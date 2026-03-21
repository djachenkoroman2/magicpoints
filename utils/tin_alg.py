#!/usr/bin/env python3
"""
TIN (Triangulated Irregular Network) construction utility for MagicPoints.

The module supports:
  - direct CLI usage;
  - import-based API usage from the main GUI application.

The public API centers around :class:`TINParameters` and
``build_tin_from_points(...)`` / ``build_tin_from_file(...)``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.spatial import Delaunay, QhullError, cKDTree
    _SCIPY_IMPORT_ERROR = None
except Exception as exc:  # noqa: BLE001
    Delaunay = None
    QhullError = Exception
    cKDTree = None
    _SCIPY_IMPORT_ERROR = exc

try:  # pragma: no cover - exercised in CLI mode rather than import mode.
    from .app_dbscan_alg import load_point_cloud_xyz
except ImportError:  # pragma: no cover - allows ``python utils/tin_alg.py``.
    from app_dbscan_alg import load_point_cloud_xyz

try:
    from rtree import index as rtree_index
except Exception:  # noqa: BLE001
    rtree_index = None


DUPLICATE_HANDLING_VALUES = ("keep_first", "average", "remove")
BOUNDARY_TYPE_VALUES = ("convex_hull", "concave_hull", "custom")
INTERPOLATION_VALUES = ("linear", "natural_neighbor")
SPATIAL_INDEX_VALUES = ("kdtree", "rtree")


class TINError(ValueError):
    """Base exception for TIN input, filtering, and triangulation failures."""


class TINInputError(TINError):
    """Raised when TIN parameters, input points, or boundary data are invalid."""


class TINGeometryError(TINError):
    """Raised when valid input cannot produce a usable triangle mesh."""


def _require_scipy() -> None:
    """Raise a clear error when scipy-backed TIN features are unavailable."""
    if _SCIPY_IMPORT_ERROR is None:
        return
    raise TINInputError(
        "TIN requires the `scipy` package. Install project dependencies with `uv sync` "
        "or `uv pip install -r requirements.txt`, then run the editor again."
    ) from _SCIPY_IMPORT_ERROR


@dataclass(frozen=True)
class TINParameters:
    """
    Normalized TIN algorithm settings.

    Parameters
    ----------
    coincidence_tolerance:
        XY-distance tolerance used to treat nearly coincident samples as duplicates
        before Delaunay triangulation. Larger values reduce numerical instability
        but can merge fine detail.
    duplicate_handling:
        Strategy for duplicate clusters inside ``coincidence_tolerance``:
        ``keep_first`` preserves the earliest sample, ``average`` merges the
        cluster into a centroid, ``remove`` drops every point participating in a
        duplicate cluster.
    max_edge_length:
        Maximum allowed triangle edge length in XY units. Values ``<= 0`` disable
        this filter. Smaller values clip long bridging triangles and keep the TIN
        tighter around dense measurements.
    min_angle:
        Minimum interior triangle angle in degrees. Larger values reject skinny
        sliver triangles but may remove more surface coverage.
    max_angle:
        Maximum interior triangle angle in degrees. Lowering the value rejects
        highly obtuse triangles and can trim unstable boundary geometry.
    outlier_filter:
        Threshold in standard deviations for nearest-neighbor spacing. Values
        ``<= 0`` disable outlier rejection. Higher values keep more isolated
        points; lower values remove sparse outliers more aggressively.
    boundary_type:
        Boundary policy: ``convex_hull`` keeps the full Delaunay hull,
        ``concave_hull`` applies an alpha-shape-style triangle filter,
        ``custom`` clips the mesh to a user-provided polygon.
    alpha:
        Alpha threshold for ``concave_hull`` mode, interpreted here as the
        maximum allowed triangle circumradius in XY units. Smaller values create
        a tighter, more concave boundary.
    interpolation_method:
        Surface interpolation strategy used when ``mesh_resolution`` performs
        spatial resampling. ``linear`` averages samples inside each resolution
        cell. ``natural_neighbor`` uses a local neighbor-weighted approximation
        to preserve smoother transitions.
    mesh_resolution:
        XY cell size used to spatially resample the input before triangulation.
        Values ``<= 0`` keep the original filtered points. Smaller cell sizes
        preserve detail; larger ones produce a coarser mesh.
    max_points:
        Maximum number of points passed to the triangulator after preprocessing.
        When exceeded, the input is spatially coarsened to stay responsive.
    spatial_index:
        Spatial index implementation used by duplicate detection, outlier
        filtering, and natural-neighbor-style resampling. ``kdtree`` is the
        default. ``rtree`` is supported when the optional dependency is present.
    """

    coincidence_tolerance: float = 1e-6
    duplicate_handling: str = "keep_first"
    max_edge_length: float = 0.0
    min_angle: float = 0.0
    max_angle: float = 175.0
    outlier_filter: float = 0.0
    boundary_type: str = "convex_hull"
    alpha: float = 5.0
    interpolation_method: str = "linear"
    mesh_resolution: float = 0.0
    max_points: int = 100_000
    spatial_index: str = "kdtree"


@dataclass(frozen=True)
class TINMesh:
    """Triangulated surface result containing compacted vertices and triangle indices."""

    vertices: np.ndarray
    triangles: np.ndarray
    source_point_count: int
    processed_point_count: int
    metadata: Mapping[str, object] = field(default_factory=dict)


def normalize_tin_parameters(
    params: TINParameters | Mapping[str, object] | None = None,
    **overrides: object,
) -> TINParameters:
    """Validate and normalize TIN settings from a dataclass or mapping."""
    raw: Dict[str, object] = {}
    if params is not None:
        if isinstance(params, TINParameters):
            raw.update(params.__dict__)
        elif isinstance(params, Mapping):
            raw.update(params)
        else:
            raise TINInputError("`params` must be a TINParameters instance, mapping, or None.")
    raw.update({key: value for key, value in overrides.items() if value is not None})

    coincidence_tolerance = _coerce_non_negative_float(
        raw.get("coincidence_tolerance", TINParameters.coincidence_tolerance),
        "coincidence_tolerance",
    )
    duplicate_handling = str(
        raw.get("duplicate_handling", TINParameters.duplicate_handling)
    ).strip().lower()
    if duplicate_handling not in DUPLICATE_HANDLING_VALUES:
        raise TINInputError(
            "`duplicate_handling` must be one of "
            f"{', '.join(DUPLICATE_HANDLING_VALUES)}."
        )

    max_edge_length = _coerce_non_negative_float(
        raw.get("max_edge_length", TINParameters.max_edge_length),
        "max_edge_length",
    )
    min_angle = _coerce_float(raw.get("min_angle", TINParameters.min_angle), "min_angle")
    max_angle = _coerce_float(raw.get("max_angle", TINParameters.max_angle), "max_angle")
    if min_angle < 0.0 or min_angle >= 180.0:
        raise TINInputError("`min_angle` must be in the range [0, 180).")
    if max_angle <= 0.0 or max_angle > 180.0:
        raise TINInputError("`max_angle` must be in the range (0, 180].")
    if max_angle <= min_angle:
        raise TINInputError("`max_angle` must be greater than `min_angle`.")

    outlier_filter = _coerce_non_negative_float(
        raw.get("outlier_filter", TINParameters.outlier_filter),
        "outlier_filter",
    )

    boundary_type = str(raw.get("boundary_type", TINParameters.boundary_type)).strip().lower()
    if boundary_type not in BOUNDARY_TYPE_VALUES:
        raise TINInputError(
            f"`boundary_type` must be one of {', '.join(BOUNDARY_TYPE_VALUES)}."
        )

    alpha = _coerce_positive_float(raw.get("alpha", TINParameters.alpha), "alpha")
    interpolation_method = str(
        raw.get("interpolation_method", TINParameters.interpolation_method)
    ).strip().lower()
    if interpolation_method not in INTERPOLATION_VALUES:
        raise TINInputError(
            "`interpolation_method` must be one of "
            f"{', '.join(INTERPOLATION_VALUES)}."
        )

    mesh_resolution = _coerce_non_negative_float(
        raw.get("mesh_resolution", TINParameters.mesh_resolution),
        "mesh_resolution",
    )

    try:
        max_points = int(raw.get("max_points", TINParameters.max_points))
    except (TypeError, ValueError) as exc:
        raise TINInputError("`max_points` must be an integer >= 3.") from exc
    if max_points < 3:
        raise TINInputError("`max_points` must be an integer >= 3.")

    spatial_index = str(raw.get("spatial_index", TINParameters.spatial_index)).strip().lower()
    if spatial_index not in SPATIAL_INDEX_VALUES:
        raise TINInputError(
            f"`spatial_index` must be one of {', '.join(SPATIAL_INDEX_VALUES)}."
        )
    if spatial_index == "rtree" and rtree_index is None:
        raise TINInputError(
            "`spatial_index='rtree'` requires the optional `rtree` package."
        )

    return TINParameters(
        coincidence_tolerance=coincidence_tolerance,
        duplicate_handling=duplicate_handling,
        max_edge_length=max_edge_length,
        min_angle=min_angle,
        max_angle=max_angle,
        outlier_filter=outlier_filter,
        boundary_type=boundary_type,
        alpha=alpha,
        interpolation_method=interpolation_method,
        mesh_resolution=mesh_resolution,
        max_points=max_points,
        spatial_index=spatial_index,
    )


def load_boundary_xy(boundary: str | Path | Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """
    Load or coerce a custom polygon boundary into an ``(N, 2)`` float64 array.

    ``boundary`` may be:
      - a TXT/PLY path containing ordered polygon vertices;
      - any array-like object with at least two numeric columns.
    """
    if isinstance(boundary, (str, Path)):
        xyz = load_point_cloud_xyz(boundary)
        polygon = np.asarray(xyz[:, :2], dtype=np.float64)
    else:
        polygon = np.asarray(boundary, dtype=np.float64)
        if polygon.ndim != 2 or polygon.shape[1] < 2:
            raise TINInputError("Custom boundary array must have shape (N, 2+) with at least 3 vertices.")
        polygon = polygon[:, :2]

    if polygon.shape[0] < 3:
        raise TINInputError("Custom boundary must contain at least 3 vertices.")
    if not np.all(np.isfinite(polygon)):
        raise TINInputError("Custom boundary contains non-finite coordinates.")

    deduped = _dedupe_polygon_vertices(polygon)
    if deduped.shape[0] < 3:
        raise TINInputError("Custom boundary collapses to fewer than 3 unique vertices.")
    return deduped


def build_tin_from_points(
    points: np.ndarray | Sequence[Sequence[float]],
    params: TINParameters | Mapping[str, object] | None = None,
    *,
    custom_boundary: str | Path | Sequence[Sequence[float]] | np.ndarray | None = None,
    output_path: str | Path | None = None,
) -> TINMesh:
    """
    Build a TIN mesh from in-memory XYZ points.

    Parameters
    ----------
    points:
        Input XYZ point array with shape ``(N, 3)``.
    params:
        :class:`TINParameters` instance or a mapping with the same field names.
        See :class:`TINParameters` for the full parameter list and how each one
        changes the resulting triangulation.
    custom_boundary:
        Optional polygon used when ``boundary_type='custom'``. The boundary can
        be a TXT/PLY path or any ``(M, 2+)`` numeric array describing ordered
        polygon vertices.
    output_path:
        Optional ``.ply`` file path for exporting the generated triangle mesh.

    Returns
    -------
    TINMesh
        Mesh vertices and compact triangle indices ready for rendering or export.
    """
    _require_scipy()
    normalized = normalize_tin_parameters(params)
    xyz = _validate_points(points)
    source_point_count = int(xyz.shape[0])
    boundary_xy = None
    if normalized.boundary_type == "custom":
        if custom_boundary is None:
            raise TINInputError(
                "`boundary_type='custom'` requires `custom_boundary` to be provided."
            )
        boundary_xy = load_boundary_xy(custom_boundary)

    working = xyz
    metadata: Dict[str, object] = {
        "boundary_type": normalized.boundary_type,
        "interpolation_method": normalized.interpolation_method,
        "mesh_resolution": float(normalized.mesh_resolution),
    }

    if boundary_xy is not None:
        working = _clip_points_to_boundary(working, boundary_xy)
        metadata["custom_boundary_vertices"] = int(boundary_xy.shape[0])

    working = _handle_duplicates(working, normalized)
    working = _filter_outliers(working, normalized)
    working = _resample_points(working, normalized)
    working = _limit_points(working, normalized)
    _ensure_minimum_points(working, "TIN preprocessing left fewer than 3 valid points.")

    triangles = _triangulate_xy(working[:, :2])
    triangles = _filter_degenerate_triangles(working, triangles, normalized.coincidence_tolerance)
    triangles = _filter_by_edge_length(working, triangles, normalized.max_edge_length)
    triangles = _filter_by_angles(working, triangles, normalized.min_angle, normalized.max_angle)

    if normalized.boundary_type == "concave_hull":
        triangles = _filter_by_alpha_shape(working, triangles, normalized.alpha)
    elif normalized.boundary_type == "custom":
        assert boundary_xy is not None
        triangles = _filter_by_polygon(working, triangles, boundary_xy)

    if triangles.size == 0:
        raise TINGeometryError(
            "TIN construction produced no valid triangles after applying the current filters."
        )

    compact_vertices, compact_triangles = _compact_mesh(working, triangles)
    mesh = TINMesh(
        vertices=compact_vertices,
        triangles=compact_triangles,
        source_point_count=source_point_count,
        processed_point_count=int(working.shape[0]),
        metadata=metadata,
    )

    if output_path is not None:
        export_mesh_to_ply(mesh, output_path)

    return mesh


def build_tin_from_file(
    input_path: str | Path,
    params: TINParameters | Mapping[str, object] | None = None,
    *,
    output_path: str | Path | None = None,
    custom_boundary: str | Path | Sequence[Sequence[float]] | np.ndarray | None = None,
) -> TINMesh:
    """
    Load a TXT/PLY cloud, triangulate it, and optionally export the mesh.

    Parameters
    ----------
    input_path:
        Source point-cloud path in any XYZ-compatible format supported by the
        project loader.
    params:
        :class:`TINParameters` instance or mapping with the same field names.
        See :class:`TINParameters` for the full list of algorithm controls and
        how each one affects the resulting triangulation.
    output_path:
        Optional ``.ply`` destination used when the caller wants the mesh saved
        in addition to the returned in-memory result.
    custom_boundary:
        Optional polygon used when ``boundary_type='custom'``.
    """
    points = load_point_cloud_xyz(input_path)
    return build_tin_from_points(
        points=points,
        params=params,
        custom_boundary=custom_boundary,
        output_path=output_path,
    )


def export_mesh_to_ply(mesh: TINMesh, output_path: str | Path) -> None:
    """Write a triangle mesh to an ASCII PLY file containing vertex and face elements."""
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    triangles = np.asarray(mesh.triangles, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise TINInputError("Mesh vertices must have shape (N, 3).")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise TINInputError("Mesh triangles must have shape (M, 3).")

    out_path = Path(output_path)
    if out_path.suffix.lower() != ".ply":
        out_path = out_path.with_suffix(".ply")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as stream:
        stream.write("ply\n")
        stream.write("format ascii 1.0\n")
        stream.write("comment Created by MagicPoints TIN utility\n")
        stream.write(f"comment source_point_count {mesh.source_point_count}\n")
        stream.write(f"comment processed_point_count {mesh.processed_point_count}\n")
        stream.write(f"element vertex {vertices.shape[0]}\n")
        stream.write("property float x\n")
        stream.write("property float y\n")
        stream.write("property float z\n")
        stream.write(f"element face {triangles.shape[0]}\n")
        stream.write("property list uchar int vertex_indices\n")
        stream.write("end_header\n")

        for x_coord, y_coord, z_coord in vertices:
            stream.write(f"{float(x_coord):.6f} {float(y_coord):.6f} {float(z_coord):.6f}\n")
        for tri in triangles:
            stream.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for TIN mesh generation."""
    parser = argparse.ArgumentParser(
        description="Build a TIN mesh from a TXT/PLY point cloud and export it as PLY."
    )
    parser.add_argument("input_cloud", help="Input point cloud file (.txt or .ply).")
    parser.add_argument("output_mesh", help="Output triangle mesh file (.ply).")
    parser.add_argument(
        "--coincidence-tolerance",
        type=float,
        default=TINParameters.coincidence_tolerance,
        help="XY tolerance for coincident samples before triangulation.",
    )
    parser.add_argument(
        "--duplicate-handling",
        choices=DUPLICATE_HANDLING_VALUES,
        default=TINParameters.duplicate_handling,
        help="How to handle duplicate points inside the coincidence tolerance.",
    )
    parser.add_argument(
        "--max-edge-length",
        type=float,
        default=TINParameters.max_edge_length,
        help="Maximum allowed triangle edge length; <=0 disables the filter.",
    )
    parser.add_argument(
        "--min-angle",
        type=float,
        default=TINParameters.min_angle,
        help="Minimum interior triangle angle in degrees.",
    )
    parser.add_argument(
        "--max-angle",
        type=float,
        default=TINParameters.max_angle,
        help="Maximum interior triangle angle in degrees.",
    )
    parser.add_argument(
        "--outlier-filter",
        type=float,
        default=TINParameters.outlier_filter,
        help="Nearest-neighbor spacing threshold in standard deviations; <=0 disables filtering.",
    )
    parser.add_argument(
        "--boundary-type",
        choices=BOUNDARY_TYPE_VALUES,
        default=TINParameters.boundary_type,
        help="Boundary policy for trimming the triangulation.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=TINParameters.alpha,
        help="Alpha threshold (maximum circumradius) for concave hull mode.",
    )
    parser.add_argument(
        "--custom-boundary",
        type=Path,
        help="Optional TXT/PLY polygon file used when --boundary-type custom.",
    )
    parser.add_argument(
        "--interpolation-method",
        choices=INTERPOLATION_VALUES,
        default=TINParameters.interpolation_method,
        help="Interpolation strategy used by mesh-resolution resampling.",
    )
    parser.add_argument(
        "--mesh-resolution",
        type=float,
        default=TINParameters.mesh_resolution,
        help="XY cell size for spatial resampling; <=0 keeps original filtered points.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=TINParameters.max_points,
        help="Maximum number of points sent to the triangulator after preprocessing.",
    )
    parser.add_argument(
        "--spatial-index",
        choices=SPATIAL_INDEX_VALUES,
        default=TINParameters.spatial_index,
        help="Spatial index used for duplicate and neighbor queries.",
    )
    return parser


def main() -> int:
    """CLI entry point for TIN mesh generation."""
    args = build_arg_parser().parse_args()
    params = TINParameters(
        coincidence_tolerance=args.coincidence_tolerance,
        duplicate_handling=args.duplicate_handling,
        max_edge_length=args.max_edge_length,
        min_angle=args.min_angle,
        max_angle=args.max_angle,
        outlier_filter=args.outlier_filter,
        boundary_type=args.boundary_type,
        alpha=args.alpha,
        interpolation_method=args.interpolation_method,
        mesh_resolution=args.mesh_resolution,
        max_points=args.max_points,
        spatial_index=args.spatial_index,
    )

    try:
        mesh = build_tin_from_file(
            input_path=args.input_cloud,
            output_path=args.output_mesh,
            params=params,
            custom_boundary=args.custom_boundary,
        )
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"TIN generation failed: {exc}") from exc

    output_mesh = Path(args.output_mesh)
    if output_mesh.suffix.lower() != ".ply":
        output_mesh = output_mesh.with_suffix(".ply")

    print(f"Source points: {mesh.source_point_count}")
    print(f"Processed points: {mesh.processed_point_count}")
    print(f"Mesh vertices: {mesh.vertices.shape[0]}")
    print(f"Mesh triangles: {mesh.triangles.shape[0]}")
    print(f"Output mesh: {output_mesh}")
    return 0


def _validate_points(points: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 3:
        raise TINInputError("Point array must have shape (N, 3+) and contain XYZ coordinates.")
    array = np.ascontiguousarray(array[:, :3], dtype=np.float64)
    if array.shape[0] == 0:
        raise TINInputError("Point array is empty.")
    if not np.all(np.isfinite(array)):
        raise TINInputError("Point array contains non-finite coordinates.")
    _ensure_minimum_points(array, "At least 3 points are required to build a TIN mesh.")
    return array


def _coerce_float(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TINInputError(f"`{name}` must be a finite float.") from exc
    if not np.isfinite(result):
        raise TINInputError(f"`{name}` must be a finite float.")
    return result


def _coerce_non_negative_float(value: object, name: str) -> float:
    result = _coerce_float(value, name)
    if result < 0.0:
        raise TINInputError(f"`{name}` must be >= 0.")
    return result


def _coerce_positive_float(value: object, name: str) -> float:
    result = _coerce_float(value, name)
    if result <= 0.0:
        raise TINInputError(f"`{name}` must be > 0.")
    return result


def _ensure_minimum_points(points: np.ndarray, message: str) -> None:
    if int(points.shape[0]) < 3:
        raise TINGeometryError(message)


def _dedupe_polygon_vertices(polygon: np.ndarray) -> np.ndarray:
    deduped: List[np.ndarray] = []
    for vertex in polygon:
        if not deduped or float(np.linalg.norm(vertex - deduped[-1])) > 1e-10:
            deduped.append(np.asarray(vertex, dtype=np.float64))
    if deduped and float(np.linalg.norm(deduped[0] - deduped[-1])) <= 1e-10:
        deduped.pop()
    return np.asarray(deduped, dtype=np.float64)


def _clip_points_to_boundary(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    point_bounds_min = np.min(points[:, :2], axis=0)
    point_bounds_max = np.max(points[:, :2], axis=0)
    poly_bounds_min = np.min(polygon, axis=0)
    poly_bounds_max = np.max(polygon, axis=0)
    intersects = np.all(point_bounds_max >= poly_bounds_min) and np.all(poly_bounds_max >= point_bounds_min)
    if not intersects:
        raise TINGeometryError(
            "Custom boundary is outside the point cloud extent and does not intersect any samples."
        )

    mask = _points_in_polygon(points[:, :2], polygon, include_boundary=True)
    clipped = np.ascontiguousarray(points[mask], dtype=np.float64)
    _ensure_minimum_points(
        clipped,
        "Custom boundary excludes too many samples; at least 3 points must remain inside it.",
    )
    return clipped


def _build_query_engine(points_xy: np.ndarray, spatial_index: str):
    if spatial_index == "kdtree":
        return _KDTreeQueryEngine(points_xy)
    if spatial_index == "rtree":
        return _RTreeQueryEngine(points_xy)
    raise TINInputError(f"Unsupported spatial index: {spatial_index}")


class _KDTreeQueryEngine:
    def __init__(self, points_xy: np.ndarray):
        _require_scipy()
        self.points_xy = np.ascontiguousarray(points_xy, dtype=np.float64)
        self.tree = cKDTree(self.points_xy)

    def radius_neighbors(self, point_xy: np.ndarray, radius: float) -> np.ndarray:
        return np.asarray(self.tree.query_ball_point(point_xy, radius), dtype=np.int32)

    def nearest(self, query_xy: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        distances, indices = self.tree.query(query_xy, k=min(k, self.points_xy.shape[0]))
        return np.asarray(distances, dtype=np.float64), np.asarray(indices, dtype=np.int32)

    def nearest_neighbor_distances(self) -> np.ndarray:
        if self.points_xy.shape[0] <= 1:
            return np.zeros(self.points_xy.shape[0], dtype=np.float64)
        distances, _ = self.tree.query(self.points_xy, k=2)
        return np.asarray(distances[:, 1], dtype=np.float64)


class _RTreeQueryEngine:
    def __init__(self, points_xy: np.ndarray):
        if rtree_index is None:
            raise TINInputError("The optional `rtree` package is not available.")
        self.points_xy = np.ascontiguousarray(points_xy, dtype=np.float64)
        properties = rtree_index.Property()
        properties.dimension = 2
        self.index = rtree_index.Index(properties=properties)
        for idx, point in enumerate(self.points_xy):
            x_coord = float(point[0])
            y_coord = float(point[1])
            self.index.insert(idx, (x_coord, y_coord, x_coord, y_coord))

    def radius_neighbors(self, point_xy: np.ndarray, radius: float) -> np.ndarray:
        x_coord = float(point_xy[0])
        y_coord = float(point_xy[1])
        hits = list(
            self.index.intersection(
                (x_coord - radius, y_coord - radius, x_coord + radius, y_coord + radius)
            )
        )
        if not hits:
            return np.zeros(0, dtype=np.int32)
        hit_indices = np.asarray(hits, dtype=np.int32)
        deltas = self.points_xy[hit_indices] - np.asarray(point_xy, dtype=np.float64)
        dist_sq = np.einsum("ij,ij->i", deltas, deltas, optimize=True)
        return np.ascontiguousarray(hit_indices[dist_sq <= radius * radius], dtype=np.int32)

    def nearest(self, query_xy: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        queries = np.asarray(query_xy, dtype=np.float64)
        single_query = queries.ndim == 1
        if single_query:
            queries = queries.reshape(1, 2)
        distances: List[np.ndarray] = []
        indices: List[np.ndarray] = []
        result_count = min(int(k), int(self.points_xy.shape[0]))
        for point_xy in queries:
            x_coord = float(point_xy[0])
            y_coord = float(point_xy[1])
            hits = list(
                self.index.nearest((x_coord, y_coord, x_coord, y_coord), num_results=result_count)
            )
            hit_indices = np.asarray(hits, dtype=np.int32)
            deltas = self.points_xy[hit_indices] - point_xy
            dist = np.sqrt(np.einsum("ij,ij->i", deltas, deltas, optimize=True))
            order = np.argsort(dist)
            distances.append(np.asarray(dist[order], dtype=np.float64))
            indices.append(np.asarray(hit_indices[order], dtype=np.int32))
        if single_query:
            return distances[0], indices[0]
        return np.asarray(distances, dtype=np.float64), np.asarray(indices, dtype=np.int32)

    def nearest_neighbor_distances(self) -> np.ndarray:
        if self.points_xy.shape[0] <= 1:
            return np.zeros(self.points_xy.shape[0], dtype=np.float64)
        distances, _ = self.nearest(self.points_xy, k=2)
        return np.asarray(distances[:, 1], dtype=np.float64)


def _handle_duplicates(points: np.ndarray, params: TINParameters) -> np.ndarray:
    tolerance = float(params.coincidence_tolerance)
    if tolerance <= 0.0:
        unique_xy, inverse = np.unique(points[:, :2], axis=0, return_inverse=True)
        if unique_xy.shape[0] == points.shape[0]:
            return points
        return _merge_duplicate_groups(points, inverse, params.duplicate_handling)

    groups = _duplicate_group_labels(points[:, :2], tolerance, params.spatial_index)
    unique_group_count = int(np.max(groups)) + 1
    if unique_group_count == int(points.shape[0]):
        return points
    merged = _merge_duplicate_groups(points, groups, params.duplicate_handling)
    _ensure_minimum_points(
        merged,
        "Duplicate-point handling removed too many samples; fewer than 3 unique points remain.",
    )
    return merged


def _duplicate_group_labels(points_xy: np.ndarray, tolerance: float, spatial_index: str) -> np.ndarray:
    engine = _build_query_engine(points_xy, spatial_index)
    parent = np.arange(points_xy.shape[0], dtype=np.int32)

    def find(index: int) -> int:
        root = index
        while parent[root] != root:
            root = int(parent[root])
        while parent[index] != index:
            next_index = int(parent[index])
            parent[index] = root
            index = next_index
        return root

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for idx, point_xy in enumerate(points_xy):
        neighbors = engine.radius_neighbors(point_xy, tolerance)
        for neighbor in neighbors:
            if int(neighbor) > idx:
                union(idx, int(neighbor))

    labels = np.empty(points_xy.shape[0], dtype=np.int32)
    remap: Dict[int, int] = {}
    next_label = 0
    for idx in range(points_xy.shape[0]):
        root = find(idx)
        label = remap.get(root)
        if label is None:
            label = next_label
            remap[root] = label
            next_label += 1
        labels[idx] = label
    return labels


def _merge_duplicate_groups(points: np.ndarray, labels: np.ndarray, strategy: str) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    unique_labels = np.unique(labels)
    merged: List[np.ndarray] = []

    for label in unique_labels:
        group = points[labels == label]
        if group.shape[0] <= 1:
            merged.append(group[0])
            continue

        if strategy == "keep_first":
            merged.append(group[0])
        elif strategy == "average":
            merged.append(np.mean(group, axis=0, dtype=np.float64))
        elif strategy == "remove":
            continue
        else:  # pragma: no cover - guarded by parameter validation.
            raise TINInputError(f"Unsupported duplicate handling strategy: {strategy}")

    if not merged:
        return np.zeros((0, 3), dtype=np.float64)
    return np.ascontiguousarray(np.vstack(merged), dtype=np.float64)


def _filter_outliers(points: np.ndarray, params: TINParameters) -> np.ndarray:
    threshold = float(params.outlier_filter)
    if threshold <= 0.0 or points.shape[0] < 4:
        return points

    engine = _build_query_engine(points[:, :2], params.spatial_index)
    nn_distances = engine.nearest_neighbor_distances()
    mean_distance = float(np.mean(nn_distances))
    std_distance = float(np.std(nn_distances))
    if std_distance <= 1e-12:
        return points

    cutoff = mean_distance + threshold * std_distance
    mask = nn_distances <= cutoff
    filtered = np.ascontiguousarray(points[mask], dtype=np.float64)
    _ensure_minimum_points(
        filtered,
        "Outlier filtering removed too many samples; fewer than 3 points remain.",
    )
    return filtered


def _resample_points(points: np.ndarray, params: TINParameters) -> np.ndarray:
    resolution = float(params.mesh_resolution)
    if resolution <= 0.0 or points.shape[0] < 4:
        return points

    xy = points[:, :2]
    min_xy = np.min(xy, axis=0)
    cell_coords = np.floor((xy - min_xy) / max(resolution, 1e-12)).astype(np.int64)
    _, inverse = np.unique(cell_coords, axis=0, return_inverse=True)
    if np.unique(inverse).shape[0] == points.shape[0]:
        return points

    if params.interpolation_method == "linear":
        return _aggregate_cells_by_mean(points, inverse)
    return _aggregate_cells_by_neighbor_interpolation(points, inverse, params.spatial_index)


def _aggregate_cells_by_mean(points: np.ndarray, labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int32)
    merged: List[np.ndarray] = []
    for label in np.unique(labels):
        group = points[labels == label]
        merged.append(np.mean(group, axis=0, dtype=np.float64))
    return np.ascontiguousarray(np.vstack(merged), dtype=np.float64)


def _aggregate_cells_by_neighbor_interpolation(
    points: np.ndarray,
    labels: np.ndarray,
    spatial_index: str,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int32)
    merged: List[np.ndarray] = []
    engine = _build_query_engine(points[:, :2], spatial_index)
    k_neighbors = max(3, min(8, int(points.shape[0])))

    for label in np.unique(labels):
        group = points[labels == label]
        query_xy = np.mean(group[:, :2], axis=0, dtype=np.float64)
        distances, indices = engine.nearest(query_xy, k_neighbors)
        distances = np.atleast_1d(np.asarray(distances, dtype=np.float64))
        indices = np.atleast_1d(np.asarray(indices, dtype=np.int32))
        if distances.size == 0:
            z_value = float(np.mean(group[:, 2], dtype=np.float64))
        else:
            exact_mask = distances <= 1e-12
            if np.any(exact_mask):
                z_value = float(np.mean(points[indices[exact_mask], 2], dtype=np.float64))
            else:
                weights = 1.0 / np.maximum(distances, 1e-9)
                z_value = float(np.sum(weights * points[indices, 2]) / np.sum(weights))
        merged.append(np.array([query_xy[0], query_xy[1], z_value], dtype=np.float64))

    return np.ascontiguousarray(np.vstack(merged), dtype=np.float64)


def _limit_points(points: np.ndarray, params: TINParameters) -> np.ndarray:
    if points.shape[0] <= params.max_points:
        return points

    xy = points[:, :2]
    min_xy = np.min(xy, axis=0)
    max_xy = np.max(xy, axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)
    target_area = float(span[0] * span[1])
    inferred_resolution = max(np.sqrt(target_area / max(float(params.max_points), 1.0)), 1e-6)
    reduced = _resample_points(
        points,
        TINParameters(
            coincidence_tolerance=params.coincidence_tolerance,
            duplicate_handling=params.duplicate_handling,
            max_edge_length=params.max_edge_length,
            min_angle=params.min_angle,
            max_angle=params.max_angle,
            outlier_filter=params.outlier_filter,
            boundary_type=params.boundary_type,
            alpha=params.alpha,
            interpolation_method=params.interpolation_method,
            mesh_resolution=inferred_resolution,
            max_points=params.max_points,
            spatial_index=params.spatial_index,
        ),
    )
    if reduced.shape[0] <= params.max_points:
        return reduced

    indices = np.linspace(0, reduced.shape[0] - 1, params.max_points, dtype=np.int64)
    return np.ascontiguousarray(reduced[indices], dtype=np.float64)


def _triangulate_xy(points_xy: np.ndarray) -> np.ndarray:
    try:
        triangulation = Delaunay(np.ascontiguousarray(points_xy, dtype=np.float64), qhull_options="Qbb Qc Qz")
    except QhullError as exc:
        raise TINGeometryError(
            "Delaunay triangulation failed. The point set may be collinear, nearly degenerate, "
            "or too heavily filtered for 2D triangulation."
        ) from exc
    simplices = np.asarray(triangulation.simplices, dtype=np.int32)
    if simplices.ndim != 2 or simplices.shape[1] != 3 or simplices.shape[0] == 0:
        raise TINGeometryError("Delaunay triangulation did not produce any triangles.")
    return np.ascontiguousarray(simplices, dtype=np.int32)


def _filter_degenerate_triangles(
    vertices: np.ndarray,
    triangles: np.ndarray,
    coincidence_tolerance: float,
) -> np.ndarray:
    areas = _triangle_areas_xy(vertices, triangles)
    area_eps = max(float(coincidence_tolerance) ** 2 * 0.5, 1e-12)
    filtered = triangles[areas > area_eps]
    if filtered.size == 0:
        raise TINGeometryError("All candidate triangles are degenerate or have near-zero area.")
    return np.ascontiguousarray(filtered, dtype=np.int32)


def _filter_by_edge_length(
    vertices: np.ndarray,
    triangles: np.ndarray,
    max_edge_length: float,
) -> np.ndarray:
    if max_edge_length <= 0.0:
        return triangles
    edge_lengths = _triangle_edge_lengths_xy(vertices, triangles)
    mask = np.max(edge_lengths, axis=1) <= max_edge_length
    filtered = triangles[mask]
    if filtered.size == 0:
        raise TINGeometryError(
            "The `max_edge_length` filter removed every triangle; increase the limit or disable the filter."
        )
    return np.ascontiguousarray(filtered, dtype=np.int32)


def _filter_by_angles(
    vertices: np.ndarray,
    triangles: np.ndarray,
    min_angle: float,
    max_angle: float,
) -> np.ndarray:
    angles = _triangle_angles_deg(vertices, triangles)
    mask = (np.min(angles, axis=1) >= min_angle) & (np.max(angles, axis=1) <= max_angle)
    filtered = triangles[mask]
    if filtered.size == 0:
        raise TINGeometryError(
            "Triangle angle filters removed every triangle; relax `min_angle`/`max_angle`."
        )
    return np.ascontiguousarray(filtered, dtype=np.int32)


def _filter_by_alpha_shape(vertices: np.ndarray, triangles: np.ndarray, alpha: float) -> np.ndarray:
    circumradius = _triangle_circumradius_xy(vertices, triangles)
    mask = circumradius <= alpha
    filtered = triangles[mask]
    if filtered.size == 0:
        raise TINGeometryError(
            "Concave hull filtering removed every triangle; increase `alpha` to allow a looser boundary."
        )
    return np.ascontiguousarray(filtered, dtype=np.int32)


def _filter_by_polygon(vertices: np.ndarray, triangles: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    tri_vertices = vertices[triangles]
    centroids = np.mean(tri_vertices[:, :, :2], axis=1, dtype=np.float64)
    midpoints_01 = 0.5 * (tri_vertices[:, 0, :2] + tri_vertices[:, 1, :2])
    midpoints_12 = 0.5 * (tri_vertices[:, 1, :2] + tri_vertices[:, 2, :2])
    midpoints_20 = 0.5 * (tri_vertices[:, 2, :2] + tri_vertices[:, 0, :2])

    mask = (
        _points_in_polygon(centroids, polygon, include_boundary=True)
        & _points_in_polygon(midpoints_01, polygon, include_boundary=True)
        & _points_in_polygon(midpoints_12, polygon, include_boundary=True)
        & _points_in_polygon(midpoints_20, polygon, include_boundary=True)
    )
    filtered = triangles[mask]
    if filtered.size == 0:
        raise TINGeometryError(
            "Custom boundary clipping removed every triangle; verify that the polygon overlaps the cloud."
        )
    return np.ascontiguousarray(filtered, dtype=np.int32)


def _compact_mesh(vertices: np.ndarray, triangles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    used = np.unique(triangles.reshape(-1))
    compact_vertices = np.ascontiguousarray(vertices[used], dtype=np.float32)
    remap = np.full(vertices.shape[0], -1, dtype=np.int32)
    remap[used] = np.arange(used.shape[0], dtype=np.int32)
    compact_triangles = np.ascontiguousarray(remap[triangles], dtype=np.int32)
    return compact_vertices, compact_triangles


def _triangle_areas_xy(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    tri_xy = vertices[triangles][:, :, :2]
    edge_a = tri_xy[:, 1] - tri_xy[:, 0]
    edge_b = tri_xy[:, 2] - tri_xy[:, 0]
    cross = edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0]
    return 0.5 * np.abs(cross)


def _triangle_edge_lengths_xy(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    tri_xy = vertices[triangles][:, :, :2]
    lengths = np.empty((triangles.shape[0], 3), dtype=np.float64)
    lengths[:, 0] = np.linalg.norm(tri_xy[:, 1] - tri_xy[:, 0], axis=1)
    lengths[:, 1] = np.linalg.norm(tri_xy[:, 2] - tri_xy[:, 1], axis=1)
    lengths[:, 2] = np.linalg.norm(tri_xy[:, 0] - tri_xy[:, 2], axis=1)
    return lengths


def _triangle_angles_deg(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    edge_lengths = _triangle_edge_lengths_xy(vertices, triangles)
    a_len = edge_lengths[:, 1]
    b_len = edge_lengths[:, 2]
    c_len = edge_lengths[:, 0]

    cos_a = np.clip((b_len * b_len + c_len * c_len - a_len * a_len) / (2.0 * b_len * c_len), -1.0, 1.0)
    cos_b = np.clip((a_len * a_len + c_len * c_len - b_len * b_len) / (2.0 * a_len * c_len), -1.0, 1.0)
    angle_a = np.degrees(np.arccos(cos_a))
    angle_b = np.degrees(np.arccos(cos_b))
    angle_c = 180.0 - angle_a - angle_b
    return np.column_stack((angle_a, angle_b, angle_c))


def _triangle_circumradius_xy(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    edge_lengths = _triangle_edge_lengths_xy(vertices, triangles)
    areas = _triangle_areas_xy(vertices, triangles)
    safe_areas = np.maximum(areas, 1e-12)
    return (edge_lengths[:, 0] * edge_lengths[:, 1] * edge_lengths[:, 2]) / (4.0 * safe_areas)


def _points_in_polygon(points_xy: np.ndarray, polygon: np.ndarray, *, include_boundary: bool) -> np.ndarray:
    points_arr = np.asarray(points_xy, dtype=np.float64)
    polygon_arr = np.asarray(polygon, dtype=np.float64)
    if points_arr.ndim != 2 or points_arr.shape[1] != 2:
        raise TINInputError("Polygon queries expect an array with shape (N, 2).")

    x_coord = points_arr[:, 0]
    y_coord = points_arr[:, 1]
    x0 = polygon_arr[:, 0]
    y0 = polygon_arr[:, 1]
    x1 = np.roll(x0, -1)
    y1 = np.roll(y0, -1)

    inside = np.zeros(points_arr.shape[0], dtype=bool)
    for edge_x0, edge_y0, edge_x1, edge_y1 in zip(x0, y0, x1, y1):
        if include_boundary:
            on_edge = _points_on_segment(
                points_arr,
                np.array([edge_x0, edge_y0], dtype=np.float64),
                np.array([edge_x1, edge_y1], dtype=np.float64),
            )
            inside |= on_edge

        cond = (edge_y0 > y_coord) != (edge_y1 > y_coord)
        safe_den = edge_y1 - edge_y0
        safe_den = safe_den if abs(float(safe_den)) > 1e-12 else np.copysign(1e-12, safe_den if safe_den != 0 else 1.0)
        cross_x = (edge_x1 - edge_x0) * (y_coord - edge_y0) / safe_den + edge_x0
        inside ^= cond & (x_coord < cross_x)

    return inside


def _points_on_segment(points_xy: np.ndarray, start_xy: np.ndarray, end_xy: np.ndarray) -> np.ndarray:
    segment = end_xy - start_xy
    point_delta = points_xy - start_xy.reshape(1, 2)
    cross = point_delta[:, 0] * segment[1] - point_delta[:, 1] * segment[0]
    close_to_line = np.abs(cross) <= 1e-9

    dot = point_delta[:, 0] * segment[0] + point_delta[:, 1] * segment[1]
    seg_len_sq = float(np.dot(segment, segment))
    if seg_len_sq <= 1e-20:
        return np.linalg.norm(point_delta, axis=1) <= 1e-9
    return close_to_line & (dot >= -1e-9) & (dot <= seg_len_sq + 1e-9)


if __name__ == "__main__":
    raise SystemExit(main())
