#!/usr/bin/env python3
"""
DBSCAN clustering utility for MagicPoints point clouds.

The module supports:
  - direct CLI usage;
  - import-based API usage from the main GUI application.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import re
import struct
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml


CLUSTER_FILE_SCHEMA = "magicpoints.dbscan.clusters.v1"


ProgressCallback = Callable[[float, str], None]


def _emit_progress(
    progress_callback: Optional[ProgressCallback],
    progress: float,
    stage: str = "",
) -> None:
    if progress_callback is None:
        return
    progress_callback(float(min(1.0, max(0.0, progress))), str(stage).strip())


def _progress_interval(count: int) -> int:
    return max(1, int(count) // 200)


class DBSCANInputError(ValueError):
    """Raised when DBSCAN parameters or input files are invalid."""


@dataclass(frozen=True)
class BoundingBox3D:
    min_corner: Tuple[float, float, float]
    max_corner: Tuple[float, float, float]

    def to_yaml_data(self) -> Dict[str, List[float]]:
        return {
            "min": [float(value) for value in self.min_corner],
            "max": [float(value) for value in self.max_corner],
        }


@dataclass(frozen=True)
class ClusterRecord:
    cluster_id: int
    point_count: int
    bounding_box: BoundingBox3D

    def to_yaml_data(self) -> Dict[str, object]:
        return {
            "id": int(self.cluster_id),
            "point_count": int(self.point_count),
            "bounding_box": self.bounding_box.to_yaml_data(),
        }


@dataclass(frozen=True)
class DBSCANClusterResult:
    epsilon: float
    min_pts: int
    source_point_count: int
    input_path: str = ""
    clusters: Tuple[ClusterRecord, ...] = ()

    def to_yaml_data(self) -> Dict[str, object]:
        data: Dict[str, object] = {
            "schema": CLUSTER_FILE_SCHEMA,
            "epsilon": float(self.epsilon),
            "min_pts": int(self.min_pts),
            "source_point_count": int(self.source_point_count),
            "clusters": [cluster.to_yaml_data() for cluster in self.clusters],
        }
        if self.input_path:
            data["input_path"] = self.input_path
        return data


@dataclass(frozen=True)
class PlyProperty:
    kind: str
    name: str
    dtype: str
    count_dtype: Optional[str] = None


@dataclass(frozen=True)
class PlyElement:
    name: str
    count: int
    properties: Tuple[PlyProperty, ...]


PLY_TO_STRUCT: Dict[str, str] = {
    "char": "b",
    "int8": "b",
    "uchar": "B",
    "uint8": "B",
    "short": "h",
    "int16": "h",
    "ushort": "H",
    "uint16": "H",
    "int": "i",
    "int32": "i",
    "uint": "I",
    "uint32": "I",
    "float": "f",
    "float32": "f",
    "double": "d",
    "float64": "d",
}
PLY_TO_NUMPY: Dict[str, str] = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}

_NEIGHBOR_OFFSETS = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
)


def normalize_field_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def _validate_points(points: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise DBSCANInputError("Point array must have shape (N, 3).")
    if array.shape[0] == 0:
        raise DBSCANInputError("Point array is empty.")
    if not np.all(np.isfinite(array)):
        raise DBSCANInputError("Point array contains non-finite coordinates.")
    return np.ascontiguousarray(array, dtype=np.float64)


def _validate_epsilon(epsilon: float) -> float:
    epsilon_value = float(epsilon)
    if not np.isfinite(epsilon_value) or epsilon_value <= 0.0:
        raise DBSCANInputError("`epsilon` must be a finite number greater than 0.")
    return epsilon_value


def _validate_min_pts(min_pts: int) -> int:
    min_pts_value = int(min_pts)
    if min_pts_value < 1:
        raise DBSCANInputError("`min_pts` must be an integer greater than or equal to 1.")
    return min_pts_value


def _build_spatial_hash(points: np.ndarray, epsilon: float) -> Tuple[np.ndarray, Dict[Tuple[int, int, int], np.ndarray]]:
    cell_coords = np.floor(points / epsilon).astype(np.int64, copy=False)
    buckets: Dict[Tuple[int, int, int], List[int]] = {}
    for index, coord in enumerate(cell_coords):
        key = (int(coord[0]), int(coord[1]), int(coord[2]))
        bucket = buckets.get(key)
        if bucket is None:
            bucket = []
            buckets[key] = bucket
        bucket.append(index)
    frozen = {
        key: np.asarray(indices, dtype=np.int32)
        for key, indices in buckets.items()
    }
    return cell_coords, frozen


def _region_query(
    point_index: int,
    points: np.ndarray,
    epsilon_sq: float,
    cell_coords: np.ndarray,
    buckets: Mapping[Tuple[int, int, int], np.ndarray],
    cache: List[Optional[np.ndarray]],
) -> np.ndarray:
    cached = cache[point_index]
    if cached is not None:
        return cached

    coord = cell_coords[point_index]
    candidate_chunks: List[np.ndarray] = []
    for dx, dy, dz in _NEIGHBOR_OFFSETS:
        key = (int(coord[0] + dx), int(coord[1] + dy), int(coord[2] + dz))
        chunk = buckets.get(key)
        if chunk is not None and chunk.size > 0:
            candidate_chunks.append(chunk)

    if not candidate_chunks:
        result = np.asarray([point_index], dtype=np.int32)
        cache[point_index] = result
        return result

    candidates = np.concatenate(candidate_chunks, axis=0)
    deltas = points[candidates] - points[point_index]
    distances_sq = np.einsum("ij,ij->i", deltas, deltas, optimize=True)
    neighbors = np.ascontiguousarray(candidates[distances_sq <= epsilon_sq], dtype=np.int32)
    cache[point_index] = neighbors
    return neighbors


def compute_dbscan_labels(
    points: np.ndarray | Sequence[Sequence[float]],
    epsilon: float,
    min_pts: int,
    progress_callback: Optional[ProgressCallback] = None,
) -> np.ndarray:
    """
    Compute DBSCAN cluster labels for a point array with shape (N, 3).

    Returns an array of shape (N,) where:
      - `-1` means noise;
      - `0..K-1` are cluster identifiers.
    """

    point_array = _validate_points(points)
    epsilon_value = _validate_epsilon(epsilon)
    min_pts_value = _validate_min_pts(min_pts)

    point_count = int(point_array.shape[0])
    labels = np.full(point_count, -2, dtype=np.int32)  # -2 = unvisited
    epsilon_sq = float(epsilon_value * epsilon_value)
    _emit_progress(progress_callback, 0.0, "Building spatial hash")
    cell_coords, buckets = _build_spatial_hash(point_array, epsilon_value)
    neighbor_cache: List[Optional[np.ndarray]] = [None] * point_count
    report_interval = _progress_interval(point_count)

    _emit_progress(progress_callback, 0.08, "Clustering points")
    cluster_id = 0
    for point_index in range(point_count):
        if labels[point_index] != -2:
            continue

        neighbors = _region_query(
            point_index,
            point_array,
            epsilon_sq,
            cell_coords,
            buckets,
            neighbor_cache,
        )
        if neighbors.size < min_pts_value:
            labels[point_index] = -1
            continue

        labels[point_index] = cluster_id
        queue: deque[int] = deque()
        queued = np.zeros(point_count, dtype=bool)
        for neighbor in neighbors:
            neighbor_index = int(neighbor)
            if neighbor_index == point_index or queued[neighbor_index]:
                continue
            queue.append(neighbor_index)
            queued[neighbor_index] = True

        while queue:
            neighbor_index = queue.popleft()
            if labels[neighbor_index] == -1:
                labels[neighbor_index] = cluster_id
            if labels[neighbor_index] != -2:
                continue

            labels[neighbor_index] = cluster_id
            neighbor_neighbors = _region_query(
                neighbor_index,
                point_array,
                epsilon_sq,
                cell_coords,
                buckets,
                neighbor_cache,
            )
            if neighbor_neighbors.size < min_pts_value:
                continue

            for nested_neighbor in neighbor_neighbors:
                nested_index = int(nested_neighbor)
                if labels[nested_index] == -1:
                    labels[nested_index] = cluster_id
                if queued[nested_index]:
                    continue
                queue.append(nested_index)
                queued[nested_index] = True

        cluster_id += 1

        if (point_index + 1) % report_interval == 0 or point_index == point_count - 1:
            _emit_progress(
                progress_callback,
                0.08 + 0.92 * ((point_index + 1) / point_count),
                f"Clustering points ({point_index + 1:,}/{point_count:,})",
            )

    labels[labels == -2] = -1
    _emit_progress(progress_callback, 1.0, "Clustering complete")
    return labels


def build_cluster_records(
    points: np.ndarray,
    labels: np.ndarray,
    progress_callback: Optional[ProgressCallback] = None,
) -> Tuple[ClusterRecord, ...]:
    point_array = _validate_points(points)
    label_array = np.asarray(labels, dtype=np.int32).reshape(-1)
    if label_array.shape[0] != point_array.shape[0]:
        raise DBSCANInputError("Labels must contain the same number of rows as points.")

    cluster_ids = sorted(int(cluster_id) for cluster_id in np.unique(label_array) if cluster_id >= 0)
    if not cluster_ids:
        _emit_progress(progress_callback, 1.0, "No clusters found")
        return ()

    clusters: List[ClusterRecord] = []
    report_interval = _progress_interval(len(cluster_ids))
    total_clusters = len(cluster_ids)
    for index, cluster_id in enumerate(cluster_ids, start=1):
        cluster_points = point_array[label_array == cluster_id]
        if cluster_points.shape[0] == 0:
            continue
        min_corner = tuple(float(value) for value in np.min(cluster_points, axis=0))
        max_corner = tuple(float(value) for value in np.max(cluster_points, axis=0))
        clusters.append(
            ClusterRecord(
                cluster_id=int(cluster_id),
                point_count=int(cluster_points.shape[0]),
                bounding_box=BoundingBox3D(
                    min_corner=min_corner,
                    max_corner=max_corner,
                ),
            )
        )
        if index % report_interval == 0 or index == total_clusters:
            _emit_progress(
                progress_callback,
                index / total_clusters,
                f"Building bounding boxes ({index:,}/{total_clusters:,})",
            )
    return tuple(clusters)


def run_dbscan_on_points(
    points: np.ndarray | Sequence[Sequence[float]],
    epsilon: float,
    min_pts: int,
    output_path: Optional[str | Path] = None,
    input_path: str = "",
    progress_callback: Optional[ProgressCallback] = None,
) -> DBSCANClusterResult:
    point_array = _validate_points(points)
    epsilon_value = _validate_epsilon(epsilon)
    min_pts_value = _validate_min_pts(min_pts)

    def scaled_progress(start: float, end: float):
        def emit(progress: float, stage: str = "") -> None:
            clamped = float(min(1.0, max(0.0, progress)))
            _emit_progress(progress_callback, start + (end - start) * clamped, stage)
        return emit

    _emit_progress(progress_callback, 0.0, "Preparing DBSCAN")
    labels = compute_dbscan_labels(
        point_array,
        epsilon_value,
        min_pts_value,
        progress_callback=scaled_progress(0.05, 0.80),
    )
    clusters = build_cluster_records(
        point_array,
        labels,
        progress_callback=scaled_progress(0.80, 0.95),
    )
    result = DBSCANClusterResult(
        epsilon=epsilon_value,
        min_pts=min_pts_value,
        source_point_count=int(point_array.shape[0]),
        input_path=str(input_path),
        clusters=clusters,
    )
    if output_path is not None:
        _emit_progress(progress_callback, 0.97, "Saving clusters YAML")
        save_cluster_result(result, output_path)
    _emit_progress(progress_callback, 1.0, "Complete")
    return result


def run_dbscan_on_file(
    input_path: str | Path,
    epsilon: float,
    min_pts: int,
    output_path: Optional[str | Path] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> DBSCANClusterResult:
    input_file = Path(input_path)
    points = load_point_cloud_xyz(input_file)
    return run_dbscan_on_points(
        points,
        epsilon=epsilon,
        min_pts=min_pts,
        output_path=output_path,
        input_path=str(input_file),
        progress_callback=progress_callback,
    )


def save_cluster_result(result: DBSCANClusterResult, output_path: str | Path) -> Path:
    out_path = Path(output_path)
    if out_path.suffix.lower() not in {".yaml", ".yml"}:
        out_path = out_path.with_suffix(".yaml")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(
            result.to_yaml_data(),
            stream,
            sort_keys=False,
            allow_unicode=False,
        )
    return out_path


def _as_float_triplet(values: object, field_name: str) -> Tuple[float, float, float]:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise DBSCANInputError(f"`{field_name}` must be a sequence of exactly 3 numbers.")
    triplet = tuple(float(value) for value in values)
    if not all(np.isfinite(triplet)):
        raise DBSCANInputError(f"`{field_name}` must contain only finite numbers.")
    return triplet


def load_cluster_result(path: str | Path) -> DBSCANClusterResult:
    input_path = Path(path)
    try:
        with input_path.open("r", encoding="utf-8") as stream:
            raw_data = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise DBSCANInputError(f"Invalid YAML cluster file: {exc}") from exc

    if not isinstance(raw_data, dict):
        raise DBSCANInputError("Cluster YAML root must be a mapping.")

    raw_clusters = raw_data.get("clusters", [])
    if raw_clusters is None:
        raw_clusters = []
    if not isinstance(raw_clusters, list):
        raise DBSCANInputError("`clusters` must be a list.")

    clusters: List[ClusterRecord] = []
    for index, raw_cluster in enumerate(raw_clusters):
        if not isinstance(raw_cluster, dict):
            raise DBSCANInputError(f"`clusters[{index}]` must be a mapping.")
        raw_bbox = raw_cluster.get("bounding_box")
        if not isinstance(raw_bbox, dict):
            raise DBSCANInputError(f"`clusters[{index}].bounding_box` must be a mapping.")

        cluster_id = int(raw_cluster.get("id", index))
        point_count = int(raw_cluster.get("point_count", 0))
        if point_count < 0:
            raise DBSCANInputError(f"`clusters[{index}].point_count` must be >= 0.")

        clusters.append(
            ClusterRecord(
                cluster_id=cluster_id,
                point_count=point_count,
                bounding_box=BoundingBox3D(
                    min_corner=_as_float_triplet(raw_bbox.get("min"), f"clusters[{index}].bounding_box.min"),
                    max_corner=_as_float_triplet(raw_bbox.get("max"), f"clusters[{index}].bounding_box.max"),
                ),
            )
        )

    epsilon = float(raw_data.get("epsilon", 0.0))
    min_pts = int(raw_data.get("min_pts", 0))
    source_point_count = int(raw_data.get("source_point_count", 0))
    input_cloud = str(raw_data.get("input_path", ""))

    if "epsilon" in raw_data:
        epsilon = _validate_epsilon(epsilon)
    if "min_pts" in raw_data:
        min_pts = _validate_min_pts(min_pts)
    if source_point_count < 0:
        raise DBSCANInputError("`source_point_count` must be >= 0.")

    return DBSCANClusterResult(
        epsilon=epsilon,
        min_pts=min_pts,
        source_point_count=source_point_count,
        input_path=input_cloud,
        clusters=tuple(clusters),
    )


def load_point_cloud_xyz(path: str | Path) -> np.ndarray:
    input_path = Path(path)
    if not input_path.is_file():
        raise DBSCANInputError(f"Input point cloud file not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix == ".txt":
        return _load_txt_xyz(input_path)
    if suffix == ".ply":
        return _load_ply_xyz(input_path)
    raise DBSCANInputError(f"Unsupported input format '{suffix}'. Supported formats: .txt, .ply.")


def _load_txt_xyz(path: Path) -> np.ndarray:
    rows: List[List[float]] = []
    header_tokens: Optional[List[str]] = None
    column_count: Optional[int] = None

    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            tokens = [token for token in re.split(r"[,\s]+", line) if token]
            if not tokens:
                continue

            values: List[float] = []
            numeric = True
            for token in tokens:
                try:
                    values.append(float(token))
                except ValueError:
                    numeric = False
                    break

            if not numeric:
                if header_tokens is None and not rows:
                    header_tokens = [normalize_field_name(token) for token in tokens]
                    continue
                raise DBSCANInputError(
                    f"Unable to parse numeric TXT point data at line {line_number}: '{raw_line.rstrip()}'."
                )

            if column_count is None:
                column_count = len(values)
            elif len(values) != column_count:
                raise DBSCANInputError(
                    f"Inconsistent TXT column count at line {line_number}: expected "
                    f"{column_count}, found {len(values)}."
                )

            rows.append(values)

    if not rows:
        raise DBSCANInputError("TXT file does not contain any point records.")

    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.shape[1] < 3:
        raise DBSCANInputError("TXT file must contain at least 3 columns for XYZ coordinates.")
    return _extract_xyz_from_matrix(matrix, header_tokens)


def _load_ply_xyz(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        first = stream.readline().decode("ascii", errors="ignore").strip().lower()
        if first != "ply":
            raise DBSCANInputError("Invalid PLY file: missing 'ply' header.")

        format_type: Optional[str] = None
        elements: List[PlyElement] = []
        current_name: Optional[str] = None
        current_count: Optional[int] = None
        current_properties: List[PlyProperty] = []

        while True:
            raw_line = stream.readline()
            if not raw_line:
                raise DBSCANInputError("Unexpected end of file while reading PLY header.")
            line = raw_line.decode("ascii", errors="ignore").strip()
            if line == "end_header":
                break
            if not line:
                continue

            parts = line.split()
            keyword = parts[0].lower()
            if keyword in {"comment", "obj_info"}:
                continue
            if keyword == "format":
                if len(parts) < 2:
                    raise DBSCANInputError("Malformed PLY format declaration.")
                format_type = parts[1].lower()
                continue
            if keyword == "element":
                if current_name is not None and current_count is not None:
                    elements.append(
                        PlyElement(
                            name=current_name,
                            count=current_count,
                            properties=tuple(current_properties),
                        )
                    )
                if len(parts) != 3:
                    raise DBSCANInputError(f"Malformed PLY element declaration: '{line}'")
                try:
                    current_count = int(parts[2])
                except ValueError as exc:
                    raise DBSCANInputError(f"Invalid PLY element count '{parts[2]}'.") from exc
                current_name = parts[1]
                current_properties = []
                continue
            if keyword == "property":
                if current_name is None:
                    raise DBSCANInputError("PLY property declared before any element.")
                if len(parts) >= 5 and parts[1].lower() == "list":
                    current_properties.append(
                        PlyProperty(
                            kind="list",
                            name=parts[4],
                            dtype=parts[3].lower(),
                            count_dtype=parts[2].lower(),
                        )
                    )
                elif len(parts) == 3:
                    current_properties.append(
                        PlyProperty(
                            kind="scalar",
                            name=parts[2],
                            dtype=parts[1].lower(),
                        )
                    )
                else:
                    raise DBSCANInputError(f"Malformed PLY property declaration: '{line}'")

        if current_name is not None and current_count is not None:
            elements.append(
                PlyElement(
                    name=current_name,
                    count=current_count,
                    properties=tuple(current_properties),
                )
            )

        if format_type is None:
            raise DBSCANInputError("PLY header missing format declaration.")
        if format_type not in {"ascii", "binary_little_endian", "binary_big_endian"}:
            raise DBSCANInputError(f"Unsupported PLY format '{format_type}'.")

        vertex_columns: Optional[Dict[str, np.ndarray]] = None
        if format_type == "ascii":
            for element in elements:
                keep = element.name.lower() == "vertex"
                columns = _read_ascii_ply_element(stream, element, keep=keep)
                if keep:
                    vertex_columns = columns
        else:
            endian = "<" if format_type == "binary_little_endian" else ">"
            for element in elements:
                keep = element.name.lower() == "vertex"
                columns = _read_binary_ply_element(stream, element, endian=endian, keep=keep)
                if keep:
                    vertex_columns = columns

    if vertex_columns is None:
        raise DBSCANInputError("PLY file does not contain a vertex element.")
    return _extract_xyz_from_ply_columns(vertex_columns)


def _read_ascii_ply_element(
    stream,
    element: PlyElement,
    keep: bool,
) -> Optional[Dict[str, np.ndarray]]:
    scalar_data: Dict[str, List[float]] = {}
    if keep:
        for prop in element.properties:
            if prop.kind == "scalar":
                scalar_data[prop.name] = []

    for row_index in range(element.count):
        raw_line = stream.readline()
        if not raw_line:
            raise DBSCANInputError(
                f"Unexpected end of ASCII PLY while reading element '{element.name}'."
            )
        tokens = raw_line.decode("ascii", errors="ignore").strip().split()
        token_index = 0
        for prop in element.properties:
            if prop.kind == "scalar":
                if token_index >= len(tokens):
                    raise DBSCANInputError(
                        f"Malformed ASCII PLY row {row_index} in element '{element.name}'."
                    )
                if keep:
                    try:
                        scalar_data[prop.name].append(float(tokens[token_index]))
                    except ValueError as exc:
                        raise DBSCANInputError(
                            f"Invalid ASCII PLY numeric value in row {row_index}."
                        ) from exc
                token_index += 1
                continue

            if token_index >= len(tokens):
                raise DBSCANInputError(
                    f"Malformed ASCII PLY list property in row {row_index}."
                )
            try:
                list_count = int(float(tokens[token_index]))
            except ValueError as exc:
                raise DBSCANInputError(
                    f"Invalid ASCII PLY list count in row {row_index}."
                ) from exc
            token_index += 1 + list_count
            if token_index > len(tokens):
                raise DBSCANInputError(
                    f"ASCII PLY list property overflow in row {row_index}."
                )

    if not keep:
        return None
    return {
        name: np.asarray(values, dtype=np.float64)
        for name, values in scalar_data.items()
    }


def _read_binary_ply_element(
    stream,
    element: PlyElement,
    endian: str,
    keep: bool,
) -> Optional[Dict[str, np.ndarray]]:
    scalar_only = all(prop.kind == "scalar" for prop in element.properties)
    if scalar_only:
        dtype_fields = []
        for prop in element.properties:
            np_type = PLY_TO_NUMPY.get(prop.dtype)
            if np_type is None:
                raise DBSCANInputError(f"Unsupported binary PLY type '{prop.dtype}'.")
            dtype_fields.append((prop.name, np.dtype(np_type).newbyteorder(endian)))
        structured_dtype = np.dtype(dtype_fields)
        array = np.fromfile(stream, dtype=structured_dtype, count=element.count)
        if int(array.size) != int(element.count):
            raise DBSCANInputError(
                f"Unexpected EOF while reading binary PLY element '{element.name}'."
            )
        if not keep:
            return None
        return {
            name: np.asarray(array[name], dtype=np.float64)
            for name in array.dtype.names or ()
        }

    scalar_names = [prop.name for prop in element.properties if prop.kind == "scalar"]
    scalar_data = (
        {name: np.empty(element.count, dtype=np.float64) for name in scalar_names}
        if keep
        else {}
    )

    for row_index in range(element.count):
        for prop in element.properties:
            if prop.kind == "scalar":
                value = _read_binary_ply_value(stream, endian, prop.dtype)
                if keep:
                    scalar_data[prop.name][row_index] = float(value)
                continue

            if prop.count_dtype is None:
                raise DBSCANInputError(
                    f"Binary PLY list property '{prop.name}' is missing count type."
                )
            list_count = int(_read_binary_ply_value(stream, endian, prop.count_dtype))
            item_size = _ply_type_size(prop.dtype)
            skip_size = list_count * item_size
            skipped = stream.read(skip_size)
            if len(skipped) != skip_size:
                raise DBSCANInputError(
                    f"Unexpected EOF while reading binary PLY list data in row {row_index}."
                )

    return scalar_data if keep else None


def _read_binary_ply_value(stream, endian: str, ply_type: str):
    fmt_char = PLY_TO_STRUCT.get(ply_type)
    if fmt_char is None:
        raise DBSCANInputError(f"Unsupported PLY scalar type '{ply_type}'.")
    fmt = endian + fmt_char
    size = struct.calcsize(fmt)
    raw = stream.read(size)
    if len(raw) != size:
        raise DBSCANInputError("Unexpected EOF while reading binary PLY scalar.")
    return struct.unpack(fmt, raw)[0]


def _ply_type_size(ply_type: str) -> int:
    fmt_char = PLY_TO_STRUCT.get(ply_type)
    if fmt_char is None:
        raise DBSCANInputError(f"Unsupported PLY type '{ply_type}'.")
    return struct.calcsize(fmt_char)


def _extract_xyz_from_ply_columns(columns: Mapping[str, np.ndarray]) -> np.ndarray:
    normalized = {normalize_field_name(name): name for name in columns.keys()}

    def pick(*keys: str) -> Optional[str]:
        for key in keys:
            name = normalized.get(key)
            if name is not None:
                return name
        return None

    x_name = pick("x")
    y_name = pick("y")
    z_name = pick("z")
    if x_name is None or y_name is None or z_name is None:
        raise DBSCANInputError("PLY vertex data must contain x, y and z fields.")

    points = np.stack(
        [columns[x_name], columns[y_name], columns[z_name]],
        axis=1,
    ).astype(np.float64, copy=False)
    return _validate_points(points)


def _extract_xyz_from_matrix(matrix: np.ndarray, header_tokens: Optional[List[str]]) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[1] < 3:
        raise DBSCANInputError("Point matrix must contain at least 3 columns for XYZ.")

    xyz_indices = [0, 1, 2]
    if header_tokens is not None and len(header_tokens) == matrix.shape[1]:
        indexed = {normalize_field_name(name): index for index, name in enumerate(header_tokens)}
        if {"x", "y", "z"}.issubset(indexed):
            xyz_indices = [indexed["x"], indexed["y"], indexed["z"]]
    points = matrix[:, xyz_indices].astype(np.float64, copy=False)
    return _validate_points(points)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DBSCAN clustering on a TXT/PLY point cloud.")
    parser.add_argument("input_cloud", help="Path to the input point cloud file (.txt or .ply).")
    parser.add_argument(
        "--epsilon",
        type=float,
        required=True,
        help="Neighborhood radius for DBSCAN.",
    )
    parser.add_argument(
        "--min-pts",
        dest="min_pts",
        type=int,
        required=True,
        help="Minimum number of points inside the epsilon-neighborhood.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output YAML file for the clustering result.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    try:
        result = run_dbscan_on_file(
            input_path=args.input_cloud,
            epsilon=args.epsilon,
            min_pts=args.min_pts,
            output_path=args.output,
        )
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"DBSCAN failed: {exc}") from exc

    output_path = Path(args.output)
    if output_path.suffix.lower() not in {".yaml", ".yml"}:
        output_path = output_path.with_suffix(".yaml")

    cluster_counts = [cluster.point_count for cluster in result.clusters]
    if cluster_counts:
        counts_text = ", ".join(str(value) for value in cluster_counts)
    else:
        counts_text = "none"

    print(f"Input points: {result.source_point_count}")
    print(f"Clusters found: {len(result.clusters)}")
    print(f"Cluster point counts: {counts_text}")
    print(f"Saved YAML: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
