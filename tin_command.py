"""Helper layer for running the TIN algorithm from the GUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from utils.tin_alg import TINMesh, TINParameters, normalize_tin_parameters


RENDER_MODE_VALUES = ("wireframe", "solid", "shaded")
ELEVATION_COLORMAP_VALUES = ("terrain", "viridis", "plasma", "grayscale")


@dataclass(frozen=True)
class TINVisualSettings:
    """Viewport settings used to display a generated TIN mesh."""

    render_mode: str = "shaded"
    elevation_colormap: str = "terrain"
    smooth_normals: bool = True


@dataclass(frozen=True)
class TINCommandParams:
    """Combined algorithm and visualization settings for the GUI TIN command."""

    algorithm: TINParameters = field(default_factory=TINParameters)
    visual: TINVisualSettings = field(default_factory=TINVisualSettings)
    custom_boundary_path: str = ""


@dataclass(frozen=True)
class TINCommandResult:
    """TIN command output returned to the GUI after a successful build."""

    mesh: TINMesh
    params: TINCommandParams
    summary: str


def normalize_visual_settings(
    settings: TINVisualSettings | Mapping[str, object] | None = None,
    **overrides: object,
) -> TINVisualSettings:
    """Validate and normalize the mesh display settings used by the GUI."""
    raw: dict[str, object] = {}
    if settings is not None:
        if isinstance(settings, TINVisualSettings):
            raw.update(settings.__dict__)
        elif isinstance(settings, Mapping):
            raw.update(settings)
        else:
            raise ValueError("`settings` must be a TINVisualSettings instance, mapping, or None.")
    raw.update({key: value for key, value in overrides.items() if value is not None})

    render_mode = str(raw.get("render_mode", TINVisualSettings.render_mode)).strip().lower()
    if render_mode not in RENDER_MODE_VALUES:
        raise ValueError(f"`render_mode` must be one of {', '.join(RENDER_MODE_VALUES)}.")

    elevation_colormap = str(
        raw.get("elevation_colormap", TINVisualSettings.elevation_colormap)
    ).strip().lower()
    if elevation_colormap not in ELEVATION_COLORMAP_VALUES:
        raise ValueError(
            "`elevation_colormap` must be one of "
            f"{', '.join(ELEVATION_COLORMAP_VALUES)}."
        )

    smooth_normals = bool(raw.get("smooth_normals", TINVisualSettings.smooth_normals))
    return TINVisualSettings(
        render_mode=render_mode,
        elevation_colormap=elevation_colormap,
        smooth_normals=smooth_normals,
    )


def normalize_command_params(
    params: TINCommandParams | Mapping[str, object] | None = None,
    **overrides: object,
) -> TINCommandParams:
    """Validate and normalize the complete GUI configuration for the TIN command."""
    raw: dict[str, object] = {}
    if params is not None:
        if isinstance(params, TINCommandParams):
            raw.update(params.__dict__)
        elif isinstance(params, Mapping):
            raw.update(params)
        else:
            raise ValueError("`params` must be a TINCommandParams instance, mapping, or None.")
    raw.update({key: value for key, value in overrides.items() if value is not None})

    algorithm_raw = raw.get("algorithm")
    visual_raw = raw.get("visual")
    custom_boundary_path = str(raw.get("custom_boundary_path", "")).strip()
    return TINCommandParams(
        algorithm=normalize_tin_parameters(algorithm_raw),
        visual=normalize_visual_settings(visual_raw),
        custom_boundary_path=custom_boundary_path,
    )


def execute_tin_for_points(
    points: np.ndarray,
    params: TINCommandParams | Mapping[str, object] | None = None,
    *,
    tin_module=None,
) -> TINCommandResult:
    """
    Run the TIN algorithm for an in-memory cloud and return mesh + GUI display settings.

    Parameters
    ----------
    points:
        XYZ point array from the currently loaded cloud.
    params:
        Combined GUI parameters including algorithm and visualization settings.
    tin_module:
        Optional pre-imported ``utils.tin_alg`` module. The GUI passes this to
        avoid repeated imports and to surface import failures early.
    """
    normalized = normalize_command_params(params)
    custom_boundary = normalized.custom_boundary_path or None
    if custom_boundary is not None and not Path(custom_boundary).is_file():
        raise FileNotFoundError(f"Custom boundary file not found: {custom_boundary}")

    module = tin_module
    if module is None:
        from utils import tin_alg as module

    mesh = module.build_tin_from_points(
        points=points,
        params=normalized.algorithm,
        custom_boundary=custom_boundary,
    )
    summary = (
        f"Vertices: {mesh.vertices.shape[0]:,} | "
        f"Triangles: {mesh.triangles.shape[0]:,} | "
        f"Processed points: {mesh.processed_point_count:,}"
    )
    return TINCommandResult(mesh=mesh, params=normalized, summary=summary)
