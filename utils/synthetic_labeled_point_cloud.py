#!/usr/bin/env python3
"""
Synthetic labeled landscape point cloud generator.

The script generates a procedural scene where each point has:
    x, y, z, label

Labels:
    0 - artificial surface
    1 - natural surface
    2 - high vegetation
    3 - low vegetation
    4 - buildings
    5 - structures
    6 - artifacts
    7 - vehicles
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence as SequenceABC
from pathlib import Path
import sys
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import yaml


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

ProgressCallback = Callable[[float, str], None]


def _emit_progress(
    progress_callback: ProgressCallback | None,
    progress: float,
    stage: str = "",
) -> None:
    if progress_callback is None:
        return
    progress_callback(
        float(min(1.0, max(0.0, float(progress)))),
        str(stage).strip(),
    )


def _make_progress_subrange(
    progress_callback: ProgressCallback | None,
    start: float,
    end: float,
) -> ProgressCallback | None:
    if progress_callback is None:
        return None

    start_f = float(start)
    end_f = max(start_f, float(end))

    def _callback(progress: float, stage: str = "") -> None:
        clamped = float(min(1.0, max(0.0, float(progress))))
        mapped = start_f + (end_f - start_f) * clamped
        _emit_progress(progress_callback, mapped, stage)

    return _callback


def ensure_data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


ARTIFICIAL_SURFACE_CLASS_ID = 0
NATURAL_SURFACE_CLASS_ID = 1
HIGH_VEGETATION_CLASS_ID = 2
LOW_VEGETATION_CLASS_ID = 3
BUILDINGS_CLASS_ID = 4
STRUCTURES_CLASS_ID = 5
ARTIFACTS_CLASS_ID = 6
VEHICLES_CLASS_ID = 7


CLASS_NAMES: Dict[int, str] = {
    ARTIFICIAL_SURFACE_CLASS_ID: "Artificial surface",
    NATURAL_SURFACE_CLASS_ID: "Natural surface",
    HIGH_VEGETATION_CLASS_ID: "High vegetation",
    LOW_VEGETATION_CLASS_ID: "Low vegetation",
    BUILDINGS_CLASS_ID: "Buildings",
    STRUCTURES_CLASS_ID: "Structures",
    ARTIFACTS_CLASS_ID: "Artifacts",
    VEHICLES_CLASS_ID: "Vehicles",
}

CLASS_COLORS: Dict[int, Tuple[float, float, float]] = {
    ARTIFICIAL_SURFACE_CLASS_ID: (0.35, 0.35, 0.35),  # asphalt gray
    NATURAL_SURFACE_CLASS_ID: (0.55, 0.39, 0.22),  # brown
    HIGH_VEGETATION_CLASS_ID: (0.05, 0.45, 0.12),  # dark green
    LOW_VEGETATION_CLASS_ID: (0.35, 0.75, 0.30),  # light green
    BUILDINGS_CLASS_ID: (0.82, 0.22, 0.18),  # red
    STRUCTURES_CLASS_ID: (0.85, 0.68, 0.20),  # yellow
    ARTIFACTS_CLASS_ID: (0.68, 0.18, 0.72),  # magenta
    VEHICLES_CLASS_ID: (0.15, 0.40, 0.85),  # blue
}

CLASS_IDS: Tuple[int, ...] = tuple(sorted(CLASS_NAMES))
DEFAULT_CLASS_RATIOS: Dict[int, float] = {
    ARTIFICIAL_SURFACE_CLASS_ID: 0.13,
    NATURAL_SURFACE_CLASS_ID: 0.38,
    HIGH_VEGETATION_CLASS_ID: 0.14,
    LOW_VEGETATION_CLASS_ID: 0.16,
    BUILDINGS_CLASS_ID: 0.10,
    STRUCTURES_CLASS_ID: 0.04,
    ARTIFACTS_CLASS_ID: 0.02,
    VEHICLES_CLASS_ID: 0.03,
}
DEFAULT_CLASS_PERCENTAGES: Tuple[float, ...] = tuple(
    DEFAULT_CLASS_RATIOS[class_id] * 100.0 for class_id in CLASS_IDS
)

CLASS_LABELS_CONFIG_KEY = "class_labels"
CLASS_GENERATION_ORDER_CONFIG_KEY = "class_generation_order"
PLY_LABEL_COMMENT_PREFIX = "magicpoints.class_label"
PLY_GENERATION_ORDER_COMMENT_PREFIX = "magicpoints.class_generation_order"

CLASS_GENERATION_ORDER: Tuple[int, ...] = (
    BUILDINGS_CLASS_ID,
    ARTIFICIAL_SURFACE_CLASS_ID,
    HIGH_VEGETATION_CLASS_ID,
    STRUCTURES_CLASS_ID,
    VEHICLES_CLASS_ID,
    LOW_VEGETATION_CLASS_ID,
    ARTIFACTS_CLASS_ID,
    NATURAL_SURFACE_CLASS_ID,
)

CLASS_GENERATION_RULES: Dict[int, str] = {
    ARTIFICIAL_SURFACE_CLASS_ID: (
        "Sampled only inside generated artificial-surface zones; natural-surface sampling is masked out there."
    ),
    NATURAL_SURFACE_CLASS_ID: (
        "Sampled after object placement and only outside artificial-surface zones."
    ),
    HIGH_VEGETATION_CLASS_ID: (
        "Trees are placed outside artificial-surface zones and away from building footprints."
    ),
    LOW_VEGETATION_CLASS_ID: (
        "Shrubs and grass patches avoid artificial-surface zones and building footprints."
    ),
    BUILDINGS_CLASS_ID: (
        "Placed before artificial-surface point sampling to anchor sidewalks/front areas; footprints avoid detached artificial surfaces and other buildings."
    ),
    STRUCTURES_CLASS_ID: (
        "Placed after buildings and vegetation; footprints avoid building footprints."
    ),
    ARTIFACTS_CLASS_ID: (
        "Generated last from scene geometry and acquisition-error models; final class share can override the base class distribution."
    ),
    VEHICLES_CLASS_ID: (
        "Placed only on vehicle-allowed artificial-surface zones and kept separated from other vehicles."
    ),
}


def format_class_label_mapping(class_names: Mapping[int, str] | None = None) -> str:
    source = class_names if isinstance(class_names, Mapping) else CLASS_NAMES
    return ", ".join(
        f"{class_id}={source.get(class_id, CLASS_NAMES[class_id])}" for class_id in CLASS_IDS
    )


def format_class_generation_order(
    order: Sequence[int] | None = None,
    class_names: Mapping[int, str] | None = None,
) -> str:
    source = class_names if isinstance(class_names, Mapping) else CLASS_NAMES
    effective_order = tuple(int(class_id) for class_id in (order or CLASS_GENERATION_ORDER))
    return " -> ".join(
        f"{class_id} ({source.get(class_id, CLASS_NAMES.get(class_id, f'Class {class_id}'))})"
        for class_id in effective_order
    )

ARTIFICIAL_SURFACE_TYPES: Tuple[str, ...] = (
    "road_network",
    "sidewalk",
    "parking_lot",
    "building_front_area",
    "industrial_concrete_pad",
    "platform",
)
ARTIFICIAL_SURFACE_TYPE_NAMES: Dict[str, str] = {
    "road_network": "Road network",
    "sidewalk": "Sidewalks along buildings",
    "parking_lot": "Parking lots",
    "building_front_area": "Areas in front of buildings",
    "industrial_concrete_pad": "Industrial concrete pads",
    "platform": "Platforms / station aprons",
}
DEFAULT_ARTIFICIAL_SURFACE_TYPE_RATIOS: Dict[str, float] = {
    "road_network": 0.30,
    "sidewalk": 0.22,
    "parking_lot": 0.16,
    "building_front_area": 0.10,
    "industrial_concrete_pad": 0.12,
    "platform": 0.10,
}
DEFAULT_ARTIFICIAL_SURFACE_TYPE_PERCENTAGES: Tuple[float, ...] = tuple(
    DEFAULT_ARTIFICIAL_SURFACE_TYPE_RATIOS[surface_type] * 100.0
    for surface_type in ARTIFICIAL_SURFACE_TYPES
)

VEHICLE_TYPES: Tuple[str, ...] = ("car", "truck", "bus")
VEHICLE_TYPE_NAMES: Dict[str, str] = {
    "car": "Passenger car",
    "truck": "Truck",
    "bus": "Bus",
}
DEFAULT_VEHICLE_TYPE_RATIOS: Dict[str, float] = {
    "car": 0.72,
    "truck": 0.18,
    "bus": 0.10,
}
DEFAULT_VEHICLE_TYPE_PERCENTAGES: Tuple[float, ...] = tuple(
    DEFAULT_VEHICLE_TYPE_RATIOS[vehicle_type] * 100.0 for vehicle_type in VEHICLE_TYPES
)

ARTIFACT_DEFAULTS: Dict[str, float | bool] = {
    "enabled": True,
    "global_intensity": 0.65,
    "point_fraction": DEFAULT_CLASS_RATIOS[ARTIFACTS_CLASS_ID],
}
ARTIFACT_TYPE_SPECS: Tuple[Dict[str, Any], ...] = (
    {
        "key": "random_outliers",
        "label": "Random outliers",
        "tooltip": "Sparse isolated returns and compact outlier clusters detached from real geometry.",
        "default_enabled": True,
        "default_intensity": 0.72,
        "default_amount": 0.24,
        "params": (
            {
                "key": "spread",
                "label": "Spread",
                "default": 0.85,
                "min": 0.05,
                "max": 4.0,
                "step": 0.05,
                "decimals": 2,
                "suffix": " m",
                "tooltip": "Maximum spatial spread of isolated outlier clusters in meters.",
            },
        ),
    },
    {
        "key": "surface_noise",
        "label": "Surface noise",
        "tooltip": "Noisy halo around terrain, roads and object surfaces caused by unstable range measurements.",
        "default_enabled": True,
        "default_intensity": 0.68,
        "default_amount": 0.22,
        "params": (
            {
                "key": "thickness",
                "label": "Thickness",
                "default": 0.16,
                "min": 0.01,
                "max": 1.5,
                "step": 0.01,
                "decimals": 3,
                "suffix": " m",
                "tooltip": "Approximate thickness of the noisy shell around sampled surfaces.",
            },
        ),
    },
    {
        "key": "hanging_points",
        "label": "Hanging points",
        "tooltip": "Floating points above objects or terrain that do not connect to any real surface.",
        "default_enabled": True,
        "default_intensity": 0.58,
        "default_amount": 0.14,
        "params": (
            {
                "key": "height",
                "label": "Height range",
                "default": 2.4,
                "min": 0.10,
                "max": 12.0,
                "step": 0.10,
                "decimals": 2,
                "suffix": " m",
                "tooltip": "Maximum elevation offset of floating points above nearby geometry.",
            },
        ),
    },
    {
        "key": "ghost_double_points",
        "label": "Ghost / double points",
        "tooltip": "Duplicated returns slightly offset from the original surface, similar to multi-path or double echoes.",
        "default_enabled": True,
        "default_intensity": 0.62,
        "default_amount": 0.14,
        "params": (
            {
                "key": "offset",
                "label": "Offset",
                "default": 0.18,
                "min": 0.02,
                "max": 1.5,
                "step": 0.01,
                "decimals": 3,
                "suffix": " m",
                "tooltip": "Typical offset between the original return and its ghost copy.",
            },
        ),
    },
    {
        "key": "blurred_boundaries",
        "label": "Blurred boundaries",
        "tooltip": "Fuzzy transitions near the edges of roads, buildings and other sharp boundaries.",
        "default_enabled": True,
        "default_intensity": 0.57,
        "default_amount": 0.16,
        "params": (
            {
                "key": "width",
                "label": "Boundary width",
                "default": 0.45,
                "min": 0.05,
                "max": 4.0,
                "step": 0.05,
                "decimals": 2,
                "suffix": " m",
                "tooltip": "Width of the transition strip sampled around geometry boundaries.",
            },
        ),
    },
    {
        "key": "false_reflections",
        "label": "False reflections",
        "tooltip": "Spurious reflected returns near vertical structures, vehicles or strong reflective surfaces.",
        "default_enabled": True,
        "default_intensity": 0.52,
        "default_amount": 0.10,
        "params": (
            {
                "key": "height",
                "label": "Reflection height",
                "default": 1.80,
                "min": 0.10,
                "max": 10.0,
                "step": 0.10,
                "decimals": 2,
                "suffix": " m",
                "tooltip": "Maximum vertical displacement of reflected returns above the anchor geometry.",
            },
        ),
    },
)
ARTIFACT_TYPES: Tuple[str, ...] = tuple(str(spec["key"]) for spec in ARTIFACT_TYPE_SPECS)
ARTIFACT_TYPE_NAMES: Dict[str, str] = {
    str(spec["key"]): str(spec["label"]) for spec in ARTIFACT_TYPE_SPECS
}
ARTIFACT_TYPE_TOOLTIPS: Dict[str, str] = {
    str(spec["key"]): str(spec["tooltip"]) for spec in ARTIFACT_TYPE_SPECS
}
ARTIFACT_TYPE_SPEC_BY_KEY: Dict[str, Dict[str, Any]] = {
    str(spec["key"]): dict(spec) for spec in ARTIFACT_TYPE_SPECS
}


def _default_artifact_type_settings() -> Dict[str, Dict[str, float | bool]]:
    settings: Dict[str, Dict[str, float | bool]] = {}
    for spec in ARTIFACT_TYPE_SPECS:
        artifact_key = str(spec["key"])
        entry: Dict[str, float | bool] = {
            "enabled": bool(spec["default_enabled"]),
            "intensity": float(spec["default_intensity"]),
            "amount": float(spec["default_amount"]),
        }
        for param_spec in spec.get("params", ()):
            entry[str(param_spec["key"])] = float(param_spec["default"])
        settings[artifact_key] = entry
    return settings


DEFAULT_ARTIFACT_TYPE_SETTINGS = _default_artifact_type_settings()

LOW_VEG_DEFAULTS: Dict[str, float] = {
    "shrub_max_diameter": 2.6,
    "shrub_max_top_height": 1.8,
    "shrub_min_bottom_height": 0.12,
    "grass_patch_max_size_x": 3.8,
    "grass_patch_max_size_y": 3.4,
    "grass_max_height": 0.65,
}

BUILDING_ROOF_TYPES: Tuple[str, ...] = (
    "single_slope",
    "gable",
    "hip",
    "tent",
    "mansard",
    "flat",
    "dome",
    "arched",
    "shell",
)
BUILDING_ROOF_TYPE_NAMES: Dict[str, str] = {
    "single_slope": "Single-slope",
    "gable": "Gable",
    "hip": "Hip",
    "tent": "Tent",
    "mansard": "Mansard",
    "flat": "Flat",
    "dome": "Dome",
    "arched": "Arched",
    "shell": "Shell",
}
DEFAULT_BUILDING_ROOF_TYPE_RATIOS: Dict[str, float] = {
    "single_slope": 0.10,
    "gable": 0.18,
    "hip": 0.14,
    "tent": 0.08,
    "mansard": 0.10,
    "flat": 0.20,
    "dome": 0.06,
    "arched": 0.08,
    "shell": 0.06,
}
DEFAULT_BUILDING_ROOF_TYPE_PERCENTAGES: Tuple[float, ...] = tuple(
    DEFAULT_BUILDING_ROOF_TYPE_RATIOS[roof_type] * 100.0 for roof_type in BUILDING_ROOF_TYPES
)
BUILDING_DEFAULTS: Dict[str, int | bool] = {
    "building_floor_min": 2,
    "building_floor_max": 9,
    "building_random_yaw": True,
}

STRUCTURE_TYPES: Tuple[str, ...] = (
    "fence",
    "railing",
    "enclosure",
    "guardrail",
    "retaining_wall",
    "parapet",
    "stone_wall",
    "pole_support",
    "lamp",
    "road_sign",
    "traffic_light",
    "bench",
    "trash_bin",
    "bike_rack",
    "bollard",
    "fountain",
    "pedestal",
    "monument",
    "stairs",
    "ramp",
    "platform",
    "footbridge",
)
STRUCTURE_TYPE_NAMES: Dict[str, str] = {
    "fence": "Fence",
    "railing": "Railing",
    "enclosure": "Enclosure",
    "guardrail": "Guardrail",
    "retaining_wall": "Retaining wall",
    "parapet": "Parapet",
    "stone_wall": "Low stone wall",
    "pole_support": "Pole / support",
    "lamp": "Lamp",
    "road_sign": "Road sign",
    "traffic_light": "Traffic light",
    "bench": "Bench",
    "trash_bin": "Trash bin",
    "bike_rack": "Bike rack",
    "bollard": "Bollard",
    "fountain": "Fountain",
    "pedestal": "Pedestal",
    "monument": "Small monument",
    "stairs": "Stairs",
    "ramp": "Ramp",
    "platform": "Platform",
    "footbridge": "Open footbridge",
}
DEFAULT_STRUCTURE_TYPE_RATIOS: Dict[str, float] = {
    "fence": 0.12,
    "railing": 0.07,
    "enclosure": 0.06,
    "guardrail": 0.06,
    "retaining_wall": 0.06,
    "parapet": 0.04,
    "stone_wall": 0.05,
    "pole_support": 0.07,
    "lamp": 0.06,
    "road_sign": 0.05,
    "traffic_light": 0.03,
    "bench": 0.05,
    "trash_bin": 0.04,
    "bike_rack": 0.03,
    "bollard": 0.03,
    "fountain": 0.02,
    "pedestal": 0.02,
    "monument": 0.02,
    "stairs": 0.03,
    "ramp": 0.02,
    "platform": 0.03,
    "footbridge": 0.04,
}
DEFAULT_STRUCTURE_TYPE_PERCENTAGES: Tuple[float, ...] = tuple(
    DEFAULT_STRUCTURE_TYPE_RATIOS[structure_type] * 100.0
    for structure_type in STRUCTURE_TYPES
)

TREE_CROWN_TYPES: Tuple[str, ...] = (
    "spherical",
    "pyramidal",
    "spreading",
    "weeping",
    "columnar",
    "umbrella",
)
TREE_CROWN_TYPE_NAMES: Dict[str, str] = {
    "spherical": "Spherical",
    "pyramidal": "Pyramidal",
    "spreading": "Spreading",
    "weeping": "Weeping",
    "columnar": "Columnar",
    "umbrella": "Umbrella",
}
DEFAULT_TREE_CROWN_TYPE_RATIOS: Dict[str, float] = {
    "spherical": 0.28,
    "pyramidal": 0.18,
    "spreading": 0.20,
    "weeping": 0.10,
    "columnar": 0.12,
    "umbrella": 0.12,
}
DEFAULT_TREE_CROWN_TYPE_PERCENTAGES: Tuple[float, ...] = tuple(
    DEFAULT_TREE_CROWN_TYPE_RATIOS[crown_type] * 100.0 for crown_type in TREE_CROWN_TYPES
)
HIGH_VEG_DEFAULTS: Dict[str, float] = {
    "tree_max_crown_diameter": 5.2,
    "tree_max_crown_top_height": 9.5,
    "tree_min_crown_bottom_height": 1.3,
}

GENERATION_CONFIG_SCHEMA = "magicpoints.synthetic_generation/v1"

Rect = Tuple[float, float, float, float]  # center_x, center_y, size_x, size_y


def default_generation_config() -> Dict[str, Any]:
    """Return normalized default generation settings used by the GUI and CLI config flow."""
    return {
        "total_points": 100_000,
        "area_width": 240.0,
        "area_length": 220.0,
        "terrain_relief": 1.0,
        "seed": 12,
        "randomize_object_counts": True,
        "custom_class_distribution": False,
        "class_percentages": tuple(DEFAULT_CLASS_PERCENTAGES),
        "custom_artificial_surface_count": False,
        "artificial_surface_count": 9,
        "custom_artificial_surface_type_distribution": False,
        "artificial_surface_type_percentages": tuple(
            DEFAULT_ARTIFICIAL_SURFACE_TYPE_PERCENTAGES
        ),
        "custom_building_count": False,
        "building_count": 14,
        "custom_building_roof_type_distribution": False,
        "building_roof_type_percentages": tuple(DEFAULT_BUILDING_ROOF_TYPE_PERCENTAGES),
        "building_floor_min": int(BUILDING_DEFAULTS["building_floor_min"]),
        "building_floor_max": int(BUILDING_DEFAULTS["building_floor_max"]),
        "building_random_yaw": bool(BUILDING_DEFAULTS["building_random_yaw"]),
        "custom_structure_count": False,
        "structure_count": 10,
        "custom_structure_type_distribution": False,
        "structure_type_percentages": tuple(DEFAULT_STRUCTURE_TYPE_PERCENTAGES),
        "custom_tree_count": False,
        "tree_count": 70,
        "custom_tree_crown_type_distribution": False,
        "tree_crown_type_percentages": tuple(DEFAULT_TREE_CROWN_TYPE_PERCENTAGES),
        "random_tree_crown_size": True,
        "tree_max_crown_diameter": HIGH_VEG_DEFAULTS["tree_max_crown_diameter"],
        "tree_max_crown_top_height": HIGH_VEG_DEFAULTS["tree_max_crown_top_height"],
        "tree_min_crown_bottom_height": HIGH_VEG_DEFAULTS["tree_min_crown_bottom_height"],
        "custom_vehicle_count": False,
        "vehicle_count": 24,
        "custom_vehicle_type_distribution": False,
        "vehicle_type_percentages": tuple(DEFAULT_VEHICLE_TYPE_PERCENTAGES),
        "artifacts_enabled": bool(ARTIFACT_DEFAULTS["enabled"]),
        "artifact_global_intensity": float(ARTIFACT_DEFAULTS["global_intensity"]),
        "artifact_point_fraction": float(ARTIFACT_DEFAULTS["point_fraction"]),
        "artifact_type_settings": _default_artifact_type_settings(),
        "custom_shrub_count": False,
        "shrub_count": 24,
        "random_shrub_size": True,
        "shrub_max_diameter": LOW_VEG_DEFAULTS["shrub_max_diameter"],
        "shrub_max_top_height": LOW_VEG_DEFAULTS["shrub_max_top_height"],
        "shrub_min_bottom_height": LOW_VEG_DEFAULTS["shrub_min_bottom_height"],
        "custom_grass_patch_count": False,
        "grass_patch_count": 18,
        "random_grass_patch_size": True,
        "grass_patch_max_size_x": LOW_VEG_DEFAULTS["grass_patch_max_size_x"],
        "grass_patch_max_size_y": LOW_VEG_DEFAULTS["grass_patch_max_size_y"],
        "grass_max_height": LOW_VEG_DEFAULTS["grass_max_height"],
    }


def _coerce_bool(value: Any, name: str) -> bool:
    """Coerce YAML scalar values into booleans with clear validation errors."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        if int(value) in {0, 1}:
            return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"`{name}` must be a boolean.")


def _coerce_int(value: Any, name: str, *, min_value: int | None = None) -> int:
    """Coerce YAML numeric values into integers and enforce optional lower bounds."""
    parsed: int
    if isinstance(value, bool):
        raise ValueError(f"`{name}` must be an integer, got boolean.")
    if isinstance(value, (int, np.integer)):
        parsed = int(value)
    elif isinstance(value, (float, np.floating)):
        float_value = float(value)
        if not np.isfinite(float_value) or not float_value.is_integer():
            raise ValueError(f"`{name}` must be an integer, got {value}.")
        parsed = int(float_value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"`{name}` must be an integer.")
        try:
            float_value = float(stripped)
        except ValueError as exc:
            raise ValueError(f"`{name}` must be an integer, got {value}.") from exc
        if not np.isfinite(float_value) or not float_value.is_integer():
            raise ValueError(f"`{name}` must be an integer, got {value}.")
        parsed = int(float_value)
    else:
        raise ValueError(f"`{name}` must be an integer.")

    if min_value is not None and parsed < min_value:
        raise ValueError(f"`{name}` must be >= {min_value}, got {parsed}.")
    return parsed


def _coerce_float(
    value: Any,
    name: str,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    """Coerce YAML numeric values into finite floats and enforce optional bounds."""
    if isinstance(value, bool):
        raise ValueError(f"`{name}` must be a number, got boolean.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"`{name}` must be a number.") from exc

    if not np.isfinite(parsed):
        raise ValueError(f"`{name}` must be finite, got {value}.")
    if min_value is not None and parsed < min_value:
        raise ValueError(f"`{name}` must be >= {min_value}, got {parsed}.")
    if max_value is not None and parsed > max_value:
        raise ValueError(f"`{name}` must be <= {max_value}, got {parsed}.")
    return parsed


def _parse_class_id_key(raw_key: Any, name: str) -> int:
    try:
        class_id = int(raw_key)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"`{name}` contains invalid class id key `{raw_key}`.") from exc
    if class_id not in CLASS_IDS:
        raise ValueError(
            f"`{name}` contains unknown class id `{raw_key}`. Supported ids: {list(CLASS_IDS)}."
        )
    return class_id


def _validate_class_labels_metadata(value: Any, name: str = CLASS_LABELS_CONFIG_KEY) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise ValueError(f"`{name}` must be a mapping.")

    parsed: Dict[int, str] = {}
    for raw_key, raw_value in value.items():
        class_id = _parse_class_id_key(raw_key, name)
        label = str(raw_value).strip()
        if not label:
            raise ValueError(f"`{name}[{raw_key!r}]` must be a non-empty string.")
        parsed[class_id] = label

    missing_ids = [class_id for class_id in CLASS_IDS if class_id not in parsed]
    if missing_ids:
        raise ValueError(
            f"`{name}` must provide labels for every class id. Missing ids: {missing_ids}."
        )

    mismatched = {
        class_id: (parsed[class_id], CLASS_NAMES[class_id])
        for class_id in CLASS_IDS
        if parsed[class_id] != CLASS_NAMES[class_id]
    }
    if mismatched:
        details = ", ".join(
            f"{class_id}: expected '{expected}', got '{actual}'"
            for class_id, (actual, expected) in mismatched.items()
        )
        raise ValueError(f"`{name}` does not match the current class label mapping: {details}.")


def _validate_class_generation_order_metadata(
    value: Any,
    name: str = CLASS_GENERATION_ORDER_CONFIG_KEY,
) -> None:
    if value is None:
        return
    if not isinstance(value, SequenceABC) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"`{name}` must be a YAML sequence.")

    parsed = tuple(_parse_class_id_key(raw_value, name) for raw_value in value)
    if len(parsed) != len(CLASS_GENERATION_ORDER):
        raise ValueError(
            f"`{name}` must contain exactly {len(CLASS_GENERATION_ORDER)} class ids, got {len(parsed)}."
        )
    if parsed != CLASS_GENERATION_ORDER:
        raise ValueError(
            f"`{name}` must match the current generation order {list(CLASS_GENERATION_ORDER)}, got {list(parsed)}."
        )


def _parse_choice_key(raw_key: Any, name: str, allowed_keys: Sequence[str]) -> str:
    key = str(raw_key).strip().lower()
    if key not in allowed_keys:
        raise ValueError(
            f"`{name}` contains unknown key `{raw_key}`. Supported keys: {list(allowed_keys)}."
        )
    return key


def _coerce_percentage_values(
    value: Any,
    name: str,
    *,
    ordered_keys: Sequence[Any],
    key_parser: Callable[[Any, str], Any],
) -> Tuple[float, ...]:
    if isinstance(value, Mapping):
        parsed_values: Dict[Any, float] = {}
        for raw_key, raw_value in value.items():
            key = key_parser(raw_key, name)
            parsed_values[key] = _coerce_float(
                raw_value,
                f"{name}[{raw_key!r}]",
                min_value=0.0,
                max_value=100.0,
            )
        return tuple(float(parsed_values.get(key, 0.0)) for key in ordered_keys)

    if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes, bytearray)):
        values = [
            _coerce_float(item, f"{name}[{index}]", min_value=0.0, max_value=100.0)
            for index, item in enumerate(value)
        ]
        if len(values) != len(ordered_keys):
            raise ValueError(
                f"`{name}` must contain exactly {len(ordered_keys)} values, got {len(values)}."
            )
        return tuple(values)

    raise ValueError(f"`{name}` must be a YAML sequence or mapping.")


def _validate_enabled_percentage_distribution(
    percentages: Sequence[float],
    name: str,
) -> None:
    weights = np.asarray(percentages, dtype=np.float64)
    total = float(weights.sum())
    if not np.any(weights > 0.0):
        raise ValueError(f"`{name}` must contain at least one value > 0.")
    if abs(total - 100.0) > 0.01:
        raise ValueError(f"`{name}` must sum to 100.0, got {total:.2f}.")


def _copy_artifact_type_settings(
    settings: Mapping[str, Mapping[str, Any]] | None = None,
) -> Dict[str, Dict[str, float | bool]]:
    source = settings if isinstance(settings, Mapping) else DEFAULT_ARTIFACT_TYPE_SETTINGS
    copied: Dict[str, Dict[str, float | bool]] = {}
    for spec in ARTIFACT_TYPE_SPECS:
        artifact_key = str(spec["key"])
        source_entry = source.get(artifact_key, {}) if isinstance(source, Mapping) else {}
        if not isinstance(source_entry, Mapping):
            source_entry = {}
        default_entry = DEFAULT_ARTIFACT_TYPE_SETTINGS[artifact_key]
        entry: Dict[str, float | bool] = {
            "enabled": bool(source_entry.get("enabled", default_entry["enabled"])),
            "intensity": float(source_entry.get("intensity", default_entry["intensity"])),
            "amount": float(source_entry.get("amount", default_entry["amount"])),
        }
        for param_spec in spec.get("params", ()):
            param_key = str(param_spec["key"])
            entry[param_key] = float(source_entry.get(param_key, default_entry[param_key]))
        copied[artifact_key] = entry
    return copied


def _normalize_artifact_type_settings_payload(
    direct_settings: Any,
    nested_settings: Any,
    *,
    defaults: Mapping[str, Mapping[str, Any]] | None = None,
) -> Dict[str, Dict[str, float | bool]]:
    defaults_map = _copy_artifact_type_settings(defaults)

    def _normalize_source_entries(raw_value: Any, field_name: str) -> Dict[str, Dict[str, Any]]:
        if raw_value is None:
            return {}
        if not isinstance(raw_value, Mapping):
            raise ValueError(f"`{field_name}` must be a mapping.")

        entries: Dict[str, Dict[str, Any]] = {}
        for raw_key, raw_entry in raw_value.items():
            artifact_key = _parse_choice_key(raw_key, field_name, ARTIFACT_TYPES)
            if raw_entry is None:
                entries[artifact_key] = {}
                continue
            if not isinstance(raw_entry, Mapping):
                raise ValueError(f"`{field_name}[{raw_key!r}]` must be a mapping.")
            normalized_entry: Dict[str, Any] = {}
            for entry_key, entry_value in raw_entry.items():
                normalized_entry[str(entry_key).strip().lower()] = entry_value
            entries[artifact_key] = normalized_entry
        return entries

    direct_entries = _normalize_source_entries(direct_settings, "artifact_type_settings")
    nested_entries = _normalize_source_entries(nested_settings, "artifacts.types")

    normalized: Dict[str, Dict[str, float | bool]] = {}
    for spec in ARTIFACT_TYPE_SPECS:
        artifact_key = str(spec["key"])
        default_entry = defaults_map[artifact_key]
        direct_entry = direct_entries.get(artifact_key, {})
        nested_entry = nested_entries.get(artifact_key, {})
        allowed_fields = {"enabled", "intensity", "amount"} | {
            str(param_spec["key"]) for param_spec in spec.get("params", ())
        }

        for field_name, entry in (
            ("artifact_type_settings", direct_entry),
            ("artifacts.types", nested_entry),
        ):
            unknown_fields = sorted(set(entry) - allowed_fields)
            if unknown_fields:
                raise ValueError(
                    f"`{field_name}[{artifact_key!r}]` contains unknown fields: {unknown_fields}."
                )

        entry: Dict[str, float | bool] = {
            "enabled": _coerce_bool(
                direct_entry.get(
                    "enabled",
                    nested_entry.get("enabled", default_entry["enabled"]),
                ),
                f"artifact_type_settings[{artifact_key!r}]['enabled']",
            ),
            "intensity": _coerce_float(
                direct_entry.get(
                    "intensity",
                    nested_entry.get("intensity", default_entry["intensity"]),
                ),
                f"artifact_type_settings[{artifact_key!r}]['intensity']",
                min_value=0.0,
                max_value=1.0,
            ),
            "amount": _coerce_float(
                direct_entry.get(
                    "amount",
                    nested_entry.get("amount", default_entry["amount"]),
                ),
                f"artifact_type_settings[{artifact_key!r}]['amount']",
                min_value=0.0,
                max_value=1.0,
            ),
        }
        for param_spec in spec.get("params", ()):
            param_key = str(param_spec["key"])
            entry[param_key] = _coerce_float(
                direct_entry.get(
                    param_key,
                    nested_entry.get(param_key, default_entry[param_key]),
                ),
                f"artifact_type_settings[{artifact_key!r}]['{param_key}']",
                min_value=float(param_spec["min"]),
                max_value=float(param_spec["max"]),
            )
        normalized[artifact_key] = entry

    return normalized


def validate_generation_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Validate YAML-loaded generation settings and return normalized values.

    Returned keys intentionally match the GUI `SyntheticGenerationParams` dataclass so the
    configuration payload can be reused by both the standalone generator and the main app.
    """
    if not isinstance(config, Mapping):
        raise ValueError("Generation configuration root must be a mapping.")

    schema = config.get("schema")
    if schema is not None and str(schema) != GENERATION_CONFIG_SCHEMA:
        raise ValueError(
            f"Unsupported config schema `{schema}`. Expected `{GENERATION_CONFIG_SCHEMA}`."
        )

    defaults = default_generation_config()
    allowed_keys = set(defaults) | {
        "schema",
        "artifacts",
        CLASS_LABELS_CONFIG_KEY,
        CLASS_GENERATION_ORDER_CONFIG_KEY,
    }
    unknown_keys = sorted(set(config) - allowed_keys)
    if unknown_keys:
        raise ValueError(f"Unknown generation config fields: {unknown_keys}.")

    artifacts_section = config.get("artifacts")
    if artifacts_section is not None and not isinstance(artifacts_section, Mapping):
        raise ValueError("`artifacts` must be a mapping.")
    artifacts_section = artifacts_section or {}
    allowed_artifact_keys = {"enabled", "global_intensity", "point_fraction", "types"}
    unknown_artifact_keys = sorted(set(artifacts_section) - allowed_artifact_keys)
    if unknown_artifact_keys:
        raise ValueError(f"`artifacts` contains unknown fields: {unknown_artifact_keys}.")
    artifact_types_section = artifacts_section.get("types")
    if artifact_types_section is not None and not isinstance(artifact_types_section, Mapping):
        raise ValueError("`artifacts.types` must be a mapping.")

    _validate_class_labels_metadata(config.get(CLASS_LABELS_CONFIG_KEY))
    _validate_class_generation_order_metadata(config.get(CLASS_GENERATION_ORDER_CONFIG_KEY))

    normalized: Dict[str, Any] = {}
    normalized["total_points"] = _coerce_int(
        config.get("total_points", defaults["total_points"]),
        "total_points",
        min_value=1,
    )
    normalized["area_width"] = _coerce_float(
        config.get("area_width", defaults["area_width"]),
        "area_width",
        min_value=1e-12,
    )
    normalized["area_length"] = _coerce_float(
        config.get("area_length", defaults["area_length"]),
        "area_length",
        min_value=1e-12,
    )
    normalized["terrain_relief"] = _coerce_float(
        config.get("terrain_relief", defaults["terrain_relief"]),
        "terrain_relief",
        min_value=0.0,
        max_value=1.0,
    )
    normalized["seed"] = _coerce_int(config.get("seed", defaults["seed"]), "seed", min_value=0)
    normalized["randomize_object_counts"] = _coerce_bool(
        config.get("randomize_object_counts", defaults["randomize_object_counts"]),
        "randomize_object_counts",
    )

    normalized["custom_class_distribution"] = _coerce_bool(
        config.get("custom_class_distribution", defaults["custom_class_distribution"]),
        "custom_class_distribution",
    )
    normalized["class_percentages"] = _coerce_percentage_values(
        config.get("class_percentages", defaults["class_percentages"]),
        "class_percentages",
        ordered_keys=CLASS_IDS,
        key_parser=_parse_class_id_key,
    )

    normalized["custom_artificial_surface_count"] = _coerce_bool(
        config.get(
            "custom_artificial_surface_count",
            defaults["custom_artificial_surface_count"],
        ),
        "custom_artificial_surface_count",
    )
    normalized["artificial_surface_count"] = _coerce_int(
        config.get("artificial_surface_count", defaults["artificial_surface_count"]),
        "artificial_surface_count",
        min_value=1,
    )
    normalized["custom_artificial_surface_type_distribution"] = _coerce_bool(
        config.get(
            "custom_artificial_surface_type_distribution",
            defaults["custom_artificial_surface_type_distribution"],
        ),
        "custom_artificial_surface_type_distribution",
    )
    normalized["artificial_surface_type_percentages"] = _coerce_percentage_values(
        config.get(
            "artificial_surface_type_percentages",
            defaults["artificial_surface_type_percentages"],
        ),
        "artificial_surface_type_percentages",
        ordered_keys=ARTIFICIAL_SURFACE_TYPES,
        key_parser=lambda raw_key, field_name: _parse_choice_key(
            raw_key,
            field_name,
            ARTIFICIAL_SURFACE_TYPES,
        ),
    )

    normalized["custom_building_count"] = _coerce_bool(
        config.get("custom_building_count", defaults["custom_building_count"]),
        "custom_building_count",
    )
    normalized["building_count"] = _coerce_int(
        config.get("building_count", defaults["building_count"]),
        "building_count",
        min_value=1,
    )
    normalized["custom_building_roof_type_distribution"] = _coerce_bool(
        config.get(
            "custom_building_roof_type_distribution",
            defaults["custom_building_roof_type_distribution"],
        ),
        "custom_building_roof_type_distribution",
    )
    normalized["building_roof_type_percentages"] = _coerce_percentage_values(
        config.get(
            "building_roof_type_percentages",
            defaults["building_roof_type_percentages"],
        ),
        "building_roof_type_percentages",
        ordered_keys=BUILDING_ROOF_TYPES,
        key_parser=lambda raw_key, field_name: _parse_choice_key(
            raw_key,
            field_name,
            BUILDING_ROOF_TYPES,
        ),
    )
    normalized["building_floor_min"] = _coerce_int(
        config.get("building_floor_min", defaults["building_floor_min"]),
        "building_floor_min",
        min_value=1,
    )
    normalized["building_floor_max"] = _coerce_int(
        config.get("building_floor_max", defaults["building_floor_max"]),
        "building_floor_max",
        min_value=1,
    )
    normalized["building_random_yaw"] = _coerce_bool(
        config.get("building_random_yaw", defaults["building_random_yaw"]),
        "building_random_yaw",
    )

    normalized["custom_structure_count"] = _coerce_bool(
        config.get("custom_structure_count", defaults["custom_structure_count"]),
        "custom_structure_count",
    )
    normalized["structure_count"] = _coerce_int(
        config.get("structure_count", defaults["structure_count"]),
        "structure_count",
        min_value=1,
    )
    normalized["custom_structure_type_distribution"] = _coerce_bool(
        config.get(
            "custom_structure_type_distribution",
            defaults["custom_structure_type_distribution"],
        ),
        "custom_structure_type_distribution",
    )
    normalized["structure_type_percentages"] = _coerce_percentage_values(
        config.get("structure_type_percentages", defaults["structure_type_percentages"]),
        "structure_type_percentages",
        ordered_keys=STRUCTURE_TYPES,
        key_parser=lambda raw_key, field_name: _parse_choice_key(
            raw_key,
            field_name,
            STRUCTURE_TYPES,
        ),
    )

    normalized["custom_tree_count"] = _coerce_bool(
        config.get("custom_tree_count", defaults["custom_tree_count"]),
        "custom_tree_count",
    )
    normalized["tree_count"] = _coerce_int(
        config.get("tree_count", defaults["tree_count"]),
        "tree_count",
        min_value=1,
    )
    normalized["custom_tree_crown_type_distribution"] = _coerce_bool(
        config.get(
            "custom_tree_crown_type_distribution",
            defaults["custom_tree_crown_type_distribution"],
        ),
        "custom_tree_crown_type_distribution",
    )
    normalized["tree_crown_type_percentages"] = _coerce_percentage_values(
        config.get("tree_crown_type_percentages", defaults["tree_crown_type_percentages"]),
        "tree_crown_type_percentages",
        ordered_keys=TREE_CROWN_TYPES,
        key_parser=lambda raw_key, field_name: _parse_choice_key(
            raw_key,
            field_name,
            TREE_CROWN_TYPES,
        ),
    )
    normalized["random_tree_crown_size"] = _coerce_bool(
        config.get("random_tree_crown_size", defaults["random_tree_crown_size"]),
        "random_tree_crown_size",
    )
    normalized["tree_max_crown_diameter"] = _coerce_float(
        config.get("tree_max_crown_diameter", defaults["tree_max_crown_diameter"]),
        "tree_max_crown_diameter",
        min_value=1e-12,
    )
    normalized["tree_max_crown_top_height"] = _coerce_float(
        config.get("tree_max_crown_top_height", defaults["tree_max_crown_top_height"]),
        "tree_max_crown_top_height",
        min_value=1e-12,
    )
    normalized["tree_min_crown_bottom_height"] = _coerce_float(
        config.get("tree_min_crown_bottom_height", defaults["tree_min_crown_bottom_height"]),
        "tree_min_crown_bottom_height",
        min_value=0.0,
    )

    normalized["custom_vehicle_count"] = _coerce_bool(
        config.get("custom_vehicle_count", defaults["custom_vehicle_count"]),
        "custom_vehicle_count",
    )
    normalized["vehicle_count"] = _coerce_int(
        config.get("vehicle_count", defaults["vehicle_count"]),
        "vehicle_count",
        min_value=1,
    )
    normalized["custom_vehicle_type_distribution"] = _coerce_bool(
        config.get(
            "custom_vehicle_type_distribution",
            defaults["custom_vehicle_type_distribution"],
        ),
        "custom_vehicle_type_distribution",
    )
    normalized["vehicle_type_percentages"] = _coerce_percentage_values(
        config.get("vehicle_type_percentages", defaults["vehicle_type_percentages"]),
        "vehicle_type_percentages",
        ordered_keys=VEHICLE_TYPES,
        key_parser=lambda raw_key, field_name: _parse_choice_key(
            raw_key,
            field_name,
            VEHICLE_TYPES,
        ),
    )

    normalized["artifacts_enabled"] = _coerce_bool(
        config.get(
            "artifacts_enabled",
            artifacts_section.get("enabled", defaults["artifacts_enabled"]),
        ),
        "artifacts_enabled",
    )
    normalized["artifact_global_intensity"] = _coerce_float(
        config.get(
            "artifact_global_intensity",
            artifacts_section.get(
                "global_intensity",
                defaults["artifact_global_intensity"],
            ),
        ),
        "artifact_global_intensity",
        min_value=0.0,
        max_value=1.0,
    )
    normalized["artifact_point_fraction"] = _coerce_float(
        config.get(
            "artifact_point_fraction",
            artifacts_section.get("point_fraction", defaults["artifact_point_fraction"]),
        ),
        "artifact_point_fraction",
        min_value=0.0,
        max_value=1.0,
    )
    normalized["artifact_type_settings"] = _normalize_artifact_type_settings_payload(
        config.get("artifact_type_settings"),
        artifact_types_section,
        defaults=defaults["artifact_type_settings"],
    )

    normalized["custom_shrub_count"] = _coerce_bool(
        config.get("custom_shrub_count", defaults["custom_shrub_count"]),
        "custom_shrub_count",
    )
    normalized["shrub_count"] = _coerce_int(
        config.get("shrub_count", defaults["shrub_count"]),
        "shrub_count",
        min_value=1,
    )
    normalized["random_shrub_size"] = _coerce_bool(
        config.get("random_shrub_size", defaults["random_shrub_size"]),
        "random_shrub_size",
    )
    normalized["shrub_max_diameter"] = _coerce_float(
        config.get("shrub_max_diameter", defaults["shrub_max_diameter"]),
        "shrub_max_diameter",
        min_value=1e-12,
    )
    normalized["shrub_max_top_height"] = _coerce_float(
        config.get("shrub_max_top_height", defaults["shrub_max_top_height"]),
        "shrub_max_top_height",
        min_value=1e-12,
    )
    normalized["shrub_min_bottom_height"] = _coerce_float(
        config.get("shrub_min_bottom_height", defaults["shrub_min_bottom_height"]),
        "shrub_min_bottom_height",
        min_value=0.0,
    )

    normalized["custom_grass_patch_count"] = _coerce_bool(
        config.get("custom_grass_patch_count", defaults["custom_grass_patch_count"]),
        "custom_grass_patch_count",
    )
    normalized["grass_patch_count"] = _coerce_int(
        config.get("grass_patch_count", defaults["grass_patch_count"]),
        "grass_patch_count",
        min_value=1,
    )
    normalized["random_grass_patch_size"] = _coerce_bool(
        config.get("random_grass_patch_size", defaults["random_grass_patch_size"]),
        "random_grass_patch_size",
    )
    normalized["grass_patch_max_size_x"] = _coerce_float(
        config.get("grass_patch_max_size_x", defaults["grass_patch_max_size_x"]),
        "grass_patch_max_size_x",
        min_value=1e-12,
    )
    normalized["grass_patch_max_size_y"] = _coerce_float(
        config.get("grass_patch_max_size_y", defaults["grass_patch_max_size_y"]),
        "grass_patch_max_size_y",
        min_value=1e-12,
    )
    normalized["grass_max_height"] = _coerce_float(
        config.get("grass_max_height", defaults["grass_max_height"]),
        "grass_max_height",
        min_value=1e-12,
    )

    if normalized["custom_class_distribution"]:
        _validate_enabled_percentage_distribution(
            normalized["class_percentages"],
            "class_percentages",
        )
    if normalized["custom_artificial_surface_type_distribution"]:
        _validate_enabled_percentage_distribution(
            normalized["artificial_surface_type_percentages"],
            "artificial_surface_type_percentages",
        )
    if normalized["custom_building_roof_type_distribution"]:
        _validate_enabled_percentage_distribution(
            normalized["building_roof_type_percentages"],
            "building_roof_type_percentages",
        )
    if normalized["custom_structure_type_distribution"]:
        _validate_enabled_percentage_distribution(
            normalized["structure_type_percentages"],
            "structure_type_percentages",
        )
    if normalized["custom_tree_crown_type_distribution"]:
        _validate_enabled_percentage_distribution(
            normalized["tree_crown_type_percentages"],
            "tree_crown_type_percentages",
        )
    if normalized["custom_vehicle_type_distribution"]:
        _validate_enabled_percentage_distribution(
            normalized["vehicle_type_percentages"],
            "vehicle_type_percentages",
        )
    if normalized["building_floor_max"] < normalized["building_floor_min"]:
        raise ValueError("`building_floor_max` must be >= `building_floor_min`.")
    if normalized["tree_min_crown_bottom_height"] >= normalized["tree_max_crown_top_height"]:
        raise ValueError(
            "`tree_min_crown_bottom_height` must be strictly less than "
            "`tree_max_crown_top_height`."
        )
    if normalized["shrub_min_bottom_height"] >= normalized["shrub_max_top_height"]:
        raise ValueError(
            "`shrub_min_bottom_height` must be strictly less than `shrub_max_top_height`."
        )
    if normalized["artifacts_enabled"] and normalized["artifact_point_fraction"] > 0.0:
        enabled_artifact_types = [
            settings
            for settings in normalized["artifact_type_settings"].values()
            if bool(settings["enabled"])
        ]
        if not enabled_artifact_types:
            raise ValueError(
                "At least one artifact type must be enabled when artifacts are enabled and "
                "`artifact_point_fraction` is > 0."
            )
        if not any(float(settings["amount"]) > 0.0 for settings in enabled_artifact_types):
            raise ValueError(
                "At least one enabled artifact type must have `amount` > 0 when artifacts "
                "are enabled and `artifact_point_fraction` is > 0."
            )

    return normalized


def generation_config_to_yaml_data(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert normalized generation settings into a readable YAML payload."""
    validated = validate_generation_config(config)
    yaml_data: Dict[str, Any] = {"schema": GENERATION_CONFIG_SCHEMA}
    yaml_data[CLASS_LABELS_CONFIG_KEY] = {
        str(class_id): CLASS_NAMES[class_id] for class_id in CLASS_IDS
    }
    yaml_data[CLASS_GENERATION_ORDER_CONFIG_KEY] = [
        int(class_id) for class_id in CLASS_GENERATION_ORDER
    ]
    artifact_settings = _copy_artifact_type_settings(validated["artifact_type_settings"])
    for key, value in validated.items():
        if key == "class_percentages":
            yaml_data[key] = {
                str(class_id): float(value[index])
                for index, class_id in enumerate(CLASS_IDS)
            }
        elif key == "artificial_surface_type_percentages":
            yaml_data[key] = {
                surface_type: float(value[index])
                for index, surface_type in enumerate(ARTIFICIAL_SURFACE_TYPES)
            }
        elif key == "building_roof_type_percentages":
            yaml_data[key] = {
                roof_type: float(value[index])
                for index, roof_type in enumerate(BUILDING_ROOF_TYPES)
            }
        elif key == "structure_type_percentages":
            yaml_data[key] = {
                structure_type: float(value[index])
                for index, structure_type in enumerate(STRUCTURE_TYPES)
            }
        elif key == "tree_crown_type_percentages":
            yaml_data[key] = {
                crown_type: float(value[index])
                for index, crown_type in enumerate(TREE_CROWN_TYPES)
            }
        elif key == "vehicle_type_percentages":
            yaml_data[key] = {
                vehicle_type: float(value[index])
                for index, vehicle_type in enumerate(VEHICLE_TYPES)
            }
        elif key in {
            "artifacts_enabled",
            "artifact_global_intensity",
            "artifact_point_fraction",
            "artifact_type_settings",
        }:
            continue
        else:
            yaml_data[key] = value

    yaml_data["artifacts"] = {
        "enabled": bool(validated["artifacts_enabled"]),
        "global_intensity": float(validated["artifact_global_intensity"]),
        "point_fraction": float(validated["artifact_point_fraction"]),
        "types": {
            artifact_key: {
                field_name: (
                    bool(field_value) if field_name == "enabled" else float(field_value)
                )
                for field_name, field_value in artifact_settings[artifact_key].items()
            }
            for artifact_key in ARTIFACT_TYPES
        },
    }
    return yaml_data


def save_generation_config(config: Mapping[str, Any], path: str | Path) -> Path:
    """Validate and save generation settings to a YAML configuration file."""
    output_path = Path(path)
    yaml_data = generation_config_to_yaml_data(config)
    with output_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(
            yaml_data,
            stream,
            allow_unicode=True,
            sort_keys=False,
        )
    return output_path


def load_generation_config(path: str | Path) -> Dict[str, Any]:
    """Load and validate generation settings from a YAML configuration file."""
    input_path = Path(path)
    try:
        with input_path.open("r", encoding="utf-8") as stream:
            raw_config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML configuration: {exc}") from exc

    if raw_config is None:
        raise ValueError("Configuration file is empty.")
    if not isinstance(raw_config, Mapping):
        raise ValueError("Configuration file root must be a mapping.")
    return validate_generation_config(raw_config)


def generation_config_to_pipeline_kwargs(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Translate validated config data into kwargs accepted by the generator pipeline."""
    validated = validate_generation_config(config)
    return {
        "total_points": int(validated["total_points"]),
        "area_width": float(validated["area_width"]),
        "area_length": float(validated["area_length"]),
        "terrain_relief": float(validated["terrain_relief"]),
        "randomize_object_counts": bool(validated["randomize_object_counts"]),
        "seed": int(validated["seed"]),
        "class_percentages": (
            validated["class_percentages"]
            if validated["custom_class_distribution"]
            else None
        ),
        "artificial_surface_count": (
            int(validated["artificial_surface_count"])
            if validated["custom_artificial_surface_count"]
            else None
        ),
        "artificial_surface_type_percentages": (
            validated["artificial_surface_type_percentages"]
            if validated["custom_artificial_surface_type_distribution"]
            else None
        ),
        "tree_count": int(validated["tree_count"]) if validated["custom_tree_count"] else None,
        "tree_crown_type_percentages": (
            validated["tree_crown_type_percentages"]
            if validated["custom_tree_crown_type_distribution"]
            else None
        ),
        "random_tree_crown_size": bool(validated["random_tree_crown_size"]),
        "tree_max_crown_diameter": float(validated["tree_max_crown_diameter"]),
        "tree_max_crown_top_height": float(validated["tree_max_crown_top_height"]),
        "tree_min_crown_bottom_height": float(validated["tree_min_crown_bottom_height"]),
        "building_count": (
            int(validated["building_count"]) if validated["custom_building_count"] else None
        ),
        "building_roof_type_percentages": (
            validated["building_roof_type_percentages"]
            if validated["custom_building_roof_type_distribution"]
            else None
        ),
        "building_floor_min": int(validated["building_floor_min"]),
        "building_floor_max": int(validated["building_floor_max"]),
        "building_random_yaw": bool(validated["building_random_yaw"]),
        "structure_count": (
            int(validated["structure_count"]) if validated["custom_structure_count"] else None
        ),
        "structure_type_percentages": (
            validated["structure_type_percentages"]
            if validated["custom_structure_type_distribution"]
            else None
        ),
        "vehicle_count": (
            int(validated["vehicle_count"]) if validated["custom_vehicle_count"] else None
        ),
        "vehicle_type_percentages": (
            validated["vehicle_type_percentages"]
            if validated["custom_vehicle_type_distribution"]
            else None
        ),
        "artifacts_enabled": bool(validated["artifacts_enabled"]),
        "artifact_global_intensity": float(validated["artifact_global_intensity"]),
        "artifact_point_fraction": float(validated["artifact_point_fraction"]),
        "artifact_type_settings": _copy_artifact_type_settings(
            validated["artifact_type_settings"]
        ),
        "shrub_count": int(validated["shrub_count"]) if validated["custom_shrub_count"] else None,
        "random_shrub_size": bool(validated["random_shrub_size"]),
        "shrub_max_diameter": float(validated["shrub_max_diameter"]),
        "shrub_max_top_height": float(validated["shrub_max_top_height"]),
        "shrub_min_bottom_height": float(validated["shrub_min_bottom_height"]),
        "grass_patch_count": (
            int(validated["grass_patch_count"])
            if validated["custom_grass_patch_count"]
            else None
        ),
        "random_grass_patch_size": bool(validated["random_grass_patch_size"]),
        "grass_patch_max_size_x": float(validated["grass_patch_max_size_x"]),
        "grass_patch_max_size_y": float(validated["grass_patch_max_size_y"]),
        "grass_max_height": float(validated["grass_max_height"]),
    }

def _validate_positive(value: float, name: str) -> None:
    """Validate positive numeric inputs."""
    if value <= 0.0:
        raise ValueError(f"`{name}` must be > 0, got {value}.")


def _validate_unit_interval(value: float, name: str) -> None:
    """Validate that a value is inside [0, 1]."""
    if value < 0.0 or value > 1.0:
        raise ValueError(f"`{name}` must be in [0, 1], got {value}.")


def _class_ratios_from_percentages(
    class_percentages: Sequence[float] | Dict[int, float] | None,
) -> Dict[int, float]:
    """
    Build normalized per-class ratios from optional percentages.
    - None -> default distribution.
    - Sequence -> 8 percentages for classes 0..7.
    - Dict -> class_id -> percentage; missing classes are treated as 0.
    """
    if class_percentages is None:
        return dict(DEFAULT_CLASS_RATIOS)

    if isinstance(class_percentages, dict):
        unknown = sorted(set(class_percentages) - set(CLASS_IDS))
        if unknown:
            raise ValueError(
                f"Unknown class ids in `class_percentages`: {unknown}. "
                f"Supported ids are {list(CLASS_IDS)}."
            )
        values = [float(class_percentages.get(class_id, 0.0)) for class_id in CLASS_IDS]
    else:
        values = [float(value) for value in class_percentages]
        if len(values) != len(CLASS_IDS):
            raise ValueError(
                f"`class_percentages` must contain exactly {len(CLASS_IDS)} values "
                f"(for classes {list(CLASS_IDS)}), got {len(values)}."
            )

    weights = np.array(values, dtype=np.float64)
    if not np.all(np.isfinite(weights)):
        raise ValueError("`class_percentages` must contain only finite numbers.")
    if np.any(weights < 0.0):
        raise ValueError("`class_percentages` must be non-negative.")
    if np.isclose(weights.sum(), 0.0):
        raise ValueError("At least one class percentage must be > 0.")

    ratios = weights / weights.sum()
    return {class_id: float(ratios[idx]) for idx, class_id in enumerate(CLASS_IDS)}


def _override_artifact_class_ratio(
    class_ratios: Mapping[int, float],
    artifact_point_fraction: float,
) -> Dict[int, float]:
    adjusted = {
        class_id: max(0.0, float(class_ratios.get(class_id, 0.0)))
        for class_id in CLASS_IDS
    }
    target_ratio = float(np.clip(float(artifact_point_fraction), 0.0, 1.0))
    if np.isclose(target_ratio, 1.0):
        return {
            class_id: (1.0 if class_id == ARTIFACTS_CLASS_ID else 0.0)
            for class_id in CLASS_IDS
        }

    other_ids = [class_id for class_id in CLASS_IDS if class_id != ARTIFACTS_CLASS_ID]
    other_total = float(sum(adjusted[class_id] for class_id in other_ids))
    if other_total <= 0.0:
        raise ValueError(
            "At least one non-artifact class must keep a positive share when "
            "`artifact_point_fraction` is below 1.0."
        )

    remaining_ratio = max(0.0, 1.0 - target_ratio)
    for class_id in other_ids:
        adjusted[class_id] = adjusted[class_id] / other_total * remaining_ratio
    adjusted[ARTIFACTS_CLASS_ID] = target_ratio
    return adjusted


def _artificial_surface_type_ratios_from_percentages(
    artificial_surface_type_percentages: Sequence[float] | Dict[str, float] | None,
) -> Dict[str, float]:
    """
    Build normalized artificial surface type ratios from optional percentages.
    Supports sequence in ARTIFICIAL_SURFACE_TYPES order or mapping by type keys.
    """
    if artificial_surface_type_percentages is None:
        return dict(DEFAULT_ARTIFICIAL_SURFACE_TYPE_RATIOS)

    if isinstance(artificial_surface_type_percentages, dict):
        normalized: Dict[str, float] = {}
        for key, value in artificial_surface_type_percentages.items():
            surface_type = str(key).strip().lower()
            if surface_type not in ARTIFICIAL_SURFACE_TYPES:
                raise ValueError(
                    "Unknown artificial surface type key "
                    f"`{key}` in `artificial_surface_type_percentages`. "
                    f"Supported keys are {list(ARTIFICIAL_SURFACE_TYPES)}."
                )
            normalized[surface_type] = float(value)
        values = [
            float(normalized.get(surface_type, 0.0))
            for surface_type in ARTIFICIAL_SURFACE_TYPES
        ]
    else:
        values = [float(value) for value in artificial_surface_type_percentages]
        if len(values) != len(ARTIFICIAL_SURFACE_TYPES):
            raise ValueError(
                "`artificial_surface_type_percentages` must contain exactly "
                f"{len(ARTIFICIAL_SURFACE_TYPES)} values "
                f"(for artificial surface types {list(ARTIFICIAL_SURFACE_TYPES)}), "
                f"got {len(values)}."
            )

    weights = np.array(values, dtype=np.float64)
    if not np.all(np.isfinite(weights)):
        raise ValueError(
            "`artificial_surface_type_percentages` must contain only finite numbers."
        )
    if np.any(weights < 0.0):
        raise ValueError("`artificial_surface_type_percentages` must be non-negative.")
    if np.isclose(weights.sum(), 0.0):
        raise ValueError("At least one artificial surface type percentage must be > 0.")

    weights = weights / weights.sum()
    return {
        surface_type: float(weights[idx])
        for idx, surface_type in enumerate(ARTIFICIAL_SURFACE_TYPES)
    }


def _vehicle_type_ratios_from_percentages(
    vehicle_type_percentages: Sequence[float] | Dict[str, float] | None,
) -> Dict[str, float]:
    """
    Build normalized vehicle type ratios from optional percentages.
    Supports sequence [car, truck, bus] or mapping by keys: car, truck, bus.
    """
    if vehicle_type_percentages is None:
        return dict(DEFAULT_VEHICLE_TYPE_RATIOS)

    if isinstance(vehicle_type_percentages, dict):
        normalized: Dict[str, float] = {}
        for key, value in vehicle_type_percentages.items():
            type_id = str(key).strip().lower()
            if type_id not in VEHICLE_TYPES:
                raise ValueError(
                    f"Unknown vehicle type key `{key}` in `vehicle_type_percentages`. "
                    f"Supported keys are {list(VEHICLE_TYPES)}."
                )
            normalized[type_id] = float(value)
        values = [float(normalized.get(vehicle_type, 0.0)) for vehicle_type in VEHICLE_TYPES]
    else:
        values = [float(value) for value in vehicle_type_percentages]
        if len(values) != len(VEHICLE_TYPES):
            raise ValueError(
                f"`vehicle_type_percentages` must contain exactly {len(VEHICLE_TYPES)} values "
                f"(for types {list(VEHICLE_TYPES)}), got {len(values)}."
            )

    weights = np.array(values, dtype=np.float64)
    if not np.all(np.isfinite(weights)):
        raise ValueError("`vehicle_type_percentages` must contain only finite numbers.")
    if np.any(weights < 0.0):
        raise ValueError("`vehicle_type_percentages` must be non-negative.")
    if np.isclose(weights.sum(), 0.0):
        raise ValueError("At least one vehicle type percentage must be > 0.")

    weights = weights / weights.sum()
    return {
        vehicle_type: float(weights[idx])
        for idx, vehicle_type in enumerate(VEHICLE_TYPES)
    }


def _structure_type_ratios_from_percentages(
    structure_type_percentages: Sequence[float] | Dict[str, float] | None,
) -> Dict[str, float]:
    """
    Build normalized structure type ratios from optional percentages.
    Supports sequence in STRUCTURE_TYPES order or mapping by structure type keys.
    """
    if structure_type_percentages is None:
        return dict(DEFAULT_STRUCTURE_TYPE_RATIOS)

    if isinstance(structure_type_percentages, dict):
        normalized: Dict[str, float] = {}
        for key, value in structure_type_percentages.items():
            structure_type = str(key).strip().lower()
            if structure_type not in STRUCTURE_TYPES:
                raise ValueError(
                    f"Unknown structure type key `{key}` in `structure_type_percentages`. "
                    f"Supported keys are {list(STRUCTURE_TYPES)}."
                )
            normalized[structure_type] = float(value)
        values = [
            float(normalized.get(structure_type, 0.0))
            for structure_type in STRUCTURE_TYPES
        ]
    else:
        values = [float(value) for value in structure_type_percentages]
        if len(values) != len(STRUCTURE_TYPES):
            raise ValueError(
                "`structure_type_percentages` must contain exactly "
                f"{len(STRUCTURE_TYPES)} values "
                f"(for structure types {list(STRUCTURE_TYPES)}), got {len(values)}."
            )

    weights = np.array(values, dtype=np.float64)
    if not np.all(np.isfinite(weights)):
        raise ValueError("`structure_type_percentages` must contain only finite numbers.")
    if np.any(weights < 0.0):
        raise ValueError("`structure_type_percentages` must be non-negative.")
    if np.isclose(weights.sum(), 0.0):
        raise ValueError("At least one structure type percentage must be > 0.")

    weights = weights / weights.sum()
    return {
        structure_type: float(weights[idx])
        for idx, structure_type in enumerate(STRUCTURE_TYPES)
    }


def _tree_crown_type_ratios_from_percentages(
    tree_crown_type_percentages: Sequence[float] | Dict[str, float] | None,
) -> Dict[str, float]:
    """
    Build normalized tree crown type ratios from optional percentages.
    Supports sequence [spherical, pyramidal, spreading, weeping, columnar, umbrella]
    or mapping by crown type keys.
    """
    if tree_crown_type_percentages is None:
        return dict(DEFAULT_TREE_CROWN_TYPE_RATIOS)

    if isinstance(tree_crown_type_percentages, dict):
        normalized: Dict[str, float] = {}
        for key, value in tree_crown_type_percentages.items():
            crown_type = str(key).strip().lower()
            if crown_type not in TREE_CROWN_TYPES:
                raise ValueError(
                    f"Unknown crown type key `{key}` in `tree_crown_type_percentages`. "
                    f"Supported keys are {list(TREE_CROWN_TYPES)}."
                )
            normalized[crown_type] = float(value)
        values = [float(normalized.get(crown_type, 0.0)) for crown_type in TREE_CROWN_TYPES]
    else:
        values = [float(value) for value in tree_crown_type_percentages]
        if len(values) != len(TREE_CROWN_TYPES):
            raise ValueError(
                f"`tree_crown_type_percentages` must contain exactly {len(TREE_CROWN_TYPES)} "
                f"values (for crown types {list(TREE_CROWN_TYPES)}), got {len(values)}."
            )

    weights = np.array(values, dtype=np.float64)
    if not np.all(np.isfinite(weights)):
        raise ValueError("`tree_crown_type_percentages` must contain only finite numbers.")
    if np.any(weights < 0.0):
        raise ValueError("`tree_crown_type_percentages` must be non-negative.")
    if np.isclose(weights.sum(), 0.0):
        raise ValueError("At least one tree crown type percentage must be > 0.")

    weights = weights / weights.sum()
    return {
        crown_type: float(weights[idx])
        for idx, crown_type in enumerate(TREE_CROWN_TYPES)
    }


def _building_roof_type_ratios_from_percentages(
    building_roof_type_percentages: Sequence[float] | Dict[str, float] | None,
) -> Dict[str, float]:
    """
    Build normalized building roof type ratios from optional percentages.
    Supports sequence in BUILDING_ROOF_TYPES order or mapping by roof type keys.
    """
    if building_roof_type_percentages is None:
        return dict(DEFAULT_BUILDING_ROOF_TYPE_RATIOS)

    if isinstance(building_roof_type_percentages, dict):
        normalized: Dict[str, float] = {}
        for key, value in building_roof_type_percentages.items():
            roof_type = str(key).strip().lower()
            if roof_type not in BUILDING_ROOF_TYPES:
                raise ValueError(
                    f"Unknown roof type key `{key}` in `building_roof_type_percentages`. "
                    f"Supported keys are {list(BUILDING_ROOF_TYPES)}."
                )
            normalized[roof_type] = float(value)
        values = [float(normalized.get(roof_type, 0.0)) for roof_type in BUILDING_ROOF_TYPES]
    else:
        values = [float(value) for value in building_roof_type_percentages]
        if len(values) != len(BUILDING_ROOF_TYPES):
            raise ValueError(
                "`building_roof_type_percentages` must contain exactly "
                f"{len(BUILDING_ROOF_TYPES)} values "
                f"(for roof types {list(BUILDING_ROOF_TYPES)}), got {len(values)}."
            )

    weights = np.array(values, dtype=np.float64)
    if not np.all(np.isfinite(weights)):
        raise ValueError("`building_roof_type_percentages` must contain only finite numbers.")
    if np.any(weights < 0.0):
        raise ValueError("`building_roof_type_percentages` must be non-negative.")
    if np.isclose(weights.sum(), 0.0):
        raise ValueError("At least one building roof type percentage must be > 0.")

    weights = weights / weights.sum()
    return {
        roof_type: float(weights[idx])
        for idx, roof_type in enumerate(BUILDING_ROOF_TYPES)
    }


def allocate_points(total_points: int, class_ratios: Dict[int, float]) -> Dict[int, int]:
    """
    Split total points across classes according to ratios.
    The output always sums to total_points.
    """
    labels = sorted(class_ratios)
    weights = np.array([class_ratios[k] for k in labels], dtype=np.float64)
    if np.any(weights < 0):
        raise ValueError("Class ratios must be non-negative.")
    if np.isclose(weights.sum(), 0.0):
        raise ValueError("At least one class ratio must be positive.")

    weights = weights / weights.sum()
    raw = weights * int(total_points)
    counts = np.floor(raw).astype(int)
    remainder = int(total_points) - int(counts.sum())
    if remainder > 0:
        frac = raw - counts
        for idx in np.argsort(frac)[::-1][:remainder]:
            counts[idx] += 1

    return {label: int(count) for label, count in zip(labels, counts)}


def _split_count_by_weights(total: int, weights: Sequence[float]) -> np.ndarray:
    """Split integer `total` by continuous `weights` while preserving sum."""
    weights_arr = np.array(weights, dtype=np.float64)
    if np.isclose(weights_arr.sum(), 0.0):
        out = np.zeros(len(weights_arr), dtype=int)
        if len(out) > 0:
            out[0] = total
        return out

    weights_arr = weights_arr / weights_arr.sum()
    raw = weights_arr * total
    base = np.floor(raw).astype(int)
    remainder = total - int(base.sum())
    if remainder > 0:
        frac = raw - base
        for idx in np.argsort(frac)[::-1][:remainder]:
            base[idx] += 1
    return base


def _split_count_evenly(total: int, n_chunks: int) -> np.ndarray:
    """Even integer split where first chunks get remainder."""
    if n_chunks <= 0:
        return np.array([], dtype=int)
    base = np.full(n_chunks, total // n_chunks, dtype=int)
    base[: (total % n_chunks)] += 1
    return base


def _split_count_random(
    total: int, n_chunks: int, rng: np.random.Generator, min_per_chunk: int = 1
) -> np.ndarray:
    """
    Random integer split that preserves the exact total.
    Falls back to even split when `total` is too small.
    """
    if n_chunks <= 0:
        return np.array([], dtype=int)
    if total <= 0:
        return np.zeros(n_chunks, dtype=int)

    min_per_chunk = max(0, int(min_per_chunk))
    if min_per_chunk * n_chunks > total:
        return _split_count_evenly(total, n_chunks)

    counts = np.full(n_chunks, min_per_chunk, dtype=int)
    remaining = total - int(counts.sum())
    if remaining > 0:
        weights = rng.dirichlet(np.ones(n_chunks, dtype=np.float64))
        raw = weights * remaining
        extra = np.floor(raw).astype(int)
        remainder = remaining - int(extra.sum())
        if remainder > 0:
            frac = raw - extra
            for idx in np.argsort(frac)[::-1][:remainder]:
                extra[idx] += 1
        counts += extra

    rng.shuffle(counts)
    return counts


def _inside_rectangles(
    x: np.ndarray, y: np.ndarray, rectangles: Sequence[Rect], margin: float = 0.0
) -> np.ndarray:
    """Mask of points that are inside at least one axis-aligned rectangle."""
    if len(rectangles) == 0:
        return np.zeros_like(x, dtype=bool)

    mask = np.zeros_like(x, dtype=bool)
    for cx, cy, sx, sy in rectangles:
        hx = max(0.0, sx * 0.5 - margin)
        hy = max(0.0, sy * 0.5 - margin)
        mask |= (np.abs(x - cx) <= hx) & (np.abs(y - cy) <= hy)
    return mask


def _sample_xy(
    rng: np.random.Generator,
    n: int,
    area_size: Tuple[float, float],
    forbidden_rects: Sequence[Rect] | None = None,
    allowed_rects: Sequence[Rect] | None = None,
    margin: float = 0.0,
    max_tries: int = 80,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample XY points in area with optional constraints.
    Falls back to unconstrained sampling if constraints are too strict.
    """
    if n <= 0:
        return np.empty(0), np.empty(0)

    forbidden_rects = list(forbidden_rects or [])
    allowed_rects = list(allowed_rects or [])

    w, h = area_size
    x_min, x_max = -w * 0.5, w * 0.5
    y_min, y_max = -h * 0.5, h * 0.5

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    remaining = n
    tries = 0

    while remaining > 0 and tries < max_tries:
        batch = max(512, remaining * 4)
        x_try = rng.uniform(x_min, x_max, size=batch)
        y_try = rng.uniform(y_min, y_max, size=batch)

        mask = np.ones(batch, dtype=bool)
        if allowed_rects:
            mask &= _inside_rectangles(x_try, y_try, allowed_rects, margin=margin)
        if forbidden_rects:
            mask &= ~_inside_rectangles(x_try, y_try, forbidden_rects, margin=margin)

        if np.any(mask):
            accepted = np.flatnonzero(mask)
            take = min(remaining, accepted.size)
            idx = accepted[:take]
            xs.append(x_try[idx])
            ys.append(y_try[idx])
            remaining -= take

        tries += 1

    # Fallback: ensure exact count even if strict constraints cannot be fully satisfied.
    if remaining > 0:
        xs.append(rng.uniform(x_min, x_max, size=remaining))
        ys.append(rng.uniform(y_min, y_max, size=remaining))

    return np.concatenate(xs), np.concatenate(ys)


def _sample_xy_outside_rects(
    rng: np.random.Generator,
    n: int,
    area_size: Tuple[float, float],
    forbidden_rects: Sequence[Rect],
    margin: float = 0.0,
    max_tries: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample XY points while strictly avoiding forbidden rectangles."""
    if n <= 0:
        return np.empty(0), np.empty(0)
    if not forbidden_rects:
        return _sample_xy(rng, n=n, area_size=area_size, margin=margin)

    w, h = area_size
    x_min, x_max = -w * 0.5, w * 0.5
    y_min, y_max = -h * 0.5, h * 0.5

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    remaining = int(n)
    tries = 0

    while remaining > 0 and tries < max_tries:
        batch = max(1024, remaining * 4)
        x_try = rng.uniform(x_min, x_max, size=batch)
        y_try = rng.uniform(y_min, y_max, size=batch)
        mask = ~_inside_rectangles(x_try, y_try, forbidden_rects, margin=margin)
        if np.any(mask):
            accepted = np.flatnonzero(mask)
            take = min(remaining, int(accepted.size))
            idx = accepted[:take]
            xs.append(x_try[idx])
            ys.append(y_try[idx])
            remaining -= take
        tries += 1

    if remaining > 0:
        raise ValueError(
            "Failed to sample natural-surface points outside artificial zones. "
            "Artificial coverage is likely too high for the requested scene settings."
        )

    return np.concatenate(xs), np.concatenate(ys)


def _sample_inside_rect(
    rng: np.random.Generator, n: int, rect: Rect, margin: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample points uniformly inside one rectangle."""
    if n <= 0:
        return np.empty(0), np.empty(0)

    cx, cy, sx, sy = rect
    hx = max(0.2, sx * 0.5 - margin)
    hy = max(0.2, sy * 0.5 - margin)
    x = rng.uniform(cx - hx, cx + hx, size=n)
    y = rng.uniform(cy - hy, cy + hy, size=n)
    return x, y


def _sample_single_xy(
    rng: np.random.Generator,
    area_size: Tuple[float, float],
    forbidden_rects: Sequence[Rect] | None = None,
    allowed_rects: Sequence[Rect] | None = None,
    margin: float = 0.0,
) -> Tuple[float, float]:
    """Sample one XY point."""
    x, y = _sample_xy(
        rng,
        n=1,
        area_size=area_size,
        forbidden_rects=forbidden_rects,
        allowed_rects=allowed_rects,
        margin=margin,
    )
    return float(x[0]), float(y[0])


def _rect_overlap_ratio(rect_a: Rect, rect_b: Rect) -> float:
    """Overlap ratio normalized by the smaller rectangle area."""
    ax, ay, aw, ah = rect_a
    bx, by, bw, bh = rect_b

    a_left, a_right = ax - 0.5 * aw, ax + 0.5 * aw
    a_bottom, a_top = ay - 0.5 * ah, ay + 0.5 * ah
    b_left, b_right = bx - 0.5 * bw, bx + 0.5 * bw
    b_bottom, b_top = by - 0.5 * bh, by + 0.5 * bh

    inter_w = max(0.0, min(a_right, b_right) - max(a_left, b_left))
    inter_h = max(0.0, min(a_top, b_top) - max(a_bottom, b_bottom))
    inter_area = inter_w * inter_h
    min_area = max(1e-9, min(aw * ah, bw * bh))
    return inter_area / min_area


def _sample_scene_artificial_surface_type_ratios(
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Sample scene-specific artificial surface ratios when no custom distribution is provided."""
    alpha = np.array(
        [
            max(
                0.55,
                14.0 * DEFAULT_ARTIFICIAL_SURFACE_TYPE_RATIOS[surface_type],
            )
            for surface_type in ARTIFICIAL_SURFACE_TYPES
        ],
        dtype=np.float64,
    )
    weights = rng.dirichlet(alpha)
    return {
        surface_type: float(weights[index])
        for index, surface_type in enumerate(ARTIFICIAL_SURFACE_TYPES)
    }


def _build_terrain_height_function(
    area_size: Tuple[float, float],
    terrain_relief: float,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Create deterministic terrain elevation function for the requested scene size."""
    area_w, area_h = area_size
    scale = max(area_w, area_h)
    relief = float(terrain_relief)
    _validate_unit_interval(relief, "terrain_relief")

    def terrain_fn(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)

        z = 2.2 * np.sin(2.0 * np.pi * x_arr / (0.75 * area_w))
        z += 1.5 * np.cos(2.0 * np.pi * y_arr / (0.60 * area_h))
        z += 1.1 * np.sin(2.0 * np.pi * (x_arr + y_arr) / (0.92 * scale))
        z += 0.7 * np.cos(2.0 * np.pi * (x_arr - 1.7 * y_arr) / (1.05 * scale))

        hill = 4.0 * np.exp(
            -(
                ((x_arr - 0.20 * area_w) ** 2) / (2.0 * (0.16 * area_w) ** 2)
                + ((y_arr + 0.24 * area_h) ** 2) / (2.0 * (0.18 * area_h) ** 2)
            )
        )
        valley = -3.0 * np.exp(
            -(
                ((x_arr + 0.28 * area_w) ** 2) / (2.0 * (0.20 * area_w) ** 2)
                + ((y_arr - 0.18 * area_h) ** 2) / (2.0 * (0.16 * area_h) ** 2)
            )
        )
        return relief * (z + hill + valley)

    return terrain_fn


def _clamp_rect_to_area(rect: Rect, area_size: Tuple[float, float]) -> Rect:
    """Clamp rectangle size and center so it stays inside the scene bounds."""
    cx, cy, sx, sy = rect
    area_w, area_h = area_size
    sx_clamped = min(max(0.8, float(sx)), max(0.8, 0.94 * area_w))
    sy_clamped = min(max(0.8, float(sy)), max(0.8, 0.94 * area_h))
    min_cx = -0.5 * area_w + 0.5 * sx_clamped
    max_cx = 0.5 * area_w - 0.5 * sx_clamped
    min_cy = -0.5 * area_h + 0.5 * sy_clamped
    max_cy = 0.5 * area_h - 0.5 * sy_clamped
    cx_clamped = float(np.clip(cx, min_cx, max_cx))
    cy_clamped = float(np.clip(cy, min_cy, max_cy))
    return (cx_clamped, cy_clamped, sx_clamped, sy_clamped)


def _surface_vehicle_allowed(surface_type: str) -> bool:
    """Whether the surface type is a plausible location for parked or moving vehicles."""
    return surface_type in {
        "road_network",
        "parking_lot",
        "building_front_area",
        "industrial_concrete_pad",
    }


def _surface_height_offset(
    rng: np.random.Generator,
    surface_type: str,
) -> float:
    """Type-specific elevation offset above terrain for flat artificial surfaces."""
    if surface_type == "platform":
        return float(rng.uniform(0.18, 0.60))
    if surface_type == "industrial_concrete_pad":
        return float(rng.uniform(0.07, 0.18))
    if surface_type == "building_front_area":
        return float(rng.uniform(0.05, 0.15))
    if surface_type == "sidewalk":
        return float(rng.uniform(0.05, 0.12))
    return float(rng.uniform(0.03, 0.11))


def _make_surface_zone(
    rng: np.random.Generator,
    terrain_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    name: str,
    surface_type: str,
    rect: Rect,
) -> Dict[str, object]:
    """Build normalized artificial surface zone metadata."""
    cx, cy, _, _ = rect
    z0 = float(
        terrain_fn(np.array([cx], dtype=np.float64), np.array([cy], dtype=np.float64))[0]
        + _surface_height_offset(rng, surface_type)
    )
    return {
        "name": name,
        "kind": surface_type,
        "surface_type": surface_type,
        "rect": rect,
        "z0": z0,
        "vehicle_allowed": _surface_vehicle_allowed(surface_type),
    }


def _zone_overlap_is_ok(
    candidate: Rect,
    surface_type: str,
    existing_zones: Sequence[Dict[str, object]],
) -> bool:
    """Check whether the candidate surface can coexist with already placed zones."""
    for zone in existing_zones:
        other_rect = zone["rect"]  # type: ignore[index]
        overlap = _rect_overlap_ratio(candidate, other_rect)
        other_type = str(zone.get("surface_type", zone.get("kind", ""))).strip().lower()

        if surface_type == "road_network" and other_type == "road_network":
            if overlap >= 0.58:
                return False
            continue
        if surface_type in {"sidewalk", "building_front_area"} and other_type in {
            "sidewalk",
            "building_front_area",
        }:
            if overlap >= 0.34:
                return False
            continue
        if overlap >= 0.22:
            return False
    return True


def _sample_detached_surface_rect(
    rng: np.random.Generator,
    area_size: Tuple[float, float],
    surface_type: str,
    existing_zones: Sequence[Dict[str, object]],
) -> Rect:
    """Sample a free-standing artificial surface rectangle."""
    area_w, area_h = area_size
    min_side = min(area_w, area_h)
    fallback: Rect = _clamp_rect_to_area((0.0, 0.0, 12.0, 8.0), area_size)

    for _ in range(48):
        if surface_type == "road_network":
            width = float(rng.uniform(max(5.5, 0.03 * min_side), max(10.5, 0.08 * min_side)))
            if bool(rng.integers(0, 2)):
                length = float(rng.uniform(max(20.0, 0.34 * area_w), max(26.0, 0.94 * area_w)))
                rect = (
                    float(rng.uniform(-0.34 * area_w, 0.34 * area_w)),
                    float(rng.uniform(-0.44 * area_h, 0.44 * area_h)),
                    length,
                    width,
                )
            else:
                length = float(rng.uniform(max(20.0, 0.34 * area_h), max(26.0, 0.94 * area_h)))
                rect = (
                    float(rng.uniform(-0.44 * area_w, 0.44 * area_w)),
                    float(rng.uniform(-0.34 * area_h, 0.34 * area_h)),
                    width,
                    length,
                )
        elif surface_type == "parking_lot":
            rect = (
                float(rng.uniform(-0.42 * area_w, 0.42 * area_w)),
                float(rng.uniform(-0.42 * area_h, 0.42 * area_h)),
                float(rng.uniform(max(12.0, 0.06 * area_w), max(18.0, 0.16 * area_w))),
                float(rng.uniform(max(10.0, 0.06 * area_h), max(16.0, 0.14 * area_h))),
            )
        elif surface_type == "industrial_concrete_pad":
            rect = (
                float(rng.uniform(-0.40 * area_w, 0.40 * area_w)),
                float(rng.uniform(-0.40 * area_h, 0.40 * area_h)),
                float(rng.uniform(max(18.0, 0.08 * area_w), max(28.0, 0.22 * area_w))),
                float(rng.uniform(max(14.0, 0.07 * area_h), max(24.0, 0.18 * area_h))),
            )
        elif surface_type == "platform":
            if bool(rng.integers(0, 2)):
                rect = (
                    float(rng.uniform(-0.40 * area_w, 0.40 * area_w)),
                    float(rng.uniform(-0.40 * area_h, 0.40 * area_h)),
                    float(rng.uniform(max(18.0, 0.10 * area_w), max(44.0, 0.30 * area_w))),
                    float(rng.uniform(4.0, max(6.0, 0.05 * area_h))),
                )
            else:
                rect = (
                    float(rng.uniform(-0.40 * area_w, 0.40 * area_w)),
                    float(rng.uniform(-0.40 * area_h, 0.40 * area_h)),
                    float(rng.uniform(4.0, max(6.0, 0.05 * area_w))),
                    float(rng.uniform(max(18.0, 0.10 * area_h), max(44.0, 0.30 * area_h))),
                )
        elif surface_type == "sidewalk":
            if bool(rng.integers(0, 2)):
                rect = (
                    float(rng.uniform(-0.42 * area_w, 0.42 * area_w)),
                    float(rng.uniform(-0.42 * area_h, 0.42 * area_h)),
                    float(rng.uniform(max(16.0, 0.08 * area_w), max(34.0, 0.22 * area_w))),
                    float(rng.uniform(1.8, 3.6)),
                )
            else:
                rect = (
                    float(rng.uniform(-0.42 * area_w, 0.42 * area_w)),
                    float(rng.uniform(-0.42 * area_h, 0.42 * area_h)),
                    float(rng.uniform(1.8, 3.6)),
                    float(rng.uniform(max(16.0, 0.08 * area_h), max(34.0, 0.22 * area_h))),
                )
        else:
            rect = (
                float(rng.uniform(-0.40 * area_w, 0.40 * area_w)),
                float(rng.uniform(-0.40 * area_h, 0.40 * area_h)),
                float(rng.uniform(max(8.0, 0.05 * area_w), max(18.0, 0.12 * area_w))),
                float(rng.uniform(max(4.0, 0.04 * area_h), max(10.0, 0.10 * area_h))),
            )

        candidate = _clamp_rect_to_area(rect, area_size)
        fallback = candidate
        if _zone_overlap_is_ok(candidate, surface_type, existing_zones):
            return candidate

    return fallback


def _sample_surface_rect_along_building(
    rng: np.random.Generator,
    area_size: Tuple[float, float],
    building_rect: Rect,
    surface_type: str,
    existing_zones: Sequence[Dict[str, object]],
) -> Rect:
    """Sample a rectangle attached to a building edge for sidewalks and front areas."""
    bx, by, bsx, bsy = building_rect
    fallback = _sample_detached_surface_rect(rng, area_size, surface_type, existing_zones)

    for _ in range(32):
        side = str(rng.choice(["north", "south", "east", "west"]))
        if surface_type == "sidewalk":
            extent_scale = float(rng.uniform(0.84, 1.16))
            thickness = float(rng.uniform(1.8, 3.4))
            gap = float(rng.uniform(0.08, 0.40))
            if side in {"north", "south"}:
                rect = (
                    bx,
                    by + (0.5 * bsy + 0.5 * thickness + gap) * (1.0 if side == "north" else -1.0),
                    max(3.0, bsx * extent_scale),
                    thickness,
                )
            else:
                rect = (
                    bx + (0.5 * bsx + 0.5 * thickness + gap) * (1.0 if side == "east" else -1.0),
                    by,
                    thickness,
                    max(3.0, bsy * extent_scale),
                )
        else:
            width_scale = float(rng.uniform(0.92, 1.35))
            depth = float(
                rng.uniform(
                    max(4.0, 0.18 * min(bsx, bsy)),
                    max(7.0, 0.48 * max(bsx, bsy)),
                )
            )
            gap = float(rng.uniform(0.20, 1.20))
            if side in {"north", "south"}:
                rect = (
                    bx,
                    by + (0.5 * bsy + 0.5 * depth + gap) * (1.0 if side == "north" else -1.0),
                    max(4.0, bsx * width_scale),
                    depth,
                )
            else:
                rect = (
                    bx + (0.5 * bsx + 0.5 * depth + gap) * (1.0 if side == "east" else -1.0),
                    by,
                    depth,
                    max(4.0, bsy * width_scale),
                )

        candidate = _clamp_rect_to_area(rect, area_size)
        fallback = candidate
        if _zone_overlap_is_ok(candidate, surface_type, existing_zones):
            return candidate

    return fallback


def _sample_natural_terrain_points(
    rng: np.random.Generator,
    terrain_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    area_size: Tuple[float, float],
    n_points: int,
    forbidden_rects: Sequence[Rect],
    terrain_relief: float,
) -> np.ndarray:
    """Sample terrain points while keeping natural-surface points out of artificial zones."""
    x_nat, y_nat = _sample_xy_outside_rects(
        rng,
        n=n_points,
        area_size=area_size,
        forbidden_rects=forbidden_rects,
        margin=0.0,
    )
    terrain_noise = 0.01 + 0.05 * float(terrain_relief)
    z_nat = terrain_fn(x_nat, y_nat) + rng.normal(0.0, terrain_noise, size=n_points)
    labels = np.full(n_points, NATURAL_SURFACE_CLASS_ID, dtype=np.int32)
    return np.column_stack((x_nat, y_nat, z_nat, labels))


def generate_terrain(
    area_size: Tuple[float, float] = (240.0, 220.0),
    n_points: int = 40_000,
    seed: int = 42,
    terrain_relief: float = 1.0,
    artificial_zones: Sequence[Dict[str, object]] | None = None,
) -> Tuple[np.ndarray, Callable[[np.ndarray, np.ndarray], np.ndarray], List[Dict[str, object]]]:
    """
    Generate natural terrain (class 1) and return:
      - terrain points (N, 4)
      - terrain height function z=f(x,y)
      - artificial zones metadata used to mask class 1 sampling
    `terrain_relief` controls elevation amplitude in [0,1]:
      0 -> almost flat, 1 -> mountainous.
    """
    rng = np.random.default_rng(seed)
    terrain_fn = _build_terrain_height_function(area_size, terrain_relief)
    zones: List[Dict[str, object]]
    if artificial_zones is None:
        zones = []
        detached_rng = np.random.default_rng(seed + 101)
        default_counts = _split_count_by_weights(
            9,
            [
                DEFAULT_ARTIFICIAL_SURFACE_TYPE_RATIOS[surface_type]
                for surface_type in ARTIFICIAL_SURFACE_TYPES
            ],
        )
        for surface_type, count in zip(ARTIFICIAL_SURFACE_TYPES, default_counts):
            for index in range(int(count)):
                rect = _sample_detached_surface_rect(
                    rng=detached_rng,
                    area_size=area_size,
                    surface_type=surface_type,
                    existing_zones=zones,
                )
                zones.append(
                    _make_surface_zone(
                        rng=detached_rng,
                        terrain_fn=terrain_fn,
                        name=f"{surface_type}_{index + 1}",
                        surface_type=surface_type,
                        rect=rect,
                    )
                )
    else:
        zones = list(artificial_zones)
    terrain_points = _sample_natural_terrain_points(
        rng=rng,
        terrain_fn=terrain_fn,
        area_size=area_size,
        n_points=n_points,
        forbidden_rects=[zone["rect"] for zone in zones],  # type: ignore[index]
        terrain_relief=terrain_relief,
    )
    return terrain_points, terrain_fn, zones


def _generate_building_points(
    rng: np.random.Generator,
    n_points: int,
    cx: float,
    cy: float,
    width: float,
    depth: float,
    floor_count: int,
    floor_height: float,
    roof_type: str,
    yaw: float,
    base_z: float,
) -> np.ndarray:
    """Generate one building with configurable roof type and Z rotation."""
    if n_points <= 0:
        return np.empty((0, 4), dtype=np.float64)

    roof_type_id = str(roof_type).strip().lower()
    if roof_type_id not in BUILDING_ROOF_TYPES:
        roof_type_id = "flat"

    floors = max(1, int(floor_count))
    floor_h = max(2.2, float(floor_height))
    body_height = float(max(2.8, floors * floor_h + rng.uniform(-0.12, 0.22) * floor_h))
    roof_height_factor = {
        "single_slope": 0.52,
        "gable": 0.62,
        "hip": 0.56,
        "tent": 0.70,
        "mansard": 0.82,
        "flat": 0.10,
        "dome": 0.88,
        "arched": 0.66,
        "shell": 0.60,
    }
    roof_height = float(
        max(0.3, floor_h * roof_height_factor.get(roof_type_id, 0.55) * rng.uniform(0.85, 1.15))
    )

    roof_share = {
        "flat": 0.22,
        "single_slope": 0.32,
        "gable": 0.34,
        "hip": 0.34,
        "tent": 0.36,
        "mansard": 0.38,
        "dome": 0.42,
        "arched": 0.37,
        "shell": 0.36,
    }.get(roof_type_id, 0.33)
    roof_n = max(1, int(n_points * roof_share))
    wall_n = n_points - roof_n

    half_w = 0.5 * float(width)
    half_d = 0.5 * float(depth)
    face = rng.integers(0, 4, size=wall_n)
    x_wall = np.empty(wall_n, dtype=np.float64) if wall_n > 0 else np.empty(0, dtype=np.float64)
    y_wall = np.empty(wall_n, dtype=np.float64) if wall_n > 0 else np.empty(0, dtype=np.float64)
    z_wall = rng.uniform(0.0, body_height, size=wall_n) if wall_n > 0 else np.empty(0)
    u = rng.uniform(-0.5, 0.5, size=wall_n)

    # Four vertical walls in local coordinates.
    left = face == 0
    right = face == 1
    front = face == 2
    back = face == 3

    if wall_n > 0:
        x_wall[left] = -half_w
        y_wall[left] = u[left] * depth
        x_wall[right] = half_w
        y_wall[right] = u[right] * depth
        x_wall[front] = u[front] * width
        y_wall[front] = half_d
        x_wall[back] = u[back] * width
        y_wall[back] = -half_d

    roof_local = _sample_building_roof_local(
        rng=rng,
        n_points=roof_n,
        width=float(width),
        depth=float(depth),
        roof_type=roof_type_id,
        roof_height=roof_height,
    )
    roof_local[:, 2] += body_height

    x_local = np.concatenate([roof_local[:, 0], x_wall])
    y_local = np.concatenate([roof_local[:, 1], y_wall])
    z_local = np.concatenate([roof_local[:, 2], z_wall])
    x_rot, y_rot = _rotate_xy(x_local, y_local, yaw=yaw)
    x = cx + x_rot + rng.normal(0.0, 0.012, size=n_points)
    y = cy + y_rot + rng.normal(0.0, 0.012, size=n_points)
    z = base_z + z_local + rng.normal(0.0, 0.016, size=n_points)
    z = np.maximum(z, base_z + 0.01)
    labels = np.full(n_points, BUILDINGS_CLASS_ID, dtype=np.int32)
    return np.column_stack((x, y, z, labels))


def _rotate_xy(x: np.ndarray, y: np.ndarray, yaw: float) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate local (x, y) around Z by `yaw` radians."""
    cos_yaw = float(np.cos(yaw))
    sin_yaw = float(np.sin(yaw))
    x_rot = x * cos_yaw - y * sin_yaw
    y_rot = x * sin_yaw + y * cos_yaw
    return x_rot, y_rot


def _sample_building_roof_local(
    rng: np.random.Generator,
    n_points: int,
    width: float,
    depth: float,
    roof_type: str,
    roof_height: float,
) -> np.ndarray:
    """Sample roof surface points in local coordinates; z is above body top."""
    if n_points <= 0:
        return np.empty((0, 3), dtype=np.float64)

    roof_type_id = str(roof_type).strip().lower()
    if roof_type_id not in BUILDING_ROOF_TYPES:
        roof_type_id = "flat"

    half_w = max(0.4, 0.5 * float(width))
    half_d = max(0.4, 0.5 * float(depth))
    roof_h = max(0.25, float(roof_height))

    if roof_type_id == "dome":
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n_points)
        radial = np.sqrt(rng.uniform(0.0, 1.0, size=n_points))
        x = half_w * radial * np.cos(theta)
        y = half_d * radial * np.sin(theta)
        r2 = np.clip((x / half_w) ** 2 + (y / half_d) ** 2, 0.0, 1.0)
        z = roof_h * np.sqrt(1.0 - r2)
        z += rng.normal(0.0, 0.01, size=n_points)
        return np.column_stack((x, y, z))

    x = rng.uniform(-half_w, half_w, size=n_points)
    y = rng.uniform(-half_d, half_d, size=n_points)
    x_norm = np.clip(np.abs(x) / half_w, 0.0, 1.0)
    y_norm = np.clip(np.abs(y) / half_d, 0.0, 1.0)

    if roof_type_id == "single_slope":
        slope = 0.5 + 0.5 * np.clip(x / half_w, -1.0, 1.0)
        z = roof_h * (0.14 + 0.86 * slope)
    elif roof_type_id == "gable":
        ridge = 1.0 - x_norm
        z = roof_h * np.clip(0.10 + 0.90 * ridge, 0.0, 1.0)
    elif roof_type_id == "hip":
        ridge = 1.0 - np.clip(0.62 * x_norm + 0.38 * y_norm, 0.0, 1.0)
        z = roof_h * np.clip(0.08 + 0.92 * ridge, 0.0, 1.0)
    elif roof_type_id == "tent":
        ridge = 1.0 - np.maximum(x_norm, y_norm)
        z = roof_h * np.clip(ridge, 0.0, 1.0)
    elif roof_type_id == "mansard":
        r = np.maximum(x_norm, y_norm)
        lower = np.clip((1.0 - r) / 0.45, 0.0, 1.0)
        upper = np.clip((0.55 - r) / 0.55, 0.0, 1.0)
        z = roof_h * np.clip(0.22 + 0.42 * lower + 0.36 * upper, 0.0, 1.0)
    elif roof_type_id == "arched":
        arch = np.sqrt(np.maximum(0.0, 1.0 - (x / half_w) ** 2))
        z = roof_h * np.clip(0.12 + 0.88 * arch, 0.0, 1.0)
    elif roof_type_id == "shell":
        x_n = np.clip(x / half_w, -1.0, 1.0)
        y_n = np.clip(y / half_d, -1.0, 1.0)
        saddle = 1.0 - 0.5 * (x_n**2 + y_n**2) + 0.30 * x_n * y_n
        z = roof_h * np.clip(0.12 + 0.88 * saddle, 0.02, 1.0)
    else:
        # Flat roof with subtle roughness.
        z = np.full(n_points, 0.03 * roof_h)

    z += rng.normal(0.0, 0.008, size=n_points)
    return np.column_stack((x, y, z))


def _generate_tree_points(
    rng: np.random.Generator,
    n_points: int,
    cx: float,
    cy: float,
    ground_z: float,
    trunk_radius: float,
    trunk_height: float,
    crown_type: str,
    crown_diameter: float,
    crown_bottom_height: float,
    crown_top_height: float,
) -> np.ndarray:
    """Points for one tree with crown shape controlled by `crown_type`."""
    if n_points <= 0:
        return np.empty((0, 4), dtype=np.float64)

    crown_height = max(0.2, float(crown_top_height) - float(crown_bottom_height))
    crown_radius = max(0.25, 0.5 * float(crown_diameter))
    crown_type_id = str(crown_type).strip().lower()
    if crown_type_id not in TREE_CROWN_TYPES:
        crown_type_id = "spherical"

    trunk_share = {
        "spherical": 0.34,
        "pyramidal": 0.30,
        "spreading": 0.36,
        "weeping": 0.33,
        "columnar": 0.31,
        "umbrella": 0.35,
    }.get(crown_type_id, 0.34)
    trunk_n = max(1, int(n_points * trunk_share))
    crown_n = n_points - trunk_n

    # Trunk as mildly tapered cylinder with small lean.
    z_rel = rng.uniform(0.0, 1.0, size=trunk_n)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=trunk_n)
    taper = np.clip(1.0 - 0.2 * z_rel, 0.75, 1.0)
    radial = taper * trunk_radius + rng.normal(0.0, trunk_radius * 0.05, size=trunk_n)
    lean_mag = float(rng.normal(0.0, min(0.14, 0.03 * max(1.0, trunk_height))))
    lean_dir = float(rng.uniform(0.0, 2.0 * np.pi))
    lean_x = lean_mag * np.cos(lean_dir)
    lean_y = lean_mag * np.sin(lean_dir)
    x_trunk = cx + radial * np.cos(theta) + lean_x * z_rel
    y_trunk = cy + radial * np.sin(theta) + lean_y * z_rel
    z_trunk = ground_z + z_rel * trunk_height + rng.normal(0.0, 0.01, size=trunk_n)

    # Crown local coordinates.
    if crown_n > 0:
        crown_local = _sample_tree_crown_local(
            rng=rng,
            n_points=crown_n,
            crown_type=crown_type_id,
            crown_radius=crown_radius,
            crown_height=crown_height,
        )
        x_crown = cx + crown_local[:, 0]
        y_crown = cy + crown_local[:, 1]
        z_crown = ground_z + float(crown_bottom_height) + crown_local[:, 2]
        # Keep crown above trunk shoulder to avoid disconnected tree parts.
        z_floor = ground_z + max(0.08, min(float(crown_bottom_height), 0.92 * trunk_height))
        z_crown = np.maximum(z_crown, z_floor)
    else:
        x_crown = np.empty(0)
        y_crown = np.empty(0)
        z_crown = np.empty(0)

    x = np.concatenate([x_trunk, x_crown])
    y = np.concatenate([y_trunk, y_crown])
    z = np.concatenate([z_trunk, z_crown])
    labels = np.full(x.size, HIGH_VEGETATION_CLASS_ID, dtype=np.int32)
    return np.column_stack((x, y, z, labels))


def _sample_tree_crown_local(
    rng: np.random.Generator,
    n_points: int,
    crown_type: str,
    crown_radius: float,
    crown_height: float,
) -> np.ndarray:
    """Sample local crown points for a tree type; z is in [0, crown_height]."""
    if n_points <= 0:
        return np.empty((0, 3), dtype=np.float64)

    crown_type_id = str(crown_type).strip().lower()
    if crown_type_id not in TREE_CROWN_TYPES:
        crown_type_id = "spherical"
    crown_radius = max(0.25, float(crown_radius))
    crown_height = max(0.2, float(crown_height))

    if crown_type_id == "spherical":
        direction = rng.normal(size=(n_points, 3))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True) + 1e-12
        radius = np.cbrt(rng.uniform(0.0, 1.0, size=n_points))
        local = direction * radius[:, None]
        local[:, 0] *= crown_radius
        local[:, 1] *= crown_radius
        local[:, 2] *= 0.5 * crown_height
        local[:, 2] += 0.5 * crown_height
        return local

    if crown_type_id == "pyramidal":
        z_rel = np.power(rng.uniform(0.0, 1.0, size=n_points), 0.75)
        max_r = crown_radius * np.clip(1.0 - z_rel, 0.0, 1.0)
        r = np.sqrt(rng.uniform(0.0, 1.0, size=n_points)) * max_r
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n_points)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = z_rel * crown_height
        return np.column_stack((x, y, z))

    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_points)
    if crown_type_id == "spreading":
        r_norm = np.power(rng.uniform(0.0, 1.0, size=n_points), 0.42)
        z_bottom = 0.10 + 0.22 * np.power(r_norm, 1.30)
        z_top = 0.94 - 0.44 * np.power(r_norm, 1.85)
    elif crown_type_id == "weeping":
        r_norm = np.power(rng.uniform(0.0, 1.0, size=n_points), 0.55)
        z_bottom = 0.35 - 0.30 * np.power(r_norm, 1.90)
        z_top = 0.95 - 0.46 * np.power(r_norm, 1.35)
    elif crown_type_id == "columnar":
        z_rel = rng.uniform(0.0, 1.0, size=n_points)
        max_r = crown_radius * np.clip(0.42 - 0.16 * np.abs(2.0 * z_rel - 1.0), 0.20, 0.42)
        r = np.sqrt(rng.uniform(0.0, 1.0, size=n_points)) * max_r
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = z_rel * crown_height
        return np.column_stack((x, y, z))
    else:
        # Umbrella crown: dense top + low hanging edge.
        r_norm = np.power(rng.uniform(0.0, 1.0, size=n_points), 0.38)
        z_top = 0.78 + 0.20 * (1.0 - np.power(r_norm, 1.45))
        inner_bottom = 0.52 + 0.10 * (1.0 - r_norm)
        edge_bottom = 0.16 + 0.30 * (1.0 - r_norm)
        z_bottom = np.where(r_norm < 0.65, inner_bottom, edge_bottom)

    r = crown_radius * r_norm
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z_span = np.maximum(0.06, z_top - z_bottom)
    z = (z_bottom + rng.uniform(0.0, 1.0, size=n_points) * z_span) * crown_height
    return np.column_stack((x, y, z))


def _sample_tree_crown_dimensions(
    rng: np.random.Generator,
    crown_type: str,
    random_size: bool,
    tree_max_crown_diameter: float,
    tree_max_crown_top_height: float,
    tree_min_crown_bottom_height: float,
) -> Tuple[float, float, float]:
    """Sample crown diameter, crown bottom and crown top heights above ground."""
    crown_type_id = str(crown_type).strip().lower()
    if crown_type_id not in TREE_CROWN_TYPES:
        crown_type_id = "spherical"

    if random_size:
        type_ranges = {
            "spherical": (2.2, 5.8, 1.2, 3.2, 5.6, 11.8),
            "pyramidal": (2.0, 4.8, 0.7, 2.0, 6.0, 13.2),
            "spreading": (3.2, 7.2, 1.8, 3.9, 6.0, 11.6),
            "weeping": (2.6, 5.8, 1.2, 3.1, 5.4, 10.8),
            "columnar": (1.5, 3.6, 0.8, 2.4, 5.8, 13.5),
            "umbrella": (3.2, 7.6, 2.0, 4.7, 6.0, 12.4),
        }
        d_min, d_max, b_min, b_max, t_min, t_max = type_ranges[crown_type_id]
        crown_diameter = float(rng.uniform(d_min, d_max))
        crown_bottom = float(rng.uniform(b_min, b_max))
        crown_top = float(rng.uniform(max(crown_bottom + 0.8, t_min), t_max))
        return crown_diameter, crown_bottom, crown_top

    type_diameter_min_factor = {
        "spherical": 0.45,
        "pyramidal": 0.42,
        "spreading": 0.50,
        "weeping": 0.44,
        "columnar": 0.34,
        "umbrella": 0.52,
    }
    type_bottom_span_factor = {
        "spherical": 0.44,
        "pyramidal": 0.36,
        "spreading": 0.52,
        "weeping": 0.42,
        "columnar": 0.34,
        "umbrella": 0.58,
    }
    type_top_low_factor = {
        "spherical": 0.68,
        "pyramidal": 0.74,
        "spreading": 0.66,
        "weeping": 0.64,
        "columnar": 0.72,
        "umbrella": 0.70,
    }

    crown_diameter = float(
        rng.uniform(
            max(0.8, type_diameter_min_factor[crown_type_id] * tree_max_crown_diameter),
            float(tree_max_crown_diameter),
        )
    )
    bottom_low = float(tree_min_crown_bottom_height)
    bottom_high = min(
        float(tree_max_crown_top_height) - 0.7,
        bottom_low
        + type_bottom_span_factor[crown_type_id]
        * max(0.8, float(tree_max_crown_top_height) - bottom_low),
    )
    bottom_high = max(bottom_low, bottom_high)
    crown_bottom = float(rng.uniform(bottom_low, bottom_high))

    top_low = max(
        crown_bottom + 0.7,
        type_top_low_factor[crown_type_id] * float(tree_max_crown_top_height),
    )
    top_low = min(top_low, float(tree_max_crown_top_height))
    crown_top = float(rng.uniform(top_low, float(tree_max_crown_top_height)))
    return crown_diameter, crown_bottom, crown_top

def _generate_fence_points(
    rng: np.random.Generator,
    terrain_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_points: int,
    x_center: float,
    y_center: float,
    length: float,
    angle: float,
) -> np.ndarray:
    """Generate elongated fence-like structure."""
    if n_points <= 0:
        return np.empty((0, 4), dtype=np.float64)

    n_rail = int(n_points * 0.65)
    n_post = n_points - n_rail

    direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)

    # Rails: two horizontal lines along fence.
    t = rng.uniform(-0.5, 0.5, size=n_rail) * length
    lateral = rng.normal(0.0, 0.05, size=n_rail)
    x_rail = x_center + direction[0] * t + normal[0] * lateral
    y_rail = y_center + direction[1] * t + normal[1] * lateral
    base_rail = terrain_fn(x_rail, y_rail)
    rail_height = rng.choice([0.75, 1.25], size=n_rail)
    z_rail = base_rail + rail_height + rng.normal(0.0, 0.03, size=n_rail)

    # Posts: vertical elements near discrete fence positions.
    if n_post > 0:
        n_segments = max(4, int(length // 3.0))
        post_t = rng.integers(0, n_segments + 1, size=n_post) / max(1, n_segments)
        post_t = (post_t - 0.5) * length
        post_lateral = rng.normal(0.0, 0.04, size=n_post)
        x_post = x_center + direction[0] * post_t + normal[0] * post_lateral
        y_post = y_center + direction[1] * post_t + normal[1] * post_lateral
        post_base = terrain_fn(x_post, y_post)
        z_post = post_base + rng.uniform(0.0, 1.9, size=n_post)
    else:
        x_post = np.empty(0)
        y_post = np.empty(0)
        z_post = np.empty(0)

    x = np.concatenate([x_rail, x_post])
    y = np.concatenate([y_rail, y_post])
    z = np.concatenate([z_rail, z_post])
    labels = np.full(x.size, STRUCTURES_CLASS_ID, dtype=np.int32)
    return np.column_stack((x, y, z, labels))


def _generate_support_points(
    rng: np.random.Generator,
    terrain_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_points: int,
    x_center: float,
    y_center: float,
    height: float,
) -> np.ndarray:
    """Generate pole/support structure with a cross-beam."""
    if n_points <= 0:
        return np.empty((0, 4), dtype=np.float64)

    n_pole = int(n_points * 0.6)
    n_beam = n_points - n_pole

    base = float(terrain_fn(np.array([x_center]), np.array([y_center]))[0])
    pole_radius = 0.18

    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_pole)
    x_pole = x_center + pole_radius * np.cos(theta)
    y_pole = y_center + pole_radius * np.sin(theta)
    z_pole = base + rng.uniform(0.0, height, size=n_pole)

    if n_beam > 0:
        beam_length = rng.uniform(4.0, 9.0)
        beam_angle = rng.uniform(0.0, 2.0 * np.pi)
        t = rng.uniform(-0.5, 0.5, size=n_beam) * beam_length
        beam_radius = rng.uniform(0.05, 0.12, size=n_beam)
        normal_angle = beam_angle + np.pi * 0.5
        x_beam = (
            x_center
            + np.cos(beam_angle) * t
            + np.cos(normal_angle) * beam_radius * rng.choice([-1.0, 1.0], size=n_beam)
        )
        y_beam = (
            y_center
            + np.sin(beam_angle) * t
            + np.sin(normal_angle) * beam_radius * rng.choice([-1.0, 1.0], size=n_beam)
        )
        z_beam = np.full(n_beam, base + 0.8 * height) + rng.normal(0.0, 0.04, size=n_beam)
    else:
        x_beam = np.empty(0)
        y_beam = np.empty(0)
        z_beam = np.empty(0)

    x = np.concatenate([x_pole, x_beam])
    y = np.concatenate([y_pole, y_beam])
    z = np.concatenate([z_pole, z_beam])
    labels = np.full(x.size, STRUCTURES_CLASS_ID, dtype=np.int32)
    return np.column_stack((x, y, z, labels))


def _generate_lattice_structure_points(
    rng: np.random.Generator,
    terrain_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_points: int,
    x_center: float,
    y_center: float,
    width: float,
    length: float,
    height: float,
    yaw: float,
) -> np.ndarray:
    """Generate random truss/lattice-like structure."""
    if n_points <= 0:
        return np.empty((0, 4), dtype=np.float64)

    base = float(terrain_fn(np.array([x_center]), np.array([y_center]))[0])
    if n_points < 3:
        x = x_center + rng.normal(0.0, 0.08, size=n_points)
        y = y_center + rng.normal(0.0, 0.08, size=n_points)
        z = base + rng.uniform(0.0, height, size=n_points)
        labels = np.full(n_points, 5, dtype=np.int32)
        return np.column_stack((x, y, z, labels))

    n_columns = max(1, int(n_points * 0.40))
    n_beams = max(1, int(n_points * 0.32))
    if n_columns + n_beams >= n_points:
        n_beams = max(1, n_points - n_columns)
    n_braces = max(0, n_points - n_columns - n_beams)

    corners = np.array(
        [
            [-0.5 * width, -0.5 * length],
            [0.5 * width, -0.5 * length],
            [0.5 * width, 0.5 * length],
            [-0.5 * width, 0.5 * length],
        ],
        dtype=np.float64,
    )

    # Vertical columns near corner points.
    col_idx = rng.integers(0, 4, size=n_columns)
    x_col = corners[col_idx, 0] + rng.normal(0.0, 0.04, size=n_columns)
    y_col = corners[col_idx, 1] + rng.normal(0.0, 0.04, size=n_columns)
    z_col = rng.uniform(0.0, height, size=n_columns)

    # Horizontal beams along perimeter at random levels.
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32)
    edge_idx = rng.integers(0, 4, size=n_beams)
    t_beam = rng.uniform(0.0, 1.0, size=n_beams)
    p0 = corners[edges[edge_idx, 0]]
    p1 = corners[edges[edge_idx, 1]]
    beam_xy = p0 + (p1 - p0) * t_beam[:, None]
    x_beam = beam_xy[:, 0] + rng.normal(0.0, 0.03, size=n_beams)
    y_beam = beam_xy[:, 1] + rng.normal(0.0, 0.03, size=n_beams)
    level = rng.choice([0.25, 0.5, 0.75, 1.0], size=n_beams, p=[0.25, 0.30, 0.30, 0.15])
    z_beam = level * height + rng.normal(0.0, 0.03, size=n_beams)

    # Diagonal braces between different corners.
    if n_braces > 0:
        brace_pairs = np.array([[0, 2], [1, 3], [0, 1], [2, 3], [1, 2], [3, 0]], dtype=np.int32)
        pair_idx = rng.integers(0, len(brace_pairs), size=n_braces)
        t_brace = rng.uniform(0.0, 1.0, size=n_braces)
        b0 = corners[brace_pairs[pair_idx, 0]]
        b1 = corners[brace_pairs[pair_idx, 1]]
        brace_xy = b0 + (b1 - b0) * t_brace[:, None]
        x_brace = brace_xy[:, 0] + rng.normal(0.0, 0.03, size=n_braces)
        y_brace = brace_xy[:, 1] + rng.normal(0.0, 0.03, size=n_braces)
        up = pair_idx % 2 == 0
        z_brace = np.where(up, t_brace * height, (1.0 - t_brace) * height)
        z_brace += rng.normal(0.0, 0.03, size=n_braces)
    else:
        x_brace = np.empty(0)
        y_brace = np.empty(0)
        z_brace = np.empty(0)

    x_local = np.concatenate([x_col, x_beam, x_brace])
    y_local = np.concatenate([y_col, y_beam, y_brace])
    z_local = np.concatenate([z_col, z_beam, z_brace])

    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    x = x_center + x_local * cos_yaw - y_local * sin_yaw
    y = y_center + x_local * sin_yaw + y_local * cos_yaw
    z = base + z_local
    labels = np.full(x.size, STRUCTURES_CLASS_ID, dtype=np.int32)
    return np.column_stack((x, y, z, labels))


def _generate_bridge_points(
    rng: np.random.Generator,
    terrain_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_points: int,
    x_center: float,
    y_center: float,
    length: float,
    width: float,
    clearance: float,
    yaw: float,
) -> np.ndarray:
    """Generate bridge-like structure: deck + side rails + vertical supports."""
    if n_points <= 0:
        return np.empty((0, 4), dtype=np.float64)

    center_ground = float(terrain_fn(np.array([x_center]), np.array([y_center]))[0])
    deck_level = center_ground + clearance
    deck_level += float(rng.uniform(0.0, 0.35))
    pier_radius = float(rng.uniform(0.14, 0.28))

    n_deck = max(1, int(n_points * 0.48))
    n_rail = max(1, int(n_points * 0.20))
    n_support = max(0, n_points - n_deck - n_rail)

    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)

    def _local_to_world(x_local: np.ndarray, y_local: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x_world = x_center + x_local * cos_yaw - y_local * sin_yaw
        y_world = y_center + x_local * sin_yaw + y_local * cos_yaw
        return x_world, y_world

    # Deck surface.
    t_deck = rng.uniform(-0.5, 0.5, size=n_deck) * length
    lat_deck = rng.uniform(-0.5, 0.5, size=n_deck) * width
    x_deck, y_deck = _local_to_world(t_deck, lat_deck)
    z_deck = np.full(n_deck, deck_level) + rng.normal(0.0, 0.02, size=n_deck)

    # Side rails.
    side = rng.choice([-1.0, 1.0], size=n_rail)
    t_rail = rng.uniform(-0.5, 0.5, size=n_rail) * length
    lat_rail = side * (0.5 * width - 0.1) + rng.normal(0.0, 0.03, size=n_rail)
    x_rail, y_rail = _local_to_world(t_rail, lat_rail)
    z_rail = deck_level + rng.uniform(0.55, 1.25, size=n_rail) + rng.normal(0.0, 0.02, size=n_rail)

    # Vertical supports.
    if n_support > 0:
        n_piers = int(np.clip(length / rng.uniform(7.0, 12.0), 2, 8))
        t_centers = np.linspace(-0.42 * length, 0.42 * length, n_piers)
        t_centers += rng.normal(0.0, 0.02 * length, size=n_piers)
        per_pier = _split_count_random(
            n_support,
            n_piers,
            rng=rng,
            min_per_chunk=1 if n_support < n_piers * 3 else 3,
        )

        support_x: List[np.ndarray] = []
        support_y: List[np.ndarray] = []
        support_z: List[np.ndarray] = []
        for t0, pier_n in zip(t_centers, per_pier):
            if pier_n <= 0:
                continue
            lat0 = float(rng.normal(0.0, 0.10 * width))
            x0_arr, y0_arr = _local_to_world(np.array([t0]), np.array([lat0]))
            x0 = float(x0_arr[0])
            y0 = float(y0_arr[0])
            ground = float(terrain_fn(np.array([x0]), np.array([y0]))[0])
            height = max(0.8, deck_level - ground)

            theta = rng.uniform(0.0, 2.0 * np.pi, size=int(pier_n))
            radial = pier_radius + rng.normal(0.0, pier_radius * 0.10, size=int(pier_n))
            support_x.append(x0 + radial * np.cos(theta))
            support_y.append(y0 + radial * np.sin(theta))
            support_z.append(ground + rng.uniform(0.0, height, size=int(pier_n)))

        if support_x:
            x_sup = np.concatenate(support_x)
            y_sup = np.concatenate(support_y)
            z_sup = np.concatenate(support_z)
        else:
            x_sup = np.empty(0)
            y_sup = np.empty(0)
            z_sup = np.empty(0)
    else:
        x_sup = np.empty(0)
        y_sup = np.empty(0)
        z_sup = np.empty(0)

    x = np.concatenate([x_deck, x_rail, x_sup])
    y = np.concatenate([y_deck, y_rail, y_sup])
    z = np.concatenate([z_deck, z_rail, z_sup])
    labels = np.full(x.size, STRUCTURES_CLASS_ID, dtype=np.int32)
    return np.column_stack((x, y, z, labels))


def _assemble_structure_points(
    rng: np.random.Generator,
    terrain_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x_center: float,
    y_center: float,
    yaw: float,
    x_local: np.ndarray,
    y_local: np.ndarray,
    z_local: np.ndarray,
    xy_noise: float = 0.01,
    z_noise: float = 0.01,
) -> np.ndarray:
    """Transform local structure coordinates into world space and assign class-5 / structures labels."""
    if x_local.size == 0:
        return np.empty((0, 4), dtype=np.float64)

    x_rot, y_rot = _rotate_xy(x_local, y_local, yaw=yaw)
    x = float(x_center) + x_rot
    y = float(y_center) + y_rot
    if xy_noise > 0.0:
        x = x + rng.normal(0.0, float(xy_noise), size=x.shape[0])
        y = y + rng.normal(0.0, float(xy_noise), size=y.shape[0])
    base = terrain_fn(x, y)
    z = base + np.asarray(z_local, dtype=np.float64)
    if z_noise > 0.0:
        z = z + rng.normal(0.0, float(z_noise), size=z.shape[0])
    z = np.maximum(z, base + 0.01)
    labels = np.full(x.shape[0], STRUCTURES_CLASS_ID, dtype=np.int32)
    return np.column_stack((x, y, z, labels))


def _sample_vertical_cylinder_local(
    rng: np.random.Generator,
    n_points: int,
    radius: float,
    height: float,
    *,
    x_center: float = 0.0,
    y_center: float = 0.0,
    z_base: float = 0.0,
    top_share: float = 0.18,
    bottom_share: float = 0.0,
) -> np.ndarray:
    """Sample points on a vertical cylinder with optional top/bottom caps."""
    if n_points <= 0:
        return np.empty((0, 3), dtype=np.float64)

    radius = max(0.03, float(radius))
    height = max(0.05, float(height))
    top_share = float(np.clip(top_share, 0.0, 0.9))
    bottom_share = float(np.clip(bottom_share, 0.0, 0.9))
    side_share = max(0.0, 1.0 - top_share - bottom_share)
    weights = np.array([side_share, top_share, bottom_share], dtype=np.float64)
    if np.isclose(weights.sum(), 0.0):
        weights[0] = 1.0
    weights /= weights.sum()

    component = rng.choice([0, 1, 2], size=int(n_points), p=weights)
    local = np.empty((int(n_points), 3), dtype=np.float64)

    side = component == 0
    if np.any(side):
        theta = rng.uniform(0.0, 2.0 * np.pi, size=int(np.sum(side)))
        local[side] = np.column_stack(
            (
                x_center + radius * np.cos(theta),
                y_center + radius * np.sin(theta),
                z_base + rng.uniform(0.0, height, size=theta.size),
            )
        )

    top = component == 1
    if np.any(top):
        theta = rng.uniform(0.0, 2.0 * np.pi, size=int(np.sum(top)))
        radial = radius * np.sqrt(rng.uniform(0.0, 1.0, size=theta.size))
        local[top] = np.column_stack(
            (
                x_center + radial * np.cos(theta),
                y_center + radial * np.sin(theta),
                np.full(theta.size, z_base + height),
            )
        )

    bottom = component == 2
    if np.any(bottom):
        theta = rng.uniform(0.0, 2.0 * np.pi, size=int(np.sum(bottom)))
        radial = radius * np.sqrt(rng.uniform(0.0, 1.0, size=theta.size))
        local[bottom] = np.column_stack(
            (
                x_center + radial * np.cos(theta),
                y_center + radial * np.sin(theta),
                np.full(theta.size, z_base),
            )
        )
    return local


def _sample_disk_plane_local(
    rng: np.random.Generator,
    n_points: int,
    radius_x: float,
    radius_y: float,
    *,
    z_value: float,
    x_center: float = 0.0,
    y_center: float = 0.0,
) -> np.ndarray:
    """Sample points on a horizontal ellipse/disc plane in local coordinates."""
    if n_points <= 0:
        return np.empty((0, 3), dtype=np.float64)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=int(n_points))
    radial = np.sqrt(rng.uniform(0.0, 1.0, size=int(n_points)))
    x = x_center + float(radius_x) * radial * np.cos(theta)
    y = y_center + float(radius_y) * radial * np.sin(theta)
    z = np.full(int(n_points), float(z_value), dtype=np.float64)
    return np.column_stack((x, y, z))


def _generate_linear_barrier_points(
    rng: np.random.Generator,
    terrain_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_points: int,
    x_center: float,
    y_center: float,
    *,
    length: float,
    height: float,
    yaw: float,
    width: float = 0.12,
    rail_levels: Sequence[float] = (),
    post_spacing: float = 2.6,
    solid: bool = False,
    xy_noise: float = 0.012,
    z_noise: float = 0.012,
) -> np.ndarray:
    """Generate a generic barrier-like structure: fence, railing, wall, parapet, etc."""
    if n_points <= 0:
        return np.empty((0, 4), dtype=np.float64)

    length = max(1.0, float(length))
    width = max(0.04, float(width))
    height = max(0.15, float(height))
    n_total = int(n_points)

    if solid:
        body_n = max(1, int(n_total * 0.78))
        post_n = max(0, n_total - body_n)
        rail_n = 0
    else:
        rail_n = max(1, int(n_total * 0.44)) if len(rail_levels) > 0 else 0
        post_n = max(1, int(n_total * 0.28))
        body_n = max(0, n_total - rail_n - post_n)

    parts: List[np.ndarray] = []

    if rail_n > 0 and len(rail_levels) > 0:
        t = rng.uniform(-0.5, 0.5, size=rail_n) * length
        lateral = rng.normal(0.0, 0.12 * width, size=rail_n)
        levels = np.asarray(rail_levels, dtype=np.float64)
        z = levels[rng.integers(0, len(levels), size=rail_n)] + rng.normal(0.0, 0.02, size=rail_n)
        parts.append(np.column_stack((t, lateral, z)))

    if post_n > 0:
        n_posts = max(2, int(np.ceil(length / max(0.8, float(post_spacing)))) + 1)
        anchors = np.linspace(-0.5 * length, 0.5 * length, n_posts)
        x_post = rng.choice(anchors, size=post_n) + rng.normal(0.0, 0.04, size=post_n)
        y_post = rng.normal(0.0, 0.18 * width, size=post_n)
        z_post = rng.uniform(0.0, height, size=post_n)
        parts.append(np.column_stack((x_post, y_post, z_post)))

    if body_n > 0:
        if solid:
            face = rng.choice([0, 1, 2, 3], size=body_n, p=[0.34, 0.25, 0.25, 0.16])
            u = rng.uniform(-0.5, 0.5, size=body_n)
            v = rng.uniform(0.0, 1.0, size=body_n)
            x_body = np.empty(body_n, dtype=np.float64)
            y_body = np.empty(body_n, dtype=np.float64)
            z_body = np.empty(body_n, dtype=np.float64)

            top = face == 0
            x_body[top] = u[top] * length
            y_body[top] = rng.uniform(-0.5, 0.5, size=int(np.sum(top))) * width
            z_body[top] = np.full(int(np.sum(top)), height)

            left = face == 1
            x_body[left] = u[left] * length
            y_body[left] = np.full(int(np.sum(left)), -0.5 * width)
            z_body[left] = v[left] * height

            right = face == 2
            x_body[right] = u[right] * length
            y_body[right] = np.full(int(np.sum(right)), 0.5 * width)
            z_body[right] = v[right] * height

            cap = face == 3
            x_body[cap] = rng.choice([-0.5 * length, 0.5 * length], size=int(np.sum(cap)))
            y_body[cap] = rng.uniform(-0.5, 0.5, size=int(np.sum(cap))) * width
            z_body[cap] = v[cap] * height
            parts.append(np.column_stack((x_body, y_body, z_body)))
        else:
            x_body = rng.uniform(-0.5, 0.5, size=body_n) * length
            y_body = rng.normal(0.0, 0.10 * width, size=body_n)
            z_body = rng.uniform(0.08 * height, 0.96 * height, size=body_n)
            parts.append(np.column_stack((x_body, y_body, z_body)))

    local = np.vstack(parts) if parts else np.empty((0, 3), dtype=np.float64)
    return _assemble_structure_points(
        rng=rng,
        terrain_fn=terrain_fn,
        x_center=x_center,
        y_center=y_center,
        yaw=yaw,
        x_local=local[:, 0],
        y_local=local[:, 1],
        z_local=local[:, 2],
        xy_noise=xy_noise,
        z_noise=z_noise,
    )


def _generate_rectangular_enclosure_points(
    rng: np.random.Generator,
    terrain_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_points: int,
    x_center: float,
    y_center: float,
    *,
    size_x: float,
    size_y: float,
    height: float,
    yaw: float,
) -> np.ndarray:
    """Generate a rectangular enclosure / perimeter barrier."""
    if n_points <= 0:
        return np.empty((0, 4), dtype=np.float64)

    size_x = max(2.0, float(size_x))
    size_y = max(2.0, float(size_y))
    counts = _split_count_by_weights(int(n_points), [size_x, size_y, size_x, size_y])
    side_defs = (
        (0.0, 0.5 * size_y, size_x, yaw),
        (0.5 * size_x, 0.0, size_y, yaw + 0.5 * np.pi),
        (0.0, -0.5 * size_y, size_x, yaw),
        (-0.5 * size_x, 0.0, size_y, yaw + 0.5 * np.pi),
    )

    parts: List[np.ndarray] = []
    for count, (local_cx, local_cy, side_length, side_yaw) in zip(counts, side_defs):
        if int(count) <= 0:
            continue
        side_offset_x, side_offset_y = _rotate_xy(
            np.array([local_cx], dtype=np.float64),
            np.array([local_cy], dtype=np.float64),
            yaw=yaw,
        )
        parts.append(
            _generate_linear_barrier_points(
                rng=rng,
                terrain_fn=terrain_fn,
                n_points=int(count),
                x_center=float(x_center + side_offset_x[0]),
                y_center=float(y_center + side_offset_y[0]),
                length=float(side_length),
                height=float(height),
                yaw=float(side_yaw),
                width=0.10,
                rail_levels=(0.58 * height, 0.98 * height),
                post_spacing=float(rng.uniform(1.7, 2.8)),
                solid=False,
                xy_noise=0.010,
                z_noise=0.010,
            )
        )
    return np.vstack(parts) if parts else np.empty((0, 4), dtype=np.float64)


def _generate_lamp_points(
    rng: np.random.Generator,
    terrain_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_points: int,
    x_center: float,
    y_center: float,
    *,
    height: float,
    yaw: float,
) -> np.ndarray:
    """Generate lamp post with pole, arm, and luminaire head."""
    if n_points <= 0:
        return np.empty((0, 4), dtype=np.float64)
    n_pole = max(1, int(n_points * 0.55))
    n_arm = max(1, int(n_points * 0.18))
    n_head = max(0, int(n_points) - n_pole - n_arm)
    arm_len = float(rng.uniform(0.55, 1.35))

    pole = _sample_vertical_cylinder_local(
        rng,
        n_pole,
        radius=float(rng.uniform(0.05, 0.11)),
        height=float(height),
        top_share=0.05,
    )
    arm = np.column_stack(
        (
            rng.uniform(0.0, arm_len, size=n_arm),
            rng.normal(0.0, 0.03, size=n_arm),
            np.full(n_arm, 0.82 * height) + rng.normal(0.0, 0.02, size=n_arm),
        )
    )
    head = _sample_cuboid_surface_local(
        rng,
        n_head,
        length=float(rng.uniform(0.25, 0.55)),
        width=float(rng.uniform(0.16, 0.32)),
        height=float(rng.uniform(0.12, 0.20)),
        x_center=arm_len,
        y_center=0.0,
        z_base=0.72 * height,
    )
    local = np.vstack([pole, arm, head]) if n_head > 0 else np.vstack([pole, arm])
    return _assemble_structure_points(
        rng, terrain_fn, x_center, y_center, yaw, local[:, 0], local[:, 1], local[:, 2]
    )


def _generate_road_sign_points(
    rng: np.random.Generator,
    terrain_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_points: int,
    x_center: float,
    y_center: float,
    *,
    height: float,
    yaw: float,
) -> np.ndarray:
    """Generate road sign with vertical pole and sign board."""
    if n_points <= 0:
        return np.empty((0, 4), dtype=np.float64)
    n_pole = max(1, int(n_points * 0.42))
    n_board = max(0, int(n_points) - n_pole)
    pole = _sample_vertical_cylinder_local(
        rng,
        n_pole,
        radius=float(rng.uniform(0.04, 0.08)),
        height=float(height),
        top_share=0.04,
    )
    board_width = float(rng.uniform(0.45, 1.10))
    board_height = float(rng.uniform(0.40, 0.95))
    board = _sample_cuboid_surface_local(
        rng,
        n_board,
        length=float(rng.uniform(0.05, 0.10)),
        width=board_width,
        height=board_height,
        x_center=float(rng.uniform(-0.02, 0.08)),
        y_center=0.0,
        z_base=float(rng.uniform(0.45, 0.65) * height),
    )
    local = np.vstack([pole, board]) if n_board > 0 else pole
    return _assemble_structure_points(
        rng, terrain_fn, x_center, y_center, yaw, local[:, 0], local[:, 1], local[:, 2]
    )


def _generate_traffic_light_points(
    rng: np.random.Generator,
    terrain_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_points: int,
    x_center: float,
    y_center: float,
    *,
    height: float,
    yaw: float,
) -> np.ndarray:
    """Generate traffic light with pole, horizontal boom, and light heads."""
    if n_points <= 0:
        return np.empty((0, 4), dtype=np.float64)
    n_pole = max(1, int(n_points * 0.40))
    n_arm = max(1, int(n_points * 0.18))
    n_heads_total = max(0, int(n_points) - n_pole - n_arm)
    pole = _sample_vertical_cylinder_local(
        rng,
        n_pole,
        radius=float(rng.uniform(0.06, 0.10)),
        height=float(height),
        top_share=0.04,
    )
    boom_len = float(rng.uniform(1.6, 3.4))
    arm = np.column_stack(
        (
            rng.uniform(0.0, boom_len, size=n_arm),
            rng.normal(0.0, 0.04, size=n_arm),
            np.full(n_arm, 0.82 * height) + rng.normal(0.0, 0.02, size=n_arm),
        )
    )
    head_count = min(3, max(1, n_heads_total))
    head_alloc = _split_count_evenly(n_heads_total, head_count)
    head_parts: List[np.ndarray] = []
    for idx, head_n in enumerate(head_alloc):
        if int(head_n) <= 0:
            continue
        head_parts.append(
            _sample_cuboid_surface_local(
                rng,
                int(head_n),
                length=0.18,
                width=0.26,
                height=0.52,
                x_center=boom_len - 0.22 * idx,
                y_center=0.0,
                z_base=0.56 * height - 0.10 * idx,
            )
        )
    local_parts = [pole, arm]
    if head_parts:
        local_parts.append(np.vstack(head_parts))
    local = np.vstack(local_parts)
    return _assemble_structure_points(
        rng, terrain_fn, x_center, y_center, yaw, local[:, 0], local[:, 1], local[:, 2]
    )


def _generate_bench_points(
    rng: np.random.Generator,
    terrain_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_points: int,
    x_center: float,
    y_center: float,
    *,
    yaw: float,
) -> np.ndarray:
    """Generate bench with seat, backrest, and legs."""
    if n_points <= 0:
        return np.empty((0, 4), dtype=np.float64)
    length = float(rng.uniform(1.3, 2.2))
    seat_depth = float(rng.uniform(0.34, 0.48))
    n_seat = max(1, int(n_points * 0.38))
    n_back = max(1, int(n_points * 0.24))
    n_legs = max(0, int(n_points) - n_seat - n_back)
    seat = _sample_cuboid_surface_local(
        rng,
        n_seat,
        length=length,
        width=seat_depth,
        height=0.08,
        x_center=0.0,
        y_center=0.02,
        z_base=0.42,
    )
    back = _sample_cuboid_surface_local(
        rng,
        n_back,
        length=length,
        width=0.08,
        height=0.42,
        x_center=0.0,
        y_center=-0.5 * seat_depth + 0.02,
        z_base=0.42,
    )
    leg_parts: List[np.ndarray] = []
    if n_legs > 0:
        leg_positions = [
            (-0.34 * length, -0.10),
            (-0.34 * length, 0.10),
            (0.34 * length, -0.10),
            (0.34 * length, 0.10),
        ]
        leg_alloc = _split_count_evenly(n_legs, len(leg_positions))
        for (leg_x, leg_y), leg_n in zip(leg_positions, leg_alloc):
            if int(leg_n) <= 0:
                continue
            leg_parts.append(
                _sample_cuboid_surface_local(
                    rng,
                    int(leg_n),
                    length=0.07,
                    width=0.07,
                    height=0.42,
                    x_center=leg_x,
                    y_center=leg_y,
                    z_base=0.0,
                )
            )
    local_parts = [seat, back]
    if leg_parts:
        local_parts.append(np.vstack(leg_parts))
    local = np.vstack(local_parts)
    return _assemble_structure_points(
        rng, terrain_fn, x_center, y_center, yaw, local[:, 0], local[:, 1], local[:, 2]
    )


def _generate_trash_bin_points(
    rng: np.random.Generator,
    terrain_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_points: int,
    x_center: float,
    y_center: float,
    *,
    yaw: float,
) -> np.ndarray:
    """Generate urban trash bin."""
    if n_points <= 0:
        return np.empty((0, 4), dtype=np.float64)
    local = _sample_vertical_cylinder_local(
        rng,
        int(n_points),
        radius=float(rng.uniform(0.16, 0.28)),
        height=float(rng.uniform(0.75, 1.15)),
        top_share=0.24,
        bottom_share=0.08,
    )
    return _assemble_structure_points(
        rng, terrain_fn, x_center, y_center, yaw, local[:, 0], local[:, 1], local[:, 2]
    )


def _generate_bike_rack_points(
    rng: np.random.Generator,
    terrain_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_points: int,
    x_center: float,
    y_center: float,
    *,
    yaw: float,
) -> np.ndarray:
    """Generate bike rack as a set of repeated U-shaped hoops."""
    if n_points <= 0:
        return np.empty((0, 4), dtype=np.float64)

    loop_count = int(rng.integers(2, 5))
    loop_spacing = float(rng.uniform(0.55, 0.85))
    loop_span = float(rng.uniform(0.45, 0.72))
    loop_height = float(rng.uniform(0.72, 1.05))
    loop_alloc = _split_count_evenly(int(n_points), loop_count)
    loop_centers = np.linspace(
        -0.5 * loop_spacing * max(0, loop_count - 1),
        0.5 * loop_spacing * max(0, loop_count - 1),
        loop_count,
    )

    local_parts: List[np.ndarray] = []
    for loop_x, loop_n in zip(loop_centers, loop_alloc):
        if int(loop_n) <= 0:
            continue
        n_side = max(1, int(loop_n * 0.60))
        n_arc = max(0, int(loop_n) - n_side)
        side_left = n_side // 2
        side_right = n_side - side_left

        left = np.column_stack(
            (
                np.full(side_left, loop_x) + rng.normal(0.0, 0.015, size=side_left),
                np.full(side_left, -0.5 * loop_span) + rng.normal(0.0, 0.015, size=side_left),
                rng.uniform(0.0, loop_height, size=side_left),
            )
        )
        right = np.column_stack(
            (
                np.full(side_right, loop_x) + rng.normal(0.0, 0.015, size=side_right),
                np.full(side_right, 0.5 * loop_span) + rng.normal(0.0, 0.015, size=side_right),
                rng.uniform(0.0, loop_height, size=side_right),
            )
        )
        if n_arc > 0:
            theta = rng.uniform(0.0, np.pi, size=n_arc)
            arc = np.column_stack(
                (
                    np.full(n_arc, loop_x) + rng.normal(0.0, 0.015, size=n_arc),
                    0.5 * loop_span * np.cos(theta),
                    loop_height + 0.22 * loop_span * np.sin(theta),
                )
            )
            local_parts.append(np.vstack([left, right, arc]))
        else:
            local_parts.append(np.vstack([left, right]))
    local = np.vstack(local_parts) if local_parts else np.empty((0, 3), dtype=np.float64)
    return _assemble_structure_points(
        rng, terrain_fn, x_center, y_center, yaw, local[:, 0], local[:, 1], local[:, 2]
    )


def _generate_bollard_points(
    rng: np.random.Generator,
    terrain_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_points: int,
    x_center: float,
    y_center: float,
    *,
    yaw: float,
) -> np.ndarray:
    """Generate short cylindrical bollard."""
    if n_points <= 0:
        return np.empty((0, 4), dtype=np.float64)
    local = _sample_vertical_cylinder_local(
        rng,
        int(n_points),
        radius=float(rng.uniform(0.08, 0.16)),
        height=float(rng.uniform(0.75, 1.15)),
        top_share=0.24,
    )
    return _assemble_structure_points(
        rng, terrain_fn, x_center, y_center, yaw, local[:, 0], local[:, 1], local[:, 2]
    )


def _generate_fountain_points(
    rng: np.random.Generator,
    terrain_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_points: int,
    x_center: float,
    y_center: float,
    *,
    yaw: float,
) -> np.ndarray:
    """Generate simple circular fountain with basin and central jet/pedestal."""
    if n_points <= 0:
        return np.empty((0, 4), dtype=np.float64)
    basin_radius = float(rng.uniform(1.1, 2.8))
    wall_height = float(rng.uniform(0.25, 0.55))
    n_wall = max(1, int(n_points * 0.42))
    n_water = max(1, int(n_points * 0.24))
    n_center = max(0, int(n_points) - n_wall - n_water)

    wall = _sample_vertical_cylinder_local(
        rng,
        n_wall,
        radius=basin_radius,
        height=wall_height,
        top_share=0.22,
    )
    water = _sample_disk_plane_local(
        rng,
        n_water,
        radius_x=0.82 * basin_radius,
        radius_y=0.82 * basin_radius,
        z_value=0.12,
    )
    center_height = float(rng.uniform(0.7, 1.8))
    center = _sample_vertical_cylinder_local(
        rng,
        n_center,
        radius=float(rng.uniform(0.10, 0.26)),
        height=center_height,
        z_base=0.0,
        top_share=0.18,
    )
    local = np.vstack([wall, water, center]) if n_center > 0 else np.vstack([wall, water])
    return _assemble_structure_points(
        rng, terrain_fn, x_center, y_center, yaw, local[:, 0], local[:, 1], local[:, 2]
    )


def _generate_pedestal_points(
    rng: np.random.Generator,
    terrain_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_points: int,
    x_center: float,
    y_center: float,
    *,
    yaw: float,
) -> np.ndarray:
    """Generate pedestal / plinth."""
    if n_points <= 0:
        return np.empty((0, 4), dtype=np.float64)
    local = _sample_cuboid_surface_local(
        rng,
        int(n_points),
        length=float(rng.uniform(0.8, 1.8)),
        width=float(rng.uniform(0.8, 1.8)),
        height=float(rng.uniform(0.8, 1.8)),
        z_base=0.0,
    )
    return _assemble_structure_points(
        rng, terrain_fn, x_center, y_center, yaw, local[:, 0], local[:, 1], local[:, 2]
    )


def _generate_monument_points(
    rng: np.random.Generator,
    terrain_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_points: int,
    x_center: float,
    y_center: float,
    *,
    yaw: float,
) -> np.ndarray:
    """Generate small monument: pedestal plus top mass."""
    if n_points <= 0:
        return np.empty((0, 4), dtype=np.float64)
    n_base = max(1, int(n_points * 0.42))
    n_top = max(0, int(n_points) - n_base)
    base_height = float(rng.uniform(0.55, 1.20))
    base = _sample_cuboid_surface_local(
        rng,
        n_base,
        length=float(rng.uniform(1.0, 2.1)),
        width=float(rng.uniform(1.0, 2.1)),
        height=base_height,
        z_base=0.0,
    )
    top = _sample_cuboid_surface_local(
        rng,
        n_top,
        length=float(rng.uniform(0.35, 0.80)),
        width=float(rng.uniform(0.35, 0.80)),
        height=float(rng.uniform(0.9, 2.4)),
        x_center=0.0,
        y_center=0.0,
        z_base=base_height,
    )
    local = np.vstack([base, top]) if n_top > 0 else base
    return _assemble_structure_points(
        rng, terrain_fn, x_center, y_center, yaw, local[:, 0], local[:, 1], local[:, 2]
    )


def _generate_stairs_points(
    rng: np.random.Generator,
    terrain_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_points: int,
    x_center: float,
    y_center: float,
    *,
    yaw: float,
) -> np.ndarray:
    """Generate small outdoor stairs with step treads and risers."""
    if n_points <= 0:
        return np.empty((0, 4), dtype=np.float64)
    step_count = int(rng.integers(4, 9))
    tread_depth = float(rng.uniform(0.26, 0.38))
    step_height = float(rng.uniform(0.14, 0.19))
    width = float(rng.uniform(1.4, 3.4))
    total_length = step_count * tread_depth

    component = rng.choice([0, 1], size=int(n_points), p=[0.64, 0.36])
    local = np.empty((int(n_points), 3), dtype=np.float64)

    top = component == 0
    if np.any(top):
        steps = rng.integers(0, step_count, size=int(np.sum(top)))
        x = -0.5 * total_length + (steps + rng.uniform(0.05, 0.95, size=steps.size)) * tread_depth
        y = rng.uniform(-0.5, 0.5, size=steps.size) * width
        z = (steps + 1) * step_height
        local[top] = np.column_stack((x, y, z))

    riser = component == 1
    if np.any(riser):
        steps = rng.integers(0, step_count, size=int(np.sum(riser)))
        x = -0.5 * total_length + (steps + 1.0) * tread_depth
        y = rng.uniform(-0.5, 0.5, size=steps.size) * width
        z = steps * step_height + rng.uniform(0.0, step_height, size=steps.size)
        local[riser] = np.column_stack((x, y, z))
    return _assemble_structure_points(
        rng, terrain_fn, x_center, y_center, yaw, local[:, 0], local[:, 1], local[:, 2]
    )


def _generate_ramp_points(
    rng: np.random.Generator,
    terrain_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_points: int,
    x_center: float,
    y_center: float,
    *,
    yaw: float,
) -> np.ndarray:
    """Generate outdoor ramp with sloped plane and optional side rails."""
    if n_points <= 0:
        return np.empty((0, 4), dtype=np.float64)
    length = float(rng.uniform(3.0, 8.0))
    width = float(rng.uniform(1.4, 3.2))
    rise = float(rng.uniform(0.35, 1.25))
    n_plane = max(1, int(n_points * 0.72))
    n_rails = max(0, int(n_points) - n_plane)

    x_plane = rng.uniform(-0.5, 0.5, size=n_plane) * length
    y_plane = rng.uniform(-0.5, 0.5, size=n_plane) * width
    alpha = (x_plane + 0.5 * length) / length
    z_plane = alpha * rise

    parts = [np.column_stack((x_plane, y_plane, z_plane))]
    if n_rails > 0:
        side = rng.choice([-1.0, 1.0], size=n_rails)
        x_rail = rng.uniform(-0.5, 0.5, size=n_rails) * length
        y_rail = side * (0.5 * width - 0.06) + rng.normal(0.0, 0.02, size=n_rails)
        alpha_rail = (x_rail + 0.5 * length) / length
        z_rail = alpha_rail * rise + rng.uniform(0.82, 1.02, size=n_rails)
        parts.append(np.column_stack((x_rail, y_rail, z_rail)))
    local = np.vstack(parts)
    return _assemble_structure_points(
        rng, terrain_fn, x_center, y_center, yaw, local[:, 0], local[:, 1], local[:, 2]
    )


def _generate_platform_points(
    rng: np.random.Generator,
    terrain_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_points: int,
    x_center: float,
    y_center: float,
    *,
    yaw: float,
) -> np.ndarray:
    """Generate raised outdoor platform/deck."""
    if n_points <= 0:
        return np.empty((0, 4), dtype=np.float64)
    deck_height = float(rng.uniform(0.35, 1.2))
    deck = _sample_cuboid_surface_local(
        rng,
        int(n_points),
        length=float(rng.uniform(1.8, 5.6)),
        width=float(rng.uniform(1.6, 4.4)),
        height=deck_height,
        z_base=0.0,
    )
    return _assemble_structure_points(
        rng, terrain_fn, x_center, y_center, yaw, deck[:, 0], deck[:, 1], deck[:, 2]
    )


def _sample_scene_structure_type_ratios(rng: np.random.Generator) -> Dict[str, float]:
    """Sample scene-specific structure type ratios when no custom distribution is provided."""
    alpha = np.array(
        [max(0.45, 18.0 * DEFAULT_STRUCTURE_TYPE_RATIOS[structure_type]) for structure_type in STRUCTURE_TYPES],
        dtype=np.float64,
    )
    weights = rng.dirichlet(alpha)
    return {
        structure_type: float(weights[index])
        for index, structure_type in enumerate(STRUCTURE_TYPES)
    }


def _generate_structure_object_points(
    rng: np.random.Generator,
    terrain_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_points: int,
    x_center: float,
    y_center: float,
    structure_type: str,
    area_size: Tuple[float, float],
) -> np.ndarray:
    """Generate one structure instance of the requested type."""
    structure_type_id = str(structure_type).strip().lower()
    if structure_type_id == "fence":
        return _generate_linear_barrier_points(
            rng,
            terrain_fn,
            n_points,
            x_center,
            y_center,
            length=float(rng.uniform(8.0, 48.0)),
            height=float(rng.uniform(1.3, 2.1)),
            yaw=float(rng.uniform(0.0, 2.0 * np.pi)),
            width=0.08,
            rail_levels=(0.72, 1.28),
            post_spacing=float(rng.uniform(2.0, 3.5)),
            solid=False,
        )
    if structure_type_id == "railing":
        return _generate_linear_barrier_points(
            rng,
            terrain_fn,
            n_points,
            x_center,
            y_center,
            length=float(rng.uniform(4.0, 26.0)),
            height=float(rng.uniform(0.95, 1.35)),
            yaw=float(rng.uniform(0.0, 2.0 * np.pi)),
            width=0.07,
            rail_levels=(0.62, 1.02),
            post_spacing=float(rng.uniform(1.5, 2.4)),
            solid=False,
        )
    if structure_type_id == "enclosure":
        return _generate_rectangular_enclosure_points(
            rng,
            terrain_fn,
            n_points,
            x_center,
            y_center,
            size_x=float(rng.uniform(3.0, 9.0)),
            size_y=float(rng.uniform(3.0, 9.0)),
            height=float(rng.uniform(1.0, 1.8)),
            yaw=float(rng.uniform(0.0, 2.0 * np.pi)),
        )
    if structure_type_id == "guardrail":
        return _generate_linear_barrier_points(
            rng,
            terrain_fn,
            n_points,
            x_center,
            y_center,
            length=float(rng.uniform(7.0, 32.0)),
            height=float(rng.uniform(0.85, 1.15)),
            yaw=float(rng.uniform(0.0, 2.0 * np.pi)),
            width=0.16,
            rail_levels=(0.54, 0.88),
            post_spacing=float(rng.uniform(2.2, 3.8)),
            solid=False,
        )
    if structure_type_id == "retaining_wall":
        return _generate_linear_barrier_points(
            rng,
            terrain_fn,
            n_points,
            x_center,
            y_center,
            length=float(rng.uniform(4.0, 20.0)),
            height=float(rng.uniform(1.2, 3.6)),
            yaw=float(rng.uniform(0.0, 2.0 * np.pi)),
            width=float(rng.uniform(0.22, 0.46)),
            solid=True,
        )
    if structure_type_id == "parapet":
        return _generate_linear_barrier_points(
            rng,
            terrain_fn,
            n_points,
            x_center,
            y_center,
            length=float(rng.uniform(4.0, 18.0)),
            height=float(rng.uniform(0.72, 1.25)),
            yaw=float(rng.uniform(0.0, 2.0 * np.pi)),
            width=float(rng.uniform(0.16, 0.32)),
            solid=True,
        )
    if structure_type_id == "stone_wall":
        return _generate_linear_barrier_points(
            rng,
            terrain_fn,
            n_points,
            x_center,
            y_center,
            length=float(rng.uniform(3.0, 16.0)),
            height=float(rng.uniform(0.7, 1.7)),
            yaw=float(rng.uniform(0.0, 2.0 * np.pi)),
            width=float(rng.uniform(0.28, 0.52)),
            solid=True,
            xy_noise=0.016,
            z_noise=0.018,
        )
    if structure_type_id == "pole_support":
        return _generate_support_points(
            rng=rng,
            terrain_fn=terrain_fn,
            n_points=int(n_points),
            x_center=x_center,
            y_center=y_center,
            height=float(rng.uniform(3.5, 12.0)),
        )
    if structure_type_id == "lamp":
        return _generate_lamp_points(
            rng,
            terrain_fn,
            n_points,
            x_center,
            y_center,
            height=float(rng.uniform(3.8, 8.5)),
            yaw=float(rng.uniform(0.0, 2.0 * np.pi)),
        )
    if structure_type_id == "road_sign":
        return _generate_road_sign_points(
            rng,
            terrain_fn,
            n_points,
            x_center,
            y_center,
            height=float(rng.uniform(2.0, 4.2)),
            yaw=float(rng.uniform(0.0, 2.0 * np.pi)),
        )
    if structure_type_id == "traffic_light":
        return _generate_traffic_light_points(
            rng,
            terrain_fn,
            n_points,
            x_center,
            y_center,
            height=float(rng.uniform(3.6, 6.2)),
            yaw=float(rng.uniform(0.0, 2.0 * np.pi)),
        )
    if structure_type_id == "bench":
        return _generate_bench_points(
            rng,
            terrain_fn,
            n_points,
            x_center,
            y_center,
            yaw=float(rng.uniform(0.0, 2.0 * np.pi)),
        )
    if structure_type_id == "trash_bin":
        return _generate_trash_bin_points(
            rng,
            terrain_fn,
            n_points,
            x_center,
            y_center,
            yaw=float(rng.uniform(0.0, 2.0 * np.pi)),
        )
    if structure_type_id == "bike_rack":
        return _generate_bike_rack_points(
            rng,
            terrain_fn,
            n_points,
            x_center,
            y_center,
            yaw=float(rng.uniform(0.0, 2.0 * np.pi)),
        )
    if structure_type_id == "bollard":
        return _generate_bollard_points(
            rng,
            terrain_fn,
            n_points,
            x_center,
            y_center,
            yaw=float(rng.uniform(0.0, 2.0 * np.pi)),
        )
    if structure_type_id == "fountain":
        return _generate_fountain_points(
            rng,
            terrain_fn,
            n_points,
            x_center,
            y_center,
            yaw=float(rng.uniform(0.0, 2.0 * np.pi)),
        )
    if structure_type_id == "pedestal":
        return _generate_pedestal_points(
            rng,
            terrain_fn,
            n_points,
            x_center,
            y_center,
            yaw=float(rng.uniform(0.0, 2.0 * np.pi)),
        )
    if structure_type_id == "monument":
        return _generate_monument_points(
            rng,
            terrain_fn,
            n_points,
            x_center,
            y_center,
            yaw=float(rng.uniform(0.0, 2.0 * np.pi)),
        )
    if structure_type_id == "stairs":
        return _generate_stairs_points(
            rng,
            terrain_fn,
            n_points,
            x_center,
            y_center,
            yaw=float(rng.uniform(0.0, 2.0 * np.pi)),
        )
    if structure_type_id == "ramp":
        return _generate_ramp_points(
            rng,
            terrain_fn,
            n_points,
            x_center,
            y_center,
            yaw=float(rng.uniform(0.0, 2.0 * np.pi)),
        )
    if structure_type_id == "platform":
        return _generate_platform_points(
            rng,
            terrain_fn,
            n_points,
            x_center,
            y_center,
            yaw=float(rng.uniform(0.0, 2.0 * np.pi)),
        )

    footbridge_length = float(rng.uniform(7.0, max(14.0, 0.32 * min(area_size))))
    return _generate_bridge_points(
        rng=rng,
        terrain_fn=terrain_fn,
        n_points=int(n_points),
        x_center=x_center,
        y_center=y_center,
        length=footbridge_length,
        width=float(rng.uniform(1.8, 3.8)),
        clearance=float(rng.uniform(0.8, 2.8)),
        yaw=float(rng.uniform(0.0, 2.0 * np.pi)),
    )


def _sample_cuboid_surface_local(
    rng: np.random.Generator,
    n_points: int,
    length: float,
    width: float,
    height: float,
    x_center: float = 0.0,
    y_center: float = 0.0,
    z_base: float = 0.0,
) -> np.ndarray:
    """Sample points on cuboid surfaces in local vehicle coordinates."""
    if n_points <= 0:
        return np.empty((0, 3), dtype=np.float64)

    face = rng.integers(0, 6, size=n_points)
    u = rng.uniform(-0.5, 0.5, size=n_points)
    v = rng.uniform(-0.5, 0.5, size=n_points)

    hx = 0.5 * float(length)
    hy = 0.5 * float(width)
    hz = 0.5 * float(height)
    cz = float(z_base) + hz

    local = np.empty((n_points, 3), dtype=np.float64)

    plus_x = face == 0
    minus_x = face == 1
    plus_y = face == 2
    minus_y = face == 3
    plus_z = face == 4
    minus_z = face == 5

    local[plus_x] = np.column_stack(
        [
            np.full(np.sum(plus_x), x_center + hx),
            y_center + u[plus_x] * 2.0 * hy,
            cz + v[plus_x] * 2.0 * hz,
        ]
    )
    local[minus_x] = np.column_stack(
        [
            np.full(np.sum(minus_x), x_center - hx),
            y_center + u[minus_x] * 2.0 * hy,
            cz + v[minus_x] * 2.0 * hz,
        ]
    )
    local[plus_y] = np.column_stack(
        [
            x_center + u[plus_y] * 2.0 * hx,
            np.full(np.sum(plus_y), y_center + hy),
            cz + v[plus_y] * 2.0 * hz,
        ]
    )
    local[minus_y] = np.column_stack(
        [
            x_center + u[minus_y] * 2.0 * hx,
            np.full(np.sum(minus_y), y_center - hy),
            cz + v[minus_y] * 2.0 * hz,
        ]
    )
    local[plus_z] = np.column_stack(
        [
            x_center + u[plus_z] * 2.0 * hx,
            y_center + v[plus_z] * 2.0 * hy,
            np.full(np.sum(plus_z), cz + hz),
        ]
    )
    local[minus_z] = np.column_stack(
        [
            x_center + u[minus_z] * 2.0 * hx,
            y_center + v[minus_z] * 2.0 * hy,
            np.full(np.sum(minus_z), cz - hz),
        ]
    )
    return local


def _sample_sloped_front_local(
    rng: np.random.Generator,
    n_points: int,
    width: float,
    x_rear: float,
    x_front: float,
    z_rear: float,
    z_front: float,
) -> np.ndarray:
    """Sample points on sloped hood / windshield-like front section."""
    if n_points <= 0:
        return np.empty((0, 3), dtype=np.float64)

    x0 = float(min(x_rear, x_front))
    x1 = float(max(x_rear, x_front))
    z0 = float(z_rear)
    z1 = float(z_front)
    half_w = 0.5 * float(width)
    if x1 - x0 < 1e-4:
        x1 = x0 + 1e-4

    part = rng.choice([0, 1, 2], size=n_points, p=[0.52, 0.30, 0.18])
    local = np.empty((n_points, 3), dtype=np.float64)

    mask_top = part == 0
    if np.any(mask_top):
        xt = rng.uniform(x0, x1, size=int(np.sum(mask_top)))
        yt = rng.uniform(-half_w, half_w, size=int(np.sum(mask_top)))
        alpha = (xt - x0) / (x1 - x0)
        zt = z0 + alpha * (z1 - z0) + rng.normal(0.0, 0.01, size=xt.size)
        local[mask_top] = np.column_stack((xt, yt, zt))

    mask_front = part == 1
    if np.any(mask_front):
        yf = rng.uniform(-half_w, half_w, size=int(np.sum(mask_front)))
        zf = rng.uniform(min(z0, z1), max(z0, z1), size=yf.size)
        local[mask_front] = np.column_stack(
            (np.full(yf.size, x1), yf, zf + rng.normal(0.0, 0.01, size=yf.size))
        )

    mask_side = part == 2
    if np.any(mask_side):
        xs = rng.uniform(x0, x1, size=int(np.sum(mask_side)))
        side = rng.choice([-1.0, 1.0], size=xs.size)
        ys = side * half_w + rng.normal(0.0, 0.01, size=xs.size)
        alpha = (xs - x0) / (x1 - x0)
        z_top = z0 + alpha * (z1 - z0)
        zs = rng.uniform(0.25 * min(z0, z1), z_top, size=xs.size)
        local[mask_side] = np.column_stack((xs, ys, zs))

    return local


def _sample_wheel_surface_local(
    rng: np.random.Generator,
    n_points: int,
    x_center: float,
    y_center: float,
    z_center: float,
    radius: float,
    thickness: float,
) -> np.ndarray:
    """Sample wheel-like points (rim + side discs)."""
    if n_points <= 0:
        return np.empty((0, 3), dtype=np.float64)

    radius = max(0.12, float(radius))
    half_t = max(0.04, 0.5 * float(thickness))
    comp = rng.choice([0, 1, 2], size=n_points, p=[0.56, 0.22, 0.22])
    local = np.empty((n_points, 3), dtype=np.float64)

    mask_rim = comp == 0
    if np.any(mask_rim):
        theta = rng.uniform(0.0, 2.0 * np.pi, size=int(np.sum(mask_rim)))
        y = y_center + rng.uniform(-half_t, half_t, size=theta.size)
        x = x_center + radius * np.cos(theta)
        z = z_center + radius * np.sin(theta)
        local[mask_rim] = np.column_stack((x, y, z))

    mask_left = comp == 1
    if np.any(mask_left):
        theta = rng.uniform(0.0, 2.0 * np.pi, size=int(np.sum(mask_left)))
        r = radius * np.sqrt(rng.uniform(0.0, 1.0, size=theta.size))
        x = x_center + r * np.cos(theta)
        z = z_center + r * np.sin(theta)
        y = np.full(theta.size, y_center + half_t)
        local[mask_left] = np.column_stack((x, y, z))

    mask_right = comp == 2
    if np.any(mask_right):
        theta = rng.uniform(0.0, 2.0 * np.pi, size=int(np.sum(mask_right)))
        r = radius * np.sqrt(rng.uniform(0.0, 1.0, size=theta.size))
        x = x_center + r * np.cos(theta)
        z = z_center + r * np.sin(theta)
        y = np.full(theta.size, y_center - half_t)
        local[mask_right] = np.column_stack((x, y, z))

    local[:, 2] = np.maximum(local[:, 2], 0.01)
    return local


def _sample_arched_roof_local(
    rng: np.random.Generator,
    n_points: int,
    length: float,
    width: float,
    x_center: float,
    z_base: float,
    arch_height: float,
) -> np.ndarray:
    """Sample points on arched roof section (typical for buses)."""
    if n_points <= 0:
        return np.empty((0, 3), dtype=np.float64)

    x = x_center + rng.uniform(-0.5 * length, 0.5 * length, size=n_points)
    side = rng.uniform(-1.0, 1.0, size=n_points)
    y = side * 0.5 * width
    lateral = np.clip(np.abs(side), 0.0, 1.0)
    arch = np.sqrt(np.maximum(0.0, 1.0 - lateral**2))
    z = z_base + arch_height * arch + rng.normal(0.0, 0.01, size=n_points)
    return np.column_stack((x, y, z))


def _sample_vehicle_dimensions(
    rng: np.random.Generator,
    vehicle_type: str,
) -> Tuple[float, float, float]:
    """Sample plausible dimensions in meters for each vehicle type."""
    vehicle_type = str(vehicle_type).lower()
    if vehicle_type == "truck":
        return (
            float(rng.uniform(6.6, 11.2)),
            float(rng.uniform(2.2, 2.8)),
            float(rng.uniform(2.8, 4.1)),
        )
    if vehicle_type == "bus":
        return (
            float(rng.uniform(9.8, 13.6)),
            float(rng.uniform(2.4, 2.9)),
            float(rng.uniform(3.0, 3.9)),
        )
    # passenger car
    return (
        float(rng.uniform(3.8, 5.2)),
        float(rng.uniform(1.65, 2.05)),
        float(rng.uniform(1.35, 1.85)),
    )


def _generate_vehicle_points(
    rng: np.random.Generator,
    n_points: int,
    x_center: float,
    y_center: float,
    base_z: float,
    length: float,
    width: float,
    height: float,
    yaw: float,
    vehicle_type: str = "car",
) -> np.ndarray:
    """Generate vehicle-like points: body + cabin/roof + wheels."""
    if n_points <= 0:
        return np.empty((0, 4), dtype=np.float64)

    vehicle_type = str(vehicle_type).lower()
    if vehicle_type not in VEHICLE_TYPES:
        vehicle_type = "car"

    parts: List[np.ndarray] = []
    if vehicle_type == "truck":
        n_chassis, n_cargo, n_cabin, n_front, n_wheels = _split_count_by_weights(
            n_points, [0.22, 0.30, 0.18, 0.10, 0.20]
        )
        chassis_h = 0.28 * height
        parts.append(
            _sample_cuboid_surface_local(
                rng, int(n_chassis), length * 0.96, width * 0.94, chassis_h, z_base=0.0
            )
        )
        parts.append(
            _sample_cuboid_surface_local(
                rng,
                int(n_cargo),
                length * 0.60,
                width * 0.92,
                height * 0.54,
                x_center=-0.13 * length,
                z_base=chassis_h,
            )
        )
        parts.append(
            _sample_cuboid_surface_local(
                rng,
                int(n_cabin),
                length * 0.25,
                width * 0.88,
                height * 0.62,
                x_center=0.30 * length,
                z_base=chassis_h,
            )
        )
        parts.append(
            _sample_sloped_front_local(
                rng=rng,
                n_points=int(n_front),
                width=width * 0.84,
                x_rear=0.31 * length,
                x_front=0.48 * length,
                z_rear=0.30 * height,
                z_front=0.76 * height,
            )
        )
        wheel_centers = [
            (-0.35 * length, +0.48 * width),
            (-0.35 * length, -0.48 * width),
            (-0.06 * length, +0.48 * width),
            (-0.06 * length, -0.48 * width),
            (0.25 * length, +0.48 * width),
            (0.25 * length, -0.48 * width),
        ]
        wheel_radius = max(0.34, min(0.72, 0.19 * height))
        wheel_thickness = max(0.18, 0.18 * width)
        wheel_counts = _split_count_evenly(int(n_wheels), len(wheel_centers))
        for n_wheel, (wx, wy) in zip(wheel_counts, wheel_centers):
            parts.append(
                _sample_wheel_surface_local(
                    rng=rng,
                    n_points=int(n_wheel),
                    x_center=float(wx),
                    y_center=float(wy),
                    z_center=wheel_radius + 0.03,
                    radius=wheel_radius,
                    thickness=wheel_thickness,
                )
            )
    elif vehicle_type == "bus":
        n_body, n_roof, n_caps, n_wheels = _split_count_by_weights(
            n_points, [0.45, 0.20, 0.15, 0.20]
        )
        body_h = 0.64 * height
        parts.append(
            _sample_cuboid_surface_local(
                rng, int(n_body), length * 0.96, width * 0.96, body_h, z_base=0.0
            )
        )
        parts.append(
            _sample_arched_roof_local(
                rng=rng,
                n_points=int(n_roof),
                length=length * 0.90,
                width=width * 0.92,
                x_center=0.0,
                z_base=body_h,
                arch_height=0.34 * height,
            )
        )
        cap_front, cap_back = _split_count_by_weights(int(n_caps), [0.5, 0.5])
        parts.append(
            _sample_sloped_front_local(
                rng=rng,
                n_points=int(cap_front),
                width=width * 0.93,
                x_rear=0.28 * length,
                x_front=0.48 * length,
                z_rear=0.28 * height,
                z_front=0.86 * height,
            )
        )
        parts.append(
            _sample_sloped_front_local(
                rng=rng,
                n_points=int(cap_back),
                width=width * 0.93,
                x_rear=-0.48 * length,
                x_front=-0.28 * length,
                z_rear=0.86 * height,
                z_front=0.28 * height,
            )
        )
        wheel_centers = [
            (-0.30 * length, +0.48 * width),
            (-0.30 * length, -0.48 * width),
            (0.30 * length, +0.48 * width),
            (0.30 * length, -0.48 * width),
        ]
        wheel_radius = max(0.36, min(0.74, 0.18 * height))
        wheel_thickness = max(0.20, 0.18 * width)
        wheel_counts = _split_count_evenly(int(n_wheels), len(wheel_centers))
        for n_wheel, (wx, wy) in zip(wheel_counts, wheel_centers):
            parts.append(
                _sample_wheel_surface_local(
                    rng=rng,
                    n_points=int(n_wheel),
                    x_center=float(wx),
                    y_center=float(wy),
                    z_center=wheel_radius + 0.03,
                    radius=wheel_radius,
                    thickness=wheel_thickness,
                )
            )
    else:
        n_body, n_cabin, n_front, n_wheels = _split_count_by_weights(
            n_points, [0.44, 0.24, 0.12, 0.20]
        )
        body_h = 0.56 * height
        parts.append(
            _sample_cuboid_surface_local(
                rng,
                int(n_body),
                length * 0.94,
                width * 0.95,
                body_h,
                x_center=-0.03 * length,
                z_base=0.0,
            )
        )
        parts.append(
            _sample_cuboid_surface_local(
                rng,
                int(n_cabin),
                length * 0.48,
                width * 0.82,
                height * 0.34,
                x_center=-0.07 * length,
                z_base=body_h * 0.72,
            )
        )
        parts.append(
            _sample_sloped_front_local(
                rng=rng,
                n_points=int(n_front),
                width=width * 0.88,
                x_rear=0.02 * length,
                x_front=0.47 * length,
                z_rear=0.24 * height,
                z_front=0.68 * height,
            )
        )
        wheel_centers = [
            (-0.28 * length, +0.47 * width),
            (-0.28 * length, -0.47 * width),
            (0.23 * length, +0.47 * width),
            (0.23 * length, -0.47 * width),
        ]
        wheel_radius = max(0.22, min(0.40, 0.24 * height))
        wheel_thickness = max(0.14, 0.18 * width)
        wheel_counts = _split_count_evenly(int(n_wheels), len(wheel_centers))
        for n_wheel, (wx, wy) in zip(wheel_counts, wheel_centers):
            parts.append(
                _sample_wheel_surface_local(
                    rng=rng,
                    n_points=int(n_wheel),
                    x_center=float(wx),
                    y_center=float(wy),
                    z_center=wheel_radius + 0.03,
                    radius=wheel_radius,
                    thickness=wheel_thickness,
                )
            )

    local = np.vstack([part for part in parts if part.size > 0])
    local[:, 2] = np.maximum(local[:, 2], 0.01)

    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    x = x_center + local[:, 0] * cos_yaw - local[:, 1] * sin_yaw
    y = y_center + local[:, 0] * sin_yaw + local[:, 1] * cos_yaw
    z = base_z + local[:, 2] + rng.normal(0.0, 0.01, size=local.shape[0])
    z = np.maximum(z, base_z + 0.005)

    labels = np.full(local.shape[0], VEHICLES_CLASS_ID, dtype=np.int32)
    return np.column_stack((x, y, z, labels))


def _resolve_low_veg_counts(
    n_low_veg_points: int,
    area_size: Tuple[float, float],
    rng: np.random.Generator,
    shrub_count: int | None,
    grass_patch_count: int | None,
) -> Tuple[int, int]:
    """Resolve shrub and grass patch counts (random or user-defined)."""
    if n_low_veg_points <= 0:
        return 0, 0

    width_m, length_m = area_size
    area_ha = (width_m * length_m) / 10_000.0

    if shrub_count is None:
        low = max(2, int(np.floor(10.0 * area_ha)))
        high = max(low + 1, int(np.ceil(28.0 * area_ha + 3.0)))
        shrub_count_eff = int(rng.integers(low, high + 1))
    else:
        shrub_count_eff = int(shrub_count)
        if shrub_count_eff <= 0:
            raise ValueError("`shrub_count` must be > 0 when provided.")

    if grass_patch_count is None:
        low = max(2, int(np.floor(7.0 * area_ha)))
        high = max(low + 1, int(np.ceil(22.0 * area_ha + 2.0)))
        grass_patch_count_eff = int(rng.integers(low, high + 1))
    else:
        grass_patch_count_eff = int(grass_patch_count)
        if grass_patch_count_eff <= 0:
            raise ValueError("`grass_patch_count` must be > 0 when provided.")

    shrub_count_eff = min(shrub_count_eff, n_low_veg_points)
    grass_patch_count_eff = min(grass_patch_count_eff, n_low_veg_points)
    return max(1, shrub_count_eff), max(1, grass_patch_count_eff)


def _generate_shrub_points(
    rng: np.random.Generator,
    terrain_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_points: int,
    center_x: float,
    center_y: float,
    random_size: bool,
    shrub_max_diameter: float,
    shrub_max_top_height: float,
    shrub_min_bottom_height: float,
) -> np.ndarray:
    """Generate one shrub cluster (class 3) with crown-like geometry."""
    if n_points <= 0:
        return np.empty((0, 4), dtype=np.float64)

    if random_size:
        diameter = float(rng.uniform(0.75, 2.9))
        top_h = float(rng.uniform(0.55, 1.85))
        bottom_h = float(rng.uniform(0.05, min(0.55, top_h - 0.10)))
    else:
        diameter = float(
            rng.uniform(max(0.45, 0.45 * shrub_max_diameter), float(shrub_max_diameter))
        )
        bottom_low = float(shrub_min_bottom_height)
        bottom_high = min(
            float(shrub_max_top_height) - 0.10,
            bottom_low + 0.30 * max(0.12, float(shrub_max_top_height) - bottom_low),
        )
        bottom_high = max(bottom_low, bottom_high)
        bottom_h = float(rng.uniform(bottom_low, bottom_high))

        top_low = max(bottom_h + 0.10, 0.65 * float(shrub_max_top_height))
        top_low = min(top_low, float(shrub_max_top_height))
        top_h = float(rng.uniform(top_low, float(shrub_max_top_height)))

    radius = max(0.18, 0.5 * diameter)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_points)
    radial_norm = np.sqrt(rng.uniform(0.0, 1.0, size=n_points))
    radial = radius * radial_norm

    x = center_x + radial * np.cos(theta)
    y = center_y + radial * np.sin(theta)
    ground = terrain_fn(x, y)

    edge_decay = np.power(np.clip(radial_norm, 0.0, 1.0), 1.7)
    crown_top = top_h * (1.0 - 0.62 * edge_decay)
    crown_top = np.maximum(crown_top, bottom_h + 0.08)
    crown_bottom = bottom_h + 0.08 * (1.0 - edge_decay)
    local_h = crown_bottom + rng.uniform(0.0, 1.0, size=n_points) * (crown_top - crown_bottom)
    z = ground + local_h + rng.normal(0.0, 0.01, size=n_points)

    labels = np.full(n_points, LOW_VEGETATION_CLASS_ID, dtype=np.int32)
    return np.column_stack((x, y, z, labels))


def _generate_grass_patch_points(
    rng: np.random.Generator,
    terrain_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_points: int,
    center_x: float,
    center_y: float,
    random_size: bool,
    grass_patch_max_size_x: float,
    grass_patch_max_size_y: float,
    grass_max_height: float,
) -> np.ndarray:
    """Generate one grassy patch (class 3) as an anisotropic ellipse."""
    if n_points <= 0:
        return np.empty((0, 4), dtype=np.float64)

    if random_size:
        size_x = float(rng.uniform(0.8, 3.8))
        size_y = float(rng.uniform(0.8, 3.6))
        height_max = float(rng.uniform(0.10, 0.72))
    else:
        size_x = float(
            rng.uniform(max(0.45, 0.40 * grass_patch_max_size_x), float(grass_patch_max_size_x))
        )
        size_y = float(
            rng.uniform(max(0.45, 0.40 * grass_patch_max_size_y), float(grass_patch_max_size_y))
        )
        height_max = float(
            rng.uniform(max(0.08, 0.35 * grass_max_height), float(grass_max_height))
        )

    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_points)
    radial_norm = np.sqrt(rng.uniform(0.0, 1.0, size=n_points))
    x = center_x + 0.5 * size_x * radial_norm * np.cos(theta)
    y = center_y + 0.5 * size_y * radial_norm * np.sin(theta)

    ground = terrain_fn(x, y)
    edge_profile = 1.0 - 0.55 * np.power(np.clip(radial_norm, 0.0, 1.0), 1.4)
    local_h_max = np.maximum(0.05, height_max * edge_profile)
    z = ground + rng.uniform(0.02, 1.0, size=n_points) * local_h_max
    z += rng.normal(0.0, 0.006, size=n_points)

    labels = np.full(n_points, LOW_VEGETATION_CLASS_ID, dtype=np.int32)
    return np.column_stack((x, y, z, labels))


def place_objects(
    terrain_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    area_size: Tuple[float, float],
    points_per_class: Dict[int, int],
    num_artificial_surfaces: int = 9,
    artificial_surface_type_ratios: Dict[str, float] | None = None,
    num_trees: int = 70,
    num_buildings: int = 14,
    building_roof_type_ratios: Dict[str, float] | None = None,
    building_floor_min: int = int(BUILDING_DEFAULTS["building_floor_min"]),
    building_floor_max: int = int(BUILDING_DEFAULTS["building_floor_max"]),
    building_random_yaw: bool = bool(BUILDING_DEFAULTS["building_random_yaw"]),
    num_structures: int = 10,
    structure_type_ratios: Dict[str, float] | None = None,
    num_vehicles: int = 24,
    vehicle_type_ratios: Dict[str, float] | None = None,
    tree_crown_type_ratios: Dict[str, float] | None = None,
    random_tree_crown_size: bool = True,
    tree_max_crown_diameter: float = HIGH_VEG_DEFAULTS["tree_max_crown_diameter"],
    tree_max_crown_top_height: float = HIGH_VEG_DEFAULTS["tree_max_crown_top_height"],
    tree_min_crown_bottom_height: float = HIGH_VEG_DEFAULTS["tree_min_crown_bottom_height"],
    num_artifact_clusters: int = 40,
    artifacts_enabled: bool = bool(ARTIFACT_DEFAULTS["enabled"]),
    artifact_global_intensity: float = float(ARTIFACT_DEFAULTS["global_intensity"]),
    artifact_type_settings: Mapping[str, Mapping[str, Any]] | None = None,
    shrub_count: int | None = None,
    random_shrub_size: bool = True,
    shrub_max_diameter: float = LOW_VEG_DEFAULTS["shrub_max_diameter"],
    shrub_max_top_height: float = LOW_VEG_DEFAULTS["shrub_max_top_height"],
    shrub_min_bottom_height: float = LOW_VEG_DEFAULTS["shrub_min_bottom_height"],
    grass_patch_count: int | None = None,
    random_grass_patch_size: bool = True,
    grass_patch_max_size_x: float = LOW_VEG_DEFAULTS["grass_patch_max_size_x"],
    grass_patch_max_size_y: float = LOW_VEG_DEFAULTS["grass_patch_max_size_y"],
    grass_max_height: float = LOW_VEG_DEFAULTS["grass_max_height"],
    seed: int = 43,
    progress_callback: ProgressCallback | None = None,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    """
    Place all non-terrain classes and return:
      - one (N, 4) array with classes 0 and 2..7
      - generated artificial surface zones metadata
    Uses classes:
      0 artificial surfaces
      2 high vegetation
      3 low vegetation
      4 buildings
      5 structures
      6 artifacts
      7 vehicles
    """
    rng = np.random.default_rng(seed)
    cloud_parts: List[np.ndarray] = []
    building_rects: List[Rect] = []

    if num_artificial_surfaces <= 0:
        raise ValueError("`num_artificial_surfaces` must be > 0.")

    _emit_progress(progress_callback, 0.02, "Generating artificial zones")

    if artificial_surface_type_ratios is None:
        artificial_surface_type_ratios_eff = _sample_scene_artificial_surface_type_ratios(rng)
    else:
        artificial_surface_type_ratios_eff = {
            surface_type: float(artificial_surface_type_ratios.get(surface_type, 0.0))
            for surface_type in ARTIFICIAL_SURFACE_TYPES
        }
        if np.isclose(sum(artificial_surface_type_ratios_eff.values()), 0.0):
            artificial_surface_type_ratios_eff = dict(DEFAULT_ARTIFICIAL_SURFACE_TYPE_RATIOS)

    surface_type_counts = {
        surface_type: int(count)
        for surface_type, count in zip(
            ARTIFICIAL_SURFACE_TYPES,
            _split_count_by_weights(
                int(num_artificial_surfaces),
                [
                    artificial_surface_type_ratios_eff[surface_type]
                    for surface_type in ARTIFICIAL_SURFACE_TYPES
                ],
            ),
        )
    }

    artificial_zones: List[Dict[str, object]] = []
    for surface_type in ARTIFICIAL_SURFACE_TYPES:
        if surface_type in {"sidewalk", "building_front_area"}:
            continue
        for index in range(surface_type_counts[surface_type]):
            rect = _sample_detached_surface_rect(
                rng=rng,
                area_size=area_size,
                surface_type=surface_type,
                existing_zones=artificial_zones,
            )
            artificial_zones.append(
                _make_surface_zone(
                    rng=rng,
                    terrain_fn=terrain_fn,
                    name=f"{surface_type}_{index + 1}",
                    surface_type=surface_type,
                    rect=rect,
                )
            )

    base_artificial_rects: List[Rect] = [
        zone["rect"] for zone in artificial_zones  # type: ignore[index]
    ]

    _emit_progress(progress_callback, 0.16, "Generating buildings")

    # ------------------------------------------------------------------
    # Class 4: buildings (cuboids sampled on walls + roof)
    # ------------------------------------------------------------------
    n_building_points = int(points_per_class.get(BUILDINGS_CLASS_ID, 0))
    if n_building_points > 0:
        n_buildings_eff = max(1, min(num_buildings, n_building_points))
        points_per_building = _split_count_random(
            n_building_points,
            n_buildings_eff,
            rng=rng,
            min_per_chunk=20,
        )
        if building_roof_type_ratios is None:
            random_weights = rng.dirichlet(
                np.array([1.3, 2.0, 1.7, 1.1, 1.2, 2.3, 0.7, 1.0, 0.8], dtype=np.float64)
            )
            building_roof_type_ratios_eff = {
                roof_type: float(random_weights[idx])
                for idx, roof_type in enumerate(BUILDING_ROOF_TYPES)
            }
        else:
            building_roof_type_ratios_eff = {
                roof_type: float(building_roof_type_ratios.get(roof_type, 0.0))
                for roof_type in BUILDING_ROOF_TYPES
            }
            if np.isclose(sum(building_roof_type_ratios_eff.values()), 0.0):
                building_roof_type_ratios_eff = dict(DEFAULT_BUILDING_ROOF_TYPE_RATIOS)

        per_roof_type = _split_count_by_weights(
            n_buildings_eff,
            [building_roof_type_ratios_eff[roof_type] for roof_type in BUILDING_ROOF_TYPES],
        )
        roof_types: List[str] = []
        for roof_type, type_count in zip(BUILDING_ROOF_TYPES, per_roof_type):
            roof_types.extend([roof_type] * int(type_count))
        if len(roof_types) < n_buildings_eff:
            roof_types.extend(["flat"] * (n_buildings_eff - len(roof_types)))
        roof_types = roof_types[:n_buildings_eff]
        rng.shuffle(roof_types)

        floor_min = max(1, int(min(building_floor_min, building_floor_max)))
        floor_max = max(floor_min, int(max(building_floor_min, building_floor_max)))
        for n_pts, roof_type in zip(points_per_building, roof_types):
            width = float(rng.uniform(8.0, 22.0))
            depth = float(rng.uniform(8.0, 20.0))
            floor_count = int(rng.integers(floor_min, floor_max + 1))
            floor_height = float(rng.uniform(2.7, 3.5))
            if building_random_yaw:
                yaw = float(rng.uniform(0.0, 2.0 * np.pi))
            else:
                yaw = float(rng.choice([0.0, 0.5 * np.pi]))

            cos_yaw = abs(float(np.cos(yaw)))
            sin_yaw = abs(float(np.sin(yaw)))
            bbox_w = cos_yaw * width + sin_yaw * depth
            bbox_d = sin_yaw * width + cos_yaw * depth

            # Keep buildings away from detached artificial surfaces and from each other.
            forbidden = base_artificial_rects + [
                (bx, by, bsx + 6.0, bsy + 6.0) for bx, by, bsx, bsy in building_rects
            ]
            cx, cy = _sample_single_xy(
                rng,
                area_size=area_size,
                forbidden_rects=forbidden,
                margin=0.2,
            )
            base_z = float(terrain_fn(np.array([cx]), np.array([cy]))[0] + 0.08)

            building_rects.append((cx, cy, bbox_w, bbox_d))
            cloud_parts.append(
                _generate_building_points(
                    rng=rng,
                    n_points=int(n_pts),
                    cx=cx,
                    cy=cy,
                    width=width,
                    depth=depth,
                    floor_count=floor_count,
                    floor_height=floor_height,
                    roof_type=roof_type,
                    yaw=yaw,
                    base_z=base_z,
                )
            )

    for surface_type in ("sidewalk", "building_front_area"):
        for index in range(surface_type_counts[surface_type]):
            if building_rects:
                anchor = building_rects[int(rng.integers(0, len(building_rects)))]
                rect = _sample_surface_rect_along_building(
                    rng=rng,
                    area_size=area_size,
                    building_rect=anchor,
                    surface_type=surface_type,
                    existing_zones=artificial_zones,
                )
            else:
                rect = _sample_detached_surface_rect(
                    rng=rng,
                    area_size=area_size,
                    surface_type=surface_type,
                    existing_zones=artificial_zones,
                )
            artificial_zones.append(
                _make_surface_zone(
                    rng=rng,
                    terrain_fn=terrain_fn,
                    name=f"{surface_type}_{index + 1}",
                    surface_type=surface_type,
                    rect=rect,
                )
            )

    artificial_rects: List[Rect] = [zone["rect"] for zone in artificial_zones]  # type: ignore[index]

    _emit_progress(progress_callback, 0.32, "Generating artificial surfaces")

    # ------------------------------------------------------------------
    # Class 0: artificial surfaces
    # ------------------------------------------------------------------
    n_artificial = int(points_per_class.get(ARTIFICIAL_SURFACE_CLASS_ID, 0))
    if n_artificial > 0 and artificial_zones:
        areas = [max(1e-6, rect[2] * rect[3]) for rect in artificial_rects]
        zone_counts = _split_count_by_weights(n_artificial, areas)
        zones_points: List[np.ndarray] = []
        for zone, zone_n in zip(artificial_zones, zone_counts):
            if zone_n <= 0:
                continue
            rect = zone["rect"]  # type: ignore[index]
            surface_type = str(zone.get("surface_type", zone.get("kind", ""))).strip().lower()
            margin = 0.30 if surface_type == "sidewalk" else 0.35
            x, y = _sample_inside_rect(rng, zone_n, rect, margin=margin)
            z0 = float(zone["z0"])  # type: ignore[index]
            roughness = 0.006 if surface_type in {"sidewalk", "platform"} else 0.010
            z = np.full(zone_n, z0) + rng.normal(0.0, roughness, size=zone_n)
            labels = np.full(zone_n, ARTIFICIAL_SURFACE_CLASS_ID, dtype=np.int32)
            zones_points.append(np.column_stack((x, y, z, labels)))
        if zones_points:
            cloud_parts.append(np.vstack(zones_points))

    _emit_progress(progress_callback, 0.48, "Generating trees")

    # ------------------------------------------------------------------
    # Class 2: high vegetation (trees)
    # ------------------------------------------------------------------
    n_tree_points = int(points_per_class.get(HIGH_VEGETATION_CLASS_ID, 0))
    if n_tree_points > 0:
        n_trees_eff = max(1, min(num_trees, n_tree_points))
        per_tree = _split_count_random(n_tree_points, n_trees_eff, rng=rng, min_per_chunk=8)
        if tree_crown_type_ratios is None:
            random_type_weights = rng.dirichlet(
                np.array([2.6, 1.8, 2.2, 1.3, 1.5, 1.6], dtype=np.float64)
            )
            tree_crown_type_ratios_eff = {
                crown_type: float(random_type_weights[idx])
                for idx, crown_type in enumerate(TREE_CROWN_TYPES)
            }
        else:
            tree_crown_type_ratios_eff = {
                crown_type: float(tree_crown_type_ratios.get(crown_type, 0.0))
                for crown_type in TREE_CROWN_TYPES
            }
            if np.isclose(sum(tree_crown_type_ratios_eff.values()), 0.0):
                tree_crown_type_ratios_eff = dict(DEFAULT_TREE_CROWN_TYPE_RATIOS)
        per_crown_type = _split_count_by_weights(
            n_trees_eff,
            [tree_crown_type_ratios_eff[crown_type] for crown_type in TREE_CROWN_TYPES],
        )
        crown_types: List[str] = []
        for crown_type, type_count in zip(TREE_CROWN_TYPES, per_crown_type):
            crown_types.extend([crown_type] * int(type_count))
        if len(crown_types) < n_trees_eff:
            crown_types.extend(["spherical"] * (n_trees_eff - len(crown_types)))
        crown_types = crown_types[:n_trees_eff]
        rng.shuffle(crown_types)

        tree_forbidden = artificial_rects + [
            (cx, cy, sx + 3.5, sy + 3.5) for cx, cy, sx, sy in building_rects
        ]
        tree_rects: List[Rect] = []
        for n_pts, crown_type in zip(per_tree, crown_types):
            crown_diameter, crown_bottom_h, crown_top_h = _sample_tree_crown_dimensions(
                rng=rng,
                crown_type=crown_type,
                random_size=bool(random_tree_crown_size),
                tree_max_crown_diameter=float(tree_max_crown_diameter),
                tree_max_crown_top_height=float(tree_max_crown_top_height),
                tree_min_crown_bottom_height=float(tree_min_crown_bottom_height),
            )
            trunk_height = max(0.8, crown_bottom_h * float(rng.uniform(0.88, 1.05)))
            trunk_radius = float(
                np.clip(
                    (0.055 * crown_diameter + 0.012 * crown_top_h)
                    * rng.uniform(0.76, 1.28),
                    0.10,
                    0.65,
                )
            )

            forbidden_rects = tree_forbidden + tree_rects
            cx, cy = _sample_single_xy(
                rng,
                area_size=area_size,
                forbidden_rects=forbidden_rects,
                margin=0.15,
            )
            tree_rects.append((cx, cy, 1.12 * crown_diameter, 1.12 * crown_diameter))
            ground_z = float(terrain_fn(np.array([cx]), np.array([cy]))[0])
            cloud_parts.append(
                _generate_tree_points(
                    rng=rng,
                    n_points=int(n_pts),
                    cx=cx,
                    cy=cy,
                    ground_z=ground_z,
                    trunk_radius=trunk_radius,
                    trunk_height=trunk_height,
                    crown_type=crown_type,
                    crown_diameter=crown_diameter,
                    crown_bottom_height=crown_bottom_h,
                    crown_top_height=crown_top_h,
                )
            )

    _emit_progress(progress_callback, 0.62, "Generating structures")

    # ------------------------------------------------------------------
    # Class 5: structures
    # ------------------------------------------------------------------
    n_structure_points = int(points_per_class.get(STRUCTURES_CLASS_ID, 0))
    if n_structure_points > 0:
        n_struct_eff = max(1, min(num_structures, n_structure_points))
        per_struct = _split_count_random(
            n_structure_points, n_struct_eff, rng=rng, min_per_chunk=8
        )
        struct_forbidden = [
            (cx, cy, sx + 2.0, sy + 2.0) for cx, cy, sx, sy in building_rects
        ]
        structure_weights = (
            structure_type_ratios
            if structure_type_ratios is not None
            else _sample_scene_structure_type_ratios(rng)
        )
        structure_types = rng.choice(
            STRUCTURE_TYPES,
            size=n_struct_eff,
            p=[structure_weights[structure_type] for structure_type in STRUCTURE_TYPES],
        )
        if (
            n_struct_eff >= 4
            and np.all(structure_types != "footbridge")
            and rng.random() < 0.45
        ):
            structure_types[int(rng.integers(0, n_struct_eff))] = "footbridge"
        for n_pts, structure_type in zip(per_struct, structure_types):
            cx, cy = _sample_single_xy(
                rng, area_size=area_size, forbidden_rects=struct_forbidden, margin=0.2
            )
            cloud_parts.append(
                _generate_structure_object_points(
                    rng=rng,
                    terrain_fn=terrain_fn,
                    n_points=int(n_pts),
                    x_center=cx,
                    y_center=cy,
                    structure_type=str(structure_type),
                    area_size=area_size,
                )
            )

    _emit_progress(progress_callback, 0.74, "Generating vehicles")

    # ------------------------------------------------------------------
    # Class 7: vehicles (car / truck / bus on drivable artificial surfaces)
    # ------------------------------------------------------------------
    n_vehicle_points = int(points_per_class.get(VEHICLES_CLASS_ID, 0))
    vehicle_zones = [zone for zone in artificial_zones if bool(zone.get("vehicle_allowed", False))]
    if not vehicle_zones:
        vehicle_zones = list(artificial_zones)
    if n_vehicle_points > 0 and vehicle_zones:
        vehicle_type_ratios = vehicle_type_ratios or dict(DEFAULT_VEHICLE_TYPE_RATIOS)
        n_vehicle_eff = max(1, min(num_vehicles, n_vehicle_points))
        per_vehicle = _split_count_evenly(n_vehicle_points, n_vehicle_eff)
        per_type = _split_count_by_weights(
            n_vehicle_eff,
            [vehicle_type_ratios[vehicle_type] for vehicle_type in VEHICLE_TYPES],
        )
        vehicle_types: List[str] = []
        for vehicle_type, type_count in zip(VEHICLE_TYPES, per_type):
            vehicle_types.extend([vehicle_type] * int(type_count))
        if len(vehicle_types) < n_vehicle_eff:
            vehicle_types.extend(["car"] * (n_vehicle_eff - len(vehicle_types)))
        vehicle_types = vehicle_types[:n_vehicle_eff]
        rng.shuffle(vehicle_types)

        zone_prob = np.array(
            [
                zone["rect"][2] * zone["rect"][3]  # type: ignore[index]
                for zone in vehicle_zones
            ],
            dtype=np.float64,
        )
        zone_prob /= zone_prob.sum()
        vehicle_rects: List[Rect] = []
        for n_pts, vehicle_type in zip(per_vehicle, vehicle_types):
            zone_idx = int(rng.choice(len(vehicle_zones), p=zone_prob))
            zone = vehicle_zones[zone_idx]
            rect = zone["rect"]  # type: ignore[index]

            length, width, height = _sample_vehicle_dimensions(rng, vehicle_type=vehicle_type)
            margin = 0.6 * max(length, width)
            cx_f, cy_f = float(rect[0]), float(rect[1])
            candidate_rect: Rect | None = None
            for _ in range(24):
                cx, cy = _sample_inside_rect(rng, 1, rect, margin=margin)
                cx_f, cy_f = float(cx[0]), float(cy[0])
                candidate_rect = (cx_f, cy_f, length * 1.14, width * 1.24)
                if all(_rect_overlap_ratio(candidate_rect, r) < 0.14 for r in vehicle_rects):
                    break

            sx, sy = rect[2], rect[3]
            base_angle = 0.0 if sx >= sy else np.pi * 0.5
            yaw = float(base_angle + rng.normal(0.0, 0.16))
            base_z = float(zone["z0"]) + 0.06  # type: ignore[index]
            if candidate_rect is None:
                candidate_rect = (cx_f, cy_f, length * 1.14, width * 1.24)
            vehicle_rects.append(candidate_rect)

            cloud_parts.append(
                _generate_vehicle_points(
                    rng=rng,
                    n_points=int(n_pts),
                    x_center=cx_f,
                    y_center=cy_f,
                    base_z=base_z,
                    length=length,
                    width=width,
                    height=height,
                    yaw=yaw,
                    vehicle_type=vehicle_type,
                )
            )

    _emit_progress(progress_callback, 0.84, "Generating low vegetation")

    # ------------------------------------------------------------------
    # Class 3: low vegetation (shrubs + grassy patches)
    # ------------------------------------------------------------------
    n_low_veg = int(points_per_class.get(LOW_VEGETATION_CLASS_ID, 0))
    if n_low_veg > 0:
        low_veg_forbidden = artificial_rects + building_rects
        shrub_count_eff, grass_patch_count_eff = _resolve_low_veg_counts(
            n_low_veg_points=n_low_veg,
            area_size=area_size,
            rng=rng,
            shrub_count=shrub_count,
            grass_patch_count=grass_patch_count,
        )

        if shrub_count_eff > 0 and grass_patch_count_eff > 0:
            shrub_points_total, grass_points_total = _split_count_by_weights(
                n_low_veg,
                [0.58, 0.42],
            )
        elif shrub_count_eff > 0:
            shrub_points_total = n_low_veg
            grass_points_total = 0
        else:
            shrub_points_total = 0
            grass_points_total = n_low_veg

        if shrub_points_total > 0:
            per_shrub = _split_count_random(
                int(shrub_points_total),
                int(shrub_count_eff),
                rng=rng,
                min_per_chunk=2,
            )
            shrub_parts: List[np.ndarray] = []
            for n_pts in per_shrub:
                cx, cy = _sample_single_xy(
                    rng=rng,
                    area_size=area_size,
                    forbidden_rects=low_veg_forbidden,
                    margin=0.1,
                )
                shrub_parts.append(
                    _generate_shrub_points(
                        rng=rng,
                        terrain_fn=terrain_fn,
                        n_points=int(n_pts),
                        center_x=float(cx),
                        center_y=float(cy),
                        random_size=bool(random_shrub_size),
                        shrub_max_diameter=float(shrub_max_diameter),
                        shrub_max_top_height=float(shrub_max_top_height),
                        shrub_min_bottom_height=float(shrub_min_bottom_height),
                    )
                )
            if shrub_parts:
                cloud_parts.append(np.vstack(shrub_parts))

        if grass_points_total > 0:
            per_grass = _split_count_random(
                int(grass_points_total),
                int(grass_patch_count_eff),
                rng=rng,
                min_per_chunk=2,
            )
            grass_parts: List[np.ndarray] = []
            for n_pts in per_grass:
                cx, cy = _sample_single_xy(
                    rng=rng,
                    area_size=area_size,
                    forbidden_rects=low_veg_forbidden,
                    margin=0.1,
                )
                grass_parts.append(
                    _generate_grass_patch_points(
                        rng=rng,
                        terrain_fn=terrain_fn,
                        n_points=int(n_pts),
                        center_x=float(cx),
                        center_y=float(cy),
                        random_size=bool(random_grass_patch_size),
                        grass_patch_max_size_x=float(grass_patch_max_size_x),
                        grass_patch_max_size_y=float(grass_patch_max_size_y),
                        grass_max_height=float(grass_max_height),
                    )
                )
            if grass_parts:
                cloud_parts.append(np.vstack(grass_parts))

    _emit_progress(progress_callback, 0.94, "Generating artifacts")

    # ------------------------------------------------------------------
    # Class 6: artifacts (measurement defects / acquisition errors)
    # ------------------------------------------------------------------
    n_artifacts = int(points_per_class.get(ARTIFACTS_CLASS_ID, 0))
    if n_artifacts > 0:
        base_cloud = np.vstack(cloud_parts) if cloud_parts else np.empty((0, 4), dtype=np.float64)
        artifact_points = _generate_artifact_points(
            rng=rng,
            terrain_fn=terrain_fn,
            area_size=area_size,
            n_points=n_artifacts,
            artificial_zones=artificial_zones,
            building_rects=building_rects,
            base_cloud=base_cloud,
            num_artifact_clusters=int(num_artifact_clusters),
            artifacts_enabled=bool(artifacts_enabled),
            artifact_global_intensity=float(artifact_global_intensity),
            artifact_type_settings=artifact_type_settings,
        )
        if artifact_points.size > 0:
            cloud_parts.append(artifact_points)

    _emit_progress(progress_callback, 1.0, "Object generation complete")
    if not cloud_parts:
        return np.empty((0, 4), dtype=np.float64), artificial_zones
    return np.vstack(cloud_parts), artificial_zones



def _artifact_empty_cloud() -> np.ndarray:
    return np.empty((0, 4), dtype=np.float64)


def _clip_xy_to_area(
    x: np.ndarray,
    y: np.ndarray,
    area_size: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    half_w = 0.5 * float(area_size[0])
    half_h = 0.5 * float(area_size[1])
    return np.clip(x, -half_w, half_w), np.clip(y, -half_h, half_h)


def _artifact_cloud_from_xyz(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    area_size: Tuple[float, float],
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    z_arr = np.asarray(z, dtype=np.float64).reshape(-1)
    if x_arr.size == 0 or y_arr.size == 0 or z_arr.size == 0:
        return _artifact_empty_cloud()
    count = min(x_arr.size, y_arr.size, z_arr.size)
    x_arr = x_arr[:count]
    y_arr = y_arr[:count]
    z_arr = z_arr[:count]
    x_arr, y_arr = _clip_xy_to_area(x_arr, y_arr, area_size)
    labels = np.full(count, ARTIFACTS_CLASS_ID, dtype=np.int32)
    return np.column_stack((x_arr, y_arr, z_arr, labels))


def _artifact_effective_strength(global_intensity: float, local_intensity: float) -> float:
    return float(np.clip(float(global_intensity) * float(local_intensity), 0.0, 1.0))


def _sample_artifact_anchors(
    rng: np.random.Generator,
    base_cloud: np.ndarray,
    n_points: int,
    preferred_labels: Sequence[int] = (),
) -> np.ndarray:
    cloud = np.asarray(base_cloud, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[0] == 0 or cloud.shape[1] < 3 or n_points <= 0:
        return np.empty((0, 3), dtype=np.float64)

    candidates = cloud
    if preferred_labels and cloud.shape[1] >= 4:
        labels = np.rint(cloud[:, 3]).astype(np.int32)
        mask = np.isin(labels, np.asarray(preferred_labels, dtype=np.int32))
        if np.any(mask):
            candidates = cloud[mask]

    indices = rng.integers(0, candidates.shape[0], size=int(n_points))
    return np.asarray(candidates[indices, :3], dtype=np.float64)


def _sample_points_near_rect_boundaries(
    rng: np.random.Generator,
    n_points: int,
    rects: Sequence[Rect],
    width: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if n_points <= 0 or not rects:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    rect_array = np.asarray(rects, dtype=np.float64)
    perimeters = 2.0 * (rect_array[:, 2] + rect_array[:, 3])
    perimeters = np.maximum(perimeters, 1e-6)
    probabilities = perimeters / perimeters.sum()
    rect_indices = rng.choice(rect_array.shape[0], size=int(n_points), p=probabilities)
    chosen_rects = rect_array[rect_indices]
    cx = chosen_rects[:, 0]
    cy = chosen_rects[:, 1]
    sx = chosen_rects[:, 2]
    sy = chosen_rects[:, 3]

    side = rng.integers(0, 4, size=int(n_points))
    tangent = rng.uniform(-0.5, 0.5, size=int(n_points))
    tangent_jitter = rng.normal(0.0, max(0.01, float(width)) * 0.15, size=int(n_points))
    normal = rng.normal(0.0, max(0.01, float(width)) * 0.55, size=int(n_points))

    x = np.empty(int(n_points), dtype=np.float64)
    y = np.empty(int(n_points), dtype=np.float64)

    left_mask = side == 0
    right_mask = side == 1
    bottom_mask = side == 2
    top_mask = side == 3

    x[left_mask] = cx[left_mask] - 0.5 * sx[left_mask] + normal[left_mask]
    y[left_mask] = cy[left_mask] + tangent[left_mask] * sy[left_mask] + tangent_jitter[left_mask]

    x[right_mask] = cx[right_mask] + 0.5 * sx[right_mask] + normal[right_mask]
    y[right_mask] = cy[right_mask] + tangent[right_mask] * sy[right_mask] + tangent_jitter[right_mask]

    x[bottom_mask] = cx[bottom_mask] + tangent[bottom_mask] * sx[bottom_mask] + tangent_jitter[bottom_mask]
    y[bottom_mask] = cy[bottom_mask] - 0.5 * sy[bottom_mask] + normal[bottom_mask]

    x[top_mask] = cx[top_mask] + tangent[top_mask] * sx[top_mask] + tangent_jitter[top_mask]
    y[top_mask] = cy[top_mask] + 0.5 * sy[top_mask] + normal[top_mask]
    return x, y


def _generate_random_outlier_artifacts(
    rng: np.random.Generator,
    terrain_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    area_size: Tuple[float, float],
    n_points: int,
    settings: Mapping[str, float | bool],
    global_intensity: float,
    num_artifact_clusters: int,
    **_: Any,
) -> np.ndarray:
    if n_points <= 0:
        return _artifact_empty_cloud()

    strength = 0.15 + 0.85 * _artifact_effective_strength(
        global_intensity,
        float(settings["intensity"]),
    )
    spread = float(settings["spread"]) * (0.25 + 1.10 * strength)
    n_clusters = max(1, min(int(n_points), int(round(num_artifact_clusters * (0.45 + 1.55 * strength)))))
    cluster_sizes = _split_count_evenly(int(n_points), int(n_clusters))

    parts: List[np.ndarray] = []
    for cluster_size in cluster_sizes:
        if int(cluster_size) <= 0:
            continue
        cx, cy = _sample_single_xy(rng, area_size=area_size)
        count = int(cluster_size)
        x = cx + rng.normal(0.0, max(0.01, spread), size=count)
        y = cy + rng.normal(0.0, max(0.01, spread), size=count)
        x, y = _clip_xy_to_area(x, y, area_size)
        z = terrain_fn(x, y) + rng.uniform(0.20, 0.45 + 2.4 * strength + 0.55 * spread, size=count)
        z += rng.normal(0.0, 0.03 + 0.16 * strength, size=count)
        parts.append(_artifact_cloud_from_xyz(x, y, z, area_size))

    return np.vstack(parts) if parts else _artifact_empty_cloud()


def _generate_surface_noise_artifacts(
    rng: np.random.Generator,
    terrain_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    area_size: Tuple[float, float],
    n_points: int,
    settings: Mapping[str, float | bool],
    global_intensity: float,
    base_cloud: np.ndarray,
    **_: Any,
) -> np.ndarray:
    if n_points <= 0:
        return _artifact_empty_cloud()

    strength = 0.15 + 0.85 * _artifact_effective_strength(
        global_intensity,
        float(settings["intensity"]),
    )
    thickness = float(settings["thickness"])
    xy_sigma = max(0.005, thickness * (0.18 + 0.95 * strength))
    z_sigma = max(0.008, thickness * (0.35 + 1.35 * strength))
    anchors = _sample_artifact_anchors(
        rng,
        base_cloud,
        int(n_points),
        preferred_labels=(ARTIFICIAL_SURFACE_CLASS_ID, NATURAL_SURFACE_CLASS_ID, LOW_VEGETATION_CLASS_ID, BUILDINGS_CLASS_ID, STRUCTURES_CLASS_ID),
    )
    if anchors.size == 0:
        x, y = _sample_xy(rng, int(n_points), area_size=area_size)
        z = terrain_fn(x, y) + rng.normal(0.0, z_sigma, size=int(n_points))
        return _artifact_cloud_from_xyz(x, y, z, area_size)

    x = anchors[:, 0] + rng.normal(0.0, xy_sigma, size=int(n_points))
    y = anchors[:, 1] + rng.normal(0.0, xy_sigma, size=int(n_points))
    x, y = _clip_xy_to_area(x, y, area_size)
    z = anchors[:, 2] + rng.normal(0.0, z_sigma, size=int(n_points))
    return _artifact_cloud_from_xyz(x, y, z, area_size)


def _generate_hanging_point_artifacts(
    rng: np.random.Generator,
    terrain_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    area_size: Tuple[float, float],
    n_points: int,
    settings: Mapping[str, float | bool],
    global_intensity: float,
    base_cloud: np.ndarray,
    **_: Any,
) -> np.ndarray:
    if n_points <= 0:
        return _artifact_empty_cloud()

    strength = 0.15 + 0.85 * _artifact_effective_strength(
        global_intensity,
        float(settings["intensity"]),
    )
    max_height = max(0.10, float(settings["height"]) * (0.35 + 1.35 * strength))
    xy_sigma = 0.04 + 0.18 * strength
    anchors = _sample_artifact_anchors(
        rng,
        base_cloud,
        int(n_points),
        preferred_labels=(HIGH_VEGETATION_CLASS_ID, BUILDINGS_CLASS_ID, STRUCTURES_CLASS_ID, VEHICLES_CLASS_ID),
    )
    if anchors.size == 0:
        x, y = _sample_xy(rng, int(n_points), area_size=area_size)
        base_z = terrain_fn(x, y)
    else:
        x = anchors[:, 0] + rng.normal(0.0, xy_sigma, size=int(n_points))
        y = anchors[:, 1] + rng.normal(0.0, xy_sigma, size=int(n_points))
        x, y = _clip_xy_to_area(x, y, area_size)
        base_z = np.maximum(terrain_fn(x, y), anchors[:, 2])
    z = base_z + rng.uniform(0.25, max_height, size=int(n_points))
    return _artifact_cloud_from_xyz(x, y, z, area_size)


def _generate_ghost_double_point_artifacts(
    rng: np.random.Generator,
    terrain_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    area_size: Tuple[float, float],
    n_points: int,
    settings: Mapping[str, float | bool],
    global_intensity: float,
    base_cloud: np.ndarray,
    **kwargs: Any,
) -> np.ndarray:
    if n_points <= 0:
        return _artifact_empty_cloud()

    anchors = _sample_artifact_anchors(rng, base_cloud, int(n_points))
    if anchors.size == 0:
        return _generate_random_outlier_artifacts(
            rng=rng,
            terrain_fn=terrain_fn,
            area_size=area_size,
            n_points=int(n_points),
            settings=settings,
            global_intensity=global_intensity,
            num_artifact_clusters=int(kwargs.get("num_artifact_clusters", 1)),
        )

    strength = 0.15 + 0.85 * _artifact_effective_strength(
        global_intensity,
        float(settings["intensity"]),
    )
    offset = max(0.01, float(settings["offset"]) * (0.35 + 1.35 * strength))
    angle = rng.uniform(0.0, 2.0 * np.pi, size=int(n_points))
    x = anchors[:, 0] + np.cos(angle) * offset + rng.normal(0.0, 0.20 * offset, size=int(n_points))
    y = anchors[:, 1] + np.sin(angle) * offset + rng.normal(0.0, 0.20 * offset, size=int(n_points))
    x, y = _clip_xy_to_area(x, y, area_size)
    z = anchors[:, 2] + rng.normal(0.0, max(0.01, offset * (0.25 + 0.85 * strength)), size=int(n_points))
    return _artifact_cloud_from_xyz(x, y, z, area_size)


def _generate_blurred_boundary_artifacts(
    rng: np.random.Generator,
    terrain_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    area_size: Tuple[float, float],
    n_points: int,
    settings: Mapping[str, float | bool],
    global_intensity: float,
    artificial_zones: Sequence[Dict[str, object]],
    building_rects: Sequence[Rect],
    base_cloud: np.ndarray,
    **_: Any,
) -> np.ndarray:
    if n_points <= 0:
        return _artifact_empty_cloud()

    rects: List[Rect] = [zone["rect"] for zone in artificial_zones] + list(building_rects)
    if not rects:
        return _generate_surface_noise_artifacts(
            rng=rng,
            terrain_fn=terrain_fn,
            area_size=area_size,
            n_points=int(n_points),
            settings={**settings, "thickness": float(settings["width"])},
            global_intensity=global_intensity,
            base_cloud=base_cloud,
        )

    strength = 0.15 + 0.85 * _artifact_effective_strength(
        global_intensity,
        float(settings["intensity"]),
    )
    boundary_width = max(0.02, float(settings["width"]) * (0.30 + 1.20 * strength))
    x, y = _sample_points_near_rect_boundaries(rng, int(n_points), rects, boundary_width)
    x, y = _clip_xy_to_area(x, y, area_size)
    z = terrain_fn(x, y) + rng.normal(0.03, max(0.01, boundary_width * (0.12 + 0.50 * strength)), size=int(n_points))
    return _artifact_cloud_from_xyz(x, y, z, area_size)


def _generate_false_reflection_artifacts(
    rng: np.random.Generator,
    terrain_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    area_size: Tuple[float, float],
    n_points: int,
    settings: Mapping[str, float | bool],
    global_intensity: float,
    base_cloud: np.ndarray,
    **kwargs: Any,
) -> np.ndarray:
    if n_points <= 0:
        return _artifact_empty_cloud()

    anchors = _sample_artifact_anchors(
        rng,
        base_cloud,
        int(n_points),
        preferred_labels=(ARTIFICIAL_SURFACE_CLASS_ID, BUILDINGS_CLASS_ID, STRUCTURES_CLASS_ID, VEHICLES_CLASS_ID),
    )
    if anchors.size == 0:
        return _generate_hanging_point_artifacts(
            rng=rng,
            terrain_fn=terrain_fn,
            area_size=area_size,
            n_points=int(n_points),
            settings=settings,
            global_intensity=global_intensity,
            base_cloud=base_cloud,
        )

    strength = 0.15 + 0.85 * _artifact_effective_strength(
        global_intensity,
        float(settings["intensity"]),
    )
    max_height = max(0.10, float(settings["height"]) * (0.25 + 1.50 * strength))
    lateral = 0.05 + 0.22 * strength * max(0.5, max_height)
    angle = rng.uniform(0.0, 2.0 * np.pi, size=int(n_points))
    x = anchors[:, 0] + np.cos(angle) * lateral + rng.normal(0.0, 0.25 * lateral, size=int(n_points))
    y = anchors[:, 1] + np.sin(angle) * lateral + rng.normal(0.0, 0.25 * lateral, size=int(n_points))
    x, y = _clip_xy_to_area(x, y, area_size)
    base_z = np.maximum(terrain_fn(x, y), anchors[:, 2])
    z = base_z + rng.uniform(0.12, max_height, size=int(n_points))
    return _artifact_cloud_from_xyz(x, y, z, area_size)


_ARTIFACT_GENERATORS: Dict[str, Callable[..., np.ndarray]] = {
    "random_outliers": _generate_random_outlier_artifacts,
    "surface_noise": _generate_surface_noise_artifacts,
    "hanging_points": _generate_hanging_point_artifacts,
    "ghost_double_points": _generate_ghost_double_point_artifacts,
    "blurred_boundaries": _generate_blurred_boundary_artifacts,
    "false_reflections": _generate_false_reflection_artifacts,
}


def _generate_artifact_points(
    rng: np.random.Generator,
    terrain_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    area_size: Tuple[float, float],
    n_points: int,
    artificial_zones: Sequence[Dict[str, object]],
    building_rects: Sequence[Rect],
    base_cloud: np.ndarray,
    num_artifact_clusters: int,
    artifacts_enabled: bool,
    artifact_global_intensity: float,
    artifact_type_settings: Mapping[str, Mapping[str, Any]] | None,
) -> np.ndarray:
    if not artifacts_enabled or n_points <= 0:
        return _artifact_empty_cloud()

    settings = _normalize_artifact_type_settings_payload(
        artifact_type_settings,
        None,
        defaults=DEFAULT_ARTIFACT_TYPE_SETTINGS,
    )
    enabled_types = [
        artifact_key
        for artifact_key in ARTIFACT_TYPES
        if bool(settings[artifact_key]["enabled"]) and float(settings[artifact_key]["amount"]) > 0.0
    ]
    if not enabled_types:
        return _artifact_empty_cloud()

    counts = _split_count_by_weights(
        int(n_points),
        [float(settings[artifact_key]["amount"]) for artifact_key in enabled_types],
    )
    parts: List[np.ndarray] = []
    for artifact_key, count in zip(enabled_types, counts):
        if int(count) <= 0:
            continue
        part = _ARTIFACT_GENERATORS[artifact_key](
            rng=rng,
            terrain_fn=terrain_fn,
            area_size=area_size,
            n_points=int(count),
            settings=settings[artifact_key],
            global_intensity=float(artifact_global_intensity),
            artificial_zones=artificial_zones,
            building_rects=building_rects,
            base_cloud=base_cloud,
            num_artifact_clusters=int(num_artifact_clusters),
        )
        if part.size > 0:
            parts.append(part)

    if not parts:
        return _artifact_empty_cloud()

    artifact_cloud = np.vstack(parts)
    if artifact_cloud.shape[0] < int(n_points):
        missing = int(n_points) - int(artifact_cloud.shape[0])
        fallback_part = _generate_random_outlier_artifacts(
            rng=rng,
            terrain_fn=terrain_fn,
            area_size=area_size,
            n_points=missing,
            settings=settings.get("random_outliers", DEFAULT_ARTIFACT_TYPE_SETTINGS["random_outliers"]),
            global_intensity=float(artifact_global_intensity),
            num_artifact_clusters=int(num_artifact_clusters),
        )
        if fallback_part.size > 0:
            artifact_cloud = np.vstack((artifact_cloud, fallback_part))
    if artifact_cloud.shape[0] > int(n_points):
        rng.shuffle(artifact_cloud, axis=0)
        artifact_cloud = artifact_cloud[: int(n_points)]
    elif artifact_cloud.shape[0] < int(n_points):
        missing = int(n_points) - int(artifact_cloud.shape[0])
        x, y = _sample_xy(rng, missing, area_size=area_size)
        z = terrain_fn(x, y) + rng.uniform(0.12, 0.85, size=missing)
        artifact_cloud = np.vstack((artifact_cloud, _artifact_cloud_from_xyz(x, y, z, area_size)))

    rng.shuffle(artifact_cloud, axis=0)
    return artifact_cloud[: int(n_points)]

def visualize_point_cloud(points: np.ndarray, labels: np.ndarray) -> None:
    """
    Interactive 3D scatter visualization with color-by-class and legend.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    xyz = np.asarray(points, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("`points` must be an array of shape (N, 3).")
    if labels_arr.shape[0] != xyz.shape[0]:
        raise ValueError("`labels` size must match number of points.")

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    legend_handles: List[Line2D] = []
    for class_id in sorted(CLASS_NAMES):
        mask = labels_arr == class_id
        count = int(mask.sum())
        if count == 0:
            continue
        color = CLASS_COLORS[class_id]
        ax.scatter(
            xyz[mask, 0],
            xyz[mask, 1],
            xyz[mask, 2],
            c=[color],
            s=1.2,
            alpha=0.95,
            linewidths=0.0,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=f"{class_id}: {CLASS_NAMES[class_id]} ({count})",
                markerfacecolor=color,
                markersize=7,
            )
        )

    ax.set_title("Synthetic Labeled Point Cloud")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()


def export_to_csv(point_cloud: np.ndarray, output_path: Path) -> None:
    """Export (N,4) cloud to CSV."""
    cloud = np.array(point_cloud, copy=True)
    cloud[:, 3] = cloud[:, 3].astype(np.int32)
    fmt = ["%.6f", "%.6f", "%.6f", "%d"]
    np.savetxt(
        output_path,
        cloud,
        delimiter=",",
        header="x,y,z,label",
        comments="",
        fmt=fmt,
    )


def export_to_ply(point_cloud: np.ndarray, output_path: Path) -> None:
    """
    Export to ASCII PLY with per-point color and label.
    Useful for CloudCompare / MeshLab import.
    """
    xyz = point_cloud[:, :3]
    labels = point_cloud[:, 3].astype(np.int32)
    rgb = np.array([CLASS_COLORS[int(l)] for l in labels], dtype=np.float64)
    rgb = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)

    with output_path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        for class_id in CLASS_IDS:
            f.write(f"comment {PLY_LABEL_COMMENT_PREFIX} {class_id} {CLASS_NAMES[class_id]}\n")
        generation_order = " ".join(str(class_id) for class_id in CLASS_GENERATION_ORDER)
        f.write(f"comment {PLY_GENERATION_ORDER_COMMENT_PREFIX} {generation_order}\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property uchar label\n")
        f.write("end_header\n")
        for p, c, l in zip(xyz, rgb, labels):
            f.write(
                f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])} {int(l)}\n"
            )


def _print_stats(labels: Iterable[int]) -> None:
    """Print total count and per-class statistics."""
    labels_arr = np.asarray(list(labels), dtype=np.int32)
    print(f"Total points: {labels_arr.size}")
    for class_id in sorted(CLASS_NAMES):
        count = int(np.sum(labels_arr == class_id))
        print(f"Class {class_id} ({CLASS_NAMES[class_id]}): {count}")


def _generate_object_counts(
    area_size: Tuple[float, float],
    rng: np.random.Generator,
    randomize_counts: bool,
) -> Dict[str, int]:
    """
    Generate object counts based on scene area.
    If randomize_counts is True, values are sampled from area-scaled ranges.
    """
    width_m, length_m = area_size
    area_m2 = width_m * length_m
    area_ha = area_m2 / 10_000.0

    # Expected object density per hectare.
    densities_per_ha = {
        "num_artificial_surfaces": 1.8,
        "num_trees": 14.0,
        "num_buildings": 2.6,
        "num_structures": 1.6,
        "num_vehicles": 4.8,
        "num_artifact_clusters": 7.5,
    }
    min_counts = {
        "num_artificial_surfaces": 5,
        "num_trees": 8,
        "num_buildings": 3,
        "num_structures": 2,
        "num_vehicles": 3,
        "num_artifact_clusters": 8,
    }

    counts: Dict[str, int] = {}
    for key, density in densities_per_ha.items():
        expected = max(float(min_counts[key]), density * area_ha)
        if randomize_counts:
            low = max(min_counts[key], int(np.floor(expected * 0.65)))
            high = max(low + 1, int(np.ceil(expected * 1.45 + 2.0)))
            counts[key] = int(rng.integers(low, high + 1))
        else:
            counts[key] = max(min_counts[key], int(round(expected)))
    return counts


def _print_scene_params(
    area_size: Tuple[float, float],
    seed: int,
    terrain_relief: float,
    randomize_object_counts: bool,
    object_counts: Dict[str, int],
    class_ratios: Dict[int, float],
    artificial_surface_type_ratios: Dict[str, float] | None,
    artificial_surface_count_overridden: bool,
    artificial_surface_type_distribution_overridden: bool,
    building_roof_type_ratios: Dict[str, float] | None,
    building_count_overridden: bool,
    building_roof_type_distribution_overridden: bool,
    building_floor_min: int,
    building_floor_max: int,
    building_random_yaw: bool,
    structure_type_ratios: Dict[str, float] | None,
    structure_count_overridden: bool,
    structure_type_distribution_overridden: bool,
    vehicle_type_ratios: Dict[str, float],
    tree_crown_type_ratios: Dict[str, float] | None,
    tree_count_overridden: bool,
    vehicle_count_overridden: bool,
    tree_crown_type_distribution_overridden: bool,
    random_tree_crown_size: bool,
    tree_max_crown_diameter: float,
    tree_max_crown_top_height: float,
    tree_min_crown_bottom_height: float,
    shrub_count_overridden: bool,
    grass_patch_count_overridden: bool,
    random_shrub_size: bool,
    random_grass_patch_size: bool,
    shrub_max_diameter: float,
    shrub_max_top_height: float,
    shrub_min_bottom_height: float,
    grass_patch_max_size_x: float,
    grass_patch_max_size_y: float,
    grass_max_height: float,
) -> None:
    """Print generated scene parameters for easier reproducibility."""
    width_m, length_m = area_size
    area_m2 = width_m * length_m
    mode = "randomized" if randomize_object_counts else "deterministic"
    print(
        f"Scene area: width={width_m:.1f} m, length={length_m:.1f} m, "
        f"total={area_m2:.1f} m^2"
    )
    print(f"Random seed: {seed}")
    print(f"Terrain relief [0..1]: {terrain_relief:.2f}")
    print(f"Object count mode: {mode}")
    print(f"Class labels: {format_class_label_mapping()}")
    print(f"Generation order: {format_class_generation_order()}")
    class_distribution = ", ".join(
        f"{class_id}={class_ratios[class_id] * 100.0:.1f}%"
        for class_id in CLASS_IDS
    )
    print(f"Class distribution: {class_distribution}")
    vehicle_distribution = ", ".join(
        f"{VEHICLE_TYPE_NAMES[vehicle_type]}={vehicle_type_ratios[vehicle_type] * 100.0:.1f}%"
        for vehicle_type in VEHICLE_TYPES
    )
    print(f"Vehicle type distribution: {vehicle_distribution}")
    if (
        artificial_surface_type_distribution_overridden
        and artificial_surface_type_ratios is not None
    ):
        surface_distribution = ", ".join(
            f"{ARTIFICIAL_SURFACE_TYPE_NAMES[surface_type]}="
            f"{artificial_surface_type_ratios[surface_type] * 100.0:.1f}%"
            for surface_type in ARTIFICIAL_SURFACE_TYPES
        )
        print(f"Artificial surface type distribution: {surface_distribution}")
    else:
        print("Artificial surface type distribution: random per scene")
    if building_roof_type_distribution_overridden and building_roof_type_ratios is not None:
        building_distribution = ", ".join(
            f"{BUILDING_ROOF_TYPE_NAMES[roof_type]}={building_roof_type_ratios[roof_type] * 100.0:.1f}%"
            for roof_type in BUILDING_ROOF_TYPES
        )
        print(f"Building roof type distribution: {building_distribution}")
    else:
        print("Building roof type distribution: random per scene")
    if structure_type_distribution_overridden and structure_type_ratios is not None:
        structure_distribution = ", ".join(
            f"{STRUCTURE_TYPE_NAMES[structure_type]}={structure_type_ratios[structure_type] * 100.0:.1f}%"
            for structure_type in STRUCTURE_TYPES
        )
        print(f"Structure type distribution: {structure_distribution}")
    else:
        print("Structure type distribution: random per scene")
    if tree_crown_type_distribution_overridden and tree_crown_type_ratios is not None:
        tree_distribution = ", ".join(
            f"{TREE_CROWN_TYPE_NAMES[crown_type]}={tree_crown_type_ratios[crown_type] * 100.0:.1f}%"
            for crown_type in TREE_CROWN_TYPES
        )
        print(f"Tree crown type distribution: {tree_distribution}")
    else:
        print("Tree crown type distribution: random per scene")
    if artificial_surface_count_overridden:
        print("Artificial surface count mode: custom override")
    if building_count_overridden:
        print("Building count mode: custom override")
    if structure_count_overridden:
        print("Structure count mode: custom override")
    if tree_count_overridden:
        print("Tree count mode: custom override")
    if vehicle_count_overridden:
        print("Vehicle count mode: custom override")
    print(
        "Building modes: "
        f"floors={int(building_floor_min)}..{int(building_floor_max)}, "
        f"yaw={'random' if building_random_yaw else 'aligned'}"
    )
    print(
        "High vegetation modes: "
        f"tree_crown_size={'random' if random_tree_crown_size else 'custom'}"
    )
    if not random_tree_crown_size:
        print(
            "Tree crown custom params: "
            f"max_diameter={tree_max_crown_diameter:.2f} m, "
            f"max_top={tree_max_crown_top_height:.2f} m, "
            f"min_bottom={tree_min_crown_bottom_height:.2f} m"
        )
    print(
        "Low vegetation modes: "
        f"shrub_count={'custom' if shrub_count_overridden else 'random'}, "
        f"shrub_size={'random' if random_shrub_size else 'custom'}, "
        f"grass_patch_count={'custom' if grass_patch_count_overridden else 'random'}, "
        f"grass_size={'random' if random_grass_patch_size else 'custom'}"
    )
    if not random_shrub_size:
        print(
            "Shrub custom params: "
            f"max_diameter={shrub_max_diameter:.2f} m, "
            f"max_top={shrub_max_top_height:.2f} m, "
            f"min_bottom={shrub_min_bottom_height:.2f} m"
        )
    if not random_grass_patch_size:
        print(
            "Grass custom params: "
            f"max_size_x={grass_patch_max_size_x:.2f} m, "
            f"max_size_y={grass_patch_max_size_y:.2f} m, "
            f"max_height={grass_max_height:.2f} m"
        )
    print(
        "Object counts: "
        f"artificial_surfaces={object_counts['num_artificial_surfaces']}, "
        f"trees={object_counts['num_trees']}, "
        f"buildings={object_counts['num_buildings']}, "
        f"structures={object_counts['num_structures']}, "
        f"vehicles={object_counts['num_vehicles']}, "
        f"artifact_clusters={object_counts['num_artifact_clusters']}"
    )


def _run_pipeline(
    total_points: int,
    show_visualization: bool,
    save_csv: bool,
    save_ply: bool,
    area_width: float,
    area_length: float,
    terrain_relief: float,
    randomize_object_counts: bool,
    seed: int,
    class_percentages: Sequence[float] | Dict[int, float] | None = None,
    artificial_surface_count: int | None = None,
    artificial_surface_type_percentages: Sequence[float] | Dict[str, float] | None = None,
    tree_count: int | None = None,
    tree_crown_type_percentages: Sequence[float] | Dict[str, float] | None = None,
    random_tree_crown_size: bool = True,
    tree_max_crown_diameter: float = HIGH_VEG_DEFAULTS["tree_max_crown_diameter"],
    tree_max_crown_top_height: float = HIGH_VEG_DEFAULTS["tree_max_crown_top_height"],
    tree_min_crown_bottom_height: float = HIGH_VEG_DEFAULTS["tree_min_crown_bottom_height"],
    building_count: int | None = None,
    building_roof_type_percentages: Sequence[float] | Dict[str, float] | None = None,
    building_floor_min: int = int(BUILDING_DEFAULTS["building_floor_min"]),
    building_floor_max: int = int(BUILDING_DEFAULTS["building_floor_max"]),
    building_random_yaw: bool = bool(BUILDING_DEFAULTS["building_random_yaw"]),
    structure_count: int | None = None,
    structure_type_percentages: Sequence[float] | Dict[str, float] | None = None,
    vehicle_count: int | None = None,
    vehicle_type_percentages: Sequence[float] | Dict[str, float] | None = None,
    artifacts_enabled: bool = bool(ARTIFACT_DEFAULTS["enabled"]),
    artifact_global_intensity: float = float(ARTIFACT_DEFAULTS["global_intensity"]),
    artifact_point_fraction: float | None = None,
    artifact_type_settings: Mapping[str, Mapping[str, Any]] | None = None,
    shrub_count: int | None = None,
    random_shrub_size: bool = True,
    shrub_max_diameter: float = LOW_VEG_DEFAULTS["shrub_max_diameter"],
    shrub_max_top_height: float = LOW_VEG_DEFAULTS["shrub_max_top_height"],
    shrub_min_bottom_height: float = LOW_VEG_DEFAULTS["shrub_min_bottom_height"],
    grass_patch_count: int | None = None,
    random_grass_patch_size: bool = True,
    grass_patch_max_size_x: float = LOW_VEG_DEFAULTS["grass_patch_max_size_x"],
    grass_patch_max_size_y: float = LOW_VEG_DEFAULTS["grass_patch_max_size_y"],
    grass_max_height: float = LOW_VEG_DEFAULTS["grass_max_height"],
    progress_callback: ProgressCallback | None = None,
) -> np.ndarray:
    """
    Internal execution pipeline with configurable runtime switches.
    """
    _emit_progress(progress_callback, 0.02, "Validating parameters")

    if total_points <= 0:
        raise ValueError("`total_points` must be > 0.")
    _validate_positive(area_width, "area_width")
    _validate_positive(area_length, "area_length")
    _validate_unit_interval(terrain_relief, "terrain_relief")
    _validate_unit_interval(artifact_global_intensity, "artifact_global_intensity")
    if artifact_point_fraction is not None:
        _validate_unit_interval(artifact_point_fraction, "artifact_point_fraction")
    _validate_positive(tree_max_crown_diameter, "tree_max_crown_diameter")
    _validate_positive(tree_max_crown_top_height, "tree_max_crown_top_height")
    if tree_min_crown_bottom_height < 0.0:
        raise ValueError("`tree_min_crown_bottom_height` must be >= 0.")
    if tree_min_crown_bottom_height >= tree_max_crown_top_height:
        raise ValueError(
            "`tree_min_crown_bottom_height` must be strictly less than "
            "`tree_max_crown_top_height`."
        )
    _validate_positive(shrub_max_diameter, "shrub_max_diameter")
    _validate_positive(shrub_max_top_height, "shrub_max_top_height")
    if shrub_min_bottom_height < 0.0:
        raise ValueError("`shrub_min_bottom_height` must be >= 0.")
    if shrub_min_bottom_height >= shrub_max_top_height:
        raise ValueError(
            "`shrub_min_bottom_height` must be strictly less than `shrub_max_top_height`."
        )
    _validate_positive(grass_patch_max_size_x, "grass_patch_max_size_x")
    _validate_positive(grass_patch_max_size_y, "grass_patch_max_size_y")
    _validate_positive(grass_max_height, "grass_max_height")
    if artificial_surface_count is not None and int(artificial_surface_count) <= 0:
        raise ValueError("`artificial_surface_count` must be > 0 when provided.")
    if shrub_count is not None and int(shrub_count) <= 0:
        raise ValueError("`shrub_count` must be > 0 when provided.")
    if grass_patch_count is not None and int(grass_patch_count) <= 0:
        raise ValueError("`grass_patch_count` must be > 0 when provided.")
    if tree_count is not None and int(tree_count) <= 0:
        raise ValueError("`tree_count` must be > 0 when provided.")
    if building_count is not None and int(building_count) <= 0:
        raise ValueError("`building_count` must be > 0 when provided.")
    if structure_count is not None and int(structure_count) <= 0:
        raise ValueError("`structure_count` must be > 0 when provided.")
    floor_min = int(building_floor_min)
    floor_max = int(building_floor_max)
    if floor_min <= 0:
        raise ValueError("`building_floor_min` must be >= 1.")
    if floor_max < floor_min:
        raise ValueError("`building_floor_max` must be >= `building_floor_min`.")
    normalized_artifact_type_settings = _normalize_artifact_type_settings_payload(
        artifact_type_settings,
        None,
        defaults=DEFAULT_ARTIFACT_TYPE_SETTINGS,
    )
    effective_artifact_fraction = (
        0.0 if not bool(artifacts_enabled) else float(artifact_point_fraction or 0.0)
    )
    if bool(artifacts_enabled) and effective_artifact_fraction > 0.0:
        enabled_artifact_types = [
            settings
            for settings in normalized_artifact_type_settings.values()
            if bool(settings["enabled"])
        ]
        if not enabled_artifact_types:
            raise ValueError(
                "At least one artifact type must be enabled when artifacts are enabled and "
                "`artifact_point_fraction` is > 0."
            )
        if not any(float(settings["amount"]) > 0.0 for settings in enabled_artifact_types):
            raise ValueError(
                "At least one enabled artifact type must have `amount` > 0 when artifacts "
                "are enabled and `artifact_point_fraction` is > 0."
            )

    _emit_progress(progress_callback, 0.10, "Preparing scene configuration")

    # ----------------------------- Scene config -----------------------------
    area_size = (float(area_width), float(area_length))
    class_ratios = _class_ratios_from_percentages(class_percentages)
    if not bool(artifacts_enabled):
        class_ratios = _override_artifact_class_ratio(class_ratios, 0.0)
    elif artifact_point_fraction is not None:
        class_ratios = _override_artifact_class_ratio(
            class_ratios,
            float(artifact_point_fraction),
        )
    artificial_surface_type_distribution_overridden = (
        artificial_surface_type_percentages is not None
    )
    artificial_surface_type_ratios = (
        _artificial_surface_type_ratios_from_percentages(artificial_surface_type_percentages)
        if artificial_surface_type_distribution_overridden
        else None
    )
    vehicle_type_ratios = _vehicle_type_ratios_from_percentages(vehicle_type_percentages)
    building_roof_type_distribution_overridden = building_roof_type_percentages is not None
    building_roof_type_ratios = (
        _building_roof_type_ratios_from_percentages(building_roof_type_percentages)
        if building_roof_type_distribution_overridden
        else None
    )
    structure_type_distribution_overridden = structure_type_percentages is not None
    structure_type_ratios = (
        _structure_type_ratios_from_percentages(structure_type_percentages)
        if structure_type_distribution_overridden
        else None
    )
    tree_crown_type_distribution_overridden = tree_crown_type_percentages is not None
    tree_crown_type_ratios = (
        _tree_crown_type_ratios_from_percentages(tree_crown_type_percentages)
        if tree_crown_type_distribution_overridden
        else None
    )
    scene_rng = np.random.default_rng(seed + 100)
    object_counts = _generate_object_counts(
        area_size=area_size,
        rng=scene_rng,
        randomize_counts=randomize_object_counts,
    )
    artificial_surface_count_overridden = artificial_surface_count is not None
    building_count_overridden = building_count is not None
    structure_count_overridden = structure_count is not None
    tree_count_overridden = tree_count is not None
    vehicle_count_overridden = vehicle_count is not None
    shrub_count_overridden = shrub_count is not None
    grass_patch_count_overridden = grass_patch_count is not None
    if artificial_surface_count_overridden:
        artificial_surface_count_int = int(artificial_surface_count)
        if artificial_surface_count_int <= 0:
            raise ValueError("`artificial_surface_count` must be > 0 when provided.")
        object_counts["num_artificial_surfaces"] = artificial_surface_count_int
    if building_count_overridden:
        building_count_int = int(building_count)
        if building_count_int <= 0:
            raise ValueError("`building_count` must be > 0 when provided.")
        object_counts["num_buildings"] = building_count_int
    if structure_count_overridden:
        structure_count_int = int(structure_count)
        if structure_count_int <= 0:
            raise ValueError("`structure_count` must be > 0 when provided.")
        object_counts["num_structures"] = structure_count_int
    if tree_count_overridden:
        tree_count_int = int(tree_count)
        if tree_count_int <= 0:
            raise ValueError("`tree_count` must be > 0 when provided.")
        object_counts["num_trees"] = tree_count_int
    if vehicle_count_overridden:
        vehicle_count_int = int(vehicle_count)
        if vehicle_count_int <= 0:
            raise ValueError("`vehicle_count` must be > 0 when provided.")
        object_counts["num_vehicles"] = vehicle_count_int

    num_trees = object_counts["num_trees"]
    num_buildings = object_counts["num_buildings"]
    num_structures = object_counts["num_structures"]
    num_vehicles = object_counts["num_vehicles"]
    num_artifact_clusters = object_counts["num_artifact_clusters"]
    _print_scene_params(
        area_size=area_size,
        seed=seed,
        terrain_relief=terrain_relief,
        randomize_object_counts=randomize_object_counts,
        object_counts=object_counts,
        class_ratios=class_ratios,
        artificial_surface_type_ratios=artificial_surface_type_ratios,
        artificial_surface_count_overridden=artificial_surface_count_overridden,
        artificial_surface_type_distribution_overridden=(
            artificial_surface_type_distribution_overridden
        ),
        building_roof_type_ratios=building_roof_type_ratios,
        building_count_overridden=building_count_overridden,
        building_roof_type_distribution_overridden=building_roof_type_distribution_overridden,
        building_floor_min=floor_min,
        building_floor_max=floor_max,
        building_random_yaw=bool(building_random_yaw),
        structure_type_ratios=structure_type_ratios,
        structure_count_overridden=structure_count_overridden,
        structure_type_distribution_overridden=structure_type_distribution_overridden,
        vehicle_type_ratios=vehicle_type_ratios,
        tree_crown_type_ratios=tree_crown_type_ratios,
        tree_count_overridden=tree_count_overridden,
        vehicle_count_overridden=vehicle_count_overridden,
        tree_crown_type_distribution_overridden=tree_crown_type_distribution_overridden,
        random_tree_crown_size=bool(random_tree_crown_size),
        tree_max_crown_diameter=float(tree_max_crown_diameter),
        tree_max_crown_top_height=float(tree_max_crown_top_height),
        tree_min_crown_bottom_height=float(tree_min_crown_bottom_height),
        shrub_count_overridden=shrub_count_overridden,
        grass_patch_count_overridden=grass_patch_count_overridden,
        random_shrub_size=bool(random_shrub_size),
        random_grass_patch_size=bool(random_grass_patch_size),
        shrub_max_diameter=float(shrub_max_diameter),
        shrub_max_top_height=float(shrub_max_top_height),
        shrub_min_bottom_height=float(shrub_min_bottom_height),
        grass_patch_max_size_x=float(grass_patch_max_size_x),
        grass_patch_max_size_y=float(grass_patch_max_size_y),
        grass_max_height=float(grass_max_height),
    )

    _emit_progress(progress_callback, 0.18, "Allocating points by class")
    points_per_class = allocate_points(total_points=total_points, class_ratios=class_ratios)

    _emit_progress(progress_callback, 0.24, "Building terrain model")
    terrain_fn = _build_terrain_height_function(area_size, terrain_relief)

    _emit_progress(progress_callback, 0.30, "Generating scene objects")
    object_points, artificial_zones = place_objects(
        terrain_fn=terrain_fn,
        area_size=area_size,
        points_per_class=points_per_class,
        num_artificial_surfaces=object_counts["num_artificial_surfaces"],
        artificial_surface_type_ratios=artificial_surface_type_ratios,
        num_trees=num_trees,
        num_buildings=num_buildings,
        building_roof_type_ratios=building_roof_type_ratios,
        building_floor_min=floor_min,
        building_floor_max=floor_max,
        building_random_yaw=bool(building_random_yaw),
        num_structures=num_structures,
        structure_type_ratios=structure_type_ratios,
        num_vehicles=num_vehicles,
        vehicle_type_ratios=vehicle_type_ratios,
        tree_crown_type_ratios=tree_crown_type_ratios,
        random_tree_crown_size=bool(random_tree_crown_size),
        tree_max_crown_diameter=float(tree_max_crown_diameter),
        tree_max_crown_top_height=float(tree_max_crown_top_height),
        tree_min_crown_bottom_height=float(tree_min_crown_bottom_height),
        num_artifact_clusters=num_artifact_clusters,
        artifacts_enabled=bool(artifacts_enabled),
        artifact_global_intensity=float(artifact_global_intensity),
        artifact_type_settings=normalized_artifact_type_settings,
        shrub_count=(int(shrub_count) if shrub_count is not None else None),
        random_shrub_size=bool(random_shrub_size),
        shrub_max_diameter=float(shrub_max_diameter),
        shrub_max_top_height=float(shrub_max_top_height),
        shrub_min_bottom_height=float(shrub_min_bottom_height),
        grass_patch_count=(int(grass_patch_count) if grass_patch_count is not None else None),
        random_grass_patch_size=bool(random_grass_patch_size),
        grass_patch_max_size_x=float(grass_patch_max_size_x),
        grass_patch_max_size_y=float(grass_patch_max_size_y),
        grass_max_height=float(grass_max_height),
        seed=seed + 1,
        progress_callback=_make_progress_subrange(progress_callback, 0.30, 0.72),
    )

    _emit_progress(progress_callback, 0.78, "Sampling natural terrain")
    terrain_points = _sample_natural_terrain_points(
        rng=np.random.default_rng(seed),
        terrain_fn=terrain_fn,
        area_size=area_size,
        n_points=int(points_per_class.get(NATURAL_SURFACE_CLASS_ID, 0)),
        forbidden_rects=[zone["rect"] for zone in artificial_zones],  # type: ignore[index]
        terrain_relief=terrain_relief,
    )
    artificial_zone_counts = {
        surface_type: sum(
            1
            for zone in artificial_zones
            if str(zone.get("surface_type", zone.get("kind", ""))).strip().lower()
            == surface_type
        )
        for surface_type in ARTIFICIAL_SURFACE_TYPES
    }
    artificial_zone_summary = ", ".join(
        f"{surface_type}={artificial_zone_counts[surface_type]}"
        for surface_type in ARTIFICIAL_SURFACE_TYPES
    )
    print(f"Artificial zones: {artificial_zone_summary}")

    _emit_progress(progress_callback, 0.88, "Assembling point cloud")
    point_cloud = np.vstack([terrain_points, object_points])

    # Shuffle rows so classes are not grouped by generation stage.
    rng = np.random.default_rng(seed + 2)
    rng.shuffle(point_cloud, axis=0)

    # Ensure exact requested count if constraints fallback produced extras.
    if point_cloud.shape[0] > total_points:
        point_cloud = point_cloud[:total_points]
    elif point_cloud.shape[0] < total_points:
        missing = total_points - point_cloud.shape[0]
        x_pad, y_pad = _sample_xy_outside_rects(
            rng,
            missing,
            area_size=area_size,
            forbidden_rects=[zone["rect"] for zone in artificial_zones],  # type: ignore[index]
            margin=0.0,
        )
        z_pad = terrain_fn(x_pad, y_pad)
        l_pad = np.full(missing, NATURAL_SURFACE_CLASS_ID, dtype=np.int32)
        point_cloud = np.vstack([point_cloud, np.column_stack((x_pad, y_pad, z_pad, l_pad))])

    _emit_progress(progress_callback, 0.94, "Computing label statistics")
    labels = point_cloud[:, 3].astype(np.int32)
    _print_stats(labels)

    if save_csv:
        _emit_progress(progress_callback, 0.97, "Saving CSV")
        csv_path = ensure_data_dir() / "synthetic_landscape_point_cloud.csv"
        export_to_csv(point_cloud, csv_path)
        print(f"Saved CSV: {csv_path}")
    if save_ply:
        _emit_progress(progress_callback, 0.99, "Saving PLY")
        ply_path = ensure_data_dir() / "synthetic_landscape_point_cloud.ply"
        export_to_ply(point_cloud, ply_path)
        print(f"Saved PLY: {ply_path}")

    if show_visualization:
        _emit_progress(progress_callback, 0.995, "Opening visualization")
        visualize_point_cloud(point_cloud[:, :3], labels)

    _emit_progress(progress_callback, 1.0, "Generation complete")
    return point_cloud


def generate_point_cloud(
    total_points: int = 100_000,
    area_width: float = 240.0,
    area_length: float = 220.0,
    terrain_relief: float = 1.0,
    randomize_object_counts: bool = True,
    seed: int = 12,
    class_percentages: Sequence[float] | Dict[int, float] | None = None,
    artificial_surface_count: int | None = None,
    artificial_surface_type_percentages: Sequence[float] | Dict[str, float] | None = None,
    tree_count: int | None = None,
    tree_crown_type_percentages: Sequence[float] | Dict[str, float] | None = None,
    random_tree_crown_size: bool = True,
    tree_max_crown_diameter: float = HIGH_VEG_DEFAULTS["tree_max_crown_diameter"],
    tree_max_crown_top_height: float = HIGH_VEG_DEFAULTS["tree_max_crown_top_height"],
    tree_min_crown_bottom_height: float = HIGH_VEG_DEFAULTS["tree_min_crown_bottom_height"],
    building_count: int | None = None,
    building_roof_type_percentages: Sequence[float] | Dict[str, float] | None = None,
    building_floor_min: int = int(BUILDING_DEFAULTS["building_floor_min"]),
    building_floor_max: int = int(BUILDING_DEFAULTS["building_floor_max"]),
    building_random_yaw: bool = bool(BUILDING_DEFAULTS["building_random_yaw"]),
    structure_count: int | None = None,
    structure_type_percentages: Sequence[float] | Dict[str, float] | None = None,
    vehicle_count: int | None = None,
    vehicle_type_percentages: Sequence[float] | Dict[str, float] | None = None,
    artifacts_enabled: bool = bool(ARTIFACT_DEFAULTS["enabled"]),
    artifact_global_intensity: float = float(ARTIFACT_DEFAULTS["global_intensity"]),
    artifact_point_fraction: float | None = None,
    artifact_type_settings: Mapping[str, Mapping[str, Any]] | None = None,
    shrub_count: int | None = None,
    random_shrub_size: bool = True,
    shrub_max_diameter: float = LOW_VEG_DEFAULTS["shrub_max_diameter"],
    shrub_max_top_height: float = LOW_VEG_DEFAULTS["shrub_max_top_height"],
    shrub_min_bottom_height: float = LOW_VEG_DEFAULTS["shrub_min_bottom_height"],
    grass_patch_count: int | None = None,
    random_grass_patch_size: bool = True,
    grass_patch_max_size_x: float = LOW_VEG_DEFAULTS["grass_patch_max_size_x"],
    grass_patch_max_size_y: float = LOW_VEG_DEFAULTS["grass_patch_max_size_y"],
    grass_max_height: float = LOW_VEG_DEFAULTS["grass_max_height"],
    progress_callback: ProgressCallback | None = None,
) -> np.ndarray:
    """
    Public API for programmatic generation without side effects:
      - no matplotlib window
      - no CSV/PLY files written
      - optional custom class percentages for classes 0..7
      - optional custom number of artificial surface objects (class 0)
      - optional custom artificial surface type percentages
      - optional custom number of tree instances
      - optional custom tree crown type percentages
      - optional custom high vegetation crown sizing controls
      - optional custom number of building instances
      - optional custom building roof type percentages
      - configurable random building floor range and arbitrary Z-rotation
      - optional custom number of structure instances
      - optional custom structure type percentages
      - optional custom number of vehicle instances
      - optional custom vehicle type percentages [car, truck, bus]
      - optional artifact controls (enable flag, global intensity, per-type settings)
      - optional custom low vegetation controls (shrubs + grass patches)
      - optional progress callback receiving `(progress, stage)` updates
    Returns point cloud array with shape (N, 4): x, y, z, label.
    """
    return _run_pipeline(
        total_points=int(total_points),
        show_visualization=False,
        save_csv=False,
        save_ply=False,
        area_width=float(area_width),
        area_length=float(area_length),
        terrain_relief=float(terrain_relief),
        randomize_object_counts=bool(randomize_object_counts),
        seed=int(seed),
        class_percentages=class_percentages,
        artificial_surface_count=artificial_surface_count,
        artificial_surface_type_percentages=artificial_surface_type_percentages,
        tree_count=tree_count,
        tree_crown_type_percentages=tree_crown_type_percentages,
        random_tree_crown_size=bool(random_tree_crown_size),
        tree_max_crown_diameter=float(tree_max_crown_diameter),
        tree_max_crown_top_height=float(tree_max_crown_top_height),
        tree_min_crown_bottom_height=float(tree_min_crown_bottom_height),
        building_count=building_count,
        building_roof_type_percentages=building_roof_type_percentages,
        building_floor_min=int(building_floor_min),
        building_floor_max=int(building_floor_max),
        building_random_yaw=bool(building_random_yaw),
        structure_count=structure_count,
        structure_type_percentages=structure_type_percentages,
        vehicle_count=vehicle_count,
        vehicle_type_percentages=vehicle_type_percentages,
        artifacts_enabled=bool(artifacts_enabled),
        artifact_global_intensity=float(artifact_global_intensity),
        artifact_point_fraction=artifact_point_fraction,
        artifact_type_settings=artifact_type_settings,
        shrub_count=shrub_count,
        random_shrub_size=bool(random_shrub_size),
        shrub_max_diameter=float(shrub_max_diameter),
        shrub_max_top_height=float(shrub_max_top_height),
        shrub_min_bottom_height=float(shrub_min_bottom_height),
        grass_patch_count=grass_patch_count,
        random_grass_patch_size=bool(random_grass_patch_size),
        grass_patch_max_size_x=float(grass_patch_max_size_x),
        grass_patch_max_size_y=float(grass_patch_max_size_y),
        grass_max_height=float(grass_max_height),
        progress_callback=progress_callback,
    )


def main(
    total_points: int = 100_000,
    area_width: float = 240.0,
    area_length: float = 220.0,
    terrain_relief: float = 1.0,
    randomize_object_counts: bool = True,
    seed: int = 12,
    class_percentages: Sequence[float] | Dict[int, float] | None = None,
    artificial_surface_count: int | None = None,
    artificial_surface_type_percentages: Sequence[float] | Dict[str, float] | None = None,
    tree_count: int | None = None,
    tree_crown_type_percentages: Sequence[float] | Dict[str, float] | None = None,
    random_tree_crown_size: bool = True,
    tree_max_crown_diameter: float = HIGH_VEG_DEFAULTS["tree_max_crown_diameter"],
    tree_max_crown_top_height: float = HIGH_VEG_DEFAULTS["tree_max_crown_top_height"],
    tree_min_crown_bottom_height: float = HIGH_VEG_DEFAULTS["tree_min_crown_bottom_height"],
    building_count: int | None = None,
    building_roof_type_percentages: Sequence[float] | Dict[str, float] | None = None,
    building_floor_min: int = int(BUILDING_DEFAULTS["building_floor_min"]),
    building_floor_max: int = int(BUILDING_DEFAULTS["building_floor_max"]),
    building_random_yaw: bool = bool(BUILDING_DEFAULTS["building_random_yaw"]),
    structure_count: int | None = None,
    structure_type_percentages: Sequence[float] | Dict[str, float] | None = None,
    vehicle_count: int | None = None,
    vehicle_type_percentages: Sequence[float] | Dict[str, float] | None = None,
    artifacts_enabled: bool = bool(ARTIFACT_DEFAULTS["enabled"]),
    artifact_global_intensity: float = float(ARTIFACT_DEFAULTS["global_intensity"]),
    artifact_point_fraction: float | None = None,
    artifact_type_settings: Mapping[str, Mapping[str, Any]] | None = None,
    shrub_count: int | None = None,
    random_shrub_size: bool = True,
    shrub_max_diameter: float = LOW_VEG_DEFAULTS["shrub_max_diameter"],
    shrub_max_top_height: float = LOW_VEG_DEFAULTS["shrub_max_top_height"],
    shrub_min_bottom_height: float = LOW_VEG_DEFAULTS["shrub_min_bottom_height"],
    grass_patch_count: int | None = None,
    random_grass_patch_size: bool = True,
    grass_patch_max_size_x: float = LOW_VEG_DEFAULTS["grass_patch_max_size_x"],
    grass_patch_max_size_y: float = LOW_VEG_DEFAULTS["grass_patch_max_size_y"],
    grass_max_height: float = LOW_VEG_DEFAULTS["grass_max_height"],
    progress_callback: ProgressCallback | None = None,
) -> np.ndarray:
    """
    Entry point required by task.
    By default:
      - generates 100k points
      - uses area 240x220 meters
      - uses mountainous terrain relief (`terrain_relief=1.0`)
      - randomizes object counts based on area
      - uses default class distribution unless custom percentages are provided
      - uses random artificial surface generation unless custom values are provided
      - uses default high-vegetation generation unless custom values are provided
      - uses default building generation unless custom values are provided
      - uses random structure generation unless custom values are provided
      - uses default vehicle count and type distribution unless custom values are provided
      - supports configurable artifact generation controls
      - uses default low-vegetation generation unless custom values are provided
      - optional progress callback receiving `(progress, stage)` updates
      - prints stats
      - saves CSV + PLY
      - opens interactive visualization
    """
    return _run_pipeline(
        total_points=int(total_points),
        show_visualization=True,
        save_csv=True,
        save_ply=True,
        area_width=float(area_width),
        area_length=float(area_length),
        terrain_relief=float(terrain_relief),
        randomize_object_counts=bool(randomize_object_counts),
        seed=int(seed),
        class_percentages=class_percentages,
        artificial_surface_count=artificial_surface_count,
        artificial_surface_type_percentages=artificial_surface_type_percentages,
        tree_count=tree_count,
        tree_crown_type_percentages=tree_crown_type_percentages,
        random_tree_crown_size=bool(random_tree_crown_size),
        tree_max_crown_diameter=float(tree_max_crown_diameter),
        tree_max_crown_top_height=float(tree_max_crown_top_height),
        tree_min_crown_bottom_height=float(tree_min_crown_bottom_height),
        building_count=building_count,
        building_roof_type_percentages=building_roof_type_percentages,
        building_floor_min=int(building_floor_min),
        building_floor_max=int(building_floor_max),
        building_random_yaw=bool(building_random_yaw),
        structure_count=structure_count,
        structure_type_percentages=structure_type_percentages,
        vehicle_count=vehicle_count,
        vehicle_type_percentages=vehicle_type_percentages,
        artifacts_enabled=bool(artifacts_enabled),
        artifact_global_intensity=float(artifact_global_intensity),
        artifact_point_fraction=artifact_point_fraction,
        artifact_type_settings=artifact_type_settings,
        shrub_count=shrub_count,
        random_shrub_size=bool(random_shrub_size),
        shrub_max_diameter=float(shrub_max_diameter),
        shrub_max_top_height=float(shrub_max_top_height),
        shrub_min_bottom_height=float(shrub_min_bottom_height),
        grass_patch_count=grass_patch_count,
        random_grass_patch_size=bool(random_grass_patch_size),
        grass_patch_max_size_x=float(grass_patch_max_size_x),
        grass_patch_max_size_y=float(grass_patch_max_size_y),
        grass_max_height=float(grass_max_height),
        progress_callback=progress_callback,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic labeled landscape point cloud."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to YAML configuration file with generation settings.",
    )
    parser.add_argument(
        "--total-points",
        type=int,
        default=100_000,
        help="Total number of generated points.",
    )
    parser.add_argument(
        "--area-width",
        type=float,
        default=240.0,
        help="Scene width along X axis in meters.",
    )
    parser.add_argument(
        "--area-length",
        type=float,
        default=220.0,
        help="Scene length along Y axis in meters.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12,
        help="Random seed for reproducible scene generation.",
    )
    parser.add_argument(
        "--terrain-relief",
        type=float,
        default=1.0,
        help="Terrain elevation intensity in [0,1]: 0=flat, 1=mountainous.",
    )
    parser.add_argument(
        "--random-object-counts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable randomized object counts (trees/buildings/structures/vehicles/artifacts) "
            "based on scene area."
        ),
    )
    parser.add_argument(
        "--class-percentages",
        type=float,
        nargs=len(CLASS_IDS),
        metavar="PCT",
        help=(
            "Optional custom class shares in percent for label order "
            f"{format_class_label_mapping()} ({len(CLASS_IDS)} values expected). "
            "Example: --class-percentages 13 38 14 16 10 4 2 3"
        ),
    )
    parser.add_argument(
        "--artificial-surface-count",
        type=int,
        help="Optional custom number of generated artificial surface objects (class 0).",
    )
    parser.add_argument(
        "--artificial-surface-type-percentages",
        type=float,
        nargs=len(ARTIFICIAL_SURFACE_TYPES),
        metavar="PCT",
        help=(
            "Optional artificial surface type shares in percent for "
            "[road_network sidewalk parking_lot building_front_area "
            "industrial_concrete_pad platform]. "
            "Example: --artificial-surface-type-percentages 30 22 16 10 12 10"
        ),
    )
    parser.add_argument(
        "--tree-count",
        type=int,
        help="Optional custom number of generated tree instances (class 2).",
    )
    parser.add_argument(
        "--tree-crown-type-percentages",
        type=float,
        nargs=len(TREE_CROWN_TYPES),
        metavar="PCT",
        help=(
            "Optional tree crown type shares in percent for "
            "[spherical pyramidal spreading weeping columnar umbrella]. "
            "Example: --tree-crown-type-percentages 30 18 20 10 12 10"
        ),
    )
    parser.add_argument(
        "--random-tree-crown-size",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable randomized tree crown sizes. Disable to use custom tree crown parameters "
            "(max diameter, max top height, min bottom height)."
        ),
    )
    parser.add_argument(
        "--tree-max-crown-diameter",
        type=float,
        default=HIGH_VEG_DEFAULTS["tree_max_crown_diameter"],
        help="Custom tree max crown diameter in meters.",
    )
    parser.add_argument(
        "--tree-max-crown-top-height",
        type=float,
        default=HIGH_VEG_DEFAULTS["tree_max_crown_top_height"],
        help="Custom tree max crown top height above ground in meters.",
    )
    parser.add_argument(
        "--tree-min-crown-bottom-height",
        type=float,
        default=HIGH_VEG_DEFAULTS["tree_min_crown_bottom_height"],
        help="Custom tree minimum crown bottom height above ground in meters.",
    )
    parser.add_argument(
        "--building-count",
        type=int,
        help=f"Optional custom number of generated buildings (class {BUILDINGS_CLASS_ID}).",
    )
    parser.add_argument(
        "--building-roof-type-percentages",
        type=float,
        nargs=len(BUILDING_ROOF_TYPES),
        metavar="PCT",
        help=(
            "Optional building roof type shares in percent for "
            "[single_slope gable hip tent mansard flat dome arched shell]. "
            "Example: --building-roof-type-percentages 10 18 14 8 10 20 6 8 6"
        ),
    )
    parser.add_argument(
        "--building-floor-min",
        type=int,
        default=int(BUILDING_DEFAULTS["building_floor_min"]),
        help="Minimum number of floors for random building generation.",
    )
    parser.add_argument(
        "--building-floor-max",
        type=int,
        default=int(BUILDING_DEFAULTS["building_floor_max"]),
        help="Maximum number of floors for random building generation.",
    )
    parser.add_argument(
        "--building-random-yaw",
        action=argparse.BooleanOptionalAction,
        default=bool(BUILDING_DEFAULTS["building_random_yaw"]),
        help="Enable arbitrary random building rotation around Z axis.",
    )
    parser.add_argument(
        "--structure-count",
        type=int,
        help=f"Optional custom number of generated structures (class {STRUCTURES_CLASS_ID}).",
    )
    parser.add_argument(
        "--structure-type-percentages",
        type=float,
        nargs=len(STRUCTURE_TYPES),
        metavar="PCT",
        help=(
            "Optional structure type shares in percent for "
            f"{list(STRUCTURE_TYPES)}."
        ),
    )
    parser.add_argument(
        "--vehicle-count",
        type=int,
        help="Optional custom number of generated vehicle instances.",
    )
    parser.add_argument(
        "--vehicle-type-percentages",
        type=float,
        nargs=len(VEHICLE_TYPES),
        metavar="PCT",
        help=(
            "Optional vehicle type shares in percent for [car truck bus]. "
            "Example: --vehicle-type-percentages 72 18 10"
        ),
    )
    parser.add_argument(
        "--artifacts-enabled",
        action=argparse.BooleanOptionalAction,
        default=bool(ARTIFACT_DEFAULTS["enabled"]),
        help="Enable generation of artifact points (class 6).",
    )
    parser.add_argument(
        "--artifact-global-intensity",
        type=float,
        default=float(ARTIFACT_DEFAULTS["global_intensity"]),
        help="Global intensity multiplier for artifact generators in [0,1].",
    )
    parser.add_argument(
        "--artifact-point-fraction",
        type=float,
        default=None,
        help=(
            "Optional target fraction of total points assigned to artifacts in [0,1]. "
            "When omitted, the artifact share follows `class_percentages`."
        ),
    )
    parser.add_argument(
        "--shrub-count",
        type=int,
        help="Optional custom number of shrub clusters for class 3 (low vegetation).",
    )
    parser.add_argument(
        "--random-shrub-size",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable randomized shrub sizes. Disable to use custom shrub parameters "
            "(max diameter, max top height, min bottom height)."
        ),
    )
    parser.add_argument(
        "--shrub-max-diameter",
        type=float,
        default=LOW_VEG_DEFAULTS["shrub_max_diameter"],
        help="Custom shrub max crown diameter in meters.",
    )
    parser.add_argument(
        "--shrub-max-top-height",
        type=float,
        default=LOW_VEG_DEFAULTS["shrub_max_top_height"],
        help="Custom shrub max top height above ground in meters.",
    )
    parser.add_argument(
        "--shrub-min-bottom-height",
        type=float,
        default=LOW_VEG_DEFAULTS["shrub_min_bottom_height"],
        help="Custom shrub minimum bottom crown height above ground in meters.",
    )
    parser.add_argument(
        "--grass-patch-count",
        type=int,
        help="Optional custom number of grass patches for class 3 (low vegetation).",
    )
    parser.add_argument(
        "--random-grass-patch-size",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable randomized grass patch sizes. Disable to use custom patch size "
            "(max X/Y spread and max grass height)."
        ),
    )
    parser.add_argument(
        "--grass-patch-max-size-x",
        type=float,
        default=LOW_VEG_DEFAULTS["grass_patch_max_size_x"],
        help="Custom grass patch max spread along X in meters.",
    )
    parser.add_argument(
        "--grass-patch-max-size-y",
        type=float,
        default=LOW_VEG_DEFAULTS["grass_patch_max_size_y"],
        help="Custom grass patch max spread along Y in meters.",
    )
    parser.add_argument(
        "--grass-max-height",
        type=float,
        default=LOW_VEG_DEFAULTS["grass_max_height"],
        help="Custom max grass height above ground in meters.",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Disable interactive visualization window.",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Disable CSV export.",
    )
    parser.add_argument(
        "--no-ply",
        action="store_true",
        help="Disable PLY export.",
    )
    return parser


def _collect_explicit_cli_dests(
    parser: argparse.ArgumentParser,
    argv: Sequence[str],
) -> set[str]:
    """Track which CLI options were explicitly provided so config files can be merged safely."""
    option_to_dest: Dict[str, str] = {}
    for action in parser._actions:
        for option in action.option_strings:
            option_to_dest[option] = action.dest

    explicit_dests: set[str] = set()
    for token in argv:
        if token == "--":
            break
        if not token.startswith("-") or token == "-":
            continue
        option = token.split("=", 1)[0]
        dest = option_to_dest.get(option)
        if dest is not None:
            explicit_dests.add(dest)
    return explicit_dests


def cli(argv: Sequence[str] | None = None) -> int:
    """Command line entry point."""
    parser = build_arg_parser()
    raw_argv = list(argv) if argv is not None else list(sys.argv[1:])
    args = parser.parse_args(raw_argv)
    explicit_dests = _collect_explicit_cli_dests(parser, raw_argv)

    config_values = default_generation_config()
    if args.config is not None:
        try:
            config_values = load_generation_config(args.config)
        except ValueError as exc:
            parser.error(str(exc))

    pipeline_kwargs = generation_config_to_pipeline_kwargs(config_values)

    if "total_points" in explicit_dests:
        pipeline_kwargs["total_points"] = int(args.total_points)
    if "area_width" in explicit_dests:
        pipeline_kwargs["area_width"] = float(args.area_width)
    if "area_length" in explicit_dests:
        pipeline_kwargs["area_length"] = float(args.area_length)
    if "terrain_relief" in explicit_dests:
        pipeline_kwargs["terrain_relief"] = float(args.terrain_relief)
    if "random_object_counts" in explicit_dests:
        pipeline_kwargs["randomize_object_counts"] = bool(args.random_object_counts)
    if "seed" in explicit_dests:
        pipeline_kwargs["seed"] = int(args.seed)
    if "class_percentages" in explicit_dests:
        pipeline_kwargs["class_percentages"] = args.class_percentages
    if "artificial_surface_count" in explicit_dests:
        pipeline_kwargs["artificial_surface_count"] = args.artificial_surface_count
    if "artificial_surface_type_percentages" in explicit_dests:
        pipeline_kwargs["artificial_surface_type_percentages"] = (
            args.artificial_surface_type_percentages
        )
    if "tree_count" in explicit_dests:
        pipeline_kwargs["tree_count"] = args.tree_count
    if "tree_crown_type_percentages" in explicit_dests:
        pipeline_kwargs["tree_crown_type_percentages"] = args.tree_crown_type_percentages
    if "random_tree_crown_size" in explicit_dests:
        pipeline_kwargs["random_tree_crown_size"] = bool(args.random_tree_crown_size)
    if "tree_max_crown_diameter" in explicit_dests:
        pipeline_kwargs["tree_max_crown_diameter"] = float(args.tree_max_crown_diameter)
    if "tree_max_crown_top_height" in explicit_dests:
        pipeline_kwargs["tree_max_crown_top_height"] = float(args.tree_max_crown_top_height)
    if "tree_min_crown_bottom_height" in explicit_dests:
        pipeline_kwargs["tree_min_crown_bottom_height"] = float(
            args.tree_min_crown_bottom_height
        )
    if "building_count" in explicit_dests:
        pipeline_kwargs["building_count"] = args.building_count
    if "building_roof_type_percentages" in explicit_dests:
        pipeline_kwargs["building_roof_type_percentages"] = args.building_roof_type_percentages
    if "building_floor_min" in explicit_dests:
        pipeline_kwargs["building_floor_min"] = int(args.building_floor_min)
    if "building_floor_max" in explicit_dests:
        pipeline_kwargs["building_floor_max"] = int(args.building_floor_max)
    if "building_random_yaw" in explicit_dests:
        pipeline_kwargs["building_random_yaw"] = bool(args.building_random_yaw)
    if "structure_count" in explicit_dests:
        pipeline_kwargs["structure_count"] = args.structure_count
    if "structure_type_percentages" in explicit_dests:
        pipeline_kwargs["structure_type_percentages"] = args.structure_type_percentages
    if "vehicle_count" in explicit_dests:
        pipeline_kwargs["vehicle_count"] = args.vehicle_count
    if "vehicle_type_percentages" in explicit_dests:
        pipeline_kwargs["vehicle_type_percentages"] = args.vehicle_type_percentages
    if "artifacts_enabled" in explicit_dests:
        pipeline_kwargs["artifacts_enabled"] = bool(args.artifacts_enabled)
    if "artifact_global_intensity" in explicit_dests:
        pipeline_kwargs["artifact_global_intensity"] = float(args.artifact_global_intensity)
    if "artifact_point_fraction" in explicit_dests:
        pipeline_kwargs["artifact_point_fraction"] = float(args.artifact_point_fraction)
    if "shrub_count" in explicit_dests:
        pipeline_kwargs["shrub_count"] = args.shrub_count
    if "random_shrub_size" in explicit_dests:
        pipeline_kwargs["random_shrub_size"] = bool(args.random_shrub_size)
    if "shrub_max_diameter" in explicit_dests:
        pipeline_kwargs["shrub_max_diameter"] = float(args.shrub_max_diameter)
    if "shrub_max_top_height" in explicit_dests:
        pipeline_kwargs["shrub_max_top_height"] = float(args.shrub_max_top_height)
    if "shrub_min_bottom_height" in explicit_dests:
        pipeline_kwargs["shrub_min_bottom_height"] = float(args.shrub_min_bottom_height)
    if "grass_patch_count" in explicit_dests:
        pipeline_kwargs["grass_patch_count"] = args.grass_patch_count
    if "random_grass_patch_size" in explicit_dests:
        pipeline_kwargs["random_grass_patch_size"] = bool(args.random_grass_patch_size)
    if "grass_patch_max_size_x" in explicit_dests:
        pipeline_kwargs["grass_patch_max_size_x"] = float(args.grass_patch_max_size_x)
    if "grass_patch_max_size_y" in explicit_dests:
        pipeline_kwargs["grass_patch_max_size_y"] = float(args.grass_patch_max_size_y)
    if "grass_max_height" in explicit_dests:
        pipeline_kwargs["grass_max_height"] = float(args.grass_max_height)

    _run_pipeline(
        show_visualization=not args.no_visualize,
        save_csv=not args.no_csv,
        save_ply=not args.no_ply,
        **pipeline_kwargs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
