#!/usr/bin/env python3
import argparse
import colorsys
import os
import re
import struct
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import yaml
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_LINES,
    GL_DYNAMIC_DRAW,
    GL_FALSE,
    GL_FLOAT,
    GL_FRAGMENT_SHADER,
    GL_POINTS,
    GL_PROGRAM_POINT_SIZE,
    GL_STATIC_DRAW,
    GL_TRIANGLES,
    GL_VERTEX_SHADER,
    glBindBuffer,
    glBufferData,
    glClear,
    glClearColor,
    glDeleteBuffers,
    glDeleteProgram,
    glDisable,
    glDisableVertexAttribArray,
    glDrawArrays,
    glEnable,
    glEnableVertexAttribArray,
    glGenBuffers,
    glGetAttribLocation,
    glGetUniformLocation,
    glLineWidth,
    glUniform1f,
    glUniform3f,
    glUniformMatrix4fv,
    glUseProgram,
    glVertexAttribPointer,
    glViewport,
)
from OpenGL.GL.shaders import compileProgram, compileShader
from PyQt5.QtCore import QRectF, QSize, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QCursor, QIcon, QKeyEvent, QMouseEvent, QPainter, QSurfaceFormat, QWheelEvent
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QOpenGLWidget,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QSpinBox,
    QStyle,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from utils.tin_alg import TINMesh, TINParameters, export_mesh_to_ply, normalize_tin_parameters

DEFAULT_MAX_POINTS = 2_000_000
DEFAULT_POINT_SIZE = 3.0
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
ICONS_DIR = PROJECT_DIR / "assets" / "icons"
SETTINGS_PATH = PROJECT_DIR / "settings.yaml"
DEFAULT_VIEWPORT_BACKGROUND = (0.08, 0.09, 0.11)
DEFAULT_VIEWPORT_BACKGROUND_HEX = "#14171C"
COLOR_PRESETS: Tuple[Tuple[str, str], ...] = (
    ("Default Dark", DEFAULT_VIEWPORT_BACKGROUND_HEX),
    ("Black", "#000000"),
    ("Slate Blue", "#1E2E4A"),
    ("Deep Gray", "#2B2F36"),
    ("Light Gray", "#D4D8DE"),
    ("White", "#FFFFFF"),
)
DEFAULT_BOUNDING_BOX_COLOR_MODE = "random"
DEFAULT_BOUNDING_BOX_COLOR_HEX = "#FFB347"
DEFAULT_BOUNDING_BOX_LINE_WIDTH = 2.0
BOX_COLOR_PRESETS: Tuple[Tuple[str, str], ...] = (
    ("Amber", DEFAULT_BOUNDING_BOX_COLOR_HEX),
    ("Lime", "#8BD450"),
    ("Cyan", "#4ED8E6"),
    ("Sky Blue", "#4FA3FF"),
    ("Magenta", "#FF5CA8"),
    ("Red", "#FF5A52"),
    ("White", "#FFFFFF"),
    ("Black", "#000000"),
)
BOUNDING_BOX_COLOR_MODES = {"random", "single"}
NAMED_COLOR_PRESETS: Tuple[Tuple[str, str], ...] = COLOR_PRESETS + BOX_COLOR_PRESETS


@dataclass
class ProjectSettings:
    output_directory: str = "data"
    point_size: float = DEFAULT_POINT_SIZE
    viewport_background: str = DEFAULT_VIEWPORT_BACKGROUND_HEX
    bounding_box_color_mode: str = DEFAULT_BOUNDING_BOX_COLOR_MODE
    bounding_box_color: str = DEFAULT_BOUNDING_BOX_COLOR_HEX
    bounding_box_line_width: float = DEFAULT_BOUNDING_BOX_LINE_WIDTH
    bounding_box_show_id: bool = False


def _clamp_point_size(value: object) -> float:
    try:
        point_size = float(value)
    except (TypeError, ValueError):
        point_size = DEFAULT_POINT_SIZE
    return float(min(10.0, max(1.0, point_size)))


def _clamp_bounding_box_line_width(value: object) -> float:
    try:
        line_width = float(value)
    except (TypeError, ValueError):
        line_width = DEFAULT_BOUNDING_BOX_LINE_WIDTH
    return float(min(10.0, max(1.0, line_width)))


def _normalize_bounding_box_color_mode(value: object) -> str:
    mode = str(value).strip().lower()
    if mode in BOUNDING_BOX_COLOR_MODES:
        return mode
    return DEFAULT_BOUNDING_BOX_COLOR_MODE


def _coerce_bool_setting(value: object, default: bool = False) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(default)


def resolve_output_directory(path_value: str) -> Path:
    raw = str(path_value).strip() or "data"
    resolved = Path(raw).expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_DIR / resolved
    return resolved.resolve()


def display_output_directory(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_DIR).as_posix()
    except ValueError:
        return str(resolved)


def color_to_hex(rgb: Sequence[float]) -> str:
    rgb_arr = np.clip(np.asarray(rgb, dtype=np.float64).reshape(3), 0.0, 1.0)
    red, green, blue = np.rint(rgb_arr * 255.0).astype(np.uint8)
    return f"#{int(red):02X}{int(green):02X}{int(blue):02X}"


def parse_color_value(value: object) -> Tuple[float, float, float]:
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size != 3:
            raise ValueError("Color sequence must contain exactly 3 components.")
        if np.any(~np.isfinite(arr)):
            raise ValueError("Color values must be finite.")
        if float(np.max(arr)) > 1.0:
            if np.any(arr < 0.0) or np.any(arr > 255.0):
                raise ValueError("RGB integer values must be within 0..255.")
            arr = arr / 255.0
        if np.any(arr < 0.0) or np.any(arr > 1.0):
            raise ValueError("RGB float values must be within 0..1.")
        return tuple(float(component) for component in arr[:3])

    text = str(value).strip()
    if not text:
        raise ValueError("Color value must not be empty.")

    named_colors = {name.lower(): color for name, color in NAMED_COLOR_PRESETS}
    lowered = text.lower()
    if lowered in named_colors:
        text = named_colors[lowered]

    if text.startswith("#"):
        hex_value = text[1:]
        if len(hex_value) == 3:
            hex_value = "".join(ch * 2 for ch in hex_value)
        if len(hex_value) != 6 or any(ch not in "0123456789abcdefABCDEF" for ch in hex_value):
            raise ValueError("Hex colors must use #RGB or #RRGGBB format.")
        return tuple(int(hex_value[i:i + 2], 16) / 255.0 for i in range(0, 6, 2))

    if lowered.startswith("rgb(") and text.endswith(")"):
        text = text[text.find("(") + 1:-1]

    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise ValueError("Color must be a preset name, #RRGGBB, rgb(r,g,b), or r,g,b.")

    try:
        values = np.asarray([float(part) for part in parts], dtype=np.float64)
    except ValueError as exc:
        raise ValueError("Color components must be numeric.") from exc

    return parse_color_value(values)


def normalize_color_value(value: object) -> str:
    return color_to_hex(parse_color_value(value))


def load_project_settings() -> ProjectSettings:
    settings = ProjectSettings()
    if not SETTINGS_PATH.exists():
        return settings

    try:
        raw = yaml.safe_load(SETTINGS_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        return settings

    if not isinstance(raw, dict):
        return settings

    output_directory = str(raw.get("output_directory", settings.output_directory)).strip()
    if output_directory:
        settings.output_directory = output_directory

    settings.point_size = _clamp_point_size(raw.get("point_size", settings.point_size))

    try:
        settings.viewport_background = normalize_color_value(
            raw.get("viewport_background", settings.viewport_background)
        )
    except ValueError:
        settings.viewport_background = DEFAULT_VIEWPORT_BACKGROUND_HEX

    settings.bounding_box_color_mode = _normalize_bounding_box_color_mode(
        raw.get("bounding_box_color_mode", settings.bounding_box_color_mode)
    )
    try:
        settings.bounding_box_color = normalize_color_value(
            raw.get("bounding_box_color", settings.bounding_box_color)
        )
    except ValueError:
        settings.bounding_box_color = DEFAULT_BOUNDING_BOX_COLOR_HEX
    settings.bounding_box_line_width = _clamp_bounding_box_line_width(
        raw.get("bounding_box_line_width", settings.bounding_box_line_width)
    )
    settings.bounding_box_show_id = _coerce_bool_setting(
        raw.get("bounding_box_show_id", settings.bounding_box_show_id),
        default=settings.bounding_box_show_id,
    )

    return settings


def save_project_settings(settings: ProjectSettings) -> None:
    payload = {
        "output_directory": str(settings.output_directory).strip() or "data",
        "point_size": _clamp_point_size(settings.point_size),
        "viewport_background": normalize_color_value(settings.viewport_background),
        "bounding_box_color_mode": _normalize_bounding_box_color_mode(settings.bounding_box_color_mode),
        "bounding_box_color": normalize_color_value(settings.bounding_box_color),
        "bounding_box_line_width": _clamp_bounding_box_line_width(settings.bounding_box_line_width),
        "bounding_box_show_id": bool(settings.bounding_box_show_id),
    }
    SETTINGS_PATH.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


APP_SETTINGS = load_project_settings()


def ensure_data_dir() -> Path:
    output_dir = resolve_output_directory(APP_SETTINGS.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def normalize_field_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.strip().lower())


def normalize_vector(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm <= 1e-12:
        return v
    return v / norm


def _interpolate_color_ramp(
    values: np.ndarray,
    stops: Sequence[float],
    colors: Sequence[Sequence[float]],
) -> np.ndarray:
    result = np.empty((values.shape[0], 3), dtype=np.float64)
    stop_arr = np.asarray(stops, dtype=np.float64)
    color_arr = np.asarray(colors, dtype=np.float64)
    for idx, value in enumerate(values):
        if value <= stop_arr[0]:
            result[idx] = color_arr[0]
            continue
        if value >= stop_arr[-1]:
            result[idx] = color_arr[-1]
            continue
        upper = int(np.searchsorted(stop_arr, value, side="right"))
        lower = max(0, upper - 1)
        span = max(float(stop_arr[upper] - stop_arr[lower]), 1e-12)
        alpha = float((value - stop_arr[lower]) / span)
        result[idx] = (1.0 - alpha) * color_arr[lower] + alpha * color_arr[upper]
    return np.ascontiguousarray(np.clip(result, 0.0, 1.0), dtype=np.float32)


def apply_elevation_colormap(name: str, z_values: np.ndarray) -> np.ndarray:
    z_array = np.asarray(z_values, dtype=np.float64).reshape(-1)
    if z_array.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    z_min = float(np.min(z_array))
    z_max = float(np.max(z_array))
    if not np.isfinite(z_min) or not np.isfinite(z_max) or abs(z_max - z_min) <= 1e-12:
        normalized = np.full(z_array.shape[0], 0.5, dtype=np.float64)
    else:
        normalized = np.clip((z_array - z_min) / (z_max - z_min), 0.0, 1.0)

    colormap = str(name).strip().lower()
    if colormap == "grayscale":
        gray = normalized.astype(np.float32, copy=False)
        return np.repeat(gray.reshape(-1, 1), 3, axis=1)
    if colormap == "viridis":
        return _interpolate_color_ramp(
            normalized,
            (0.0, 0.25, 0.5, 0.75, 1.0),
            (
                (0.267, 0.005, 0.329),
                (0.283, 0.141, 0.458),
                (0.254, 0.265, 0.530),
                (0.207, 0.478, 0.553),
                (0.993, 0.906, 0.144),
            ),
        )
    if colormap == "plasma":
        return _interpolate_color_ramp(
            normalized,
            (0.0, 0.25, 0.5, 0.75, 1.0),
            (
                (0.050, 0.030, 0.528),
                (0.494, 0.012, 0.658),
                (0.798, 0.281, 0.469),
                (0.973, 0.585, 0.252),
                (0.940, 0.975, 0.131),
            ),
        )
    return _interpolate_color_ramp(
        normalized,
        (0.0, 0.18, 0.36, 0.62, 0.82, 1.0),
        (
            (0.082, 0.204, 0.118),
            (0.208, 0.396, 0.145),
            (0.455, 0.596, 0.263),
            (0.608, 0.486, 0.325),
            (0.722, 0.707, 0.612),
            (0.960, 0.955, 0.938),
        ),
    )


def perspective_matrix(fov_y_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / np.tan(np.radians(fov_y_deg) * 0.5)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def look_at_matrix(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = normalize_vector(target - eye)
    right = normalize_vector(np.cross(forward, up))
    if float(np.linalg.norm(right)) <= 1e-7:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    cam_up = normalize_vector(np.cross(right, forward))

    m = np.eye(4, dtype=np.float32)
    m[0, :3] = right
    m[1, :3] = cam_up
    m[2, :3] = -forward
    m[0, 3] = -float(np.dot(right, eye))
    m[1, 3] = -float(np.dot(cam_up, eye))
    m[2, 3] = float(np.dot(forward, eye))
    return m


def generate_distinct_palette(count: int) -> np.ndarray:
    if count <= 0:
        return np.zeros((0, 3), dtype=np.float32)

    palette = np.zeros((count, 3), dtype=np.float32)
    golden_ratio = 0.618033988749895
    for i in range(count):
        hue = (i * golden_ratio) % 1.0
        saturation = 0.68 + 0.2 * ((i % 3) / 2.0)
        value = 0.95
        palette[i] = colorsys.hsv_to_rgb(hue, saturation, value)
    return palette


BOX_EDGE_VERTEX_PAIRS: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def build_bounding_box_line_vertices(
    min_corner: Sequence[float],
    max_corner: Sequence[float],
) -> np.ndarray:
    min_xyz = np.asarray(min_corner, dtype=np.float32).reshape(3)
    max_xyz = np.asarray(max_corner, dtype=np.float32).reshape(3)
    corners = np.asarray(
        [
            [min_xyz[0], min_xyz[1], min_xyz[2]],
            [max_xyz[0], min_xyz[1], min_xyz[2]],
            [max_xyz[0], max_xyz[1], min_xyz[2]],
            [min_xyz[0], max_xyz[1], min_xyz[2]],
            [min_xyz[0], min_xyz[1], max_xyz[2]],
            [max_xyz[0], min_xyz[1], max_xyz[2]],
            [max_xyz[0], max_xyz[1], max_xyz[2]],
            [min_xyz[0], max_xyz[1], max_xyz[2]],
        ],
        dtype=np.float32,
    )
    vertices = np.empty((len(BOX_EDGE_VERTEX_PAIRS) * 2, 3), dtype=np.float32)
    for edge_index, (start_idx, end_idx) in enumerate(BOX_EDGE_VERTEX_PAIRS):
        target = edge_index * 2
        vertices[target] = corners[start_idx]
        vertices[target + 1] = corners[end_idx]
    return vertices


@dataclass
class PointCloudData:
    points: np.ndarray
    labels: Optional[np.ndarray] = None
    rgb: Optional[np.ndarray] = None
    file_path: str = ""
    original_count: int = 0
    _label_color_cache: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.points = np.ascontiguousarray(self.points, dtype=np.float32)
        if self.labels is not None:
            self.labels = np.ascontiguousarray(self.labels.astype(np.int32, copy=False))
        if self.rgb is not None:
            self.rgb = np.ascontiguousarray(self.rgb.astype(np.float32, copy=False))
        if self.original_count <= 0:
            self.original_count = int(self.points.shape[0])

    @property
    def loaded_count(self) -> int:
        return int(self.points.shape[0])

    @property
    def has_labels(self) -> bool:
        return self.labels is not None and self.labels.size > 0

    @property
    def has_rgb(self) -> bool:
        return self.rgb is not None and self.rgb.size > 0

    @property
    def center(self) -> np.ndarray:
        if self.points.size == 0:
            return np.zeros(3, dtype=np.float32)
        # Use float64 accumulation so large world coordinates do not skew the camera target.
        return np.mean(self.points, axis=0, dtype=np.float64).astype(np.float32)

    @property
    def radius(self) -> float:
        if self.points.size == 0:
            return 1.0
        mins = np.min(self.points, axis=0).astype(np.float64, copy=False)
        maxs = np.max(self.points, axis=0).astype(np.float64, copy=False)
        return max(1e-4, float(np.linalg.norm(maxs - mins) * 0.5))

    def subset(self, indices: np.ndarray) -> "PointCloudData":
        sub_points = np.ascontiguousarray(self.points[indices], dtype=np.float32)
        sub_labels = None
        sub_rgb = None
        if self.labels is not None:
            sub_labels = np.ascontiguousarray(self.labels[indices], dtype=np.int32)
        if self.rgb is not None:
            sub_rgb = np.ascontiguousarray(self.rgb[indices], dtype=np.float32)
        return PointCloudData(
            points=sub_points,
            labels=sub_labels,
            rgb=sub_rgb,
            file_path=self.file_path,
            original_count=self.original_count,
        )

    def label_colors(self) -> np.ndarray:
        if not self.has_labels:
            return np.zeros((self.loaded_count, 3), dtype=np.float32)
        if self._label_color_cache is not None and self._label_color_cache.shape[0] == self.loaded_count:
            return self._label_color_cache

        assert self.labels is not None
        unique_labels = np.unique(self.labels)
        palette = generate_distinct_palette(int(unique_labels.shape[0]))
        mapped_indices = np.searchsorted(unique_labels, self.labels)
        self._label_color_cache = np.ascontiguousarray(palette[mapped_indices], dtype=np.float32)
        return self._label_color_cache


@dataclass(frozen=True)
class ClusterBoundingBoxData:
    cluster_id: int
    point_count: int
    min_corner: Tuple[float, float, float]
    max_corner: Tuple[float, float, float]


@dataclass(frozen=True)
class DBSCANDialogParams:
    epsilon: float
    min_pts: int
    output_path: str


class PointCloudLoaderError(Exception):
    pass


class SplitByLabelDialog(QDialog):
    def __init__(
        self,
        default_prefix: str,
        default_directory: str,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Split Point Cloud by Label")
        self.setModal(True)

        self.prefix_edit = QLineEdit(self)
        self.prefix_edit.setText(default_prefix)
        self.prefix_edit.setPlaceholderText("file prefix")
        self.prefix_edit.textChanged.connect(self._update_ok_button_state)

        self.directory_edit = QLineEdit(self)
        self.directory_edit.setText(default_directory)
        self.directory_edit.textChanged.connect(self._update_ok_button_state)

        self.browse_button = QPushButton("Browse...", self)
        self.browse_button.clicked.connect(self._choose_directory)

        directory_row = QHBoxLayout()
        directory_row.addWidget(self.directory_edit)
        directory_row.addWidget(self.browse_button)

        form = QFormLayout()
        form.addRow("File prefix:", self.prefix_edit)
        form.addRow("Output directory:", directory_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.ok_button = buttons.button(QDialogButtonBox.Ok)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)
        self.setLayout(layout)

        self._update_ok_button_state()

    def _choose_directory(self) -> None:
        start_dir = self.output_directory() or str(ensure_data_dir())
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            start_dir,
        )
        if path:
            self.directory_edit.setText(path)

    def _update_ok_button_state(self) -> None:
        if self.ok_button is None:
            return
        self.ok_button.setEnabled(bool(self.file_prefix()) and bool(self.output_directory()))

    def file_prefix(self) -> str:
        return self.prefix_edit.text().strip()

    def output_directory(self) -> str:
        return self.directory_edit.text().strip()

    def accept(self) -> None:
        prefix = self.file_prefix()
        directory = self.output_directory()
        if not prefix:
            QMessageBox.warning(self, "Invalid Prefix", "Please enter a file prefix.")
            return
        if os.sep in prefix or (os.altsep and os.altsep in prefix):
            QMessageBox.warning(
                self,
                "Invalid Prefix",
                "The file prefix must not contain path separators.",
            )
            return
        if not directory:
            QMessageBox.warning(self, "Invalid Directory", "Please choose an output directory.")
            return
        if not Path(directory).is_dir():
            QMessageBox.warning(
                self,
                "Invalid Directory",
                "The selected output directory does not exist.",
            )
            return
        super().accept()


class DBSCANDialog(QDialog):
    def __init__(
        self,
        cloud_name: str,
        default_epsilon: float,
        default_min_pts: int,
        default_output_path: str,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("DBSCAN Settings")
        self.setModal(True)

        self.cloud_edit = QLineEdit(self)
        self.cloud_edit.setReadOnly(True)
        self.cloud_edit.setText(cloud_name)

        self.epsilon_spin = QDoubleSpinBox(self)
        self.epsilon_spin.setDecimals(6)
        self.epsilon_spin.setRange(0.000001, 1_000_000.0)
        self.epsilon_spin.setValue(max(0.000001, float(default_epsilon)))

        self.min_pts_spin = QSpinBox(self)
        self.min_pts_spin.setRange(1, 1_000_000_000)
        self.min_pts_spin.setValue(max(1, int(default_min_pts)))

        self.output_edit = QLineEdit(self)
        self.output_edit.setText(default_output_path)
        self.output_edit.textChanged.connect(self._update_ok_button_state)

        self.output_browse_button = QPushButton("Browse...", self)
        self.output_browse_button.clicked.connect(self._choose_output_file)

        output_row = QHBoxLayout()
        output_row.addWidget(self.output_edit)
        output_row.addWidget(self.output_browse_button)

        form = QFormLayout()
        form.addRow("Current cloud:", self.cloud_edit)
        form.addRow("epsilon:", self.epsilon_spin)
        form.addRow("MinPts:", self.min_pts_spin)
        form.addRow("Output YAML:", output_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.ok_button = buttons.button(QDialogButtonBox.Ok)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)
        self.setLayout(layout)

        self._update_ok_button_state()

    def _choose_output_file(self) -> None:
        start_path = self.output_path()
        if not start_path:
            start_path = str(ensure_data_dir() / "dbscan_clusters.yaml")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save DBSCAN Result",
            start_path,
            "YAML Files (*.yaml *.yml);;All Files (*)",
        )
        if not path:
            return
        out_path = Path(path)
        if out_path.suffix.lower() not in {".yaml", ".yml"}:
            out_path = out_path.with_suffix(".yaml")
        self.output_edit.setText(str(out_path))

    def _update_ok_button_state(self) -> None:
        if self.ok_button is None:
            return
        self.ok_button.setEnabled(bool(self.output_path()))

    def output_path(self) -> str:
        return self.output_edit.text().strip()

    def params(self) -> DBSCANDialogParams:
        output_path = Path(self.output_path())
        if output_path.suffix.lower() not in {".yaml", ".yml"}:
            output_path = output_path.with_suffix(".yaml")
        return DBSCANDialogParams(
            epsilon=float(self.epsilon_spin.value()),
            min_pts=int(self.min_pts_spin.value()),
            output_path=str(output_path),
        )

    def accept(self) -> None:
        params = self.params()
        if params.epsilon <= 0.0:
            QMessageBox.warning(self, "Invalid epsilon", "`epsilon` must be greater than 0.")
            return
        if params.min_pts < 1:
            QMessageBox.warning(self, "Invalid MinPts", "`MinPts` must be at least 1.")
            return
        if not params.output_path:
            QMessageBox.warning(self, "Invalid Output", "Please choose an output YAML file.")
            return

        output_path = Path(params.output_path)
        parent_dir = output_path.parent
        if str(parent_dir) and not parent_dir.exists():
            QMessageBox.warning(
                self,
                "Invalid Output",
                "The selected output directory does not exist.",
            )
            return

        super().accept()


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


RENDER_MODE_VALUES = ("wireframe", "solid", "shaded")
ELEVATION_COLORMAP_VALUES = ("terrain", "viridis", "plasma", "grayscale")


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
    """Run the TIN algorithm for an in-memory cloud and return mesh + GUI settings."""
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


class TINDialog(QDialog):
    """Collect TIN algorithm and viewport settings before mesh generation."""

    def __init__(
        self,
        cloud_name: str,
        point_count: int,
        default_params: TINCommandParams | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("TIN Settings")
        self.setModal(True)
        self.resize(620, 760)

        self._cloud_name = str(cloud_name).strip() or "<in-memory point cloud>"
        self._point_count = max(0, int(point_count))
        self._default_params = normalize_command_params(default_params)

        self.cloud_edit = QLineEdit(self)
        self.cloud_edit.setReadOnly(True)
        self.cloud_edit.setText(self._cloud_name)

        self.point_count_edit = QLineEdit(self)
        self.point_count_edit.setReadOnly(True)
        self.point_count_edit.setText(f"{self._point_count:,}")

        self.coincidence_tolerance_spin = self._make_float_spin(0.0, 1_000_000.0, 6)
        self.coincidence_tolerance_spin.setValue(self._default_params.algorithm.coincidence_tolerance)

        self.duplicate_handling_combo = QComboBox(self)
        self._populate_combo(
            self.duplicate_handling_combo,
            (
                ("Keep first", "keep_first"),
                ("Average duplicates", "average"),
                ("Remove duplicates", "remove"),
            ),
            self._default_params.algorithm.duplicate_handling,
        )

        self.max_edge_length_spin = self._make_float_spin(0.0, 1_000_000.0, 3)
        self.max_edge_length_spin.setValue(self._default_params.algorithm.max_edge_length)
        self.max_edge_length_spin.setSpecialValueText("Disabled")

        self.min_angle_spin = self._make_float_spin(0.0, 179.9, 1)
        self.min_angle_spin.setValue(self._default_params.algorithm.min_angle)

        self.max_angle_spin = self._make_float_spin(0.1, 180.0, 1)
        self.max_angle_spin.setValue(self._default_params.algorithm.max_angle)

        self.outlier_filter_spin = self._make_float_spin(0.0, 100.0, 2)
        self.outlier_filter_spin.setValue(self._default_params.algorithm.outlier_filter)
        self.outlier_filter_spin.setSpecialValueText("Disabled")

        self.boundary_type_combo = QComboBox(self)
        self._populate_combo(
            self.boundary_type_combo,
            (
                ("Convex hull", "convex_hull"),
                ("Concave hull", "concave_hull"),
                ("Custom polygon", "custom"),
            ),
            self._default_params.algorithm.boundary_type,
        )

        self.alpha_spin = self._make_float_spin(0.001, 1_000_000.0, 3)
        self.alpha_spin.setValue(self._default_params.algorithm.alpha)

        self.custom_boundary_edit = QLineEdit(self)
        self.custom_boundary_edit.setText(self._default_params.custom_boundary_path)
        self.custom_boundary_edit.setPlaceholderText("Ordered polygon vertices in TXT/PLY")
        self.custom_boundary_browse = QPushButton("Browse...", self)
        self.custom_boundary_browse.clicked.connect(self._choose_custom_boundary)

        self.interpolation_combo = QComboBox(self)
        self._populate_combo(
            self.interpolation_combo,
            (
                ("Linear", "linear"),
                ("Natural neighbor (approx.)", "natural_neighbor"),
            ),
            self._default_params.algorithm.interpolation_method,
        )

        self.mesh_resolution_spin = self._make_float_spin(0.0, 1_000_000.0, 3)
        self.mesh_resolution_spin.setValue(self._default_params.algorithm.mesh_resolution)
        self.mesh_resolution_spin.setSpecialValueText("Disabled")

        self.render_mode_combo = QComboBox(self)
        self._populate_combo(
            self.render_mode_combo,
            (
                ("Wireframe", "wireframe"),
                ("Solid", "solid"),
                ("Shaded", "shaded"),
            ),
            self._default_params.visual.render_mode,
        )

        self.colormap_combo = QComboBox(self)
        self._populate_combo(
            self.colormap_combo,
            (
                ("Terrain", "terrain"),
                ("Viridis", "viridis"),
                ("Plasma", "plasma"),
                ("Grayscale", "grayscale"),
            ),
            self._default_params.visual.elevation_colormap,
        )

        self.smooth_normals_check = QCheckBox("Smooth normals", self)
        self.smooth_normals_check.setChecked(self._default_params.visual.smooth_normals)

        self.max_points_spin = QSpinBox(self)
        self.max_points_spin.setRange(3, 1_000_000_000)
        self.max_points_spin.setValue(self._default_params.algorithm.max_points)

        self.spatial_index_combo = QComboBox(self)
        self._populate_combo(
            self.spatial_index_combo,
            (
                ("KD-tree", "kdtree"),
                ("R-tree", "rtree"),
            ),
            self._default_params.algorithm.spatial_index,
        )

        self.preview_label = QLabel(self)
        self.preview_label.setWordWrap(True)
        self.preview_label.setStyleSheet("padding: 8px; background: #F4F6F9; border: 1px solid #D6DCE5;")

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.ok_button = buttons.button(QDialogButtonBox.Ok)

        content = QWidget(self)
        content_layout = QVBoxLayout(content)
        content_layout.addWidget(self._build_source_group())
        content_layout.addWidget(self._build_triangulation_group())
        content_layout.addWidget(self._build_filter_group())
        content_layout.addWidget(self._build_boundary_group())
        content_layout.addWidget(self._build_interpolation_group())
        content_layout.addWidget(self._build_display_group())
        content_layout.addWidget(self._build_performance_group())
        content_layout.addWidget(self.preview_label)
        content_layout.addStretch(1)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)

        layout = QVBoxLayout(self)
        layout.addWidget(scroll)
        layout.addWidget(buttons)
        self.setLayout(layout)

        self._wire_preview_updates()
        self._update_dependent_fields()
        self._update_preview()

    def _make_float_spin(self, minimum: float, maximum: float, decimals: int) -> QDoubleSpinBox:
        spin = QDoubleSpinBox(self)
        spin.setDecimals(decimals)
        spin.setRange(minimum, maximum)
        spin.setSingleStep(max(10 ** (-decimals), 0.001))
        return spin

    def _populate_combo(self, combo: QComboBox, options, selected_value: str) -> None:
        current_index = 0
        for index, (label, value) in enumerate(options):
            combo.addItem(label, value)
            if value == selected_value:
                current_index = index
        combo.setCurrentIndex(current_index)

    def _build_source_group(self) -> QGroupBox:
        group = QGroupBox("Source", self)
        form = QFormLayout(group)
        form.addRow("Current cloud:", self.cloud_edit)
        form.addRow("Input points:", self.point_count_edit)
        return group

    def _build_triangulation_group(self) -> QGroupBox:
        group = QGroupBox("Delaunay Triangulation", self)
        form = QFormLayout(group)
        form.addRow("Coincidence tolerance:", self.coincidence_tolerance_spin)
        form.addRow("Duplicate handling:", self.duplicate_handling_combo)
        return group

    def _build_filter_group(self) -> QGroupBox:
        group = QGroupBox("Point Filtering", self)
        form = QFormLayout(group)
        form.addRow("Max edge length:", self.max_edge_length_spin)
        form.addRow("Min angle:", self.min_angle_spin)
        form.addRow("Max angle:", self.max_angle_spin)
        form.addRow("Outlier threshold (std):", self.outlier_filter_spin)
        return group

    def _build_boundary_group(self) -> QGroupBox:
        group = QGroupBox("Boundary Conditions", self)
        form = QFormLayout(group)
        self.custom_boundary_row = QWidget(self)
        row_layout = QHBoxLayout(self.custom_boundary_row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(self.custom_boundary_edit)
        row_layout.addWidget(self.custom_boundary_browse)

        form.addRow("Boundary type:", self.boundary_type_combo)
        form.addRow("Alpha:", self.alpha_spin)
        form.addRow("Custom boundary:", self.custom_boundary_row)
        return group

    def _build_interpolation_group(self) -> QGroupBox:
        group = QGroupBox("Interpolation", self)
        form = QFormLayout(group)
        form.addRow("Method:", self.interpolation_combo)
        form.addRow("Mesh resolution:", self.mesh_resolution_spin)
        return group

    def _build_display_group(self) -> QGroupBox:
        group = QGroupBox("Display", self)
        form = QFormLayout(group)
        form.addRow("Render mode:", self.render_mode_combo)
        form.addRow("Elevation colormap:", self.colormap_combo)
        form.addRow("Normals:", self.smooth_normals_check)
        return group

    def _build_performance_group(self) -> QGroupBox:
        group = QGroupBox("Performance", self)
        form = QFormLayout(group)
        form.addRow("Max points:", self.max_points_spin)
        form.addRow("Spatial index:", self.spatial_index_combo)
        return group

    def _wire_preview_updates(self) -> None:
        widgets = (
            self.coincidence_tolerance_spin,
            self.duplicate_handling_combo,
            self.max_edge_length_spin,
            self.min_angle_spin,
            self.max_angle_spin,
            self.outlier_filter_spin,
            self.boundary_type_combo,
            self.alpha_spin,
            self.custom_boundary_edit,
            self.interpolation_combo,
            self.mesh_resolution_spin,
            self.render_mode_combo,
            self.colormap_combo,
            self.smooth_normals_check,
            self.max_points_spin,
            self.spatial_index_combo,
        )
        for widget in widgets:
            if isinstance(widget, QLineEdit):
                widget.textChanged.connect(self._update_preview)
                widget.textChanged.connect(self._update_dependent_fields)
            elif isinstance(widget, QCheckBox):
                widget.toggled.connect(self._update_preview)
                widget.toggled.connect(self._update_dependent_fields)
            elif isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(self._update_preview)
                widget.currentIndexChanged.connect(self._update_dependent_fields)
            else:
                widget.valueChanged.connect(self._update_preview)
                widget.valueChanged.connect(self._update_dependent_fields)

    def _selected_value(self, combo: QComboBox) -> str:
        return str(combo.currentData()).strip().lower()

    def _choose_custom_boundary(self) -> None:
        start_path = self.custom_boundary_path() or self._cloud_name
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Custom Boundary",
            start_path,
            "Point/Boundary Files (*.txt *.ply);;All Files (*)",
        )
        if path:
            self.custom_boundary_edit.setText(path)

    def custom_boundary_path(self) -> str:
        return self.custom_boundary_edit.text().strip()

    def params(self) -> TINCommandParams:
        algorithm = TINParameters(
            coincidence_tolerance=float(self.coincidence_tolerance_spin.value()),
            duplicate_handling=self._selected_value(self.duplicate_handling_combo),
            max_edge_length=float(self.max_edge_length_spin.value()),
            min_angle=float(self.min_angle_spin.value()),
            max_angle=float(self.max_angle_spin.value()),
            outlier_filter=float(self.outlier_filter_spin.value()),
            boundary_type=self._selected_value(self.boundary_type_combo),
            alpha=float(self.alpha_spin.value()),
            interpolation_method=self._selected_value(self.interpolation_combo),
            mesh_resolution=float(self.mesh_resolution_spin.value()),
            max_points=int(self.max_points_spin.value()),
            spatial_index=self._selected_value(self.spatial_index_combo),
        )
        visual = TINVisualSettings(
            render_mode=self._selected_value(self.render_mode_combo),
            elevation_colormap=self._selected_value(self.colormap_combo),
            smooth_normals=bool(self.smooth_normals_check.isChecked()),
        )
        return normalize_command_params(
            TINCommandParams(
                algorithm=algorithm,
                visual=visual,
                custom_boundary_path=self.custom_boundary_path(),
            )
        )

    def _update_dependent_fields(self) -> None:
        boundary_type = self._selected_value(self.boundary_type_combo)
        use_alpha = boundary_type == "concave_hull"
        use_custom_boundary = boundary_type == "custom"
        self.alpha_spin.setEnabled(use_alpha)
        self.custom_boundary_edit.setEnabled(use_custom_boundary)
        self.custom_boundary_browse.setEnabled(use_custom_boundary)

        render_mode = self._selected_value(self.render_mode_combo)
        allow_normal_controls = render_mode in {"solid", "shaded"}
        self.smooth_normals_check.setEnabled(allow_normal_controls)

    def _update_preview(self) -> None:
        boundary_type = self._selected_value(self.boundary_type_combo)
        render_mode = self._selected_value(self.render_mode_combo)
        max_points = int(self.max_points_spin.value())
        processed = min(self._point_count, max_points)

        if boundary_type == "concave_hull":
            boundary_summary = f"Concave hull (alpha={self.alpha_spin.value():.3f})"
        elif boundary_type == "custom":
            boundary_summary = "Custom polygon"
            if self.custom_boundary_path():
                boundary_summary += f" from {Path(self.custom_boundary_path()).name}"
        else:
            boundary_summary = "Convex hull"

        if float(self.mesh_resolution_spin.value()) > 0.0:
            resampling = (
                f"Enabled at {self.mesh_resolution_spin.value():.3f} using "
                f"{self._selected_value(self.interpolation_combo)}"
            )
        else:
            resampling = "Disabled"

        normals = "smoothed" if self.smooth_normals_check.isChecked() else "flat"
        if render_mode == "wireframe":
            normals = "not used"

        preview = (
            f"Cloud: {self._cloud_name}\n"
            f"Input points: {self._point_count:,}\n"
            f"Up to {processed:,} points will reach the triangulator after performance limits.\n"
            f"Boundary: {boundary_summary}.\n"
            f"Resampling: {resampling}.\n"
            f"Display: {render_mode} with {self._selected_value(self.colormap_combo)} colormap and {normals} normals."
        )
        self.preview_label.setText(preview)

    def accept(self) -> None:
        boundary_type = self._selected_value(self.boundary_type_combo)
        if self.min_angle_spin.value() >= self.max_angle_spin.value():
            QMessageBox.warning(
                self,
                "Invalid Angles",
                "Minimum triangle angle must be smaller than maximum triangle angle.",
            )
            return

        if boundary_type == "custom":
            path = self.custom_boundary_path()
            if not path:
                QMessageBox.warning(
                    self,
                    "Missing Boundary",
                    "Choose a custom boundary TXT/PLY file when custom boundary mode is enabled.",
                )
                return
            if not Path(path).is_file():
                QMessageBox.warning(
                    self,
                    "Missing Boundary",
                    "The selected custom boundary file does not exist.",
                )
                return

        try:
            self.params()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Invalid Settings", str(exc))
            return

        super().accept()


@dataclass
class PlyProperty:
    kind: str  # "scalar" or "list"
    name: str
    dtype: str
    count_dtype: Optional[str] = None


@dataclass
class PlyElement:
    name: str
    count: int
    properties: List[PlyProperty]


class PointCloudLoader:
    LABEL_FIELD_NAMES = [
        "label",
        "class",
        "cls",
        "category",
        "semantic",
        "segment",
        "segmentation",
        "id",
    ]
    PLY_TO_STRUCT = {
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
    PLY_TO_NUMPY = {
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

    @classmethod
    def load(cls, path: str, max_points: int = DEFAULT_MAX_POINTS) -> Tuple[PointCloudData, bool]:
        if max_points <= 0:
            raise PointCloudLoaderError("Point threshold must be a positive integer.")
        if not os.path.isfile(path):
            raise PointCloudLoaderError(f"File not found: {path}")

        ext = os.path.splitext(path)[1].lower()
        if ext == ".txt":
            cloud = cls._load_txt(path)
        elif ext == ".ply":
            cloud = cls._load_ply(path)
        else:
            raise PointCloudLoaderError(f"Unsupported file extension '{ext}'. Use TXT or PLY.")

        cloud.file_path = path
        cloud.original_count = cloud.loaded_count

        if cloud.loaded_count > max_points:
            rng = np.random.default_rng()
            indices = rng.choice(cloud.loaded_count, size=max_points, replace=False)
            cloud = cloud.subset(indices)
            cloud.file_path = path
            return cloud, True

        return cloud, False

    @classmethod
    def _load_txt(cls, path: str) -> PointCloudData:
        rows: List[List[float]] = []
        header_tokens: Optional[List[str]] = None
        ncols: Optional[int] = None

        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for line_number, raw_line in enumerate(fh, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue

                tokens = [token for token in re.split(r"[,\s]+", line) if token]
                if not tokens:
                    continue

                numeric_values: List[float] = []
                numeric_ok = True
                for token in tokens:
                    try:
                        numeric_values.append(float(token))
                    except ValueError:
                        numeric_ok = False
                        break

                if not numeric_ok:
                    if header_tokens is None and not rows:
                        header_tokens = [normalize_field_name(token) for token in tokens]
                        continue
                    raise PointCloudLoaderError(
                        f"Unable to parse numeric data in TXT at line {line_number}: '{raw_line.rstrip()}'"
                    )

                if ncols is None:
                    ncols = len(numeric_values)
                elif len(numeric_values) != ncols:
                    raise PointCloudLoaderError(
                        f"Inconsistent column count in TXT at line {line_number}: "
                        f"expected {ncols}, found {len(numeric_values)}."
                    )

                rows.append(numeric_values)

        if not rows:
            raise PointCloudLoaderError("TXT file does not contain point records.")

        matrix = np.asarray(rows, dtype=np.float32)
        if matrix.shape[1] < 3:
            raise PointCloudLoaderError("TXT data must contain at least 3 columns for XYZ coordinates.")

        return cls._extract_channels_from_matrix(matrix, header_tokens)

    @classmethod
    def _load_ply(cls, path: str) -> PointCloudData:
        with open(path, "rb") as fh:
            first = fh.readline().decode("ascii", errors="ignore").strip()
            if first.lower() != "ply":
                raise PointCloudLoaderError("Invalid PLY file: missing 'ply' magic header.")

            format_type: Optional[str] = None
            elements: List[PlyElement] = []
            current_element: Optional[PlyElement] = None

            while True:
                line = fh.readline()
                if not line:
                    raise PointCloudLoaderError("Unexpected end of file while reading PLY header.")
                text = line.decode("ascii", errors="ignore").strip()
                if text == "end_header":
                    break
                if not text:
                    continue

                parts = text.split()
                keyword = parts[0].lower()
                if keyword in {"comment", "obj_info"}:
                    continue
                if keyword == "format":
                    if len(parts) < 2:
                        raise PointCloudLoaderError("Malformed PLY format declaration.")
                    format_type = parts[1].lower()
                elif keyword == "element":
                    if len(parts) != 3:
                        raise PointCloudLoaderError(f"Malformed PLY element declaration: '{text}'")
                    try:
                        count = int(parts[2])
                    except ValueError as exc:
                        raise PointCloudLoaderError(f"Invalid element count in PLY header: '{parts[2]}'") from exc
                    current_element = PlyElement(name=parts[1], count=count, properties=[])
                    elements.append(current_element)
                elif keyword == "property":
                    if current_element is None:
                        raise PointCloudLoaderError("PLY property declared before any element.")
                    if len(parts) >= 5 and parts[1].lower() == "list":
                        current_element.properties.append(
                            PlyProperty(
                                kind="list",
                                count_dtype=parts[2].lower(),
                                dtype=parts[3].lower(),
                                name=parts[4],
                            )
                        )
                    elif len(parts) == 3:
                        current_element.properties.append(
                            PlyProperty(kind="scalar", dtype=parts[1].lower(), name=parts[2])
                        )
                    else:
                        raise PointCloudLoaderError(f"Malformed PLY property declaration: '{text}'")

            if format_type is None:
                raise PointCloudLoaderError("PLY header missing format declaration.")
            if format_type not in {"ascii", "binary_little_endian", "binary_big_endian"}:
                raise PointCloudLoaderError(f"Unsupported PLY format '{format_type}'.")

            vertex_columns: Optional[Dict[str, np.ndarray]] = None
            if format_type == "ascii":
                for element in elements:
                    keep = element.name.lower() == "vertex"
                    columns = cls._read_ascii_element(fh, element, keep=keep)
                    if keep:
                        vertex_columns = columns
            else:
                endian = "<" if format_type == "binary_little_endian" else ">"
                for element in elements:
                    keep = element.name.lower() == "vertex"
                    columns = cls._read_binary_element(fh, element, endian=endian, keep=keep)
                    if keep:
                        vertex_columns = columns

            if vertex_columns is None:
                raise PointCloudLoaderError("PLY file does not contain a 'vertex' element.")
            return cls._extract_channels_from_ply_columns(vertex_columns)

    @classmethod
    def _read_ascii_element(
        cls, fh, element: PlyElement, keep: bool
    ) -> Optional[Dict[str, np.ndarray]]:
        scalar_data: Dict[str, List[float]] = {}
        if keep:
            for prop in element.properties:
                if prop.kind == "scalar":
                    scalar_data[prop.name] = []

        for row_index in range(element.count):
            line = fh.readline()
            if not line:
                raise PointCloudLoaderError(
                    f"Unexpected end of PLY file while reading ASCII element '{element.name}'."
                )
            tokens = line.decode("ascii", errors="ignore").strip().split()
            token_idx = 0
            for prop in element.properties:
                if prop.kind == "scalar":
                    if token_idx >= len(tokens):
                        raise PointCloudLoaderError(
                            f"Malformed ASCII PLY row {row_index} in element '{element.name}'."
                        )
                    if keep:
                        try:
                            scalar_data[prop.name].append(float(tokens[token_idx]))
                        except ValueError as exc:
                            raise PointCloudLoaderError(
                                f"Invalid numeric value in ASCII PLY row {row_index}."
                            ) from exc
                    token_idx += 1
                else:
                    if token_idx >= len(tokens):
                        raise PointCloudLoaderError(
                            f"Malformed list property in ASCII PLY row {row_index}."
                        )
                    try:
                        list_count = int(float(tokens[token_idx]))
                    except ValueError as exc:
                        raise PointCloudLoaderError(
                            f"Invalid list count in ASCII PLY row {row_index}."
                        ) from exc
                    token_idx += 1 + list_count
                    if token_idx > len(tokens):
                        raise PointCloudLoaderError(
                            f"List property overflow in ASCII PLY row {row_index}."
                        )

        if not keep:
            return None
        return {name: np.asarray(values, dtype=np.float64) for name, values in scalar_data.items()}

    @classmethod
    def _read_binary_element(
        cls, fh, element: PlyElement, endian: str, keep: bool
    ) -> Optional[Dict[str, np.ndarray]]:
        scalar_only = all(prop.kind == "scalar" for prop in element.properties)

        if scalar_only:
            dtype_fields = []
            for prop in element.properties:
                np_type = cls.PLY_TO_NUMPY.get(prop.dtype)
                if np_type is None:
                    raise PointCloudLoaderError(f"Unsupported PLY type '{prop.dtype}' in element '{element.name}'.")
                dtype_fields.append((prop.name, np.dtype(np_type).newbyteorder(endian)))
            structured_dtype = np.dtype(dtype_fields)
            array = np.fromfile(fh, dtype=structured_dtype, count=element.count)
            if array.size != element.count:
                raise PointCloudLoaderError(
                    f"Unexpected EOF while reading binary PLY element '{element.name}'."
                )
            if not keep:
                return None
            return {name: np.asarray(array[name], dtype=np.float64) for name in array.dtype.names or []}

        scalar_names = [prop.name for prop in element.properties if prop.kind == "scalar"]
        scalar_data = (
            {name: np.empty(element.count, dtype=np.float64) for name in scalar_names}
            if keep
            else {}
        )

        for row_index in range(element.count):
            for prop in element.properties:
                if prop.kind == "scalar":
                    value = cls._read_binary_value(fh, endian, prop.dtype)
                    if keep:
                        scalar_data[prop.name][row_index] = float(value)
                else:
                    assert prop.count_dtype is not None
                    list_count = int(cls._read_binary_value(fh, endian, prop.count_dtype))
                    item_size = cls._type_size(prop.dtype)
                    skip_size = list_count * item_size
                    if skip_size > 0:
                        skipped = fh.read(skip_size)
                        if len(skipped) != skip_size:
                            raise PointCloudLoaderError(
                                f"Unexpected EOF while skipping binary list data in element '{element.name}'."
                            )

        return scalar_data if keep else None

    @classmethod
    def _read_binary_value(cls, fh, endian: str, ply_type: str):
        fmt_char = cls.PLY_TO_STRUCT.get(ply_type)
        if fmt_char is None:
            raise PointCloudLoaderError(f"Unsupported PLY scalar type '{ply_type}'.")
        fmt = endian + fmt_char
        size = struct.calcsize(fmt)
        raw = fh.read(size)
        if len(raw) != size:
            raise PointCloudLoaderError("Unexpected EOF in binary PLY scalar read.")
        return struct.unpack(fmt, raw)[0]

    @classmethod
    def _type_size(cls, ply_type: str) -> int:
        fmt_char = cls.PLY_TO_STRUCT.get(ply_type)
        if fmt_char is None:
            raise PointCloudLoaderError(f"Unsupported PLY type '{ply_type}'.")
        return struct.calcsize(fmt_char)

    @classmethod
    def _extract_channels_from_ply_columns(cls, columns: Dict[str, np.ndarray]) -> PointCloudData:
        normalized_to_original = {normalize_field_name(name): name for name in columns.keys()}

        def pick(*keys: str) -> Optional[str]:
            for key in keys:
                hit = normalized_to_original.get(key)
                if hit is not None:
                    return hit
            return None

        x_name = pick("x")
        y_name = pick("y")
        z_name = pick("z")
        if x_name is None or y_name is None or z_name is None:
            raise PointCloudLoaderError("PLY vertex data does not provide required x/y/z fields.")

        points = np.stack(
            [columns[x_name], columns[y_name], columns[z_name]],
            axis=1,
        ).astype(np.float32, copy=False)

        r_name = pick("red", "r")
        g_name = pick("green", "g")
        b_name = pick("blue", "b")
        rgb = None
        if r_name and g_name and b_name:
            rgb = np.stack(
                [columns[r_name], columns[g_name], columns[b_name]],
                axis=1,
            ).astype(np.float32, copy=False)
            rgb = cls._normalize_rgb(rgb)

        label_name = None
        for candidate in cls.LABEL_FIELD_NAMES:
            label_name = normalized_to_original.get(candidate)
            if label_name is not None:
                break
        labels = None
        if label_name is not None:
            labels = np.rint(columns[label_name]).astype(np.int32)

        return PointCloudData(points=points, labels=labels, rgb=rgb)

    @classmethod
    def _extract_channels_from_matrix(
        cls, matrix: np.ndarray, header_tokens: Optional[List[str]] = None
    ) -> PointCloudData:
        col_count = matrix.shape[1]
        if col_count < 3:
            raise PointCloudLoaderError("Point data must contain at least 3 columns for XYZ.")

        xyz_idx = [0, 1, 2]
        rgb_idx: Optional[List[int]] = None
        label_idx: Optional[int] = None

        if header_tokens is not None and len(header_tokens) == col_count:
            indexed = {normalize_field_name(name): idx for idx, name in enumerate(header_tokens)}
            if {"x", "y", "z"}.issubset(indexed):
                xyz_idx = [indexed["x"], indexed["y"], indexed["z"]]
            if {"red", "green", "blue"}.issubset(indexed):
                rgb_idx = [indexed["red"], indexed["green"], indexed["blue"]]
            elif {"r", "g", "b"}.issubset(indexed):
                rgb_idx = [indexed["r"], indexed["g"], indexed["b"]]
            for candidate in cls.LABEL_FIELD_NAMES:
                if candidate in indexed:
                    label_idx = indexed[candidate]
                    break
        else:
            if col_count >= 6 and cls._looks_like_rgb(matrix[:, 3:6]):
                rgb_idx = [3, 4, 5]
                if col_count >= 7:
                    label_idx = 6
            elif col_count >= 4:
                label_idx = 3

        points = matrix[:, xyz_idx].astype(np.float32, copy=False)
        rgb = None
        if rgb_idx is not None:
            rgb = matrix[:, rgb_idx].astype(np.float32, copy=False)
            rgb = cls._normalize_rgb(rgb)

        labels = None
        if label_idx is not None and 0 <= label_idx < col_count:
            labels = np.rint(matrix[:, label_idx]).astype(np.int32)

        return PointCloudData(points=points, labels=labels, rgb=rgb)

    @staticmethod
    def _looks_like_rgb(rgb_values: np.ndarray) -> bool:
        if rgb_values.shape[1] != 3:
            return False
        if not np.all(np.isfinite(rgb_values)):
            return False
        min_val = float(np.min(rgb_values))
        max_val = float(np.max(rgb_values))
        if min_val < -1e-4 or max_val > 255.0 + 1e-3:
            return False
        if max_val <= 1.0 + 1e-4:
            return True
        near_int = np.abs(rgb_values - np.round(rgb_values)) < 1e-3
        return float(np.mean(near_int)) > 0.98

    @staticmethod
    def _normalize_rgb(rgb: np.ndarray) -> np.ndarray:
        rgb = np.asarray(rgb, dtype=np.float32)
        if rgb.size == 0:
            return rgb
        max_value = float(np.max(rgb))
        if max_value > 1.0:
            rgb = rgb / 255.0
        return np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=False)


def export_point_cloud_data_to_ply(cloud: PointCloudData, output_path: Path) -> None:
    points = np.ascontiguousarray(cloud.points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Point cloud points must have shape (N, 3).")

    rgb_uint8 = None
    if cloud.rgb is not None and cloud.rgb.size > 0:
        rgb = np.ascontiguousarray(cloud.rgb, dtype=np.float32)
        if rgb.shape != (points.shape[0], 3):
            raise ValueError("Point cloud RGB data must have shape (N, 3).")
        rgb_uint8 = np.clip(np.round(rgb * 255.0), 0.0, 255.0).astype(np.uint8)

    labels = None
    if cloud.labels is not None and cloud.labels.size > 0:
        labels = np.ascontiguousarray(cloud.labels, dtype=np.int32).reshape(-1)
        if labels.shape[0] != points.shape[0]:
            raise ValueError("Point cloud labels must contain exactly one value per point.")

    columns: List[np.ndarray] = [points[:, 0], points[:, 1], points[:, 2]]
    fmt = ["%.6f", "%.6f", "%.6f"]
    property_lines = [
        "property float x",
        "property float y",
        "property float z",
    ]

    if rgb_uint8 is not None:
        columns.extend([rgb_uint8[:, 0], rgb_uint8[:, 1], rgb_uint8[:, 2]])
        fmt.extend(["%d", "%d", "%d"])
        property_lines.extend(
            [
                "property uchar red",
                "property uchar green",
                "property uchar blue",
            ]
        )

    if labels is not None:
        columns.append(labels)
        fmt.append("%d")
        property_lines.append("property int label")

    matrix = np.column_stack(columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        fh.write("ply\n")
        fh.write("format ascii 1.0\n")
        fh.write(f"element vertex {points.shape[0]}\n")
        for line in property_lines:
            fh.write(f"{line}\n")
        fh.write("end_header\n")
        np.savetxt(fh, matrix, fmt=fmt)


FALLBACK_SYNTHETIC_CLASS_NAMES: Dict[int, str] = {
    0: "Natural surface",
    1: "Artificial surface",
    2: "Low vegetation",
    3: "High vegetation",
    4: "Buildings",
    5: "Structures",
    6: "Vehicles",
    7: "Artifacts",
}
FALLBACK_CLASS_PERCENTAGES: Tuple[float, ...] = (38.0, 13.0, 16.0, 14.0, 10.0, 4.0, 3.0, 2.0)
FALLBACK_ARTIFICIAL_SURFACE_TYPES: Tuple[str, ...] = (
    "road_network",
    "sidewalk",
    "parking_lot",
    "building_front_area",
    "industrial_concrete_pad",
    "platform",
)
FALLBACK_ARTIFICIAL_SURFACE_TYPE_NAMES: Dict[str, str] = {
    "road_network": "Road network",
    "sidewalk": "Sidewalks along buildings",
    "parking_lot": "Parking lots",
    "building_front_area": "Areas in front of buildings",
    "industrial_concrete_pad": "Industrial concrete pads",
    "platform": "Platforms / station aprons",
}
FALLBACK_ARTIFICIAL_SURFACE_TYPE_PERCENTAGES: Tuple[float, ...] = (
    30.0,
    22.0,
    16.0,
    10.0,
    12.0,
    10.0,
)
FALLBACK_VEHICLE_TYPES: Tuple[str, ...] = ("car", "truck", "bus")
FALLBACK_VEHICLE_TYPE_NAMES: Dict[str, str] = {
    "car": "Passenger car",
    "truck": "Truck",
    "bus": "Bus",
}
FALLBACK_VEHICLE_TYPE_PERCENTAGES: Tuple[float, ...] = (72.0, 18.0, 10.0)
FALLBACK_LOW_VEG_DEFAULTS: Dict[str, float] = {
    "shrub_max_diameter": 2.6,
    "shrub_max_top_height": 1.8,
    "shrub_min_bottom_height": 0.12,
    "grass_patch_max_size_x": 3.8,
    "grass_patch_max_size_y": 3.4,
    "grass_max_height": 0.65,
}
FALLBACK_BUILDING_ROOF_TYPES: Tuple[str, ...] = (
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
FALLBACK_BUILDING_ROOF_TYPE_NAMES: Dict[str, str] = {
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
FALLBACK_BUILDING_ROOF_TYPE_PERCENTAGES: Tuple[float, ...] = (
    10.0,
    18.0,
    14.0,
    8.0,
    10.0,
    20.0,
    6.0,
    8.0,
    6.0,
)
FALLBACK_BUILDING_DEFAULTS: Dict[str, int | bool] = {
    "building_floor_min": 2,
    "building_floor_max": 9,
    "building_random_yaw": True,
}
FALLBACK_STRUCTURE_TYPES: Tuple[str, ...] = (
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
FALLBACK_STRUCTURE_TYPE_NAMES: Dict[str, str] = {
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
FALLBACK_STRUCTURE_TYPE_PERCENTAGES: Tuple[float, ...] = (
    12.0,
    7.0,
    6.0,
    6.0,
    6.0,
    4.0,
    5.0,
    7.0,
    6.0,
    5.0,
    3.0,
    5.0,
    4.0,
    3.0,
    3.0,
    2.0,
    2.0,
    2.0,
    3.0,
    2.0,
    3.0,
    4.0,
)
FALLBACK_TREE_CROWN_TYPES: Tuple[str, ...] = (
    "spherical",
    "pyramidal",
    "spreading",
    "weeping",
    "columnar",
    "umbrella",
)
FALLBACK_TREE_CROWN_TYPE_NAMES: Dict[str, str] = {
    "spherical": "Spherical",
    "pyramidal": "Pyramidal",
    "spreading": "Spreading",
    "weeping": "Weeping",
    "columnar": "Columnar",
    "umbrella": "Umbrella",
}
FALLBACK_TREE_CROWN_TYPE_PERCENTAGES: Tuple[float, ...] = (28.0, 18.0, 20.0, 10.0, 12.0, 12.0)
FALLBACK_HIGH_VEG_DEFAULTS: Dict[str, float] = {
    "tree_max_crown_diameter": 5.2,
    "tree_max_crown_top_height": 9.5,
    "tree_min_crown_bottom_height": 1.3,
}


@dataclass
class SyntheticGenerationParams:
    total_points: int = 100_000
    area_width: float = 240.0
    area_length: float = 220.0
    terrain_relief: float = 1.0
    seed: int = 12
    randomize_object_counts: bool = True
    custom_class_distribution: bool = False
    class_percentages: Tuple[float, ...] = FALLBACK_CLASS_PERCENTAGES
    custom_artificial_surface_count: bool = False
    artificial_surface_count: int = 9
    custom_artificial_surface_type_distribution: bool = False
    artificial_surface_type_percentages: Tuple[float, ...] = (
        FALLBACK_ARTIFICIAL_SURFACE_TYPE_PERCENTAGES
    )
    custom_building_count: bool = False
    building_count: int = 14
    custom_building_roof_type_distribution: bool = False
    building_roof_type_percentages: Tuple[float, ...] = FALLBACK_BUILDING_ROOF_TYPE_PERCENTAGES
    building_floor_min: int = int(FALLBACK_BUILDING_DEFAULTS["building_floor_min"])
    building_floor_max: int = int(FALLBACK_BUILDING_DEFAULTS["building_floor_max"])
    building_random_yaw: bool = bool(FALLBACK_BUILDING_DEFAULTS["building_random_yaw"])
    custom_structure_count: bool = False
    structure_count: int = 10
    custom_structure_type_distribution: bool = False
    structure_type_percentages: Tuple[float, ...] = FALLBACK_STRUCTURE_TYPE_PERCENTAGES
    custom_tree_count: bool = False
    tree_count: int = 70
    custom_tree_crown_type_distribution: bool = False
    tree_crown_type_percentages: Tuple[float, ...] = FALLBACK_TREE_CROWN_TYPE_PERCENTAGES
    random_tree_crown_size: bool = True
    tree_max_crown_diameter: float = FALLBACK_HIGH_VEG_DEFAULTS["tree_max_crown_diameter"]
    tree_max_crown_top_height: float = FALLBACK_HIGH_VEG_DEFAULTS["tree_max_crown_top_height"]
    tree_min_crown_bottom_height: float = FALLBACK_HIGH_VEG_DEFAULTS[
        "tree_min_crown_bottom_height"
    ]
    custom_vehicle_count: bool = False
    vehicle_count: int = 24
    custom_vehicle_type_distribution: bool = False
    vehicle_type_percentages: Tuple[float, ...] = FALLBACK_VEHICLE_TYPE_PERCENTAGES
    custom_shrub_count: bool = False
    shrub_count: int = 24
    random_shrub_size: bool = True
    shrub_max_diameter: float = FALLBACK_LOW_VEG_DEFAULTS["shrub_max_diameter"]
    shrub_max_top_height: float = FALLBACK_LOW_VEG_DEFAULTS["shrub_max_top_height"]
    shrub_min_bottom_height: float = FALLBACK_LOW_VEG_DEFAULTS["shrub_min_bottom_height"]
    custom_grass_patch_count: bool = False
    grass_patch_count: int = 18
    random_grass_patch_size: bool = True
    grass_patch_max_size_x: float = FALLBACK_LOW_VEG_DEFAULTS["grass_patch_max_size_x"]
    grass_patch_max_size_y: float = FALLBACK_LOW_VEG_DEFAULTS["grass_patch_max_size_y"]
    grass_max_height: float = FALLBACK_LOW_VEG_DEFAULTS["grass_max_height"]


class SyntheticGenerationDialog(QDialog):
    def __init__(
        self,
        default_params: Optional[SyntheticGenerationParams] = None,
        class_names: Optional[Dict[int, str]] = None,
        default_class_percentages: Optional[Sequence[float]] = None,
        artificial_surface_type_names: Optional[Dict[str, str]] = None,
        default_artificial_surface_type_percentages: Optional[Sequence[float]] = None,
        building_roof_type_names: Optional[Dict[str, str]] = None,
        default_building_roof_type_percentages: Optional[Sequence[float]] = None,
        building_defaults: Optional[Dict[str, int | bool]] = None,
        structure_type_names: Optional[Dict[str, str]] = None,
        default_structure_type_percentages: Optional[Sequence[float]] = None,
        tree_crown_type_names: Optional[Dict[str, str]] = None,
        default_tree_crown_type_percentages: Optional[Sequence[float]] = None,
        high_veg_defaults: Optional[Dict[str, float]] = None,
        vehicle_type_names: Optional[Dict[str, str]] = None,
        default_vehicle_type_percentages: Optional[Sequence[float]] = None,
        low_veg_defaults: Optional[Dict[str, float]] = None,
        synthetic_module=None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Generate Synthetic Point Cloud")
        self.setModal(True)
        self._synthetic_module = synthetic_module

        params = default_params or SyntheticGenerationParams()
        self._class_names = self._normalize_class_names(class_names)
        self._class_ids = sorted(self._class_names)
        fallback_percentages = self._normalize_percentages(
            values=default_class_percentages,
            class_count=len(self._class_ids),
            fallback=FALLBACK_CLASS_PERCENTAGES,
        )
        initial_percentages = self._normalize_percentages(
            values=params.class_percentages,
            class_count=len(self._class_ids),
            fallback=fallback_percentages,
        )
        self._artificial_surface_type_names = self._normalize_artificial_surface_type_names(
            artificial_surface_type_names
        )
        self._artificial_surface_types = list(FALLBACK_ARTIFICIAL_SURFACE_TYPES)
        fallback_artificial_surface_percentages = self._normalize_percentages(
            values=default_artificial_surface_type_percentages,
            class_count=len(self._artificial_surface_types),
            fallback=FALLBACK_ARTIFICIAL_SURFACE_TYPE_PERCENTAGES,
        )
        initial_artificial_surface_percentages = self._normalize_percentages(
            values=params.artificial_surface_type_percentages,
            class_count=len(self._artificial_surface_types),
            fallback=fallback_artificial_surface_percentages,
        )
        self._building_roof_type_names = self._normalize_building_roof_type_names(
            building_roof_type_names
        )
        self._building_roof_types = list(FALLBACK_BUILDING_ROOF_TYPES)
        self._building_defaults = self._normalize_building_defaults(building_defaults)
        fallback_building_roof_percentages = self._normalize_percentages(
            values=default_building_roof_type_percentages,
            class_count=len(self._building_roof_types),
            fallback=FALLBACK_BUILDING_ROOF_TYPE_PERCENTAGES,
        )
        initial_building_roof_percentages = self._normalize_percentages(
            values=params.building_roof_type_percentages,
            class_count=len(self._building_roof_types),
            fallback=fallback_building_roof_percentages,
        )
        self._structure_type_names = self._normalize_structure_type_names(structure_type_names)
        self._structure_types = list(FALLBACK_STRUCTURE_TYPES)
        fallback_structure_percentages = self._normalize_percentages(
            values=default_structure_type_percentages,
            class_count=len(self._structure_types),
            fallback=FALLBACK_STRUCTURE_TYPE_PERCENTAGES,
        )
        initial_structure_percentages = self._normalize_percentages(
            values=params.structure_type_percentages,
            class_count=len(self._structure_types),
            fallback=fallback_structure_percentages,
        )
        self._tree_crown_type_names = self._normalize_tree_crown_type_names(
            tree_crown_type_names
        )
        self._tree_crown_types = list(FALLBACK_TREE_CROWN_TYPES)
        self._high_veg_defaults = self._normalize_high_veg_defaults(high_veg_defaults)
        fallback_tree_crown_percentages = self._normalize_percentages(
            values=default_tree_crown_type_percentages,
            class_count=len(self._tree_crown_types),
            fallback=FALLBACK_TREE_CROWN_TYPE_PERCENTAGES,
        )
        initial_tree_crown_percentages = self._normalize_percentages(
            values=params.tree_crown_type_percentages,
            class_count=len(self._tree_crown_types),
            fallback=fallback_tree_crown_percentages,
        )
        self._vehicle_type_names = self._normalize_vehicle_type_names(vehicle_type_names)
        self._vehicle_types = list(FALLBACK_VEHICLE_TYPES)
        self._low_veg_defaults = self._normalize_low_veg_defaults(low_veg_defaults)
        fallback_vehicle_percentages = self._normalize_percentages(
            values=default_vehicle_type_percentages,
            class_count=len(self._vehicle_types),
            fallback=FALLBACK_VEHICLE_TYPE_PERCENTAGES,
        )
        initial_vehicle_percentages = self._normalize_percentages(
            values=params.vehicle_type_percentages,
            class_count=len(self._vehicle_types),
            fallback=fallback_vehicle_percentages,
        )
        default_floor_min = int(self._building_defaults["building_floor_min"])
        default_floor_max = int(self._building_defaults["building_floor_max"])
        initial_building_floor_min = max(1, int(params.building_floor_min))
        initial_building_floor_max = max(initial_building_floor_min, int(params.building_floor_max))
        if initial_building_floor_min <= 0:
            initial_building_floor_min = max(1, default_floor_min)
        if initial_building_floor_max < initial_building_floor_min:
            initial_building_floor_max = max(initial_building_floor_min, default_floor_max)
        initial_building_random_yaw = bool(params.building_random_yaw)
        initial_tree_max_crown_diameter = max(
            0.05,
            float(
                params.tree_max_crown_diameter
                if params.tree_max_crown_diameter > 0.0
                else self._high_veg_defaults["tree_max_crown_diameter"]
            ),
        )
        initial_tree_max_crown_top_height = max(
            0.05,
            float(
                params.tree_max_crown_top_height
                if params.tree_max_crown_top_height > 0.0
                else self._high_veg_defaults["tree_max_crown_top_height"]
            ),
        )
        initial_tree_min_crown_bottom_height = max(
            0.0,
            float(
                params.tree_min_crown_bottom_height
                if params.tree_min_crown_bottom_height >= 0.0
                else self._high_veg_defaults["tree_min_crown_bottom_height"]
            ),
        )
        initial_shrub_max_diameter = max(
            0.05,
            float(
                params.shrub_max_diameter
                if params.shrub_max_diameter > 0.0
                else self._low_veg_defaults["shrub_max_diameter"]
            ),
        )
        initial_shrub_max_top_height = max(
            0.05,
            float(
                params.shrub_max_top_height
                if params.shrub_max_top_height > 0.0
                else self._low_veg_defaults["shrub_max_top_height"]
            ),
        )
        initial_shrub_min_bottom_height = max(
            0.0,
            float(
                params.shrub_min_bottom_height
                if params.shrub_min_bottom_height >= 0.0
                else self._low_veg_defaults["shrub_min_bottom_height"]
            ),
        )
        initial_grass_patch_max_size_x = max(
            0.05,
            float(
                params.grass_patch_max_size_x
                if params.grass_patch_max_size_x > 0.0
                else self._low_veg_defaults["grass_patch_max_size_x"]
            ),
        )
        initial_grass_patch_max_size_y = max(
            0.05,
            float(
                params.grass_patch_max_size_y
                if params.grass_patch_max_size_y > 0.0
                else self._low_veg_defaults["grass_patch_max_size_y"]
            ),
        )
        initial_grass_max_height = max(
            0.05,
            float(
                params.grass_max_height
                if params.grass_max_height > 0.0
                else self._low_veg_defaults["grass_max_height"]
            ),
        )

        self.total_points_spin = QSpinBox(self)
        self.total_points_spin.setRange(1_000, 50_000_000)
        self.total_points_spin.setSingleStep(10_000)
        self.total_points_spin.setValue(int(params.total_points))

        self.area_width_spin = QDoubleSpinBox(self)
        self.area_width_spin.setRange(10.0, 20_000.0)
        self.area_width_spin.setDecimals(1)
        self.area_width_spin.setSingleStep(10.0)
        self.area_width_spin.setValue(float(params.area_width))
        self.area_width_spin.setSuffix(" m")

        self.area_length_spin = QDoubleSpinBox(self)
        self.area_length_spin.setRange(10.0, 20_000.0)
        self.area_length_spin.setDecimals(1)
        self.area_length_spin.setSingleStep(10.0)
        self.area_length_spin.setValue(float(params.area_length))
        self.area_length_spin.setSuffix(" m")

        self.terrain_relief_spin = QDoubleSpinBox(self)
        self.terrain_relief_spin.setRange(0.0, 1.0)
        self.terrain_relief_spin.setDecimals(2)
        self.terrain_relief_spin.setSingleStep(0.05)
        self.terrain_relief_spin.setValue(float(params.terrain_relief))

        self.seed_spin = QSpinBox(self)
        self.seed_spin.setRange(0, 2_147_483_647)
        self.seed_spin.setValue(int(params.seed))

        self.random_counts_check = QCheckBox("Randomize object counts", self)
        self.random_counts_check.setChecked(bool(params.randomize_object_counts))

        self.custom_distribution_check = QCheckBox("Use custom class distribution (%)", self)
        self.custom_distribution_check.setChecked(bool(params.custom_class_distribution))
        self.custom_distribution_check.toggled.connect(self._update_class_distribution_state)

        self.class_percentage_spins: Dict[int, QDoubleSpinBox] = {}
        for idx, class_id in enumerate(self._class_ids):
            spin = QDoubleSpinBox(self)
            spin.setRange(0.0, 100.0)
            spin.setDecimals(2)
            spin.setSingleStep(0.5)
            spin.setSuffix(" %")
            spin.setValue(float(initial_percentages[idx]))
            spin.valueChanged.connect(self._update_class_distribution_summary)
            self.class_percentage_spins[class_id] = spin

        self.class_distribution_sum_label = QLabel(self)
        self.custom_artificial_surface_count_check = QCheckBox(
            "Use custom artificial surface count",
            self,
        )
        self.custom_artificial_surface_count_check.setChecked(
            bool(params.custom_artificial_surface_count)
        )
        self.custom_artificial_surface_count_check.toggled.connect(
            self._update_artificial_surface_settings_state
        )

        self.artificial_surface_count_spin = QSpinBox(self)
        self.artificial_surface_count_spin.setRange(1, 1_000_000)
        self.artificial_surface_count_spin.setSingleStep(1)
        self.artificial_surface_count_spin.setValue(max(1, int(params.artificial_surface_count)))

        self.custom_artificial_surface_type_check = QCheckBox(
            "Use custom artificial surface type distribution (%)",
            self,
        )
        self.custom_artificial_surface_type_check.setChecked(
            bool(params.custom_artificial_surface_type_distribution)
        )
        self.custom_artificial_surface_type_check.toggled.connect(
            self._update_artificial_surface_settings_state
        )

        self.artificial_surface_type_percentage_spins: Dict[str, QDoubleSpinBox] = {}
        for idx, surface_type in enumerate(self._artificial_surface_types):
            spin = QDoubleSpinBox(self)
            spin.setRange(0.0, 100.0)
            spin.setDecimals(2)
            spin.setSingleStep(0.5)
            spin.setSuffix(" %")
            spin.setValue(float(initial_artificial_surface_percentages[idx]))
            spin.valueChanged.connect(self._update_artificial_surface_distribution_summary)
            self.artificial_surface_type_percentage_spins[surface_type] = spin

        self.artificial_surface_distribution_sum_label = QLabel(self)
        self.custom_building_count_check = QCheckBox("Use custom building count", self)
        self.custom_building_count_check.setChecked(bool(params.custom_building_count))
        self.custom_building_count_check.toggled.connect(self._update_building_state)

        self.building_count_spin = QSpinBox(self)
        self.building_count_spin.setRange(1, 1_000_000)
        self.building_count_spin.setSingleStep(1)
        self.building_count_spin.setValue(max(1, int(params.building_count)))

        self.custom_building_roof_type_check = QCheckBox(
            "Use custom building roof type distribution (%)",
            self,
        )
        self.custom_building_roof_type_check.setChecked(
            bool(params.custom_building_roof_type_distribution)
        )
        self.custom_building_roof_type_check.toggled.connect(self._update_building_state)

        self.building_roof_type_percentage_spins: Dict[str, QDoubleSpinBox] = {}
        for idx, roof_type in enumerate(self._building_roof_types):
            spin = QDoubleSpinBox(self)
            spin.setRange(0.0, 100.0)
            spin.setDecimals(2)
            spin.setSingleStep(0.5)
            spin.setSuffix(" %")
            spin.setValue(float(initial_building_roof_percentages[idx]))
            spin.valueChanged.connect(self._update_building_distribution_summary)
            self.building_roof_type_percentage_spins[roof_type] = spin

        self.building_roof_distribution_sum_label = QLabel(self)
        self.building_floor_min_spin = QSpinBox(self)
        self.building_floor_min_spin.setRange(1, 120)
        self.building_floor_min_spin.setSingleStep(1)
        self.building_floor_min_spin.setValue(initial_building_floor_min)
        self.building_floor_min_spin.valueChanged.connect(self._update_building_state)

        self.building_floor_max_spin = QSpinBox(self)
        self.building_floor_max_spin.setRange(1, 120)
        self.building_floor_max_spin.setSingleStep(1)
        self.building_floor_max_spin.setValue(initial_building_floor_max)
        self.building_floor_max_spin.valueChanged.connect(self._update_building_state)

        self.building_random_yaw_check = QCheckBox("Arbitrary random Z rotation", self)
        self.building_random_yaw_check.setChecked(initial_building_random_yaw)

        self.building_validation_label = QLabel(self)
        self.custom_structure_count_check = QCheckBox("Use custom structure count", self)
        self.custom_structure_count_check.setChecked(bool(params.custom_structure_count))
        self.custom_structure_count_check.toggled.connect(self._update_structure_settings_state)

        self.structure_count_spin = QSpinBox(self)
        self.structure_count_spin.setRange(1, 1_000_000)
        self.structure_count_spin.setSingleStep(1)
        self.structure_count_spin.setValue(max(1, int(params.structure_count)))

        self.custom_structure_type_check = QCheckBox(
            "Use custom structure type distribution (%)",
            self,
        )
        self.custom_structure_type_check.setChecked(bool(params.custom_structure_type_distribution))
        self.custom_structure_type_check.toggled.connect(self._update_structure_settings_state)

        self.structure_type_percentage_spins: Dict[str, QDoubleSpinBox] = {}
        for idx, structure_type in enumerate(self._structure_types):
            spin = QDoubleSpinBox(self)
            spin.setRange(0.0, 100.0)
            spin.setDecimals(2)
            spin.setSingleStep(0.5)
            spin.setSuffix(" %")
            spin.setValue(float(initial_structure_percentages[idx]))
            spin.valueChanged.connect(self._update_structure_distribution_summary)
            self.structure_type_percentage_spins[structure_type] = spin

        self.structure_distribution_sum_label = QLabel(self)
        self.custom_tree_count_check = QCheckBox("Use custom tree count", self)
        self.custom_tree_count_check.setChecked(bool(params.custom_tree_count))
        self.custom_tree_count_check.toggled.connect(self._update_high_vegetation_state)

        self.tree_count_spin = QSpinBox(self)
        self.tree_count_spin.setRange(1, 1_000_000)
        self.tree_count_spin.setSingleStep(1)
        self.tree_count_spin.setValue(max(1, int(params.tree_count)))

        self.custom_tree_crown_type_check = QCheckBox(
            "Use custom tree crown type distribution (%)",
            self,
        )
        self.custom_tree_crown_type_check.setChecked(bool(params.custom_tree_crown_type_distribution))
        self.custom_tree_crown_type_check.toggled.connect(self._update_high_vegetation_state)

        self.tree_crown_type_percentage_spins: Dict[str, QDoubleSpinBox] = {}
        for idx, crown_type in enumerate(self._tree_crown_types):
            spin = QDoubleSpinBox(self)
            spin.setRange(0.0, 100.0)
            spin.setDecimals(2)
            spin.setSingleStep(0.5)
            spin.setSuffix(" %")
            spin.setValue(float(initial_tree_crown_percentages[idx]))
            spin.valueChanged.connect(self._update_tree_crown_distribution_summary)
            self.tree_crown_type_percentage_spins[crown_type] = spin

        self.tree_crown_distribution_sum_label = QLabel(self)
        self.random_tree_crown_size_check = QCheckBox("Random tree crown size", self)
        self.random_tree_crown_size_check.setChecked(bool(params.random_tree_crown_size))
        self.random_tree_crown_size_check.toggled.connect(self._update_high_vegetation_state)

        self.tree_max_crown_diameter_spin = QDoubleSpinBox(self)
        self.tree_max_crown_diameter_spin.setRange(0.05, 60.0)
        self.tree_max_crown_diameter_spin.setDecimals(2)
        self.tree_max_crown_diameter_spin.setSingleStep(0.1)
        self.tree_max_crown_diameter_spin.setValue(initial_tree_max_crown_diameter)
        self.tree_max_crown_diameter_spin.setSuffix(" m")
        self.tree_max_crown_diameter_spin.valueChanged.connect(self._update_high_vegetation_state)

        self.tree_max_crown_top_height_spin = QDoubleSpinBox(self)
        self.tree_max_crown_top_height_spin.setRange(0.05, 80.0)
        self.tree_max_crown_top_height_spin.setDecimals(2)
        self.tree_max_crown_top_height_spin.setSingleStep(0.1)
        self.tree_max_crown_top_height_spin.setValue(initial_tree_max_crown_top_height)
        self.tree_max_crown_top_height_spin.setSuffix(" m")
        self.tree_max_crown_top_height_spin.valueChanged.connect(
            self._update_high_vegetation_state
        )

        self.tree_min_crown_bottom_height_spin = QDoubleSpinBox(self)
        self.tree_min_crown_bottom_height_spin.setRange(0.0, 80.0)
        self.tree_min_crown_bottom_height_spin.setDecimals(2)
        self.tree_min_crown_bottom_height_spin.setSingleStep(0.1)
        self.tree_min_crown_bottom_height_spin.setValue(initial_tree_min_crown_bottom_height)
        self.tree_min_crown_bottom_height_spin.setSuffix(" m")
        self.tree_min_crown_bottom_height_spin.valueChanged.connect(
            self._update_high_vegetation_state
        )

        self.high_vegetation_validation_label = QLabel(self)
        self.custom_vehicle_count_check = QCheckBox("Use custom vehicle count", self)
        self.custom_vehicle_count_check.setChecked(bool(params.custom_vehicle_count))
        self.custom_vehicle_count_check.toggled.connect(self._update_vehicle_settings_state)

        self.vehicle_count_spin = QSpinBox(self)
        self.vehicle_count_spin.setRange(1, 1_000_000)
        self.vehicle_count_spin.setSingleStep(1)
        self.vehicle_count_spin.setValue(max(1, int(params.vehicle_count)))

        self.custom_vehicle_type_check = QCheckBox("Use custom vehicle type distribution (%)", self)
        self.custom_vehicle_type_check.setChecked(bool(params.custom_vehicle_type_distribution))
        self.custom_vehicle_type_check.toggled.connect(self._update_vehicle_settings_state)

        self.vehicle_type_percentage_spins: Dict[str, QDoubleSpinBox] = {}
        for idx, vehicle_type in enumerate(self._vehicle_types):
            spin = QDoubleSpinBox(self)
            spin.setRange(0.0, 100.0)
            spin.setDecimals(2)
            spin.setSingleStep(0.5)
            spin.setSuffix(" %")
            spin.setValue(float(initial_vehicle_percentages[idx]))
            spin.valueChanged.connect(self._update_vehicle_distribution_summary)
            self.vehicle_type_percentage_spins[vehicle_type] = spin

        self.vehicle_distribution_sum_label = QLabel(self)
        self.custom_shrub_count_check = QCheckBox("Use custom shrub count", self)
        self.custom_shrub_count_check.setChecked(bool(params.custom_shrub_count))
        self.custom_shrub_count_check.toggled.connect(self._update_low_vegetation_state)

        self.shrub_count_spin = QSpinBox(self)
        self.shrub_count_spin.setRange(1, 1_000_000)
        self.shrub_count_spin.setSingleStep(1)
        self.shrub_count_spin.setValue(max(1, int(params.shrub_count)))

        self.random_shrub_size_check = QCheckBox("Random shrub size", self)
        self.random_shrub_size_check.setChecked(bool(params.random_shrub_size))
        self.random_shrub_size_check.toggled.connect(self._update_low_vegetation_state)

        self.shrub_max_diameter_spin = QDoubleSpinBox(self)
        self.shrub_max_diameter_spin.setRange(0.05, 20.0)
        self.shrub_max_diameter_spin.setDecimals(2)
        self.shrub_max_diameter_spin.setSingleStep(0.1)
        self.shrub_max_diameter_spin.setValue(initial_shrub_max_diameter)
        self.shrub_max_diameter_spin.setSuffix(" m")
        self.shrub_max_diameter_spin.valueChanged.connect(self._update_low_vegetation_state)

        self.shrub_max_top_height_spin = QDoubleSpinBox(self)
        self.shrub_max_top_height_spin.setRange(0.05, 20.0)
        self.shrub_max_top_height_spin.setDecimals(2)
        self.shrub_max_top_height_spin.setSingleStep(0.05)
        self.shrub_max_top_height_spin.setValue(initial_shrub_max_top_height)
        self.shrub_max_top_height_spin.setSuffix(" m")
        self.shrub_max_top_height_spin.valueChanged.connect(self._update_low_vegetation_state)

        self.shrub_min_bottom_height_spin = QDoubleSpinBox(self)
        self.shrub_min_bottom_height_spin.setRange(0.0, 20.0)
        self.shrub_min_bottom_height_spin.setDecimals(2)
        self.shrub_min_bottom_height_spin.setSingleStep(0.05)
        self.shrub_min_bottom_height_spin.setValue(initial_shrub_min_bottom_height)
        self.shrub_min_bottom_height_spin.setSuffix(" m")
        self.shrub_min_bottom_height_spin.valueChanged.connect(self._update_low_vegetation_state)

        self.custom_grass_patch_count_check = QCheckBox("Use custom grass patch count", self)
        self.custom_grass_patch_count_check.setChecked(bool(params.custom_grass_patch_count))
        self.custom_grass_patch_count_check.toggled.connect(self._update_low_vegetation_state)

        self.grass_patch_count_spin = QSpinBox(self)
        self.grass_patch_count_spin.setRange(1, 1_000_000)
        self.grass_patch_count_spin.setSingleStep(1)
        self.grass_patch_count_spin.setValue(max(1, int(params.grass_patch_count)))

        self.random_grass_patch_size_check = QCheckBox("Random grass patch size", self)
        self.random_grass_patch_size_check.setChecked(bool(params.random_grass_patch_size))
        self.random_grass_patch_size_check.toggled.connect(self._update_low_vegetation_state)

        self.grass_patch_max_size_x_spin = QDoubleSpinBox(self)
        self.grass_patch_max_size_x_spin.setRange(0.05, 50.0)
        self.grass_patch_max_size_x_spin.setDecimals(2)
        self.grass_patch_max_size_x_spin.setSingleStep(0.1)
        self.grass_patch_max_size_x_spin.setValue(initial_grass_patch_max_size_x)
        self.grass_patch_max_size_x_spin.setSuffix(" m")
        self.grass_patch_max_size_x_spin.valueChanged.connect(self._update_low_vegetation_state)

        self.grass_patch_max_size_y_spin = QDoubleSpinBox(self)
        self.grass_patch_max_size_y_spin.setRange(0.05, 50.0)
        self.grass_patch_max_size_y_spin.setDecimals(2)
        self.grass_patch_max_size_y_spin.setSingleStep(0.1)
        self.grass_patch_max_size_y_spin.setValue(initial_grass_patch_max_size_y)
        self.grass_patch_max_size_y_spin.setSuffix(" m")
        self.grass_patch_max_size_y_spin.valueChanged.connect(self._update_low_vegetation_state)

        self.grass_max_height_spin = QDoubleSpinBox(self)
        self.grass_max_height_spin.setRange(0.05, 10.0)
        self.grass_max_height_spin.setDecimals(2)
        self.grass_max_height_spin.setSingleStep(0.05)
        self.grass_max_height_spin.setValue(initial_grass_max_height)
        self.grass_max_height_spin.setSuffix(" m")
        self.grass_max_height_spin.valueChanged.connect(self._update_low_vegetation_state)

        self.low_vegetation_validation_label = QLabel(self)

        form_general = QFormLayout()
        form_general.addRow("Total points:", self.total_points_spin)
        form_general.addRow("Area width:", self.area_width_spin)
        form_general.addRow("Area length:", self.area_length_spin)
        form_general.addRow("Terrain relief [0..1]:", self.terrain_relief_spin)
        form_general.addRow("Random seed:", self.seed_spin)
        form_general.addRow("", self.random_counts_check)
        form_general.addRow("", self.custom_distribution_check)
        for class_id in self._class_ids:
            form_general.addRow(
                f"Class {class_id} ({self._class_names[class_id]}):",
                self.class_percentage_spins[class_id],
            )
        form_general.addRow("Class sum:", self.class_distribution_sum_label)

        form_artificial_surface = QFormLayout()
        form_artificial_surface.addRow("", self.custom_artificial_surface_count_check)
        form_artificial_surface.addRow(
            "Artificial surface objects:",
            self.artificial_surface_count_spin,
        )
        form_artificial_surface.addRow("", self.custom_artificial_surface_type_check)
        for surface_type in self._artificial_surface_types:
            form_artificial_surface.addRow(
                f"{self._artificial_surface_type_names[surface_type]}:",
                self.artificial_surface_type_percentage_spins[surface_type],
            )
        form_artificial_surface.addRow(
            "Type sum:",
            self.artificial_surface_distribution_sum_label,
        )

        form_building = QFormLayout()
        form_building.addRow("", self.custom_building_count_check)
        form_building.addRow("Building instances:", self.building_count_spin)
        form_building.addRow("", self.custom_building_roof_type_check)
        for roof_type in self._building_roof_types:
            form_building.addRow(
                f"{self._building_roof_type_names[roof_type]}:",
                self.building_roof_type_percentage_spins[roof_type],
            )
        form_building.addRow("Type sum:", self.building_roof_distribution_sum_label)
        form_building.addRow("Min building floors:", self.building_floor_min_spin)
        form_building.addRow("Max building floors:", self.building_floor_max_spin)
        form_building.addRow("", self.building_random_yaw_check)
        form_building.addRow("Validation:", self.building_validation_label)

        form_structure = QFormLayout()
        form_structure.addRow("", self.custom_structure_count_check)
        form_structure.addRow("Structure instances:", self.structure_count_spin)
        form_structure.addRow("", self.custom_structure_type_check)
        for structure_type in self._structure_types:
            form_structure.addRow(
                f"{self._structure_type_names[structure_type]}:",
                self.structure_type_percentage_spins[structure_type],
            )
        form_structure.addRow("Type sum:", self.structure_distribution_sum_label)

        form_high_veg = QFormLayout()
        form_high_veg.addRow("", self.custom_tree_count_check)
        form_high_veg.addRow("Tree instances:", self.tree_count_spin)
        form_high_veg.addRow("", self.custom_tree_crown_type_check)
        for crown_type in self._tree_crown_types:
            form_high_veg.addRow(
                f"{self._tree_crown_type_names[crown_type]}:",
                self.tree_crown_type_percentage_spins[crown_type],
            )
        form_high_veg.addRow("Type sum:", self.tree_crown_distribution_sum_label)
        form_high_veg.addRow("", self.random_tree_crown_size_check)
        form_high_veg.addRow("Max tree crown diameter:", self.tree_max_crown_diameter_spin)
        form_high_veg.addRow("Max tree crown top height:", self.tree_max_crown_top_height_spin)
        form_high_veg.addRow(
            "Min tree crown bottom height:",
            self.tree_min_crown_bottom_height_spin,
        )
        form_high_veg.addRow("Validation:", self.high_vegetation_validation_label)

        form_vehicle = QFormLayout()
        form_vehicle.addRow("", self.custom_vehicle_count_check)
        form_vehicle.addRow("Vehicle instances:", self.vehicle_count_spin)
        form_vehicle.addRow("", self.custom_vehicle_type_check)
        for vehicle_type in self._vehicle_types:
            form_vehicle.addRow(
                f"{self._vehicle_type_names[vehicle_type]}:",
                self.vehicle_type_percentage_spins[vehicle_type],
            )
        form_vehicle.addRow("Type sum:", self.vehicle_distribution_sum_label)

        form_low_veg = QFormLayout()
        form_low_veg.addRow("", self.custom_shrub_count_check)
        form_low_veg.addRow("Shrub clusters:", self.shrub_count_spin)
        form_low_veg.addRow("", self.random_shrub_size_check)
        form_low_veg.addRow("Max shrub crown diameter:", self.shrub_max_diameter_spin)
        form_low_veg.addRow("Max shrub top height:", self.shrub_max_top_height_spin)
        form_low_veg.addRow("Min shrub crown bottom height:", self.shrub_min_bottom_height_spin)
        form_low_veg.addRow("", self.custom_grass_patch_count_check)
        form_low_veg.addRow("Grass patches:", self.grass_patch_count_spin)
        form_low_veg.addRow("", self.random_grass_patch_size_check)
        form_low_veg.addRow("Max grass patch size X:", self.grass_patch_max_size_x_spin)
        form_low_veg.addRow("Max grass patch size Y:", self.grass_patch_max_size_y_spin)
        form_low_veg.addRow("Max grass height:", self.grass_max_height_spin)
        form_low_veg.addRow("Validation:", self.low_vegetation_validation_label)

        tabs = QTabWidget(self)
        general_tab = QWidget(self)
        general_layout = QVBoxLayout(general_tab)
        general_layout.addLayout(form_general)
        general_layout.addStretch(1)
        tabs.addTab(general_tab, "General")

        artificial_surface_tab = QWidget(self)
        artificial_surface_layout = QVBoxLayout(artificial_surface_tab)
        artificial_surface_layout.addLayout(form_artificial_surface)
        artificial_surface_layout.addStretch(1)
        tabs.addTab(artificial_surface_tab, "Artificial Surface")

        building_tab = QWidget(self)
        building_layout = QVBoxLayout(building_tab)
        building_layout.addLayout(form_building)
        building_layout.addStretch(1)
        tabs.addTab(building_tab, "Buildings")

        structure_tab = QWidget(self)
        structure_container = QWidget(self)
        structure_container_layout = QVBoxLayout(structure_container)
        structure_container_layout.addLayout(form_structure)
        structure_container_layout.addStretch(1)
        structure_scroll = QScrollArea(structure_tab)
        structure_scroll.setWidgetResizable(True)
        structure_scroll.setWidget(structure_container)
        structure_layout = QVBoxLayout(structure_tab)
        structure_layout.addWidget(structure_scroll)
        tabs.addTab(structure_tab, "Structures")

        high_veg_tab = QWidget(self)
        high_veg_layout = QVBoxLayout(high_veg_tab)
        high_veg_layout.addLayout(form_high_veg)
        high_veg_layout.addStretch(1)
        tabs.addTab(high_veg_tab, "High Vegetation")

        vehicle_tab = QWidget(self)
        vehicle_layout = QVBoxLayout(vehicle_tab)
        vehicle_layout.addLayout(form_vehicle)
        vehicle_layout.addStretch(1)
        tabs.addTab(vehicle_tab, "Vehicles")

        low_veg_tab = QWidget(self)
        low_veg_layout = QVBoxLayout(low_veg_tab)
        low_veg_layout.addLayout(form_low_veg)
        low_veg_layout.addStretch(1)
        tabs.addTab(low_veg_tab, "Low Vegetation")

        note = QLabel(
            (
                "Generation uses utils/synthetic_labeled_point_cloud.py pipeline. "
                "When custom distributions are enabled, percentages must sum to 100%."
            ),
            self,
        )
        note.setWordWrap(True)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.import_config_button = buttons.addButton(
            "Import Configuration...",
            QDialogButtonBox.ActionRole,
        )
        self.import_config_button.clicked.connect(self.import_configuration)
        self.export_config_button = buttons.addButton(
            "Export Configuration...",
            QDialogButtonBox.ActionRole,
        )
        self.export_config_button.clicked.connect(self.export_configuration)
        self.ok_button = buttons.button(QDialogButtonBox.Ok)

        layout = QVBoxLayout(self)
        layout.addWidget(tabs)
        layout.addWidget(note)
        layout.addWidget(buttons)
        self.setLayout(layout)

        self._update_class_distribution_state()
        self._update_artificial_surface_settings_state()
        self._update_building_state()
        self._update_structure_settings_state()
        self._update_high_vegetation_state()
        self._update_vehicle_settings_state()
        self._update_low_vegetation_state()
        self._update_class_distribution_summary()
        self._update_artificial_surface_distribution_summary()
        self._update_building_distribution_summary()
        self._update_structure_distribution_summary()
        self._update_tree_crown_distribution_summary()
        self._update_vehicle_distribution_summary()

    def _resolve_synthetic_module(self):
        if self._synthetic_module is not None:
            return self._synthetic_module
        try:
            from utils import synthetic_labeled_point_cloud as synthetic_module
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self,
                "Generator Error",
                f"Failed to import utils/synthetic_labeled_point_cloud.py:\n{exc}",
            )
            return None
        self._synthetic_module = synthetic_module
        return self._synthetic_module

    def _apply_params(self, params: SyntheticGenerationParams) -> None:
        def _set_spin_value(spin_box, value, name: str) -> None:
            minimum = spin_box.minimum()
            maximum = spin_box.maximum()
            if value < minimum or value > maximum:
                raise ValueError(
                    f"`{name}` is outside the supported dialog range [{minimum}, {maximum}]: {value}."
                )
            spin_box.setValue(value)

        class_percentages = self._normalize_percentages(
            values=params.class_percentages,
            class_count=len(self._class_ids),
            fallback=FALLBACK_CLASS_PERCENTAGES,
        )
        artificial_surface_percentages = self._normalize_percentages(
            values=params.artificial_surface_type_percentages,
            class_count=len(self._artificial_surface_types),
            fallback=FALLBACK_ARTIFICIAL_SURFACE_TYPE_PERCENTAGES,
        )
        building_roof_percentages = self._normalize_percentages(
            values=params.building_roof_type_percentages,
            class_count=len(self._building_roof_types),
            fallback=FALLBACK_BUILDING_ROOF_TYPE_PERCENTAGES,
        )
        structure_percentages = self._normalize_percentages(
            values=params.structure_type_percentages,
            class_count=len(self._structure_types),
            fallback=FALLBACK_STRUCTURE_TYPE_PERCENTAGES,
        )
        tree_crown_percentages = self._normalize_percentages(
            values=params.tree_crown_type_percentages,
            class_count=len(self._tree_crown_types),
            fallback=FALLBACK_TREE_CROWN_TYPE_PERCENTAGES,
        )
        vehicle_percentages = self._normalize_percentages(
            values=params.vehicle_type_percentages,
            class_count=len(self._vehicle_types),
            fallback=FALLBACK_VEHICLE_TYPE_PERCENTAGES,
        )

        _set_spin_value(self.total_points_spin, int(params.total_points), "total_points")
        _set_spin_value(self.area_width_spin, float(params.area_width), "area_width")
        _set_spin_value(self.area_length_spin, float(params.area_length), "area_length")
        _set_spin_value(self.terrain_relief_spin, float(params.terrain_relief), "terrain_relief")
        _set_spin_value(self.seed_spin, int(params.seed), "seed")
        self.random_counts_check.setChecked(bool(params.randomize_object_counts))
        self.custom_distribution_check.setChecked(bool(params.custom_class_distribution))
        for index, class_id in enumerate(self._class_ids):
            _set_spin_value(
                self.class_percentage_spins[class_id],
                float(class_percentages[index]),
                f"class_percentages[{class_id}]",
            )

        self.custom_artificial_surface_count_check.setChecked(
            bool(params.custom_artificial_surface_count)
        )
        _set_spin_value(
            self.artificial_surface_count_spin,
            int(params.artificial_surface_count),
            "artificial_surface_count",
        )
        self.custom_artificial_surface_type_check.setChecked(
            bool(params.custom_artificial_surface_type_distribution)
        )
        for index, surface_type in enumerate(self._artificial_surface_types):
            _set_spin_value(
                self.artificial_surface_type_percentage_spins[surface_type],
                float(artificial_surface_percentages[index]),
                f"artificial_surface_type_percentages[{surface_type}]",
            )

        self.custom_building_count_check.setChecked(bool(params.custom_building_count))
        _set_spin_value(self.building_count_spin, int(params.building_count), "building_count")
        self.custom_building_roof_type_check.setChecked(
            bool(params.custom_building_roof_type_distribution)
        )
        for index, roof_type in enumerate(self._building_roof_types):
            _set_spin_value(
                self.building_roof_type_percentage_spins[roof_type],
                float(building_roof_percentages[index]),
                f"building_roof_type_percentages[{roof_type}]",
            )
        floor_min = int(params.building_floor_min)
        floor_max = int(params.building_floor_max)
        _set_spin_value(self.building_floor_min_spin, floor_min, "building_floor_min")
        _set_spin_value(self.building_floor_max_spin, floor_max, "building_floor_max")
        self.building_random_yaw_check.setChecked(bool(params.building_random_yaw))

        self.custom_structure_count_check.setChecked(bool(params.custom_structure_count))
        _set_spin_value(self.structure_count_spin, int(params.structure_count), "structure_count")
        self.custom_structure_type_check.setChecked(bool(params.custom_structure_type_distribution))
        for index, structure_type in enumerate(self._structure_types):
            _set_spin_value(
                self.structure_type_percentage_spins[structure_type],
                float(structure_percentages[index]),
                f"structure_type_percentages[{structure_type}]",
            )

        self.custom_tree_count_check.setChecked(bool(params.custom_tree_count))
        _set_spin_value(self.tree_count_spin, int(params.tree_count), "tree_count")
        self.custom_tree_crown_type_check.setChecked(
            bool(params.custom_tree_crown_type_distribution)
        )
        for index, crown_type in enumerate(self._tree_crown_types):
            _set_spin_value(
                self.tree_crown_type_percentage_spins[crown_type],
                float(tree_crown_percentages[index]),
                f"tree_crown_type_percentages[{crown_type}]",
            )
        self.random_tree_crown_size_check.setChecked(bool(params.random_tree_crown_size))
        _set_spin_value(
            self.tree_max_crown_diameter_spin,
            float(params.tree_max_crown_diameter),
            "tree_max_crown_diameter",
        )
        _set_spin_value(
            self.tree_max_crown_top_height_spin,
            float(params.tree_max_crown_top_height),
            "tree_max_crown_top_height",
        )
        _set_spin_value(
            self.tree_min_crown_bottom_height_spin,
            float(params.tree_min_crown_bottom_height),
            "tree_min_crown_bottom_height",
        )

        self.custom_vehicle_count_check.setChecked(bool(params.custom_vehicle_count))
        _set_spin_value(self.vehicle_count_spin, int(params.vehicle_count), "vehicle_count")
        self.custom_vehicle_type_check.setChecked(bool(params.custom_vehicle_type_distribution))
        for index, vehicle_type in enumerate(self._vehicle_types):
            _set_spin_value(
                self.vehicle_type_percentage_spins[vehicle_type],
                float(vehicle_percentages[index]),
                f"vehicle_type_percentages[{vehicle_type}]",
            )

        self.custom_shrub_count_check.setChecked(bool(params.custom_shrub_count))
        _set_spin_value(self.shrub_count_spin, int(params.shrub_count), "shrub_count")
        self.random_shrub_size_check.setChecked(bool(params.random_shrub_size))
        _set_spin_value(
            self.shrub_max_diameter_spin,
            float(params.shrub_max_diameter),
            "shrub_max_diameter",
        )
        _set_spin_value(
            self.shrub_max_top_height_spin,
            float(params.shrub_max_top_height),
            "shrub_max_top_height",
        )
        _set_spin_value(
            self.shrub_min_bottom_height_spin,
            float(params.shrub_min_bottom_height),
            "shrub_min_bottom_height",
        )

        self.custom_grass_patch_count_check.setChecked(bool(params.custom_grass_patch_count))
        _set_spin_value(
            self.grass_patch_count_spin,
            int(params.grass_patch_count),
            "grass_patch_count",
        )
        self.random_grass_patch_size_check.setChecked(bool(params.random_grass_patch_size))
        _set_spin_value(
            self.grass_patch_max_size_x_spin,
            float(params.grass_patch_max_size_x),
            "grass_patch_max_size_x",
        )
        _set_spin_value(
            self.grass_patch_max_size_y_spin,
            float(params.grass_patch_max_size_y),
            "grass_patch_max_size_y",
        )
        _set_spin_value(
            self.grass_max_height_spin,
            float(params.grass_max_height),
            "grass_max_height",
        )

        self._update_class_distribution_state()
        self._update_artificial_surface_settings_state()
        self._update_building_state()
        self._update_structure_settings_state()
        self._update_high_vegetation_state()
        self._update_vehicle_settings_state()
        self._update_low_vegetation_state()
        self._update_class_distribution_summary()
        self._update_artificial_surface_distribution_summary()
        self._update_building_distribution_summary()
        self._update_structure_distribution_summary()
        self._update_tree_crown_distribution_summary()
        self._update_vehicle_distribution_summary()

    def _collect_validation_errors(self) -> List[str]:
        errors: List[str] = []
        if not self._is_class_distribution_valid():
            errors.append(
                "When custom class distribution is enabled, class percentages must sum to 100%."
            )
        if not self._is_artificial_surface_distribution_valid():
            errors.append(
                "When custom artificial surface type distribution is enabled, percentages must sum to 100%."
            )
        if not self._is_building_roof_distribution_valid():
            errors.append(
                "When custom building roof type distribution is enabled, percentages must sum to 100%."
            )
        if not self._is_structure_distribution_valid():
            errors.append(
                "When custom structure type distribution is enabled, percentages must sum to 100%."
            )
        if not self._is_tree_crown_distribution_valid():
            errors.append(
                "When custom tree crown type distribution is enabled, percentages must sum to 100%."
            )
        if not self._is_vehicle_distribution_valid():
            errors.append(
                "When custom vehicle type distribution is enabled, vehicle type percentages must sum to 100%."
            )
        building_error = self._building_validation_error()
        if building_error is not None:
            errors.append(building_error)
        high_veg_error = self._high_vegetation_validation_error()
        if high_veg_error is not None:
            errors.append(high_veg_error)
        low_veg_error = self._low_vegetation_validation_error()
        if low_veg_error is not None:
            errors.append(low_veg_error)
        return errors

    def export_configuration(self) -> None:
        synthetic_module = self._resolve_synthetic_module()
        if synthetic_module is None:
            return

        errors = self._collect_validation_errors()
        if errors:
            QMessageBox.warning(
                self,
                "Export Error",
                "Configuration cannot be exported because current settings are invalid:\n"
                + "\n".join(errors),
            )
            return

        default_name = f"synthetic_generation_seed_{int(self.seed_spin.value())}.yaml"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Generation Configuration",
            str(ensure_data_dir() / default_name),
            "YAML Files (*.yaml *.yml);;All Files (*)",
        )
        if not path:
            return

        out_path = Path(path)
        if out_path.suffix.lower() not in {".yaml", ".yml"}:
            out_path = out_path.with_suffix(".yaml")

        try:
            synthetic_module.save_generation_config(asdict(self.params()), out_path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to save generation configuration:\n{exc}",
            )
            return

    def import_configuration(self) -> None:
        synthetic_module = self._resolve_synthetic_module()
        if synthetic_module is None:
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Generation Configuration",
            str(ensure_data_dir()),
            "YAML Files (*.yaml *.yml);;All Files (*)",
        )
        if not path:
            return

        try:
            config_data = synthetic_module.load_generation_config(path)
            self._apply_params(SyntheticGenerationParams(**config_data))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self,
                "Import Error",
                f"Failed to load generation configuration:\n{exc}",
            )
            return

    @staticmethod
    def _normalize_class_names(class_names: Optional[Dict[int, str]]) -> Dict[int, str]:
        if not isinstance(class_names, dict):
            return dict(FALLBACK_SYNTHETIC_CLASS_NAMES)

        normalized: Dict[int, str] = {}
        for key, value in class_names.items():
            try:
                class_id = int(key)
            except (TypeError, ValueError):
                continue
            normalized[class_id] = str(value)

        if not normalized:
            return dict(FALLBACK_SYNTHETIC_CLASS_NAMES)
        return {class_id: normalized[class_id] for class_id in sorted(normalized)}

    @staticmethod
    def _normalize_artificial_surface_type_names(
        artificial_surface_type_names: Optional[Dict[str, str]],
    ) -> Dict[str, str]:
        out: Dict[str, str] = {}
        source = (
            artificial_surface_type_names
            if isinstance(artificial_surface_type_names, dict)
            else {}
        )
        for surface_type in FALLBACK_ARTIFICIAL_SURFACE_TYPES:
            label = source.get(surface_type)
            if label is None:
                label = FALLBACK_ARTIFICIAL_SURFACE_TYPE_NAMES[surface_type]
            out[surface_type] = str(label)
        return out

    @staticmethod
    def _normalize_building_roof_type_names(
        building_roof_type_names: Optional[Dict[str, str]],
    ) -> Dict[str, str]:
        out: Dict[str, str] = {}
        source = building_roof_type_names if isinstance(building_roof_type_names, dict) else {}
        for roof_type in FALLBACK_BUILDING_ROOF_TYPES:
            label = source.get(roof_type)
            if label is None:
                label = FALLBACK_BUILDING_ROOF_TYPE_NAMES[roof_type]
            out[roof_type] = str(label)
        return out

    @staticmethod
    def _normalize_building_defaults(
        building_defaults: Optional[Dict[str, int | bool]],
    ) -> Dict[str, int | bool]:
        out: Dict[str, int | bool] = dict(FALLBACK_BUILDING_DEFAULTS)
        if not isinstance(building_defaults, dict):
            return out

        floor_min = building_defaults.get("building_floor_min")
        floor_max = building_defaults.get("building_floor_max")
        if floor_min is not None:
            try:
                floor_min_i = int(floor_min)
                if floor_min_i >= 1:
                    out["building_floor_min"] = floor_min_i
            except (TypeError, ValueError):
                pass
        if floor_max is not None:
            try:
                floor_max_i = int(floor_max)
                if floor_max_i >= 1:
                    out["building_floor_max"] = floor_max_i
            except (TypeError, ValueError):
                pass

        if int(out["building_floor_max"]) < int(out["building_floor_min"]):
            out["building_floor_max"] = int(out["building_floor_min"])

        yaw_value = building_defaults.get("building_random_yaw")
        if yaw_value is not None:
            out["building_random_yaw"] = bool(yaw_value)

        return out

    @staticmethod
    def _normalize_structure_type_names(
        structure_type_names: Optional[Dict[str, str]],
    ) -> Dict[str, str]:
        out: Dict[str, str] = {}
        source = structure_type_names if isinstance(structure_type_names, dict) else {}
        for structure_type in FALLBACK_STRUCTURE_TYPES:
            label = source.get(structure_type)
            if label is None:
                label = FALLBACK_STRUCTURE_TYPE_NAMES[structure_type]
            out[structure_type] = str(label)
        return out

    @staticmethod
    def _normalize_tree_crown_type_names(
        tree_crown_type_names: Optional[Dict[str, str]],
    ) -> Dict[str, str]:
        out: Dict[str, str] = {}
        source = tree_crown_type_names if isinstance(tree_crown_type_names, dict) else {}
        for crown_type in FALLBACK_TREE_CROWN_TYPES:
            label = source.get(crown_type)
            if label is None:
                label = FALLBACK_TREE_CROWN_TYPE_NAMES[crown_type]
            out[crown_type] = str(label)
        return out

    @staticmethod
    def _normalize_vehicle_type_names(
        vehicle_type_names: Optional[Dict[str, str]],
    ) -> Dict[str, str]:
        out: Dict[str, str] = {}
        source = vehicle_type_names if isinstance(vehicle_type_names, dict) else {}
        for vehicle_type in FALLBACK_VEHICLE_TYPES:
            label = source.get(vehicle_type)
            if label is None:
                label = FALLBACK_VEHICLE_TYPE_NAMES[vehicle_type]
            out[vehicle_type] = str(label)
        return out

    @staticmethod
    def _normalize_high_veg_defaults(
        high_veg_defaults: Optional[Dict[str, float]],
    ) -> Dict[str, float]:
        out = dict(FALLBACK_HIGH_VEG_DEFAULTS)
        if not isinstance(high_veg_defaults, dict):
            return out

        for key in out:
            value = high_veg_defaults.get(key)
            if value is None:
                continue
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                continue
            if key == "tree_min_crown_bottom_height":
                if value_f < 0.0:
                    continue
            elif value_f <= 0.0:
                continue
            out[key] = value_f
        return out

    @staticmethod
    def _normalize_low_veg_defaults(
        low_veg_defaults: Optional[Dict[str, float]],
    ) -> Dict[str, float]:
        out = dict(FALLBACK_LOW_VEG_DEFAULTS)
        if not isinstance(low_veg_defaults, dict):
            return out

        for key in out:
            value = low_veg_defaults.get(key)
            if value is None:
                continue
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                continue
            if key == "shrub_min_bottom_height":
                if value_f < 0.0:
                    continue
            elif value_f <= 0.0:
                continue
            out[key] = value_f
        return out

    @staticmethod
    def _normalize_percentages(
        values: Optional[Sequence[float]],
        class_count: int,
        fallback: Sequence[float],
    ) -> Tuple[float, ...]:
        target_count = max(1, int(class_count))
        fallback_values: List[float] = []
        for idx in range(target_count):
            fallback_values.append(float(fallback[idx]) if idx < len(fallback) else 0.0)

        if values is None:
            return tuple(fallback_values)

        try:
            normalized = [float(value) for value in values]
        except (TypeError, ValueError):
            return tuple(fallback_values)

        if len(normalized) != target_count:
            return tuple(fallback_values)

        as_array = np.asarray(normalized, dtype=np.float64)
        if np.any(~np.isfinite(as_array)) or np.any(as_array < 0.0):
            return tuple(fallback_values)

        return tuple(float(value) for value in normalized)

    def _current_class_percentages(self) -> Tuple[float, ...]:
        return tuple(
            float(self.class_percentage_spins[class_id].value())
            for class_id in self._class_ids
        )

    def _current_artificial_surface_type_percentages(self) -> Tuple[float, ...]:
        return tuple(
            float(self.artificial_surface_type_percentage_spins[surface_type].value())
            for surface_type in self._artificial_surface_types
        )

    def _current_building_roof_type_percentages(self) -> Tuple[float, ...]:
        return tuple(
            float(self.building_roof_type_percentage_spins[roof_type].value())
            for roof_type in self._building_roof_types
        )

    def _current_structure_type_percentages(self) -> Tuple[float, ...]:
        return tuple(
            float(self.structure_type_percentage_spins[structure_type].value())
            for structure_type in self._structure_types
        )

    def _current_tree_crown_type_percentages(self) -> Tuple[float, ...]:
        return tuple(
            float(self.tree_crown_type_percentage_spins[crown_type].value())
            for crown_type in self._tree_crown_types
        )

    def _current_vehicle_type_percentages(self) -> Tuple[float, ...]:
        return tuple(
            float(self.vehicle_type_percentage_spins[vehicle_type].value())
            for vehicle_type in self._vehicle_types
        )

    def _is_class_distribution_valid(self) -> bool:
        if not self.custom_distribution_check.isChecked():
            return True

        percentages = np.asarray(self._current_class_percentages(), dtype=np.float64)
        total = float(percentages.sum())
        has_positive = bool(np.any(percentages > 0.0))
        return has_positive and abs(total - 100.0) <= 0.01

    def _is_artificial_surface_distribution_valid(self) -> bool:
        if not self.custom_artificial_surface_type_check.isChecked():
            return True

        percentages = np.asarray(
            self._current_artificial_surface_type_percentages(),
            dtype=np.float64,
        )
        total = float(percentages.sum())
        has_positive = bool(np.any(percentages > 0.0))
        return has_positive and abs(total - 100.0) <= 0.01

    def _is_building_roof_distribution_valid(self) -> bool:
        if not self.custom_building_roof_type_check.isChecked():
            return True

        percentages = np.asarray(self._current_building_roof_type_percentages(), dtype=np.float64)
        total = float(percentages.sum())
        has_positive = bool(np.any(percentages > 0.0))
        return has_positive and abs(total - 100.0) <= 0.01

    def _is_structure_distribution_valid(self) -> bool:
        if not self.custom_structure_type_check.isChecked():
            return True

        percentages = np.asarray(self._current_structure_type_percentages(), dtype=np.float64)
        total = float(percentages.sum())
        has_positive = bool(np.any(percentages > 0.0))
        return has_positive and abs(total - 100.0) <= 0.01

    def _is_tree_crown_distribution_valid(self) -> bool:
        if not self.custom_tree_crown_type_check.isChecked():
            return True

        percentages = np.asarray(self._current_tree_crown_type_percentages(), dtype=np.float64)
        total = float(percentages.sum())
        has_positive = bool(np.any(percentages > 0.0))
        return has_positive and abs(total - 100.0) <= 0.01

    def _is_vehicle_distribution_valid(self) -> bool:
        if not self.custom_vehicle_type_check.isChecked():
            return True

        percentages = np.asarray(self._current_vehicle_type_percentages(), dtype=np.float64)
        total = float(percentages.sum())
        has_positive = bool(np.any(percentages > 0.0))
        return has_positive and abs(total - 100.0) <= 0.01

    def _building_validation_error(self) -> Optional[str]:
        floor_min = int(self.building_floor_min_spin.value())
        floor_max = int(self.building_floor_max_spin.value())
        if floor_min <= 0:
            return "Minimum building floors must be >= 1."
        if floor_max < floor_min:
            return "Maximum building floors must be greater or equal to minimum."
        return None

    def _high_vegetation_validation_error(self) -> Optional[str]:
        tree_top = float(self.tree_max_crown_top_height_spin.value())
        tree_bottom = float(self.tree_min_crown_bottom_height_spin.value())
        if tree_bottom >= tree_top:
            return "Tree min crown bottom height must be less than tree max crown top height."
        return None

    def _low_vegetation_validation_error(self) -> Optional[str]:
        shrub_top = float(self.shrub_max_top_height_spin.value())
        shrub_bottom = float(self.shrub_min_bottom_height_spin.value())
        if shrub_bottom >= shrub_top:
            return "Shrub min bottom height must be less than shrub max top height."
        return None

    def _update_class_distribution_state(self) -> None:
        enabled = bool(self.custom_distribution_check.isChecked())
        for spin in self.class_percentage_spins.values():
            spin.setEnabled(enabled)
        self._update_class_distribution_summary()

    def _update_artificial_surface_settings_state(self) -> None:
        self.artificial_surface_count_spin.setEnabled(
            bool(self.custom_artificial_surface_count_check.isChecked())
        )
        is_custom_distribution = bool(self.custom_artificial_surface_type_check.isChecked())
        for spin in self.artificial_surface_type_percentage_spins.values():
            spin.setEnabled(is_custom_distribution)
        self._update_artificial_surface_distribution_summary()
        self._update_ok_button_state()

    def _update_building_state(self) -> None:
        self.building_count_spin.setEnabled(bool(self.custom_building_count_check.isChecked()))
        is_custom_distribution = bool(self.custom_building_roof_type_check.isChecked())
        for spin in self.building_roof_type_percentage_spins.values():
            spin.setEnabled(is_custom_distribution)

        error = self._building_validation_error()
        if error is None:
            self.building_validation_label.setText("Valid")
            self.building_validation_label.setStyleSheet("color: #1f7a1f;")
        else:
            self.building_validation_label.setText(error)
            self.building_validation_label.setStyleSheet("color: #b00020;")

        self._update_building_distribution_summary()
        self._update_ok_button_state()

    def _update_structure_settings_state(self) -> None:
        self.structure_count_spin.setEnabled(bool(self.custom_structure_count_check.isChecked()))
        is_custom_distribution = bool(self.custom_structure_type_check.isChecked())
        for spin in self.structure_type_percentage_spins.values():
            spin.setEnabled(is_custom_distribution)
        self._update_structure_distribution_summary()
        self._update_ok_button_state()

    def _update_high_vegetation_state(self) -> None:
        self.tree_count_spin.setEnabled(bool(self.custom_tree_count_check.isChecked()))
        is_custom_distribution = bool(self.custom_tree_crown_type_check.isChecked())
        for spin in self.tree_crown_type_percentage_spins.values():
            spin.setEnabled(is_custom_distribution)

        crown_custom_size = not bool(self.random_tree_crown_size_check.isChecked())
        self.tree_max_crown_diameter_spin.setEnabled(crown_custom_size)
        self.tree_max_crown_top_height_spin.setEnabled(crown_custom_size)
        self.tree_min_crown_bottom_height_spin.setEnabled(crown_custom_size)

        error = self._high_vegetation_validation_error()
        if error is None:
            self.high_vegetation_validation_label.setText("Valid")
            self.high_vegetation_validation_label.setStyleSheet("color: #1f7a1f;")
        else:
            self.high_vegetation_validation_label.setText(error)
            self.high_vegetation_validation_label.setStyleSheet("color: #b00020;")

        self._update_tree_crown_distribution_summary()
        self._update_ok_button_state()

    def _update_vehicle_settings_state(self) -> None:
        self.vehicle_count_spin.setEnabled(bool(self.custom_vehicle_count_check.isChecked()))
        is_custom_distribution = bool(self.custom_vehicle_type_check.isChecked())
        for spin in self.vehicle_type_percentage_spins.values():
            spin.setEnabled(is_custom_distribution)
        self._update_vehicle_distribution_summary()

    def _update_low_vegetation_state(self) -> None:
        self.shrub_count_spin.setEnabled(bool(self.custom_shrub_count_check.isChecked()))
        shrub_custom_size = not bool(self.random_shrub_size_check.isChecked())
        self.shrub_max_diameter_spin.setEnabled(shrub_custom_size)
        self.shrub_max_top_height_spin.setEnabled(shrub_custom_size)
        self.shrub_min_bottom_height_spin.setEnabled(shrub_custom_size)

        self.grass_patch_count_spin.setEnabled(bool(self.custom_grass_patch_count_check.isChecked()))
        grass_custom_size = not bool(self.random_grass_patch_size_check.isChecked())
        self.grass_patch_max_size_x_spin.setEnabled(grass_custom_size)
        self.grass_patch_max_size_y_spin.setEnabled(grass_custom_size)
        self.grass_max_height_spin.setEnabled(grass_custom_size)

        error = self._low_vegetation_validation_error()
        if error is None:
            self.low_vegetation_validation_label.setText("Valid")
            self.low_vegetation_validation_label.setStyleSheet("color: #1f7a1f;")
        else:
            self.low_vegetation_validation_label.setText(error)
            self.low_vegetation_validation_label.setStyleSheet("color: #b00020;")
        self._update_ok_button_state()

    def _update_class_distribution_summary(self) -> None:
        total = float(sum(self._current_class_percentages()))
        is_custom = bool(self.custom_distribution_check.isChecked())
        is_valid = self._is_class_distribution_valid()

        if not is_custom:
            text = f"{total:.2f}% (custom disabled, defaults from generator will be used)"
            color = "#666666"
        elif is_valid:
            text = f"{total:.2f}% (valid)"
            color = "#1f7a1f"
        elif total <= 0.0:
            text = f"{total:.2f}% (invalid: at least one class must be > 0)"
            color = "#b00020"
        else:
            text = f"{total:.2f}% (invalid: must equal 100.00%)"
            color = "#b00020"

        self.class_distribution_sum_label.setText(text)
        self.class_distribution_sum_label.setStyleSheet(f"color: {color};")
        self._update_ok_button_state()

    def _update_artificial_surface_distribution_summary(self) -> None:
        total = float(sum(self._current_artificial_surface_type_percentages()))
        is_custom = bool(self.custom_artificial_surface_type_check.isChecked())
        is_valid = self._is_artificial_surface_distribution_valid()

        if not is_custom:
            text = f"{total:.2f}% (custom disabled, random distribution will be used)"
            color = "#666666"
        elif is_valid:
            text = f"{total:.2f}% (valid)"
            color = "#1f7a1f"
        elif total <= 0.0:
            text = f"{total:.2f}% (invalid: at least one type must be > 0)"
            color = "#b00020"
        else:
            text = f"{total:.2f}% (invalid: must equal 100.00%)"
            color = "#b00020"

        self.artificial_surface_distribution_sum_label.setText(text)
        self.artificial_surface_distribution_sum_label.setStyleSheet(f"color: {color};")
        self._update_ok_button_state()

    def _update_building_distribution_summary(self) -> None:
        total = float(sum(self._current_building_roof_type_percentages()))
        is_custom = bool(self.custom_building_roof_type_check.isChecked())
        is_valid = self._is_building_roof_distribution_valid()

        if not is_custom:
            text = f"{total:.2f}% (custom disabled, random distribution will be used)"
            color = "#666666"
        elif is_valid:
            text = f"{total:.2f}% (valid)"
            color = "#1f7a1f"
        elif total <= 0.0:
            text = f"{total:.2f}% (invalid: at least one type must be > 0)"
            color = "#b00020"
        else:
            text = f"{total:.2f}% (invalid: must equal 100.00%)"
            color = "#b00020"

        self.building_roof_distribution_sum_label.setText(text)
        self.building_roof_distribution_sum_label.setStyleSheet(f"color: {color};")
        self._update_ok_button_state()

    def _update_structure_distribution_summary(self) -> None:
        total = float(sum(self._current_structure_type_percentages()))
        is_custom = bool(self.custom_structure_type_check.isChecked())
        is_valid = self._is_structure_distribution_valid()

        if not is_custom:
            text = f"{total:.2f}% (custom disabled, random distribution will be used)"
            color = "#666666"
        elif is_valid:
            text = f"{total:.2f}% (valid)"
            color = "#1f7a1f"
        elif total <= 0.0:
            text = f"{total:.2f}% (invalid: at least one type must be > 0)"
            color = "#b00020"
        else:
            text = f"{total:.2f}% (invalid: must equal 100.00%)"
            color = "#b00020"

        self.structure_distribution_sum_label.setText(text)
        self.structure_distribution_sum_label.setStyleSheet(f"color: {color};")
        self._update_ok_button_state()

    def _update_tree_crown_distribution_summary(self) -> None:
        total = float(sum(self._current_tree_crown_type_percentages()))
        is_custom = bool(self.custom_tree_crown_type_check.isChecked())
        is_valid = self._is_tree_crown_distribution_valid()

        if not is_custom:
            text = f"{total:.2f}% (custom disabled, random distribution will be used)"
            color = "#666666"
        elif is_valid:
            text = f"{total:.2f}% (valid)"
            color = "#1f7a1f"
        elif total <= 0.0:
            text = f"{total:.2f}% (invalid: at least one type must be > 0)"
            color = "#b00020"
        else:
            text = f"{total:.2f}% (invalid: must equal 100.00%)"
            color = "#b00020"

        self.tree_crown_distribution_sum_label.setText(text)
        self.tree_crown_distribution_sum_label.setStyleSheet(f"color: {color};")
        self._update_ok_button_state()

    def _update_vehicle_distribution_summary(self) -> None:
        total = float(sum(self._current_vehicle_type_percentages()))
        is_custom = bool(self.custom_vehicle_type_check.isChecked())
        is_valid = self._is_vehicle_distribution_valid()

        if not is_custom:
            text = f"{total:.2f}% (custom disabled, defaults from generator will be used)"
            color = "#666666"
        elif is_valid:
            text = f"{total:.2f}% (valid)"
            color = "#1f7a1f"
        elif total <= 0.0:
            text = f"{total:.2f}% (invalid: at least one type must be > 0)"
            color = "#b00020"
        else:
            text = f"{total:.2f}% (invalid: must equal 100.00%)"
            color = "#b00020"

        self.vehicle_distribution_sum_label.setText(text)
        self.vehicle_distribution_sum_label.setStyleSheet(f"color: {color};")
        self._update_ok_button_state()

    def _update_ok_button_state(self) -> None:
        if self.ok_button is None:
            return
        self.ok_button.setEnabled(
            self._is_class_distribution_valid()
            and self._is_artificial_surface_distribution_valid()
            and self._is_building_roof_distribution_valid()
            and self._is_structure_distribution_valid()
            and self._is_tree_crown_distribution_valid()
            and self._is_vehicle_distribution_valid()
            and self._building_validation_error() is None
            and self._high_vegetation_validation_error() is None
            and self._low_vegetation_validation_error() is None
        )

    def accept(self) -> None:
        errors = self._collect_validation_errors()
        if errors:
            QMessageBox.warning(
                self,
                "Invalid Generation Settings",
                "\n".join(errors),
            )
            return
        super().accept()

    def params(self) -> SyntheticGenerationParams:
        return SyntheticGenerationParams(
            total_points=int(self.total_points_spin.value()),
            area_width=float(self.area_width_spin.value()),
            area_length=float(self.area_length_spin.value()),
            terrain_relief=float(self.terrain_relief_spin.value()),
            seed=int(self.seed_spin.value()),
            randomize_object_counts=bool(self.random_counts_check.isChecked()),
            custom_class_distribution=bool(self.custom_distribution_check.isChecked()),
            class_percentages=self._current_class_percentages(),
            custom_artificial_surface_count=bool(
                self.custom_artificial_surface_count_check.isChecked()
            ),
            artificial_surface_count=int(self.artificial_surface_count_spin.value()),
            custom_artificial_surface_type_distribution=bool(
                self.custom_artificial_surface_type_check.isChecked()
            ),
            artificial_surface_type_percentages=(
                self._current_artificial_surface_type_percentages()
            ),
            custom_building_count=bool(self.custom_building_count_check.isChecked()),
            building_count=int(self.building_count_spin.value()),
            custom_building_roof_type_distribution=bool(
                self.custom_building_roof_type_check.isChecked()
            ),
            building_roof_type_percentages=self._current_building_roof_type_percentages(),
            building_floor_min=int(self.building_floor_min_spin.value()),
            building_floor_max=int(self.building_floor_max_spin.value()),
            building_random_yaw=bool(self.building_random_yaw_check.isChecked()),
            custom_structure_count=bool(self.custom_structure_count_check.isChecked()),
            structure_count=int(self.structure_count_spin.value()),
            custom_structure_type_distribution=bool(self.custom_structure_type_check.isChecked()),
            structure_type_percentages=self._current_structure_type_percentages(),
            custom_tree_count=bool(self.custom_tree_count_check.isChecked()),
            tree_count=int(self.tree_count_spin.value()),
            custom_tree_crown_type_distribution=bool(self.custom_tree_crown_type_check.isChecked()),
            tree_crown_type_percentages=self._current_tree_crown_type_percentages(),
            random_tree_crown_size=bool(self.random_tree_crown_size_check.isChecked()),
            tree_max_crown_diameter=float(self.tree_max_crown_diameter_spin.value()),
            tree_max_crown_top_height=float(self.tree_max_crown_top_height_spin.value()),
            tree_min_crown_bottom_height=float(self.tree_min_crown_bottom_height_spin.value()),
            custom_vehicle_count=bool(self.custom_vehicle_count_check.isChecked()),
            vehicle_count=int(self.vehicle_count_spin.value()),
            custom_vehicle_type_distribution=bool(self.custom_vehicle_type_check.isChecked()),
            vehicle_type_percentages=self._current_vehicle_type_percentages(),
            custom_shrub_count=bool(self.custom_shrub_count_check.isChecked()),
            shrub_count=int(self.shrub_count_spin.value()),
            random_shrub_size=bool(self.random_shrub_size_check.isChecked()),
            shrub_max_diameter=float(self.shrub_max_diameter_spin.value()),
            shrub_max_top_height=float(self.shrub_max_top_height_spin.value()),
            shrub_min_bottom_height=float(self.shrub_min_bottom_height_spin.value()),
            custom_grass_patch_count=bool(self.custom_grass_patch_count_check.isChecked()),
            grass_patch_count=int(self.grass_patch_count_spin.value()),
            random_grass_patch_size=bool(self.random_grass_patch_size_check.isChecked()),
            grass_patch_max_size_x=float(self.grass_patch_max_size_x_spin.value()),
            grass_patch_max_size_y=float(self.grass_patch_max_size_y_spin.value()),
            grass_max_height=float(self.grass_max_height_spin.value()),
        )


class SettingsDialog(QDialog):
    def __init__(self, settings: ProjectSettings, parent=None):
        super().__init__(parent)
        self._accepted_settings = ProjectSettings(
            output_directory=str(settings.output_directory).strip() or "data",
            point_size=_clamp_point_size(settings.point_size),
            viewport_background=normalize_color_value(settings.viewport_background),
            bounding_box_color_mode=_normalize_bounding_box_color_mode(settings.bounding_box_color_mode),
            bounding_box_color=normalize_color_value(settings.bounding_box_color),
            bounding_box_line_width=_clamp_bounding_box_line_width(settings.bounding_box_line_width),
            bounding_box_show_id=bool(settings.bounding_box_show_id),
        )

        self.setWindowTitle("Project Settings")
        self.resize(640, 0)

        self.tabs = QTabWidget(self)

        self.output_dir_edit = QLineEdit(self)
        self.output_dir_edit.setText(self._accepted_settings.output_directory)

        browse_output_button = QPushButton("Browse...", self)
        browse_output_button.clicked.connect(self._browse_output_directory)

        output_dir_row = QWidget(self)
        output_dir_layout = QHBoxLayout(output_dir_row)
        output_dir_layout.setContentsMargins(0, 0, 0, 0)
        output_dir_layout.addWidget(self.output_dir_edit, 1)
        output_dir_layout.addWidget(browse_output_button)

        self.point_size_spin = QSpinBox(self)
        self.point_size_spin.setRange(1, 10)
        self.point_size_spin.setValue(int(round(self._accepted_settings.point_size)))
        self.point_size_spin.setToolTip("Point size used for viewport rendering, similar to CloudCompare.")

        self.background_preset_combo = QComboBox(self)
        self.background_preset_combo.addItem("Custom", "")
        for label, color_value in COLOR_PRESETS:
            self.background_preset_combo.addItem(label, color_value)
        self.background_preset_combo.currentIndexChanged.connect(self._apply_selected_background_preset)

        self.background_edit = QLineEdit(self)
        self.background_edit.setPlaceholderText("#14171C, rgb(20, 23, 28), or 0.08, 0.09, 0.11")
        self.background_edit.textChanged.connect(self._update_background_color_preview)

        self.background_pick_color_button = QPushButton("Pick Color...", self)
        self.background_pick_color_button.clicked.connect(self._choose_background_color)

        self.background_preview = QLabel(self)
        self.background_preview.setFixedSize(56, 24)

        background_row = QWidget(self)
        background_layout = QHBoxLayout(background_row)
        background_layout.setContentsMargins(0, 0, 0, 0)
        background_layout.addWidget(self.background_preset_combo)
        background_layout.addWidget(self.background_edit, 1)
        background_layout.addWidget(self.background_preview)
        background_layout.addWidget(self.background_pick_color_button)

        background_help_label = QLabel(
            "Background color accepts preset names, hex (#RRGGBB), rgb(r,g,b), or r,g,b values.",
            self,
        )
        background_help_label.setWordWrap(True)
        background_help_label.setStyleSheet("color: #5a6777;")

        general_form = QFormLayout()
        general_form.addRow("Output directory:", output_dir_row)
        general_form.addRow("Point size:", self.point_size_spin)
        general_form.addRow("Viewport background:", background_row)
        general_form.addRow("", background_help_label)

        general_tab = QWidget(self)
        general_layout = QVBoxLayout(general_tab)
        general_layout.setContentsMargins(0, 0, 0, 0)
        general_layout.addLayout(general_form)
        general_layout.addStretch(1)

        self.bounding_box_color_mode_combo = QComboBox(self)
        self.bounding_box_color_mode_combo.addItem("Random per box", "random")
        self.bounding_box_color_mode_combo.addItem("Single color", "single")
        self.bounding_box_color_mode_combo.currentIndexChanged.connect(self._update_bounding_box_color_mode_state)

        self.bounding_box_preset_combo = QComboBox(self)
        self.bounding_box_preset_combo.addItem("Custom", "")
        for label, color_value in BOX_COLOR_PRESETS:
            self.bounding_box_preset_combo.addItem(label, color_value)
        self.bounding_box_preset_combo.currentIndexChanged.connect(self._apply_selected_bounding_box_preset)

        self.bounding_box_color_edit = QLineEdit(self)
        self.bounding_box_color_edit.setPlaceholderText("#FFB347, rgb(255, 179, 71), or 1.0, 0.7, 0.28")
        self.bounding_box_color_edit.textChanged.connect(self._update_bounding_box_color_preview)

        self.bounding_box_pick_color_button = QPushButton("Pick Color...", self)
        self.bounding_box_pick_color_button.clicked.connect(self._choose_bounding_box_color)

        self.bounding_box_color_preview = QLabel(self)
        self.bounding_box_color_preview.setFixedSize(56, 24)

        bounding_box_color_row = QWidget(self)
        bounding_box_color_layout = QHBoxLayout(bounding_box_color_row)
        bounding_box_color_layout.setContentsMargins(0, 0, 0, 0)
        bounding_box_color_layout.addWidget(self.bounding_box_preset_combo)
        bounding_box_color_layout.addWidget(self.bounding_box_color_edit, 1)
        bounding_box_color_layout.addWidget(self.bounding_box_color_preview)
        bounding_box_color_layout.addWidget(self.bounding_box_pick_color_button)

        self.bounding_box_line_width_spin = QDoubleSpinBox(self)
        self.bounding_box_line_width_spin.setDecimals(1)
        self.bounding_box_line_width_spin.setRange(1.0, 10.0)
        self.bounding_box_line_width_spin.setSingleStep(0.5)
        self.bounding_box_line_width_spin.setValue(self._accepted_settings.bounding_box_line_width)
        self.bounding_box_line_width_spin.setToolTip("Line width used when drawing cluster bounding boxes.")

        self.bounding_box_show_id_check = QCheckBox("Display near each box", self)
        self.bounding_box_show_id_check.setChecked(bool(self._accepted_settings.bounding_box_show_id))

        bounding_box_help_label = QLabel(
            "Random mode assigns a distinct color to each box. Single color accepts preset names, hex (#RRGGBB), rgb(r,g,b), or r,g,b values.",
            self,
        )
        bounding_box_help_label.setWordWrap(True)
        bounding_box_help_label.setStyleSheet("color: #5a6777;")

        bounding_box_form = QFormLayout()
        bounding_box_form.addRow("Color mode:", self.bounding_box_color_mode_combo)
        bounding_box_form.addRow("Box color:", bounding_box_color_row)
        bounding_box_form.addRow("Line width:", self.bounding_box_line_width_spin)
        bounding_box_form.addRow("Cluster ID:", self.bounding_box_show_id_check)
        bounding_box_form.addRow("", bounding_box_help_label)

        bounding_box_tab = QWidget(self)
        bounding_box_layout = QVBoxLayout(bounding_box_tab)
        bounding_box_layout.setContentsMargins(0, 0, 0, 0)
        bounding_box_layout.addLayout(bounding_box_form)
        bounding_box_layout.addStretch(1)

        self.tabs.addTab(general_tab, "General")
        self.tabs.addTab(bounding_box_tab, "Bounding Boxes")

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs)
        layout.addWidget(buttons)

        self.background_edit.setText(self._accepted_settings.viewport_background)
        self._sync_background_preset_selection(self._accepted_settings.viewport_background)
        self._update_background_color_preview()

        self._set_bounding_box_color_mode(self._accepted_settings.bounding_box_color_mode)
        self.bounding_box_color_edit.setText(self._accepted_settings.bounding_box_color)
        self._sync_bounding_box_preset_selection(self._accepted_settings.bounding_box_color)
        self._update_bounding_box_color_preview()
        self._update_bounding_box_color_mode_state()

    def settings(self) -> ProjectSettings:
        return ProjectSettings(
            output_directory=self._accepted_settings.output_directory,
            point_size=self._accepted_settings.point_size,
            viewport_background=self._accepted_settings.viewport_background,
            bounding_box_color_mode=self._accepted_settings.bounding_box_color_mode,
            bounding_box_color=self._accepted_settings.bounding_box_color,
            bounding_box_line_width=self._accepted_settings.bounding_box_line_width,
            bounding_box_show_id=self._accepted_settings.bounding_box_show_id,
        )

    def _browse_output_directory(self) -> None:
        start_dir = resolve_output_directory(self.output_dir_edit.text())
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            str(start_dir),
        )
        if selected:
            self.output_dir_edit.setText(display_output_directory(Path(selected)))

    def _apply_selected_background_preset(self, _index: int) -> None:
        preset_value = self.background_preset_combo.currentData()
        if preset_value:
            self.background_edit.setText(str(preset_value))

    def _choose_background_color(self) -> None:
        initial = QColor(self.background_edit.text().strip())
        if not initial.isValid():
            try:
                initial = QColor(normalize_color_value(self.background_edit.text()))
            except ValueError:
                initial = QColor(DEFAULT_VIEWPORT_BACKGROUND_HEX)

        selected = QColorDialog.getColor(initial, self, "Select Viewport Background Color")
        if selected.isValid():
            self.background_edit.setText(selected.name().upper())

    def _sync_background_preset_selection(self, color_value: str) -> None:
        normalized = normalize_color_value(color_value)
        target_index = 0
        for index in range(1, self.background_preset_combo.count()):
            if str(self.background_preset_combo.itemData(index)).upper() == normalized:
                target_index = index
                break
        self.background_preset_combo.blockSignals(True)
        self.background_preset_combo.setCurrentIndex(target_index)
        self.background_preset_combo.blockSignals(False)

    def _update_background_color_preview(self) -> None:
        text = self.background_edit.text().strip()
        try:
            normalized = normalize_color_value(text)
        except ValueError:
            self.background_preview.setStyleSheet(
                "border: 1px solid #C44A4A; background-color: #F4D7D7;"
            )
            self.background_edit.setStyleSheet("border: 1px solid #C44A4A;")
            self.background_preset_combo.blockSignals(True)
            self.background_preset_combo.setCurrentIndex(0)
            self.background_preset_combo.blockSignals(False)
            return

        self.background_preview.setStyleSheet(
            f"border: 1px solid #768397; background-color: {normalized};"
        )
        self.background_edit.setStyleSheet("")
        self._sync_background_preset_selection(normalized)

    def _selected_bounding_box_color_mode(self) -> str:
        return _normalize_bounding_box_color_mode(self.bounding_box_color_mode_combo.currentData())

    def _set_bounding_box_color_mode(self, mode: str) -> None:
        normalized = _normalize_bounding_box_color_mode(mode)
        target_index = 0
        for index in range(self.bounding_box_color_mode_combo.count()):
            if self.bounding_box_color_mode_combo.itemData(index) == normalized:
                target_index = index
                break
        self.bounding_box_color_mode_combo.blockSignals(True)
        self.bounding_box_color_mode_combo.setCurrentIndex(target_index)
        self.bounding_box_color_mode_combo.blockSignals(False)

    def _apply_selected_bounding_box_preset(self, _index: int) -> None:
        preset_value = self.bounding_box_preset_combo.currentData()
        if preset_value:
            self.bounding_box_color_edit.setText(str(preset_value))

    def _choose_bounding_box_color(self) -> None:
        initial = QColor(self.bounding_box_color_edit.text().strip())
        if not initial.isValid():
            try:
                initial = QColor(normalize_color_value(self.bounding_box_color_edit.text()))
            except ValueError:
                initial = QColor(DEFAULT_BOUNDING_BOX_COLOR_HEX)

        selected = QColorDialog.getColor(initial, self, "Select Bounding Box Color")
        if selected.isValid():
            self.bounding_box_color_edit.setText(selected.name().upper())

    def _sync_bounding_box_preset_selection(self, color_value: str) -> None:
        normalized = normalize_color_value(color_value)
        target_index = 0
        for index in range(1, self.bounding_box_preset_combo.count()):
            if str(self.bounding_box_preset_combo.itemData(index)).upper() == normalized:
                target_index = index
                break
        self.bounding_box_preset_combo.blockSignals(True)
        self.bounding_box_preset_combo.setCurrentIndex(target_index)
        self.bounding_box_preset_combo.blockSignals(False)

    def _update_bounding_box_color_preview(self) -> None:
        text = self.bounding_box_color_edit.text().strip()
        try:
            normalized = normalize_color_value(text)
        except ValueError:
            self.bounding_box_color_preview.setStyleSheet(
                "border: 1px solid #C44A4A; background-color: #F4D7D7;"
            )
            self.bounding_box_color_edit.setStyleSheet("border: 1px solid #C44A4A;")
            self.bounding_box_preset_combo.blockSignals(True)
            self.bounding_box_preset_combo.setCurrentIndex(0)
            self.bounding_box_preset_combo.blockSignals(False)
            return

        self.bounding_box_color_preview.setStyleSheet(
            f"border: 1px solid #768397; background-color: {normalized};"
        )
        self.bounding_box_color_edit.setStyleSheet("")
        self._sync_bounding_box_preset_selection(normalized)

    def _update_bounding_box_color_mode_state(self) -> None:
        single_color_mode = self._selected_bounding_box_color_mode() == "single"
        self.bounding_box_preset_combo.setEnabled(single_color_mode)
        self.bounding_box_color_edit.setEnabled(single_color_mode)
        self.bounding_box_color_preview.setEnabled(single_color_mode)
        self.bounding_box_pick_color_button.setEnabled(single_color_mode)

    def accept(self) -> None:
        output_directory = self.output_dir_edit.text().strip()
        if not output_directory:
            self.tabs.setCurrentIndex(0)
            QMessageBox.warning(self, "Invalid Settings", "Output directory must not be empty.")
            return

        try:
            resolved_output_dir = resolve_output_directory(output_directory)
            resolved_output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            self.tabs.setCurrentIndex(0)
            QMessageBox.warning(
                self,
                "Invalid Settings",
                f"Failed to create or access the output directory:\n{exc}",
            )
            return

        try:
            normalized_background = normalize_color_value(self.background_edit.text())
        except ValueError as exc:
            self.tabs.setCurrentIndex(0)
            QMessageBox.warning(self, "Invalid Settings", f"Invalid viewport background color:\n{exc}")
            return

        try:
            normalized_bounding_box_color = normalize_color_value(
                self.bounding_box_color_edit.text() or self._accepted_settings.bounding_box_color
            )
        except ValueError as exc:
            self.tabs.setCurrentIndex(1)
            QMessageBox.warning(self, "Invalid Settings", f"Invalid bounding box color:\n{exc}")
            return

        self._accepted_settings = ProjectSettings(
            output_directory=display_output_directory(resolved_output_dir),
            point_size=float(self.point_size_spin.value()),
            viewport_background=normalized_background,
            bounding_box_color_mode=self._selected_bounding_box_color_mode(),
            bounding_box_color=normalized_bounding_box_color,
            bounding_box_line_width=_clamp_bounding_box_line_width(self.bounding_box_line_width_spin.value()),
            bounding_box_show_id=bool(self.bounding_box_show_id_check.isChecked()),
        )
        super().accept()


class PointCloudGLWidget(QOpenGLWidget):
    colorModeChanged = pyqtSignal(str)
    navigationModeChanged = pyqtSignal(str)

    COLOR_MODE_LABELS = {
        "neutral": "Neutral",
        "label": "Label",
        "rgb": "RGB",
    }
    NAVIGATION_MODE_LABELS = {
        "orbit": "Orbit",
        "game": "Game",
    }
    DEFAULT_YAW = 35.0
    DEFAULT_PITCH = -25.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cloud: Optional[PointCloudData] = None
        self._cluster_boxes: Tuple[ClusterBoundingBoxData, ...] = ()
        self._color_mode: str = "neutral"
        self._background = parse_color_value(APP_SETTINGS.viewport_background)
        self._neutral_color = np.array([0.82, 0.84, 0.88], dtype=np.float32)
        self._cluster_box_color_mode = _normalize_bounding_box_color_mode(
            APP_SETTINGS.bounding_box_color_mode
        )
        self._cluster_box_color = normalize_color_value(APP_SETTINGS.bounding_box_color)
        self._cluster_box_show_id = bool(APP_SETTINGS.bounding_box_show_id)
        self._overlay_box_colors = np.zeros((0, 3), dtype=np.float32)
        self._surface_vertices = np.zeros((0, 3), dtype=np.float32)
        self._surface_triangles = np.zeros((0, 3), dtype=np.int32)
        self._surface_render_mode = "shaded"
        self._surface_elevation_colormap = "terrain"
        self._surface_smooth_normals = True

        self._program: int = 0
        self._mesh_program: int = 0
        self._vbo_points: int = 0
        self._vbo_colors: int = 0
        self._vbo_overlay_points: int = 0
        self._vbo_overlay_colors: int = 0
        self._vbo_mesh_fill_points: int = 0
        self._vbo_mesh_fill_colors: int = 0
        self._vbo_mesh_fill_normals: int = 0
        self._vbo_mesh_line_points: int = 0
        self._vbo_mesh_line_colors: int = 0
        self._a_pos: int = -1
        self._a_color: int = -1
        self._u_mvp: int = -1
        self._u_point_size: int = -1
        self._mesh_a_pos: int = -1
        self._mesh_a_color: int = -1
        self._mesh_a_normal: int = -1
        self._mesh_u_mvp: int = -1
        self._mesh_u_light_dir: int = -1
        self._mesh_u_shading_strength: int = -1
        self._initialized = False
        self._point_count = 0
        self._overlay_vertex_count = 0
        self._mesh_fill_vertex_count = 0
        self._mesh_line_vertex_count = 0
        self._overlay_line_width = _clamp_bounding_box_line_width(APP_SETTINGS.bounding_box_line_width)

        self._fov_y_deg = 50.0
        self._scene_center = np.zeros(3, dtype=np.float32)
        self._scene_radius = 1.0
        self._pan = np.zeros(3, dtype=np.float32)
        self._yaw = self.DEFAULT_YAW
        self._pitch = self.DEFAULT_PITCH
        self._distance = 3.0
        self._last_mouse_pos = None
        self._game_position = np.zeros(3, dtype=np.float32)
        self._navigation_mode = "orbit"
        self._orbit_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self._game_mouse_sensitivity = 0.20
        self._game_pitch_limit = 85.0
        self._game_target_yaw = self._yaw
        self._game_target_pitch = self._pitch
        self._game_look_response = 12.0
        self._game_move_speed = 0.0
        self._game_velocity = np.zeros(3, dtype=np.float32)
        self._game_velocity_response = 7.5
        self._game_velocity_drag = 5.0
        self._pressed_keys: Set[int] = set()
        self._last_move_time = time.monotonic()
        self._move_timer = QTimer(self)
        self._move_timer.setInterval(16)
        self._move_timer.timeout.connect(self._update_game_movement)
        self._point_size = _clamp_point_size(APP_SETTINGS.point_size)

        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(False)

    def has_rgb_data(self) -> bool:
        return self._cloud is not None and self._cloud.has_rgb

    def has_label_data(self) -> bool:
        return self._cloud is not None and self._cloud.has_labels

    def active_color_mode(self) -> str:
        return self._color_mode

    def active_color_mode_label(self) -> str:
        return self.COLOR_MODE_LABELS.get(self._color_mode, "Unknown")

    def is_game_navigation_enabled(self) -> bool:
        return self._navigation_mode == "game"

    def active_navigation_mode_label(self) -> str:
        return self.NAVIGATION_MODE_LABELS.get(self._navigation_mode, "Unknown")

    def set_point_size(self, point_size: float) -> None:
        self._point_size = _clamp_point_size(point_size)
        self.update()

    def set_background_color(self, color: Sequence[float]) -> None:
        self._background = parse_color_value(color)
        self.update()

    def set_cluster_box_display_settings(
        self,
        *,
        color_mode: str,
        color: object,
        line_width: float,
        show_id: bool,
    ) -> None:
        normalized_mode = _normalize_bounding_box_color_mode(color_mode)
        normalized_color = normalize_color_value(color)
        normalized_line_width = _clamp_bounding_box_line_width(line_width)
        normalized_show_id = bool(show_id)

        overlay_colors_changed = (
            normalized_mode != self._cluster_box_color_mode
            or normalized_color != self._cluster_box_color
        )
        display_changed = (
            overlay_colors_changed
            or normalized_line_width != self._overlay_line_width
            or normalized_show_id != self._cluster_box_show_id
        )

        self._cluster_box_color_mode = normalized_mode
        self._cluster_box_color = normalized_color
        self._overlay_line_width = normalized_line_width
        self._cluster_box_show_id = normalized_show_id

        if overlay_colors_changed and self._initialized:
            self._upload_cluster_overlay()
        if display_changed:
            self.update()

    def set_game_navigation_enabled(self, enabled: bool) -> None:
        next_mode = "game" if enabled else "orbit"
        if next_mode == self._navigation_mode:
            return

        if next_mode == "game":
            if self._cloud is None:
                return
            self._game_position = self._current_orbit_eye()
            self._sync_game_angles_from_orbit_view()
            self._game_target_yaw = self._yaw
            self._game_target_pitch = self._pitch
            self._game_velocity.fill(0.0)
            self._pressed_keys.clear()
            self._last_move_time = time.monotonic()
            self._last_mouse_pos = None
            self.setMouseTracking(True)
            self.setCursor(Qt.CrossCursor)
            self.setFocus()
            QTimer.singleShot(0, self._center_cursor_in_view)
        else:
            forward = self._game_forward_direction()
            self._sync_orbit_angles_from_game_forward(forward)
            self._game_target_yaw = self._yaw
            self._game_target_pitch = self._pitch
            self._pressed_keys.clear()
            self._game_velocity.fill(0.0)
            self._move_timer.stop()
            self.setMouseTracking(False)
            self.unsetCursor()
            # Keep the exact world-space camera position when switching back to orbit.
            target = self._game_position - self._camera_direction() * self._distance
            self._pan = target - self._scene_center
            self._last_mouse_pos = None

        self._navigation_mode = next_mode
        self.navigationModeChanged.emit(self._navigation_mode)
        self.update()

    def _center_cursor_in_view(self) -> None:
        if not self.isVisible():
            return
        center_local = self.rect().center()
        QCursor.setPos(self.mapToGlobal(center_local))
        self._last_mouse_pos = center_local

    def set_point_cloud(self, cloud: PointCloudData) -> None:
        self.clear_surface_mesh(update=False)
        self._cloud = cloud
        self._scene_center = cloud.center.astype(np.float32)
        self._scene_radius = max(cloud.radius, 1e-4)
        self._pan = np.zeros(3, dtype=np.float32)
        self._distance = self._fit_distance()
        self._point_count = cloud.loaded_count
        self._game_move_speed = max(0.35, self._scene_radius * 0.28)
        if self._navigation_mode == "game":
            self._game_position = self._scene_center - self._game_forward_direction() * self._distance
        else:
            self._game_position = self._current_orbit_eye()
        self._game_velocity.fill(0.0)
        self._pressed_keys.clear()
        if self._navigation_mode == "game":
            self._last_move_time = time.monotonic()

        resolved_mode = self._resolve_mode(self._color_mode)
        if resolved_mode != self._color_mode:
            self._color_mode = resolved_mode
            self.colorModeChanged.emit(self._color_mode)

        if self._initialized:
            self._upload_geometry()
            self._upload_colors()
            self._upload_cluster_overlay()
        self.update()

    def has_surface_mesh(self) -> bool:
        return bool(self._surface_vertices.shape[0] > 0 and self._surface_triangles.shape[0] > 0)

    def set_surface_mesh(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        *,
        render_mode: str,
        elevation_colormap: str,
        smooth_normals: bool,
    ) -> None:
        surface_vertices = np.ascontiguousarray(np.asarray(vertices, dtype=np.float32), dtype=np.float32)
        surface_triangles = np.ascontiguousarray(np.asarray(triangles, dtype=np.int32), dtype=np.int32)
        if surface_vertices.ndim != 2 or surface_vertices.shape[1] != 3:
            raise ValueError("Surface mesh vertices must have shape (N, 3).")
        if surface_triangles.ndim != 2 or surface_triangles.shape[1] != 3:
            raise ValueError("Surface mesh triangles must have shape (M, 3).")
        if surface_vertices.shape[0] == 0 or surface_triangles.shape[0] == 0:
            raise ValueError("Surface mesh is empty.")
        if np.min(surface_triangles) < 0 or np.max(surface_triangles) >= surface_vertices.shape[0]:
            raise ValueError("Surface mesh triangle indices are out of bounds.")

        self._surface_vertices = surface_vertices
        self._surface_triangles = surface_triangles
        self._surface_render_mode = str(render_mode).strip().lower() or "shaded"
        self._surface_elevation_colormap = str(elevation_colormap).strip().lower() or "terrain"
        self._surface_smooth_normals = bool(smooth_normals)

        if self._cloud is None:
            self._scene_center = np.mean(surface_vertices, axis=0, dtype=np.float64).astype(np.float32)
            mins = np.min(surface_vertices, axis=0).astype(np.float64, copy=False)
            maxs = np.max(surface_vertices, axis=0).astype(np.float64, copy=False)
            self._scene_radius = max(1e-4, float(np.linalg.norm(maxs - mins) * 0.5))
            self._pan = np.zeros(3, dtype=np.float32)
            self._distance = self._fit_distance()
            self._game_move_speed = max(0.35, self._scene_radius * 0.28)

        if self._initialized:
            self._upload_surface_mesh()
        self.update()

    def clear_surface_mesh(self, update: bool = True) -> None:
        self._surface_vertices = np.zeros((0, 3), dtype=np.float32)
        self._surface_triangles = np.zeros((0, 3), dtype=np.int32)
        self._surface_render_mode = "shaded"
        self._surface_elevation_colormap = "terrain"
        self._surface_smooth_normals = True
        self._mesh_fill_vertex_count = 0
        self._mesh_line_vertex_count = 0
        if update:
            self.update()

    def set_cluster_boxes(self, boxes: Sequence[ClusterBoundingBoxData]) -> None:
        self._cluster_boxes = tuple(boxes)
        if self._initialized:
            self._upload_cluster_overlay()
        self.update()

    def clear_viewport(self) -> None:
        if self._navigation_mode == "game":
            self.set_game_navigation_enabled(False)

        self._cloud = None
        self._cluster_boxes = ()
        self.clear_surface_mesh(update=False)
        self._point_count = 0
        self._overlay_vertex_count = 0
        self._overlay_box_colors = np.zeros((0, 3), dtype=np.float32)
        self._color_mode = "neutral"
        self._scene_center = np.zeros(3, dtype=np.float32)
        self._scene_radius = 1.0
        self._pan = np.zeros(3, dtype=np.float32)
        self._yaw = self.DEFAULT_YAW
        self._pitch = self.DEFAULT_PITCH
        self._distance = 3.0
        self._game_position = np.zeros(3, dtype=np.float32)
        self._game_target_yaw = self._yaw
        self._game_target_pitch = self._pitch
        self._game_move_speed = 0.0
        self._game_velocity.fill(0.0)
        self._pressed_keys.clear()
        self._last_mouse_pos = None
        self._last_move_time = time.monotonic()
        self._orbit_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self._move_timer.stop()
        self.setMouseTracking(False)
        self.unsetCursor()
        self.colorModeChanged.emit(self._color_mode)
        self.update()

    def set_color_mode(self, mode: str) -> None:
        resolved_mode = self._resolve_mode(mode)
        if resolved_mode == self._color_mode and self._cloud is not None:
            return
        self._color_mode = resolved_mode
        if self._initialized and self._cloud is not None:
            self._upload_colors()
        self.colorModeChanged.emit(self._color_mode)
        self.update()

    def toggle_rgb_mode(self) -> None:
        if not self.has_rgb_data():
            return
        if self._color_mode == "rgb":
            fallback = "label" if self.has_label_data() else "neutral"
            self.set_color_mode(fallback)
        else:
            self.set_color_mode("rgb")

    def fit_to_view(self) -> None:
        if self._cloud is None:
            return
        self._pan = np.zeros(3, dtype=np.float32)
        self._scene_center = self._cloud.center.astype(np.float32)
        self._distance = self._fit_distance()
        if self._navigation_mode == "game":
            self._game_position = self._scene_center - self._game_forward_direction() * self._distance
        self.update()

    def reset_view(self) -> None:
        if self._cloud is None:
            return
        self._yaw = self.DEFAULT_YAW
        self._pitch = self.DEFAULT_PITCH
        self._game_target_yaw = self._yaw
        self._game_target_pitch = self._pitch
        self._orbit_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.fit_to_view()

    def set_view_top(self) -> None:
        # Z-up top view: camera above cloud, looking downward along -Z.
        self._set_projection_view(yaw=0.0, pitch=0.0, up_axis="y")

    def set_view_front(self) -> None:
        # Front view for Z-up scenes: looking from +Y toward -Y.
        self._set_projection_view(yaw=0.0, pitch=90.0, up_axis="z")

    def set_view_left(self) -> None:
        self._set_projection_view(yaw=-90.0, pitch=0.0, up_axis="z")

    def set_view_right(self) -> None:
        self._set_projection_view(yaw=90.0, pitch=0.0, up_axis="z")

    def set_view_back(self) -> None:
        # Back view for Z-up scenes: looking from -Y toward +Y.
        self._set_projection_view(yaw=0.0, pitch=-90.0, up_axis="z")

    def set_view_bottom(self) -> None:
        # Z-up bottom view: camera below cloud, looking upward along +Z.
        self._set_projection_view(yaw=180.0, pitch=0.0, up_axis="y")

    def set_view_front_isometric(self) -> None:
        # Front-right-top isometric view for Z-up scenes.
        self._set_projection_view(yaw=45.0, pitch=35.264, up_axis="z")

    def initializeGL(self) -> None:
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_PROGRAM_POINT_SIZE)

        point_vertex_shader = """
        #version 120
        attribute vec3 a_pos;
        attribute vec3 a_color;
        uniform mat4 u_mvp;
        uniform float u_point_size;
        varying vec3 v_color;
        void main() {
            gl_Position = u_mvp * vec4(a_pos, 1.0);
            gl_PointSize = u_point_size;
            v_color = a_color;
        }
        """

        point_fragment_shader = """
        #version 120
        varying vec3 v_color;
        void main() {
            gl_FragColor = vec4(v_color, 1.0);
        }
        """

        mesh_vertex_shader = """
        #version 120
        attribute vec3 a_pos;
        attribute vec3 a_color;
        attribute vec3 a_normal;
        uniform mat4 u_mvp;
        varying vec3 v_color;
        varying vec3 v_normal;
        void main() {
            gl_Position = u_mvp * vec4(a_pos, 1.0);
            v_color = a_color;
            v_normal = a_normal;
        }
        """

        mesh_fragment_shader = """
        #version 120
        varying vec3 v_color;
        varying vec3 v_normal;
        uniform vec3 u_light_dir;
        uniform float u_shading_strength;
        void main() {
            float diffuse = max(dot(normalize(v_normal), normalize(u_light_dir)), 0.0);
            float lit = mix(1.0, 0.25 + 0.75 * diffuse, clamp(u_shading_strength, 0.0, 1.0));
            gl_FragColor = vec4(v_color * lit, 1.0);
        }
        """

        self._program = compileProgram(
            compileShader(point_vertex_shader, GL_VERTEX_SHADER),
            compileShader(point_fragment_shader, GL_FRAGMENT_SHADER),
        )
        self._a_pos = glGetAttribLocation(self._program, "a_pos")
        self._a_color = glGetAttribLocation(self._program, "a_color")
        self._u_mvp = glGetUniformLocation(self._program, "u_mvp")
        self._u_point_size = glGetUniformLocation(self._program, "u_point_size")

        self._mesh_program = compileProgram(
            compileShader(mesh_vertex_shader, GL_VERTEX_SHADER),
            compileShader(mesh_fragment_shader, GL_FRAGMENT_SHADER),
        )
        self._mesh_a_pos = glGetAttribLocation(self._mesh_program, "a_pos")
        self._mesh_a_color = glGetAttribLocation(self._mesh_program, "a_color")
        self._mesh_a_normal = glGetAttribLocation(self._mesh_program, "a_normal")
        self._mesh_u_mvp = glGetUniformLocation(self._mesh_program, "u_mvp")
        self._mesh_u_light_dir = glGetUniformLocation(self._mesh_program, "u_light_dir")
        self._mesh_u_shading_strength = glGetUniformLocation(self._mesh_program, "u_shading_strength")

        self._vbo_points = glGenBuffers(1)
        self._vbo_colors = glGenBuffers(1)
        self._vbo_overlay_points = glGenBuffers(1)
        self._vbo_overlay_colors = glGenBuffers(1)
        self._vbo_mesh_fill_points = glGenBuffers(1)
        self._vbo_mesh_fill_colors = glGenBuffers(1)
        self._vbo_mesh_fill_normals = glGenBuffers(1)
        self._vbo_mesh_line_points = glGenBuffers(1)
        self._vbo_mesh_line_colors = glGenBuffers(1)

        self._initialized = True
        if self._cloud is not None:
            self._upload_geometry()
            self._upload_colors()
        if self.has_surface_mesh():
            self._upload_surface_mesh()
        self._upload_cluster_overlay()

        context = self.context()
        if context is not None:
            context.aboutToBeDestroyed.connect(self._cleanup_gl)

    def resizeGL(self, width: int, height: int) -> None:
        glViewport(0, 0, max(1, width), max(1, height))

    def paintGL(self) -> None:
        glViewport(0, 0, max(1, self.width()), max(1, self.height()))
        glClearColor(*self._background, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        show_surface_mesh = self.has_surface_mesh()
        show_point_cloud = self._cloud is not None and self._point_count > 0 and not show_surface_mesh
        if not self._initialized or (not show_point_cloud and not show_surface_mesh):
            return

        aspect = max(1e-4, float(self.width()) / max(1.0, float(self.height())))
        near = max(0.01, self._distance * 0.01)

        if self._navigation_mode == "game":
            eye = self._game_position
            forward = self._game_forward_direction()
            target = eye + forward
            dist_to_center = float(np.linalg.norm(eye - self._scene_center))
            far = max(near + 1.0, dist_to_center + self._scene_radius * 8.0 + 10.0)
            proj = perspective_matrix(self._fov_y_deg, aspect, near, far)
            view = look_at_matrix(eye, target, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        else:
            far = max(near + 1.0, self._distance + self._scene_radius * 8.0 + 10.0)
            proj = perspective_matrix(self._fov_y_deg, aspect, near, far)
            target = self._scene_center + self._pan
            eye = target + self._camera_direction() * self._distance
            view = look_at_matrix(eye, target, self._orbit_up)
        mvp = proj @ view

        if show_point_cloud:
            glUseProgram(self._program)
            glUniformMatrix4fv(self._u_mvp, 1, GL_FALSE, mvp.T)
            glUniform1f(self._u_point_size, float(self._point_size))

            glBindBuffer(GL_ARRAY_BUFFER, self._vbo_points)
            glEnableVertexAttribArray(self._a_pos)
            glVertexAttribPointer(self._a_pos, 3, GL_FLOAT, GL_FALSE, 0, None)

            glBindBuffer(GL_ARRAY_BUFFER, self._vbo_colors)
            glEnableVertexAttribArray(self._a_color)
            glVertexAttribPointer(self._a_color, 3, GL_FLOAT, GL_FALSE, 0, None)

            glDrawArrays(GL_POINTS, 0, self._point_count)
            glDisableVertexAttribArray(self._a_pos)
            glDisableVertexAttribArray(self._a_color)
            glBindBuffer(GL_ARRAY_BUFFER, 0)

        if show_surface_mesh and self._surface_render_mode in {"solid", "shaded"} and self._mesh_fill_vertex_count > 0:
            glUseProgram(self._mesh_program)
            glUniformMatrix4fv(self._mesh_u_mvp, 1, GL_FALSE, mvp.T)
            glUniform3f(self._mesh_u_light_dir, 0.35, 0.45, 0.82)
            glUniform1f(self._mesh_u_shading_strength, 1.0 if self._surface_render_mode == "shaded" else 0.0)

            glBindBuffer(GL_ARRAY_BUFFER, self._vbo_mesh_fill_points)
            glEnableVertexAttribArray(self._mesh_a_pos)
            glVertexAttribPointer(self._mesh_a_pos, 3, GL_FLOAT, GL_FALSE, 0, None)

            glBindBuffer(GL_ARRAY_BUFFER, self._vbo_mesh_fill_colors)
            glEnableVertexAttribArray(self._mesh_a_color)
            glVertexAttribPointer(self._mesh_a_color, 3, GL_FLOAT, GL_FALSE, 0, None)

            glBindBuffer(GL_ARRAY_BUFFER, self._vbo_mesh_fill_normals)
            glEnableVertexAttribArray(self._mesh_a_normal)
            glVertexAttribPointer(self._mesh_a_normal, 3, GL_FLOAT, GL_FALSE, 0, None)

            glDrawArrays(GL_TRIANGLES, 0, self._mesh_fill_vertex_count)
            glDisableVertexAttribArray(self._mesh_a_pos)
            glDisableVertexAttribArray(self._mesh_a_color)
            glDisableVertexAttribArray(self._mesh_a_normal)
            glBindBuffer(GL_ARRAY_BUFFER, 0)

        if show_surface_mesh and self._surface_render_mode == "wireframe" and self._mesh_line_vertex_count > 0:
            glUseProgram(self._program)
            glUniformMatrix4fv(self._u_mvp, 1, GL_FALSE, mvp.T)
            glUniform1f(self._u_point_size, 1.0)

            glBindBuffer(GL_ARRAY_BUFFER, self._vbo_mesh_line_points)
            glEnableVertexAttribArray(self._a_pos)
            glVertexAttribPointer(self._a_pos, 3, GL_FLOAT, GL_FALSE, 0, None)

            glBindBuffer(GL_ARRAY_BUFFER, self._vbo_mesh_line_colors)
            glEnableVertexAttribArray(self._a_color)
            glVertexAttribPointer(self._a_color, 3, GL_FLOAT, GL_FALSE, 0, None)

            glDrawArrays(GL_LINES, 0, self._mesh_line_vertex_count)
            glDisableVertexAttribArray(self._a_pos)
            glDisableVertexAttribArray(self._a_color)
            glBindBuffer(GL_ARRAY_BUFFER, 0)

        if self._overlay_vertex_count > 0:
            glDisable(GL_DEPTH_TEST)
            glLineWidth(float(self._overlay_line_width))

            glUseProgram(self._program)
            glUniformMatrix4fv(self._u_mvp, 1, GL_FALSE, mvp.T)
            glUniform1f(self._u_point_size, 1.0)

            glBindBuffer(GL_ARRAY_BUFFER, self._vbo_overlay_points)
            glEnableVertexAttribArray(self._a_pos)
            glVertexAttribPointer(self._a_pos, 3, GL_FLOAT, GL_FALSE, 0, None)

            glBindBuffer(GL_ARRAY_BUFFER, self._vbo_overlay_colors)
            glEnableVertexAttribArray(self._a_color)
            glVertexAttribPointer(self._a_color, 3, GL_FLOAT, GL_FALSE, 0, None)

            glDrawArrays(GL_LINES, 0, self._overlay_vertex_count)
            glDisableVertexAttribArray(self._a_pos)
            glDisableVertexAttribArray(self._a_color)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glEnable(GL_DEPTH_TEST)

        glUseProgram(0)
        self._paint_cluster_box_labels(mvp)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._last_mouse_pos = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._navigation_mode == "game":
            if self._cloud is None:
                super().mouseMoveEvent(event)
                return

            current = event.pos()
            if self._last_mouse_pos is None:
                self._last_mouse_pos = current
                super().mouseMoveEvent(event)
                return

            dx = float(current.x() - self._last_mouse_pos.x())
            dy = float(current.y() - self._last_mouse_pos.y())
            self._game_target_yaw += dx * self._game_mouse_sensitivity
            self._game_target_pitch += dy * self._game_mouse_sensitivity
            self._game_target_pitch = max(
                -self._game_pitch_limit,
                min(self._game_pitch_limit, self._game_target_pitch),
            )
            if not self._move_timer.isActive():
                self._last_move_time = time.monotonic()
                self._move_timer.start()
            self._last_mouse_pos = current
            self.update()
            super().mouseMoveEvent(event)
            return

        if self._cloud is None or self._last_mouse_pos is None:
            self._last_mouse_pos = event.pos()
            super().mouseMoveEvent(event)
            return

        current = event.pos()
        dx = float(current.x() - self._last_mouse_pos.x())
        dy = float(current.y() - self._last_mouse_pos.y())

        buttons = event.buttons()
        if buttons & Qt.LeftButton:
            if event.modifiers() & Qt.ShiftModifier:
                self._apply_pan(dx, dy)
            else:
                self._yaw -= dx * 0.35
                self._pitch -= dy * 0.35
                self._pitch = max(-89.0, min(89.0, self._pitch))
        elif buttons & Qt.MidButton:
            self._apply_pan(dx, dy)
        elif buttons & Qt.RightButton:
            self._apply_dolly_drag(dy)

        self._last_mouse_pos = current
        self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._last_mouse_pos = None
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._cloud is None:
            super().wheelEvent(event)
            return
        if self._navigation_mode == "game":
            super().wheelEvent(event)
            return
        steps = event.angleDelta().y() / 120.0
        scale = pow(0.88, steps)
        self._distance = max(1e-3, self._distance * scale)
        self.update()
        super().wheelEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if self._navigation_mode != "game" or self._cloud is None:
            super().keyPressEvent(event)
            return

        if event.isAutoRepeat():
            return

        key = event.key()
        if key in {Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D, Qt.Key_Space, Qt.Key_Control}:
            self._pressed_keys.add(int(key))
            self._last_move_time = time.monotonic()
            if not self._move_timer.isActive():
                self._move_timer.start()
            event.accept()
            return

        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if self._navigation_mode != "game":
            super().keyReleaseEvent(event)
            return

        if event.isAutoRepeat():
            return

        key = int(event.key())
        if key in self._pressed_keys:
            self._pressed_keys.discard(key)
            event.accept()
            return

        super().keyReleaseEvent(event)

    def focusOutEvent(self, event) -> None:
        self._pressed_keys.clear()
        self._game_velocity.fill(0.0)
        self._move_timer.stop()
        super().focusOutEvent(event)

    def _camera_direction(self) -> np.ndarray:
        yaw_rad = np.radians(self._yaw)
        pitch_rad = np.radians(self._pitch)
        x = np.cos(pitch_rad) * np.sin(yaw_rad)
        y = np.sin(pitch_rad)
        z = np.cos(pitch_rad) * np.cos(yaw_rad)
        return np.array([x, y, z], dtype=np.float32)

    def _game_forward_direction(self) -> np.ndarray:
        yaw_rad = np.radians(self._yaw)
        pitch_rad = np.radians(self._pitch)
        x = np.sin(yaw_rad) * np.cos(pitch_rad)
        y = np.cos(yaw_rad) * np.cos(pitch_rad)
        z = np.sin(pitch_rad)
        return normalize_vector(np.array([x, y, z], dtype=np.float32))

    def _current_orbit_eye(self) -> np.ndarray:
        target = self._scene_center + self._pan
        return target + self._camera_direction() * self._distance

    def _sync_game_angles_from_orbit_view(self) -> None:
        orbit_target = self._scene_center + self._pan
        forward = normalize_vector(orbit_target - self._game_position)
        self._pitch = float(np.degrees(np.arcsin(np.clip(forward[2], -1.0, 1.0))))
        self._pitch = max(-self._game_pitch_limit, min(self._game_pitch_limit, self._pitch))
        self._yaw = float(np.degrees(np.arctan2(forward[0], forward[1])))
        self._game_target_yaw = self._yaw
        self._game_target_pitch = self._pitch

    def _sync_orbit_angles_from_game_forward(self, forward: np.ndarray) -> None:
        camera_dir = normalize_vector(-forward)
        self._pitch = float(np.degrees(np.arcsin(np.clip(camera_dir[1], -1.0, 1.0))))
        self._pitch = max(-89.0, min(89.0, self._pitch))
        self._yaw = float(np.degrees(np.arctan2(camera_dir[0], camera_dir[2])))
        self._game_target_yaw = self._yaw
        self._game_target_pitch = self._pitch

    def _orbit_camera_basis(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        forward = normalize_vector(-self._camera_direction())
        world_up = self._orbit_up
        right = normalize_vector(np.cross(forward, world_up))
        if float(np.linalg.norm(right)) <= 1e-7:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        up = normalize_vector(np.cross(right, forward))
        return forward, right, up

    def _game_camera_basis(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        forward = self._game_forward_direction()
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = normalize_vector(np.cross(forward, world_up))
        if float(np.linalg.norm(right)) <= 1e-7:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        up = normalize_vector(np.cross(right, forward))
        return forward, right, up

    def _update_game_movement(self) -> None:
        if self._navigation_mode != "game" or self._cloud is None:
            self._move_timer.stop()
            return

        now = time.monotonic()
        dt = max(0.0, min(0.2, now - self._last_move_time))
        self._last_move_time = now
        if dt <= 0.0:
            return

        look_response = max(0.0, float(self._game_look_response))
        look_alpha = 1.0 - np.exp(-look_response * float(dt))
        self._yaw += (self._game_target_yaw - self._yaw) * float(look_alpha)
        self._pitch += (self._game_target_pitch - self._pitch) * float(look_alpha)
        self._pitch = max(-self._game_pitch_limit, min(self._game_pitch_limit, self._pitch))

        look_eps = 1e-3
        if abs(self._game_target_yaw - self._yaw) < look_eps:
            self._yaw = self._game_target_yaw
        if abs(self._game_target_pitch - self._pitch) < look_eps:
            self._pitch = self._game_target_pitch

        look_pending = (
            abs(self._game_target_yaw - self._yaw) >= look_eps
            or abs(self._game_target_pitch - self._pitch) >= look_eps
        )

        forward, right, _ = self._game_camera_basis()
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        move = np.zeros(3, dtype=np.float32)
        if Qt.Key_W in self._pressed_keys:
            move += forward
        if Qt.Key_S in self._pressed_keys:
            move -= forward
        if Qt.Key_A in self._pressed_keys:
            move -= right
        if Qt.Key_D in self._pressed_keys:
            move += right
        if Qt.Key_Space in self._pressed_keys:
            if Qt.Key_Control in self._pressed_keys:
                move -= world_up
            else:
                move += world_up

        norm = float(np.linalg.norm(move))
        if norm > 1e-8:
            move = move / norm
            target_velocity = move * self._game_move_speed
        else:
            target_velocity = np.zeros(3, dtype=np.float32)

        response = max(0.0, float(self._game_velocity_response))
        alpha = 1.0 - np.exp(-response * float(dt))
        self._game_velocity += (target_velocity - self._game_velocity) * float(alpha)

        if norm <= 1e-8:
            drag = np.exp(-max(0.0, float(self._game_velocity_drag)) * float(dt))
            self._game_velocity *= float(drag)

        speed_now = float(np.linalg.norm(self._game_velocity))
        if norm <= 1e-8 and speed_now < 1e-3 and not look_pending:
            self._game_velocity.fill(0.0)
            self._move_timer.stop()
            return

        self._game_position += self._game_velocity * float(dt)
        self.update()

    def _apply_pan(self, dx: float, dy: float) -> None:
        _, right, up = self._orbit_camera_basis()
        h = max(1.0, float(self.height()))
        world_per_pixel = 2.0 * self._distance * np.tan(np.radians(self._fov_y_deg) * 0.5) / h
        self._pan += (dx * world_per_pixel) * right + (-dy * world_per_pixel) * up

    def _apply_dolly_drag(self, dy: float) -> None:
        # Inverted RMB drag zoom: up -> zoom out, down -> zoom in.
        scale = pow(1.01, float(-dy))
        self._distance = max(1e-3, self._distance * float(scale))

    def _resolve_mode(self, mode: str) -> str:
        if self._cloud is None:
            return "neutral"
        mode = mode.lower()
        if mode == "rgb" and self._cloud.has_rgb:
            return "rgb"
        if mode == "label" and self._cloud.has_labels:
            return "label"
        if self._cloud.has_labels:
            return "label"
        if self._cloud.has_rgb:
            return "rgb"
        return "neutral"

    def _set_projection_view(self, yaw: float, pitch: float, up_axis: str = "y") -> None:
        if self._cloud is None:
            return
        self._yaw = float(yaw)
        self._pitch = max(-90.0, min(90.0, float(pitch)))
        self._game_target_yaw = self._yaw
        self._game_target_pitch = self._pitch
        if up_axis == "z":
            self._orbit_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            self._orbit_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.fit_to_view()

    def _fit_distance(self) -> float:
        radius = max(self._scene_radius, 1e-4)
        aspect = max(1e-4, float(self.width()) / max(1.0, float(self.height())))
        fov_y = np.radians(self._fov_y_deg)
        fov_x = 2.0 * np.arctan(np.tan(fov_y * 0.5) * aspect)
        min_half_fov = max(1e-3, min(fov_x, fov_y) * 0.5)
        distance = radius / np.sin(min_half_fov)
        return float(distance * 1.1)

    def _current_cluster_box_colors(self) -> np.ndarray:
        count = len(self._cluster_boxes)
        if count <= 0:
            return np.zeros((0, 3), dtype=np.float32)
        if self._cluster_box_color_mode == "single":
            color = np.asarray(parse_color_value(self._cluster_box_color), dtype=np.float32).reshape(1, 3)
            return np.repeat(color, count, axis=0)
        return generate_distinct_palette(count)

    def _project_world_to_screen(
        self,
        point: Sequence[float],
        mvp: np.ndarray,
    ) -> Optional[Tuple[float, float]]:
        clip = mvp @ np.array([float(point[0]), float(point[1]), float(point[2]), 1.0], dtype=np.float32)
        w = float(clip[3])
        if w <= 1e-6:
            return None
        ndc = clip[:3] / w
        if not np.all(np.isfinite(ndc)):
            return None
        if float(ndc[2]) < -1.0 or float(ndc[2]) > 1.0:
            return None
        x = (float(ndc[0]) * 0.5 + 0.5) * float(self.width())
        y = (1.0 - (float(ndc[1]) * 0.5 + 0.5)) * float(self.height())
        return x, y

    def _paint_cluster_box_labels(self, mvp: np.ndarray) -> None:
        if not self._cluster_box_show_id or not self._cluster_boxes:
            return
        if self._overlay_box_colors.shape[0] != len(self._cluster_boxes):
            self._overlay_box_colors = np.ascontiguousarray(self._current_cluster_box_colors(), dtype=np.float32)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        metrics = painter.fontMetrics()
        viewport_rect = QRectF(0.0, 0.0, float(self.width()), float(self.height()))

        for index, box in enumerate(self._cluster_boxes):
            anchor = 0.5 * (
                np.asarray(box.min_corner, dtype=np.float32) + np.asarray(box.max_corner, dtype=np.float32)
            )
            projected = self._project_world_to_screen(anchor, mvp)
            if projected is None:
                continue

            x, y = projected
            text = f"ID {box.cluster_id}"
            text_width = float(metrics.horizontalAdvance(text))
            text_height = float(metrics.height())
            rect = QRectF(x + 8.0, y - text_height - 10.0, text_width + 10.0, text_height + 6.0)
            rect.moveLeft(min(max(6.0, rect.left()), max(6.0, viewport_rect.width() - rect.width() - 6.0)))
            rect.moveTop(min(max(6.0, rect.top()), max(6.0, viewport_rect.height() - rect.height() - 6.0)))

            painter.fillRect(rect, QColor(10, 14, 20, 180))
            painter.setPen(QColor(255, 255, 255, 38))
            painter.drawRect(rect)
            painter.setPen(QColor(color_to_hex(self._overlay_box_colors[index])))
            baseline_x = int(round(rect.x() + 5.0))
            baseline_y = int(round(rect.y() + metrics.ascent() + 3.0))
            painter.drawText(baseline_x, baseline_y, text)

        painter.end()

    def _build_color_array(self) -> np.ndarray:
        if self._cloud is None:
            return np.zeros((0, 3), dtype=np.float32)

        if self._color_mode == "rgb" and self._cloud.has_rgb:
            assert self._cloud.rgb is not None
            return np.ascontiguousarray(self._cloud.rgb, dtype=np.float32)
        if self._color_mode == "label" and self._cloud.has_labels:
            return self._cloud.label_colors()
        return np.tile(self._neutral_color, (self._cloud.loaded_count, 1)).astype(np.float32, copy=False)

    def _upload_geometry(self) -> None:
        if not self._initialized or self._cloud is None:
            return

        points = np.ascontiguousarray(self._cloud.points, dtype=np.float32)
        self._point_count = points.shape[0]

        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_points)
        glBufferData(GL_ARRAY_BUFFER, points.nbytes, points, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def _upload_colors(self) -> None:
        if not self._initialized or self._cloud is None:
            return

        colors = np.ascontiguousarray(self._build_color_array(), dtype=np.float32)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_colors)
        glBufferData(GL_ARRAY_BUFFER, colors.nbytes, colors, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def _upload_cluster_overlay(self) -> None:
        if not self._initialized:
            return

        if not self._cluster_boxes:
            self._overlay_vertex_count = 0
            self._overlay_box_colors = np.zeros((0, 3), dtype=np.float32)
            return

        box_colors = np.ascontiguousarray(self._current_cluster_box_colors(), dtype=np.float32)
        self._overlay_box_colors = box_colors
        all_vertices: List[np.ndarray] = []
        all_colors: List[np.ndarray] = []
        for index, box in enumerate(self._cluster_boxes):
            vertices = build_bounding_box_line_vertices(box.min_corner, box.max_corner)
            color = np.tile(box_colors[index], (vertices.shape[0], 1)).astype(np.float32, copy=False)
            all_vertices.append(vertices)
            all_colors.append(color)

        overlay_vertices = np.ascontiguousarray(np.vstack(all_vertices), dtype=np.float32)
        overlay_colors = np.ascontiguousarray(np.vstack(all_colors), dtype=np.float32)
        self._overlay_vertex_count = int(overlay_vertices.shape[0])

        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_overlay_points)
        glBufferData(GL_ARRAY_BUFFER, overlay_vertices.nbytes, overlay_vertices, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_overlay_colors)
        glBufferData(GL_ARRAY_BUFFER, overlay_colors.nbytes, overlay_colors, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def _surface_vertex_colors(self) -> np.ndarray:
        if not self.has_surface_mesh():
            return np.zeros((0, 3), dtype=np.float32)
        return np.ascontiguousarray(
            apply_elevation_colormap(self._surface_elevation_colormap, self._surface_vertices[:, 2]),
            dtype=np.float32,
        )

    def _upload_surface_mesh(self) -> None:
        if not self._initialized:
            return
        if not self.has_surface_mesh():
            self._mesh_fill_vertex_count = 0
            self._mesh_line_vertex_count = 0
            return

        tri_vertices = self._surface_vertices[self._surface_triangles]
        tri_normals = np.cross(tri_vertices[:, 1] - tri_vertices[:, 0], tri_vertices[:, 2] - tri_vertices[:, 0])
        tri_norms = np.linalg.norm(tri_normals, axis=1, keepdims=True)
        valid = tri_norms[:, 0] > 1e-8
        tri_normals = tri_normals.astype(np.float32, copy=False)
        tri_normals[valid] = tri_normals[valid] / tri_norms[valid]
        tri_normals[~valid] = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        vertex_colors = self._surface_vertex_colors()
        fill_points = np.ascontiguousarray(tri_vertices.reshape(-1, 3), dtype=np.float32)
        fill_colors = np.ascontiguousarray(vertex_colors[self._surface_triangles].reshape(-1, 3), dtype=np.float32)

        if self._surface_smooth_normals:
            vertex_normals = np.zeros_like(self._surface_vertices, dtype=np.float64)
            np.add.at(vertex_normals, self._surface_triangles[:, 0], tri_normals.astype(np.float64))
            np.add.at(vertex_normals, self._surface_triangles[:, 1], tri_normals.astype(np.float64))
            np.add.at(vertex_normals, self._surface_triangles[:, 2], tri_normals.astype(np.float64))
            normal_lengths = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
            normal_mask = normal_lengths[:, 0] > 1e-8
            vertex_normals[normal_mask] /= normal_lengths[normal_mask]
            vertex_normals[~normal_mask] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            fill_normals = np.ascontiguousarray(
                vertex_normals[self._surface_triangles].reshape(-1, 3),
                dtype=np.float32,
            )
        else:
            fill_normals = np.ascontiguousarray(
                np.repeat(tri_normals[:, None, :], 3, axis=1).reshape(-1, 3),
                dtype=np.float32,
            )

        line_points = np.ascontiguousarray(
            tri_vertices[:, [0, 1, 1, 2, 2, 0], :].reshape(-1, 3),
            dtype=np.float32,
        )
        line_colors = np.ascontiguousarray(
            vertex_colors[self._surface_triangles][:, [0, 1, 1, 2, 2, 0], :].reshape(-1, 3),
            dtype=np.float32,
        )

        self._mesh_fill_vertex_count = int(fill_points.shape[0])
        self._mesh_line_vertex_count = int(line_points.shape[0])

        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_mesh_fill_points)
        glBufferData(GL_ARRAY_BUFFER, fill_points.nbytes, fill_points, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_mesh_fill_colors)
        glBufferData(GL_ARRAY_BUFFER, fill_colors.nbytes, fill_colors, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_mesh_fill_normals)
        glBufferData(GL_ARRAY_BUFFER, fill_normals.nbytes, fill_normals, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_mesh_line_points)
        glBufferData(GL_ARRAY_BUFFER, line_points.nbytes, line_points, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo_mesh_line_colors)
        glBufferData(GL_ARRAY_BUFFER, line_colors.nbytes, line_colors, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def _cleanup_gl(self) -> None:
        self.makeCurrent()
        if self._vbo_points:
            glDeleteBuffers(1, [self._vbo_points])
            self._vbo_points = 0
        if self._vbo_colors:
            glDeleteBuffers(1, [self._vbo_colors])
            self._vbo_colors = 0
        if self._vbo_overlay_points:
            glDeleteBuffers(1, [self._vbo_overlay_points])
            self._vbo_overlay_points = 0
        if self._vbo_overlay_colors:
            glDeleteBuffers(1, [self._vbo_overlay_colors])
            self._vbo_overlay_colors = 0
        if self._vbo_mesh_fill_points:
            glDeleteBuffers(1, [self._vbo_mesh_fill_points])
            self._vbo_mesh_fill_points = 0
        if self._vbo_mesh_fill_colors:
            glDeleteBuffers(1, [self._vbo_mesh_fill_colors])
            self._vbo_mesh_fill_colors = 0
        if self._vbo_mesh_fill_normals:
            glDeleteBuffers(1, [self._vbo_mesh_fill_normals])
            self._vbo_mesh_fill_normals = 0
        if self._vbo_mesh_line_points:
            glDeleteBuffers(1, [self._vbo_mesh_line_points])
            self._vbo_mesh_line_points = 0
        if self._vbo_mesh_line_colors:
            glDeleteBuffers(1, [self._vbo_mesh_line_colors])
            self._vbo_mesh_line_colors = 0
        if self._program:
            glDeleteProgram(self._program)
            self._program = 0
        if self._mesh_program:
            glDeleteProgram(self._mesh_program)
            self._mesh_program = 0
        self.doneCurrent()
        self._initialized = False


class MainWindow(QMainWindow):
    def __init__(self, max_points: int = DEFAULT_MAX_POINTS):
        super().__init__()
        self.max_points = max_points
        self._settings = ProjectSettings(
            output_directory=APP_SETTINGS.output_directory,
            point_size=APP_SETTINGS.point_size,
            viewport_background=APP_SETTINGS.viewport_background,
            bounding_box_color_mode=APP_SETTINGS.bounding_box_color_mode,
            bounding_box_color=APP_SETTINGS.bounding_box_color,
            bounding_box_line_width=APP_SETTINGS.bounding_box_line_width,
            bounding_box_show_id=APP_SETTINGS.bounding_box_show_id,
        )
        self.current_cloud: Optional[PointCloudData] = None
        self._split_module = None
        self._synthetic_module = None
        self._dbscan_module = None
        self._tin_module = None
        self._cluster_boxes: Tuple[ClusterBoundingBoxData, ...] = ()
        self._cluster_source_path: str = ""
        self._last_generation_params = SyntheticGenerationParams()
        self._last_split_prefix = "split"
        self._last_split_dir = str(ensure_data_dir())
        self._last_dbscan_epsilon = 1.0
        self._last_dbscan_min_pts = 8
        self._last_dbscan_output_path = str(ensure_data_dir() / "dbscan_clusters.yaml")
        self._last_tin_params = TINCommandParams()
        self._last_tin_mesh: Optional[TINMesh] = None
        self._last_cluster_yaml_dir = str(ensure_data_dir())
        self._last_view_image_dir = str(ensure_data_dir())
        self._last_mesh_export_dir = str(ensure_data_dir())

        self.setWindowTitle("MagicPoints")
        self.resize(1200, 800)
        self.setWindowIcon(self._icon("app", QStyle.SP_ComputerIcon))

        self.gl_widget = PointCloudGLWidget(self)
        self.setCentralWidget(self.gl_widget)
        self.gl_widget.colorModeChanged.connect(self._on_color_mode_changed)
        self.gl_widget.navigationModeChanged.connect(self._on_navigation_mode_changed)

        self._ensure_settings_file_exists()
        self._apply_project_settings(self._settings, persist=False, update_recent_paths=False)
        self._create_actions()
        self._create_menus()
        self._create_toolbar()
        self.statusBar().showMessage("No file loaded")

    def _icon(self, name: str, fallback: QStyle.StandardPixmap) -> QIcon:
        icon_path = ICONS_DIR / f"{name}.svg"
        if icon_path.exists():
            return QIcon(str(icon_path))
        return self.style().standardIcon(fallback)

    def _create_actions(self) -> None:
        self.open_action = QAction(
            self._icon("open", QStyle.SP_DialogOpenButton),
            "Open Point Cloud File",
            self,
        )
        self.open_action.setShortcut("Ctrl+O")
        self.open_action.setToolTip("Open TXT or PLY point cloud")
        self.open_action.triggered.connect(self.open_file_dialog)

        self.open_clusters_action = QAction(
            self._icon("open_clusters", QStyle.SP_DirOpenIcon),
            "Open Clusters File",
            self,
        )
        self.open_clusters_action.setToolTip("Load a clusters file with DBSCAN cluster bounding boxes")
        self.open_clusters_action.triggered.connect(self.open_clusters_file_dialog)

        self.settings_action = QAction(
            self._icon("settings", QStyle.SP_FileDialogDetailedView),
            "Settings",
            self,
        )
        self.settings_action.setToolTip("Open project settings")
        self.settings_action.triggered.connect(self.open_settings_dialog)

        self.split_action = QAction(
            self._icon("split", QStyle.SP_FileDialogListView),
            "Split...",
            self,
        )
        self.split_action.setToolTip("Split labeled point cloud into one PLY file per class")
        self.split_action.setEnabled(False)
        self.split_action.triggered.connect(self.split_point_cloud_by_label)

        self.dbscan_action = QAction(
            self._icon("dbscan", QStyle.SP_FileDialogDetailedView),
            "DBSCAN...",
            self,
        )
        self.dbscan_action.setToolTip("Run DBSCAN clustering on the current point cloud")
        self.dbscan_action.setEnabled(False)
        self.dbscan_action.triggered.connect(self.run_dbscan_dialog)

        self.tin_action = QAction(
            self._icon("tin", QStyle.SP_FileDialogDetailedView),
            "TIN...",
            self,
        )
        self.tin_action.setToolTip("Build and display a triangulated irregular network mesh")
        self.tin_action.setEnabled(False)
        self.tin_action.triggered.connect(self.run_tin_dialog)

        self.save_view_png_action = QAction(
            self._icon("save_view_png", QStyle.SP_DialogSaveButton),
            "Save View as PNG...",
            self,
        )
        self.save_view_png_action.setToolTip("Save the current 3D view to a PNG image")
        self.save_view_png_action.setEnabled(False)
        self.save_view_png_action.triggered.connect(self.save_current_view_as_png)

        self.clear_viewport_action = QAction(
            self._icon("clear_viewport", QStyle.SP_TrashIcon),
            "Clear Viewport",
            self,
        )
        self.clear_viewport_action.setToolTip(
            "Remove the current point cloud and any cluster bounding boxes from the viewport"
        )
        self.clear_viewport_action.setEnabled(False)
        self.clear_viewport_action.triggered.connect(self.clear_viewport)

        self.fit_action = QAction(self._icon("fit", QStyle.SP_ArrowUp), "Fit to View", self)
        self.fit_action.setToolTip("Frame the whole cloud in the viewport")
        self.fit_action.setEnabled(False)
        self.fit_action.triggered.connect(self.fit_to_view)

        self.reset_action = QAction(self._icon("reset", QStyle.SP_BrowserReload), "Reset View", self)
        self.reset_action.setToolTip("Reset default camera orientation")
        self.reset_action.setEnabled(False)
        self.reset_action.triggered.connect(self.reset_view)

        self.view_top_action = QAction(self._icon("view_top", QStyle.SP_ArrowUp), "Top View", self)
        self.view_top_action.setToolTip("View from top")
        self.view_top_action.setEnabled(False)
        self.view_top_action.triggered.connect(self.set_view_top)

        self.view_front_action = QAction(self._icon("view_front", QStyle.SP_ArrowRight), "Front View", self)
        self.view_front_action.setToolTip("View from front")
        self.view_front_action.setEnabled(False)
        self.view_front_action.triggered.connect(self.set_view_front)

        self.view_left_action = QAction(self._icon("view_left", QStyle.SP_ArrowLeft), "Left Side View", self)
        self.view_left_action.setToolTip("View from left side")
        self.view_left_action.setEnabled(False)
        self.view_left_action.triggered.connect(self.set_view_left)

        self.view_right_action = QAction(self._icon("view_right", QStyle.SP_ArrowRight), "Right Side View", self)
        self.view_right_action.setToolTip("View from right side")
        self.view_right_action.setEnabled(False)
        self.view_right_action.triggered.connect(self.set_view_right)

        self.view_back_action = QAction(self._icon("view_back", QStyle.SP_ArrowDown), "Back View", self)
        self.view_back_action.setToolTip("View from back")
        self.view_back_action.setEnabled(False)
        self.view_back_action.triggered.connect(self.set_view_back)

        self.view_bottom_action = QAction(self._icon("view_bottom", QStyle.SP_ArrowDown), "Bottom View", self)
        self.view_bottom_action.setToolTip("View from bottom")
        self.view_bottom_action.setEnabled(False)
        self.view_bottom_action.triggered.connect(self.set_view_bottom)

        self.view_front_iso_action = QAction(self._icon("view_front_iso", QStyle.SP_FileDialogDetailedView), "Front Isometric", self)
        self.view_front_iso_action.setToolTip("Front isometric view")
        self.view_front_iso_action.setEnabled(False)
        self.view_front_iso_action.triggered.connect(self.set_view_front_isometric)

        self.toggle_rgb_action = QAction(self._icon("toggle_rgb", QStyle.SP_DialogYesButton), "Toggle RGB Mode", self)
        self.toggle_rgb_action.setToolTip("Switch between RGB and label/neutral coloring")
        self.toggle_rgb_action.setEnabled(False)
        self.toggle_rgb_action.triggered.connect(self.toggle_rgb_mode)

        self.game_navigation_action = QAction(self._icon("game_navigation", QStyle.SP_ComputerIcon), "Game Navigation Mode", self)
        self.game_navigation_action.setCheckable(True)
        self.game_navigation_action.setShortcut("F2")
        self.game_navigation_action.setToolTip("Toggle WASD + mouse navigation mode")
        self.game_navigation_action.setEnabled(False)
        self.game_navigation_action.triggered.connect(self.toggle_game_navigation_mode)

        self.generate_synthetic_action = QAction(
            self._icon("generate_synthetic", QStyle.SP_MediaPlay),
            "Generate Exterior Synthetic Cloud...",
            self,
        )
        self.generate_synthetic_action.setToolTip("Generate procedural labeled point cloud")
        self.generate_synthetic_action.triggered.connect(self.generate_synthetic_cloud)

        self.save_cloud_ply_action = QAction(
            self._icon("save_generated_ply", QStyle.SP_DialogSaveButton),
            "Save Cloud as PLY...",
            self,
        )
        self.save_cloud_ply_action.setToolTip("Save the currently displayed point cloud to a PLY file")
        self.save_cloud_ply_action.setEnabled(False)
        self.save_cloud_ply_action.triggered.connect(self.save_current_cloud_as_ply)

        self.save_mesh_ply_action = QAction(
            self._icon("save_mesh_ply", QStyle.SP_DialogSaveButton),
            "Save Mesh as PLY...",
            self,
        )
        self.save_mesh_ply_action.setToolTip("Save the current TIN surface mesh to a PLY file")
        self.save_mesh_ply_action.setEnabled(False)
        self.save_mesh_ply_action.triggered.connect(self.save_current_mesh_as_ply)

        self.exit_action = QAction(self._icon("exit", QStyle.SP_DialogCloseButton), "Exit", self)
        self.exit_action.setShortcut("Ctrl+Q")
        self.exit_action.triggered.connect(self.close)

        self.about_action = QAction(self._icon("about", QStyle.SP_MessageBoxInformation), "About", self)
        self.about_action.triggered.connect(self.show_about)

    def _create_menus(self) -> None:
        menu = self.menuBar()

        file_menu = menu.addMenu("File")
        file_menu.addAction(self.clear_viewport_action)
        file_menu.addSeparator()
        file_menu.addAction(self.open_action)
        file_menu.addAction(self.open_clusters_action)
        file_menu.addSeparator()
        file_menu.addAction(self.save_cloud_ply_action)
        file_menu.addAction(self.save_mesh_ply_action)
        file_menu.addAction(self.save_view_png_action)
        file_menu.addSeparator()
        file_menu.addAction(self.settings_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

        edit_menu = menu.addMenu("Tools")
        edit_menu.addAction(self.split_action)
        edit_menu.addAction(self.dbscan_action)
        edit_menu.addAction(self.tin_action)

        generate_menu = menu.addMenu("Generate")
        generate_menu.addAction(self.generate_synthetic_action)

        view_menu = menu.addMenu("View")
        view_menu.addAction(self.fit_action)
        view_menu.addAction(self.reset_action)
        view_menu.addSeparator()
        view_menu.addAction(self.view_top_action)
        view_menu.addAction(self.view_front_action)
        view_menu.addAction(self.view_left_action)
        view_menu.addAction(self.view_right_action)
        view_menu.addAction(self.view_back_action)
        view_menu.addAction(self.view_bottom_action)
        view_menu.addAction(self.view_front_iso_action)
        view_menu.addSeparator()
        view_menu.addAction(self.game_navigation_action)
        view_menu.addSeparator()
        view_menu.addAction(self.toggle_rgb_action)

        help_menu = menu.addMenu("Help")
        help_menu.addAction(self.about_action)

    def _create_toolbar(self) -> None:
        toolbar = QToolBar("Main Toolbar", self)
        toolbar.setMovable(False)
        base_icon_px = max(1, int(self.style().pixelMetric(QStyle.PM_ToolBarIconSize)))
        icon_px = base_icon_px * 2
        toolbar.setIconSize(QSize(icon_px, icon_px))
        toolbar.setStyleSheet(
            f"QToolButton {{ min-width: {icon_px + 10}px; min-height: {icon_px + 10}px; }}"
        )
        toolbar.addAction(self.open_action)
        toolbar.addAction(self.open_clusters_action)
        toolbar.addAction(self.save_cloud_ply_action)
        toolbar.addAction(self.save_mesh_ply_action)
        toolbar.addAction(self.save_view_png_action)
        toolbar.addAction(self.settings_action)
        toolbar.addSeparator()
        toolbar.addAction(self.split_action)
        toolbar.addAction(self.dbscan_action)
        toolbar.addAction(self.tin_action)
        toolbar.addAction(self.clear_viewport_action)
        toolbar.addSeparator()
        toolbar.addAction(self.fit_action)
        toolbar.addAction(self.reset_action)
        toolbar.addAction(self.view_top_action)
        toolbar.addAction(self.view_front_action)
        toolbar.addAction(self.view_left_action)
        toolbar.addAction(self.view_right_action)
        toolbar.addAction(self.view_back_action)
        toolbar.addAction(self.view_bottom_action)
        toolbar.addAction(self.view_front_iso_action)
        toolbar.addAction(self.game_navigation_action)
        toolbar.addAction(self.toggle_rgb_action)
        self.addToolBar(toolbar)

    def _ensure_settings_file_exists(self) -> None:
        if SETTINGS_PATH.exists():
            return
        try:
            save_project_settings(self._settings)
        except Exception:
            return

    def _apply_project_settings(
        self,
        settings: ProjectSettings,
        *,
        persist: bool,
        update_recent_paths: bool,
    ) -> None:
        normalized = ProjectSettings(
            output_directory=display_output_directory(resolve_output_directory(settings.output_directory)),
            point_size=_clamp_point_size(settings.point_size),
            viewport_background=normalize_color_value(settings.viewport_background),
            bounding_box_color_mode=_normalize_bounding_box_color_mode(settings.bounding_box_color_mode),
            bounding_box_color=normalize_color_value(settings.bounding_box_color),
            bounding_box_line_width=_clamp_bounding_box_line_width(settings.bounding_box_line_width),
            bounding_box_show_id=_coerce_bool_setting(settings.bounding_box_show_id),
        )

        self._settings = normalized
        APP_SETTINGS.output_directory = normalized.output_directory
        APP_SETTINGS.point_size = normalized.point_size
        APP_SETTINGS.viewport_background = normalized.viewport_background
        APP_SETTINGS.bounding_box_color_mode = normalized.bounding_box_color_mode
        APP_SETTINGS.bounding_box_color = normalized.bounding_box_color
        APP_SETTINGS.bounding_box_line_width = normalized.bounding_box_line_width
        APP_SETTINGS.bounding_box_show_id = normalized.bounding_box_show_id

        output_dir = ensure_data_dir()
        if update_recent_paths:
            self._last_split_dir = str(output_dir)
            dbscan_name = Path(self._last_dbscan_output_path).name if self._last_dbscan_output_path else "dbscan_clusters.yaml"
            if Path(dbscan_name).suffix.lower() not in {".yaml", ".yml"}:
                dbscan_name = Path(dbscan_name).with_suffix(".yaml").name
            self._last_dbscan_output_path = str(output_dir / dbscan_name)
            self._last_cluster_yaml_dir = str(output_dir)
            self._last_view_image_dir = str(output_dir)

        self.gl_widget.set_point_size(normalized.point_size)
        self.gl_widget.set_background_color(parse_color_value(normalized.viewport_background))
        self.gl_widget.set_cluster_box_display_settings(
            color_mode=normalized.bounding_box_color_mode,
            color=normalized.bounding_box_color,
            line_width=normalized.bounding_box_line_width,
            show_id=normalized.bounding_box_show_id,
        )

        if persist:
            save_project_settings(normalized)

        self._update_status_bar()

    def open_settings_dialog(self) -> None:
        dialog = SettingsDialog(self._settings, parent=self)
        if dialog.exec_() != QDialog.Accepted:
            return

        try:
            self._apply_project_settings(
                dialog.settings(),
                persist=True,
                update_recent_paths=True,
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Settings Error", f"Failed to apply project settings:\n{exc}")
            return

        self.statusBar().showMessage(f"Settings saved to {SETTINGS_PATH}", 5000)

    def open_file_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Point Cloud",
            str(ensure_data_dir()),
            "Point Cloud Files (*.txt *.ply);;TXT Files (*.txt);;PLY Files (*.ply);;All Files (*)",
        )
        if path:
            self.load_file(path)

    def load_file(self, path: str) -> None:
        try:
            cloud, was_subsampled = PointCloudLoader.load(path, max_points=self.max_points)
        except PointCloudLoaderError as exc:
            QMessageBox.critical(self, "Load Error", str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Load Error", f"Unexpected error while loading file:\n{exc}")
            return

        self._apply_cloud(cloud)

        if was_subsampled:
            QMessageBox.information(
                self,
                "Point Cloud Subsampled",
                f"Original points: {cloud.original_count:,}\n"
                f"Loaded points: {cloud.loaded_count:,}\n"
                f"Sampling limit: {self.max_points:,}",
            )

    def _default_split_prefix(self) -> str:
        if self.current_cloud is not None and self.current_cloud.file_path:
            stem = Path(self.current_cloud.file_path).stem.strip()
            if stem:
                return re.sub(r"[\\\\/:*?\"<>|]+", "_", stem)
        return self._last_split_prefix or "split"

    def split_point_cloud_by_label(self) -> None:
        if self.current_cloud is None:
            QMessageBox.information(self, "No Cloud", "Open a point cloud first.")
            return
        if not self.current_cloud.has_labels or self.current_cloud.labels is None:
            QMessageBox.information(
                self,
                "No Labels",
                "The current point cloud does not contain a label field.",
            )
            return

        dialog = SplitByLabelDialog(
            default_prefix=self._default_split_prefix(),
            default_directory=self._last_split_dir,
            parent=self,
        )
        if dialog.exec_() != QDialog.Accepted:
            return

        prefix = dialog.file_prefix()
        output_dir = Path(dialog.output_directory())
        self._last_split_prefix = prefix
        self._last_split_dir = str(output_dir)

        split_module = self._get_split_module()
        if split_module is None:
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            result = split_module.split_point_cloud_by_label_arrays(
                points=self.current_cloud.points,
                labels=self.current_cloud.labels,
                prefix=prefix,
                output_dir=output_dir,
                rgb=self.current_cloud.rgb,
                source_path=self.current_cloud.file_path,
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self,
                "Split Error",
                f"Failed to split point cloud into class files:\n{exc}",
            )
            return
        finally:
            QApplication.restoreOverrideCursor()

        self.statusBar().showMessage(
            f"Saved {len(result.files)} class PLY files to {output_dir}",
            7000,
        )
        QMessageBox.information(
            self,
            "Split Complete",
            f"Saved {len(result.files)} PLY files to:\n{output_dir}",
        )

    def _current_cloud_display_name(self) -> str:
        if self.current_cloud is None:
            return "<no cloud>"
        file_path = self.current_cloud.file_path.strip()
        if file_path:
            return file_path
        return "<in-memory point cloud>"

    def _default_dbscan_output_path(self) -> str:
        if self.current_cloud is not None and self.current_cloud.file_path:
            current_path = Path(self.current_cloud.file_path)
            stem = current_path.stem.strip()
            if stem:
                return str(ensure_data_dir() / f"{stem}_dbscan.yaml")
        return self._last_dbscan_output_path

    def _default_view_image_path(self) -> str:
        if self.current_cloud is not None and self.current_cloud.file_path:
            current_path = Path(self.current_cloud.file_path)
            stem = current_path.stem.strip()
            if stem:
                return str(ensure_data_dir() / f"{stem}_view.png")
        return str(Path(self._last_view_image_dir) / "current_view.png")

    def _default_cloud_export_path(self) -> str:
        if self.current_cloud is not None and self.current_cloud.file_path:
            stem = Path(self.current_cloud.file_path).stem.strip()
            if stem:
                safe_stem = re.sub(r"[\\\\/:*?\"<>|]+", "_", stem)
                return str(ensure_data_dir() / f"{safe_stem}_export.ply")
        return str(ensure_data_dir() / "point_cloud_export.ply")

    def _default_mesh_export_path(self) -> str:
        if self.current_cloud is not None and self.current_cloud.file_path:
            stem = Path(self.current_cloud.file_path).stem.strip()
            if stem:
                safe_stem = re.sub(r"[\\/:*?\"<>|]+", "_", stem)
                return str(ensure_data_dir() / f"{safe_stem}_tin_mesh.ply")
        return str(Path(self._last_mesh_export_dir) / "tin_mesh_export.ply")

    def _cluster_boxes_from_result(self, result) -> Tuple[ClusterBoundingBoxData, ...]:
        boxes: List[ClusterBoundingBoxData] = []
        for cluster in getattr(result, "clusters", ()):
            bbox = getattr(cluster, "bounding_box", None)
            if bbox is None:
                continue
            min_corner = tuple(float(value) for value in getattr(bbox, "min_corner", (0.0, 0.0, 0.0)))
            max_corner = tuple(float(value) for value in getattr(bbox, "max_corner", (0.0, 0.0, 0.0)))
            if len(min_corner) != 3 or len(max_corner) != 3:
                continue
            boxes.append(
                ClusterBoundingBoxData(
                    cluster_id=int(getattr(cluster, "cluster_id", len(boxes))),
                    point_count=int(getattr(cluster, "point_count", 0)),
                    min_corner=min_corner,  # type: ignore[arg-type]
                    max_corner=max_corner,  # type: ignore[arg-type]
                )
            )
        return tuple(boxes)

    def _apply_cluster_overlay(self, boxes: Sequence[ClusterBoundingBoxData], source_path: str) -> None:
        self._cluster_boxes = tuple(boxes)
        self._cluster_source_path = source_path
        if source_path:
            self._last_cluster_yaml_dir = str(Path(source_path).resolve().parent)
        self.gl_widget.set_cluster_boxes(self._cluster_boxes)
        self._update_view_actions_enabled(self.current_cloud)
        self._update_status_bar()

    def open_clusters_file_dialog(self) -> None:
        start_dir = self._last_cluster_yaml_dir
        if self.current_cloud is not None and self.current_cloud.file_path:
            current_path = Path(self.current_cloud.file_path)
            if current_path.suffix.lower() in {".txt", ".ply"}:
                start_dir = str(current_path.parent)

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Clusters File",
            start_dir,
            "YAML Files (*.yaml *.yml);;All Files (*)",
        )
        if not path:
            return

        dbscan_module = self._get_dbscan_module()
        if dbscan_module is None:
            return

        try:
            result = dbscan_module.load_cluster_result(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self,
                "Cluster Load Error",
                f"Failed to load clusters file:\n{exc}",
            )
            return

        self._apply_cluster_overlay(
            self._cluster_boxes_from_result(result),
            source_path=str(Path(path)),
        )

        cluster_count = len(self._cluster_boxes)
        self.statusBar().showMessage(
            f"Loaded clusters file: {path} | Clusters: {cluster_count}",
            7000,
        )
        if self.current_cloud is None:
            QMessageBox.information(
                self,
                "Clusters File Loaded",
                "Clusters file loaded. Open a point cloud to see the overlay.",
            )

    def run_dbscan_dialog(self) -> None:
        if self.current_cloud is None:
            QMessageBox.information(self, "No Cloud", "Open a point cloud first.")
            return

        dbscan_module = self._get_dbscan_module()
        if dbscan_module is None:
            return

        dialog = DBSCANDialog(
            cloud_name=self._current_cloud_display_name(),
            default_epsilon=self._last_dbscan_epsilon,
            default_min_pts=self._last_dbscan_min_pts,
            default_output_path=self._default_dbscan_output_path(),
            parent=self,
        )
        if dialog.exec_() != QDialog.Accepted:
            return

        params = dialog.params()
        self._last_dbscan_epsilon = params.epsilon
        self._last_dbscan_min_pts = params.min_pts
        self._last_dbscan_output_path = params.output_path

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            dbscan_module.run_dbscan_on_points(
                self.current_cloud.points,
                epsilon=params.epsilon,
                min_pts=params.min_pts,
                output_path=params.output_path,
                input_path=self.current_cloud.file_path,
            )
            saved_result = dbscan_module.load_cluster_result(params.output_path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self,
                "DBSCAN Error",
                f"Failed to run DBSCAN:\n{exc}",
            )
            return
        finally:
            QApplication.restoreOverrideCursor()

        saved_path = Path(params.output_path)
        if saved_path.suffix.lower() not in {".yaml", ".yml"}:
            saved_path = saved_path.with_suffix(".yaml")

        self._apply_cluster_overlay(
            self._cluster_boxes_from_result(saved_result),
            source_path=str(saved_path),
        )

        cluster_count = len(self._cluster_boxes)
        self.statusBar().showMessage(
            f"DBSCAN complete | Clusters: {cluster_count} | YAML: {saved_path}",
            7000,
        )
        QMessageBox.information(
            self,
            "DBSCAN Complete",
            f"Clusters found: {cluster_count}\nSaved YAML:\n{saved_path}",
        )

    def run_tin_dialog(self) -> None:
        if self.current_cloud is None or self.current_cloud.loaded_count <= 0:
            QMessageBox.information(self, "No Cloud", "Open a point cloud first.")
            return

        tin_module = self._get_tin_module()
        if tin_module is None:
            return

        dialog = TINDialog(
            cloud_name=self._current_cloud_display_name(),
            point_count=self.current_cloud.loaded_count,
            default_params=self._last_tin_params,
            parent=self,
        )
        if dialog.exec_() != QDialog.Accepted:
            return

        params = dialog.params()

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            result = execute_tin_for_points(
                self.current_cloud.points,
                params=params,
                tin_module=tin_module,
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self,
                "TIN Error",
                f"Failed to build TIN mesh:\n{exc}",
            )
            return
        finally:
            QApplication.restoreOverrideCursor()

        self._last_tin_params = result.params
        self._last_tin_mesh = result.mesh
        self.gl_widget.set_surface_mesh(
            result.mesh.vertices,
            result.mesh.triangles,
            render_mode=result.params.visual.render_mode,
            elevation_colormap=result.params.visual.elevation_colormap,
            smooth_normals=result.params.visual.smooth_normals,
        )
        self._update_view_actions_enabled(self.current_cloud)
        self.statusBar().showMessage(f"TIN mesh loaded | {result.summary}", 7000)
        self._update_status_bar()

    def save_current_mesh_as_ply(self) -> None:
        if not self.gl_widget.has_surface_mesh() or self._last_tin_mesh is None:
            QMessageBox.information(
                self,
                "No Mesh",
                "Build a TIN surface mesh first. This command saves only triangulated meshes, not point clouds or cluster boxes.",
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Mesh as PLY",
            self._default_mesh_export_path(),
            "PLY Files (*.ply);;All Files (*)",
        )
        if not path:
            return

        out_path = Path(path)
        if out_path.suffix.lower() != ".ply":
            out_path = out_path.with_suffix(".ply")

        try:
            export_mesh_to_ply(self._last_tin_mesh, out_path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Save Error", f"Failed to save mesh PLY file:\n{exc}")
            return

        self._last_mesh_export_dir = str(out_path.parent)
        self.statusBar().showMessage(f"Saved TIN mesh: {out_path}", 7000)

    def save_current_view_as_png(self) -> None:
        if self.current_cloud is None:
            QMessageBox.information(self, "No Cloud", "Open a point cloud first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Current View as PNG",
            self._default_view_image_path(),
            "PNG Images (*.png);;All Files (*)",
        )
        if not path:
            return

        out_path = Path(path)
        if out_path.suffix.lower() != ".png":
            out_path = out_path.with_suffix(".png")

        image = self.gl_widget.grabFramebuffer()
        if image.isNull():
            QMessageBox.critical(
                self,
                "Save Error",
                "Failed to capture the current view from the OpenGL viewport.",
            )
            return

        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not image.save(str(out_path), "PNG"):
            QMessageBox.critical(
                self,
                "Save Error",
                f"Failed to save PNG image:\n{out_path}",
            )
            return

        self._last_view_image_dir = str(out_path.parent)
        self.statusBar().showMessage(f"Saved current view: {out_path}", 7000)

    def fit_to_view(self) -> None:
        self.gl_widget.fit_to_view()
        self._update_status_bar()

    def clear_viewport(self) -> None:
        if self.current_cloud is None and not self._cluster_boxes and not self.gl_widget.has_surface_mesh():
            return

        self.current_cloud = None
        self._last_tin_mesh = None
        self._cluster_boxes = ()
        self._cluster_source_path = ""
        self.gl_widget.clear_viewport()
        self._update_view_actions_enabled(None)
        self.split_action.setEnabled(False)
        self.dbscan_action.setEnabled(False)
        self.tin_action.setEnabled(False)
        self.save_view_png_action.setEnabled(False)
        self._update_status_bar()

    def reset_view(self) -> None:
        self.gl_widget.reset_view()
        self._update_status_bar()

    def set_view_top(self) -> None:
        self.gl_widget.set_view_top()
        self._update_status_bar()

    def set_view_front(self) -> None:
        self.gl_widget.set_view_front()
        self._update_status_bar()

    def set_view_left(self) -> None:
        self.gl_widget.set_view_left()
        self._update_status_bar()

    def set_view_right(self) -> None:
        self.gl_widget.set_view_right()
        self._update_status_bar()

    def set_view_back(self) -> None:
        self.gl_widget.set_view_back()
        self._update_status_bar()

    def set_view_bottom(self) -> None:
        self.gl_widget.set_view_bottom()
        self._update_status_bar()

    def set_view_front_isometric(self) -> None:
        self.gl_widget.set_view_front_isometric()
        self._update_status_bar()

    def toggle_rgb_mode(self) -> None:
        self.gl_widget.toggle_rgb_mode()
        self._update_status_bar()

    def toggle_game_navigation_mode(self, enabled: bool) -> None:
        self.gl_widget.set_game_navigation_enabled(enabled)
        actual = self.gl_widget.is_game_navigation_enabled()
        if self.game_navigation_action.isChecked() != actual:
            self.game_navigation_action.blockSignals(True)
            self.game_navigation_action.setChecked(actual)
            self.game_navigation_action.blockSignals(False)
        self._update_status_bar()

    def generate_synthetic_cloud(self) -> None:
        synthetic_module = self._get_synthetic_module()
        if synthetic_module is None:
            return

        class_names = getattr(synthetic_module, "CLASS_NAMES", None)
        default_class_percentages = getattr(synthetic_module, "DEFAULT_CLASS_PERCENTAGES", None)
        if default_class_percentages is None:
            default_class_ratios = getattr(synthetic_module, "DEFAULT_CLASS_RATIOS", None)
            if isinstance(default_class_ratios, dict):
                default_class_percentages = tuple(
                    float(default_class_ratios[class_id]) * 100.0
                    for class_id in sorted(default_class_ratios)
                )
        artificial_surface_type_names = getattr(
            synthetic_module,
            "ARTIFICIAL_SURFACE_TYPE_NAMES",
            None,
        )
        default_artificial_surface_type_percentages = getattr(
            synthetic_module,
            "DEFAULT_ARTIFICIAL_SURFACE_TYPE_PERCENTAGES",
            None,
        )
        if default_artificial_surface_type_percentages is None:
            default_artificial_surface_type_ratios = getattr(
                synthetic_module,
                "DEFAULT_ARTIFICIAL_SURFACE_TYPE_RATIOS",
                None,
            )
            if isinstance(default_artificial_surface_type_ratios, dict):
                default_artificial_surface_type_percentages = tuple(
                    float(default_artificial_surface_type_ratios.get(key, 0.0)) * 100.0
                    for key in FALLBACK_ARTIFICIAL_SURFACE_TYPES
                )
        building_roof_type_names = getattr(synthetic_module, "BUILDING_ROOF_TYPE_NAMES", None)
        default_building_roof_type_percentages = getattr(
            synthetic_module,
            "DEFAULT_BUILDING_ROOF_TYPE_PERCENTAGES",
            None,
        )
        if default_building_roof_type_percentages is None:
            default_building_roof_type_ratios = getattr(
                synthetic_module,
                "DEFAULT_BUILDING_ROOF_TYPE_RATIOS",
                None,
            )
            if isinstance(default_building_roof_type_ratios, dict):
                default_building_roof_type_percentages = tuple(
                    float(default_building_roof_type_ratios.get(key, 0.0)) * 100.0
                    for key in FALLBACK_BUILDING_ROOF_TYPES
                )
        building_defaults = getattr(synthetic_module, "BUILDING_DEFAULTS", None)
        structure_type_names = getattr(synthetic_module, "STRUCTURE_TYPE_NAMES", None)
        default_structure_type_percentages = getattr(
            synthetic_module,
            "DEFAULT_STRUCTURE_TYPE_PERCENTAGES",
            None,
        )
        if default_structure_type_percentages is None:
            default_structure_type_ratios = getattr(
                synthetic_module,
                "DEFAULT_STRUCTURE_TYPE_RATIOS",
                None,
            )
            if isinstance(default_structure_type_ratios, dict):
                default_structure_type_percentages = tuple(
                    float(default_structure_type_ratios.get(key, 0.0)) * 100.0
                    for key in FALLBACK_STRUCTURE_TYPES
                )
        tree_crown_type_names = getattr(synthetic_module, "TREE_CROWN_TYPE_NAMES", None)
        default_tree_crown_type_percentages = getattr(
            synthetic_module,
            "DEFAULT_TREE_CROWN_TYPE_PERCENTAGES",
            None,
        )
        if default_tree_crown_type_percentages is None:
            default_tree_crown_type_ratios = getattr(
                synthetic_module,
                "DEFAULT_TREE_CROWN_TYPE_RATIOS",
                None,
            )
            if isinstance(default_tree_crown_type_ratios, dict):
                default_tree_crown_type_percentages = tuple(
                    float(default_tree_crown_type_ratios.get(key, 0.0)) * 100.0
                    for key in FALLBACK_TREE_CROWN_TYPES
                )
        high_veg_defaults = getattr(synthetic_module, "HIGH_VEG_DEFAULTS", None)
        vehicle_type_names = getattr(synthetic_module, "VEHICLE_TYPE_NAMES", None)
        default_vehicle_type_percentages = getattr(
            synthetic_module,
            "DEFAULT_VEHICLE_TYPE_PERCENTAGES",
            None,
        )
        if default_vehicle_type_percentages is None:
            default_vehicle_type_ratios = getattr(
                synthetic_module,
                "DEFAULT_VEHICLE_TYPE_RATIOS",
                None,
            )
            if isinstance(default_vehicle_type_ratios, dict):
                ordered_keys = ("car", "truck", "bus")
                default_vehicle_type_percentages = tuple(
                    float(default_vehicle_type_ratios.get(key, 0.0)) * 100.0
                    for key in ordered_keys
                )
        low_veg_defaults = getattr(synthetic_module, "LOW_VEG_DEFAULTS", None)

        dialog = SyntheticGenerationDialog(
            default_params=self._last_generation_params,
            class_names=class_names if isinstance(class_names, dict) else None,
            default_class_percentages=default_class_percentages,
            artificial_surface_type_names=(
                artificial_surface_type_names
                if isinstance(artificial_surface_type_names, dict)
                else None
            ),
            default_artificial_surface_type_percentages=(
                default_artificial_surface_type_percentages
            ),
            building_roof_type_names=(
                building_roof_type_names if isinstance(building_roof_type_names, dict) else None
            ),
            default_building_roof_type_percentages=default_building_roof_type_percentages,
            building_defaults=building_defaults if isinstance(building_defaults, dict) else None,
            structure_type_names=(
                structure_type_names if isinstance(structure_type_names, dict) else None
            ),
            default_structure_type_percentages=default_structure_type_percentages,
            tree_crown_type_names=(
                tree_crown_type_names if isinstance(tree_crown_type_names, dict) else None
            ),
            default_tree_crown_type_percentages=default_tree_crown_type_percentages,
            high_veg_defaults=high_veg_defaults if isinstance(high_veg_defaults, dict) else None,
            vehicle_type_names=vehicle_type_names if isinstance(vehicle_type_names, dict) else None,
            default_vehicle_type_percentages=default_vehicle_type_percentages,
            low_veg_defaults=low_veg_defaults if isinstance(low_veg_defaults, dict) else None,
            synthetic_module=synthetic_module,
            parent=self,
        )
        if dialog.exec_() != QDialog.Accepted:
            return
        params = dialog.params()

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            generated_cloud = synthetic_module.generate_point_cloud(
                total_points=params.total_points,
                area_width=params.area_width,
                area_length=params.area_length,
                terrain_relief=params.terrain_relief,
                randomize_object_counts=params.randomize_object_counts,
                seed=params.seed,
                class_percentages=(
                    params.class_percentages
                    if params.custom_class_distribution
                    else None
                ),
                artificial_surface_count=(
                    params.artificial_surface_count
                    if params.custom_artificial_surface_count
                    else None
                ),
                artificial_surface_type_percentages=(
                    params.artificial_surface_type_percentages
                    if params.custom_artificial_surface_type_distribution
                    else None
                ),
                building_count=(
                    params.building_count
                    if params.custom_building_count
                    else None
                ),
                building_roof_type_percentages=(
                    params.building_roof_type_percentages
                    if params.custom_building_roof_type_distribution
                    else None
                ),
                building_floor_min=int(params.building_floor_min),
                building_floor_max=int(params.building_floor_max),
                building_random_yaw=bool(params.building_random_yaw),
                structure_count=(
                    params.structure_count
                    if params.custom_structure_count
                    else None
                ),
                structure_type_percentages=(
                    params.structure_type_percentages
                    if params.custom_structure_type_distribution
                    else None
                ),
                tree_count=(
                    params.tree_count
                    if params.custom_tree_count
                    else None
                ),
                tree_crown_type_percentages=(
                    params.tree_crown_type_percentages
                    if params.custom_tree_crown_type_distribution
                    else None
                ),
                random_tree_crown_size=bool(params.random_tree_crown_size),
                tree_max_crown_diameter=float(params.tree_max_crown_diameter),
                tree_max_crown_top_height=float(params.tree_max_crown_top_height),
                tree_min_crown_bottom_height=float(params.tree_min_crown_bottom_height),
                vehicle_count=(
                    params.vehicle_count
                    if params.custom_vehicle_count
                    else None
                ),
                vehicle_type_percentages=(
                    params.vehicle_type_percentages
                    if params.custom_vehicle_type_distribution
                    else None
                ),
                shrub_count=(
                    params.shrub_count
                    if params.custom_shrub_count
                    else None
                ),
                random_shrub_size=bool(params.random_shrub_size),
                shrub_max_diameter=float(params.shrub_max_diameter),
                shrub_max_top_height=float(params.shrub_max_top_height),
                shrub_min_bottom_height=float(params.shrub_min_bottom_height),
                grass_patch_count=(
                    params.grass_patch_count
                    if params.custom_grass_patch_count
                    else None
                ),
                random_grass_patch_size=bool(params.random_grass_patch_size),
                grass_patch_max_size_x=float(params.grass_patch_max_size_x),
                grass_patch_max_size_y=float(params.grass_patch_max_size_y),
                grass_max_height=float(params.grass_max_height),
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Generation Error", f"Failed to generate point cloud:\n{exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        try:
            cloud = self._cloud_from_generated(generated_cloud, params, synthetic_module)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self,
                "Generation Error",
                f"Generated data has invalid format and cannot be displayed:\n{exc}",
            )
            return

        self._last_generation_params = params
        self._apply_cloud(cloud)
        self.statusBar().showMessage(
            f"Generated cloud loaded | Points: {cloud.loaded_count:,} | Seed: {params.seed}",
            7000,
        )

    def save_current_cloud_as_ply(self) -> None:
        if self.current_cloud is None or self.current_cloud.loaded_count <= 0:
            QMessageBox.information(self, "No Cloud", "Open or generate a point cloud first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Cloud as PLY",
            self._default_cloud_export_path(),
            "PLY Files (*.ply);;All Files (*)",
        )
        if not path:
            return

        out_path = Path(path)
        if out_path.suffix.lower() != ".ply":
            out_path = out_path.with_suffix(".ply")

        try:
            export_point_cloud_data_to_ply(self.current_cloud, out_path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Save Error", f"Failed to save PLY file:\n{exc}")
            return

        self.statusBar().showMessage(f"Saved point cloud: {out_path}", 7000)

    def _update_view_actions_enabled(self, cloud: Optional[PointCloudData]) -> None:
        has_cloud = cloud is not None and cloud.loaded_count > 0
        has_viewport_content = has_cloud or bool(self._cluster_boxes) or self.gl_widget.has_surface_mesh()
        for action in (
            self.fit_action,
            self.reset_action,
            self.view_top_action,
            self.view_front_action,
            self.view_left_action,
            self.view_right_action,
            self.view_back_action,
            self.view_bottom_action,
            self.view_front_iso_action,
            self.game_navigation_action,
        ):
            action.setEnabled(has_cloud)

        if not has_cloud and self.game_navigation_action.isChecked():
            self.game_navigation_action.blockSignals(True)
            self.game_navigation_action.setChecked(False)
            self.game_navigation_action.blockSignals(False)

        self.toggle_rgb_action.setEnabled(bool(has_cloud and cloud is not None and cloud.has_rgb))
        self.save_cloud_ply_action.setEnabled(has_cloud)
        self.save_mesh_ply_action.setEnabled(bool(self.gl_widget.has_surface_mesh()))
        self.tin_action.setEnabled(has_cloud)
        self.clear_viewport_action.setEnabled(has_viewport_content)

    def _apply_cloud(self, cloud: PointCloudData) -> None:
        self.current_cloud = cloud
        self._last_tin_mesh = None
        self.gl_widget.set_point_cloud(cloud)
        self.gl_widget.reset_view()

        if cloud.has_labels:
            self.gl_widget.set_color_mode("label")
        elif cloud.has_rgb:
            self.gl_widget.set_color_mode("rgb")
        else:
            self.gl_widget.set_color_mode("neutral")

        self._update_view_actions_enabled(cloud)
        self.split_action.setEnabled(cloud.has_labels)
        self.dbscan_action.setEnabled(cloud.loaded_count > 0)
        self.tin_action.setEnabled(cloud.loaded_count > 0)
        self.save_view_png_action.setEnabled(cloud.loaded_count > 0)
        self._update_status_bar()

    def _get_synthetic_module(self):
        if self._synthetic_module is not None:
            return self._synthetic_module
        try:
            from utils import synthetic_labeled_point_cloud as synthetic_module
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self,
                "Generator Error",
                f"Failed to import utils/synthetic_labeled_point_cloud.py:\n{exc}",
            )
            return None
        self._synthetic_module = synthetic_module
        return self._synthetic_module

    def _get_split_module(self):
        if self._split_module is not None:
            return self._split_module
        try:
            from utils import app_split_by_label as split_module
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self,
                "Split Error",
                f"Failed to import utils/app_split_by_label.py:\n{exc}",
            )
            return None
        self._split_module = split_module
        return self._split_module

    def _get_dbscan_module(self):
        if self._dbscan_module is not None:
            return self._dbscan_module
        try:
            from utils import app_dbscan_alg as dbscan_module
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self,
                "DBSCAN Error",
                f"Failed to import utils/app_dbscan_alg.py:\n{exc}",
            )
            return None
        self._dbscan_module = dbscan_module
        return self._dbscan_module

    def _get_tin_module(self):
        if self._tin_module is not None:
            return self._tin_module
        try:
            from utils import tin_alg as tin_module
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self,
                "TIN Error",
                f"Failed to import utils/tin_alg.py:\n{exc}",
            )
            return None
        self._tin_module = tin_module
        return self._tin_module

    def _cloud_from_generated(
        self,
        generated_cloud: np.ndarray,
        params: SyntheticGenerationParams,
        synthetic_module,
    ) -> PointCloudData:
        cloud = np.asarray(generated_cloud)
        if cloud.ndim != 2 or cloud.shape[1] < 4:
            raise ValueError("Expected generated array shape (N, 4): x y z label.")
        if cloud.shape[0] == 0:
            raise ValueError("Generated cloud is empty.")

        points = np.ascontiguousarray(cloud[:, :3], dtype=np.float32)
        labels = np.ascontiguousarray(np.rint(cloud[:, 3]).astype(np.int32), dtype=np.int32)
        rgb = self._build_generated_rgb(labels, synthetic_module)

        file_name = (
            f"synthetic_seed_{params.seed}_"
            f"n_{int(points.shape[0])}.generated"
        )
        return PointCloudData(
            points=points,
            labels=labels,
            rgb=rgb,
            file_path=file_name,
            original_count=int(points.shape[0]),
        )

    def _build_generated_rgb(self, labels: np.ndarray, synthetic_module) -> Optional[np.ndarray]:
        class_colors = getattr(synthetic_module, "CLASS_COLORS", None)
        if not isinstance(class_colors, dict):
            return None

        unique_labels = np.unique(labels)
        if unique_labels.size == 0:
            return None

        fallback = generate_distinct_palette(int(unique_labels.size))
        label_to_color: Dict[int, np.ndarray] = {}
        for idx, label in enumerate(unique_labels):
            value = class_colors.get(int(label))
            if value is None:
                label_to_color[int(label)] = fallback[idx]
                continue

            color = np.asarray(value, dtype=np.float32).reshape(-1)
            if color.size != 3:
                label_to_color[int(label)] = fallback[idx]
                continue
            label_to_color[int(label)] = np.clip(color[:3], 0.0, 1.0)

        rgb = np.empty((labels.shape[0], 3), dtype=np.float32)
        for label, color in label_to_color.items():
            rgb[labels == label] = color
        return rgb

    def _on_color_mode_changed(self, _mode: str) -> None:
        self._update_status_bar()

    def _on_navigation_mode_changed(self, _mode: str) -> None:
        actual = self.gl_widget.is_game_navigation_enabled()
        if self.game_navigation_action.isChecked() != actual:
            self.game_navigation_action.blockSignals(True)
            self.game_navigation_action.setChecked(actual)
            self.game_navigation_action.blockSignals(False)
        self._update_status_bar()

    def _update_status_bar(self) -> None:
        cluster_text = f" | Clusters: {len(self._cluster_boxes)}" if self._cluster_boxes else ""
        surface_text = " | Surface: TIN" if self.gl_widget.has_surface_mesh() else ""
        if self.current_cloud is None:
            if self._cluster_source_path:
                source_name = os.path.basename(self._cluster_source_path) or "<clusters>"
                self.statusBar().showMessage(f"No file loaded{cluster_text}{surface_text} | Overlay: {source_name}")
            else:
                self.statusBar().showMessage(f"No file loaded{surface_text}")
            return

        file_name = os.path.basename(self.current_cloud.file_path) or "<unknown>"
        mode_label = self.gl_widget.active_color_mode_label()
        nav_label = self.gl_widget.active_navigation_mode_label()
        self.statusBar().showMessage(
            f"File: {file_name} | Points: {self.current_cloud.loaded_count:,} / "
            f"{self.current_cloud.original_count:,} | Color: {mode_label} | Nav: {nav_label}"
            f"{cluster_text}{surface_text}"
        )

    def show_about(self) -> None:
        QMessageBox.about(
            self,
            "About MagicPoints",
            "MagicPoints\n\n"
            "Features:\n"
            "- TXT/PLY loading (ASCII and binary PLY)\n"
            "- OpenGL VBO rendering\n"
            "- Label and RGB color modes\n"
            "- DBSCAN clustering with YAML bounding-box overlays\n"
            "- Orbit, pan, zoom camera controls\n"
            "- Optional game-style WASD + mouse navigation mode\n"
            "- Synthetic cloud generation with configurable parameters\n"
            "- Automatic subsampling for very large clouds\n\n"
            "Copyright owner: Dyachenko Roman",
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MagicPoints (OpenGL point cloud visualizer)")
    parser.add_argument("file", nargs="?", help="Optional point cloud file to open at startup.")
    parser.add_argument(
        "--max-points",
        type=int,
        default=DEFAULT_MAX_POINTS,
        help=f"Maximum number of points to load (default: {DEFAULT_MAX_POINTS:,}).",
    )
    return parser


def configure_opengl() -> None:
    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.OpenGL)
    fmt.setVersion(2, 1)
    fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
    fmt.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(fmt)


def main() -> int:
    args = build_arg_parser().parse_args()
    configure_opengl()

    app = QApplication(sys.argv)
    app_icon_path = ICONS_DIR / "app.svg"
    if app_icon_path.exists():
        app.setWindowIcon(QIcon(str(app_icon_path)))
    else:
        app.setWindowIcon(app.style().standardIcon(QStyle.SP_ComputerIcon))
    window = MainWindow(max_points=args.max_points)
    window.show()

    if args.file:
        window.load_file(args.file)

    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
