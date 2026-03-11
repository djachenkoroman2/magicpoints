#!/usr/bin/env python3
import argparse
import colorsys
import os
import re
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_DYNAMIC_DRAW,
    GL_FALSE,
    GL_FLOAT,
    GL_FRAGMENT_SHADER,
    GL_POINTS,
    GL_PROGRAM_POINT_SIZE,
    GL_STATIC_DRAW,
    GL_VERTEX_SHADER,
    glBindBuffer,
    glBufferData,
    glClear,
    glClearColor,
    glDeleteBuffers,
    glDeleteProgram,
    glDisableVertexAttribArray,
    glDrawArrays,
    glEnable,
    glEnableVertexAttribArray,
    glGenBuffers,
    glGetAttribLocation,
    glGetUniformLocation,
    glUniform1f,
    glUniformMatrix4fv,
    glUseProgram,
    glVertexAttribPointer,
    glViewport,
)
from OpenGL.GL.shaders import compileProgram, compileShader
from PyQt5.QtCore import QSize, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QCursor, QIcon, QKeyEvent, QMouseEvent, QSurfaceFormat, QWheelEvent
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QOpenGLWidget,
    QSpinBox,
    QStyle,
    QToolBar,
    QVBoxLayout,
)

DEFAULT_MAX_POINTS = 2_000_000
DEFAULT_POINT_SIZE = 3.0
ICONS_DIR = Path(__file__).resolve().parent / "assets" / "icons"


def normalize_field_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.strip().lower())


def normalize_vector(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm <= 1e-12:
        return v
    return v / norm


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
        return np.mean(self.points, axis=0).astype(np.float32)

    @property
    def radius(self) -> float:
        if self.points.size == 0:
            return 1.0
        mins = np.min(self.points, axis=0)
        maxs = np.max(self.points, axis=0)
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


class PointCloudLoaderError(Exception):
    pass


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


@dataclass
class SyntheticGenerationParams:
    total_points: int = 100_000
    area_width: float = 240.0
    area_length: float = 220.0
    terrain_relief: float = 1.0
    seed: int = 12
    randomize_object_counts: bool = True


class SyntheticGenerationDialog(QDialog):
    def __init__(
        self,
        default_params: Optional[SyntheticGenerationParams] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Generate Synthetic Point Cloud")
        self.setModal(True)

        params = default_params or SyntheticGenerationParams()

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

        form = QFormLayout()
        form.addRow("Total points:", self.total_points_spin)
        form.addRow("Area width:", self.area_width_spin)
        form.addRow("Area length:", self.area_length_spin)
        form.addRow("Terrain relief [0..1]:", self.terrain_relief_spin)
        form.addRow("Random seed:", self.seed_spin)
        form.addRow("", self.random_counts_check)

        note = QLabel("Generation uses synthetic_labeled_point_cloud.py pipeline.", self)
        note.setWordWrap(True)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(note)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def params(self) -> SyntheticGenerationParams:
        return SyntheticGenerationParams(
            total_points=int(self.total_points_spin.value()),
            area_width=float(self.area_width_spin.value()),
            area_length=float(self.area_length_spin.value()),
            terrain_relief=float(self.terrain_relief_spin.value()),
            seed=int(self.seed_spin.value()),
            randomize_object_counts=bool(self.random_counts_check.isChecked()),
        )


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
        self._color_mode: str = "neutral"
        self._background = (0.08, 0.09, 0.11)
        self._neutral_color = np.array([0.82, 0.84, 0.88], dtype=np.float32)

        self._program: int = 0
        self._vbo_points: int = 0
        self._vbo_colors: int = 0
        self._a_pos: int = -1
        self._a_color: int = -1
        self._u_mvp: int = -1
        self._u_point_size: int = -1
        self._initialized = False
        self._point_count = 0

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
        self._point_size = DEFAULT_POINT_SIZE

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
            target = self._game_position + forward * self._distance
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

        vertex_shader = """
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

        fragment_shader = """
        #version 120
        varying vec3 v_color;
        void main() {
            gl_FragColor = vec4(v_color, 1.0);
        }
        """

        self._program = compileProgram(
            compileShader(vertex_shader, GL_VERTEX_SHADER),
            compileShader(fragment_shader, GL_FRAGMENT_SHADER),
        )
        self._a_pos = glGetAttribLocation(self._program, "a_pos")
        self._a_color = glGetAttribLocation(self._program, "a_color")
        self._u_mvp = glGetUniformLocation(self._program, "u_mvp")
        self._u_point_size = glGetUniformLocation(self._program, "u_point_size")

        self._vbo_points = glGenBuffers(1)
        self._vbo_colors = glGenBuffers(1)

        self._initialized = True
        if self._cloud is not None:
            self._upload_geometry()
            self._upload_colors()

        context = self.context()
        if context is not None:
            context.aboutToBeDestroyed.connect(self._cleanup_gl)

    def resizeGL(self, width: int, height: int) -> None:
        glViewport(0, 0, max(1, width), max(1, height))

    def paintGL(self) -> None:
        glViewport(0, 0, max(1, self.width()), max(1, self.height()))
        glClearColor(*self._background, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if not self._initialized or self._cloud is None or self._point_count <= 0:
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
        glUseProgram(0)

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
            self._yaw += dx * 0.35
            self._pitch += dy * 0.35
            self._pitch = max(-89.0, min(89.0, self._pitch))
        elif buttons & (Qt.MidButton | Qt.RightButton):
            self._apply_pan(dx, dy)

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
        self._pan += (-dx * world_per_pixel) * right + (dy * world_per_pixel) * up

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

    def _cleanup_gl(self) -> None:
        self.makeCurrent()
        if self._vbo_points:
            glDeleteBuffers(1, [self._vbo_points])
            self._vbo_points = 0
        if self._vbo_colors:
            glDeleteBuffers(1, [self._vbo_colors])
            self._vbo_colors = 0
        if self._program:
            glDeleteProgram(self._program)
            self._program = 0
        self.doneCurrent()
        self._initialized = False


class MainWindow(QMainWindow):
    def __init__(self, max_points: int = DEFAULT_MAX_POINTS):
        super().__init__()
        self.max_points = max_points
        self.current_cloud: Optional[PointCloudData] = None
        self._synthetic_module = None
        self._generated_cloud_raw: Optional[np.ndarray] = None
        self._last_generation_params = SyntheticGenerationParams()

        self.setWindowTitle("MagicPoints")
        self.resize(1200, 800)
        self.setWindowIcon(self._icon("app", QStyle.SP_ComputerIcon))

        self.gl_widget = PointCloudGLWidget(self)
        self.setCentralWidget(self.gl_widget)
        self.gl_widget.colorModeChanged.connect(self._on_color_mode_changed)
        self.gl_widget.navigationModeChanged.connect(self._on_navigation_mode_changed)

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
        self.open_action = QAction(self._icon("open", QStyle.SP_DialogOpenButton), "Open File", self)
        self.open_action.setShortcut("Ctrl+O")
        self.open_action.setToolTip("Open TXT or PLY point cloud")
        self.open_action.triggered.connect(self.open_file_dialog)

        self.fit_action = QAction(self._icon("fit", QStyle.SP_ArrowUp), "Fit to View", self)
        self.fit_action.setToolTip("Frame the whole cloud in the viewport")
        self.fit_action.triggered.connect(self.fit_to_view)

        self.reset_action = QAction(self._icon("reset", QStyle.SP_BrowserReload), "Reset View", self)
        self.reset_action.setToolTip("Reset default camera orientation")
        self.reset_action.triggered.connect(self.reset_view)

        self.view_top_action = QAction(self._icon("view_top", QStyle.SP_ArrowUp), "Top View", self)
        self.view_top_action.setToolTip("View from top")
        self.view_top_action.triggered.connect(self.set_view_top)

        self.view_front_action = QAction(self._icon("view_front", QStyle.SP_ArrowRight), "Front View", self)
        self.view_front_action.setToolTip("View from front")
        self.view_front_action.triggered.connect(self.set_view_front)

        self.view_left_action = QAction(self._icon("view_left", QStyle.SP_ArrowLeft), "Left Side View", self)
        self.view_left_action.setToolTip("View from left side")
        self.view_left_action.triggered.connect(self.set_view_left)

        self.view_right_action = QAction(self._icon("view_right", QStyle.SP_ArrowRight), "Right Side View", self)
        self.view_right_action.setToolTip("View from right side")
        self.view_right_action.triggered.connect(self.set_view_right)

        self.view_back_action = QAction(self._icon("view_back", QStyle.SP_ArrowDown), "Back View", self)
        self.view_back_action.setToolTip("View from back")
        self.view_back_action.triggered.connect(self.set_view_back)

        self.view_bottom_action = QAction(self._icon("view_bottom", QStyle.SP_ArrowDown), "Bottom View", self)
        self.view_bottom_action.setToolTip("View from bottom")
        self.view_bottom_action.triggered.connect(self.set_view_bottom)

        self.view_front_iso_action = QAction(self._icon("view_front_iso", QStyle.SP_FileDialogDetailedView), "Front Isometric", self)
        self.view_front_iso_action.setToolTip("Front isometric view")
        self.view_front_iso_action.triggered.connect(self.set_view_front_isometric)

        self.toggle_rgb_action = QAction(self._icon("toggle_rgb", QStyle.SP_DialogYesButton), "Toggle RGB Mode", self)
        self.toggle_rgb_action.setToolTip("Switch between RGB and label/neutral coloring")
        self.toggle_rgb_action.setEnabled(False)
        self.toggle_rgb_action.triggered.connect(self.toggle_rgb_mode)

        self.game_navigation_action = QAction(self._icon("game_navigation", QStyle.SP_ComputerIcon), "Game Navigation Mode", self)
        self.game_navigation_action.setCheckable(True)
        self.game_navigation_action.setShortcut("F2")
        self.game_navigation_action.setToolTip("Toggle WASD + mouse navigation mode")
        self.game_navigation_action.triggered.connect(self.toggle_game_navigation_mode)

        self.generate_synthetic_action = QAction(self._icon("generate_synthetic", QStyle.SP_MediaPlay), "Generate Synthetic Cloud...", self)
        self.generate_synthetic_action.setToolTip("Generate procedural labeled point cloud")
        self.generate_synthetic_action.triggered.connect(self.generate_synthetic_cloud)

        self.save_generated_ply_action = QAction(self._icon("save_generated_ply", QStyle.SP_DialogSaveButton), "Save Generated Cloud as PLY...", self)
        self.save_generated_ply_action.setToolTip("Save the latest generated cloud to a PLY file")
        self.save_generated_ply_action.setEnabled(False)
        self.save_generated_ply_action.triggered.connect(self.save_generated_cloud_as_ply)

        self.exit_action = QAction(self._icon("exit", QStyle.SP_DialogCloseButton), "Exit", self)
        self.exit_action.setShortcut("Ctrl+Q")
        self.exit_action.triggered.connect(self.close)

        self.about_action = QAction(self._icon("about", QStyle.SP_MessageBoxInformation), "About", self)
        self.about_action.triggered.connect(self.show_about)

    def _create_menus(self) -> None:
        menu = self.menuBar()

        file_menu = menu.addMenu("File")
        file_menu.addAction(self.open_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

        generate_menu = menu.addMenu("Generate")
        generate_menu.addAction(self.generate_synthetic_action)
        generate_menu.addAction(self.save_generated_ply_action)

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

    def open_file_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Point Cloud",
            "",
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

        self._generated_cloud_raw = None
        self.save_generated_ply_action.setEnabled(False)
        self._apply_cloud(cloud)

        if was_subsampled:
            QMessageBox.information(
                self,
                "Point Cloud Subsampled",
                f"Original points: {cloud.original_count:,}\n"
                f"Loaded points: {cloud.loaded_count:,}\n"
                f"Sampling limit: {self.max_points:,}",
            )

    def fit_to_view(self) -> None:
        self.gl_widget.fit_to_view()
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
        dialog = SyntheticGenerationDialog(default_params=self._last_generation_params, parent=self)
        if dialog.exec_() != QDialog.Accepted:
            return
        params = dialog.params()

        synthetic_module = self._get_synthetic_module()
        if synthetic_module is None:
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            generated_cloud = synthetic_module.generate_point_cloud(
                total_points=params.total_points,
                area_width=params.area_width,
                area_length=params.area_length,
                terrain_relief=params.terrain_relief,
                randomize_object_counts=params.randomize_object_counts,
                seed=params.seed,
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

        self._generated_cloud_raw = np.asarray(generated_cloud, dtype=np.float64)
        self._last_generation_params = params
        self.save_generated_ply_action.setEnabled(True)
        self._apply_cloud(cloud)
        self.statusBar().showMessage(
            f"Generated cloud loaded | Points: {cloud.loaded_count:,} | Seed: {params.seed}",
            7000,
        )

    def save_generated_cloud_as_ply(self) -> None:
        if self._generated_cloud_raw is None:
            QMessageBox.information(self, "No Generated Cloud", "Generate a synthetic cloud first.")
            return

        synthetic_module = self._get_synthetic_module()
        if synthetic_module is None:
            return

        default_name = (
            f"synthetic_seed_{self._last_generation_params.seed}_"
            f"n_{int(self._generated_cloud_raw.shape[0])}.ply"
        )
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Generated Cloud as PLY",
            default_name,
            "PLY Files (*.ply);;All Files (*)",
        )
        if not path:
            return

        out_path = Path(path)
        if out_path.suffix.lower() != ".ply":
            out_path = out_path.with_suffix(".ply")

        try:
            synthetic_module.export_to_ply(self._generated_cloud_raw, out_path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Save Error", f"Failed to save PLY file:\n{exc}")
            return

        self.statusBar().showMessage(f"Saved generated cloud: {out_path}", 7000)

    def _apply_cloud(self, cloud: PointCloudData) -> None:
        self.current_cloud = cloud
        self.gl_widget.set_point_cloud(cloud)
        self.gl_widget.reset_view()

        if cloud.has_labels:
            self.gl_widget.set_color_mode("label")
        elif cloud.has_rgb:
            self.gl_widget.set_color_mode("rgb")
        else:
            self.gl_widget.set_color_mode("neutral")

        self.toggle_rgb_action.setEnabled(cloud.has_rgb)
        self._update_status_bar()

    def _get_synthetic_module(self):
        if self._synthetic_module is not None:
            return self._synthetic_module
        try:
            import synthetic_labeled_point_cloud as synthetic_module
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self,
                "Generator Error",
                f"Failed to import synthetic_labeled_point_cloud.py:\n{exc}",
            )
            return None
        self._synthetic_module = synthetic_module
        return self._synthetic_module

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
        if self.current_cloud is None:
            self.statusBar().showMessage("No file loaded")
            return

        file_name = os.path.basename(self.current_cloud.file_path) or "<unknown>"
        mode_label = self.gl_widget.active_color_mode_label()
        nav_label = self.gl_widget.active_navigation_mode_label()
        self.statusBar().showMessage(
            f"File: {file_name} | Points: {self.current_cloud.loaded_count:,} / "
            f"{self.current_cloud.original_count:,} | Color: {mode_label} | Nav: {nav_label}"
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
