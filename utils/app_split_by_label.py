#!/usr/bin/env python3
"""
Split a labeled point cloud into one PLY file per label.

The module supports:
  - direct CLI usage;
  - import-based API usage from the main GUI application.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
import re
import struct
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"


def ensure_data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


ProgressCallback = Callable[[float, str], None]


def _emit_progress(
    progress_callback: Optional[ProgressCallback],
    progress: float,
    stage: str = "",
) -> None:
    if progress_callback is None:
        return
    progress_callback(float(min(1.0, max(0.0, progress))), str(stage).strip())


def normalize_field_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


class SplitInputError(ValueError):
    """Raised when split parameters or source data are invalid."""


@dataclass
class PointCloudData:
    points: np.ndarray
    labels: Optional[np.ndarray] = None
    rgb: Optional[np.ndarray] = None
    file_path: str = ""

    def __post_init__(self) -> None:
        self.points = np.ascontiguousarray(self.points, dtype=np.float32)
        if self.labels is not None:
            self.labels = np.ascontiguousarray(self.labels.astype(np.int32, copy=False))
        if self.rgb is not None:
            self.rgb = np.ascontiguousarray(self.rgb.astype(np.float32, copy=False))

    @property
    def loaded_count(self) -> int:
        return int(self.points.shape[0])

    @property
    def has_labels(self) -> bool:
        return self.labels is not None and self.labels.size > 0

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
        )


@dataclass(frozen=True)
class SplitFileRecord:
    label: int
    point_count: int
    output_path: str


@dataclass(frozen=True)
class SplitResult:
    prefix: str
    output_dir: str
    source_path: str
    source_point_count: int
    files: Tuple[SplitFileRecord, ...]


@dataclass
class PlyProperty:
    kind: str
    name: str
    dtype: str
    count_dtype: Optional[str] = None


@dataclass
class PlyElement:
    name: str
    count: int
    properties: List[PlyProperty]


class PointCloudLoaderError(SplitInputError):
    pass


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
    def load(cls, path: str | Path) -> PointCloudData:
        file_path = Path(path)
        if not file_path.is_file():
            raise PointCloudLoaderError(f"File not found: {file_path}")

        ext = file_path.suffix.lower()
        if ext == ".txt":
            cloud = cls._load_txt(file_path)
        elif ext == ".ply":
            cloud = cls._load_ply(file_path)
        else:
            raise PointCloudLoaderError(f"Unsupported file extension '{ext}'. Use TXT or PLY.")

        cloud.file_path = str(file_path)
        return cloud

    @classmethod
    def _load_txt(cls, path: Path) -> PointCloudData:
        rows: List[List[float]] = []
        header_tokens: Optional[List[str]] = None
        ncols: Optional[int] = None

        with path.open("r", encoding="utf-8", errors="ignore") as fh:
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
    def _load_ply(cls, path: Path) -> PointCloudData:
        with path.open("rb") as fh:
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
                        raise PointCloudLoaderError(
                            f"Invalid element count in PLY header: '{parts[2]}'"
                        ) from exc
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
        cls,
        fh,
        element: PlyElement,
        keep: bool,
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
        cls,
        fh,
        element: PlyElement,
        endian: str,
        keep: bool,
    ) -> Optional[Dict[str, np.ndarray]]:
        scalar_only = all(prop.kind == "scalar" for prop in element.properties)

        if scalar_only:
            dtype_fields = []
            for prop in element.properties:
                np_type = cls.PLY_TO_NUMPY.get(prop.dtype)
                if np_type is None:
                    raise PointCloudLoaderError(
                        f"Unsupported PLY type '{prop.dtype}' in element '{element.name}'."
                    )
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
        cls,
        matrix: np.ndarray,
        header_tokens: Optional[List[str]] = None,
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
        raise SplitInputError("Point cloud points must have shape (N, 3).")

    rgb_uint8 = None
    if cloud.rgb is not None and cloud.rgb.size > 0:
        rgb = np.ascontiguousarray(cloud.rgb, dtype=np.float32)
        rgb_uint8 = np.clip(np.round(rgb * 255.0), 0.0, 255.0).astype(np.uint8)

    labels = None
    if cloud.labels is not None and cloud.labels.size > 0:
        labels = np.ascontiguousarray(cloud.labels, dtype=np.int32)

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


def _validate_prefix(prefix: str) -> str:
    value = str(prefix).strip()
    if not value:
        raise SplitInputError("File prefix must not be empty.")
    if os.sep in value or (os.altsep and os.altsep in value):
        raise SplitInputError("File prefix must not contain path separators.")
    return value


def split_point_cloud_by_label_arrays(
    points: np.ndarray,
    labels: np.ndarray,
    prefix: str,
    output_dir: str | Path,
    rgb: Optional[np.ndarray] = None,
    source_path: str = "",
    progress_callback: Optional[ProgressCallback] = None,
) -> SplitResult:
    point_array = np.ascontiguousarray(np.asarray(points, dtype=np.float32))
    if point_array.ndim != 2 or point_array.shape[1] != 3:
        raise SplitInputError("Point cloud points must have shape (N, 3).")

    label_array = np.asarray(labels, dtype=np.int32).reshape(-1)
    if label_array.shape[0] != point_array.shape[0]:
        raise SplitInputError("Label array size must match point count.")
    if label_array.size == 0:
        raise SplitInputError("Point cloud does not contain any label values.")

    rgb_array = None
    if rgb is not None:
        rgb_array = np.ascontiguousarray(np.asarray(rgb, dtype=np.float32))
        if rgb_array.shape != (point_array.shape[0], 3):
            raise SplitInputError("RGB array must have shape (N, 3) when provided.")

    valid_prefix = _validate_prefix(prefix)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cloud = PointCloudData(
        points=point_array,
        labels=label_array,
        rgb=rgb_array,
        file_path=str(source_path),
    )
    unique_labels = np.unique(label_array)
    if unique_labels.size == 0:
        raise SplitInputError("Point cloud does not contain any label values.")

    total_labels = int(unique_labels.size)
    _emit_progress(progress_callback, 0.05, f"Preparing {total_labels} label file(s)")

    saved_files: List[SplitFileRecord] = []
    for index, label in enumerate(unique_labels, start=1):
        label_value = int(label)
        start_progress = 0.05 + 0.90 * ((index - 1) / total_labels)
        _emit_progress(
            progress_callback,
            start_progress,
            f"Saving label {label_value} ({index}/{total_labels})",
        )
        indices = np.flatnonzero(label_array == label)
        subset = cloud.subset(indices)
        out_path = out_dir / f"{valid_prefix}_{label_value}.ply"
        export_point_cloud_data_to_ply(subset, out_path)
        saved_files.append(
            SplitFileRecord(
                label=label_value,
                point_count=int(indices.size),
                output_path=str(out_path),
            )
        )
        end_progress = 0.05 + 0.90 * (index / total_labels)
        _emit_progress(
            progress_callback,
            end_progress,
            f"Saved label {label_value} ({index}/{total_labels})",
        )

    _emit_progress(progress_callback, 1.0, "Complete")

    return SplitResult(
        prefix=valid_prefix,
        output_dir=str(out_dir),
        source_path=str(source_path),
        source_point_count=int(point_array.shape[0]),
        files=tuple(saved_files),
    )


def split_point_cloud_file(
    input_path: str | Path,
    prefix: Optional[str] = None,
    output_dir: str | Path | None = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> SplitResult:
    cloud = PointCloudLoader.load(input_path)
    if not cloud.has_labels or cloud.labels is None:
        raise SplitInputError("Input point cloud does not contain a label field.")

    resolved_prefix = prefix
    if resolved_prefix is None or not str(resolved_prefix).strip():
        stem = Path(input_path).stem.strip()
        resolved_prefix = stem or "split"

    resolved_output_dir = Path(output_dir) if output_dir is not None else ensure_data_dir()
    return split_point_cloud_by_label_arrays(
        points=cloud.points,
        labels=cloud.labels,
        prefix=resolved_prefix,
        output_dir=resolved_output_dir,
        rgb=cloud.rgb,
        source_path=str(input_path),
        progress_callback=progress_callback,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split a labeled TXT/PLY point cloud into one PLY file per class label."
    )
    parser.add_argument("input_cloud", help="Path to the input point cloud file (.txt or .ply).")
    parser.add_argument(
        "--prefix",
        help="Output file prefix. Defaults to the input file stem.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DATA_DIR),
        help=f"Directory for output PLY files (default: {DATA_DIR}).",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    try:
        result = split_point_cloud_file(
            input_path=args.input_cloud,
            prefix=args.prefix,
            output_dir=args.output_dir,
        )
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Split failed: {exc}") from exc

    print(f"Input points: {result.source_point_count}")
    print(f"Output directory: {result.output_dir}")
    print(f"Files written: {len(result.files)}")
    for file_record in result.files:
        print(
            f"Label {file_record.label}: {file_record.point_count} points -> {file_record.output_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
