"""Qt dialog for configuring TIN generation and visualization."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtWidgets import (
    QCheckBox,
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
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from tin_command import (
    ELEVATION_COLORMAP_VALUES,
    RENDER_MODE_VALUES,
    TINCommandParams,
    TINVisualSettings,
    normalize_command_params,
)
from utils.tin_alg import (
    BOUNDARY_TYPE_VALUES,
    DUPLICATE_HANDLING_VALUES,
    INTERPOLATION_VALUES,
    SPATIAL_INDEX_VALUES,
    TINParameters,
)


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
