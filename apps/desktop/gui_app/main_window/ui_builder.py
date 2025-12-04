"""
UI Builder
==========

Handles the construction of the main window UI components.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTextEdit, 
    QProgressBar, QGroupBox, QFrame, QScrollArea, QCheckBox, QSlider, 
    QSpinBox, QComboBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPalette, QColor
from qtrangeslider import QRangeSlider
import logging

from ..vtk.viewer import VTK3DViewer

class UIBuilder:
    def __init__(self, window):
        self.window = window

    def init_ui(self):
        self.window.setWindowTitle("Khorium MeshGen - Parallel Edition")
        self.window.setGeometry(100, 50, 1600, 850)

        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(248, 249, 250))
        palette.setColor(QPalette.WindowText, QColor(33, 37, 41))
        palette.setColor(QPalette.Base, QColor(255, 255, 255))
        palette.setColor(QPalette.Text, QColor(33, 37, 41))
        palette.setColor(QPalette.Button, QColor(255, 255, 255))
        palette.setColor(QPalette.ButtonText, QColor(33, 37, 41))
        self.window.setPalette(palette)

        central = QWidget()
        self.window.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Left Container
        self.window.left_container = QWidget()
        left_container_layout = QVBoxLayout(self.window.left_container)
        left_container_layout.setContentsMargins(0, 0, 0, 0)
        left_container_layout.setSpacing(0)

        self.window.left_panel = self.create_left_panel()
        left_container_layout.addWidget(self.window.left_panel)
        
        # Chatbox placeholder (initialized in window)
        
        main_layout.addWidget(self.window.left_container)

        # Right Panel
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 1)

    def create_left_panel(self):
        panel = QFrame()
        panel.setMaximumWidth(380)
        panel.setStyleSheet("QFrame { background-color: white; border-right: 1px solid #dee2e6; }")

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { background-color: white; border: none; }")

        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        # Header
        header_layout = QHBoxLayout()
        title = QLabel("Khorium MeshGen")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        header_layout.addWidget(title)
        header_layout.addStretch()

        self.window.chat_toggle_btn = QPushButton("💬 AI Chat")
        self.window.chat_toggle_btn.setCheckable(True)
        self.window.chat_toggle_btn.clicked.connect(self.window.toggle_chatbox)
        header_layout.addWidget(self.window.chat_toggle_btn)
        layout.addLayout(header_layout)

        subtitle = QLabel("Parallel Mesh Generation (3-5x Faster)")
        subtitle.setFont(QFont("Arial", 10))
        subtitle.setStyleSheet("color: #198754; font-weight: bold;")
        layout.addWidget(subtitle)

        # System Info
        from multiprocessing import cpu_count
        cores = cpu_count()
        workers = max(1, cores - 2)
        system_info = QLabel(f"System: {cores} cores | {workers} parallel workers")
        system_info.setFont(QFont("Arial", 9))
        system_info.setStyleSheet("color: #6c757d; padding: 5px 0;")
        layout.addWidget(system_info)

        # Upload Group
        self._create_upload_group(layout)

        # Quality Group
        self._create_quality_group(layout)

        # Cross-Section Group
        self._create_crosssection_group(layout)

        # Visualization Group
        self._create_viz_group(layout)

        # Paintbrush placeholder (added dynamically in window)

        # Buttons
        self.window.generate_btn = QPushButton("Generate Mesh (Parallel)")
        self.window.generate_btn.setEnabled(False)
        self.window.generate_btn.setStyleSheet(self._get_btn_style("#198754", "#157347"))
        self.window.generate_btn.clicked.connect(self.window.start_mesh_generation)
        layout.addWidget(self.window.generate_btn)

        self.window.stop_btn = QPushButton("Stop Meshing")
        self.window.stop_btn.setEnabled(False)
        self.window.stop_btn.setStyleSheet(self._get_btn_style("#dc3545", "#bb2d3b"))
        self.window.stop_btn.clicked.connect(self.window.stop_mesh_generation)
        layout.addWidget(self.window.stop_btn)

        self.window.refine_btn = QPushButton("Refine Mesh Quality")
        self.window.refine_btn.setEnabled(False)
        self.window.refine_btn.setStyleSheet(self._get_btn_style("#17a2b8", "#138496"))
        self.window.refine_btn.clicked.connect(self.window.refine_mesh_quality)
        layout.addWidget(self.window.refine_btn)

        # Progress
        self._create_progress_group(layout)

        layout.addStretch()

        # Hex Mode Secret Button
        self.window.hex_mode_btn = QPushButton("🔳")
        self.window.hex_mode_btn.setFixedSize(30, 30)
        self.window.hex_mode_btn.setCheckable(True)
        self.window.hex_mode_btn.setStyleSheet("""
            QPushButton { background-color: #343a40; color: white; border: none; border-radius: 3px; font-size: 16px; padding: 2px; }
            QPushButton:hover { background-color: #495057; }
            QPushButton:checked { background-color: #0d6efd; border: 2px solid #0a58ca; }
        """)
        self.window.hex_mode_btn.clicked.connect(self.window.toggle_hex_visualization)
        layout.addWidget(self.window.hex_mode_btn, alignment=Qt.AlignLeft)

        scroll.setWidget(content_widget)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.addWidget(scroll)

        return panel

    def create_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 3D Viewer (create FIRST so controls can reference it)
        self.window.viewer = VTK3DViewer(parent=self.window)
        layout.addWidget(self.window.viewer, 2)

        # Viewer Controls
        controls = QFrame()
        controls.setStyleSheet("background-color: #f8f9fa; border-bottom: 1px solid #dee2e6;")
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(10, 5, 10, 5)

        axes_cb = QCheckBox("Show XYZ Axes")
        axes_cb.setChecked(True)
        axes_cb.setStyleSheet("QCheckBox { color: black; font-size: 11px; }")
        axes_cb.stateChanged.connect(lambda s: self.window.viewer.axes_widget.SetEnabled(s == Qt.Checked))
        controls_layout.addWidget(axes_cb)

        self.window.show_ghost_cb = QCheckBox("Show Ghost (Above Cut)")
        self.window.show_ghost_cb.setChecked(False)
        self.window.show_ghost_cb.setStyleSheet("QCheckBox { color: black; font-size: 11px; }")
        self.window.show_ghost_cb.stateChanged.connect(self.window.on_ghost_visibility_toggled)
        controls_layout.addWidget(self.window.show_ghost_cb)

        controls_layout.addStretch()
        layout.addWidget(controls)

        # Console
        console_frame = QFrame()
        console_frame.setStyleSheet("QFrame { background-color: white; border-top: 1px solid #dee2e6; }")
        console_layout = QVBoxLayout(console_frame)
        console_layout.setContentsMargins(10, 10, 10, 10)

        header_layout = QHBoxLayout()
        header = QLabel("Console")
        header.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        header_layout.addWidget(header)

        copy_btn = QPushButton("Copy Console")
        copy_btn.setMaximumWidth(120)
        copy_btn.setStyleSheet(self._get_btn_style("#6c757d", "#5a6268", padding="4px 12px", font_size="11px"))
        copy_btn.clicked.connect(self.window.copy_console_to_clipboard)
        header_layout.addStretch()
        header_layout.addWidget(copy_btn)
        console_layout.addLayout(header_layout)

        self.window.console = QTextEdit()
        self.window.console.setReadOnly(True)
        self.window.console.setMaximumHeight(200)
        self.window.console.setStyleSheet("""
            QTextEdit { background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px;
                        font-family: 'Courier New', monospace; font-size: 10px; color: #212529; padding: 8px; }
        """)
        console_layout.addWidget(self.window.console)
        layout.addWidget(console_frame)

        return panel

    def _create_upload_group(self, layout):
        group = self._create_group_box("Load CAD File")
        group_layout = QVBoxLayout()

        self.window.load_btn = QPushButton("Browse CAD/Mesh File")
        self.window.load_btn.setStyleSheet(self._get_btn_style("#0d6efd", "#0b5ed7"))
        self.window.load_btn.clicked.connect(self.window.load_cad_file)
        group_layout.addWidget(self.window.load_btn)

        self.window.file_label = QLabel("No file loaded")
        self.window.file_label.setWordWrap(True)
        self.window.file_label.setStyleSheet("color: #6c757d; font-size: 11px;")
        group_layout.addWidget(self.window.file_label)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def _create_quality_group(self, layout):
        group = self._create_group_box("Mesh Quality Settings")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(10)

        # Preset
        h = QHBoxLayout()
        h.addWidget(self._create_label("Quality Preset:"))
        self.window.quality_preset = QComboBox()
        self.window.quality_preset.addItems(["Coarse", "Medium", "Fine", "Very Fine", "Custom"])
        self.window.quality_preset.setCurrentIndex(1)
        self.window.quality_preset.setStyleSheet(self._get_combo_style())
        self.window.quality_preset.currentTextChanged.connect(self.window.on_quality_preset_changed)
        h.addWidget(self.window.quality_preset, 1)
        group_layout.addLayout(h)

        # Target Elements
        h = QHBoxLayout()
        h.addWidget(self._create_label("Target Elements:"))
        self.window.target_elements = QSpinBox()
        self.window.target_elements.setRange(100, 1000000)
        self.window.target_elements.setValue(10000)
        self.window.target_elements.setSingleStep(1000)
        self.window.target_elements.setStyleSheet(self._get_spin_style())
        h.addWidget(self.window.target_elements, 1)
        group_layout.addLayout(h)

        # Max Size
        h = QHBoxLayout()
        h.addWidget(self._create_label("Max Element Size:"))
        self.window.max_size = QSpinBox()
        self.window.max_size.setRange(1, 10000)
        self.window.max_size.setValue(100)
        self.window.max_size.setSuffix(" mm")
        self.window.max_size.setFixedWidth(100)
        self.window.max_size.setStyleSheet(self._get_spin_style())
        h.addWidget(self.window.max_size, 1)
        group_layout.addLayout(h)

        # Strategy
        h = QHBoxLayout()
        h.addWidget(self._create_label("Mesh Strategy:"))
        self.window.mesh_strategy = QComboBox()
        self.window.mesh_strategy.addItems([
            "Tetrahedral (Delaunay)", "Hex Dominant (Subdivision)", 
            "Hex Dominant Testing", "Polyhedral (Dual)"
        ])
        self.window.mesh_strategy.setCurrentIndex(0)
        self.window.mesh_strategy.setStyleSheet(self._get_combo_style())
        h.addWidget(self.window.mesh_strategy, 1)
        group_layout.addLayout(h)

        # Checkboxes
        self.window.curvature_adaptive = QCheckBox("Curvature-Adaptive Meshing")
        self.window.curvature_adaptive.setStyleSheet("color: black; font-size: 11px;")
        group_layout.addWidget(self.window.curvature_adaptive)

        self.window.save_stl = QCheckBox("Save intermediate STL files")
        self.window.save_stl.setStyleSheet("color: black; font-size: 11px;")
        group_layout.addWidget(self.window.save_stl)

        self.window.export_ansys = QCheckBox("Export for ANSYS Fluent (.cgns)")
        self.window.export_ansys.setStyleSheet("color: black; font-size: 11px;")
        self.window.export_ansys.setToolTip("Exports .cgns file with Physical Groups for boundary conditions")
        group_layout.addWidget(self.window.export_ansys)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def _create_crosssection_group(self, layout):
        group = self._create_group_box("Cross-Section Viewer")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(8)

        self.window.crosssection_enabled = QCheckBox("Enable Cross-Section")
        self.window.crosssection_enabled.setStyleSheet("font-size: 11px; color: #495057;")
        self.window.crosssection_enabled.stateChanged.connect(self.window.on_crosssection_toggled)
        group_layout.addWidget(self.window.crosssection_enabled)

        # Axis
        h = QHBoxLayout()
        h.addWidget(self._create_label("Clip Axis:"))
        self.window.clip_axis_combo = QComboBox()
        self.window.clip_axis_combo.addItems(["X", "Y", "Z"])
        self.window.clip_axis_combo.setCurrentText("Z")
        self.window.clip_axis_combo.setEnabled(False)
        self.window.clip_axis_combo.setStyleSheet(self._get_combo_style())
        self.window.clip_axis_combo.currentTextChanged.connect(self.window.on_clip_axis_changed)
        h.addWidget(self.window.clip_axis_combo, 1)
        group_layout.addLayout(h)

        # Cell Mode
        h = QHBoxLayout()
        h.addWidget(self._create_label("Slice Cells:"))
        self.window.crosssection_cell_combo = QComboBox()
        self.window.crosssection_cell_combo.addItems(["Auto", "Tetrahedra", "Hexahedra"])
        self.window.crosssection_cell_combo.setCurrentText("Auto")
        self.window.crosssection_cell_combo.setEnabled(False)
        self.window.crosssection_cell_combo.setStyleSheet(self._get_combo_style())
        self.window.crosssection_cell_combo.currentTextChanged.connect(self.window.on_crosssection_element_mode_changed)
        h.addWidget(self.window.crosssection_cell_combo, 1)
        group_layout.addLayout(h)

        # Offset
        h = QHBoxLayout()
        h.addWidget(self._create_label("Offset:"))
        self.window.clip_offset_value_label = QLabel("0%")
        self.window.clip_offset_value_label.setStyleSheet("font-size: 11px; color: #007bff; font-weight: 600;")
        h.addWidget(self.window.clip_offset_value_label)
        group_layout.addLayout(h)

        self.window.clip_offset_slider = QSlider(Qt.Horizontal)
        self.window.clip_offset_slider.setRange(-50, 50)
        self.window.clip_offset_slider.setValue(0)
        self.window.clip_offset_slider.setEnabled(False)
        self.window.clip_offset_slider.valueChanged.connect(self.window.on_clip_offset_changed)
        group_layout.addWidget(self.window.clip_offset_slider)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def _create_viz_group(self, layout):
        group = self._create_group_box("Quality Visualization")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(8)

        # Metric
        h = QHBoxLayout()
        h.addWidget(self._create_label("Metric:"))
        self.window.viz_metric_combo = QComboBox()
        self.window.viz_metric_combo.addItems(["SICN (Min)", "Gamma", "Skewness", "Aspect Ratio"])
        self.window.viz_metric_combo.setStyleSheet(self._get_combo_style())
        self.window.viz_metric_combo.currentTextChanged.connect(self.window.on_viz_metric_changed)
        h.addWidget(self.window.viz_metric_combo, 1)
        group_layout.addLayout(h)

        # Opacity
        h = QHBoxLayout()
        h.addWidget(self._create_label("Opacity:"))
        self.window.viz_opacity_spin = QDoubleSpinBox()
        self.window.viz_opacity_spin.setRange(0.0, 1.0)
        self.window.viz_opacity_spin.setSingleStep(0.1)
        self.window.viz_opacity_spin.setValue(1.0)
        self.window.viz_opacity_spin.setStyleSheet(self._get_spin_style())
        self.window.viz_opacity_spin.valueChanged.connect(self.window.on_viz_opacity_changed)
        h.addWidget(self.window.viz_opacity_spin, 1)
        group_layout.addLayout(h)

        # Range Slider
        group_layout.addWidget(self._create_label("Show Quality Range:", margin_top=5))
        
        h = QHBoxLayout()
        self.window.viz_range_min_label = QLabel("Min: 0.00")
        self.window.viz_range_min_label.setStyleSheet("font-size: 10px; color: #0d6efd; font-weight: bold;")
        self.window.viz_range_max_label = QLabel("Max: 1.00")
        self.window.viz_range_max_label.setStyleSheet("font-size: 10px; color: #dc3545; font-weight: bold;")
        h.addWidget(self.window.viz_range_min_label)
        h.addStretch()
        h.addWidget(self.window.viz_range_max_label)
        group_layout.addLayout(h)

        self.window.viz_range_slider = QRangeSlider(Qt.Horizontal)
        self.window.viz_range_slider.setMinimum(0)
        self.window.viz_range_slider.setMaximum(100)
        self.window.viz_range_slider.setValue((0, 100))
        self.window.viz_range_slider.setStyleSheet("""
            QRangeSlider {
                qproperty-barColor: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #dc3545, stop:0.5 #ffc107, stop:1 #28a745);
            }
        """)
        self.window.viz_range_slider.valueChanged.connect(self.window.on_viz_range_slider_changed)
        group_layout.addWidget(self.window.viz_range_slider)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def _create_progress_group(self, layout):
        group = self._create_group_box("Progress")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(8)

        group_layout.addWidget(self._create_label("Overall Progress", bold=True))
        self.window.master_bar = QProgressBar()
        self.window.master_bar.setStyleSheet(self._get_progress_style(height=24, color="#0d6efd"))
        self.window.master_bar.setValue(0)
        self.window.master_bar.setFormat("0% - Ready")
        group_layout.addWidget(self.window.master_bar)

        group_layout.addWidget(self._create_label("Current Stage", bold=True, margin_top=8))
        self.window.current_process_label = QLabel("Waiting to start...")
        self.window.current_process_label.setStyleSheet("font-size: 9px; color: #6c757d; font-style: italic;")
        group_layout.addWidget(self.window.current_process_label)

        self.window.current_process_bar = QProgressBar()
        self.window.current_process_bar.setStyleSheet(self._get_progress_style(height=18, color="#6c757d"))
        self.window.current_process_bar.setValue(0)
        group_layout.addWidget(self.window.current_process_bar)

        group_layout.addWidget(self._create_label("Completed Stages", bold=True, margin_top=8))
        
        self.window.completed_stages_scroll = QScrollArea()
        self.window.completed_stages_scroll.setWidgetResizable(True)
        self.window.completed_stages_scroll.setMaximumHeight(120)
        self.window.completed_stages_scroll.setStyleSheet("QScrollArea { border: 1px solid #dee2e6; background-color: #f8f9fa; }")
        
        self.window.completed_stages_widget = QWidget()
        self.window.completed_stages_layout = QVBoxLayout(self.window.completed_stages_widget)
        self.window.completed_stages_layout.setSpacing(2)
        self.window.completed_stages_layout.setContentsMargins(5, 5, 5, 5)
        self.window.completed_stages_layout.addStretch()
        
        self.window.completed_stages_scroll.setWidget(self.window.completed_stages_widget)
        group_layout.addWidget(self.window.completed_stages_scroll)

        group.setLayout(group_layout)
        layout.addWidget(group)

    # Helpers
    def _create_group_box(self, title):
        group = QGroupBox(title)
        group.setStyleSheet("""
            QGroupBox { font-weight: bold; border: 1px solid #dee2e6; border-radius: 5px; margin-top: 10px; padding-top: 10px; color: #212529; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #212529; }
        """)
        return group

    def _create_label(self, text, bold=False, margin_top=0):
        label = QLabel(text)
        style = f"font-size: 11px; color: #495057; margin-top: {margin_top}px;"
        if bold: style += " font-weight: 600;"
        label.setStyleSheet(style)
        return label

    def _get_btn_style(self, bg, hover, padding="12px", font_size="13px"):
        return f"""
            QPushButton {{ background-color: {bg}; color: white; border: none; padding: {padding}; border-radius: 5px; font-size: {font_size}; font-weight: bold; }}
            QPushButton:hover:enabled {{ background-color: {hover}; }}
            QPushButton:disabled {{ background-color: #e9ecef; color: #adb5bd; }}
        """

    def _get_combo_style(self):
        return """
            QComboBox { padding: 4px; border: 1px solid #ced4da; border-radius: 4px; background-color: white; color: black; font-size: 11px; }
            QComboBox:disabled { background-color: #e9ecef; color: #6c757d; }
        """

    def _get_spin_style(self):
        return "QSpinBox, QDoubleSpinBox { padding: 4px; border: 1px solid #ced4da; border-radius: 4px; background-color: white; color: #212529; font-size: 11px; }"

    def _get_progress_style(self, height, color):
        return f"""
            QProgressBar {{ border: 1px solid {color}; border-radius: 3px; text-align: center; background-color: #f8f9fa; height: {height}px; font-size: 9px; font-weight: bold; color: #212529; }}
            QProgressBar::chunk {{ background-color: {color}; border-radius: 2px; }}
        """
