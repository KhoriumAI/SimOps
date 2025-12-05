"""
Main GUI Window
================

ModernMeshGenGUI - Main application window with mesh generation controls.
"""

import sys
import os
import json
import logging
import time
from pathlib import Path
from typing import Optional, Dict

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QProgressBar, QGroupBox,
    QSplitter, QFileDialog, QFrame, QScrollArea, QGridLayout,
    QCheckBox, QSizePolicy, QSlider, QSpinBox, QComboBox, QDoubleSpinBox
)
from qtrangeslider import QRangeSlider
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor

from .workers import MeshWorker
from .vtk_viewer import VTK3DViewer
from .utils import setup_logging

# Paintbrush imports
PAINTBRUSH_AVAILABLE = False
try:
    # Add project root to path
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from paintbrush_widget import PaintbrushWidget
    from core.paintbrush_geometry import PaintbrushSelector
    from strategies.paintbrush_strategy import PaintbrushStrategy
    PAINTBRUSH_AVAILABLE = True
    print("[OK] Paintbrush feature loaded successfully")
except ImportError as e:
    print(f"[!] Paintbrush feature not available: {e}")


class ModernMeshGenGUI(QMainWindow):
    """Final polished GUI with AI chatbox"""

    def __init__(self):
        super().__init__()
        self.cad_file = None
        self.mesh_file = None
        self.worker = MeshWorker()
        self.phase_bars = {}
        self.phase_labels = {}
        self.phase_base_names = {}
        self.active_phase = None
        self.dot_count = 0
        
        # Master progress tracking
        self.phase_weights = {}  # {phase_name: percentage_weight}
        self.completed_phases = []  # [phase_name, ...]
        self.phase_completion_times = {}  # {phase_name: seconds}
        self.mesh_start_time = None
        self.master_progress = 0.0  # 0-100

        # AI chatbox
        self.chatbox = None
        self.chatbox_visible = False

        # Paintbrush refinement
        self.paintbrush_selector = None
        self.paintbrush_widget = None
        if PAINTBRUSH_AVAILABLE:
            self.paintbrush_selector = PaintbrushSelector()

        # Animation timer for jumping dots
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_timer.setInterval(400)  # Update every 400ms

        self.worker.signals.log.connect(self.add_log)
        self.worker.signals.progress.connect(self.update_progress)
        self.worker.signals.phase_complete.connect(self.mark_phase_complete)
        self.worker.signals.finished.connect(self.on_mesh_finished)

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Khorium MeshGen - Parallel Edition")
        self.setGeometry(100, 50, 1600, 850)  # Reduced height to fit more screens

        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(248, 249, 250))
        palette.setColor(QPalette.WindowText, QColor(33, 37, 41))
        palette.setColor(QPalette.Base, QColor(255, 255, 255))
        palette.setColor(QPalette.Text, QColor(33, 37, 41))
        palette.setColor(QPalette.Button, QColor(255, 255, 255))
        palette.setColor(QPalette.ButtonText, QColor(33, 37, 41))
        self.setPalette(palette)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Create a container for left panel and chatbox (they'll swap)
        self.left_container = QWidget()
        left_container_layout = QVBoxLayout(self.left_container)
        left_container_layout.setContentsMargins(0, 0, 0, 0)
        left_container_layout.setSpacing(0)

        # Create left panel
        logging.info("Creating left panel...")
        self.left_panel = self.create_left_panel()
        left_container_layout.addWidget(self.left_panel)
        logging.info("Left panel created successfully")

        # Try to create chatbox (optional - may fail if dependencies not installed)
        logging.info("Attempting to initialize chatbox...")
        try:
            logging.info("Importing ChatboxWidget from ui.chatbox_widget...")
            from ui.chatbox_widget import ChatboxWidget
            logging.info("ChatboxWidget imported successfully")

            logging.info("Creating ChatboxWidget instance...")
            self.chatbox = ChatboxWidget()
            logging.info(f"ChatboxWidget instance created: {self.chatbox}")

            self.chatbox.setVisible(False)  # Hidden by default
            left_container_layout.addWidget(self.chatbox)

            # Connect close button signal to toggle function
            self.chatbox.close_requested.connect(self.toggle_chatbox)

            logging.info("AI chatbox initialized successfully")
            print("AI chatbox initialized successfully")

            # Re-enable the chat button now that chatbox is available
            if hasattr(self, 'chat_toggle_btn'):
                logging.info("Re-enabling chat button...")
                self.chat_toggle_btn.setEnabled(True)
                self.chat_toggle_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #0d6efd;
                        color: white;
                        border: none;
                        padding: 8px 16px;
                        border-radius: 5px;
                        font-size: 12px;
                        font-weight: bold;
                    }
                    QPushButton:hover { background-color: #0b5ed7; }
                    QPushButton:checked {
                        background-color: #198754;
                    }
                """)
                self.chat_toggle_btn.setToolTip("Toggle AI assistant (Claude)")
                logging.info("Chat button re-enabled successfully")
                print("Chat button re-enabled")
            else:
                logging.warning("chat_toggle_btn not found - cannot re-enable")
        except ImportError as e:
            logging.error(f"ImportError during chatbox initialization: {e}")
            print(f"[!] AI chatbox not available (ImportError): {e}")
            print("  Install dependencies: pip install anthropic python-dotenv")
            self.chatbox = None
        except Exception as e:
            logging.error(f"Exception during chatbox initialization: {e}")
            logging.error(f"Exception type: {type(e)}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            print(f"AI chatbox initialization failed: {e}")
            traceback.print_exc()
            self.chatbox = None

        main_layout.addWidget(self.left_container)

        logging.info("Creating right panel...")
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 1)
        logging.info("Right panel created successfully")

        logging.info("="*60)
        logging.info("init_ui() completed successfully")
        logging.info(f"Final chatbox status: {self.chatbox}")
        logging.info("="*60)

    def create_left_panel(self):
        panel = QFrame()
        panel.setMaximumWidth(380)
        panel.setStyleSheet("QFrame { background-color: white; border-right: 1px solid #dee2e6; }")

        # Use scroll area to handle overflow
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { background-color: white; border: none; }")

        # Content widget inside scroll area
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)  # Reduced from 15

        # Header with title and chat button
        header_layout = QHBoxLayout()

        title = QLabel("Khorium MeshGen")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        header_layout.addWidget(title)

        header_layout.addStretch()

        # AI Chat toggle button
        self.chat_toggle_btn = QPushButton("💬 AI Chat")
        self.chat_toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #0d6efd;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #0b5ed7; }
            QPushButton:checked {
                background-color: #198754;
            }
        """)
        self.chat_toggle_btn.setCheckable(True)
        self.chat_toggle_btn.clicked.connect(self.toggle_chatbox)
        self.chat_toggle_btn.setToolTip("Toggle AI assistant (Claude)")
        logging.info(f"Checking chatbox status: self.chatbox = {self.chatbox}")
        if not self.chatbox:
            logging.warning("Chatbox is None - disabling button")
            self.chat_toggle_btn.setEnabled(False)
            self.chat_toggle_btn.setStyleSheet("""
                QPushButton {
                    background-color: #6c757d;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 5px;
                    font-size: 12px;
                    font-weight: bold;
                    opacity: 0.6;
                }
            """)
            self.chat_toggle_btn.setToolTip("AI Chat unavailable - Check /tmp/meshgen_gui_debug.log")
            print(f"[DEBUG] Chat button disabled - chatbox is None")
            logging.info("Chat button set to disabled state")
        else:
            logging.info("Chatbox is initialized - enabling button")
            print(f"[DEBUG] Chat button enabled - chatbox initialized")
            logging.info("Chat button set to enabled state")
        header_layout.addWidget(self.chat_toggle_btn)

        layout.addLayout(header_layout)

        subtitle = QLabel("Parallel Mesh Generation (3-5x Faster)")
        subtitle.setFont(QFont("Arial", 10))
        subtitle.setStyleSheet("color: #198754; font-weight: bold;")
        layout.addWidget(subtitle)

        # System info
        from multiprocessing import cpu_count
        cores = cpu_count()
        workers = max(1, cores - 2)
        system_info = QLabel(f"System: {cores} cores | {workers} parallel workers")
        system_info.setFont(QFont("Arial", 9))
        system_info.setStyleSheet("color: #6c757d; padding: 5px 0;")
        layout.addWidget(system_info)

        # Upload section
        upload_group = QGroupBox("Load CAD File")
        upload_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #dee2e6;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                color: #212529;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #212529;
            }
        """)
        upload_layout = QVBoxLayout()

        self.load_btn = QPushButton("Browse CAD/Mesh File")
        self.load_btn.setStyleSheet("""
            QPushButton {
                background-color: #0d6efd;
                color: white;
                border: none;
                padding: 12px;
                border-radius: 5px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #0b5ed7; }
        """)
        self.load_btn.clicked.connect(self.load_cad_file)
        upload_layout.addWidget(self.load_btn)

        self.file_label = QLabel("No file loaded")
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet("color: #6c757d; font-size: 11px;")
        upload_layout.addWidget(self.file_label)

        upload_group.setLayout(upload_layout)
        layout.addWidget(upload_group)

        # Mesh Quality Settings
        quality_group = QGroupBox("Mesh Quality Settings")
        quality_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #dee2e6;
                border-radius: 5px;
                margin-top: 12px;
                padding-top: 12px;
                color: #212529;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #212529;
            }
        """)
        quality_layout = QVBoxLayout()
        quality_layout.setSpacing(10)

        # Quality preset dropdown
        preset_layout = QHBoxLayout()
        preset_label = QLabel("Quality Preset:")
        preset_label.setStyleSheet("font-size: 11px; color: #495057;")
        preset_layout.addWidget(preset_label)

        self.quality_preset = QComboBox()
        self.quality_preset.addItems(["Coarse", "Medium", "Fine", "Very Fine", "Custom"])
        self.quality_preset.setCurrentIndex(1)  # Default to Medium
        self.quality_preset.setStyleSheet("""
            QComboBox {
                padding: 5px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                background-color: white;
                color: black;
                font-size: 11px;
            }
        """)
        self.quality_preset.currentTextChanged.connect(self.on_quality_preset_changed)
        preset_layout.addWidget(self.quality_preset, 1)
        quality_layout.addLayout(preset_layout)

        # Target element count
        target_layout = QHBoxLayout()
        target_label = QLabel("Target Elements:")
        target_label.setStyleSheet("font-size: 11px; color: #495057;")
        target_layout.addWidget(target_label)

        self.target_elements = QSpinBox()
        self.target_elements.setRange(100, 1000000)
        self.target_elements.setValue(10000)
        self.target_elements.setSingleStep(1000)
        self.target_elements.setStyleSheet("""
            QSpinBox {
                padding: 5px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                background-color: white;
                color: #212529;
                font-size: 11px;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                width: 16px;
                background-color: #f8f9fa;
                border: 1px solid #ced4da;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background-color: #e9ecef;
            }
        """)
        target_layout.addWidget(self.target_elements, 1)
        quality_layout.addLayout(target_layout)

        # Max element size only (ANSYS-style) - label and control on same line
        size_layout = QHBoxLayout()
        size_label = QLabel("Max Element Size:")
        size_label.setStyleSheet("font-size: 11px; color: #495057;")
        size_layout.addWidget(size_label)

        self.max_size = QSpinBox()
        self.max_size.setRange(1, 10000)
        self.max_size.setValue(100)
        self.max_size.setSuffix(" mm")
        self.max_size.setFixedWidth(100)  # Compact width
        self.max_size.setButtonSymbols(QSpinBox.UpDownArrows)
        self.max_size.setStyleSheet("""
            QSpinBox {
                padding: 4px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                background-color: white;
                color: #212529;
                font-size: 11px;
            }
        """)
        size_layout.addWidget(self.max_size, 1)
        quality_layout.addLayout(size_layout)

        # Mesh strategy selector
        strategy_layout = QHBoxLayout()
        strategy_label = QLabel("Mesh Strategy:")
        strategy_label.setStyleSheet("font-size: 11px; color: #495057;")
        strategy_layout.addWidget(strategy_label)

        self.mesh_strategy = QComboBox()
        self.mesh_strategy.addItems([
            "Tetrahedral (Delaunay)",
            "Tetrahedral (GPU Delaunay)",
            "Hex Dominant (Subdivision)",
            "Hex Dominant Testing",
            "Polyhedral (Dual)"
        ])
        self.mesh_strategy.setCurrentIndex(0)  # Default to Delaunay
        self.mesh_strategy.setToolTip(
            "Tetrahedral (Delaunay): Robust conformal tet mesh (CPU)\n"
            "Tetrahedral (GPU Delaunay): Ultra-fast GPU Fill & Filter pipeline\n"
            "Hex Dominant (Subdivision): 100% hex mesh via CoACD + subdivision\n"
            "Hex Dominant Testing: Visualize CoACD components\n"
            "Polyhedral (Dual): Polyhedral cells from tet dual"
        )
        self.mesh_strategy.setStyleSheet("""
            QComboBox {
                padding: 5px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                background-color: white;
                color: black;
                font-size: 11px;
            }
        """)
        strategy_layout.addWidget(self.mesh_strategy, 1)
        quality_layout.addLayout(strategy_layout)

        # Curvature-adaptive checkbox
        self.curvature_adaptive = QCheckBox("Curvature-Adaptive Meshing")
        self.curvature_adaptive.setStyleSheet("color: black; font-size: 11px;")
        self.curvature_adaptive.setToolTip("Refine mesh on curved surfaces")
        quality_layout.addWidget(self.curvature_adaptive)
        
        # STL Export checkbox
        self.save_stl = QCheckBox("Save intermediate STL files")
        self.save_stl.setStyleSheet("color: black; font-size: 11px;")
        self.save_stl.setToolTip("Save STL files from STEP conversion and CoACD decomposition")
        self.save_stl.setChecked(False)  # Default off
        quality_layout.addWidget(self.save_stl)

        # ANSYS Export mode selector
        ansys_layout = QHBoxLayout()
        ansys_label = QLabel("ANSYS Export:")
        ansys_label.setStyleSheet("color: black; font-size: 11px; font-weight: bold;")
        self.ansys_mode = QComboBox()
        self.ansys_mode.addItems(["None", "CFD (Fluent)", "FEA (Mechanical)"])
        self.ansys_mode.setStyleSheet("color: black; font-size: 11px;")
        self.ansys_mode.setToolTip(
            "CFD: Linear elements (Tet4), .msh v2.2 for Fluent\n"
            "FEA: Quadratic elements (Tet10), .bdf for Mechanical"
        )
        ansys_layout.addWidget(ansys_label)
        ansys_layout.addWidget(self.ansys_mode, 1)
        quality_layout.addLayout(ansys_layout)

        quality_group.setLayout(quality_layout)
        layout.addWidget(quality_group)

        # Cross-Section Viewer
        crosssection_group = QGroupBox("Cross-Section Viewer")
        crosssection_group.setStyleSheet("""
            QGroupBox {
                font-weight: 600;
                font-size: 12px;
                color: #2c3e50;
                border: 1px solid #dee2e6;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 12px;
                background-color: #f8f9fa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                background-color: #f8f9fa;
            }
        """)
        crosssection_layout = QVBoxLayout()
        crosssection_layout.setSpacing(8)

        # Enable cross-section checkbox
        self.crosssection_enabled = QCheckBox("Enable Cross-Section")
        self.crosssection_enabled.setStyleSheet("font-size: 11px; color: #495057;")
        self.crosssection_enabled.setChecked(False)
        self.crosssection_enabled.stateChanged.connect(self.on_crosssection_toggled)
        crosssection_layout.addWidget(self.crosssection_enabled)

        # Axis selection dropdown
        axis_layout = QHBoxLayout()
        axis_label = QLabel("Clip Axis:")
        axis_label.setStyleSheet("font-size: 11px; color: #495057;")
        axis_layout.addWidget(axis_label)

        self.clip_axis_combo = QComboBox()
        self.clip_axis_combo.addItems(["X", "Y", "Z"])
        self.clip_axis_combo.setCurrentText("Z")
        self.clip_axis_combo.setEnabled(False)
        self.clip_axis_combo.setStyleSheet("""
            QComboBox {
                padding: 4px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                background-color: white;
                color: #212529;
                font-size: 11px;
            }
            QComboBox:disabled {
                background-color: #e9ecef;
                color: #6c757d;
            }
        """)
        self.clip_axis_combo.currentTextChanged.connect(self.on_clip_axis_changed)
        axis_layout.addWidget(self.clip_axis_combo, 1)
        crosssection_layout.addLayout(axis_layout)
        
        # Generate button
        self.generate_btn = QPushButton("Generate Mesh (Parallel)")
        self.generate_btn.setEnabled(False)
        self.generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #198754;
                color: white;
                border: none;
                padding: 15px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover:enabled { background-color: #157347; }
            QPushButton:disabled { background-color: #e9ecef; color: #adb5bd; }
        """)
        self.generate_btn.clicked.connect(self.start_mesh_generation)
        layout.addWidget(self.generate_btn)

        # Stop button
        self.stop_btn = QPushButton("Stop Meshing")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover:enabled { background-color: #bb2d3b; }
            QPushButton:disabled { background-color: #e9ecef; color: #adb5bd; }
        """)
        self.stop_btn.clicked.connect(self.stop_mesh_generation)
        layout.addWidget(self.stop_btn)

        # Refine Mesh Quality button
        self.refine_btn = QPushButton("Refine Mesh Quality")
        self.refine_btn.setEnabled(False)
        self.refine_btn.setStyleSheet("""
            QPushButton {
                background-color: #17a2b8;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover:enabled { background-color: #138496; }
            QPushButton:disabled { background-color: #e9ecef; color: #adb5bd; }
        """)
        self.refine_btn.clicked.connect(self.refine_mesh_quality)
        layout.addWidget(self.refine_btn)

        # Progress bars - COMPACT
        progress_group = QGroupBox("Progress")
        progress_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #dee2e6;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                color: #212529;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #212529;
            }
        """)
        progress_layout = QVBoxLayout()
        progress_layout.setSpacing(8)
        
        # === MASTER PROGRESS BAR ===
        master_label = QLabel("Overall Progress")
        master_label.setStyleSheet("font-size: 10px; color: #495057; font-weight: 600; margin-bottom: 2px;")
        progress_layout.addWidget(master_label)
        
        self.master_bar = QProgressBar()
        self.master_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #0d6efd;
                border-radius: 4px;
                text-align: center;
                background-color: #e7f1ff;
                height: 24px;
                font-size: 11px;
                font-weight: bold;
                color: #212529;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0d6efd, stop:1 #0a58ca);
                border-radius: 2px;
            }
        """)
        self.master_bar.setMaximum(100)
        self.master_bar.setValue(0)
        self.master_bar.setFormat("0% - Ready")
        progress_layout.addWidget(self.master_bar)
        
        # === CURRENT PROCESS BAR ===
        current_label = QLabel("Current Stage")
        current_label.setStyleSheet("font-size: 10px; color: #495057; font-weight: 600; margin-top: 8px; margin-bottom: 2px;")
        progress_layout.addWidget(current_label)
        
        self.current_process_label = QLabel("Waiting to start...")
        self.current_process_label.setStyleSheet("font-size: 9px; color: #6c757d; font-style: italic; margin-bottom: 2px;")
        progress_layout.addWidget(self.current_process_label)
        
        self.current_process_bar = QProgressBar()
        self.current_process_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #6c757d;
                border-radius: 3px;
                text-align: center;
                background-color: #f8f9fa;
                height: 18px;
                font-size: 9px;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #6c757d;
                border-radius: 2px;
            }
        """)
        self.current_process_bar.setMaximum(100)
        self.current_process_bar.setValue(0)
        self.current_process_bar.setFormat("%p%")
        progress_layout.addWidget(self.current_process_bar)
        
        # === COMPLETED STAGES LIST ===
        completed_label = QLabel("Completed Stages")
        completed_label.setStyleSheet("font-size: 10px; color: #495057; font-weight: 600; margin-top: 8px; margin-bottom: 2px;")
        progress_layout.addWidget(completed_label)
        
        # Scroll area for completed stages
        self.completed_stages_scroll = QScrollArea()
        self.completed_stages_scroll.setWidgetResizable(True)
        self.completed_stages_scroll.setMaximumHeight(120)
        self.completed_stages_scroll.setStyleSheet("""
            QScrollArea {
                border: 1px solid #dee2e6;
                border-radius: 3px;
                background-color: #f8f9fa;
            }
        """)
        
        # Widget to hold completed stages
        self.completed_stages_widget = QWidget()
        self.completed_stages_layout = QVBoxLayout(self.completed_stages_widget)
        self.completed_stages_layout.setSpacing(2)
        self.completed_stages_layout.setContentsMargins(5, 5, 5, 5)
        self.completed_stages_layout.addStretch()  # Push items to top
        
        self.completed_stages_scroll.setWidget(self.completed_stages_widget)
        progress_layout.addWidget(self.completed_stages_scroll)

        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        # Cross-section Controls
        crosssection_group = QGroupBox("Cross-Section View")
        crosssection_group.setCheckable(True)
        crosssection_group.setChecked(False)
        self.crosssection_enabled = crosssection_group
        crosssection_group.toggled.connect(self.on_crosssection_toggled)
        crosssection_group.setStyleSheet("""
            QGroupBox {
                font-weight: 600;
                font-size: 12px;
                color: #2c3e50;
                border: 1px solid #dee2e6;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 12px;
                background-color: #f8f9fa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                background-color: #f8f9fa;
            }
        """)
        crosssection_layout = QVBoxLayout()
        crosssection_layout.setSpacing(8)

        # Axis selection
        axis_layout = QHBoxLayout()
        axis_label = QLabel("Axis:")
        axis_label.setStyleSheet("font-size: 11px; color: #495057;")
        axis_layout.addWidget(axis_label)

        self.clip_axis_combo = QComboBox()
        self.clip_axis_combo.addItems(["X", "Y", "Z"])
        self.clip_axis_combo.setEnabled(False)
        self.clip_axis_combo.setStyleSheet("""
            QComboBox {
                padding: 4px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                background-color: white;
                color: #212529;
                font-size: 11px;
            }
            QComboBox:disabled {
                background-color: #e9ecef;
                color: #6c757d;
            }
        """)
        self.clip_axis_combo.currentTextChanged.connect(self.on_clip_axis_changed)
        axis_layout.addWidget(self.clip_axis_combo, 1)
        crosssection_layout.addLayout(axis_layout)

        # Cell type selector
        cell_mode_layout = QHBoxLayout()
        cell_mode_label = QLabel("Slice Cells:")
        cell_mode_label.setStyleSheet("font-size: 11px; color: #495057;")
        cell_mode_layout.addWidget(cell_mode_label)
        
        self.crosssection_cell_combo = QComboBox()
        self.crosssection_cell_combo.addItems(["Auto", "Tetrahedra", "Hexahedra"])
        self.crosssection_cell_combo.setCurrentText("Auto")
        self.crosssection_cell_combo.setEnabled(False)
        self.crosssection_cell_combo.setStyleSheet("""
            QComboBox {
                padding: 4px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                background-color: white;
                color: #212529;
                font-size: 11px;
            }
            QComboBox:disabled {
                background-color: #e9ecef;
                color: #6c757d;
            }
        """)
        self.crosssection_cell_combo.currentTextChanged.connect(self.on_crosssection_element_mode_changed)
        cell_mode_layout.addWidget(self.crosssection_cell_combo, 1)
        crosssection_layout.addLayout(cell_mode_layout)

        # Offset slider
        offset_layout = QVBoxLayout()
        offset_layout.setSpacing(3)

        offset_label_layout = QHBoxLayout()
        offset_label = QLabel("Offset:")
        offset_label.setStyleSheet("font-size: 11px; color: #495057;")
        offset_label_layout.addWidget(offset_label)

        self.clip_offset_value_label = QLabel("0%")
        self.clip_offset_value_label.setStyleSheet("font-size: 11px; color: #007bff; font-weight: 600;")
        self.clip_offset_value_label.setAlignment(Qt.AlignRight)
        offset_label_layout.addWidget(self.clip_offset_value_label)
        offset_layout.addLayout(offset_label_layout)

        self.clip_offset_slider = QSlider(Qt.Horizontal)
        self.clip_offset_slider.setRange(-50, 50)
        self.clip_offset_slider.setValue(0)
        self.clip_offset_slider.setEnabled(False)
        self.clip_offset_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background: #e9ecef;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #007bff;
                width: 14px;
                height: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QSlider::handle:horizontal:hover {
                background: #0056b3;
            }
            QSlider::handle:horizontal:disabled {
                background: #6c757d;
            }
        """)
        self.clip_offset_slider.valueChanged.connect(self.on_clip_offset_changed)
        offset_layout.addWidget(self.clip_offset_slider)
        crosssection_layout.addLayout(offset_layout)

        crosssection_group.setLayout(crosssection_layout)
        self.crosssection_group = crosssection_group
        self.crosssection_group.setVisible(False)
        layout.addWidget(self.crosssection_group)

        # Quality Visualization Controls
        viz_group = QGroupBox("Quality Visualization")
        viz_group.setStyleSheet("""
            QGroupBox {
                font-weight: 600;
                font-size: 12px;
                color: #2c3e50;
                border: 1px solid #dee2e6;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 12px;
                background-color: #f8f9fa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                background-color: #f8f9fa;
            }
        """)
        viz_layout = QVBoxLayout()
        viz_layout.setSpacing(8)

        # Metric Selector
        metric_layout = QHBoxLayout()
        metric_label = QLabel("Metric:")
        metric_label.setStyleSheet("font-size: 11px; color: #495057;")
        metric_layout.addWidget(metric_label)

        self.viz_metric_combo = QComboBox()
        self.viz_metric_combo.addItems(["SICN (Min)", "Gamma", "Skewness", "Aspect Ratio"])
        self.viz_metric_combo.setCurrentIndex(0)
        self.viz_metric_combo.setStyleSheet("""
            QComboBox {
                padding: 4px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                background-color: white;
                color: #212529;
                font-size: 11px;
            }
        """)
        self.viz_metric_combo.currentTextChanged.connect(self.on_viz_metric_changed)
        metric_layout.addWidget(self.viz_metric_combo, 1)
        viz_layout.addLayout(metric_layout)

        # Opacity Control (SpinBox)
        opacity_layout = QHBoxLayout()
        opacity_label = QLabel("Opacity:")
        opacity_label.setStyleSheet("font-size: 11px; color: #495057;")
        opacity_layout.addWidget(opacity_label)
        
        self.viz_opacity_spin = QDoubleSpinBox()
        self.viz_opacity_spin.setRange(0.0, 1.0)
        self.viz_opacity_spin.setSingleStep(0.1)
        self.viz_opacity_spin.setValue(1.0)
        self.viz_opacity_spin.setDecimals(1)
        self.viz_opacity_spin.setStyleSheet("""
            QDoubleSpinBox {
                padding: 4px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                background-color: white;
                font-size: 11px;
            }
        """)
        self.viz_opacity_spin.valueChanged.connect(self.on_viz_opacity_changed)
        opacity_layout.addWidget(self.viz_opacity_spin, 1)
        viz_layout.addLayout(opacity_layout)


        # Filter Range (Dual-Handled Range Slider)
        filter_label = QLabel("Show Quality Range:")
        filter_label.setStyleSheet("font-size: 11px; color: #495057; margin-top: 5px;")
        viz_layout.addWidget(filter_label)

        # Value display (shows current min/max)
        range_value_layout = QHBoxLayout()
        self.viz_range_min_label = QLabel("Min: 0.00")
        self.viz_range_min_label.setStyleSheet("font-size: 10px; color: #0d6efd; font-weight: bold;")
        self.viz_range_max_label = QLabel("Max: 1.00")
        self.viz_range_max_label.setStyleSheet("font-size: 10px; color: #dc3545; font-weight: bold;")
        range_value_layout.addWidget(self.viz_range_min_label)
        range_value_layout.addStretch()
        range_value_layout.addWidget(self.viz_range_max_label)
        viz_layout.addLayout(range_value_layout)
        
        # Dual-handled range slider
        self.viz_range_slider = QRangeSlider(Qt.Horizontal)
        self.viz_range_slider.setMinimum(0)
        self.viz_range_slider.setMaximum(100)
        self.viz_range_slider.setValue((0, 100))
        self.viz_range_slider.setStyleSheet("""
            QRangeSlider {
                qproperty-barColor: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #dc3545, stop:0.5 #ffc107, stop:1 #28a745);
            }
        """)
        self.viz_range_slider.valueChanged.connect(self.on_viz_range_slider_changed)
        viz_layout.addWidget(self.viz_range_slider)
        
        # Store quality data ranges for auto-update
        self.quality_data_ranges = {
            'SICN': (0.0, 1.0),
            'Gamma': (0.0, 1.0),
            'Skewness': (0.0, 1.0),
            'Aspect Ratio': (1.0, 10.0)
        }

        viz_group.setLayout(viz_layout)
        self.viz_group = viz_group
        self.viz_group.setVisible(False)
        layout.addWidget(self.viz_group)

        # Paintbrush refinement widget
        print(f"[DEBUG] Paintbrush available: {PAINTBRUSH_AVAILABLE}")
        print(f"[DEBUG] Paintbrush selector: {self.paintbrush_selector}")

        if PAINTBRUSH_AVAILABLE and self.paintbrush_selector:
            try:
                print("[DEBUG] Creating paintbrush widget...")
                self.paintbrush_widget = PaintbrushWidget()
                print("[DEBUG] Paintbrush widget created successfully")

                self.paintbrush_widget.paintbrush_enabled.connect(self.on_paintbrush_toggled)
                self.paintbrush_widget.radius_changed.connect(self.on_brush_radius_changed)
                self.paintbrush_widget.refinement_changed.connect(self.on_refinement_changed)
                self.paintbrush_widget.clear_requested.connect(self.on_clear_painted_regions)
                self.paintbrush_widget.preview_requested.connect(self.on_preview_refinement)
                self.paintbrush_widget.region_deleted.connect(self.on_region_deleted)

                print("[DEBUG] Adding paintbrush widget to layout...")
                layout.addWidget(self.paintbrush_widget)
                print("[DEBUG] Paintbrush widget added to GUI!")
            except Exception as e:
                print(f"[ERROR] Failed to create paintbrush widget: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"[DEBUG] Paintbrush widget NOT added - Available:{PAINTBRUSH_AVAILABLE}, Selector:{self.paintbrush_selector}")

        layout.addStretch()

        # Secret buttons at the bottom
        secret_buttons_layout = QHBoxLayout()
        secret_buttons_layout.setSpacing(5)
        
        # Secret button 1 - Surface mesh only
        self.surface_only_btn = QPushButton("📐")
        self.surface_only_btn.setFixedSize(30, 30)
        self.surface_only_btn.setToolTip("Secret: Generate surface mesh only (2D)")
        self.surface_only_btn.setStyleSheet("""
            QPushButton {
                background-color: #343a40;
                color: white;
                border: none;
                border-radius: 3px;
                font-size: 16px;
                padding: 2px;
            }
            QPushButton:hover { background-color: #495057; }
            QPushButton:pressed { background-color: #0d6efd; }
        """)
        self.surface_only_btn.clicked.connect(self.generate_surface_mesh_only)
        secret_buttons_layout.addWidget(self.surface_only_btn)
        
        # Secret button 2 - hex visualization
        self.hex_mode_btn = QPushButton("🔳")
        self.hex_mode_btn.setFixedSize(30, 30)
        self.hex_mode_btn.setToolTip("Secret: Make tets look like hexes")
        self.hex_mode_btn.setCheckable(True)
        self.hex_mode_btn.setStyleSheet("""
            QPushButton {
                background-color: #343a40;
                color: white;
                border: none;
                border-radius: 3px;
                font-size: 16px;
                padding: 2px;
            }
            QPushButton:hover { background-color: #495057; }
            QPushButton:checked {
                background-color: #0d6efd;
                border: 2px solid #0a58ca;
            }
        """)
        self.hex_mode_btn.clicked.connect(self.toggle_hex_visualization)
        secret_buttons_layout.addWidget(self.hex_mode_btn)
        
        secret_buttons_layout.addStretch()
        layout.addLayout(secret_buttons_layout)

        # Add content to scroll area
        scroll.setWidget(content_widget)

        # Add scroll area to panel
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.addWidget(scroll)

        return panel

    def create_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Viewer controls
        controls = QFrame()
        controls.setStyleSheet("background-color: #f8f9fa; border-bottom: 1px solid #dee2e6;")
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(10, 5, 10, 5)

        axes_cb = QCheckBox("Show XYZ Axes")
        axes_cb.setChecked(True)
        axes_cb.setStyleSheet("QCheckBox { color: black; font-size: 11px; }")
        axes_cb.stateChanged.connect(lambda s: self.viewer.toggle_axes(s == Qt.Checked))
        controls_layout.addWidget(axes_cb)
        
        # Cross-section mode toggle
        # Ghost visibility toggle
        self.show_ghost_cb = QCheckBox("Show Ghost (Above Cut)")
        self.show_ghost_cb.setChecked(False)  # Default: hide ghost for full brightness
        self.show_ghost_cb.setStyleSheet("QCheckBox { color: black; font-size: 11px; }")
        self.show_ghost_cb.setToolTip("Show transparent outline of portion above cross-section")
        self.show_ghost_cb.stateChanged.connect(self.on_ghost_visibility_toggled)
        controls_layout.addWidget(self.show_ghost_cb)
        
        controls_layout.addStretch()

        layout.addWidget(controls)

        # 3D Viewer - Pass self as parent so CustomInteractorStyle can access main GUI
        self.viewer = VTK3DViewer(parent=self)
        layout.addWidget(self.viewer, 2)

        # Console
        console_frame = QFrame()
        console_frame.setStyleSheet("QFrame { background-color: white; border-top: 1px solid #dee2e6; }")
        console_layout = QVBoxLayout(console_frame)
        console_layout.setContentsMargins(10, 10, 10, 10)

        # Console header with copy button
        console_header_layout = QHBoxLayout()
        console_header = QLabel("Console")
        console_header.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        console_header_layout.addWidget(console_header)

        copy_console_btn = QPushButton("Copy Console")
        copy_console_btn.setMaximumWidth(120)
        copy_console_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 4px 12px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
            QPushButton:pressed {
                background-color: #545b62;
            }
        """)
        copy_console_btn.clicked.connect(self.copy_console_to_clipboard)
        console_header_layout.addStretch()
        console_header_layout.addWidget(copy_console_btn)
        console_layout.addLayout(console_header_layout)

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumHeight(200)
        self.console.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                font-family: 'Courier New', monospace;
                font-size: 10px;
                color: #212529;
                padding: 8px;
            }
        """)
        console_layout.addWidget(self.console)

        layout.addWidget(console_frame)
        return panel

    def calculate_suggested_element_counts(self, geom_info: dict):
        """Calculate appropriate element counts based on geometry size"""
        volume = geom_info.get('volume', 0.001)  # cubic meters
        bbox_diag = geom_info.get('bbox_diagonal', 1.0)  # meters

        # Work backwards: Given desired element count, calculate mesh size
        # Formula: element_size = (volume / num_elements)^(1/3)
        # Then set min/max around this average size

        # Helper function to calculate sizes from target count
        def calc_sizes(target_elements):
            avg_size_m = (volume / target_elements) ** (1/3)  # meters
            avg_size_mm = avg_size_m * 1000  # convert to mm

            # Set min/max as ±30% of average
            min_mm = max(0.1, avg_size_mm * 0.7)
            max_mm = max(0.5, avg_size_mm * 1.3)

            return int(target_elements), int(round(min_mm)), int(round(max_mm))

        # Coarse: ~500-2000 elements (fast preview)
        target_coarse = max(500, min(2000, int(volume * 1e6)))  # ~1mm³ per element
        count_c, min_c, max_c = calc_sizes(target_coarse)

        # Medium: ~2000-8000 elements (balanced)
        target_medium = max(2000, min(8000, int(volume * 4e6)))  # ~0.25mm³ per element
        count_m, min_m, max_m = calc_sizes(target_medium)

        # Fine: ~8000-30000 elements (high quality)
        target_fine = max(8000, min(30000, int(volume * 16e6)))  # ~0.0625mm³ per element
        count_f, min_f, max_f = calc_sizes(target_fine)

        # Very Fine: ~30000-100000 elements (max quality)
        target_vfine = max(30000, min(100000, int(volume * 64e6)))  # ~0.015625mm³ per element
        count_vf, min_vf, max_vf = calc_sizes(target_vfine)

        presets = {
            "Coarse": {"target": count_c, "max": max_c},
            "Medium": {"target": count_m, "max": max_m},
            "Fine": {"target": count_f, "max": max_f},
            "Very Fine": {"target": count_vf, "max": max_vf}
        }

        # Store calculated presets for future use
        self.calculated_presets = presets

        # Update current preset values
        current_preset = self.quality_preset.currentText()
        if current_preset in presets:
            values = presets[current_preset]
            self.target_elements.setValue(values["target"])
            self.max_size.setValue(values["max"])
            self.add_log(f"Calculated element counts for geometry (volume={volume:.6f} m³):")
            self.add_log(f"   Coarse: ~{count_c:,} elements (max size: {max_c:.1f} mm)")
            self.add_log(f"   Medium: ~{count_m:,} elements (max size: {max_m:.1f} mm)")
            self.add_log(f"   Fine: ~{count_f:,} elements (max size: {max_f:.1f} mm)")
            self.add_log(f"   Very Fine: ~{count_vf:,} elements (max size: {max_vf:.1f} mm)")

    def on_quality_preset_changed(self, preset: str):
        """Update mesh quality settings based on preset"""
        # Use calculated presets if available, otherwise use defaults
        if hasattr(self, 'calculated_presets') and preset in self.calculated_presets:
            presets = self.calculated_presets
        else:
            presets = {
                "Coarse": {"target": 5000, "max": 200},
                "Medium": {"target": 10000, "max": 100},
                "Fine": {"target": 50000, "max": 50},
                "Very Fine": {"target": 200000, "max": 20}
            }

        presets["Custom"] = None  # Don't change values for custom

        if preset in presets and presets[preset]:
            values = presets[preset]
            self.target_elements.setValue(values["target"])
            self.max_size.setValue(values["max"])

    def on_crosssection_toggled(self, state):
        """Handle cross-section checkbox toggle"""
        enabled = bool(state)
        self.clip_axis_combo.setEnabled(enabled)
        self.clip_offset_slider.setEnabled(enabled)
        if hasattr(self, 'crosssection_cell_combo'):
            self.crosssection_cell_combo.setEnabled(enabled)
        
        self.viewer.set_clipping(
            enabled=enabled,
            axis=self.clip_axis_combo.currentText(),
            offset=self.clip_offset_slider.value()
        )

    def on_clip_axis_changed(self, text):
        """Handle clip axis change"""
        self.viewer.set_clipping(
            enabled=self.crosssection_enabled.isChecked(),
            axis=text,
            offset=self.clip_offset_slider.value()
        )

    def on_clip_offset_changed(self, value):
        """Handle clip offset slider change"""
        self.clip_offset_value_label.setText(f"{value}%")
        self.viewer.set_clipping(
            enabled=self.crosssection_enabled.isChecked(),
            axis=self.clip_axis_combo.currentText(),
            offset=value
        )
    
    def on_crosssection_element_mode_changed(self, text: str):
        """Handle cell type selector for cross-section slices"""
        if not hasattr(self, 'viewer') or not self.viewer:
            return
        mode_map = {
            "Auto": "auto",
            "Tetrahedra": "tetrahedra",
            "Hexahedra": "hexahedra"
        }
        mode = mode_map.get(text, "auto")
        self.viewer.set_cross_section_element_mode(mode)
    def on_ghost_visibility_toggled(self, state):
        """Handle ghost visibility toggle"""
        if hasattr(self, 'above_cut_actor') and self.above_cut_actor:
            if state:  # Checked = show ghost
                self.above_cut_actor.VisibilityOn()
            else:  # Unchecked = hide ghost for full brightness
                self.above_cut_actor.VisibilityOff()
            self.vtk_widget.GetRenderWindow().Render()
            self.add_log(f"Ghost visibility: {'ON' if state else 'OFF'}")

# --- VISUALIZATION HANDLERS ---
    def on_viz_range_slider_changed(self, value):
        """Handle range slider change - value is (min, max) tuple"""
        min_slider, max_slider = value
        
        # Get current metric's data range
        metric = self.viz_metric_combo.currentText()
        data_min, data_max = self.quality_data_ranges.get(metric, (0.0, 1.0))
        
        # Convert slider position (0-100) to actual quality values
        range_span = data_max - data_min
        min_val = data_min + (min_slider / 100.0) * range_span
        max_val = data_min + (max_slider / 100.0) * range_span
        
        # Update labels
        self.viz_range_min_label.setText(f"Min: {min_val:.2f}")
        self.viz_range_max_label.setText(f"Max: {max_val:.2f}")
        
        # Trigger visualization update
        if hasattr(self, 'viewer') and self.viewer:
            self.viewer.update_quality_visualization(
                metric=metric,
                opacity=self.viz_opacity_spin.value(),
                min_val=min_val,
                max_val=max_val
            )

    def on_viz_metric_changed(self, text):
        """Handle metric selection change - auto-update slider range"""
        # Update slider range based on new metric
        data_min, data_max = self.quality_data_ranges.get(text, (0.0, 1.0))
        
        # Reset slider to full range for new metric
        self.viz_range_slider.setValue((0, 100))
        
        # Update labels with new metric's range
        self.viz_range_min_label.setText(f"Min: {data_min:.2f}")
        self.viz_range_max_label.setText(f"Max: {data_max:.2f}")
        
        # Trigger visualization update
        if hasattr(self, 'viewer') and self.viewer:
            self.viewer.update_quality_visualization(
                metric=text,
                opacity=self.viz_opacity_spin.value(),
                min_val=data_min,
                max_val=data_max
            )

    def on_viz_opacity_changed(self, value):
        """Handle opacity change"""
        if hasattr(self, 'viewer') and self.viewer:
            # Get current range from slider
            min_slider, max_slider = self.viz_range_slider.value()
            metric = self.viz_metric_combo.currentText()
            data_min, data_max = self.quality_data_ranges.get(metric, (0.0, 1.0))
            range_span = data_max - data_min
            min_val = data_min + (min_slider / 100.0) * range_span
            max_val = data_min + (max_slider / 100.0) * range_span
            
            self.viewer.update_quality_visualization(
                metric=metric,
                opacity=value,
                min_val=min_val,
                max_val=max_val
            )

    def update_quality_data_ranges(self, per_element_quality, per_element_gamma, per_element_skewness, per_element_aspect_ratio):
        """Update quality data ranges from loaded mesh data"""
        if per_element_quality:
            quality_values = list(per_element_quality.values())
            self.quality_data_ranges['SICN'] = (min(quality_values), max(quality_values))
        
        if per_element_gamma:
            gamma_values = list(per_element_gamma.values())
            self.quality_data_ranges['Gamma'] = (min(gamma_values), max(gamma_values))
        
        if per_element_skewness:
            skew_values = list(per_element_skewness.values())
            self.quality_data_ranges['Skewness'] = (min(skew_values), max(skew_values))
        
        if per_element_aspect_ratio:
            ar_values = list(per_element_aspect_ratio.values())
            # Clamp max to reasonable value
            self.quality_data_ranges['Aspect Ratio'] = (min(ar_values), min(max(ar_values), 20.0))
        
        # Update current metric's display
        metric = self.viz_metric_combo.currentText()
        data_min, data_max = self.quality_data_ranges.get(metric, (0.0, 1.0))
        self.viz_range_min_label.setText(f"Min: {data_min:.2f}")
        self.viz_range_max_label.setText(f"Max: {data_max:.2f}")
    # --------------------------------------

    def load_cad_file(self):
        # Kill any running mesh generation workers before loading new CAD
        if self.worker and self.worker.is_running:
            self.add_log("Stopping previous mesh generation...")
            self.worker.stop()
            self.add_log("[OK] Previous mesh generation stopped")

        # Reset UI state (progress bars, completed stages, etc.)
        self.reset_progress_bars()

        # Clear cached iteration data
        self.iteration_meshes = []
        self.current_iteration = 0

        # Use cad_files folder as default
        default_dir = str(Path(__file__).parent / "cad_files")
        if not Path(default_dir).exists():
            default_dir = str(Path.home())

        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select CAD or Mesh File", default_dir,
            "CAD Files (*.step *.stp *.stl);;Mesh Files (*.msh);;All Files (*)"
        )

        if filepath:
            file_ext = Path(filepath).suffix.lower()

            # Check if it's a mesh file
            if file_ext == '.msh':
                # Load mesh directly
                self.cad_file = None  # No CAD file
                self.file_label.setText(f"{Path(filepath).name}")
                self.generate_btn.setEnabled(False)  # Can't generate from mesh
                self.add_log(f"Loaded mesh: {filepath}")

                # Clear iterations
                self.viewer.clear_iterations()

                # Load the mesh file directly through the viewer
                self.viewer.load_mesh_file(filepath)
                
                # Show post-processing UI sections (quality visualization and cross-section)
                if hasattr(self, 'crosssection_group'):
                    self.crosssection_group.setVisible(True)
                if hasattr(self, 'viz_group'):
                    self.viz_group.setVisible(True)
                
                return

            # Otherwise treat as CAD file
            self.cad_file = filepath
            self.file_label.setText(f"{Path(filepath).name}")
            self.generate_btn.setEnabled(True)
            self.add_log(f"Loaded CAD: {filepath}")

            # Clear any existing AI iteration meshes
            self.viewer.clear_iterations()

            # Load CAD and get geometry info
            geom_info = self.viewer.load_step_file(filepath)
            self.current_geom_info = geom_info  # Store for logging

            # Load geometry for paintbrush (after CAD is loaded)
            if self.paintbrush_selector:
                try:
                    import gmsh
                    gmsh.initialize()
                    gmsh.open(filepath)
                    if self.paintbrush_selector.load_cad_geometry():
                        self.add_log(f"Paintbrush: Loaded {len(self.paintbrush_selector.available_surfaces)} surfaces")
                    gmsh.finalize()
                except Exception as e:
                    print(f"Could not load geometry for paintbrush: {e}")

            # Calculate suggested element counts based on geometry
            if geom_info and 'volume' in geom_info:
                self.calculate_suggested_element_counts(geom_info)
            else:
                self.add_log("[!] Could not calculate geometry volume - using default element counts")


    def calculate_phase_weights(self, element_target, quality_preset):
        """
        Calculate relative time weights for each phase based on element count and quality.
        
        Args:
            element_target: Target number of elements
            quality_preset: "Draft", "Standard", or "High Fidelity"
            
        Returns:
            dict {phase_name: weight_percentage}
        """
        # Base weights (for 10k elements, Standard quality)
        base_weights = {
            'cad': 5,
            'surf': 20,
            'refine': 15,
            '3d': 35,
            'opt': 20,
            'quality': 5
        }
        
        # Complexity multiplier based on element count
        if element_target < 5000:
            complexity = 0.7  # Simpler, faster
        elif element_target < 50000:
            complexity = 1.0  # Standard
        else:
            complexity = 1.5  # Complex, slower
        
        # Quality multipliers
        quality_mult = {
            'Draft': 0.6,
            'Standard': 1.0,
            'High Fidelity': 1.8
        }.get(quality_preset, 1.0)
        
        # Adjust weights
        adjusted = base_weights.copy()
        adjusted['opt'] *= quality_mult  # Optimization scales with quality
        adjusted['refine'] *= quality_mult  # Refinement scales with quality
        adjusted['3d'] *= complexity  # 3D meshing scales with element count
        
        # Normalize to 100%
        total = sum(adjusted.values())
        return {k: (v/total)*100 for k, v in adjusted.items()}
    
    def update_eta(self, current_progress):
        """Update ETA display on master progress bar"""
        if not self.mesh_start_time or current_progress < 5:
            return  # Wait for meaningful data
        
        import time
        elapsed = time.time() - self.mesh_start_time
        
        if current_progress > 5:
            estimated_total = elapsed / (current_progress / 100)
            remaining = estimated_total - elapsed
            
            # Format as "Xm Ys" or "Xs"
            if remaining < 0:
                eta_text = "Finishing..."
            elif remaining < 60:
                eta_text = f"{int(remaining)}s"
            else:
                mins = int(remaining // 60)
                secs = int(remaining % 60)
                eta_text = f"{mins}m {secs}s"
            
            # Update master bar format
            if hasattr(self, 'master_bar'):
                self.master_bar.setFormat(f"{int(current_progress)}% - ETA: {eta_text}")

    def start_mesh_generation(self):
        if not self.cad_file:
            return

        self.generate_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)  # Enable stop button
        self.console.clear()

        for bar in self.phase_bars.values():
            bar.setValue(0)
            bar.setStyleSheet(bar.styleSheet().replace("background-color: #198754", "background-color: #0d6efd"))
        
        # Initialize master progress tracking
        import time
        self.mesh_start_time = time.time()
        self.completed_phases = []
        self.phase_completion_times = {}
        self.master_progress = 0.0
        
        # Calculate phase weights based on target elements and quality preset
        element_target = self.target_elements.value()
        quality_preset = self.quality_preset.currentText()
        self.phase_weights = self.calculate_phase_weights(element_target, quality_preset)
        
        self.add_log(f"[DEBUG] Phase weights calculated: {self.phase_weights}")
        self.add_log(f"[DEBUG] Target elements: {element_target}, Quality: {quality_preset}")
        
        # Reset master bar if it exists
        if hasattr(self, 'master_bar'):
            self.master_bar.setValue(0)
            self.master_bar.setFormat("0% - Starting...")
        if hasattr(self, 'current_process_bar'):
            self.current_process_bar.setValue(0)
        if hasattr(self, 'current_process_label'):
            self.current_process_label.setText("Initializing...")

        # Collect quality parameters from GUI
        quality_params = {
            "quality_preset": self.quality_preset.currentText(),
            "target_elements": self.target_elements.value(),
            "max_size_mm": self.max_size.value(),
            "curvature_adaptive": self.curvature_adaptive.isChecked(),
            "mesh_strategy": self.mesh_strategy.currentText(),
            "save_stl": self.save_stl.isChecked(),  # Export intermediate STL files
            "ansys_mode": self.ansys_mode.currentText()  # ANSYS export mode: None, CFD, or FEA
        }

        # Add painted regions if any exist
        if self.paintbrush_selector and self.paintbrush_selector.get_painted_regions():
            painted_regions = self.paintbrush_selector.get_painted_regions()
            quality_params["painted_regions"] = [region.to_dict() for region in painted_regions]

            # Log that paintbrush refinement will be applied
            stats = self.paintbrush_selector.get_statistics()
            self.add_log(f"[ART] Paintbrush Refinement: {stats['num_regions']} regions, "
                        f"{stats['num_surfaces']} surfaces, "
                        f"avg {stats['avg_refinement']:.1f}x refinement")

            # Debug: Show what's being serialized
            print(f"\n[DEBUG GUI] Serialized {len(painted_regions)} painted regions:")
            for i, region in enumerate(painted_regions, 1):
                region_dict = region.to_dict()
                print(f"  Region {i}: {len(region_dict['surface_tags'])} surfaces, "
                      f"{region_dict['refinement_level']:.1f}x, "
                      f"center: {region_dict.get('center_point')}")
                print(f"    Surface tags: {region_dict['surface_tags'][:10]}")
            print()

        # Store for chatbox experiments
        self.last_quality_params = quality_params

        self.add_log("=" * 70)
        self.add_log("Starting PARALLEL mesh generation...")
        self.add_log(f"Quality: {quality_params['quality_preset']}, Target: {quality_params['target_elements']:,} elements")
        self.add_log(f"Max element size: {quality_params['max_size_mm']} mm (ANSYS-style, no minimum)")
        self.add_log(f"Strategy: {quality_params['mesh_strategy']}")
        if quality_params['curvature_adaptive']:
            self.add_log(f"Curvature-Adaptive: ON")
        self.add_log(f"Parallel execution will test multiple strategies simultaneously")
        self.add_log("=" * 70)

        self.worker.start(self.cad_file, quality_params)

    def stop_mesh_generation(self):
        """Stop the currently running mesh generation"""
        if self.worker and self.worker.is_running:
            self.add_log("\n[!] Stopping mesh generation...")
            self.worker.stop()
            self.add_log("[OK] Mesh generation stopped by user")

            # Re-enable generate button, disable stop button
            self.generate_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)

            # Stop animation
            if self.animation_timer.isActive():
                self.animation_timer.stop()
            self.active_phase = None

    def refine_mesh_quality(self):
        """Refine existing mesh quality with Netgen optimization and progress feedback"""
        if not self.mesh_file or not Path(self.mesh_file).exists():
            self.add_log("[!] No mesh loaded - generate mesh first")
            return

        self.add_log("\n" + "=" * 70)
        self.add_log("MESH QUALITY REFINEMENT")
        self.add_log("=" * 70)
        self.add_log("Applying Netgen optimization to improve element quality...")
        self.add_log("This may take several minutes for complex meshes.\n")

        # Disable buttons during refinement
        self.refine_btn.setEnabled(False)
        self.generate_btn.setEnabled(False)

        try:
            import gmsh
            gmsh.initialize()
            gmsh.open(self.mesh_file)

            # Enable optimization
            gmsh.option.setNumber("Mesh.Optimize", 1)
            gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
            gmsh.option.setNumber("Mesh.Smoothing", 10)

            # Get initial quality
            try:
                qualities_before = []
                for elem_type in [2, 4]:  # Triangles and tetrahedra
                    elem_tags, elem_quality = gmsh.model.mesh.getElementQualities([], elem_type, "sicn")
                    if elem_quality:
                        qualities_before.extend(elem_quality)

                if qualities_before:
                    avg_quality_before = sum(qualities_before) / len(qualities_before)
                    min_quality_before = min(qualities_before)
                    self.add_log(f"Initial quality: avg={avg_quality_before:.3f}, min={min_quality_before:.3f}")
            except:
                avg_quality_before = None

            # Run optimization passes with progress feedback
            num_passes = 5
            for i in range(num_passes):
                self.add_log(f"\nOptimization pass {i+1}/{num_passes}...")

                try:
                    gmsh.model.mesh.optimize("Netgen")

                    # Check quality after each pass
                    qualities = []
                    for elem_type in [2, 4]:  # Triangles and tetrahedra
                        elem_tags, elem_quality = gmsh.model.mesh.getElementQualities([], elem_type, "sicn")
                        if elem_quality:
                            qualities.extend(elem_quality)

                    if qualities:
                        avg_quality = sum(qualities) / len(qualities)
                        min_quality = min(qualities)
                        self.add_log(f"  Quality: avg={avg_quality:.3f}, min={min_quality:.3f}")

                        # Stop if quality is excellent
                        if avg_quality > 0.85:
                            self.add_log(f"  [OK] Quality excellent, stopping early")
                            break
                    else:
                        self.add_log(f"  [!] Could not measure quality")

                except Exception as e:
                    self.add_log(f"  [!] Optimization pass failed: {e}")
                    break

            # Save refined mesh
            gmsh.write(self.mesh_file)
            gmsh.finalize()

            self.add_log("\n" + "=" * 70)
            self.add_log("[OK] MESH REFINEMENT COMPLETE")
            self.add_log("=" * 70)

            # Reload mesh in viewer
            self.add_log("\nReloading refined mesh in viewer...")
            self.viewer.load_mesh_file(self.mesh_file, {})
            self.add_log("[OK] Refined mesh loaded\n")

        except Exception as e:
            self.add_log(f"\n[!] Refinement failed: {e}")
            import traceback
            self.add_log(traceback.format_exc())

        finally:
            # Re-enable buttons
            self.refine_btn.setEnabled(True)
            self.generate_btn.setEnabled(True)

    def reset_progress_bars(self):
        """Reset all progress bars to 0% and initial blue color"""
        for bar in self.phase_bars.values():
            bar.setValue(0)
            bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #dee2e6;
                    border-radius: 2px;
                    text-align: center;
                    background-color: #f8f9fa;
                    height: 14px;
                    font-size: 8px;
                    font-weight: bold;
                }
                QProgressBar::chunk {
                    background-color: #0d6efd;
                    border-radius: 1px;
                }
            """)

        # Reset phase labels
        for phase_id in self.phase_labels:
            if phase_id in self.phase_base_names:
                self.phase_labels[phase_id].setText(self.phase_base_names[phase_id])

        # Stop animation timer
        if self.animation_timer.isActive():
            self.animation_timer.stop()
        self.active_phase = None
        
        # Reset master progress
        self.master_progress = 0.0
        if hasattr(self, 'master_bar'):
            self.master_bar.setValue(0)
            self.master_bar.setFormat("0% - Ready")
            
        # Reset internal tracking
        self.completed_phases = []
        self.phase_completion_times = {}
        
        # Clear completed stages UI
        if hasattr(self, 'completed_stages_layout'):
            while self.completed_stages_layout.count() > 1: # Keep the stretch item
                item = self.completed_stages_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

    def add_log(self, message: str):
        self.console.append(message)
        self.console.verticalScrollBar().setValue(self.console.verticalScrollBar().maximum())

    def copy_console_to_clipboard(self):
        """Copy all console text to clipboard"""
        clipboard = QApplication.clipboard()
        console_text = self.console.toPlainText()
        clipboard.setText(console_text)
        self.add_log("Console output copied to clipboard!")

    def update_progress(self, phase: str, percentage: int):
        """Update progress bars and track master progress"""
        import time
        
        # Map phase names to standardized names (map all possible phase IDs)
        phase_map = {
            'strategy': 'cad', 'CAD': 'cad', 'cad_preprocessing': 'cad', 'cad': 'cad',
            '1d': 'surf', '2d': 'surf', 'surface_meshing': 'surf', 'surf': 'surf',
            'refine': 'refine', 'refinement': 'refine',
            '3d': '3d', 'meshing_3d': '3d',
            'opt': 'opt', 'netgen': 'opt', 'optimization': 'opt',
            'order2': 'quality', 'quality': 'quality', 'quality_assessment': 'quality',
            # Hexahedral strategy phases
            'hex_generation': '3d', 'hex_conversion': 'opt', 'hex_smoothing': 'opt',
            'component_generation': '3d', 'hex_testing': 'quality',
            'convex_decomposition': 'surf', 'merging_convex_hulls': 'refine',
            'meshing_convex_hull': '3d', 'gluing_convex_hulls': 'opt',
            # Polyhedral strategy phases  
            'poly_generation': '3d', 'dual_graph': '3d', 'poly_construction': 'opt',
            'face_generation': 'opt', 'polyhedral_meshing': '3d',
            # Additional mappings for complete coverage
            'complete': 'quality',  # Final phase
            'error': 'quality'  # Map errors to last phase
        }
        
        normalized_phase = phase_map.get(phase, phase)
        
        # Update current process bar and color
        self.current_process_bar.setValue(percentage)
        
        # Turn current bar green when complete
        if percentage >= 100:
            self.current_process_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #198754;
                    border-radius: 3px;
                    text-align: center;
                    background-color: #d1e7dd;
                    height: 18px;
                    font-size: 9px;
                    font-weight: bold;
                }
                QProgressBar::chunk {
                    background-color: #198754;
                    border-radius: 2px;
                }
            """)
        else:
            # Reset to gray if not complete
            self.current_process_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #6c757d;
                    border-radius: 3px;
                    text-align: center;
                    background-color: #f8f9fa;
                    height: 18px;
                    font-size: 9px;
                    font-weight: bold;
                }
                QProgressBar::chunk {
                    background-color: #6c757d;
                    border-radius: 2px;
                }
            """)
        
        # Update current process label
        phase_display_names = {
            'cad': 'CAD Preprocessing',
            'surf': 'Surface Meshing',
            'refine': 'Refinement',
            '3d': '3D Meshing',
            'opt': 'Optimization',
            'quality': 'Quality Assessment'
        }
        display_name = phase_display_names.get(normalized_phase, phase.upper())
        self.current_process_label.setText(f"{display_name}")
        
        # Calculate master progress
        if self.phase_weights and normalized_phase in self.phase_weights:
            phase_weight = self.phase_weights[normalized_phase]
            phase_contribution = (percentage / 100.0) * phase_weight
            
            # Sum completed phases
            completed_weight = sum(
                self.phase_weights.get(p, 0) for p in self.completed_phases 
                if p != normalized_phase
            )
            
            # Master progress = completed + current phase contribution
            self.master_progress = completed_weight + phase_contribution
            self.master_bar.setValue(int(self.master_progress))
            
            # Update ETA
            self.update_eta(self.master_progress)
        
        # Force master bar to 100% if we receive a 'complete' phase
        if phase == 'complete' or (percentage >= 100 and normalized_phase == 'quality'):
            self.master_progress = 100.0
            self.master_bar.setValue(100)
            self.master_bar.setFormat("100% - Complete!")
        
        # Mark phase complete if 100%
        if percentage >= 100 and normalized_phase not in self.completed_phases:
            self.completed_phases.append(normalized_phase)
            
            # Calculate time taken for this phase
            if self.mesh_start_time:
                phase_time = time.time() - self.mesh_start_time
                # Estimate time for this phase based on weight
                if normalized_phase in self.phase_weights and self.master_progress > 0:
                    phase_weight = self.phase_weights[normalized_phase]
                    phase_duration = (phase_weight / 100.0) * (phase_time / (self.master_progress / 100.0))
                else:
                    phase_duration = 0
                
                self.phase_completion_times[normalized_phase] = phase_duration
                self.add_completed_stage_to_ui(display_name, phase_duration)
        
        # Update old phase bars if they exist (backwards compatibility)
        if phase in self.phase_bars:
            bar = self.phase_bars[phase]
            bar.setValue(percentage)
            
            # Start animation if this is a new active phase
            if self.active_phase != phase:
                self.active_phase = phase
                self.dot_count = 0
                if not self.animation_timer.isActive():
                    self.animation_timer.start()
    
    def add_completed_stage_to_ui(self, stage_name: str, duration: float):
        """Add a completed stage to the UI list"""
        # Format duration
        if duration < 1:
            time_str = "< 1s"
        elif duration < 60:
            time_str = f"{int(duration)}s"
        else:
            mins = int(duration // 60)
            secs = int(duration % 60)
            time_str = f"{mins}m {secs}s"
        
        # Create stage label with checkmark
        stage_label = QLabel(f"✓ {stage_name} ({time_str})")
        stage_label.setStyleSheet("""
            QLabel {
                font-size: 9px;
                color: #198754;
                font-weight: 600;
                padding: 2px;
            }
        """)
        
        # Insert at the top (before the stretch)
        self.completed_stages_layout.insertWidget(0, stage_label)

    def mark_phase_complete(self, phase: str):
        """Turn bar green when complete and reset label"""
        if phase in self.phase_bars:
            bar = self.phase_bars[phase]
            bar.setValue(100)
            # Change to green
            bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #198754;
                    border-radius: 4px;
                    text-align: center;
                    background-color: #d1e7dd;
                    height: 24px;
                    font-size: 11px;
                    font-weight: bold;
                }
                QProgressBar::chunk {
                    background-color: #198754;
                    border-radius: 3px;
                }
            """)

            # Reset label to base name (remove dots)
            if phase in self.phase_labels and phase in self.phase_base_names:
                self.phase_labels[phase].setText(self.phase_base_names[phase])

    def update_animation(self):
        """Update the animated dots for the active phase"""
        if self.active_phase and self.active_phase in self.phase_labels:
            # Cycle through 0, 1, 2, 3 dots
            self.dot_count = (self.dot_count + 1) % 4

            # Create animated dots with vertical offset effect
            dots = ['⠀', '⠄', '⠆', '⠇']  # Braille dots that look like jumping dots
            # Or use simpler dots:
            # dots = ['.  ', '.. ', '...', '   ']
            dots_simple = ['   ', '.  ', '.. ', '...']

            base_name = self.phase_base_names[self.active_phase]
            animated_text = f"{base_name}{dots_simple[self.dot_count]}"

            self.phase_labels[self.active_phase].setText(animated_text)
            
            # Update ETA live
            if self.master_progress > 0:
                self.update_eta(self.master_progress)

    def on_mesh_finished(self, success: bool, result: dict):
        self.add_log("[DEBUG] on_mesh_finished CALLBACK TRIGGERED!")
        self.add_log(f"[DEBUG] success: {success}")
        self.add_log(f"[DEBUG] result keys: {list(result.keys()) if result else 'None'}")

        # Stop animation timer
        if self.animation_timer.isActive():
            self.animation_timer.stop()
        self.active_phase = None

        self.generate_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)  # Disable stop button

        if success:
            self.refine_btn.setEnabled(True)  # Enable refine button after successful mesh
            
            # Show post-processing UI sections
            if hasattr(self, 'crosssection_group'):
                self.crosssection_group.setVisible(True)
            if hasattr(self, 'viz_group'):
                self.viz_group.setVisible(True)
            
            # Force progress bar to 100% and show total time
            import time
            if self.mesh_start_time:
                total_time = time.time() - self.mesh_start_time
                if total_time < 60:
                    time_str = f"{int(total_time)}s"
                else:
                    mins = int(total_time // 60)
                    secs = int(total_time % 60)
                    time_str = f"{mins}m {secs}s"
                
                self.master_bar.setValue(100)
                self.master_bar.setFormat(f"100% - Complete! (Total: {time_str})")
            else:
                self.master_bar.setValue(100)
                self.master_bar.setFormat("100% - Complete!")
            
            # Mark current process as complete
            self.current_process_bar.setValue(100)
            self.current_process_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #198754;
                    border-radius: 3px;
                    text-align: center;
                    background-color: #d1e7dd;
                    height: 18px;
                    font-size: 9px;
                    font-weight: bold;
                }
                QProgressBar::chunk {
                    background-color: #198754;
                    border-radius: 2px;
                }
            """)
            
            self.add_log("=" * 70)
            self.add_log("MESH GENERATION COMPLETE")
            self.add_log("=" * 70)

            self.mesh_file = result.get('output_file')
            metrics = result.get('metrics', {})

            self.add_log(f"[DEBUG] mesh_file: {self.mesh_file}")
            self.add_log(f"[DEBUG] File exists: {Path(self.mesh_file).exists() if self.mesh_file else 'N/A'}")

            if self.mesh_file and Path(self.mesh_file).exists():
                # Check if post-processing conversion is needed
                strategy = self.mesh_strategy.currentText()

                if strategy == "Polyhedral (Tet->Poly)":
                    self.add_log("\n" + "="*70)
                    self.add_log("POST-PROCESSING: Tetrahedral -> Polyhedral")
                    self.add_log("="*70)
                    self.mesh_file = self._convert_to_polyhedral(self.mesh_file)
                    self.add_log("="*70 + "\n")

                elif strategy == "Hexahedral (Tet->Hex)":
                    self.add_log("\n" + "="*70)
                    self.add_log("POST-PROCESSING: Tetrahedral -> Hexahedral (THex)")
                    self.add_log("="*70)
                    self.mesh_file = self._convert_to_hexahedral(self.mesh_file)
                    self.add_log("="*70 + "\n")

                # Check for polyhedral visualization mode (use JSON data)
                if result.get('visualization_mode') == 'polyhedral' and result.get('polyhedral_data_file'):
                    poly_json = result.get('polyhedral_data_file')
                    self.add_log(f"[POLY-VIZ] Loading polyhedral mesh from {poly_json}...")
                    load_result = self.viewer.load_polyhedral_file(poly_json)
                    self.add_log(f"[POLY-VIZ] Result: {load_result}")
                else:
                    # Standard mesh loading
                    self.add_log(f"[DEBUG] Calling viewer.load_mesh_file...")
                    load_result = self.viewer.load_mesh_file(self.mesh_file, result)  # Pass result dict!
                    self.add_log(f"[DEBUG] load_mesh_file returned: {load_result}")
            
            # Check for hex testing component visualization
            elif result.get('visualization_mode') == 'components' and result.get('component_files'):
                self.add_log(f"[DEBUG] Loading component visualization with {result.get('num_components')} parts...")
                load_result = self.viewer.load_component_visualization(result)
                self.add_log(f"[DEBUG] Component visualization loaded")

            # Check for surface visualization (e.g. Polyhedral Dual fallback)
            elif result.get('visualization_mode') == 'surface':
                self.add_log(f"[DEBUG] Loading surface visualization...")
                load_result = self.viewer.load_mesh_file(self.mesh_file, result)
                self.add_log(f"[DEBUG] Surface visualization loaded")


                # Check if colors were applied
                if self.viewer.current_actor and result.get('per_element_quality'):
                    mapper = self.viewer.current_actor.GetMapper()
                    poly_data = mapper.GetInput()
                    scalars = poly_data.GetCellData().GetScalars()
                    self.add_log(f"[DEBUG] Actor has {poly_data.GetNumberOfCells()} cells")
                    self.add_log(f"[DEBUG] Scalars array: {scalars.GetNumberOfTuples() if scalars else 'NONE'} tuples")
                    self.add_log(f"[DEBUG] Mapper ScalarVisibility: {mapper.GetScalarVisibility()}")
                    self.add_log(f"[DEBUG] Mapper ColorMode: {mapper.GetColorMode()}")
                    if scalars:
                        # Sample first few colors
                        sample_colors = []
                        for i in range(min(5, scalars.GetNumberOfTuples())):
                            color = scalars.GetTuple3(i)
                            sample_colors.append(f"({int(color[0])},{int(color[1])},{int(color[2])})")
                        self.add_log(f"[DEBUG] First 5 colors: {', '.join(sample_colors)}")

                self.add_log(f"[DEBUG] Calling viewer.show_quality_report...")
                self.viewer.show_quality_report(metrics)
                
                # Update quality data ranges for the range slider
                self.update_quality_data_ranges(
                    result.get('per_element_quality', {}),
                    result.get('per_element_gamma', {}),
                    result.get('per_element_skewness', {}),
                    result.get('per_element_aspect_ratio', {})
                )
                
                self.add_log(f"[DEBUG] on_mesh_finished complete!")

                # Update chatbox with mesh data, CAD file, and config
                if self.chatbox:
                    mesh_data = {
                        'file_name': Path(self.mesh_file).name,
                        'total_elements': result.get('total_elements', 0),
                        'total_nodes': result.get('total_nodes', 0),
                        **metrics
                    }

                    # Pass CAD file and config for experiments
                    config = getattr(self, 'last_quality_params', {
                        'quality_preset': 'Medium',
                        'target_elements': 10000,
                        'max_size_mm': 100,
                        'curvature_adaptive': False
                    })

                    self.chatbox.update_mesh_data(mesh_data, self.cad_file, config)
                    print("[DEBUG] Chatbox updated with mesh data, CAD file, and config")

        else:
            self.add_log("=" * 70)
            self.add_log("MESH GENERATION FAILED")
            self.add_log(f"Error: {result.get('error')}")
            self.add_log("=" * 70)

    def on_ai_iteration_mesh_ready(self, mesh_path: str, metrics: Dict):
        """Called when an AI iteration completes - auto-display the mesh"""
        logging.info(f"on_ai_iteration_mesh_ready: {mesh_path}")

        if mesh_path and Path(mesh_path).exists():
            # Add to viewer's iteration list
            self.viewer.add_iteration_mesh(mesh_path, metrics)
        if hasattr(self, 'viewer') and self.viewer:
            axis = self.clip_axis_combo.currentText().lower()
            offset = self.clip_offset_slider.value()
            self.viewer.set_clipping(enabled, axis, offset)

            if enabled:
                self.add_log(f"Cross-section enabled - Axis: {axis.upper()}, Offset: {offset}%")
            else:
                self.add_log("Cross-section disabled")

    def on_clip_axis_changed(self, axis_text):
        """Handle clip axis change"""
        if self.crosssection_enabled.isChecked() and hasattr(self, 'viewer') and self.viewer:
            axis = axis_text.lower()
            offset = self.clip_offset_slider.value()
            self.viewer.set_clipping(True, axis, offset)
            self.add_log(f"Cross-section axis changed to: {axis.upper()}")

    def on_clip_offset_changed(self, value):
        """Handle clip offset slider change"""
        self.clip_offset_value_label.setText(f"{value}%")

        if self.crosssection_enabled.isChecked() and hasattr(self, 'viewer') and self.viewer:
            axis = self.clip_axis_combo.currentText().lower()
            self.viewer.set_clipping(True, axis, value)

    # Paintbrush refinement methods
    def on_paintbrush_toggled(self, enabled: bool):
        """Handle paintbrush mode toggle"""
        if hasattr(self.viewer, 'interactor_style'):
            self.viewer.interactor_style.painting_mode = enabled

            if enabled:
                # Create brush cursor with current radius
                if self.paintbrush_widget:
                    radius = self.paintbrush_widget.get_current_radius()
                    self.viewer.create_brush_cursor(radius)
                self.add_log("Paintbrush mode enabled - Left click to paint, Right click to rotate")
            else:
                # Hide brush cursor
                if self.viewer.brush_cursor_actor:
                    self.viewer.brush_cursor_actor.VisibilityOff()
                    self.viewer.brush_cursor_visible = False
                    self.viewer.vtk_widget.GetRenderWindow().Render()
                self.add_log("Paintbrush mode disabled - Normal rotation mode")

    def on_brush_radius_changed(self, radius: float):
        """Handle brush radius change"""
        # Update brush cursor size if paintbrush is enabled
        if hasattr(self.viewer, 'interactor_style') and self.viewer.interactor_style.painting_mode:
            self.viewer.create_brush_cursor(radius)

    def on_refinement_changed(self, level: float):
        """Handle refinement level change"""
        pass  # Refinement level is stored in widget, used when painting

    def on_paint_at_cursor(self, x: int, y: int):
        """Handle painting at cursor position"""
        if not self.paintbrush_selector or not self.paintbrush_widget:
            return

        try:
            # Use cell picker to get actual surface point
            picker = vtk.vtkCellPicker()
            picker.SetTolerance(0.005)
            if not picker.Pick(x, y, 0, self.viewer.renderer):
                # No surface picked, skip painting
                return
            point = picker.GetPickPosition()

            # Get current brush settings
            radius = self.paintbrush_widget.get_current_radius()
            refinement = self.paintbrush_widget.get_current_refinement()

            # Find VTK cells within brush radius for visual feedback
            cell_ids = self.viewer._get_cells_within_radius(point, radius)

            # Find gmsh surfaces near point for mesh refinement
            surfaces = self.paintbrush_selector.get_surfaces_near_point(point, radius)

            if surfaces:
                # Add painted region (for gmsh mesh refinement)
                region = self.paintbrush_selector.add_painted_region(
                    surfaces, radius, refinement, center_point=point
                )

                # Update UI
                self.paintbrush_widget.add_region_to_list(
                    f"Region: {len(surfaces)} surfaces, {refinement:.1f}x"
                )

                # Update statistics
                stats = self.paintbrush_selector.get_statistics()
                self.paintbrush_widget.update_statistics(
                    stats['num_regions'],
                    stats['num_surfaces'],
                    stats['avg_refinement']
                )

                # Color the surface cells to show painted area
                if cell_ids:
                    self.viewer.mark_cells_as_painted(cell_ids, refinement)

                self.add_log(f"Painted {len(surfaces)} surfaces ({len(cell_ids)} cells) with {refinement:.1f}x refinement")

        except Exception as e:
            print(f"Error painting at cursor: {e}")

    def on_clear_painted_regions(self):
        """Clear all painted regions"""
        if self.paintbrush_selector:
            self.paintbrush_selector.clear_all_regions()
            if self.paintbrush_widget:
                self.paintbrush_widget.update_statistics(0, 0, 0.0)
            # Clear visual markers
            self.viewer.clear_paint_markers()
            self.add_log("Cleared all painted regions")

    def on_preview_refinement(self):
        """Preview refinement field (future enhancement)"""
        if self.paintbrush_selector:
            stats = self.paintbrush_selector.get_statistics()
            self.add_log(f"Preview: {stats['num_regions']} regions, "
                        f"{stats['num_surfaces']} surfaces painted")
            self.add_log(f"Avg refinement: {stats['avg_refinement']:.1f}x, "
                        f"Max: {stats['max_refinement']:.1f}x")

    def on_region_deleted(self, index: int):
        """Handle region deletion"""
        if self.paintbrush_selector:
            self.paintbrush_selector.remove_region(index)
            stats = self.paintbrush_selector.get_statistics()
            if self.paintbrush_widget:
                self.paintbrush_widget.update_statistics(
                    stats['num_regions'],
                    stats['num_surfaces'],
                    stats['avg_refinement']
                )
            self.add_log(f"Deleted region {index + 1}")

    def _convert_to_polyhedral(self, tet_mesh_file: str) -> str:
        """Convert tetrahedral mesh to polyhedral mesh"""
        try:
            from converters.poly_hex_converter import TetToPolyConverter

            self.add_log("Loading tetrahedral mesh...")
            converter = TetToPolyConverter()
            converter.load_from_gmsh(tet_mesh_file)

            self.add_log("Converting to polyhedral cells...")
            poly_cells = converter.convert()

            # Export to VTK
            output_file = tet_mesh_file.replace('.msh', '_poly.vtk')
            self.add_log(f"Exporting to {Path(output_file).name}...")
            converter.export_to_vtk(poly_cells, output_file)

            self.add_log(f"[OK] Polyhedral conversion complete!")
            self.add_log(f"  Created {len(poly_cells)} polyhedral cells")

            return output_file

        except Exception as e:
            self.add_log(f"[X] Polyhedral conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return tet_mesh_file  # Return original on failure

    def _convert_to_hexahedral(self, tet_mesh_file: str) -> str:
        """Convert tetrahedral mesh to hexahedral mesh via THex splitting"""
        try:
            from converters.poly_hex_converter import TetToHexConverter

            self.add_log("Loading tetrahedral mesh...")
            converter = TetToHexConverter()
            converter.load_from_gmsh(tet_mesh_file)

            self.add_log("Splitting tets into hexes (THex algorithm)...")
            hexes = converter.convert()

            # Export to Gmsh format
            output_file = tet_mesh_file.replace('.msh', '_hex.msh')
            self.add_log(f"Exporting to {Path(output_file).name}...")
            converter.export_to_gmsh(hexes, output_file)

            self.add_log(f"[OK] Hexahedral conversion complete!")
            self.add_log(f"  Created {len(hexes)} hexahedra (4x original tets)")
            self.add_log(f"  Total nodes: {len(converter.nodes)}")

            return output_file

        except Exception as e:
            self.add_log(f"[X] Hexahedral conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return tet_mesh_file  # Return original on failure

    def toggle_hex_visualization(self, checked: bool):
        """Toggle between tet and hex-like visualization by hiding diagonal edges"""
        if not hasattr(self.viewer, 'current_actor') or not self.viewer.current_actor:
            return

        import vtk
        from collections import defaultdict

        if checked:
            # Debug mode: paint longest edges RED to see which ones are being identified
            self.console.append("🔍 Highlighting longest edges in RED...")

            try:
                # Check if we have mesh data
                if not hasattr(self.viewer, 'current_mesh_elements') or not self.viewer.current_mesh_elements:
                    self.console.append("[X] No mesh data available")
                    return

                import numpy as np

                nodes = self.viewer.current_mesh_nodes
                elements = self.viewer.current_mesh_elements
                node_map = self.viewer.current_node_map
                polydata = self.viewer.current_poly_data

                # Build edge-to-triangles map to find shared edges (quad diagonals)
                edge_to_triangles = {}  # edge -> list of triangle indices
                triangles = []  # Store triangle data
                triangle_count = 0

                for element in elements:
                    if element['type'] == 'triangle':
                        node_ids = element['nodes']
                        pt_indices = [node_map[nid] for nid in node_ids]

                        triangles.append({
                            'pt_indices': pt_indices,
                            'coords': [np.array(nodes[nid]) for nid in node_ids]
                        })

                        # Add all three edges to the map
                        edges = [
                            tuple(sorted([pt_indices[0], pt_indices[1]])),
                            tuple(sorted([pt_indices[1], pt_indices[2]])),
                            tuple(sorted([pt_indices[2], pt_indices[0]]))
                        ]

                        for edge in edges:
                            if edge not in edge_to_triangles:
                                edge_to_triangles[edge] = []
                            edge_to_triangles[edge].append(triangle_count)

                        triangle_count += 1

                # GREEDY TOPOLOGICAL PAIRING ALGORITHM
                # Pair triangles based on connectivity only (no geometry heuristics)
                # Hide edges between paired triangles (diagonals)
                # Draw all other edges

                # Step 1: Build neighbor list per triangle
                tri_neighbors = [[] for _ in range(triangle_count)]

                for edge, tri_list in edge_to_triangles.items():
                    if len(tri_list) == 2:
                        # Internal edge - these triangles are neighbors
                        t1, t2 = tri_list[0], tri_list[1]
                        tri_neighbors[t1].append({'neighbor': t2, 'edge': edge})
                        tri_neighbors[t2].append({'neighbor': t1, 'edge': edge})

                # Step 2: Greedy pairing loop
                visited_tris = set()
                hidden_edges = set()  # Edges to HIDE (diagonals of quads)

                for tri_id in range(triangle_count):
                    if tri_id in visited_tris:
                        continue

                    # Look for an unvisited neighbor to pair with
                    partner_found = False
                    for neighbor_info in tri_neighbors[tri_id]:
                        neighbor_id = neighbor_info['neighbor']
                        shared_edge = neighbor_info['edge']

                        if neighbor_id not in visited_tris:
                            # MATCH! Form a quad
                            visited_tris.add(tri_id)
                            visited_tris.add(neighbor_id)
                            hidden_edges.add(shared_edge)
                            partner_found = True
                            break

                    if not partner_found:
                        visited_tris.add(tri_id)

                # Step 3: Collect edges to draw (all except hidden diagonals)
                edges_to_draw = set()
                for edge in edge_to_triangles.keys():
                    if edge not in hidden_edges:
                        edges_to_draw.add(edge)

                longest_edges = edges_to_draw

                quads_formed = len(hidden_edges)
                orphans = triangle_count - (quads_formed * 2)
                print(f"[DEBUG] Greedy Topological Pairing:")
                print(f"  Triangles: {triangle_count}")
                print(f"  Quad pairs: {quads_formed}")
                print(f"  Orphans: {orphans}")
                print(f"  Hidden diagonals: {len(hidden_edges)}")
                print(f"  Visible edges: {len(edges_to_draw)}")

                # Compute surface normals to offset lines outward
                normals_filter = vtk.vtkPolyDataNormals()
                normals_filter.SetInputData(polydata)
                normals_filter.ComputePointNormalsOn()
                normals_filter.Update()
                normals = normals_filter.GetOutput().GetPointData().GetNormals()

                # Create offset points (shifted outward by 0.5mm along normals)
                offset_points = vtk.vtkPoints()
                offset_points.SetNumberOfPoints(polydata.GetNumberOfPoints())

                for i in range(polydata.GetNumberOfPoints()):
                    pt = polydata.GetPoint(i)
                    normal = normals.GetTuple3(i) if normals else (0, 0, 1)
                    # Shift outward by 0.5mm
                    offset_pt = [pt[j] + normal[j] * 0.01 for j in range(3)]
                    offset_points.SetPoint(i, offset_pt)

                # Create a SINGLE vtkPolyData with ALL the longest edges as lines
                lines = vtk.vtkCellArray()

                for edge in longest_edges:
                    line = vtk.vtkLine()
                    line.GetPointIds().SetId(0, edge[0])
                    line.GetPointIds().SetId(1, edge[1])
                    lines.InsertNextCell(line)

                # Create new polydata for edges (use OFFSET points)
                edge_polydata = vtk.vtkPolyData()
                edge_polydata.SetPoints(offset_points)
                edge_polydata.SetLines(lines)

                # Create ONE mapper and ONE actor for all red lines
                edge_mapper = vtk.vtkPolyDataMapper()
                edge_mapper.SetInputData(edge_polydata)

                if not hasattr(self.viewer, 'hex_edge_actor') or self.viewer.hex_edge_actor is None:
                    self.viewer.hex_edge_actor = vtk.vtkActor()

                self.viewer.hex_edge_actor.SetMapper(edge_mapper)
                self.viewer.hex_edge_actor.GetProperty().SetColor(0, 0, 0)  # BLACK
                self.viewer.hex_edge_actor.GetProperty().SetLineWidth(1.0)  # Normal thickness
                self.viewer.hex_edge_actor.GetProperty().SetRenderLinesAsTubes(False)

                # Polygon offset to render lines in front
                edge_mapper.SetResolveCoincidentTopologyToPolygonOffset()
                edge_mapper.SetRelativeCoincidentTopologyLineOffsetParameters(-1, -1)
                edge_mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-1, -1)

                # Hide original wireframe, show only quad wireframe
                self.viewer.current_actor.GetProperty().EdgeVisibilityOff()

                # Add the new quad wireframe actor
                self.viewer.renderer.AddActor(self.viewer.hex_edge_actor)
                self.viewer.hex_edge_actor.SetUseBounds(False)

                self.console.append(f"[OK] Hex mode: {quads_formed} quads, {orphans} orphans, {len(edges_to_draw)} edges visible")
                print(f"[DEBUG] Hex wireframe: {quads_formed} quads, {orphans} orphans")

            except Exception as e:
                self.console.append(f"[X] Error computing hex visualization: {e}")
                import traceback
                traceback.print_exc()
        else:
            # Normal mode: restore original triangle wireframe
            if hasattr(self.viewer, 'hex_edge_actor') and self.viewer.hex_edge_actor:
                self.viewer.renderer.RemoveActor(self.viewer.hex_edge_actor)
                self.viewer.hex_edge_actor = None

            self.viewer.current_actor.GetProperty().EdgeVisibilityOn()
            self.viewer.current_actor.GetProperty().SetEdgeColor(0.0, 0.0, 0.0)
            self.console.append("🔲 Triangle mode: Original wireframe restored")

        # Re-render
        self.viewer.vtk_widget.GetRenderWindow().Render()

    def toggle_chatbox(self):
        """Toggle AI chatbox visibility and swap with left panel"""
        logging.info("="*60)
        logging.info("toggle_chatbox() called")
        logging.info(f"  chatbox = {self.chatbox}")
        logging.info(f"  chatbox_visible = {getattr(self, 'chatbox_visible', False)}")
        print(f"[DEBUG] toggle_chatbox called, chatbox={self.chatbox}, visible={getattr(self, 'chatbox_visible', False)}")

        if not self.chatbox:
            logging.warning("Chatbox is None - aborting toggle")
            print("[DEBUG] Chatbox is None, returning")
            return

        self.chatbox_visible = not self.chatbox_visible
        logging.info(f"New chatbox_visible state: {self.chatbox_visible}")
        print(f"[DEBUG] New chatbox_visible state: {self.chatbox_visible}")

        # Swap visibility between left panel and chatbox
        if self.chatbox_visible:
            logging.info("Opening chatbox...")
            print("[DEBUG] Opening chatbox...")
            self.left_panel.setVisible(False)
            self.chatbox.setVisible(True)
            self.chat_toggle_btn.setText("✕ Close Chat")
            self.chat_toggle_btn.setChecked(True)
            self.add_log("AI assistant opened - Ask Claude about your mesh!")
            logging.info("Chatbox opened successfully")
        else:
            logging.info("Closing chatbox...")
            print("[DEBUG] Closing chatbox...")
            self.chatbox.setVisible(False)
            self.left_panel.setVisible(True)
            self.chat_toggle_btn.setText("💬 AI Chat")
            self.chat_toggle_btn.setChecked(False)
            self.add_log("AI assistant closed")
            logging.info("Chatbox closed successfully")
        logging.info("="*60)

    def generate_surface_mesh_only(self):
        """Secret button: Generate surface mesh only (2D, no volume)"""
        if not self.cad_file:
            self.add_log("[!] No CAD file loaded")
            return
        
        self.add_log("=" * 70)
        self.add_log("SURFACE MESH ONLY (2D)")
        self.add_log("=" * 70)
        
        try:
            import gmsh
            import multiprocessing
            
            # Initialize
            gmsh.initialize()
            num_cores = multiprocessing.cpu_count()
            gmsh.option.setNumber("General.NumThreads", num_cores)
            gmsh.option.setNumber("General.Terminal", 1)
            
            # Load CAD
            gmsh.model.occ.importShapes(self.cad_file)
            gmsh.model.occ.synchronize()
            
            # Generate 2D mesh only
            self.add_log("Generating 2D surface mesh...")
            gmsh.model.mesh.generate(2)
            
            # Save
            output_file = str(Path(self.cad_file).stem) + "_surface.msh"
            output_path = Path.home() / "Downloads" / "MeshPackageLean" / output_file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            gmsh.write(str(output_path))
            gmsh.finalize()
            
            self.add_log(f"[OK] Surface mesh saved to: {output_file}")
            self.add_log("=" * 70)
            
            # Load it in the viewer
            self.viewer.load_mesh_file(str(output_path))
            
        except Exception as e:
            self.add_log(f"[!] Surface mesh failed: {e}")
            import traceback
            self.add_log(traceback.format_exc())


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Force light mode palette (prevents dark mode on macOS)
    palette = QPalette()
    # Window background
    palette.setColor(QPalette.Window, QColor(240, 240, 240))
    # Base (text input backgrounds)
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    # AlternateBase (alternating rows)
    palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
    # ToolTipBase
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))
    # ToolTipText
    palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
    # Text (general text)
    palette.setColor(QPalette.Text, QColor(0, 0, 0))
    # Button
    palette.setColor(QPalette.Button, QColor(240, 240, 240))
    # ButtonText
    palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))
    # BrightText
    palette.setColor(QPalette.BrightText, QColor(255, 255, 255))
    # Highlight (selections)
    palette.setColor(QPalette.Highlight, QColor(0, 120, 215))
    # HighlightedText
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))

    app.setPalette(palette)

    gui = ModernMeshGenGUI()
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
