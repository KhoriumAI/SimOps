#!/usr/bin/env python
"""
Montage Video Configuration GUI
================================
Visual tool for creating professional multi-mesh showcase videos.

Features:
- Drag-and-drop mesh ordering
- Visual camera path configuration
- Per-mesh orientation control
- Real-time preview of settings
- One-click video generation
"""

import sys
import os
import json
from pathlib import Path

# Fix Qt platform plugin path for macOS (must be done before importing PyQt6)
if sys.platform == 'darwin' and 'QT_PLUGIN_PATH' not in os.environ:
    # Try to find PyQt6 plugins in common conda/pip locations
    possible_paths = [
        Path(sys.executable).parent.parent / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "PyQt6" / "Qt6" / "plugins",
        Path.home() / "anaconda3" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "PyQt6" / "Qt6" / "plugins",
        Path("/opt/anaconda3") / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "PyQt6" / "Qt6" / "plugins",
    ]

    for path in possible_paths:
        if path.exists():
            os.environ['QT_PLUGIN_PATH'] = str(path)
            break
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QListWidget, QLabel, QSpinBox, QDoubleSpinBox,
    QFileDialog, QGroupBox, QSlider, QLineEdit, QComboBox,
    QMessageBox, QListWidgetItem, QAbstractItemView, QDialog, QTextEdit,
    QCheckBox
)
from PyQt6.QtCore import Qt, QTimer, QProcess
from PyQt6.QtGui import QFont, QPalette, QColor
import subprocess
import tempfile


class MeshOrientationDialog(QDialog):
    """Dialog for setting X/Y/Z rotation for a mesh"""

    def __init__(self, parent, mesh_name, x_rot, y_rot, z_rot):
        super().__init__(parent)
        self.setWindowTitle(f"Set Orientation: {mesh_name}")
        self.x_rotation = x_rot
        self.y_rotation = y_rot
        self.z_rotation = z_rot
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # X-axis rotation
        x_label = QLabel("X-axis rotation (pitch):")
        x_label.setToolTip("Rotate around X-axis (e.g., to stand tree upright)")
        self.x_spin = QSpinBox()
        self.x_spin.setRange(-180, 180)
        self.x_spin.setValue(self.x_rotation)
        self.x_spin.setSuffix("°")

        # Y-axis rotation
        y_label = QLabel("Y-axis rotation (yaw - animation axis):")
        y_label.setToolTip("Initial rotation around Y-axis (vertical)")
        self.y_spin = QSpinBox()
        self.y_spin.setRange(0, 360)
        self.y_spin.setValue(self.y_rotation)
        self.y_spin.setSuffix("°")

        # Z-axis rotation
        z_label = QLabel("Z-axis rotation (roll):")
        z_label.setToolTip("Rotate around Z-axis")
        self.z_spin = QSpinBox()
        self.z_spin.setRange(-180, 180)
        self.z_spin.setValue(self.z_rotation)
        self.z_spin.setSuffix("°")

        # OK/Cancel buttons
        button_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)

        layout.addWidget(x_label)
        layout.addWidget(self.x_spin)
        layout.addWidget(y_label)
        layout.addWidget(self.y_spin)
        layout.addWidget(z_label)
        layout.addWidget(self.z_spin)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def accept(self):
        self.x_rotation = self.x_spin.value()
        self.y_rotation = self.y_spin.value()
        self.z_rotation = self.z_spin.value()
        super().accept()

    def get_rotations(self):
        return (self.x_rotation, self.y_rotation, self.z_rotation)


class CameraPathWidget(QWidget):
    """Visual widget for configuring camera path"""

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Start position
        start_group = QGroupBox("Camera Start Position")
        start_layout = QVBoxLayout()

        self.start_elevation = QSpinBox()
        self.start_elevation.setRange(-90, 90)
        self.start_elevation.setValue(-5)
        self.start_elevation.setSuffix("° elevation")

        self.start_azimuth = QSpinBox()
        self.start_azimuth.setRange(-180, 180)
        self.start_azimuth.setValue(5)
        self.start_azimuth.setSuffix("° azimuth")

        start_layout.addWidget(QLabel("Angle variation (-5° to +5°):"))
        start_layout.addWidget(self.start_elevation)
        start_layout.addWidget(QLabel("Angle variation (-5° to +5°):"))
        start_layout.addWidget(self.start_azimuth)

        start_group.setLayout(start_layout)

        # End position
        end_group = QGroupBox("Camera End Position")
        end_layout = QVBoxLayout()

        self.end_elevation = QSpinBox()
        self.end_elevation.setRange(-90, 90)
        self.end_elevation.setValue(5)
        self.end_elevation.setSuffix("° elevation")

        self.end_azimuth = QSpinBox()
        self.end_azimuth.setRange(-180, 180)
        self.end_azimuth.setValue(-5)
        self.end_azimuth.setSuffix("° azimuth")

        end_layout.addWidget(QLabel("Angle variation (-5° to +5°):"))
        end_layout.addWidget(self.end_elevation)
        end_layout.addWidget(QLabel("Angle variation (-5° to +5°):"))
        end_layout.addWidget(self.end_azimuth)

        end_group.setLayout(end_layout)

        # Presets
        preset_label = QLabel("Camera Presets:")
        preset_label.setStyleSheet("font-weight: bold; margin-top: 10px;")

        preset_buttons = QHBoxLayout()

        btn_subtle = QPushButton("Subtle (Default)")
        btn_subtle.clicked.connect(lambda: self.apply_preset(-5, 5, 5, -5))

        btn_no_motion = QPushButton("No Angle Change")
        btn_no_motion.clicked.connect(lambda: self.apply_preset(0, 0, 0, 0))

        btn_moderate = QPushButton("Moderate Variation")
        btn_moderate.clicked.connect(lambda: self.apply_preset(-3, 3, 3, -3))

        preset_buttons.addWidget(btn_subtle)
        preset_buttons.addWidget(btn_no_motion)
        preset_buttons.addWidget(btn_moderate)

        layout.addWidget(start_group)
        layout.addWidget(end_group)
        layout.addWidget(preset_label)
        layout.addLayout(preset_buttons)

        self.setLayout(layout)

    def apply_preset(self, start_elev, start_azim, end_elev, end_azim):
        self.start_elevation.setValue(start_elev)
        self.start_azimuth.setValue(start_azim)
        self.end_elevation.setValue(end_elev)
        self.end_azimuth.setValue(end_azim)

    def get_camera_start(self):
        return (self.start_elevation.value(), self.start_azimuth.value())

    def get_camera_end(self):
        return (self.end_elevation.value(), self.end_azimuth.value())


class MontageConfigGUI(QMainWindow):
    """Main GUI for montage configuration"""

    def __init__(self):
        super().__init__()
        self.mesh_files = []
        self.mesh_x_rotations = []  # X-axis rotations (pitch)
        self.mesh_orientations = []  # Y-axis rotations (yaw - animation axis)
        self.mesh_z_rotations = []  # Z-axis rotations (roll)
        self.render_process = None
        self.temp_script_path = None
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("Mesh Montage Video Creator")
        self.setGeometry(100, 100, 1200, 800)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Left panel: Mesh list
        left_panel = self.create_mesh_panel()
        main_layout.addWidget(left_panel, stretch=2)

        # Right panel: Settings
        right_panel = self.create_settings_panel()
        main_layout.addWidget(right_panel, stretch=1)

    def create_mesh_panel(self):
        panel = QGroupBox("Mesh Sequence")
        layout = QVBoxLayout()

        # Instructions
        instructions = QLabel(
            "Add meshes in the order they should appear.\n"
            "Drag to reorder. Double-click to set X/Y/Z rotation."
        )
        instructions.setStyleSheet("color: #666; padding: 5px; background: #f0f0f0; border-radius: 3px;")

        # Mesh list
        self.mesh_list = QListWidget()
        self.mesh_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.mesh_list.itemDoubleClicked.connect(self.edit_mesh_orientation)

        # Buttons
        button_layout = QHBoxLayout()

        btn_add = QPushButton("Add Meshes")
        btn_add.clicked.connect(self.add_meshes)

        btn_remove = QPushButton("Remove Selected")
        btn_remove.clicked.connect(self.remove_selected_mesh)

        btn_clear = QPushButton("Clear All")
        btn_clear.clicked.connect(self.clear_meshes)

        button_layout.addWidget(btn_add)
        button_layout.addWidget(btn_remove)
        button_layout.addWidget(btn_clear)

        layout.addWidget(instructions)
        layout.addWidget(self.mesh_list)
        layout.addLayout(button_layout)

        panel.setLayout(layout)
        return panel

    def create_settings_panel(self):
        panel = QGroupBox("Video Settings")
        layout = QVBoxLayout()

        # Rotation settings
        rotation_group = QGroupBox("Rotation")
        rotation_layout = QVBoxLayout()

        self.rotation_degrees = QSpinBox()
        self.rotation_degrees.setRange(30, 360)
        self.rotation_degrees.setValue(60)
        self.rotation_degrees.setSuffix("° per mesh")
        self.rotation_degrees.valueChanged.connect(self.update_status)

        self.seconds_per_rev = QSpinBox()
        self.seconds_per_rev.setRange(5, 60)
        self.seconds_per_rev.setValue(30)
        self.seconds_per_rev.setSuffix(" sec/360°")
        self.seconds_per_rev.valueChanged.connect(self.update_status)

        rotation_layout.addWidget(QLabel("Rotation per mesh:"))
        rotation_layout.addWidget(self.rotation_degrees)
        rotation_layout.addWidget(QLabel("Rotation speed:"))
        rotation_layout.addWidget(self.seconds_per_rev)

        # Speed ramping controls
        self.enable_ramping = QComboBox()
        self.enable_ramping.addItems(["Constant Speed", "Speed Ramping (Accelerate)"])
        self.enable_ramping.currentTextChanged.connect(self.on_ramping_mode_changed)

        self.ramping_start = QSpinBox()
        self.ramping_start.setRange(5, 360)
        self.ramping_start.setValue(60)
        self.ramping_start.setSuffix("° start")
        self.ramping_start.valueChanged.connect(self.update_status)

        self.ramping_end = QSpinBox()
        self.ramping_end.setRange(5, 360)
        self.ramping_end.setValue(5)
        self.ramping_end.setSuffix("° end")
        self.ramping_end.valueChanged.connect(self.update_status)

        self.ramping_pattern = QComboBox()
        self.ramping_pattern.addItems(["Linear", "Exponential"])
        self.ramping_pattern.currentTextChanged.connect(self.update_status)

        rotation_layout.addWidget(QLabel("Speed mode:"))
        rotation_layout.addWidget(self.enable_ramping)

        # Container for ramping settings (hidden by default)
        self.ramping_container = QWidget()
        ramping_container_layout = QVBoxLayout()
        ramping_container_layout.setContentsMargins(0, 0, 0, 0)
        ramping_container_layout.addWidget(QLabel("Starting rotation:"))
        ramping_container_layout.addWidget(self.ramping_start)
        ramping_container_layout.addWidget(QLabel("Ending rotation:"))
        ramping_container_layout.addWidget(self.ramping_end)
        ramping_container_layout.addWidget(QLabel("Ramping pattern:"))
        ramping_container_layout.addWidget(self.ramping_pattern)
        self.ramping_container.setLayout(ramping_container_layout)
        self.ramping_container.setVisible(False)

        rotation_layout.addWidget(self.ramping_container)

        rotation_group.setLayout(rotation_layout)

        # Cross Section settings
        cross_section_group = QGroupBox("Cross Section")
        cross_section_layout = QVBoxLayout()

        self.enable_clipping = QCheckBox("Enable Cross Section")
        self.enable_clipping.toggled.connect(self.on_clipping_toggled)

        self.clip_axis = QComboBox()
        self.clip_axis.addItems(["X", "Y", "Z"])
        self.clip_axis.setEnabled(False)

        self.clip_offset = QSpinBox()
        self.clip_offset.setRange(-50, 50)
        self.clip_offset.setValue(0)
        self.clip_offset.setSuffix("% offset")
        self.clip_offset.setEnabled(False)

        cross_section_layout.addWidget(self.enable_clipping)
        cross_section_layout.addWidget(QLabel("Cut Axis:"))
        cross_section_layout.addWidget(self.clip_axis)
        cross_section_layout.addWidget(QLabel("Cut Position:"))
        cross_section_layout.addWidget(self.clip_offset)

        cross_section_group.setLayout(cross_section_layout)

        # Camera path
        self.camera_widget = CameraPathWidget()

        # Output settings
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout()

        self.output_name = QLineEdit("montage_showcase.mp4")

        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["high", "medium", "web"])

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(24, 60)
        self.fps_spin.setValue(30)
        self.fps_spin.setSuffix(" fps")

        # Background mode selection
        self.background_mode = QComboBox()
        self.background_mode.addItems(["Cinematic (Gradient)", "Green Screen", "Blue Screen", "Black"])
        self.background_mode.setToolTip("Choose background for easy keying in your video editor")

        output_layout.addWidget(QLabel("Output filename:"))
        output_layout.addWidget(self.output_name)
        output_layout.addWidget(QLabel("Video quality:"))
        output_layout.addWidget(self.quality_combo)
        output_layout.addWidget(QLabel("Frame rate:"))
        output_layout.addWidget(self.fps_spin)
        output_layout.addWidget(QLabel("Background:"))
        output_layout.addWidget(self.background_mode)

        output_group.setLayout(output_layout)

        # Generate button
        self.btn_generate = QPushButton("Generate Montage Video")
        self.btn_generate.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.btn_generate.clicked.connect(self.generate_video)

        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("""
            padding: 8px;
            background: #e3f2fd;
            border-radius: 3px;
            color: #1a1a1a;
            font-size: 12px;
        """)
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumHeight(60)

        # Layout assembly
        layout.addWidget(rotation_group)
        layout.addWidget(cross_section_group)
        layout.addWidget(self.camera_widget)
        layout.addWidget(output_group)
        layout.addStretch()
        layout.addWidget(self.btn_generate)
        layout.addWidget(self.status_label)

        panel.setLayout(layout)
        return panel

    def add_meshes(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Mesh Files",
            str(Path.home() / "Downloads" / "MeshTest" / "generated_meshes"),
            "Mesh Files (*.msh);;All Files (*)"
        )

        for file in files:
            self.mesh_files.append(file)
            self.mesh_x_rotations.append(0)  # Default X rotation
            self.mesh_orientations.append(0)  # Default Y rotation
            self.mesh_z_rotations.append(0)  # Default Z rotation

            item = QListWidgetItem(f"{Path(file).name} [X:0° Y:0° Z:0°]")
            item.setData(Qt.ItemDataRole.UserRole, file)
            self.mesh_list.addItem(item)

        self.update_status()

    def remove_selected_mesh(self):
        current_row = self.mesh_list.currentRow()
        if current_row >= 0:
            self.mesh_list.takeItem(current_row)
            self.mesh_files.pop(current_row)
            self.mesh_x_rotations.pop(current_row)
            self.mesh_orientations.pop(current_row)
            self.mesh_z_rotations.pop(current_row)
            self.update_status()

    def clear_meshes(self):
        self.mesh_list.clear()
        self.mesh_files.clear()
        self.mesh_x_rotations.clear()
        self.mesh_orientations.clear()
        self.mesh_z_rotations.clear()
        self.update_status()

    def edit_mesh_orientation(self, item):
        row = self.mesh_list.row(item)
        mesh_name = Path(self.mesh_files[row]).name

        # Create dialog with current rotations
        dialog = MeshOrientationDialog(
            self,
            mesh_name,
            self.mesh_x_rotations[row],
            self.mesh_orientations[row],
            self.mesh_z_rotations[row]
        )

        # Show as modal dialog and wait for result
        if dialog.exec() == QDialog.DialogCode.Accepted:
            x_rot, y_rot, z_rot = dialog.get_rotations()
            self.mesh_x_rotations[row] = x_rot
            self.mesh_orientations[row] = y_rot
            self.mesh_z_rotations[row] = z_rot
            item.setText(f"{mesh_name} [X:{x_rot}° Y:{y_rot}° Z:{z_rot}°]")

    def on_ramping_mode_changed(self, text):
        """Toggle ramping settings visibility"""
        is_ramping = "Ramping" in text
        self.ramping_container.setVisible(is_ramping)
        self.rotation_degrees.setEnabled(not is_ramping)
        self.update_status()

    def on_clipping_toggled(self, checked):
        """Enable/disable clipping controls"""
        self.clip_axis.setEnabled(checked)
        self.clip_offset.setEnabled(checked)

    def calculate_rotation_degrees(self):
        """Calculate rotation degrees for each mesh based on ramping settings"""
        num_meshes = len(self.mesh_files)

        if num_meshes == 0:
            return []

        # Check if ramping is enabled
        if "Ramping" not in self.enable_ramping.currentText():
            # Constant speed - all meshes use same rotation
            return [self.rotation_degrees.value()] * num_meshes

        # Speed ramping mode
        start_degrees = self.ramping_start.value()
        end_degrees = self.ramping_end.value()
        pattern = self.ramping_pattern.currentText()

        if num_meshes == 1:
            return [start_degrees]

        degrees_list = []

        if pattern == "Linear":
            # Linear interpolation
            for i in range(num_meshes):
                t = i / (num_meshes - 1)  # 0 to 1
                degrees = start_degrees + (end_degrees - start_degrees) * t
                degrees_list.append(int(degrees))
        else:  # Exponential
            # Exponential decay for smooth acceleration
            import math
            for i in range(num_meshes):
                t = i / (num_meshes - 1)  # 0 to 1
                # Exponential curve: more gradual at start, sharper at end
                exp_t = (math.exp(t * 2) - 1) / (math.exp(2) - 1)
                degrees = start_degrees + (end_degrees - start_degrees) * exp_t
                degrees_list.append(int(degrees))

        return degrees_list

    def update_status(self):
        if len(self.mesh_files) == 0:
            self.status_label.setText("Add meshes to get started")
            self.btn_generate.setEnabled(False)
        else:
            rotation_list = self.calculate_rotation_degrees()
            total_rotation = sum(rotation_list)
            duration = (total_rotation / 360) * self.seconds_per_rev.value()

            status_text = f"Ready: {len(self.mesh_files)} meshes\n"

            # Show rotation info based on mode
            if "Ramping" in self.enable_ramping.currentText():
                status_text += f"Speed ramping: {rotation_list[0]}° -> {rotation_list[-1]}°\n"
                status_text += f"Pattern: {rotation_list}\n"
            else:
                status_text += f"Rotation per mesh: {rotation_list[0]}°\n"

            status_text += f"Total rotation: {total_rotation}°\n"
            status_text += f"Video duration: {duration:.1f} seconds"

            self.status_label.setText(status_text)
            self.btn_generate.setEnabled(True)

    def generate_video(self):
        if len(self.mesh_files) < 2:
            QMessageBox.warning(self, "Error", "Add at least 2 meshes to create a montage")
            return

        # Sync mesh list order with files (in case user reordered)
        ordered_files = []
        ordered_x_rotations = []
        ordered_orientations = []
        ordered_z_rotations = []
        for i in range(self.mesh_list.count()):
            item = self.mesh_list.item(i)
            file = item.data(Qt.ItemDataRole.UserRole)
            idx = self.mesh_files.index(file)
            ordered_files.append(file)
            ordered_x_rotations.append(self.mesh_x_rotations[idx])
            ordered_orientations.append(self.mesh_orientations[idx])
            ordered_z_rotations.append(self.mesh_z_rotations[idx])

        self.mesh_files = ordered_files
        self.mesh_x_rotations = ordered_x_rotations
        self.mesh_orientations = ordered_orientations
        self.mesh_z_rotations = ordered_z_rotations

        # Calculate rotation degrees (may be list for ramping)
        rotation_degrees = self.calculate_rotation_degrees()

        # Build config
        config = {
            'mesh_files': self.mesh_files,
            'mesh_x_rotations': self.mesh_x_rotations,
            'mesh_orientations': self.mesh_orientations,
            'mesh_z_rotations': self.mesh_z_rotations,
            'camera_start': self.camera_widget.get_camera_start(),
            'camera_end': self.camera_widget.get_camera_end(),
            'rotation_degrees': rotation_degrees,
            'output_file': self.output_name.text(),
            'fps': self.fps_spin.value(),
            'seconds_per_revolution': self.seconds_per_rev.value(),
            'video_quality': self.quality_combo.currentText(),
            'resolution': [1920, 1080],
            'background_mode': self.background_mode.currentText(),
            'clipping_enabled': self.enable_clipping.isChecked(),
            'clipping_axis': self.clip_axis.currentText().lower(),
            'clipping_offset': self.clip_offset.value()
        }

        # Create a temporary Python script to run the rendering
        # This avoids VTK threading issues on macOS
        # Use sys.path to ensure mesh_showcase can be imported
        script_dir = str(Path(__file__).parent.absolute())
        script_content = f'''#!/usr/bin/env python3
import sys
import json

# Add script directory to path to find mesh_showcase
sys.path.insert(0, {repr(script_dir)})

from mesh_showcase import create_montage_video

config = {json.dumps(config, indent=2)}

print(f"Starting montage with {{len(config['mesh_files'])}} meshes...", flush=True)

result = create_montage_video(
    mesh_files=config['mesh_files'],
    output_file=config['output_file'],
    rotation_degrees=config['rotation_degrees'],
    fps=config['fps'],
    video_quality=config['video_quality'],
    seconds_per_revolution=config['seconds_per_revolution'],
    resolution=tuple(config['resolution']),
    mesh_x_rotations=config['mesh_x_rotations'],
    mesh_orientations=config['mesh_orientations'],
    mesh_z_rotations=config['mesh_z_rotations'],
    camera_start_pos=tuple(config['camera_start']),
    camera_end_pos=tuple(config['camera_end']),
    background_mode=config['background_mode'],
    clipping_enabled=config.get('clipping_enabled', False),
    clipping_axis=config.get('clipping_axis', 'x'),
    clipping_offset=config.get('clipping_offset', 0.0)
)

if result:
    print(f"SUCCESS: {{result}}", flush=True)
else:
    print("ERROR: Video creation failed", flush=True)
'''

        # Write script to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script_content)
            self.temp_script_path = f.name

        # Disable UI
        self.btn_generate.setEnabled(False)
        self.status_label.setText("Initializing render...")

        # Create QProcess for async execution with progress
        self.render_process = QProcess(self)
        self.render_process.setWorkingDirectory(script_dir)

        # Connect signals
        self.render_process.readyReadStandardOutput.connect(self.on_process_output)
        self.render_process.readyReadStandardError.connect(self.on_process_error)
        self.render_process.finished.connect(self.on_process_finished)

        # Start the process
        self.render_process.start('python3', [self.temp_script_path])

    def on_process_output(self):
        """Handle stdout from render process"""
        if self.render_process:
            output = bytes(self.render_process.readAllStandardOutput()).decode('utf-8')

            # Update status with latest output
            for line in output.strip().split('\n'):
                if line:
                    # Check for progress indicators
                    if 'Progress:' in line or '/' in line:
                        self.status_label.setText(f"Rendering: {line.strip()}")
                    elif 'Processing:' in line:
                        self.status_label.setText(line.strip())
                    elif 'SUCCESS:' in line:
                        pass  # Handle in finished
                    elif 'ERROR:' in line:
                        pass  # Handle in finished
                    else:
                        self.status_label.setText(line.strip())

    def on_process_error(self):
        """Handle stderr from render process"""
        if self.render_process:
            error = bytes(self.render_process.readAllStandardError()).decode('utf-8')
            if error.strip():
                print(f"Process error: {error}")

    def on_process_finished(self, exit_code, exit_status):
        """Handle process completion"""
        # Read any remaining output
        if self.render_process:
            output = bytes(self.render_process.readAllStandardOutput()).decode('utf-8')
            error = bytes(self.render_process.readAllStandardError()).decode('utf-8')

            # Clean up temp script
            if self.temp_script_path and Path(self.temp_script_path).exists():
                Path(self.temp_script_path).unlink()

            # Check for success
            if exit_code == 0 and 'SUCCESS:' in output:
                # Extract output file path
                for line in output.split('\n'):
                    if 'SUCCESS:' in line:
                        output_file = line.split('SUCCESS:')[1].strip()
                        self.on_render_complete(True, output_file)
                        return

            # Failed
            error_msg = error if error.strip() else output
            self.on_render_complete(False, error_msg)

    def on_render_complete(self, success, message):
        """Called when rendering completes"""
        self.btn_generate.setEnabled(True)
        self.render_process = None

        if success:
            self.status_label.setText(f"Complete! Video saved: {message}")
            QMessageBox.information(
                self,
                "Success",
                f"Montage video created successfully!\n\n{message}"
            )

            # Ask if user wants to open the video
            reply = QMessageBox.question(
                self,
                "Open Video?",
                "Would you like to open the video now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                subprocess.run(['open', message])

        else:
            self.status_label.setText(f"Error: {message}")
            QMessageBox.critical(self, "Error", f"Failed to create video:\n{message}")


def main():
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle('Fusion')

    # Create and show GUI
    gui = MontageConfigGUI()
    gui.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
