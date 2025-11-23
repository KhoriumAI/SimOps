#!/usr/bin/env python3
"""
Improved Khorium MeshGen GUI
=============================

Enhanced features:
- Toggleable XYZ axes with fixed size
- Detailed progress tracking for each mesh phase
- Better info labels without truncation
- Multi-phase progress bars
- Optimization step tracking

Dependencies:
    pip install PyQt5 vtk numpy pyvista
"""

import sys
import os
import json
import subprocess
import threading
import tempfile
import re
from pathlib import Path
from typing import Optional
from queue import Queue

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QProgressBar, QGroupBox,
    QSplitter, QFileDialog, QFrame, QScrollArea, QGridLayout,
    QCheckBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QDragEnterEvent, QDropEvent, QPalette, QColor

import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

# Add core modules to path
sys.path.insert(0, str(Path(__file__).parent))


class WorkerSignals(QObject):
    """Signals for mesh worker thread communication"""
    log = pyqtSignal(str)
    progress = pyqtSignal(str, int, str)  # (phase, percentage, status)
    finished = pyqtSignal(bool, dict)  # (success, result)


class MeshWorker:
    """Subprocess-based mesh generation worker with detailed progress tracking"""

    def __init__(self):
        self.signals = WorkerSignals()
        self.thread = None
        self.process = None
        self.is_running = False

    def start(self, cad_file: str):
        """Start mesh generation in background thread"""
        if self.is_running:
            return

        self.is_running = True
        self.thread = threading.Thread(target=self._run, args=(cad_file,), daemon=True)
        self.thread.start()

    def _run(self, cad_file: str):
        """Run mesh generation subprocess with detailed progress parsing"""
        try:
            self.signals.log.emit(f"Loading: {Path(cad_file).name}")
            self.signals.progress.emit("init", 10, "Initializing...")

            # Launch subprocess
            worker_script = Path(__file__).parent / "mesh_worker_subprocess.py"
            self.signals.log.emit("Starting mesh generation...")

            self.process = subprocess.Popen(
                [sys.executable, str(worker_script), cad_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )

            current_phase = "init"
            strategy_count = 0

            # Read output line by line
            for line in self.process.stdout:
                line = line.strip()
                if line:
                    # Parse gmsh Info lines for progress
                    if "Info" in line:
                        # Meshing 1D
                        if "Meshing 1D" in line:
                            self.signals.progress.emit("1d", 20, "Meshing curves (1D)...")
                            current_phase = "1d"
                        elif "Meshing curve" in line:
                            match = re.search(r'\[\s*(\d+)%\]', line)
                            if match:
                                pct = int(match.group(1))
                                self.signals.progress.emit("1d", 20 + pct // 10, f"Meshing curves ({pct}%)...")
                        elif "Done meshing 1D" in line:
                            self.signals.progress.emit("1d", 30, "1D meshing complete")

                        # Meshing 2D
                        elif "Meshing 2D" in line:
                            self.signals.progress.emit("2d", 30, "Meshing surfaces (2D)...")
                            current_phase = "2d"
                        elif "Meshing surface" in line:
                            match = re.search(r'\[\s*(\d+)%\]', line)
                            if match:
                                pct = int(match.group(1))
                                self.signals.progress.emit("2d", 30 + pct // 5, f"Meshing surfaces ({pct}%)...")
                        elif "Done meshing 2D" in line:
                            self.signals.progress.emit("2d", 50, "2D meshing complete")

                        # Meshing 3D
                        elif "Meshing 3D" in line:
                            self.signals.progress.emit("3d", 50, "Generating volume mesh (3D)...")
                            current_phase = "3d"
                        elif "Tetrahedrizing" in line:
                            self.signals.progress.emit("3d", 55, "Creating tetrahedra...")
                        elif "Reconstructing mesh" in line:
                            self.signals.progress.emit("3d", 60, "Reconstructing mesh...")
                        elif "3D refinement" in line:
                            self.signals.progress.emit("3d", 65, "Refining tetrahedra...")
                        elif "Done meshing 3D" in line:
                            self.signals.progress.emit("3d", 70, "3D meshing complete")

                        # Optimization
                        elif "Optimizing mesh" in line and "Netgen" not in line:
                            self.signals.progress.emit("opt", 70, "Optimizing (Gmsh)...")
                            current_phase = "opt"
                        elif "edge swaps" in line or "node relocations" in line:
                            self.signals.progress.emit("opt", 75, "Edge swaps & node relocation...")
                        elif "Optimizing mesh (Netgen)" in line:
                            self.signals.progress.emit("netgen", 75, "Optimizing (Netgen)...")
                            current_phase = "netgen"
                        elif "Remove Illegal Elements" in line:
                            self.signals.progress.emit("netgen", 76, "Removing illegal elements...")
                        elif "SplitImprove" in line:
                            self.signals.progress.emit("netgen", 78, "Split improvement...")
                        elif "SwapImprove" in line:
                            self.signals.progress.emit("netgen", 80, "Swap improvement...")
                        elif "CombineImprove" in line:
                            self.signals.progress.emit("netgen", 82, "Combine improvement...")
                        elif "ImproveMesh" in line:
                            self.signals.progress.emit("netgen", 84, "General improvement...")
                        elif "Done optimizing mesh" in line:
                            if current_phase == "netgen":
                                self.signals.progress.emit("netgen", 87, "Netgen optimization complete")
                            else:
                                self.signals.progress.emit("opt", 75, "Gmsh optimization complete")

                        # Higher order meshing
                        elif "Meshing order 2" in line:
                            self.signals.progress.emit("order2", 87, "Creating higher-order mesh...")
                            current_phase = "order2"
                        elif "Meshing curve" in line and "order 2" in line:
                            self.signals.progress.emit("order2", 88, "Higher-order curves...")
                        elif "Meshing surface" in line and "order 2" in line:
                            self.signals.progress.emit("order2", 90, "Higher-order surfaces...")
                        elif "Meshing volume" in line and "order 2" in line:
                            self.signals.progress.emit("order2", 92, "Higher-order volume...")
                        elif "Done meshing order 2" in line:
                            self.signals.progress.emit("order2", 95, "Higher-order complete")

                    # Strategy attempts
                    if "ATTEMPT" in line:
                        strategy_count += 1
                        match = re.search(r'ATTEMPT\s+(\d+)/(\d+):\s+(\w+)', line)
                        if match:
                            current = match.group(1)
                            total = match.group(2)
                            strategy = match.group(3)
                            self.signals.progress.emit("strategy", 15, f"Strategy {current}/{total}: {strategy}")

                    # Quality analysis
                    if "Analyzing mesh quality" in line:
                        self.signals.progress.emit("quality", 95, "Analyzing quality...")

                    # Check if it's the final JSON result
                    if line.startswith('{') and '"success"' in line:
                        try:
                            result = json.loads(line)
                            if result.get('success'):
                                self.signals.log.emit("Mesh generation completed!")
                                self.signals.progress.emit("complete", 100, "Complete!")
                                self.signals.finished.emit(True, result)
                            else:
                                error = result.get('error', 'Unknown error')
                                self.signals.log.emit(f"Failed: {error}")
                                self.signals.progress.emit("error", 0, "Failed")
                                self.signals.finished.emit(False, result)
                        except json.JSONDecodeError:
                            pass
                    else:
                        # Regular log line
                        self.signals.log.emit(line)

            # Wait for process
            self.process.wait()

        except Exception as e:
            self.signals.log.emit(f"Exception: {str(e)}")
            self.signals.progress.emit("error", 0, "Error")
            self.signals.finished.emit(False, {"error": str(e)})
        finally:
            self.is_running = False

    def stop(self):
        """Stop the running process"""
        if self.process:
            self.process.terminate()
            self.is_running = False


class VTK3DViewer(QFrame):
    """Modern 3D viewer with toggleable axes and better info display"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Box | QFrame.Sunken)
        self.setLineWidth(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # VTK widget
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtk_widget)

        # Renderer
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.95, 0.95, 0.97)  # Light gray background
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)

        # Interactor style
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.vtk_widget.GetRenderWindow().GetInteractor().SetInteractorStyle(style)

        # Orientation marker (axes with constant screen size)
        self.axes_widget = vtk.vtkOrientationMarkerWidget()
        self.axes_actor = vtk.vtkAxesActor()
        self.axes_actor.SetShaftTypeToCylinder()
        self.axes_actor.SetXAxisLabelText("X")
        self.axes_actor.SetYAxisLabelText("Y")
        self.axes_actor.SetZAxisLabelText("Z")
        self.axes_actor.SetTotalLength(1.0, 1.0, 1.0)
        self.axes_actor.SetCylinderRadius(0.02)
        self.axes_actor.SetConeRadius(0.05)
        self.axes_actor.SetSphereRadius(0.02)

        # Make labels smaller
        self.axes_actor.GetXAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        self.axes_actor.GetYAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        self.axes_actor.GetZAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        self.axes_actor.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(12)
        self.axes_actor.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(12)
        self.axes_actor.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(12)

        self.axes_widget.SetOrientationMarker(self.axes_actor)
        self.axes_widget.SetInteractor(self.vtk_widget.GetRenderWindow().GetInteractor())
        self.axes_widget.SetViewport(0, 0, 0.2, 0.2)  # Bottom-left corner
        self.axes_widget.SetEnabled(1)
        self.axes_widget.InteractiveOff()

        self.axes_visible = True

        # Current actor
        self.current_actor = None

        # Start interactor
        self.vtk_widget.Initialize()
        self.vtk_widget.Start()

        # Info overlay (larger to prevent truncation)
        self.info_label = QLabel("No file loaded", self)
        self.info_label.setStyleSheet("""
            QLabel {
                background-color: rgba(255, 255, 255, 220);
                padding: 10px 15px;
                border-radius: 6px;
                font-size: 12px;
                color: #333;
                font-weight: 500;
            }
        """)
        self.info_label.setMinimumWidth(300)
        self.info_label.setWordWrap(True)
        self.info_label.move(10, 10)
        self.info_label.adjustSize()

    def toggle_axes(self, visible: bool):
        """Toggle axes visibility"""
        self.axes_visible = visible
        self.axes_widget.SetEnabled(1 if visible else 0)
        self.vtk_widget.GetRenderWindow().Render()

    def clear_view(self):
        """Clear the current view"""
        if self.current_actor:
            self.renderer.RemoveActor(self.current_actor)
            self.current_actor = None
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

    def load_step_file(self, filepath: str):
        """Load and display STEP file"""
        self.clear_view()
        self.info_label.setText(f"Loading: {Path(filepath).name}")
        self.info_label.adjustSize()

        try:
            import pyvista as pv

            # Convert STEP to STL using gmsh in subprocess
            with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp:
                tmp_stl = tmp.name

            gmsh_script = f"""
import gmsh
import sys

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)

try:
    gmsh.open("{filepath}")
    gmsh.model.mesh.generate(2)
    gmsh.write("{tmp_stl}")
    gmsh.finalize()
    print("SUCCESS")
except Exception as e:
    gmsh.finalize()
    print(f"ERROR: {{e}}")
    sys.exit(1)
"""

            result = subprocess.run(
                [sys.executable, "-c", gmsh_script],
                capture_output=True,
                text=True,
                timeout=30
            )

            if "SUCCESS" not in result.stdout:
                raise Exception(f"Failed to convert STEP file")

            # Load STL with pyvista
            mesh = pv.read(tmp_stl)

            try:
                os.unlink(tmp_stl)
            except:
                pass

            # Convert to VTK polydata
            poly_data = mesh.cast_to_unstructured_grid().extract_surface()

            # Create mapper and actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(poly_data)

            self.current_actor = vtk.vtkActor()
            self.current_actor.SetMapper(mapper)
            self.current_actor.GetProperty().SetColor(0.3, 0.5, 0.8)
            self.current_actor.GetProperty().EdgeVisibilityOn()
            self.current_actor.GetProperty().SetEdgeColor(0.2, 0.2, 0.2)
            self.current_actor.GetProperty().SetLineWidth(1)

            self.renderer.AddActor(self.current_actor)
            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()

            # Update info label with details
            num_points = poly_data.GetNumberOfPoints()
            num_cells = poly_data.GetNumberOfCells()
            self.info_label.setText(f"CAD: {Path(filepath).name}\n{num_points:,} points, {num_cells:,} faces")
            self.info_label.adjustSize()

        except Exception as e:
            self.show_placeholder(f"CAD: {Path(filepath).name}\n\nClick 'Generate Mesh' to create 3D mesh")

    def load_mesh_file(self, filepath: str):
        """Load and display mesh file"""
        self.clear_view()
        self.info_label.setText(f"Loading mesh...")
        self.info_label.adjustSize()

        try:
            # Parse Gmsh .msh file
            nodes, elements = self._parse_msh_file(filepath)

            # Create VTK data structures
            points = vtk.vtkPoints()
            cells = vtk.vtkCellArray()

            # Add nodes
            node_map = {}
            for idx, (node_id, coords) in enumerate(nodes.items()):
                points.InsertNextPoint(coords)
                node_map[node_id] = idx

            # Add elements
            tet_count = 0
            tri_count = 0

            for element in elements:
                if element['type'] == 'tetrahedron':
                    tet = vtk.vtkTetra()
                    for i, node_id in enumerate(element['nodes']):
                        tet.GetPointIds().SetId(i, node_map[node_id])
                    cells.InsertNextCell(tet)
                    tet_count += 1
                elif element['type'] == 'triangle':
                    tri = vtk.vtkTriangle()
                    for i, node_id in enumerate(element['nodes']):
                        tri.GetPointIds().SetId(i, node_map[node_id])
                    cells.InsertNextCell(tri)
                    tri_count += 1

            # Create unstructured grid
            mesh = vtk.vtkUnstructuredGrid()
            mesh.SetPoints(points)
            mesh.SetCells(vtk.VTK_TETRA if tet_count > 0 else vtk.VTK_TRIANGLE, cells)

            # Extract surface
            surface_filter = vtk.vtkDataSetSurfaceFilter()
            surface_filter.SetInputData(mesh)
            surface_filter.Update()

            # Create mapper and actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(surface_filter.GetOutputPort())

            self.current_actor = vtk.vtkActor()
            self.current_actor.SetMapper(mapper)
            self.current_actor.GetProperty().SetColor(0.2, 0.7, 0.4)
            self.current_actor.GetProperty().EdgeVisibilityOn()
            self.current_actor.GetProperty().SetEdgeColor(0.1, 0.1, 0.1)
            self.current_actor.GetProperty().SetLineWidth(1)

            self.renderer.AddActor(self.current_actor)
            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()

            # Update info with full details
            self.info_label.setText(
                f"Mesh: {Path(filepath).name}\n"
                f"Nodes: {len(nodes):,} | Elements: {len(elements):,}\n"
                f"Tets: {tet_count:,} | Tris: {tri_count:,}"
            )
            self.info_label.adjustSize()

        except Exception as e:
            self.show_placeholder(f"Error loading mesh:\n{str(e)}")

    def _parse_msh_file(self, filepath: str):
        """Parse Gmsh .msh file"""
        nodes = {}
        elements = []

        with open(filepath, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if line == "$Nodes":
                i += 2
                while lines[i].strip() != "$EndNodes":
                    parts = lines[i].strip().split()
                    if len(parts) == 4:
                        num_nodes = int(parts[3])
                        i += 1
                        node_tags = []
                        for _ in range(num_nodes):
                            node_tags.append(int(lines[i].strip()))
                            i += 1
                        for tag in node_tags:
                            coords = lines[i].strip().split()
                            nodes[tag] = [float(coords[0]), float(coords[1]), float(coords[2])]
                            i += 1
                    else:
                        i += 1

            elif line == "$Elements":
                i += 2
                while lines[i].strip() != "$EndElements":
                    parts = lines[i].strip().split()
                    if len(parts) == 4:
                        element_type = int(parts[2])
                        num_elements = int(parts[3])
                        i += 1

                        if element_type == 4:  # Tetrahedron
                            for _ in range(num_elements):
                                data = lines[i].strip().split()
                                if len(data) >= 5:
                                    elements.append({
                                        "type": "tetrahedron",
                                        "nodes": [int(data[1]), int(data[2]), int(data[3]), int(data[4])]
                                    })
                                i += 1
                        elif element_type == 2:  # Triangle
                            for _ in range(num_elements):
                                data = lines[i].strip().split()
                                if len(data) >= 4:
                                    elements.append({
                                        "type": "triangle",
                                        "nodes": [int(data[1]), int(data[2]), int(data[3])]
                                    })
                                i += 1
                        else:
                            for _ in range(num_elements):
                                i += 1
                    else:
                        i += 1
            else:
                i += 1

        return nodes, elements

    def show_placeholder(self, message: str):
        """Show placeholder text"""
        self.clear_view()
        self.info_label.setText(message)
        self.info_label.adjustSize()


class ModernMeshGenGUI(QMainWindow):
    """Improved Khorium MeshGen GUI"""

    def __init__(self):
        super().__init__()
        self.cad_file = None
        self.mesh_file = None
        self.worker = MeshWorker()

        # Phase progress bars
        self.phase_bars = {}

        # Connect worker signals
        self.worker.signals.log.connect(self.add_log)
        self.worker.signals.progress.connect(self.update_progress)
        self.worker.signals.finished.connect(self.on_mesh_finished)

        self.init_ui()

    def init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle("Khorium MeshGen - Exhaustive Mesh Generator")
        self.setGeometry(100, 100, 1500, 950)

        # Light mode palette
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(248, 249, 250))
        palette.setColor(QPalette.WindowText, QColor(33, 37, 41))
        palette.setColor(QPalette.Base, QColor(255, 255, 255))
        palette.setColor(QPalette.AlternateBase, QColor(248, 249, 250))
        palette.setColor(QPalette.Text, QColor(33, 37, 41))
        palette.setColor(QPalette.Button, QColor(255, 255, 255))
        palette.setColor(QPalette.ButtonText, QColor(33, 37, 41))
        self.setPalette(palette)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Left panel
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel)

        # Right panel
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 1)

    def create_left_panel(self):
        """Create left control panel"""
        panel = QFrame()
        panel.setMaximumWidth(380)
        panel.setStyleSheet("""
            QFrame {
                background-color: white;
                border-right: 1px solid #dee2e6;
            }
        """)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Title
        title = QLabel("Khorium MeshGen")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setStyleSheet("color: #212529;")
        layout.addWidget(title)

        subtitle = QLabel("Exhaustive Mesh Generation")
        subtitle.setFont(QFont("Arial", 10))
        subtitle.setStyleSheet("color: #6c757d; margin-bottom: 10px;")
        layout.addWidget(subtitle)

        # File upload section
        upload_group = QGroupBox("Load CAD File")
        upload_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #dee2e6;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)
        upload_layout = QVBoxLayout()

        self.load_btn = QPushButton("Browse CAD File")
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
            QPushButton:hover {
                background-color: #0b5ed7;
            }
        """)
        self.load_btn.clicked.connect(self.load_cad_file)
        upload_layout.addWidget(self.load_btn)

        self.file_label = QLabel("No file loaded")
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet("color: #6c757d; font-size: 11px; margin-top: 5px;")
        upload_layout.addWidget(self.file_label)

        upload_group.setLayout(upload_layout)
        layout.addWidget(upload_group)

        # Generate button
        self.generate_btn = QPushButton("Generate Mesh")
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
            QPushButton:hover:enabled {
                background-color: #157347;
            }
            QPushButton:disabled {
                background-color: #e9ecef;
                color: #adb5bd;
            }
        """)
        self.generate_btn.clicked.connect(self.start_mesh_generation)
        layout.addWidget(self.generate_btn)

        # Progress section with phase bars
        progress_group = QGroupBox("Progress Tracking")
        progress_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #dee2e6;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)
        progress_layout = QVBoxLayout()

        # Create phase progress bars
        phases = [
            ("strategy", "Strategy Selection"),
            ("1d", "1D Meshing (Curves)"),
            ("2d", "2D Meshing (Surfaces)"),
            ("3d", "3D Meshing (Volume)"),
            ("opt", "Optimization (Gmsh)"),
            ("netgen", "Optimization (Netgen)"),
            ("order2", "Higher-Order Elements"),
            ("quality", "Quality Analysis")
        ]

        bar_style = """
            QProgressBar {
                border: 1px solid #dee2e6;
                border-radius: 3px;
                text-align: center;
                background-color: #f8f9fa;
                height: 18px;
                font-size: 10px;
            }
            QProgressBar::chunk {
                background-color: #0d6efd;
                border-radius: 2px;
            }
        """

        for phase_id, phase_name in phases:
            phase_label = QLabel(phase_name)
            phase_label.setStyleSheet("font-size: 10px; color: #6c757d; font-weight: normal;")
            progress_layout.addWidget(phase_label)

            phase_bar = QProgressBar()
            phase_bar.setStyleSheet(bar_style)
            phase_bar.setMaximum(100)
            phase_bar.setValue(0)
            progress_layout.addWidget(phase_bar)

            self.phase_bars[phase_id] = phase_bar

        progress_group.setLayout(progress_layout)

        # Make it scrollable
        scroll = QScrollArea()
        scroll.setWidget(progress_group)
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(350)
        layout.addWidget(scroll)

        # Quality metrics
        self.metrics_group = QGroupBox("Quality Metrics")
        self.metrics_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #dee2e6;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
        """)
        self.metrics_layout = QGridLayout()
        self.metrics_group.setLayout(self.metrics_layout)
        self.metrics_group.setVisible(False)
        layout.addWidget(self.metrics_group)

        layout.addStretch()

        return panel

    def create_right_panel(self):
        """Create right panel with viewer and console"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Viewer controls
        viewer_controls = QFrame()
        viewer_controls.setStyleSheet("background-color: #f8f9fa; border-bottom: 1px solid #dee2e6;")
        controls_layout = QHBoxLayout(viewer_controls)
        controls_layout.setContentsMargins(10, 5, 10, 5)

        axes_checkbox = QCheckBox("Show XYZ Axes")
        axes_checkbox.setChecked(True)
        axes_checkbox.setStyleSheet("font-size: 11px;")
        axes_checkbox.stateChanged.connect(lambda state: self.viewer.toggle_axes(state == Qt.Checked))
        controls_layout.addWidget(axes_checkbox)

        controls_layout.addStretch()

        layout.addWidget(viewer_controls)

        # 3D Viewer
        self.viewer = VTK3DViewer()
        layout.addWidget(self.viewer, 2)

        # Console
        console_frame = QFrame()
        console_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-top: 1px solid #dee2e6;
            }
        """)
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
                font-size: 11px;
                color: #212529;
                padding: 8px;
            }
        """)
        console_layout.addWidget(self.console)

        layout.addWidget(console_frame)

        return panel

    def load_cad_file(self):
        """Load CAD file"""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select CAD File",
            str(Path.home()),
            "CAD Files (*.step *.stp *.stl);;All Files (*)"
        )

        if filepath:
            self.cad_file = filepath
            self.file_label.setText(f"{Path(filepath).name}")
            self.generate_btn.setEnabled(True)
            self.add_log(f"Loaded: {filepath}")
            self.viewer.load_step_file(filepath)

    def start_mesh_generation(self):
        """Start mesh generation"""
        if not self.cad_file:
            return

        self.generate_btn.setEnabled(False)
        self.console.clear()

        # Reset all progress bars
        for bar in self.phase_bars.values():
            bar.setValue(0)

        self.metrics_group.setVisible(False)

        # Clear metrics
        while self.metrics_layout.count():
            item = self.metrics_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.add_log("=" * 60)
        self.add_log("Starting mesh generation...")
        self.add_log("=" * 60)

        self.worker.start(self.cad_file)

    def add_log(self, message: str):
        """Add message to console"""
        self.console.append(message)
        self.console.verticalScrollBar().setValue(
            self.console.verticalScrollBar().maximum()
        )

    def copy_console_to_clipboard(self):
        """Copy all console text to clipboard"""
        clipboard = QApplication.clipboard()
        console_text = self.console.toPlainText()
        clipboard.setText(console_text)
        self.add_log("Console output copied to clipboard!")

    def update_progress(self, phase: str, percentage: int, status: str):
        """Update progress bar for specific phase"""
        if phase in self.phase_bars:
            self.phase_bars[phase].setValue(percentage)
            self.phase_bars[phase].setFormat(f"{percentage}% - {status}")

    def on_mesh_finished(self, success: bool, result: dict):
        """Handle mesh generation completion"""
        self.generate_btn.setEnabled(True)

        if success:
            self.add_log("=" * 60)
            self.add_log("MESH GENERATION COMPLETE")
            self.add_log("=" * 60)

            # Mark all phases complete
            for bar in self.phase_bars.values():
                if bar.value() > 0:
                    bar.setValue(100)

            self.mesh_file = result.get('output_file')
            metrics = result.get('metrics', {})
            self.display_metrics(metrics)

            if self.mesh_file and Path(self.mesh_file).exists():
                self.viewer.load_mesh_file(self.mesh_file)

        else:
            self.add_log("=" * 60)
            self.add_log("MESH GENERATION FAILED")
            self.add_log(f"Error: {result.get('error', 'Unknown error')}")
            self.add_log("=" * 60)

    def display_metrics(self, metrics: dict):
        """Display quality metrics"""
        self.metrics_group.setVisible(True)

        row = 0

        sicn_min = metrics.get('sicn_min') or metrics.get('SICN (Gmsh)', {}).get('min')
        gamma_min = metrics.get('gamma_min') or metrics.get('Gamma (Gmsh)', {}).get('min')
        max_skew = metrics.get('max_skewness') or metrics.get('Skewness (converted)', {}).get('max')
        max_ar = metrics.get('max_aspect_ratio') or metrics.get('Aspect Ratio (converted)', {}).get('max')

        metric_data = [
            ("SICN (min)", sicn_min, lambda v: v >= 0.3),
            ("Gamma (min)", gamma_min, lambda v: v >= 0.2),
            ("Max Skewness", max_skew, lambda v: v <= 0.7),
            ("Max Aspect Ratio", max_ar, lambda v: v <= 5.0),
        ]

        for name, value, is_good in metric_data:
            if value is not None:
                name_label = QLabel(name + ":")
                name_label.setStyleSheet("font-size: 11px; color: #495057;")
                self.metrics_layout.addWidget(name_label, row, 0)

                value_label = QLabel(f"{value:.4f}" if isinstance(value, float) else str(value))
                value_label.setStyleSheet("font-size: 11px; font-weight: bold;")
                self.metrics_layout.addWidget(value_label, row, 1)

                status_label = QLabel("[OK]" if is_good(value) else "[!]")
                status_label.setStyleSheet(f"font-size: 14px; color: {'#198754' if is_good(value) else '#ffc107'};")
                self.metrics_layout.addWidget(status_label, row, 2)

                row += 1


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    gui = ModernMeshGenGUI()
    gui.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
