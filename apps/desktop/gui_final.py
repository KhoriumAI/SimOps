#!/usr/bin/env python3
"""
Final Khorium MeshGen GUI
==========================

Polished features:
- Clean progress bars showing only percentages
- Green bars when complete
- All bars reach 100% properly
- Visual quality report overlay
- Working mesh display using parsed .msh data
- No jitter or text overflow

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
import logging
from pathlib import Path
from typing import Optional, Dict
from queue import Queue

# Set up file logging for debugging
log_file = os.path.join(tempfile.gettempdir(), "meshgen_gui_debug.log")
logging.basicConfig(
    filename=log_file,
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'  # Overwrite on each launch
)
logging.info("="*60)
logging.info("GUI Starting - Log initialized")
logging.info("="*60)

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QProgressBar, QGroupBox,
    QSplitter, QFileDialog, QFrame, QScrollArea, QGridLayout,
    QCheckBox, QSizePolicy, QSlider, QSpinBox, QComboBox, QDoubleSpinBox
)
from qtrangeslider import QRangeSlider
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QPalette, QColor

import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

# Paintbrush refinement imports
PAINTBRUSH_AVAILABLE = False
try:
    # Add paths for imports
    import sys
    from pathlib import Path as PathLib

    app_dir = PathLib(__file__).parent
    project_root = app_dir.parent.parent

    # Add to path if not already there
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from paintbrush_widget import PaintbrushWidget
    from core.paintbrush_geometry import PaintbrushSelector
    from strategies.paintbrush_strategy import PaintbrushStrategy

    PAINTBRUSH_AVAILABLE = True
    print("[OK] Paintbrush feature loaded successfully")
except ImportError as e:
    print(f"[!] Paintbrush feature not available: {e}")
    import traceback
    traceback.print_exc()

# Custom VTK interactor style for better pan control
class CustomInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    """
    Custom interactor style with improved pan controls and paintbrush support:
    - Left mouse: Rotate (or paint if paintbrush mode enabled)
    - Right mouse: Rotate (in paint mode) or Pan (in normal mode)
    - Middle mouse or Shift+Left: Pan (alternative)
    - Scroll wheel: Zoom
    """

    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.painting_mode = False
        self.is_painting = False
        self.AddObserver("RightButtonPressEvent", self.right_button_press)
        self.AddObserver("RightButtonReleaseEvent", self.right_button_release)
        self.AddObserver("LeftButtonPressEvent", self.left_button_press)
        self.AddObserver("LeftButtonReleaseEvent", self.left_button_release)
        self.AddObserver("MouseMoveEvent", self.mouse_move)

    def right_button_press(self, obj, event):
        if self.painting_mode:
            # In paint mode, right mouse rotates
            self.OnLeftButtonDown()
        else:
            # In normal mode, right mouse pans
            self.OnMiddleButtonDown()
        return

    def right_button_release(self, obj, event):
        if self.painting_mode:
            # In paint mode, right mouse rotates
            self.OnLeftButtonUp()
        else:
            # In normal mode, right mouse pans
            self.OnMiddleButtonUp()
        return

    def left_button_press(self, obj, event):
        print(f"[DEBUG] Left button press - Paint mode: {self.painting_mode}, Parent: {self.parent is not None}")
        if self.painting_mode and self.parent:
            # Start painting - consume event to prevent rotation
            print("[DEBUG] Starting paint operation")
            self.is_painting = True
            self.paint_at_cursor()
            # Abort event to prevent further processing
            self.GetInteractor().SetAbortFlag(1)
        else:
            # Normal rotate
            print("[DEBUG] Normal rotation mode")
            self.OnLeftButtonDown()

    def left_button_release(self, obj, event):
        if self.painting_mode:
            self.is_painting = False
            # Abort event
            self.GetInteractor().SetAbortFlag(1)
        else:
            self.OnLeftButtonUp()

    def mouse_move(self, obj, event):
        if self.painting_mode and self.is_painting and self.parent:
            # Continue painting while dragging - DON'T call OnMouseMove to prevent rotation
            self.paint_at_cursor()
            # Abort event to prevent camera rotation
            self.GetInteractor().SetAbortFlag(1)
        elif self.painting_mode and self.parent:
            # In paint mode but not actively painting - update cursor position only
            x, y = self.GetInteractor().GetEventPosition()
            if hasattr(self.parent, 'viewer') and self.parent.viewer:
                self.parent.viewer.update_brush_cursor_position(x, y)
            # Abort event to prevent rotation while in paint mode
            self.GetInteractor().SetAbortFlag(1)
        else:
            # Normal rotation mode
            self.OnMouseMove()

    def paint_at_cursor(self):
        """Paint surfaces at cursor location"""
        if not self.parent or not hasattr(self.parent, 'on_paint_at_cursor'):
            return

        # Get mouse position
        x, y = self.GetInteractor().GetEventPosition()

        # Pass to parent for handling
        self.parent.on_paint_at_cursor(x, y)

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class WorkerSignals(QObject):
    """Signals for mesh worker thread"""
    log = pyqtSignal(str)
    progress = pyqtSignal(str, int)  # (phase, percentage) - simplified
    phase_complete = pyqtSignal(str)  # Signal when phase is 100% done
    finished = pyqtSignal(bool, dict)


class MeshWorker:
    """Subprocess mesh generation worker"""

    def __init__(self):
        self.signals = WorkerSignals()
        self.thread = None
        self.process = None
        self.is_running = False
        self.phase_max = {}  # Track max value reached for each phase

    def start(self, cad_file: str, quality_params: dict = None):
        if self.is_running:
            return

        self.is_running = True
        self.phase_max = {}
        self.thread = threading.Thread(target=self._run, args=(cad_file, quality_params), daemon=True)
        self.thread.start()

    def _emit_progress(self, phase: str, value: int):
        """Emit progress only if it's higher than before (prevent jitter)"""
        if phase not in self.phase_max:
            self.phase_max[phase] = 0

        if value > self.phase_max[phase]:
            self.phase_max[phase] = value
            self.signals.progress.emit(phase, value)

    def _complete_phase(self, phase: str):
        """Mark phase as 100% complete"""
        self.phase_max[phase] = 100
        self.signals.progress.emit(phase, 100)
        self.signals.phase_complete.emit(phase)

    def _parse_and_emit_line(self, line: str):
        """Parse a single log line and emit appropriate signals"""
        if not line:
            return

        # Parse gmsh Info lines
        if "Info" in line:
            # 1D Meshing
            if "Meshing 1D" in line:
                self._emit_progress("1d", 10)
            elif "Meshing curve" in line and "order 2" not in line:
                match = re.search(r'\[\s*(\d+)%\]', line)
                if match:
                    pct = int(match.group(1))
                    self._emit_progress("1d", 10 + int(pct * 0.9))
            elif "Done meshing 1D" in line:
                self._complete_phase("1d")

            # 2D Meshing
            elif "Meshing 2D" in line:
                self._emit_progress("2d", 10)
            elif "Meshing surface" in line and "order 2" not in line:
                match = re.search(r'\[\s*(\d+)%\]', line)
                if match:
                    pct = int(match.group(1))
                    self._emit_progress("2d", 10 + int(pct * 0.9))
            elif "Done meshing 2D" in line:
                self._complete_phase("2d")

            # 3D Meshing
            elif "Meshing 3D" in line:
                self._emit_progress("3d", 10)
            elif "Tetrahedrizing" in line:
                self._emit_progress("3d", 30)
            elif "Reconstructing mesh" in line:
                self._emit_progress("3d", 50)
            elif "3D refinement" in line:
                self._emit_progress("3d", 70)
            elif "Done meshing 3D" in line:
                self._complete_phase("3d")

            # Optimization (Gmsh)
            elif "Optimizing mesh..." in line and "Netgen" not in line:
                self._emit_progress("opt", 10)
            elif "edge swaps" in line:
                self._emit_progress("opt", 60)
            elif "No ill-shaped tets" in line:
                self._emit_progress("opt", 90)
            elif "Done optimizing mesh" in line and "Netgen" not in line:
                self._complete_phase("opt")

            # Optimization (Netgen)
            elif "Optimizing mesh (Netgen)" in line:
                self._emit_progress("netgen", 10)
            elif "SplitImprove" in line:
                self._emit_progress("netgen", min(self.phase_max.get("netgen", 10) + 10, 80))
            elif "SwapImprove" in line:
                self._emit_progress("netgen", min(self.phase_max.get("netgen", 10) + 5, 85))
            elif "CombineImprove" in line:
                self._emit_progress("netgen", min(self.phase_max.get("netgen", 10) + 5, 90))
            elif "Done optimizing mesh (Wall" in line:
                self._complete_phase("netgen")

            # Higher order
            elif "Meshing order 2" in line:
                self._emit_progress("order2", 10)
            elif "order 2" in line:
                match = re.search(r'\[\s*(\d+)%\]', line)
                if match:
                    pct = int(match.group(1))
                    self._emit_progress("order2", 10 + int(pct * 0.9))
            elif "Done meshing order 2" in line:
                self._complete_phase("order2")

        # Strategy attempts
        if "ATTEMPT" in line:
            match = re.search(r'ATTEMPT\s+(\d+)/(\d+)', line)
            if match:
                current = int(match.group(1))
                total = int(match.group(2))
                pct = int((current / total) * 100)
                self._emit_progress("strategy", pct)

        # Quality analysis
        if "Analyzing mesh quality" in line:
            self._emit_progress("quality", 50)
        elif "Quality Score" in line:
            self._emit_progress("quality", 90)

        # Final result
        if line.startswith('{') and '"success"' in line:
            print(f"[GUI-WORKER] Found JSON result line: {line[:100]}...")
            try:
                result = json.loads(line)
                print(f"[GUI-WORKER] Parsed result, success={result.get('success')}")
                if result.get('success'):
                    # Mark all active phases as complete
                    for phase in ["strategy", "1d", "2d", "3d", "opt", "netgen", "order2", "quality"]:
                        if self.phase_max.get(phase, 0) > 0:
                            self._complete_phase(phase)

                    self.signals.log.emit("Mesh generation completed!")
                    self.signals.log.emit(f"[DEBUG] About to emit finished signal with {len(result)} result keys")
                    self.signals.log.emit(f"[DEBUG] Result keys: {list(result.keys())}")
                    self.signals.log.emit(f"[DEBUG] per_element_quality present: {'per_element_quality' in result}")
                    print(f"[GUI-WORKER] Emitting finished signal with result keys: {list(result.keys())}")
                    self.signals.finished.emit(True, result)
                    self.signals.log.emit("[DEBUG] Finished signal emitted!")
                else:
                    self.signals.log.emit(f"Failed: {result.get('error')}")
                    self.signals.finished.emit(False, result)
            except Exception as e:
                print(f"[GUI-WORKER] Failed to parse JSON: {e}")
                pass
        else:
            # Debug: Show lines that look like they might be JSON
            if line.startswith('{') or ('"success"' in line and not line.startswith('Info')):
                print(f"[GUI-WORKER] Potential JSON line (didn't match): {line[:100]}")
            self.signals.log.emit(line)

    def _run(self, cad_file: str, quality_params: dict = None):
        try:
            import tempfile
            from multiprocessing import cpu_count

            self.signals.log.emit(f"Loading: {Path(cad_file).name}")
            self._emit_progress("strategy", 5)

            # Show parallel execution info
            cores = cpu_count()
            workers = max(1, cores - 2)
            self.signals.log.emit("=" * 70)
            self.signals.log.emit("PARALLEL EXECUTION MODE ENABLED")
            self.signals.log.emit(f"System: {cores} CPU cores detected")
            self.signals.log.emit(f"Using: {workers} parallel workers (strategies run concurrently)")
            self.signals.log.emit(f"Expected speedup: 3-5x faster than sequential mode")
            self.signals.log.emit("=" * 70)

            # Path to mesh worker (now in apps/cli/)
            worker_script = Path(__file__).parent.parent / "cli" / "mesh_worker_subprocess.py"
            if not worker_script.exists():
                # Fallback: check if mesh_worker.py exists instead
                worker_script = Path(__file__).parent.parent / "cli" / "mesh_worker.py"
            self.signals.log.emit("Starting parallel mesh generation...")

            # Prepare command with quality parameters
            cmd = [sys.executable, str(worker_script), cad_file]
            
            # Use a temporary file for configuration to avoid command line length limits
            # This is critical for paintbrush feature which can have large data
            self.temp_config_file = None
            if quality_params:
                try:
                    # Create a named temp file that persists after close
                    fd, config_path = tempfile.mkstemp(suffix='.json', prefix='mesh_config_')
                    os.close(fd)
                    
                    with open(config_path, 'w') as f:
                        json.dump(quality_params, f)
                    
                    cmd.extend(["--config-file", config_path])
                    self.temp_config_file = config_path
                    self.signals.log.emit(f"[DEBUG] Wrote config to: {config_path}")
                except Exception as e:
                    self.signals.log.emit(f"[ERROR] Failed to create config file: {e}")
                    # Fallback to CLI args if file creation fails (though unlikely to work if too long)
                    cmd.extend(["--quality-params", json.dumps(quality_params)])

            self.signals.log.emit(f"[DEBUG] Executing command: {' '.join(cmd)}")
            print(f"[DEBUG] Executing command: {' '.join(cmd)}")

            # Use subprocess.PIPE instead of tempfile for better stability
            self.signals.log.emit("[DEBUG] Starting subprocess with PIPE...")
            
            # Start subprocess with PIPE
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1  # Line buffered
            )
            self.signals.log.emit(f"[DEBUG] Subprocess started with PID: {self.process.pid}")

            # Read line by line from stdout
            for line in self.process.stdout:
                line = line.strip()
                if not line:
                    continue
                
                self._parse_and_emit_line(line)

            self.process.wait()
            self.signals.log.emit(f"[DEBUG] Subprocess finished with return code: {self.process.returncode}")

        except Exception as e:
            self.signals.log.emit(f"Exception: {str(e)}")
            self.signals.finished.emit(False, {"error": str(e)})
        finally:
            self.is_running = False
            # Clean up temp config file
            if hasattr(self, 'temp_config_file') and self.temp_config_file and os.path.exists(self.temp_config_file):
                try:
                    os.remove(self.temp_config_file)
                except:
                    pass

    def stop(self):
        """Stop the running mesh generation subprocess"""
        import signal
        if self.process and self.process.poll() is None:
            # Process is still running
            try:
                # Try graceful termination first (SIGTERM)
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    # If still running after 3 seconds, force kill (SIGKILL)
                    self.process.kill()
                    self.process.wait()
            except Exception as e:
                # Best effort - if it fails, process might already be dead
                pass
        self.is_running = False


class VTK3DViewer(QFrame):
    """3D viewer with quality report overlay"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Box | QFrame.Sunken)
        self.setLineWidth(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtk_widget)

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.95, 0.95, 0.97)
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)

        # Add ambient lighting to prevent pitch-black shadows
        # This ensures all areas have a minimum illumination level
        self.renderer.SetAmbient(0.4, 0.4, 0.4)  # Soft ambient light

        # Add lights to enhance 3D appearance and show mesh facets clearly
        # Key light (main light from upper front-right)
        light1 = vtk.vtkLight()
        light1.SetPosition(1, 1, 1)
        light1.SetFocalPoint(0, 0, 0)
        light1.SetColor(1.0, 1.0, 1.0)
        light1.SetIntensity(0.6)  # Reduced from 0.8 to balance with ambient
        self.renderer.AddLight(light1)

        # Fill light (softer light from upper left to fill shadows)
        light2 = vtk.vtkLight()
        light2.SetPosition(-0.5, 1, 0.5)
        light2.SetFocalPoint(0, 0, 0)
        light2.SetColor(1.0, 1.0, 1.0)
        light2.SetIntensity(0.3)  # Reduced from 0.4 to balance with ambient
        self.renderer.AddLight(light2)

        # Back light (subtle rim light to show edges)
        light3 = vtk.vtkLight()
        light3.SetPosition(0, 0.5, -1)
        light3.SetFocalPoint(0, 0, 0)
        light3.SetColor(1.0, 1.0, 1.0)
        light3.SetIntensity(0.2)  # Reduced from 0.3 to balance with ambient
        self.renderer.AddLight(light3)

        # Use custom interactor style with right-click pan and paintbrush support
        self.interactor_style = CustomInteractorStyle(parent=parent)
        self.vtk_widget.GetRenderWindow().GetInteractor().SetInteractorStyle(self.interactor_style)

        # Orientation marker
        self.axes_widget = vtk.vtkOrientationMarkerWidget()
        self.axes_actor = vtk.vtkAxesActor()
        self.axes_actor.SetShaftTypeToCylinder()
        self.axes_actor.SetTotalLength(1.0, 1.0, 1.0)
        self.axes_actor.SetCylinderRadius(0.02)
        self.axes_actor.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(12)
        self.axes_actor.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(12)
        self.axes_actor.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(12)
        self.axes_widget.SetOrientationMarker(self.axes_actor)
        self.axes_widget.SetInteractor(self.vtk_widget.GetRenderWindow().GetInteractor())
        self.axes_widget.SetViewport(0, 0, 0.15, 0.15)
        self.axes_widget.SetEnabled(1)
        self.axes_widget.InteractiveOff()

        self.current_actor = None
        self.current_poly_data = None  # Store unclipped mesh data
        self.current_mesh_nodes = None  # Store nodes dict {id: [x,y,z]}
        self.current_mesh_elements = None  # Store all elements list
        self.current_node_map = None  # Store node_id -> vtk_index mapping
        self.current_tetrahedra = None  # Store tetrahedra separately for cross-section
        self.current_hexahedra = None  # Store hexahedra for volume visualization
        self.current_volumetric_grid = None  # Store full volumetric grid for 3D clipping
        self.clipping_enabled = False
        self.clip_plane = None
        self.clip_axis = 'x'
        self.clip_offset = 0.0  # -50 to +50 percentage
        self.cross_section_actor = None  # Actor for cross-section visualization
        self.cross_section_mode = "layered"  # Always use layered mode (show complete volume cells)
        self.cross_section_element_mode = "auto"  # Auto-switch between tet/hex slicing

        # Paintbrush visual feedback
        self.brush_cursor_actor = None
        self.brush_cursor_visible = False
        self.painted_cells = set()  # Set of VTK cell IDs that have been painted
        self.paint_colors = vtk.vtkUnsignedCharArray()  # RGB colors per cell
        self.paint_colors.SetNumberOfComponents(3)
        self.paint_colors.SetName("PaintColors")

        self.vtk_widget.Initialize()
        self.vtk_widget.Start()

        # Info overlay (top-left) - FIXED WIDTH to prevent truncation
        self.info_label = QLabel("No CAD file loaded", self)
        self.info_label.setStyleSheet("""
            QLabel {
                background-color: rgba(255, 255, 255, 230);
                padding: 12px 16px;
                border-radius: 6px;
                font-size: 11px;
                color: #212529;
                font-weight: 500;
                border: 1px solid rgba(0,0,0,0.1);
            }
        """)
        self.info_label.setFixedWidth(450)  # Increased from 400 to 450
        self.info_label.setMinimumHeight(80)  # Minimum height to prevent vertical truncation
        self.info_label.setMaximumHeight(300)  # Increased from 200 to 300px for iterations
        self.info_label.setWordWrap(True)
        self.info_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.info_label.move(15, 15)

        # Quality report overlay (top-right) - initially hidden
        self.quality_label = QLabel("", self)
        self.quality_label.setTextFormat(Qt.RichText)  # Enable HTML rendering
        self.quality_label.setTextInteractionFlags(Qt.TextSelectableByMouse)  # Allow selection
        self.quality_label.setStyleSheet("""
            QLabel {
                background-color: rgba(255, 255, 255, 230);
                padding: 12px 16px;
                border-radius: 6px;
                font-size: 11px;
                color: #212529;
                border: 1px solid rgba(0,0,0,0.1);
            }
        """)
        self.quality_label.setMinimumWidth(250)
        self.quality_label.setMaximumWidth(300)
        self.quality_label.setWordWrap(True)
        self.quality_label.setVisible(False)

        # Iteration selector buttons (bottom-right)
        self.iteration_buttons = []
        self.iteration_meshes = []  # Store mesh file paths for each iteration
        self.current_iteration = 0

        self.iteration_container = QFrame(self)
        self.iteration_container.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 230);
                border-radius: 6px;
                border: 1px solid rgba(0,0,0,0.1);
                padding: 5px;
            }
        """)
        iteration_layout = QHBoxLayout(self.iteration_container)
        iteration_layout.setContentsMargins(5, 5, 5, 5)
        iteration_layout.setSpacing(5)

        # Create buttons for iterations 1-5
        for i in range(1, 6):
            btn = QPushButton(str(i))
            btn.setFixedSize(30, 30)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #e9ecef;
                    border: 1px solid #adb5bd;
                    border-radius: 4px;
                    font-size: 12px;
                    font-weight: bold;
                    color: #495057;
                }
                QPushButton:hover {
                    background-color: #dee2e6;
                    border-color: #6c757d;
                }
                QPushButton:disabled {
                    background-color: #f8f9fa;
                    color: #ced4da;
                    border-color: #dee2e6;
                }
                QPushButton[selected="true"] {
                    background-color: #0d6efd;
                    color: white;
                    border-color: #0d6efd;
                }
            """)
            btn.setEnabled(False)  # Disabled until iteration exists
            btn.clicked.connect(lambda checked, idx=i-1: self.switch_to_iteration(idx))
            iteration_layout.addWidget(btn)
            self.iteration_buttons.append(btn)

        self.iteration_container.setVisible(False)  # Hidden until we have iterations

    def resizeEvent(self, event):
        """Reposition quality label and iteration buttons on resize"""
        super().resizeEvent(event)
        if self.quality_label.isVisible():
            self.quality_label.move(
                self.width() - self.quality_label.width() - 15,
                15
            )

        # Position iteration container in bottom-right corner
        if self.iteration_container.isVisible():
            self.iteration_container.adjustSize()
            self.iteration_container.move(
                self.width() - self.iteration_container.width() - 15,
                self.height() - self.iteration_container.height() - 15
            )

    def toggle_axes(self, visible: bool):
        self.axes_widget.SetEnabled(1 if visible else 0)
        self.vtk_widget.GetRenderWindow().Render()

    def clear_view(self):
        if self.current_actor:
            self.renderer.RemoveActor(self.current_actor)
            self.current_actor = None
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

    def show_quality_report(self, metrics: Dict):
        """Show quality metrics overlay in top-right"""
        # Extract from nested structure (gmsh_sicn: {min, avg, max})
        sicn_min = None
        if 'gmsh_sicn' in metrics and isinstance(metrics['gmsh_sicn'], dict):
            sicn_min = metrics['gmsh_sicn'].get('min')
        elif 'sicn_min' in metrics:
            sicn_min = metrics['sicn_min']

        gamma_min = None
        if 'gmsh_gamma' in metrics and isinstance(metrics['gmsh_gamma'], dict):
            gamma_min = metrics['gmsh_gamma'].get('min')
        elif 'gamma_min' in metrics:
            gamma_min = metrics['gamma_min']

        max_skew = None
        if 'skewness' in metrics and isinstance(metrics['skewness'], dict):
            max_skew = metrics['skewness'].get('max')
        elif 'max_skewness' in metrics:
            max_skew = metrics['max_skewness']

        max_ar = None
        if 'aspect_ratio' in metrics and isinstance(metrics['aspect_ratio'], dict):
            max_ar = metrics['aspect_ratio'].get('max')
        elif 'max_aspect_ratio' in metrics:
            max_ar = metrics['max_aspect_ratio']

        # Extract badness (element quality measure from Netgen)
        badness = None
        if 'badness' in metrics and isinstance(metrics['badness'], dict):
            badness = metrics['badness'].get('max')
        elif 'max_badness' in metrics:
            badness = metrics['max_badness']

        # Extract worst tet radius ratio
        worst_tet_radius = None
        if 'tet_radius_ratio' in metrics and isinstance(metrics['tet_radius_ratio'], dict):
            worst_tet_radius = metrics['tet_radius_ratio'].get('min')
        elif 'worst_tet_radius_ratio' in metrics:
            worst_tet_radius = metrics['worst_tet_radius_ratio']

        # Get geometric accuracy (most important for shape fidelity)
        geom_accuracy = metrics.get('geometric_accuracy')
        mean_dev = metrics.get('mean_deviation_mm')

        # Determine grade based on geometric accuracy (prioritized)
        grade = "Excellent"
        grade_color = "#198754"
        if geom_accuracy is not None:
            # Use geometric accuracy as primary metric
            if geom_accuracy >= 0.95:
                grade = "Excellent"
                grade_color = "#198754"
            elif geom_accuracy >= 0.85:
                grade = "Good"
                grade_color = "#0d6efd"
            elif geom_accuracy >= 0.75:
                grade = "Fair"
                grade_color = "#ffc107"
            elif geom_accuracy >= 0.60:
                grade = "Poor"
                grade_color = "#fd7e14"
            else:
                grade = "Critical"
                grade_color = "#dc3545"
        elif sicn_min is not None:
            # Fallback to SICN if no geometric accuracy
            if sicn_min < 0.0001:
                grade = "Critical"
                grade_color = "#dc3545"
            elif sicn_min < 0.1:
                grade = "Very Poor"
                grade_color = "#dc3545"
            elif sicn_min < 0.3:
                grade = "Poor"
                grade_color = "#fd7e14"
            elif sicn_min < 0.5:
                grade = "Fair"
                grade_color = "#ffc107"
            elif sicn_min < 0.7:
                grade = "Good"
                grade_color = "#0d6efd"

        report_html = f"""
        <div style='font-family: Arial; font-size: 11px;'>
            <div style='font-size: 13px; font-weight: bold; margin-bottom: 8px; color: {grade_color};'>
                Quality: {grade}
            </div>
        """

        # Show geometric accuracy FIRST (most important)
        if geom_accuracy is not None:
            icon = "[OK]" if geom_accuracy >= 0.85 else "[X]"
            color = "#198754" if geom_accuracy >= 0.85 else "#dc3545"
            report_html += f"<div><b>Shape Accuracy:</b> {geom_accuracy:.3f} <span style='color: {color};'>{icon}</span></div>"
            if mean_dev is not None:
                report_html += f"<div style='font-size: 10px; color: #6c757d;'>   (deviation: {mean_dev:.3f}mm)</div>"

        if sicn_min is not None:
            icon = "[OK]" if sicn_min >= 0.3 else "[X]"
            color = "#198754" if sicn_min >= 0.3 else "#dc3545"
            report_html += f"<div><b>SICN (min):</b> {sicn_min:.4f} <span style='color: {color};'>{icon}</span></div>"

        if gamma_min is not None:
            icon = "[OK]" if gamma_min >= 0.2 else "[X]"
            color = "#198754" if gamma_min >= 0.2 else "#dc3545"
            report_html += f"<div><b>Gamma (min):</b> {gamma_min:.4f} <span style='color: {color};'>{icon}</span></div>"

        if max_skew is not None:
            icon = "[OK]" if max_skew <= 0.7 else "[X]"
            color = "#198754" if max_skew <= 0.7 else "#dc3545"
            report_html += f"<div><b>Max Skewness:</b> {max_skew:.4f} <span style='color: {color};'>{icon}</span></div>"

        if max_ar is not None:
            icon = "[OK]" if max_ar <= 5.0 else "[X]"
            color = "#198754" if max_ar <= 5.0 else "#dc3545"
            report_html += f"<div><b>Max Aspect Ratio:</b> {max_ar:.2f} <span style='color: {color};'>{icon}</span></div>"

        if badness is not None:
            # Badness: lower is better. <10 is good, >100 is bad
            icon = "[OK]" if badness <= 10 else "[X]"
            color = "#198754" if badness <= 10 else "#dc3545"
            report_html += f"<div><b>Badness (max):</b> {badness:.2f} <span style='color: {color};'>{icon}</span></div>"

        if worst_tet_radius is not None:
            # Tet radius ratio: closer to 1.0 is better (perfect tetrahedron)
            icon = "[OK]" if worst_tet_radius >= 0.5 else "[X]"
            color = "#198754" if worst_tet_radius >= 0.5 else "#dc3545"
            report_html += f"<div><b>Worst Tet Radius:</b> {worst_tet_radius:.4f} <span style='color: {color};'>{icon}</span></div>"

        report_html += "</div>"

        self.quality_label.setText(report_html)
        self.quality_label.adjustSize()
        self.quality_label.move(
            self.width() - self.quality_label.width() - 15,
            15
        )
        self.quality_label.setVisible(True)

    def add_iteration_mesh(self, mesh_path: str, metrics: Dict = None):
        """Add a new iteration mesh and display it"""
        logging.info(f"add_iteration_mesh called: {mesh_path}")

        if len(self.iteration_meshes) >= 5:
            logging.warning("Maximum 5 iterations reached, not adding more")
            return

        # Store the mesh path
        self.iteration_meshes.append({
            'path': mesh_path,
            'metrics': metrics
        })

        iteration_idx = len(self.iteration_meshes) - 1
        logging.info(f"Added iteration {iteration_idx + 1} at path: {mesh_path}")

        # Enable the corresponding button
        self.iteration_buttons[iteration_idx].setEnabled(True)

        # Show the iteration container if this is the first iteration
        if len(self.iteration_meshes) == 1:
            self.iteration_container.setVisible(True)
            self.iteration_container.adjustSize()
            self.iteration_container.move(
                self.width() - self.iteration_container.width() - 15,
                self.height() - self.iteration_container.height() - 15
            )

        # Auto-switch to the new iteration
        self.switch_to_iteration(iteration_idx)

    def switch_to_iteration(self, iteration_idx: int):
        """Switch to display a specific iteration's mesh"""
        logging.info(f"switch_to_iteration called: {iteration_idx}")

        if iteration_idx < 0 or iteration_idx >= len(self.iteration_meshes):
            logging.warning(f"Invalid iteration index: {iteration_idx}")
            return

        self.current_iteration = iteration_idx
        iteration_data = self.iteration_meshes[iteration_idx]

        # Update button styling to show which is selected
        for i, btn in enumerate(self.iteration_buttons):
            if i == iteration_idx:
                btn.setProperty("selected", "true")
            else:
                btn.setProperty("selected", "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

        # Load and display the mesh
        mesh_path = iteration_data['path']
        metrics = iteration_data.get('metrics')

        logging.info(f"Loading iteration {iteration_idx + 1}: {mesh_path}")

        if Path(mesh_path).exists():
            self.load_mesh_file(mesh_path, {'quality_metrics': metrics} if metrics else None)

            # Update info label to show iteration number (prepend to existing HTML)
            info_text = self.info_label.text()
            # Insert iteration text at the beginning, preserving HTML structure
            info_text = f"<b>Iteration {iteration_idx + 1}</b><br>" + info_text
            self.info_label.setText(info_text)

            # Show quality report if we have metrics
            if metrics:
                self.show_quality_report(metrics)
        else:
            logging.error(f"Mesh file not found: {mesh_path}")

    def clear_iterations(self):
        """Clear all iteration data (e.g., when loading a new CAD file)"""
        logging.info("Clearing all iterations")
        self.iteration_meshes = []
        self.current_iteration = 0
        self.iteration_container.setVisible(False)

        for btn in self.iteration_buttons:
            btn.setEnabled(False)
            btn.setProperty("selected", "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def set_clipping(self, enabled: bool, axis: str = 'x', offset: float = 0.0):
        """
        Enable/disable cross-section clipping with live updates.
        
        Args:
            enabled: Whether to enable clipping
            axis: 'x', 'y', or 'z'
            offset: Offset from center as percentage (-50 to +50)
        """
        self.clipping_enabled = enabled
        self.clip_axis = axis.lower()
        self.clip_offset = offset
        
        if not self.current_poly_data:
            return
        
        if enabled:
            self._apply_clipping()
        else:
            self._remove_clipping()
    
    def set_cross_section_mode(self, mode: str):
        """
        Set cross-section visualization mode.
        
        Args:
            mode: Either "perfect" for geometric slice or "layered" for complete volume cells
        """
        if mode not in ["perfect", "layered"]:
            print(f"[WARNING] Invalid cross-section mode '{mode}', using 'perfect'")
            mode = "perfect"
        
        self.cross_section_mode = mode
        print(f"[DEBUG] Cross-section mode set to: {mode}")
        
        # Re-apply clipping if currently enabled
        if self.clipping_enabled and self.current_poly_data:
            self._apply_clipping()
    
    def set_cross_section_element_mode(self, mode: str):
        """
        Choose which volume elements to display in the slice
        mode: 'auto', 'tetrahedra', or 'hexahedra'
        """
        mode = mode.lower()
        if mode not in ("auto", "tetrahedra", "hexahedra"):
            mode = "auto"
        self.cross_section_element_mode = mode
        print(f"[DEBUG] Cross-section element mode set to: {mode}")
        if self.clipping_enabled and self.current_poly_data:
            self._apply_clipping()
    
    def _signed_distance_to_plane(self, point, plane_origin, plane_normal):
        """
        Compute signed distance from a point to a plane.
        Positive = point is on the side the normal points to.
        Negative = point is on the opposite side.
        """
        import numpy as np
        diff = np.array(point) - np.array(plane_origin)
        return np.dot(diff, plane_normal)
    
    def _iter_volume_elements(self):
        """Yield stored volume cells for cross-section operations, respecting mode."""
        if not self.current_mesh_nodes:
            return
        
        mode = getattr(self, 'cross_section_element_mode', 'auto')
        tets = self.current_tetrahedra or []
        hexes = self.current_hexahedra or []
        
        if mode == "auto":
            if hexes and (len(hexes) >= len(tets)):
                mode = "hexahedra"
            elif tets:
                mode = "tetrahedra"
            elif hexes:
                mode = "hexahedra"
            else:
                mode = "tetrahedra"
        
        if mode == "tetrahedra" and tets:
            for tet in tets:
                yield tet
        elif mode == "hexahedra" and hexes:
            for hexa in hexes:
                yield hexa
        else:
            # Fallback: yield whatever is available
            for tet in tets:
                yield tet
            for hexa in hexes:
                yield hexa
    
    def _get_volume_elements_intersecting_plane(self, plane_origin, plane_normal):
        """
        Find all stored volume elements (tets/hexes) that intersect the plane.
        """
        intersecting = []
        for element in self._iter_volume_elements() or []:
            node_ids = element['nodes']
            vertices = [self.current_mesh_nodes[nid] for nid in node_ids]
            distances = [self._signed_distance_to_plane(v, plane_origin, plane_normal)
                         for v in vertices]
            has_positive = any(d > 1e-10 for d in distances)
            has_negative = any(d < -1e-10 for d in distances)
            if has_positive and has_negative:
                intersecting.append(element)
        return intersecting
    
    def _intersect_edge_with_plane(self, v1, v2, plane_origin, plane_normal):
        """
        Find the intersection point of a line segment (v1-v2) with a plane.
        Returns None if no intersection, otherwise returns the intersection point.
        """
        import numpy as np
        
        # Compute signed distances
        d1 = self._signed_distance_to_plane(v1, plane_origin, plane_normal)
        d2 = self._signed_distance_to_plane(v2, plane_origin, plane_normal)
        
        # Check if edge crosses plane (different signs)
        if d1 * d2 > 0:
            return None  # Both on same side
        
        # Linear interpolation to find intersection point
        # t is the parameter: intersection_point = v1 + t * (v2 - v1)
        t = d1 / (d1 - d2)
        
        v1_arr = np.array(v1)
        v2_arr = np.array(v2)
        intersection = v1_arr + t * (v2_arr - v1_arr)
        
        return intersection.tolist()
    
    def _intersect_element_with_plane(self, element, plane_origin, plane_normal):
        """
        Compute the intersection polygon of a volume element (tet or hex) with a plane.
        """
        import numpy as np
        
        node_ids = element['nodes']
        vertices = [self.current_mesh_nodes[nid] for nid in node_ids]
        
        if element['type'] == 'tetrahedron':
            edges = [
                (0, 1), (0, 2), (0, 3),
                (1, 2), (1, 3), (2, 3)
            ]
        elif element['type'] == 'hexahedron':
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7)
            ]
        else:
            return []
        
        intersection_points = []
        for i, j in edges:
            point = self._intersect_edge_with_plane(
                vertices[i], vertices[j], plane_origin, plane_normal
            )
            if point is not None:
                intersection_points.append(point)
        
        # Deduplicate points (when plane passes through tet vertex, multiple edges report same point)
        EPSILON = 1e-9
        unique_points = []
        for pt in intersection_points:
            is_duplicate = False
            for existing_pt in unique_points:
                dist_sq = sum((pt[k] - existing_pt[k])**2 for k in range(3))
                if dist_sq < EPSILON * EPSILON:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_points.append(pt)
        
        # Need at least 3 points to form a polygon
        if len(unique_points) < 3:
            return unique_points
        
        # Order points by angle around centroid for correct winding
        centroid = np.mean(unique_points, axis=0)
        
        # Create two orthogonal vectors in the plane for 2D projection
        normal = np.array(plane_normal)
        normal = normal / np.linalg.norm(normal)
        
        # Pick arbitrary vector not parallel to normal
        if abs(normal[0]) < 0.9:
            arbitrary = np.array([1, 0, 0])
        else:
            arbitrary = np.array([0, 1, 0])
        
        # Create orthonormal basis in the plane
        u = np.cross(normal, arbitrary)
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        
        # Project points to 2D and compute angles
        def get_angle(pt):
            relative = np.array(pt) - centroid
            x = np.dot(relative, u)
            y = np.dot(relative, v)
            return np.arctan2(y, x)
        
        # Sort points by angle
        unique_points.sort(key=get_angle)
        
        return unique_points
    
    def _generate_cross_section_mesh(self, plane_origin, plane_normal):
        """
        Generate VTK PolyData for the cross-section by intersecting all volume elements with the plane.
        
        Returns:
            vtk.vtkPolyData containing the cross-section geometry
        """
        import numpy as np
        
        intersecting_elements = self._get_volume_elements_intersecting_plane(plane_origin, plane_normal)
        
        if not intersecting_elements:
            # No intersection, return empty polydata
            return vtk.vtkPolyData()
        
        # Collect all intersection polygons
        all_points = []
        all_triangles = []
        point_offset = 0
        
        skipped_degenerate = 0
        
        for element in intersecting_elements:
            polygon_points = self._intersect_element_with_plane(element, plane_origin, plane_normal)
            
            if len(polygon_points) < 3:
                skipped_degenerate += 1
                continue  # Skip degenerate polygons
            
            # Add points to the global list
            num_points = len(polygon_points)
            all_points.extend(polygon_points)
            
            # Triangulate the polygon (points are now ordered, so simple fan works)
            # For 3 points: single triangle
            # For 4 points: split into 2 triangles
            if num_points == 3:
                all_triangles.append([point_offset, point_offset + 1, point_offset + 2])
            elif num_points == 4:
                # Create two triangles with consistent winding
                all_triangles.append([point_offset, point_offset + 1, point_offset + 2])
                all_triangles.append([point_offset, point_offset + 2, point_offset + 3])
            else:
                # For more points (rare), use a fan triangulation from first point
                for i in range(1, num_points - 1):
                    all_triangles.append([point_offset, point_offset + i, point_offset + i + 1])
            
            point_offset += num_points
        
        if skipped_degenerate > 0:
            print(f"[DEBUG] Cross-section: skipped {skipped_degenerate} degenerate polygons")
        
        # Create VTK PolyData
        points = vtk.vtkPoints()
        for pt in all_points:
            points.InsertNextPoint(pt)
        
        triangles = vtk.vtkCellArray()
        for tri in all_triangles:
            triangle = vtk.vtkTriangle()
            for i, idx in enumerate(tri):
                triangle.GetPointIds().SetId(i, idx)
            triangles.InsertNextCell(triangle)
        
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetPolys(triangles)
        
        print(f"[DEBUG] Cross-section: {len(intersecting_elements)} intersecting volume cells -> {len(all_triangles)} triangles, {len(all_points)} points")
        
        return poly_data
    
    def _generate_layered_cross_section(self, plane_origin, plane_normal):
        """
        Generate VTK PolyData for layered cross-section showing complete volume elements.
        
        Instead of slicing geometrically, this shows the complete faces for all
        tets/hexes that intersect the plane (ANSYS-style coarse visualization).
        
        Returns:
            vtk.vtkPolyData containing complete faces with quality colors if available
        """
        import numpy as np
        
        intersecting_elements = self._get_volume_elements_intersecting_plane(plane_origin, plane_normal)
        
        if not intersecting_elements:
            return vtk.vtkPolyData()
        
        tet_faces = [
            (0, 1, 2),
            (0, 1, 3),
            (0, 2, 3),
            (1, 2, 3)
        ]
        hex_faces = [
            (0, 1, 2, 3),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 0, 4, 7)
        ]
        
        all_points = []
        all_triangles = []
        point_offset = 0
        
        # Track tet IDs for quality coloring
        face_to_element_id = []  # Maps each output triangle to its source element ID
        
        for element in intersecting_elements:
            node_ids = element['nodes']
            vertices = [self.current_mesh_nodes[nid] for nid in node_ids]
            
            local_offset = point_offset
            all_points.extend(vertices)
            
            if element['type'] == 'tetrahedron':
                for face_indices in tet_faces:
                    tri_indices = [local_offset + i for i in face_indices]
                    all_triangles.append(tri_indices)
                    face_to_element_id.append(element['id'])
            elif element['type'] == 'hexahedron':
                for quad in hex_faces:
                    a, b, c, d = quad
                    all_triangles.append([local_offset + a, local_offset + b, local_offset + c])
                    face_to_element_id.append(element['id'])
                    all_triangles.append([local_offset + a, local_offset + c, local_offset + d])
                    face_to_element_id.append(element['id'])
            
            point_offset += len(vertices)
        
        # Create VTK PolyData
        points = vtk.vtkPoints()
        for pt in all_points:
            points.InsertNextPoint(pt)
        
        triangles = vtk.vtkCellArray()
        for tri in all_triangles:
            triangle = vtk.vtkTriangle()
            for i, idx in enumerate(tri):
                triangle.GetPointIds().SetId(i, idx)
            triangles.InsertNextCell(triangle)
        
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetPolys(triangles)
        
        # Apply quality colors if available
        if hasattr(self, 'current_quality_data') and self.current_quality_data:
            per_elem_quality = self.current_quality_data.get('per_element_quality', {})
            
            # Helper: HSL to RGB
            def hsl_to_rgb(h, s, l):
                def hue_to_rgb(p, q, t):
                    if t < 0: t += 1
                    if t > 1: t -= 1
                    if t < 1/6: return p + (q - p) * 6 * t
                    if t < 1/2: return q
                    if t < 2/3: return p + (q - p) * (2/3 - t) * 6
                    return p
                
                if s == 0:
                    r = g = b = l
                else:
                    q = l * (1 + s) if l < 0.5 else l + s - l * s
                    p = 2 * l - q
                    r = hue_to_rgb(p, q, h + 1/3)
                    g = hue_to_rgb(p, q, h)
                    b = hue_to_rgb(p, q, h - 1/3)
                
                return int(r * 255), int(g * 255), int(b * 255)
            
            # Create color array
            colors = vtk.vtkUnsignedCharArray()
            colors.SetNumberOfComponents(3)
            colors.SetName("Colors")
            
            colored_faces = 0
            for face_idx, elem_id in enumerate(face_to_element_id):
                elem_id_str = str(elem_id)
                quality = per_elem_quality.get(elem_id_str)
                
                if quality is not None:
                    # Map quality [0, 1] to hue [0° (red), 120° (green)]
                    # 0 = poor (red), 1 = good (green)
                    hue = quality * 0.333  # 0.333 = 120°/360°
                    r, g, b = hsl_to_rgb(hue, 0.8, 0.5)
                    colors.InsertNextTuple3(r, g, b)
                    colored_faces += 1
                else:
                    # No quality data for this tet, use gray
                    colors.InsertNextTuple3(128, 128, 128)
            
            if colored_faces > 0:
                poly_data.GetCellData().SetScalars(colors)
                print(f"[DEBUG] Applied quality colors to {colored_faces}/{len(all_triangles)} layered faces")
        
        print(f"[DEBUG] Layered cross-section: {len(intersecting_elements)} intersecting volume cells -> {len(all_triangles)} faces")
        
        return poly_data
    
    def _apply_clipping(self):
        """Apply cross-section with clear above/below separation"""
        if not self.current_poly_data or not self.current_actor:
            return
        
        bounds = self.current_poly_data.GetBounds()
        center = [
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2
        ]
        
        # Determine plane normal and origin based on axis and offset
        if self.clip_axis == 'x':
            normal = (-1, 0, 0)
            dim_size = bounds[1] - bounds[0]
            origin = [center[0] + (dim_size * self.clip_offset / 100.0), center[1], center[2]]
        elif self.clip_axis == 'y':
            normal = (0, -1, 0)
            dim_size = bounds[3] - bounds[2]
            origin = [center[0], center[1] + (dim_size * self.clip_offset / 100.0), center[2]]
        else:  # z
            normal = (0, 0, -1)
            dim_size = bounds[5] - bounds[4]
            origin = [center[0], center[1], center[2] + (dim_size * self.clip_offset / 100.0)]
        
        # Create clipping plane
        plane = vtk.vtkPlane()
        plane.SetNormal(normal)
        plane.SetOrigin(origin)
        
        # --- PART 1: BELOW THE CUT (100% visible) ---
        clipper_below = vtk.vtkClipPolyData()
        clipper_below.SetInputData(self.current_poly_data)
        clipper_below.SetClipFunction(plane)
        clipper_below.InsideOutOff()  # Keep the side the normal points AWAY from
        clipper_below.Update()
        
        # Update main actor with below-cut mesh (fully visible)
        mapper = self.current_actor.GetMapper()
        mapper.SetInputData(clipper_below.GetOutput())
        self.current_actor.GetProperty().SetOpacity(1.0)  # Fully visible
        
        # --- PART 2: ABOVE THE CUT (5% visible for context) ---
        clipper_above = vtk.vtkClipPolyData()
        clipper_above.SetInputData(self.current_poly_data)
        clipper_above.SetClipFunction(plane)
        clipper_above.InsideOutOn()  # Keep the side the normal points TO
        clipper_above.Update()
        
        # Create/update actor for above-cut mesh (transparent)
        if not hasattr(self, 'above_cut_actor') or not self.above_cut_actor:
            self.above_cut_actor = vtk.vtkActor()
            self.renderer.AddActor(self.above_cut_actor)
            
        above_mapper = vtk.vtkPolyDataMapper()
        above_mapper.SetInputData(clipper_above.GetOutput())
        
        # IMPORTANT: Render ghost BEHIND cross-section using depth offset
        # Positive offset pushes away from camera
        above_mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(1, 1)
        
        self.above_cut_actor.SetMapper(above_mapper)
        self.above_cut_actor.GetProperty().SetOpacity(0.05)  # Very transparent ghost
        self.above_cut_actor.GetProperty().SetColor(0.7, 0.7, 0.7)  # Light gray
        # Set ghost visibility based on checkbox
        if hasattr(self, 'show_ghost_cb') and self.show_ghost_cb.isChecked():
            self.above_cut_actor.VisibilityOn()
        else:
            self.above_cut_actor.VisibilityOff()
        
        # --- PART 3: CROSS-SECTION (LAYERED TETS) ---
        # Generate cross-section showing complete tetrahedra
        cross_section_poly = self._generate_layered_cross_section(origin, normal)
        
        if cross_section_poly.GetNumberOfCells() > 0:
            # Create mapper for cross-section
            cs_mapper = vtk.vtkPolyDataMapper()
            cs_mapper.SetInputData(cross_section_poly)
            
            # CRITICAL: Use depth offset to render cross-section ABOVE everything else
            # Negative offset shifts toward camera, eliminating z-fighting
            cs_mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-1, -1)
            cs_mapper.SetRelativeCoincidentTopologyLineOffsetParameters(-1, -1)
            
            # Create actor if it doesn't exist
            if not self.cross_section_actor:
                self.cross_section_actor = vtk.vtkActor()
                self.renderer.AddActor(self.cross_section_actor)
            
            self.cross_section_actor.SetMapper(cs_mapper)
            
            # Check if we have quality colors
            has_quality_colors = (cross_section_poly.GetCellData().GetScalars() is not None)
            
            if has_quality_colors:
                # Use quality colors
                cs_mapper.SetScalarModeToUseCellData()
                cs_mapper.ScalarVisibilityOn()
                cs_mapper.SetColorModeToDirectScalars()  # RGB 0-255 values
                print(f"[DEBUG] Layered mode: using quality colors")
            else:
                # Fallback to solid red if no quality data
                self.cross_section_actor.GetProperty().SetColor(0.9, 0.3, 0.3)
                cs_mapper.ScalarVisibilityOff()
                print(f"[DEBUG] Layered mode: no quality data, using solid red")
            
            # Full opacity and BRIGHT lighting for cross-section tets
            self.cross_section_actor.GetProperty().SetOpacity(1.0)  
            self.cross_section_actor.GetProperty().SetLighting(True)
            
            # CRITICAL: Match surface mesh lighting for equal brightness
            # Increase ambient to reduce shadow darkness
            self.cross_section_actor.GetProperty().SetAmbient(0.6)  # Higher ambient = brighter
            self.cross_section_actor.GetProperty().SetDiffuse(0.8)  # Strong diffuse
            self.cross_section_actor.GetProperty().SetSpecular(0.2)  # Some shine
            
            # Edge visibility
            self.cross_section_actor.GetProperty().EdgeVisibilityOn()
            self.cross_section_actor.GetProperty().SetEdgeColor(0.1, 0.1, 0.1)
            self.cross_section_actor.GetProperty().SetLineWidth(1.0)
            
            self.cross_section_actor.VisibilityOn()
            print(f"[DEBUG] Cross-section: {cross_section_poly.GetNumberOfCells()} triangles")
        else:
            if self.cross_section_actor:
                self.cross_section_actor.VisibilityOff()
        
        print(f"[DEBUG] Below cut: 100% opacity, Above cut: 5% opacity (ghost)")
        
        self.vtk_widget.GetRenderWindow().Render()
    
    def _remove_clipping(self):
        """Remove clipping and restore original mesh"""
        if not self.current_poly_data or not self.current_actor:
            return
        
        # Restore original unclipped data
        mapper = self.current_actor.GetMapper()
        mapper.SetInputData(self.current_poly_data)
        self.current_actor.GetProperty().SetOpacity(1.0)  # Restore full opacity
        
        # Hide above-cut actor
        if hasattr(self, 'above_cut_actor') and self.above_cut_actor:
            self.above_cut_actor.VisibilityOff()
        
        # Hide cross-section actor
        if self.cross_section_actor:
            self.cross_section_actor.VisibilityOff()
        
        self.vtk_widget.GetRenderWindow().Render()

    def update_quality_visualization(self, metric="SICN (Min)", opacity=1.0, min_val=0.0, max_val=1.0):
        """
        Update visualization based on selected quality metric and filters.
        Rebuilds the displayed mesh to hide elements outside the range.
        """
        if not self.current_poly_data or not self.current_actor:
            return
            
        # Update opacity
        self.current_actor.GetProperty().SetOpacity(opacity)
        
        # If no quality data, just return
        if not hasattr(self, 'current_quality_data') or not self.current_quality_data:
            self.vtk_widget.GetRenderWindow().Render()
            return

        # Determine which data to use
        quality_map = {}
        metric_key = 'per_element_quality' # Default (SICN)
        
        if "Gamma" in metric:
            metric_key = 'per_element_gamma'
        elif "Skewness" in metric:
            metric_key = 'per_element_skewness'
        elif "Aspect" in metric:
            metric_key = 'per_element_aspect_ratio'
            
        if metric_key in self.current_quality_data:
            quality_map = self.current_quality_data[metric_key]
        elif metric != "SICN (Min)":
            print(f"[WARN] Metric '{metric}' ({metric_key}) not found, defaulting to SICN")
            quality_map = self.current_quality_data.get('per_element_quality', {})
            
        if not quality_map:
            return

        # Create new PolyData for filtered mesh
        filtered_poly_data = vtk.vtkPolyData()
        filtered_poly_data.SetPoints(self.current_poly_data.GetPoints()) # Share points
        
        # New cell array and colors
        new_cells = vtk.vtkCellArray()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("Colors")
        
        # Helper for HSL to RGB
        def hsl_to_rgb(h, s, l):
            def hue_to_rgb(p, q, t):
                if t < 0: t += 1
                if t > 1: t -= 1
                if t < 1/6: return p + (q - p) * 6 * t
                if t < 1/2: return q
                if t < 2/3: return p + (q - p) * (2/3 - t) * 6
                return p
            if s == 0: r = g = b = l
            else:
                q = l * (1 + s) if l < 0.5 else l + s - l * s
                p = 2 * l - q
                r = hue_to_rgb(p, q, h + 1/3)
                g = hue_to_rgb(p, q, h)
                b = hue_to_rgb(p, q, h - 1/3)
            return int(r * 255), int(g * 255), int(b * 255)

        # Iterate through ORIGINAL elements to filter
        if not self.current_mesh_elements:
            return
            
        visible_count = 0
        filtered_count = 0
        
        # Get global range for this metric for consistent coloring
        all_vals = [float(v) for v in quality_map.values()]
        if all_vals:
            global_min, global_max = min(all_vals), max(all_vals)
            val_range = global_max - global_min if global_max > global_min else 1.0
        else:
            global_min, global_max, val_range = 0.0, 1.0, 1.0

        # Rebuild cells
        # Note: self.current_mesh_elements contains ALL elements (tets + tris)
        # We only want to display triangles (surface)
        # But for "seeing inside", we might want to display tets? 
        # No, usually we display the surface triangles. 
        # If the user wants to see "internal tets", we would need to extract faces of internal tets.
        # For now, let's stick to filtering the SURFACE mesh.
        # If the surface triangle is hidden, we see through it.
        
        # We need to map from element ID to VTK cell index? 
        # No, we just rebuild the vtkCellArray from scratch based on visible elements.
        
        for element in self.current_mesh_elements:
            if element['type'] == 'triangle':
                eid = str(element['id'])
                # Try int key too
                val = quality_map.get(eid)
                if val is None:
                    val = quality_map.get(int(eid))
                
                if val is not None:
                    if min_val <= val <= max_val:
                        # Visible! Add cell and color
                        
                        # Get node indices for this triangle
                        node_ids = element['nodes']
                        vtk_ids = [self.current_node_map[nid] for nid in node_ids]
                        
                        tri = vtk.vtkTriangle()
                        for i, vid in enumerate(vtk_ids):
                            tri.GetPointIds().SetId(i, vid)
                        new_cells.InsertNextCell(tri)
                        
                        # Calculate color
                        # Normalize to 0-1 based on global range
                        norm = (val - global_min) / val_range
                        norm = max(0.0, min(1.0, norm))
                        
                        # Color map: Red (0.0) -> Green (0.33)
                        # For Skewness/Aspect Ratio, lower is better (Green), higher is worse (Red)
                        # For SICN/Gamma, higher is better (Green), lower is worse (Red)
                        
                        if "Skewness" in metric or "Aspect" in metric:
                            # 0 (Good/Green) -> 1 (Bad/Red)
                            # Invert norm so 0->Green, 1->Red
                            hue = (1.0 - norm) * 0.33
                        else:
                            # 0 (Bad/Red) -> 1 (Good/Green)
                            hue = norm * 0.33
                            
                        r, g, b = hsl_to_rgb(hue, 1.0, 0.5)
                        colors.InsertNextTuple3(r, g, b)
                        visible_count += 1
                    else:
                        filtered_count += 1

        filtered_poly_data.SetPolys(new_cells)
        filtered_poly_data.GetCellData().SetScalars(colors)
        
        # Update mapper
        mapper = self.current_actor.GetMapper()
        mapper.SetInputData(filtered_poly_data)
        mapper.ScalarVisibilityOn() # CRITICAL: Ensure colors are shown
        mapper.SetScalarModeToUseCellData()
        mapper.SetColorModeToDirectScalars() # Use RGB values directly
        
        self.vtk_widget.GetRenderWindow().Render()
        print(f"[DEBUG] Updated visualization: {visible_count} visible, {filtered_count} filtered")
        print(f"[DEBUG] Metric: {metric}, Range: {min_val}-{max_val}, Opacity: {opacity}")

    def create_brush_cursor(self, radius: float):
        """Create or update brush cursor sphere"""
        print(f"[DEBUG] Creating brush cursor with radius: {radius}")

        # Remove old cursor if exists
        if self.brush_cursor_actor:
            self.renderer.RemoveActor(self.brush_cursor_actor)

        # Create sphere for brush cursor
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(radius)
        sphere.SetThetaResolution(20)
        sphere.SetPhiResolution(20)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())

        self.brush_cursor_actor = vtk.vtkActor()
        self.brush_cursor_actor.SetMapper(mapper)
        # Bright cyan color for better visibility
        self.brush_cursor_actor.GetProperty().SetColor(0.0, 1.0, 1.0)  # Cyan
        self.brush_cursor_actor.GetProperty().SetOpacity(0.5)
        self.brush_cursor_actor.GetProperty().SetRepresentationToWireframe()
        self.brush_cursor_actor.GetProperty().SetLineWidth(3)
        # Render as tubes for better visibility
        self.brush_cursor_actor.GetProperty().SetRenderLinesAsTubes(True)

        # Initially hidden
        self.brush_cursor_actor.VisibilityOff()
        self.renderer.AddActor(self.brush_cursor_actor)
        self.brush_cursor_visible = False
        print(f"[DEBUG] Brush cursor actor created and added to renderer")

    def update_brush_cursor_position(self, x: int, y: int):
        """Update brush cursor position to follow mouse"""
        if not self.brush_cursor_actor:
            print("[DEBUG] No brush cursor actor to update")
            return

        # Use cell picker to pick actual surface points
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.005)  # Set picking tolerance

        if picker.Pick(x, y, 0, self.renderer):
            # Successfully picked a surface point
            point = picker.GetPickPosition()
            self.brush_cursor_actor.SetPosition(point)

            if not self.brush_cursor_visible:
                print(f"[DEBUG] Showing brush cursor at {point}")
                self.brush_cursor_actor.VisibilityOn()
                self.brush_cursor_visible = True

            self.vtk_widget.GetRenderWindow().Render()
        else:
            # No surface picked - hide cursor
            if self.brush_cursor_visible:
                self.brush_cursor_actor.VisibilityOff()
                self.brush_cursor_visible = False
                self.vtk_widget.GetRenderWindow().Render()

    def _get_cells_within_radius(self, center_point, radius):
        """Find all VTK cells within brush radius of center point"""
        cell_ids = []
        poly_data = self.current_poly_data

        if not poly_data:
            print("[DEBUG] No polydata available for cell selection")
            return cell_ids

        # Iterate through all cells and check distance
        num_cells = poly_data.GetNumberOfCells()
        radius_squared = radius * radius

        for i in range(num_cells):
            cell = poly_data.GetCell(i)
            # Get cell center from bounds
            bounds = cell.GetBounds()
            cell_center = [
                (bounds[0] + bounds[1]) / 2.0,
                (bounds[2] + bounds[3]) / 2.0,
                (bounds[4] + bounds[5]) / 2.0
            ]

            # Calculate squared distance (faster than sqrt)
            dist_squared = sum((cell_center[j] - center_point[j])**2 for j in range(3))

            if dist_squared <= radius_squared:
                cell_ids.append(i)

        print(f"[DEBUG] Found {len(cell_ids)} cells within radius {radius}")
        return cell_ids

    def mark_cells_as_painted(self, cell_ids, refinement_level):
        """Color mesh cells to show painted regions on surface"""
        if not self.current_poly_data or not cell_ids:
            print("[DEBUG] Cannot mark cells - no polydata or empty cell list")
            return

        print(f"[DEBUG] Marking {len(cell_ids)} cells as painted with refinement {refinement_level}x")

        # Update painted cells set
        self.painted_cells.update(cell_ids)

        # Initialize color array if needed
        num_cells = self.current_poly_data.GetNumberOfCells()
        if self.paint_colors.GetNumberOfTuples() != num_cells:
            print(f"[DEBUG] Initializing color array for {num_cells} cells")

            # Check if polydata already has colors (from mesh quality or previous operations)
            existing_colors = self.current_poly_data.GetCellData().GetScalars()

            self.paint_colors.SetNumberOfTuples(num_cells)

            if existing_colors and existing_colors.GetNumberOfTuples() == num_cells and existing_colors.GetNumberOfComponents() >= 3:
                # Preserve existing colors (e.g., mesh quality colors)
                print(f"[DEBUG] Preserving existing scalar colors from polydata")
                for i in range(num_cells):
                    if existing_colors.GetNumberOfComponents() == 3:
                        color = existing_colors.GetTuple3(i)
                        self.paint_colors.SetTuple3(i, int(color[0]), int(color[1]), int(color[2]))
                    else:
                        # If it's a different format, default to base color
                        self.paint_colors.SetTuple3(i, 76, 128, 204)
            else:
                # Initialize to base CAD color (light blue)
                print(f"[DEBUG] Initializing to base CAD color")
                for i in range(num_cells):
                    self.paint_colors.SetTuple3(i, 76, 128, 204)  # RGB(0.3, 0.5, 0.8)*255

        # Color newly painted cells with intensity based on refinement level
        # Higher refinement = more intense cyan color
        intensity = min(255, int(50 + refinement_level * 30))
        print(f"[DEBUG] Painting {len(cell_ids)} cells with intensity {intensity}")
        for cell_id in cell_ids:
            # Bright cyan/turquoise for painted areas
            self.paint_colors.SetTuple3(cell_id, 0, intensity, intensity)

        # Apply colors to mesh
        print(f"[DEBUG] Applying paint_colors array to polydata (has {self.paint_colors.GetNumberOfTuples()} tuples)")
        self.current_poly_data.GetCellData().SetScalars(self.paint_colors)

        # Verify scalar data was set
        check_scalars = self.current_poly_data.GetCellData().GetScalars()
        print(f"[DEBUG] Polydata now has scalars: {check_scalars is not None}, tuples: {check_scalars.GetNumberOfTuples() if check_scalars else 0}")

        # Update mapper to use cell colors
        mapper = self.current_actor.GetMapper()
        mapper.SetScalarModeToUseCellData()
        mapper.ScalarVisibilityOn()
        mapper.SetColorModeToDirectScalars()  # RGB 0-255 values

        print(f"[DEBUG] Mapper configured - ScalarVisibility: {mapper.GetScalarVisibility()}, ScalarMode: {mapper.GetScalarMode()}")
        print(f"[DEBUG] Surface coloring applied, {len(self.painted_cells)} total cells painted")

        # Force render update
        self.current_actor.Modified()
        mapper.Modified()
        self.vtk_widget.GetRenderWindow().Render()

    def clear_paint_markers(self):
        """Remove all paint markers and reset surface colors"""
        print("[DEBUG] Clearing all paint markers")
        self.painted_cells.clear()

        if self.current_poly_data and self.current_actor:
            # Remove scalar colors
            self.current_poly_data.GetCellData().SetScalars(None)

            # Reset mapper
            mapper = self.current_actor.GetMapper()
            mapper.ScalarVisibilityOff()

            # Reset to uniform CAD color (light blue)
            self.current_actor.GetProperty().SetColor(0.3, 0.5, 0.8)

            self.vtk_widget.GetRenderWindow().Render()

    def load_step_file(self, filepath: str):
        self.clear_view()
        self.quality_label.setVisible(False)
        self.info_label.setText(f"Loading: {Path(filepath).name}")

        try:
            import pyvista as pv
            
            # WINDOWS FIX 1: File Locking
            # We must create the path, but CLOSE the file immediately.
            # Windows prevents a subprocess from writing to a file open in the main process.
            tmp = tempfile.NamedTemporaryFile(suffix='.stl', delete=False)
            tmp_stl = tmp.name
            tmp.close() 

            # Tessellate CAD for preview display
            gmsh_script = f"""
import gmsh
import json
import sys

# Redirect stdout/stderr to ensure we capture errors
try:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1) # Enable terminal output for debugging
    gmsh.option.setNumber("General.Verbosity", 2) # Reduce spam, keep errors

    gmsh.open(r"{filepath}") # Use raw string for Windows paths

    # --- GEOMETRY ANALYSIS (Volume & Units) ---
    bbox = gmsh.model.getBoundingBox(-1, -1)
    bbox_dims = [bbox[3]-bbox[0], bbox[4]-bbox[1], bbox[5]-bbox[2]]

    volumes_3d = gmsh.model.getEntities(dim=3)
    total_volume_raw = 0.0
    for vol_dim, vol_tag in volumes_3d:
        total_volume_raw += gmsh.model.occ.getMass(vol_dim, vol_tag)

    bbox_volume_raw = bbox_dims[0] * bbox_dims[1] * bbox_dims[2]

    # Heuristic unit detection
    unit_name = "m"
    unit_scale = 1.0
    if total_volume_raw > 10000: # Likely mm^3
        unit_scale = 0.001
        unit_name = "mm"
    elif max(bbox_dims) > 1000: # Large dimensions
        unit_scale = 0.001
        unit_name = "mm"

    total_volume = total_volume_raw * (unit_scale ** 3)
    bbox_volume = bbox_volume_raw * (unit_scale ** 3)
    bbox_diag = (bbox_dims[0]**2 + bbox_dims[1]**2 + bbox_dims[2]**2)**0.5 * unit_scale

    if total_volume > bbox_volume or total_volume <= 0:
        total_volume = bbox_volume * 0.4

    geom_info = {{
        "volume": total_volume, 
        "bbox_diagonal": bbox_diag, 
        "units_detected": unit_name
    }}
    print("GEOM_INFO:" + json.dumps(geom_info))

    # --- TESSELLATION ---
    # Generate 2D mesh for visualization
    gmsh.model.mesh.generate(2)
    gmsh.write(r"{tmp_stl}")
    gmsh.finalize()
    print("SUCCESS_MARKER")

except Exception as e:
    print("GMSH_ERROR:" + str(e))
    sys.exit(1)
"""

            # WINDOWS FIX 2: Environment Variables
            # Conda environments rely on PATH to find DLLs. We must pass the current env.
            current_env = os.environ.copy()

            result = subprocess.run(
                [sys.executable, "-c", gmsh_script],
                capture_output=True,
                text=True,
                timeout=45,
                env=current_env  # CRITICAL: Passes DLL paths to subprocess
            )

            # Debugging Output
            if result.returncode != 0 or "SUCCESS_MARKER" not in result.stdout:
                print(f"--- GMSH SUBPROCESS FAILED ---")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                raise Exception(f"CAD conversion subprocess failed. See console for details.")

            # Parse geometry info
            geom_info = None
            for line in result.stdout.split('\n'):
                if line.startswith("GEOM_INFO:"):
                    geom_info = json.loads(line[10:])
                    break

            # Check if file actually exists and has content
            if not os.path.exists(tmp_stl) or os.path.getsize(tmp_stl) < 100:
                raise Exception("STL file was not created or is empty.")

            # Load into PyVista
            mesh = pv.read(tmp_stl)
            os.unlink(tmp_stl) # Cleanup temp file

            # --- VTK VISUALIZATION SETUP ---
            if mesh.n_points == 0:
                raise Exception("Empty mesh - no geometry in CAD file")

            poly_data = mesh.cast_to_unstructured_grid().extract_surface()
            self.current_poly_data = poly_data # Store for clipping

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(poly_data)

            self.current_actor = vtk.vtkActor()
            self.current_actor.SetMapper(mapper)
            
            # Styling: Smooth Blue CAD look
            self.current_actor.GetProperty().SetColor(0.3, 0.5, 0.8)
            self.current_actor.GetProperty().SetInterpolationToPhong()
            self.current_actor.GetProperty().EdgeVisibilityOff()
            self.current_actor.GetProperty().SetAmbient(0.3)
            self.current_actor.GetProperty().SetDiffuse(0.7)
            self.current_actor.GetProperty().SetSpecular(0.2)

            self.renderer.AddActor(self.current_actor)
            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()

            # Update Info Label
            volume_text = ""
            if geom_info and 'volume' in geom_info:
                v = geom_info['volume']
                if v > 0.001: 
                    volume_text = f"<br>Volume: {v:.4f} m³"
                else: 
                    volume_text = f"<br>Volume: {v*1e9:.0f} mm³"

            self.info_label.setText(
                f"<b>CAD Preview</b><br>"
                f"{Path(filepath).name}<br>"
                f"<span style='color: #6c757d;'>{poly_data.GetNumberOfPoints():,} nodes{volume_text}</span>"
            )

            return geom_info

        except Exception as e:
            # Fallback for errors
            print(f"Load Error: {e}")
            import traceback
            traceback.print_exc()
            self.info_label.setText(f"CAD Loaded<br><small>(Preview Unavailable)</small><br>Click 'Generate Mesh'")
            return None

    def load_mesh_file(self, filepath: str, result: dict = None):
        """Load and display mesh file with optional result dict for counts"""
        print(f"\n{'='*70}")
        print(f"[DEBUG] load_mesh_file called:")
        print(f"  filepath: {filepath}")
        print(f"  File exists: {Path(filepath).exists()}")
        if result:
            print(f"  result keys: {list(result.keys())}")
            if 'per_element_quality' in result:
                print(f"  [OK][OK]per_element_quality present: {len(result['per_element_quality'])} elements")
            else:
                print(f"  [X][X]per_element_quality NOT in result dict!")
            if 'quality_metrics' in result:
                print(f"  [OK][OK]quality_metrics present: {result['quality_metrics']}")
            else:
                print(f"  [X][X]quality_metrics NOT in result dict!")
        else:
            print(f"  [X][X]result dict is None!")
        print(f"{'='*70}")

        self.clear_view()
        self.info_label.setText("Loading mesh...")

        try:
            print(f"[DEBUG] Parsing .msh file...")
            nodes, elements = self._parse_msh_file(filepath)
            print(f"[DEBUG] Parsed {len(nodes)} nodes, {len(elements)} elements")

            # Try to load surface quality data if not provided
            if not (result and result.get('per_element_quality')):
                quality_file = Path(filepath).with_suffix('.quality.json')
                print(f"[DEBUG] Checking for quality file: {quality_file}")
                print(f"[DEBUG] Quality file exists: {quality_file.exists()}")
                if quality_file.exists():
                    print(f"[DEBUG] Loading surface quality data from {quality_file}")
                    try:
                        import json
                        with open(quality_file, 'r') as f:
                            surface_quality = json.load(f)

                        if not result:
                            result = {}
                        result['per_element_quality'] = surface_quality.get('per_element_quality', {})
                        
                        # Store full quality data for visualization switching
                        self.current_quality_data = surface_quality
                        
                        result['quality_metrics'] = {
                            'sicn_10_percentile': surface_quality.get('quality_threshold_10', 0.3),
                            'sicn_min': surface_quality.get('statistics', {}).get('min_quality', 0.0),
                            'sicn_avg': surface_quality.get('statistics', {}).get('avg_quality', 0.5),
                            'sicn_max': surface_quality.get('statistics', {}).get('max_quality', 1.0)
                        }
                        print(f"[DEBUG] [OK][OK]Loaded quality data for {len(result['per_element_quality'])} triangles")
                        print(f"[DEBUG] [OK][OK]Quality threshold: {result['quality_metrics']['sicn_10_percentile']:.3f}")
                    except Exception as e:
                        print(f"[DEBUG] [X][X]Could not load quality data: {e}")
                        import traceback
                        traceback.print_exc()
            else:
                print(f"[DEBUG] Quality data already in result dict")
                # Ensure we store it for visualization
                self.current_quality_data = result

            print(f"[DEBUG] Creating VTK data structures...")
            points = vtk.vtkPoints()
            cells = vtk.vtkCellArray()

            node_map = {}
            for idx, (node_id, coords) in enumerate(nodes.items()):
                points.InsertNextPoint(coords)
                node_map[node_id] = idx

            print(f"[DEBUG] Added {points.GetNumberOfPoints()} points to VTK")

            # Count elements for display
            tet_count = sum(1 for e in elements if e['type'] == 'tetrahedron')
            hex_count = sum(1 for e in elements if e['type'] == 'hexahedron')
            tri_count = sum(1 for e in elements if e['type'] == 'triangle')
            quad_count = sum(1 for e in elements if e['type'] == 'quadrilateral')

            print(f"[DEBUG] Element counts: {tet_count} tets, {hex_count} hexes, {tri_count} triangles, {quad_count} quads")

            # CRITICAL: Only visualize SURFACE TRIANGLES, not volume tetrahedra!
            # Include quadrilateral surfaces for hex meshes.
            for element in elements:
                if element['type'] == 'triangle':
                    tri = vtk.vtkTriangle()
                    for i, node_id in enumerate(element['nodes']):
                        tri.GetPointIds().SetId(i, node_map[node_id])
                    cells.InsertNextCell(tri)
                elif element['type'] == 'quadrilateral':
                    quad = vtk.vtkQuad()
                    for i, node_id in enumerate(element['nodes']):
                        quad.GetPointIds().SetId(i, node_map[node_id])
                    cells.InsertNextCell(quad)

            print(f"[DEBUG] Rendering {cells.GetNumberOfCells()} surface facets (triangles/quads only)")

            # Create PolyData directly from triangles (no geometry filter needed)
            poly_data = vtk.vtkPolyData()
            poly_data.SetPoints(points)
            poly_data.SetPolys(cells)

            # Store for cross-section clipping and hex visualization
            self.current_poly_data = poly_data
            self.current_mesh_nodes = nodes
            self.current_mesh_elements = elements
            self.current_node_map = node_map
            
            # Separate tetrahedra for cross-section computation
            self.current_tetrahedra = [e for e in elements if e['type'] == 'tetrahedron']
            self.current_hexahedra = [e for e in elements if e['type'] == 'hexahedron']
            
            # Build volumetric grid for 3D clipping
            if self.current_tetrahedra or self.current_hexahedra:
                self.current_volumetric_grid = vtk.vtkUnstructuredGrid()
                self.current_volumetric_grid.SetPoints(points)
                
                total_cells = 0
                for tet in self.current_tetrahedra:
                    vtk_tet = vtk.vtkTetra()
                    for i, nid in enumerate(tet['nodes']):
                        vtk_tet.GetPointIds().SetId(i, node_map[nid])
                    self.current_volumetric_grid.InsertNextCell(vtk_tet.GetCellType(), vtk_tet.GetPointIds())
                    total_cells += 1
                
                for hex_elem in self.current_hexahedra:
                    vtk_hex = vtk.vtkHexahedron()
                    for i, nid in enumerate(hex_elem['nodes']):
                        vtk_hex.GetPointIds().SetId(i, node_map[nid])
                    self.current_volumetric_grid.InsertNextCell(vtk_hex.GetCellType(), vtk_hex.GetPointIds())
                    total_cells += 1
                
                print(f"[DEBUG] Built volumetric grid with {total_cells} volume cells")
            else:
                self.current_volumetric_grid = None
            
            print(f"[DEBUG] PolyData created with {poly_data.GetNumberOfCells()} cells")
            print(f"[DEBUG] Stored {len(self.current_tetrahedra)} tetrahedra and {len(self.current_hexahedra)} hexahedra for cross-section visualization")
            
            # Reset cross-section element mode combo availability
            if hasattr(self, 'crosssection_cell_combo') and self.crosssection_cell_combo:
                model = self.crosssection_cell_combo.model()
                if model and hasattr(model, "item"):
                    # Index 0 = Auto, 1 = Tetrahedra, 2 = Hexahedra
                    if model.rowCount() >= 3:
                        model.item(1).setEnabled(bool(tet_count))
                        model.item(2).setEnabled(bool(hex_count))
                self.crosssection_cell_combo.blockSignals(True)
                self.crosssection_cell_combo.setCurrentText("Auto")
                self.crosssection_cell_combo.blockSignals(False)
            
            if hasattr(self, 'viewer') and self.viewer:
                self.viewer.set_cross_section_element_mode("auto")

            # Add per-cell colors based on quality (if available)
            print(f"[DEBUG] About to check quality coloring conditions...")
            print(f"[DEBUG]   result exists: {result is not None}")
            print(f"[DEBUG]   quality_metrics in result: {'quality_metrics' in result if result else False}")
            print(f"[DEBUG]   per_element_quality in result: {'per_element_quality' in result if result else False}")

            if result and result.get('quality_metrics') and result.get('per_element_quality'):
                try:
                    print(f"[DEBUG] ENTERING quality coloring block!")
                    per_elem_quality = result['per_element_quality']
                    threshold = result['quality_metrics'].get('sicn_10_percentile', 0.3)

                    print(f"[DEBUG] Per-element quality data found: {len(per_elem_quality)} elements")
                    print(f"[DEBUG] Quality threshold (10th percentile): {threshold:.3f}")
                    print(f"[DEBUG] Number of elements to iterate: {len(elements)}")
                    print(f"[DEBUG] Element types: {set(e['type'] for e in elements)}")
                    
                    surface_elements = [e for e in elements if e['type'] in ('triangle', 'quadrilateral')]
                    
                    # Show sample of quality keys vs element IDs
                    quality_keys_sample = list(per_elem_quality.keys())[:10]
                    element_ids_sample = [e['id'] for e in surface_elements][:10]
                    print(f"[DEBUG] Sample quality keys: {quality_keys_sample}")
                    print(f"[DEBUG] Sample surface IDs: {element_ids_sample}")
                    print(f"[DEBUG] Sample quality values: {[per_elem_quality.get(k, 'MISSING') for k in element_ids_sample[:5]]}")

                    # Calculate global quality range for color mapping
                    all_qualities = [q for q in per_elem_quality.values() if q is not None]
                    if all_qualities:
                        global_min = min(all_qualities)
                        global_max = max(all_qualities)
                        print(f"[DEBUG] Global quality range: {global_min:.3f} to {global_max:.3f}")
                    else:
                        global_min, global_max = 0.0, 1.0
                    
                    # Create color array for cells
                    colors = vtk.vtkUnsignedCharArray()
                    colors.SetNumberOfComponents(3)
                    colors.SetName("Colors")

                    # Helper function for HSL to RGB conversion
                    def hsl_to_rgb(h, s, l):
                        """Convert HSL to RGB (0-255)"""
                        def hue_to_rgb(p, q, t):
                            if t < 0: t += 1
                            if t > 1: t -= 1
                            if t < 1/6: return p + (q - p) * 6 * t
                            if t < 1/2: return q
                            if t < 2/3: return p + (q - p) * (2/3 - t) * 6
                            return p
                        
                        if s == 0:
                            r = g = b = l
                        else:
                            q = l * (1 + s) if l < 0.5 else l + s - l * s
                            p = 2 * l - q
                            r = hue_to_rgb(p, q, h + 1/3)
                            g = hue_to_rgb(p, q, h)
                            b = hue_to_rgb(p, q, h - 1/3)
                        
                        return int(r * 255), int(g * 255), int(b * 255)

                    # Color each surface cell with smooth gradient
                    for element in surface_elements:
                        elem_id = element['id']
                        quality = per_elem_quality.get(elem_id, per_elem_quality.get(str(elem_id), None))

                        if quality is None:
                            colors.InsertNextTuple3(150, 150, 150)
                        else:
                            quality_range = global_max - global_min
                            if quality_range > 0.0001:
                                normalized = (quality - global_min) / quality_range
                            else:
                                normalized = 1.0
                            
                            normalized = max(0.0, min(1.0, normalized))
                            hue = normalized * 0.33
                            r, g, b = hsl_to_rgb(hue, 1.0, 0.5)
                            colors.InsertNextTuple3(r, g, b)

                    poly_data.GetCellData().SetScalars(colors)
                    print(f"[DEBUG] [OK][OK]Applied smooth quality gradient colors")
                    print(f"[DEBUG] [OK][OK]Quality range: {global_min:.3f} (red) to {global_max:.3f} (green)")
                    print(f"[DEBUG] [OK][OK]Total colored surface cells: {len(surface_elements)}")

                    # Verify scalars were set
                    check_scalars = poly_data.GetCellData().GetScalars()
                    print(f"[DEBUG] [OK][OK]Scalars check after SetScalars: {check_scalars.GetNumberOfTuples() if check_scalars else 'NONE'}")

                except Exception as e:
                    print(f"[DEBUG ERROR] Could not apply quality colors: {e}")
                    import traceback
                    print(f"[DEBUG ERROR] Traceback:")
                    traceback.print_exc()
            else:
                print(f"[DEBUG] Quality coloring conditions NOT met - skipping color application")

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(poly_data)

            # Enable scalar coloring if quality colors were applied
            if result and result.get('per_element_quality') and poly_data.GetCellData().GetScalars():
                mapper.SetScalarModeToUseCellData()
                mapper.ScalarVisibilityOn()
                # CRITICAL: Tell VTK these are direct RGB colors (0-255), not scalar values
                mapper.SetColorModeToDirectScalars()
                print(f"[DEBUG] [OK][OK]Mapper configured for scalar coloring")
                print(f"[DEBUG] [OK][OK]ScalarVisibility: {mapper.GetScalarVisibility()}")
                print(f"[DEBUG] [OK][OK]ScalarMode: {mapper.GetScalarMode()}")
                print(f"[DEBUG] [OK][OK]ColorMode: DirectScalars (RGB 0-255)")
            else:
                mapper.ScalarVisibilityOff()
                print(f"[DEBUG] Scalar coloring disabled")

            self.current_actor = vtk.vtkActor()
            self.current_actor.SetMapper(mapper)

            # Set surface appearance - SOLID mesh visualization
            if not (result and result.get('per_element_quality')):
                # Only set uniform color if no per-element coloring
                self.current_actor.GetProperty().SetColor(0.2, 0.7, 0.4)  # Green

            self.current_actor.GetProperty().SetOpacity(1.0)  # Fully opaque

            # Use FLAT shading to show individual triangular facets
            # This makes the mesh structure visible instead of looking like smooth CAD
            self.current_actor.GetProperty().SetInterpolationToFlat()

            # Set material properties based on whether we have quality coloring
            if result and result.get('per_element_quality'):
                # Quality coloring: use high ambient to see colors clearly
                self.current_actor.GetProperty().SetAmbient(0.8)  # High ambient = bright colors
                self.current_actor.GetProperty().SetDiffuse(0.5)  # Lower diffuse
                self.current_actor.GetProperty().SetSpecular(0.0)  # No specular highlights
                print(f"[DEBUG] [OK][OK]Using high ambient lighting for quality visualization")
            else:
                # Normal mesh: balanced lighting
                self.current_actor.GetProperty().SetAmbient(0.4)  # Reflects ambient light
                self.current_actor.GetProperty().SetDiffuse(0.7)  # Main surface color
                self.current_actor.GetProperty().SetSpecular(0.2)  # Reduced specular for softer highlights
                self.current_actor.GetProperty().SetSpecularPower(15)  # Softer highlights

            # ALWAYS show mesh edges to distinguish mesh from CAD
            self.current_actor.GetProperty().EdgeVisibilityOn()
            self.current_actor.GetProperty().SetEdgeColor(0.0, 0.0, 0.0)  # Black edges
            self.current_actor.GetProperty().SetLineWidth(1.0)  # Visible edge width

            print(f"[DEBUG] Actor created, adding to renderer...")
            self.renderer.AddActor(self.current_actor)
            print(f"[DEBUG] Number of actors in renderer: {self.renderer.GetActors().GetNumberOfItems()}")

            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()
            print(f"[DEBUG] Render complete!")

            # Use result dict counts if available, otherwise parsed counts
            if result:
                display_nodes = result.get('total_nodes', len(nodes))
                display_elements = result.get('total_elements', len(elements))
            else:
                display_nodes = len(nodes)
                display_elements = len(elements)

            # Build info text with quality metrics if available
            info_lines = [
                f"<b>Mesh Generated</b><br>",
                f"{Path(filepath).name}<br>",
                f"<span style='color: #6c757d;'>",
                f"Nodes: {display_nodes:,} * Elements: {display_elements:,}<br>",
                f"Tetrahedra: {tet_count:,} * Triangles: {tri_count:,}"
            ]

            # Add quality metrics if available in result
            if result and result.get('quality_metrics'):
                metrics = result['quality_metrics']
                info_lines.append("<br><b>Quality Metrics (avg):</b><br>")

                # SICN (primary gmsh metric) - use AVERAGE not min
                if 'sicn_avg' in metrics:
                    sicn = metrics['sicn_avg']
                    sicn_color = "#198754" if sicn >= 0.7 else "#ffc107" if sicn >= 0.5 else "#dc3545"
                    info_lines.append(f"<span style='color: {sicn_color};'>SICN: {sicn:.3f}</span> ")

                # Gamma - use AVERAGE not min
                if 'gamma_avg' in metrics:
                    gamma = metrics['gamma_avg']
                    gamma_color = "#198754" if gamma >= 0.6 else "#ffc107" if gamma >= 0.4 else "#dc3545"
                    info_lines.append(f"<span style='color: {gamma_color};'>γ: {gamma:.3f}</span><br>")

                # Skewness - use AVERAGE not max
                if 'avg_skewness' in metrics:
                    skew = metrics['avg_skewness']
                    skew_color = "#198754" if skew <= 0.3 else "#ffc107" if skew <= 0.5 else "#dc3545"
                    info_lines.append(f"<span style='color: {skew_color};'>Skew: {skew:.3f}</span> ")

                # Aspect Ratio - use AVERAGE not max
                if 'avg_aspect_ratio' in metrics:
                    ar = metrics['avg_aspect_ratio']
                    ar_color = "#198754" if ar <= 2.0 else "#ffc107" if ar <= 3.0 else "#dc3545"
                    info_lines.append(f"<span style='color: {ar_color};'>AR: {ar:.2f}</span>")

            info_lines.append("</span>")
            info_text = "".join(info_lines)
            self.info_label.setText(info_text)
            self.info_label.adjustSize()  # Force label to resize to fit content
            print(f"[DEBUG] Info label updated: {info_text}")
            print(f"[DEBUG] load_mesh_file completed successfully!")
            return "SUCCESS"

        except Exception as e:
            error_msg = f"Error loading mesh: {str(e)}"
            print(f"[DEBUG ERROR] {error_msg}")
            import traceback
            traceback.print_exc()
            self.info_label.setText(error_msg)
            return f"ERROR: {e}"

    def _parse_msh_file(self, filepath: str):
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

                        # Handle linear tetrahedra (4-node)
                        if element_type == 4:
                            for _ in range(num_elements):
                                data = lines[i].strip().split()
                                if len(data) >= 5:
                                    elements.append({
                                        "id": int(data[0]),
                                        "type": "tetrahedron",
                                        "nodes": [int(data[1]), int(data[2]), int(data[3]), int(data[4])]
                                    })
                                i += 1
                        # Handle quadratic tetrahedra (10-node) - use first 4 corner nodes
                        elif element_type == 11:
                            for _ in range(num_elements):
                                data = lines[i].strip().split()
                                if len(data) >= 11:  # 10 nodes + element tag
                                    elements.append({
                                        "id": int(data[0]),
                                        "type": "tetrahedron",
                                        "nodes": [int(data[1]), int(data[2]), int(data[3]), int(data[4])]
                                    })
                                i += 1
                        # Handle linear hexahedra (8-node)
                        elif element_type == 5:
                            for _ in range(num_elements):
                                data = lines[i].strip().split()
                                if len(data) >= 9:
                                    elements.append({
                                        "id": int(data[0]),
                                        "type": "hexahedron",
                                        "nodes": [int(data[j]) for j in range(1, 9)]
                                    })
                                i += 1
                        # Handle quadratic hexahedra (20-node) - use first 8 corner nodes
                        elif element_type == 12:
                            for _ in range(num_elements):
                                data = lines[i].strip().split()
                                if len(data) >= 13:
                                    elements.append({
                                        "id": int(data[0]),
                                        "type": "hexahedron",
                                        "nodes": [int(data[j]) for j in range(1, 9)]
                                    })
                                i += 1
                        # Handle linear triangles (3-node)
                        elif element_type == 2:
                            for _ in range(num_elements):
                                data = lines[i].strip().split()
                                if len(data) >= 4:
                                    elements.append({
                                        "id": int(data[0]),
                                        "type": "triangle",
                                        "nodes": [int(data[1]), int(data[2]), int(data[3])]
                                    })
                                i += 1
                        # Handle quadratic triangles (6-node) - use first 3 corner nodes
                        elif element_type == 9:
                            for _ in range(num_elements):
                                data = lines[i].strip().split()
                                if len(data) >= 7:  # 6 nodes + element tag
                                    elements.append({
                                        "id": int(data[0]),
                                        "type": "triangle",
                                        "nodes": [int(data[1]), int(data[2]), int(data[3])]
                                    })
                                i += 1
                        # Handle linear quads (4-node)
                        elif element_type == 3:
                            for _ in range(num_elements):
                                data = lines[i].strip().split()
                                if len(data) >= 5:
                                    elements.append({
                                        "id": int(data[0]),
                                        "type": "quadrilateral",
                                        "nodes": [int(data[1]), int(data[2]), int(data[3]), int(data[4])]
                                    })
                                i += 1
                        # Handle quadratic quads (8-node) - use first 4 corners
                        elif element_type == 10:
                            for _ in range(num_elements):
                                data = lines[i].strip().split()
                                if len(data) >= 9:
                                    elements.append({
                                        "id": int(data[0]),
                                        "type": "quadrilateral",
                                        "nodes": [int(data[1]), int(data[2]), int(data[3]), int(data[4])]
                                    })
                                i += 1
                        else:
                            # Skip other element types (edges, points, etc.)
                            for _ in range(num_elements):
                                i += 1
                    else:
                        i += 1
            else:
                i += 1

        return nodes, elements


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
            "Hex Dominant (Subdivision)"
        ])
        self.mesh_strategy.setCurrentIndex(0)  # Default to Delaunay
        self.mesh_strategy.setToolTip(
            "Tetrahedral (Delaunay): Robust conformal tet mesh\n"
            "Hex Dominant (Subdivision): 100% hex mesh via CoACD + subdivision (4x elements)"
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
        layout.addWidget(crosssection_group)

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
        layout.addWidget(viz_group)

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
        progress_layout.setSpacing(3)  # More compact spacing

        phases = [
            ("strategy", "Strategy"),
            ("1d", "1D"),
            ("2d", "2D"),
            ("3d", "3D"),
            ("opt", "Optimize"),
            ("netgen", "Netgen"),
            ("order2", "Order 2"),
            ("quality", "Quality")
        ]

        for phase_id, phase_name in phases:
            phase_label = QLabel(phase_name)
            phase_label.setStyleSheet("font-size: 8px; color: #495057; font-weight: 600;")
            progress_layout.addWidget(phase_label)

            # Store label reference and base name for animation
            self.phase_labels[phase_id] = phase_label
            self.phase_base_names[phase_id] = phase_name

            phase_bar = QProgressBar()
            phase_bar.setStyleSheet("""
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
            phase_bar.setMaximum(100)
            phase_bar.setValue(0)
            phase_bar.setFormat("%p%")  # Only show percentage
            progress_layout.addWidget(phase_bar)

            self.phase_bars[phase_id] = phase_bar

        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        layout.addStretch()

        # Secret button at the bottom - converts tet mesh display to hex-like
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
        layout.addWidget(self.hex_mode_btn, alignment=Qt.AlignLeft)

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

        # Reset progress bars to initial state
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

    def start_mesh_generation(self):
        if not self.cad_file:
            return

        self.generate_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)  # Enable stop button
        self.console.clear()

        for bar in self.phase_bars.values():
            bar.setValue(0)
            bar.setStyleSheet(bar.styleSheet().replace("background-color: #198754", "background-color: #0d6efd"))

        # Collect quality parameters from GUI
        quality_params = {
            "quality_preset": self.quality_preset.currentText(),
            "target_elements": self.target_elements.value(),
            "max_size_mm": self.max_size.value(),
            "curvature_adaptive": self.curvature_adaptive.isChecked(),
            "mesh_strategy": self.mesh_strategy.currentText(),
            "save_stl": self.save_stl.isChecked()  # Export intermediate STL files
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
        """Update progress bar and animate phase label"""
        if phase in self.phase_bars:
            bar = self.phase_bars[phase]
            bar.setValue(percentage)

            # Start animation if this is a new active phase
            if self.active_phase != phase:
                self.active_phase = phase
                self.dot_count = 0
                if not self.animation_timer.isActive():
                    self.animation_timer.start()

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

                self.add_log(f"[DEBUG] Calling viewer.load_mesh_file...")
                load_result = self.viewer.load_mesh_file(self.mesh_file, result)  # Pass result dict!
                self.add_log(f"[DEBUG] load_mesh_file returned: {load_result}")

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
