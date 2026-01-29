"""
VTK 3D Viewer
==============

3D mesh visualization with quality overlay, cross-sections, and paintbrush support.
"""

import vtk
import numpy as np
import logging
import tempfile
import subprocess
import sys
import os
import json
from pathlib import Path
from typing import Dict, Optional
from PyQt5.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel,
    QSizePolicy, QWidget, QSlider, QSpinBox,
    QPushButton, QCheckBox, QComboBox, QShortcut
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QEvent
from PyQt5.QtGui import QFont, QColor, QKeySequence
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from .interactor import CustomInteractorStyle
from .utils import hsl_to_rgb
from core.zone_manager import FluentZoneManager


class HQMeshWorker(QThread):
    """Background worker for high-quality visualization mesh generation"""
    finished = pyqtSignal(str, dict, str) # stl_path, geom_info, original_cad_path

    def __init__(self, cad_path: str):
        super().__init__()
        self.cad_path = cad_path
        self.process = None
        
    def stop(self):
        """Force kill the worker and its subprocess"""
        if self.process:
            try:
                self.process.kill()
            except:
                pass
        self.quit()
        self.wait() # Quickly clean up thread resources

    def run(self):
        try:
            # Create temp file for HQ mesh
            fd, tmp_stl = tempfile.mkstemp(suffix=".stl")
            os.close(fd)
            
            # --- HQ TESSELLATION SETTINGS ---
            # Using the "User Approved" high-fidelity settings
            gmsh_script = f"""
import gmsh
import json
import sys
import os

try:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("General.Verbosity", 1) # Less spam

    # Open config
    gmsh.open(r"{self.cad_path}")

    # --- GEOMETRY ANALYSIS (Fast, BBox only) ---
    bbox = gmsh.model.getBoundingBox(-1, -1)
    bbox_dims = [bbox[3]-bbox[0], bbox[4]-bbox[1], bbox[5]-bbox[2]]
    bbox_diag = (bbox_dims[0]**2 + bbox_dims[1]**2 + bbox_dims[2]**2)**0.5
    
    # Estimate volume from bbox to avoid OCC getMass hang
    bbox_volume = bbox_dims[0] * bbox_dims[1] * bbox_dims[2]
        
    # Heuristic unit detection
    # If it's big (> 10), it's likely mm. If it's tiny (< 1), it's likely m.
    unit_name = "m"
    if max(bbox_dims) > 10 or bbox_volume > 1000:
        unit_name = "mm"
        
    geom_info = {{
        "volume": bbox_volume * 0.5, # Rough estimate (50% fill)
        "bbox_diagonal": bbox_diag,
        "units_detected": unit_name
    }}
    print("GEOM_INFO:" + json.dumps(geom_info))

    # --- HIGH QUALITY TESSELLATION ---
    # "Perfect Circle" Logic using Curvature
    # 100 nodes per full circle = 3.6 degrees per segment.
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 100) 
    
    # Constrain max element size based on scale
    gmsh.option.setNumber("Mesh.MeshSizeMax", bbox_diag / 20.0)
    
    # Generate 2D mesh
    gmsh.model.mesh.generate(2)
    gmsh.write(r"{tmp_stl}")
    gmsh.finalize()
    print("SUCCESS_MARKER")

except Exception as e:
    print("GMSH_ERROR:" + str(e))
    sys.exit(1)
"""
            # Run subprocess
            current_env = os.environ.copy()
            
            # Use same python executable
            # Use same python executable
            # Use Popen to allow killing
            self.process = subprocess.Popen(
                [sys.executable, "-c", gmsh_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=current_env
            )
            
            # Wait for result (blocking this thread only, not UI)
            stdout, stderr = self.process.communicate(timeout=180)
            
            if self.process.returncode != 0 or "SUCCESS_MARKER" not in stdout:
                print(f"[HQ WORKER] Failed: {stderr}")
                return

            # Parse info
            geom_info = {}
            for line in stdout.splitlines():
                if line.startswith("GEOM_INFO:"):
                    geom_info = json.loads(line[10:])
                    break
            
            self.finished.emit(tmp_stl, geom_info, self.cad_path)
            
        except Exception as e:
            print(f"[HQ WORKER] Exception: {e}")


class LoadingOverlay(QWidget):
    """Semi-transparent loading overlay with spinner"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setStyleSheet("""
            QWidget {
                background-color: rgba(0, 0, 0, 0.6);
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        
        # Spinner label (larger and more visible)
        self.spinner_label = QLabel("◐", self)
        self.spinner_label.setStyleSheet("""
            QLabel {
                color: #4A9EFF;
                font-size: 64px;
                background-color: transparent;
                font-weight: bold;
            }
        """)
        self.spinner_label.setAlignment(Qt.AlignCenter)
        
        # Loading text
        self.text_label = QLabel("Loading CAD file...", self)
        self.text_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 18px;
                background-color: transparent;
                padding: 10px;
                font-weight: bold;
            }
        """)
        self.text_label.setAlignment(Qt.AlignCenter)
        
        # Subtitle for time estimates (hidden by default)
        self.subtitle_label = QLabel("", self)
        self.subtitle_label.setStyleSheet("""
            QLabel {
                color: #CCCCCC;
                font-size: 14px;
                background-color: transparent;
                padding: 5px;
            }
        """)
        self.subtitle_label.setAlignment(Qt.AlignCenter)
        self.subtitle_label.hide()
        
        layout.addWidget(self.spinner_label)
        layout.addWidget(self.text_label)
        layout.addWidget(self.subtitle_label)
        
        # Animation timer
        self.frame = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        
    def set_message(self, message: str, subtitle: str = ""):
        """Update loading message and optional subtitle"""
        self.text_label.setText(message)
        if subtitle:
            self.subtitle_label.setText(subtitle)
            self.subtitle_label.show()
        else:
            self.subtitle_label.hide()
        
    def show(self):
        """Show overlay and start animation"""
        super().show()
        self.frame = 0
        self.timer.start(100)  # 100ms = ~10 fps (smooth enough, not too fast)
        
    def hide(self):
        """Hide overlay and stop animation"""
        self.timer.stop()
        self.subtitle_label.hide()
        super().hide()
        
    def animate(self):
        """Rotate spinner with better visual feedback"""
        # 8-frame animation cycle for smooth rotation
        chars = ['◐', '◓', '◑', '◒', '◐', '◓', '◑', '◒']
        self.frame = (self.frame + 1) % len(chars)
        self.spinner_label.setText(chars[self.frame])


class CadLoadWorker(QThread):
    """Background worker for CAD file loading"""
    finished = pyqtSignal(object, object)  # poly_data, geom_info
    error = pyqtSignal(str)  # error_message
    
    def __init__(self, filepath: str):
        super().__init__()
        self.filepath = filepath
        self._is_running = True
        
    def stop(self):
        """Request worker to stop"""
        self._is_running = False
        
    def run(self):
        """Execute CAD loading in background thread"""
        try:
            import pyvista as pv
            import gmsh
            import tempfile
            
            if not self._is_running:
                return
            
            # CRITICAL: Ensure gmsh is not already initialized
            # This can happen if paintbrush or other code opened gmsh before us
            try:
                if gmsh.is_initialized():
                    print("[CAD WORKER] WARNING: gmsh already initialized, finalizing first...")
                    gmsh.finalize()
            except:
                pass  # Ignore errors during cleanup
            
            # Create temp file path (close immediately so gmsh can write to it)
            tmp = tempfile.NamedTemporaryFile(suffix='.stl', delete=False)
            tmp_stl = tmp.name
            tmp.close()

            # --- IN-PROCESS GMSH CONVERSION ---
            print("[CAD WORKER] Starting in-process Gmsh conversion...")
            
            # Ensure fresh start
            gmsh.initialize()
            
            # Silent mode
            gmsh.option.setNumber("General.Terminal", 1) 
            gmsh.option.setNumber("General.Verbosity", 2)
            
            if not self._is_running:
                gmsh.finalize()
                return

            try:
                gmsh.open(self.filepath)

                # --- FAST GEOMETRY ANALYSIS ---
                bbox = gmsh.model.getBoundingBox(-1, -1)
                bbox_dims = [bbox[3]-bbox[0], bbox[4]-bbox[1], bbox[5]-bbox[2]]
                bbox_diag = (bbox_dims[0]**2 + bbox_dims[1]**2 + bbox_dims[2]**2)**0.5
                
                print(f"[CAD WORKER] BBox Diag: {bbox_diag}")

                # --- FAST TESSELLATION SETTINGS ---
                # Basic Coarse Constraints
                gmsh.option.setNumber("Mesh.MeshSizeMin", bbox_diag / 100.0)
                gmsh.option.setNumber("Mesh.MeshSizeMax", bbox_diag / 20.0)
                
                # Disable curvature for speed
                gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
                gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
                
                if not self._is_running:
                    gmsh.finalize()
                    return
                
                # Generate 2D mesh
                gmsh.model.mesh.generate(2)
                
                # Write to temp STL
                gmsh.write(tmp_stl)
                
            except Exception as e:
                print(f"[CAD WORKER ERROR] Gmsh operations failed: {e}")
                gmsh.finalize()
                self.error.emit(str(e))
                return
            finally:
                if gmsh.is_initialized():
                    gmsh.finalize()

            # Check result
            if not os.path.exists(tmp_stl) or os.path.getsize(tmp_stl) < 100:
                self.error.emit("STL file was not created or is empty.")
                return

            if not self._is_running:
                try:
                    os.unlink(tmp_stl)
                except:
                    pass
                return

            # Load into PyVista
            mesh = pv.read(tmp_stl)
            try:
                os.unlink(tmp_stl)  # Cleanup temp file
            except:
                pass

            print(f"[CAD WORKER] STL loaded: {mesh.n_points} points, {mesh.n_cells} cells", flush=True)

            # --- VTK DATA PREPARATION ---
            if mesh.n_points == 0:
                self.error.emit("Empty mesh - no geometry in CAD file")
                return

            poly_data = mesh.cast_to_unstructured_grid().extract_surface()
            print(f"[CAD WORKER] Surface extracted: {poly_data.GetNumberOfPoints()} pts, {poly_data.GetNumberOfCells()} cells", flush=True)

            # --- IMPROVED CAD VISUALIZATION PIPELINE ---
            try:
                # 0. Subdivision (The "Quality Boost")
                processing_poly_data = poly_data
                cell_count = poly_data.GetNumberOfCells()
                
                # Only apply light subdivision for the preview to keep it fast
                if cell_count < 10000:
                    print(f"[CAD WORKER] Applying Fast Subdivision to {cell_count} cells...")
                    subdivision = vtk.vtkLinearSubdivisionFilter()
                    subdivision.SetInputData(poly_data)
                    subdivision.SetNumberOfSubdivisions(1)
                    subdivision.Update()
                    processing_poly_data = subdivision.GetOutput()
                
                # 1. Generate Smooth Normals 
                print("[CAD WORKER] Generating smooth normals...")
                normals_gen = vtk.vtkPolyDataNormals()
                normals_gen.SetInputData(processing_poly_data)
                normals_gen.ComputePointNormalsOn()
                normals_gen.ComputeCellNormalsOff()
                normals_gen.SplittingOn() 
                normals_gen.SetFeatureAngle(60.0)
                normals_gen.Update()
                smooth_poly_data = normals_gen.GetOutput()
                
            except Exception as e:
                print(f"[CAD WORKER ERROR] Pipeline failed: {e}. Using basic data.")
                import traceback
                traceback.print_exc()
                smooth_poly_data = poly_data

            # Accurate Volume Calculation via VTK
            mass = vtk.vtkMassProperties()
            mass.SetInputData(smooth_poly_data)
            mass.Update()
            volume = mass.GetVolume()
            
            # Unit Heuristic: If bounds > 5.0, assume mm
            bounds = smooth_poly_data.GetBounds()
            dim_x = bounds[1]-bounds[0]
            dim_y = bounds[3]-bounds[2]
            dim_z = bounds[5]-bounds[4]
            max_dim = max(dim_x, dim_y, dim_z)
            bbox_diag = (dim_x**2 + dim_y**2 + dim_z**2)**0.5
            
            unit_scale = 1.0
            unit_name = 'm'
            
            if max_dim > 5.0:
                 # Likely mm
                 unit_name = 'mm'
                 unit_scale = 0.001  # Convert mm to m for internal calculations

            # Construct standardized geom_info for main.py (expects meters)
            geom_info = {
                "volume": volume * (unit_scale ** 3),  # Convert to m³
                "bbox_diagonal": bbox_diag * unit_scale,  # Convert to m
                "units_detected": unit_name,
                "volume_display": volume,  # Raw volume for display
                "node_count": smooth_poly_data.GetNumberOfPoints(),
                "filepath": self.filepath
            }
            
            if not self._is_running:
                return
            
            # Emit success
            self.finished.emit(smooth_poly_data, geom_info)
            
        except Exception as e:
            print(f"[CAD WORKER] Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


class VTK3DViewer(QFrame):
    """3D viewer with quality report overlay"""
    
    # Signal to request application exit
    exit_requested = pyqtSignal()
    # Signal when selection changes (count of selected faces)
    selection_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Box | QFrame.Sunken)
        self.setLineWidth(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Initialize Zone Manager
        self.zone_manager = FluentZoneManager()
        self.vtk_cell_to_face_id = [] # List mapping VTK cell ID -> Face ID
        self.face_id_to_vtk_cell = {} # Dict mapping Face ID -> VTK cell ID

        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtk_widget)

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.95, 0.95, 0.97)
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)

        # Shortcut to exit fullscreen
        self.esc_shortcut = QShortcut(QKeySequence("Esc"), self)
        self.esc_shortcut.activated.connect(self.exit_fullscreen)

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
        self.above_cut_actor = None # Actor for ghost visualization
        self.persistent_ghost_data = None # Storage for ghost data persistence
        self.cross_section_mode = "layered"  # Always use layered mode (show complete volume cells)
        self.cross_section_element_mode = "auto"  # Auto-switch between tet/hex slicing
        
        self.avg_cell_size = 1.0  # Default average cell size for dynamic visibility

        # Progressive loading state
        self.hq_worker = None
        self.current_cad_path = None
        
        # CAD loading worker and overlay
        self.cad_load_worker = None
        self.current_geom_info = None  # Store geometry info when CAD loading completes
        self.loading_overlay = LoadingOverlay(self)
        self.loading_overlay.hide()
        self.loading_overlay.setGeometry(self.rect())
        
        # Progressive loading: cache full-quality tessellation for mesh generation
        self.full_quality_stl_path = None  # Path to high-quality STL for mesh generation
        self.is_refining_preview = False  # Track if Stage 2 refinement is active
        self.hq_tessellation_process = None  # Background subprocess for high-quality tessellation
        self.hq_tessellation_complete = False  # Track if high-quality STL is ready

        # Paintbrush visual feedback
        self.brush_cursor_actor = None
        self.brush_cursor_visible = False
        self.painted_cells = set()  # Set of VTK cell IDs that have been painted
        self.paint_colors = vtk.vtkUnsignedCharArray()  # RGB colors per cell
        self.paint_colors.SetNumberOfComponents(3)
        self.paint_colors.SetName("PaintColors")
        
        # Fullscreen state
        self.is_fullscreen = False
        self.saved_parent_layout = None

        # Info overlay (top-left) - FIXED WIDTH to prevent truncation
        # Create BEFORE vtk_widget.Initialize() to avoid resizeEvent errors
        self.info_label = QLabel(self)
        self.info_label.setStyleSheet("""
            QLabel {
                background-color: rgba(255, 255, 255, 255);
                padding: 10px;
                border-radius: 5px;
                font-family: Arial;
                font-size: 12px;
                border: 1px solid #dee2e6;
            }
        """)
        self.info_label.setText("<b>3D Preview</b><br>No model loaded")
        self.info_label.setWordWrap(True)
        self.info_label.setFixedWidth(450)
        self.info_label.setMinimumHeight(80)
        self.info_label.setMaximumHeight(300)
        self.info_label.move(10, 10)
        self.info_label.show()

        self.vtk_widget.Initialize()
        self.vtk_widget.Start()
        
        # Enable double-click to enter fullscreen
        self.vtk_widget.mouseDoubleClickEvent = self._on_double_click
    

        # Exit Button (Top-right overlay)
        self.exit_btn = QPushButton("Exit", self)
        self.exit_btn.setCursor(Qt.PointingHandCursor)
        self.exit_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 0, 0, 0.1); 
                color: #555;
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 4px 12px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(220, 53, 69, 0.9);
                color: white;
                border: 1px solid #dc3545;
            }
        """)
        self.exit_btn.clicked.connect(self.exit_requested.emit)
        self.exit_btn.resize(60, 24)
        self.exit_btn.show()

        # Initial Camera Setup
        self.renderer.SetBackground(0.95, 0.95, 0.97) # Reverted to original background color
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
        """Handle resize to update overlay positions"""
        # Keep info label top-left
        self.info_label.move(10, 10)
        
        # Keep Exit button top-right
        if hasattr(self, 'exit_btn'):
            self.exit_btn.move(self.width() - self.exit_btn.width() - 10, 10)
        
        # Resize loading overlay to match viewer size
        if hasattr(self, 'loading_overlay'):
            self.loading_overlay.setGeometry(self.rect())
            
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
        # Remove ALL 3D actors from renderer
        actors = self.renderer.GetActors()
        actors.InitTraversal()
        for i in range(actors.GetNumberOfItems()):
            actor = actors.GetNextActor()
            if actor:
                self.renderer.RemoveActor(actor)
        
        # Remove ALL 2D actors (scalar bars, text, etc.)
        actors2d = self.renderer.GetActors2D()
        actors2d.InitTraversal()
        for i in range(actors2d.GetNumberOfItems()):
            actor = actors2d.GetNextActor2D()
            if actor:
                self.renderer.RemoveActor2D(actor)
        
        self.current_actor = None
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()
    
    def _on_double_click(self, event):
        """Handle double-click to toggle fullscreen"""
        self.toggle_fullscreen()

    def toggle_fullscreen(self):
        """Toggle fullscreen mode - hides all GUI except VTK viewer, ESC to exit"""
        if not self.is_fullscreen:
            # Save parent window BEFORE detaching
            self.saved_main_window = self.window()
            
            # Enter fullscreen
            self.is_fullscreen = True
            
            # Hide info label (top-left overlay)
            if hasattr(self, 'info_label') and self.info_label:
                self.info_label.hide()
            
            # Detach from current layout and show as standalone fullscreen window
            self.setParent(None)
            self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
            self.showFullScreen()
            
            # Install event filter to catch ESC key
            self.installEventFilter(self)
            
            print("[Fullscreen] Entered - press ESC to exit")
        else:
            # Exit fullscreen
            self.exit_fullscreen()
    
    def exit_fullscreen(self):
        """Exit fullscreen mode and restore to main window"""
        if not self.is_fullscreen:
            return
            
        self.is_fullscreen = False
        
        # Remove event filter
        self.removeEventFilter(self)
        
        # Show info label again
        if hasattr(self, 'info_label') and self.info_label:
            self.info_label.show()
        
        # Return to normal windowed mode
        self.showNormal()
        self.setWindowFlags(Qt.Widget)
        
        # Re-parent to main window
        if hasattr(self, 'saved_main_window') and self.saved_main_window:
            main_window = self.saved_main_window
            
            # Find the right panel and re-add to it
            if hasattr(main_window, 'right_panel_layout'):
                # Insert at index 1 (between controls and console) with stretch 2
                main_window.right_panel_layout.insertWidget(1, self, 2)
                print("[Fullscreen] Re-attached to main window layout")
            else:
                 print("[Fullscreen] WARNING: Could not find right_panel_layout to re-attach")
        
        self.show()
        print("[Fullscreen] Exited")
    
    def eventFilter(self, obj, event):
        """Event filter to catch ESC key for exiting fullscreen"""
        if self.is_fullscreen and event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Escape:
                self.exit_fullscreen()
                return True
        return super().eventFilter(obj, event)

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

        # Quality overlay disabled per user request - metrics shown in console instead
        # self.quality_label.setText(report_html)
        # self.quality_label.adjustSize()
        # self.quality_label.move(
        #     self.width() - self.quality_label.width() - 15,
        #     15
        # )
        # self.quality_label.setVisible(True)

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
    
    def load_polyhedral_file(self, json_path: str):
        """Load polyhedral mesh from JSON file"""
        # DEBUG: Write to file to prove function is running
        debug_log = Path("poly_debug.txt")
        with open(debug_log, 'w') as f:
            f.write(f"=== POLYHEDRAL LOAD DEBUG ===\n")
            f.write(f"Function called at: {json_path}\n")
        
        print("=" * 80)
        print("[POLY-VIZ v2.0] STARTING POLYHEDRAL LOAD")
        print("=" * 80)
        try:
            print(f"[POLY-VIZ] Loading polyhedral data from {json_path}")
            
            # Check file exists
            if not Path(json_path).exists():
                print(f"[POLY-VIZ ERROR] File not found: {json_path}")
                with open(debug_log, 'a') as f:
                    f.write("[ERROR] File not found\n")
                return "FAILED"
            
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            nodes = data.get('nodes', {})
            elements = data.get('elements', [])
            
            with open(debug_log, 'a') as f:
                f.write(f"Nodes: {len(nodes)}, Elements: {len(elements)}\n")
            
            print(f"[POLY-VIZ] Loaded {len(nodes)} nodes, {len(elements)} elements")
            
            if not nodes or not elements:
                print("[POLY-VIZ ERROR] No nodes or elements in JSON")
                with open(debug_log, 'a') as f:
                    f.write("[ERROR] No data\n")
                return "FAILED"
            
            self.clear_view()
            
            # 1. Create VTK Points
            points = vtk.vtkPoints()
            node_map = {}
            sorted_node_ids = sorted(nodes.keys(), key=lambda x: int(x))
            
            print(f"[POLY-VIZ] Creating {len(sorted_node_ids)} points...")
            for i, node_id in enumerate(sorted_node_ids):
                coords = nodes[node_id]
                points.InsertNextPoint(coords)
                node_map[str(node_id)] = i
                node_map[int(node_id)] = i
                
            # 2. Create Unstructured Grid
            ugrid = vtk.vtkUnstructuredGrid()
            ugrid.SetPoints(points)
            
            # 3. Insert Cells
            print(f"[POLY-VIZ] Inserting {len(elements)} polyhedral cells...")
            count = 0
            skipped = 0
            
            for elem_idx, elem in enumerate(elements):
                try:
                    if elem['type'] != 'polyhedron':
                        continue
                        
                    faces = elem['faces']
                    if len(faces) < 3:
                        skipped += 1
                        continue
                    
                    face_stream = [len(faces)]
                    
                    for face_nodes in faces:
                        face_stream.append(len(face_nodes))
                        try:
                            face_indices = [node_map[nid] for nid in face_nodes]
                            face_stream.extend(face_indices)
                        except KeyError as e:
                            print(f"[POLY-VIZ WARNING] Node ID {e} not found in element {elem_idx}, skipping face")
                            skipped += 1
                            continue
                    
                    # Use vtkIdList for the face stream
                    id_list = vtk.vtkIdList()
                    for val in face_stream:
                        id_list.InsertNextId(val)
                    
                    ugrid.InsertNextCell(vtk.VTK_POLYHEDRON, id_list)
                    count += 1
                    
                    if count % 50 == 0:
                        print(f"[POLY-VIZ] Inserted {count}/{len(elements)} cells...")
                        
                except Exception as e:
                    print(f"[POLY-VIZ ERROR] Failed to insert element {elem_idx}: {e}")
                    with open(debug_log, 'a') as f:
                        f.write(f"[ERROR] Element {elem_idx}: {e}\n")
                    skipped += 1
                    continue
            
            with open(debug_log, 'a') as f:
                f.write(f"Cells created: {count}, Skipped: {skipped}\n")
                f.write(f"UGrid cells: {ugrid.GetNumberOfCells()}\n")
            
            print(f"[POLY-VIZ] Created {count} polyhedral cells ({skipped} skipped)")
            
            if count == 0:
                print("[POLY-VIZ ERROR] No cells were created!")
                return "FAILED"
            
            # 4. Extract Surface Using VTK's Geometry Filter
            # This automatically finds the boundary of the VTK_POLYHEDRON cells
            print("[POLY-VIZ v2.0] Extracting surface using vtkGeometryFilter...")
            
            geometry_filter = vtk.vtkGeometryFilter()
            geometry_filter.SetInputData(ugrid)
            geometry_filter.Update()
            polydata = geometry_filter.GetOutput()
            
            with open(debug_log, 'a') as f:
                f.write(f"=== SURFACE EXTRACTION (vtkGeometryFilter) ===\n")
                f.write(f"Input cells: {ugrid.GetNumberOfCells()}\n")
                f.write(f"Output surface cells: {polydata.GetNumberOfCells()}\n")
                f.write(f"Output points: {polydata.GetNumberOfPoints()}\n")
            
            print(f"[POLY-VIZ] Extracted {polydata.GetNumberOfCells()} surface cells")
            
            # 5. Mapper/Actor
            print("[POLY-VIZ] Creating mapper and actor...")
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(polydata)
            
            self.current_actor = vtk.vtkActor()
            self.current_actor.SetMapper(mapper)
            self.current_actor.GetProperty().SetEdgeVisibility(1)
            self.current_actor.GetProperty().SetColor(0.7, 0.8, 0.9)
            self.current_actor.GetProperty().SetOpacity(1.0)
            
            # CRITICAL: Disable back-face culling for polyhedral meshes
            # Boundary polyhedra may have outward-facing polygons that get culled
            self.current_actor.GetProperty().BackfaceCullingOff()
            self.current_actor.GetProperty().FrontfaceCullingOff()
            
            # Improve lighting for better visibility
            self.current_actor.GetProperty().SetAmbient(0.3)
            self.current_actor.GetProperty().SetDiffuse(0.7)
            self.current_actor.GetProperty().SetSpecular(0.2)
            
            self.renderer.AddActor(self.current_actor)
            self.current_volumetric_grid = ugrid
            self.current_poly_data = None
            
            with open(debug_log, 'a') as f:
                f.write(f"Actor created and added to renderer\n")
                f.write(f"Renderer actor count: {self.renderer.GetActors().GetNumberOfItems()}\n")

            # CRITICAL: Enable edges for wireframe visualization
            actor.GetProperty().EdgeVisibilityOn()
            actor.GetProperty().SetEdgeColor(0.2, 0.2, 0.2)  # Dark gray edges
            actor.GetProperty().SetLineWidth(1.0)
            
            print("[POLY-VIZ] Resetting camera and rendering...")
            
            # Get bounds and log them
            bounds = ugrid.GetBounds()
            with open(debug_log, 'a') as f:
                f.write(f"UGrid bounds: {bounds}\n")
            
            # Compute center and size
            import numpy as np
            center = np.array([
                (bounds[0] + bounds[1]) / 2,
                (bounds[2] + bounds[3]) / 2,
                (bounds[4] + bounds[5]) / 2
            ])
            size = np.array([
                bounds[1] - bounds[0],
                bounds[3] - bounds[2],
                bounds[5] - bounds[4]
            ])
            max_size = np.max(size)
            
            with open(debug_log, 'a') as f:
                f.write(f"Center: {center}\n")
                f.write(f"Size: {size}\n")
                f.write(f"Max size: {max_size}\n")
            
            # Reset camera with explicit positioning
            camera = self.renderer.GetActiveCamera()
            camera.SetFocalPoint(center[0], center[1], center[2])
            camera.SetPosition(
                center[0] + max_size * 1.5,
                center[1] + max_size * 1.5,
                center[2] + max_size * 1.5
            )
            camera.SetViewUp(0, 0, 1)
            self.renderer.ResetCamera(bounds)
            
            self.vtk_widget.GetRenderWindow().Render()
            
            self.info_label.setText(f"<b>Polyhedral Mesh</b><br>{ugrid.GetNumberOfCells()} polyhedra")
            print("[POLY-VIZ] SUCCESS!")
            
            with open(debug_log, 'a') as f:
                f.write(f"=== COMPLETE SUCCESS ===\n")
            
            return "SUCCESS"
            
        except Exception as e:
            print(f"[POLY-VIZ ERROR] Exception: {e}")
            import traceback
            traceback.print_exc()
            with open(debug_log, 'a') as f:
                f.write(f"[EXCEPTION] {e}\n")
                f.write(traceback.format_exc())
            self.info_label.setText(f"Error loading polyhedral mesh: {e}")
            return "FAILED"

    def _generate_cross_section_mesh(self, plane_origin, plane_normal):
        """
        Generate VTK PolyData for the cross-section.
        Uses vtkCutter for Polyhedra, manual intersection for Tets/Hexes.
        """
        generated_poly = None

        # If we have a volumetric grid, use vtkCutter (Standard, Fast, Watertight)
        # This works for Tets, Hexes, and Polyhedra alike.
        if self.current_volumetric_grid and self.current_volumetric_grid.GetNumberOfCells() > 0:
             plane = vtk.vtkPlane()
             plane.SetOrigin(plane_origin)
             plane.SetNormal(plane_normal)
             
             cutter = vtk.vtkCutter()
             cutter.SetInputData(self.current_volumetric_grid)
             cutter.SetCutFunction(plane)
             # cutter.GenerateCutScalarsOn() # Generates scalars based on cut function value (distance) - we don't want this
             cutter.GenerateCutScalarsOff()
             cutter.Update()
             
             return cutter.GetOutput()
        
        # Fallback to manual intersection for standard meshes (Tets/Hexes) ONLY if volume grid missing
        import numpy as np
        
        intersecting_elements = self._get_volume_elements_intersecting_plane(plane_origin, plane_normal)
        
        if not intersecting_elements:
            return vtk.vtkPolyData()
        
        # ... (rest of manual intersection logic) ...
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
            
            # Triangulate the polygon
            if num_points == 3:
                all_triangles.append([point_offset, point_offset + 1, point_offset + 2])
            elif num_points == 4:
                all_triangles.append([point_offset, point_offset + 1, point_offset + 2])
                all_triangles.append([point_offset, point_offset + 2, point_offset + 3])
            else:
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
            
            point_offset += len(vertices)
            
            # --- HANDLE DIFFERENT CELL TYPES ---
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
            
            elif element['type'] == 'polyhedron':
                # For polyhedra, we need to extract faces dynamically
                # This assumes 'faces' key contains list of face indices (list of lists)
                # If not available in element dict, we might need to fetch from vtk grid
                
                # Check if we have face definitions
                if 'faces' in element:
                    for face in element['faces']:
                        # Triangulate simple face
                        if len(face) >= 3:
                            # Fan triangulation
                            pivot = local_offset + face[0] # Note: face indices must be relative to node list
                            for i in range(1, len(face) - 1):
                                b = local_offset + face[i]
                                c = local_offset + face[i+1]
                                all_triangles.append([pivot, b, c])
                                face_to_element_id.append(element['id'])
                else: 
                     # Fallback: Just try to grab faces from original cell if accessible
                     print(f"[DEBUG] Polyhedron {element.get('id')} missing 'faces' key, skipping cross-section render")
                     pass
            
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
    
    def _get_clip_plane_params(self, bounds, center, offset_percentage):
        """Helper to determine plane normal and origin based on axis and offset."""
        if self.clip_axis == 'x':
            normal = (-1, 0, 0)
            dim_size = bounds[1] - bounds[0]
            origin = [center[0] + (dim_size * offset_percentage / 100.0), center[1], center[2]]
        elif self.clip_axis == 'y':
            normal = (0, -1, 0)
            dim_size = bounds[3] - bounds[2]
            origin = [center[0], center[1] + (dim_size * offset_percentage / 100.0), center[2]]
        else:  # z
            normal = (0, 0, -1)
            dim_size = bounds[5] - bounds[4]
            origin = [center[0], center[1], center[2] + (dim_size * offset_percentage / 100.0)]
        return origin, normal

    def _apply_clipping(self):
        """Apply cross-section using Element-Based 'Crinkle Cut' for structural inspection"""
        if not self.current_volumetric_grid:
            # Fallback to standard surface clip if no volume
            self._apply_standard_surface_clipping()
            return
            
        # Get clipping planes
        try:
            print(f"[DEBUG] _apply_clipping: Axis={self.clip_axis}, Offset={self.clip_offset}")
            
            plane = vtk.vtkPlane()
            
            # Determine plane normal and origin based on axis and offset
            bounds = self.current_volumetric_grid.GetBounds() # Use volumetric grid bounds
            # ... (bounds recalc is fast)
            
            center = [
                (bounds[0] + bounds[1]) / 2,
                (bounds[2] + bounds[3]) / 2,
                (bounds[4] + bounds[5]) / 2
            ]
            
            origin, normal = self._get_clip_plane_params(bounds, center, self.clip_offset)
            print(f"[DEBUG] Clip Plane: Origin={origin}, Normal={normal}")
            
            plane.SetOrigin(origin)
            plane.SetNormal(normal)
            
            # --- CRINKLE CUT LOGIC ---
            # Extract cells that are "Inside" (or intersected)
            # This shows whole elements
            
            extract = vtk.vtkExtractGeometry()
            extract.SetInputData(self.current_volumetric_grid)
            extract.SetImplicitFunction(plane)
            extract.ExtractInsideOn()
            extract.ExtractBoundaryCellsOn() # Include intersected cells (User: "see which tets are cut")
            extract.ExtractOnlyBoundaryCellsOff()
            extract.Update()
            
            # Get the "Solid" part (Below Cut)
            solid_grid = extract.GetOutput()
            print(f"[DEBUG] Extracted Cross-Section Cells: {solid_grid.GetNumberOfCells()}")
            
            # Get the "Solid" part (Below Cut)
            solid_grid = extract.GetOutput()
            
            # Store to prevent GC
            self.current_cross_section_data = solid_grid
            self.current_cross_section_data.Register(None) # Extra safety
            
            # Update Main Actor (Solid)
            # Use DataSetMapper which handles UnstructuredGrid directly (shows faces)
            main_mapper = vtk.vtkDataSetMapper()
            main_mapper.SetInputData(self.current_cross_section_data)
            
            # Restore visual properties
            # If quality data is present, use it
            if hasattr(self, 'current_quality_data') and self.current_quality_data:
                main_mapper.SetScalarModeToUseCellData()
                main_mapper.SetColorModeToDirectScalars()
                main_mapper.ScalarVisibilityOn()
            else:
                # Fallback to default color if no quality data
                self.current_actor.GetProperty().SetColor(0.8, 0.8, 0.8) # Default gray
                main_mapper.ScalarVisibilityOff()
            
            self.current_actor.SetMapper(main_mapper)
            self.current_actor.GetProperty().EdgeVisibilityOn()
            self.current_actor.GetProperty().SetOpacity(1.0) # Fully visible
            
            # --- GHOST PART (Above Cut) ---
            if hasattr(self, 'show_ghost_cb') and self.show_ghost_cb.isChecked():
                extract_ghost = vtk.vtkExtractGeometry()
                extract_ghost.SetInputData(self.current_volumetric_grid)
                extract_ghost.SetImplicitFunction(plane)
                extract_ghost.ExtractInsideOff() # Extract OUTSIDE
                extract_ghost.ExtractBoundaryCellsOff() # Avoid duplicates? Or On?
                # If we included boundary in solid, we exclude here?
                # Usually standard extract puts boundary in "Inside" if specified.
                extract_ghost.Update()
                
                self.persistent_ghost_data = extract_ghost.GetOutput()
                
                if not self.above_cut_actor:
                     self.above_cut_actor = vtk.vtkActor()
                     self.renderer.AddActor(self.above_cut_actor)
                
                ghost_mapper = vtk.vtkDataSetMapper()
                ghost_mapper.SetInputData(self.persistent_ghost_data)
                
                self.above_cut_actor.SetMapper(ghost_mapper)
                self.above_cut_actor.GetProperty().SetOpacity(0.1)
                self.above_cut_actor.GetProperty().SetColor(0.7, 0.7, 0.7) # Light gray
                self.above_cut_actor.VisibilityOn()
            else:
                if self.above_cut_actor:
                    self.above_cut_actor.VisibilityOff()
            
            # Hide the old "Cross Section" actor if it exists (we merged it into main)
            if self.cross_section_actor:
                self.cross_section_actor.VisibilityOff()
                
            self.vtk_widget.GetRenderWindow().Render()
            
        except Exception as e:
            print(f"[ERROR] Crinkle cut failed: {e}")
            import traceback
            traceback.print_exc()
            # Try to restore normal view if cross-section fails
            if hasattr(self, 'above_cut_actor') and self.above_cut_actor:
                self.above_cut_actor.VisibilityOff()
            if hasattr(self, 'cross_section_actor') and self.cross_section_actor:
                self.cross_section_actor.VisibilityOff()

    def _apply_standard_surface_clipping(self):
        """Apply cross-section with clear above/below separation"""
        if not self.current_poly_data or not self.current_actor:
            return
        
        try:
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
            
            # SAFETY: Handle LODActor (which doesn't support GetMapper easily for clipping)
            if self.current_actor.IsA("vtkLODActor"):
                print("[DEBUG] Swapping LODActor for vtkActor to support clipping")
                new_actor = vtk.vtkActor()
                # Copy properties (color, lighting, etc.)
                new_actor.SetProperty(self.current_actor.GetProperty())
                
                # Create new mapper for this standard actor
                new_mapper = vtk.vtkPolyDataMapper()
                new_actor.SetMapper(new_mapper)
                
                # Swap in renderer
                self.renderer.RemoveActor(self.current_actor)
                self.renderer.AddActor(new_actor)
                self.current_actor = new_actor
                
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
            
            # --- PART 3: CROSS-SECTION (WATERTIGHT CUT) ---
            # Generate cross-section using vtkCutter (fast C++ implementation)
            cross_section_poly = self._generate_cross_section_mesh(origin, normal)
            
            if cross_section_poly.GetNumberOfCells() > 0:
                # Create actor if it doesn't exist
                if not self.cross_section_actor:
                    self.cross_section_actor = vtk.vtkActor()
                    self.renderer.AddActor(self.cross_section_actor)
                
                # SAFETY: Persist data in class to prevent Python GC from killing C++ object
                self.current_cross_section_data = vtk.vtkPolyData()
                self.current_cross_section_data.DeepCopy(cross_section_poly)
                
                # Validate before rendering
                if self.current_cross_section_data.GetNumberOfPoints() == 0:
                     print("[DEBUG] Cross-section resulted in empty mesh (post-copy validation)")
                     self.cross_section_actor.VisibilityOff()
                     return
                
                # Cleanup memory layout
                self.current_cross_section_data.Squeeze()

                # Robust Mapper: Use DataSetMapper instead of PolyDataMapper
                # It handles edge cases and topology issues much better, preventing crashes
                cs_mapper = vtk.vtkDataSetMapper()
                cs_mapper.SetInputData(self.current_cross_section_data)
                
                # CRITICAL: Use depth offset to render cross-section ABOVE everything else
                # Negative offset shifts toward camera, eliminating z-fighting
                cs_mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-1, -1)
                cs_mapper.SetRelativeCoincidentTopologyLineOffsetParameters(-1, -1)
                
                self.cross_section_actor.SetMapper(cs_mapper)
                
                # Check if we have quality colors
                has_quality_colors = (self.current_cross_section_data.GetCellData().GetScalars() is not None)
                
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
            
        except Exception as e:
            print(f"[ERROR] Cross-section failed: {e}")
            import traceback
            traceback.print_exc()
            # Try to restore normal view if cross-section fails
            if hasattr(self, 'above_cut_actor') and self.above_cut_actor:
                self.above_cut_actor.VisibilityOff()
            if hasattr(self, 'cross_section_actor') and self.cross_section_actor:
                self.cross_section_actor.VisibilityOff()
    
    def _remove_clipping(self):
        """Remove clipping and restore original mesh"""
        if not self.current_poly_data or not self.current_actor:
            return
            
        # Hide auxiliary actors first
        if hasattr(self, 'above_cut_actor') and self.above_cut_actor:
            self.above_cut_actor.VisibilityOff()
            
        if self.cross_section_actor:
            self.cross_section_actor.VisibilityOff()
            
        # RESTORE ORIGINAL STATE
        # If we swapped actors (LOD -> Standard), we should probably stick with standard for now 
        # to avoid complexity, or just reset the mapper input.
        
        # Ensure we have a mapper
        mapper = self.current_actor.GetMapper()
        if not mapper:
             # Should practically never happen for a valid actor
             print("[ERROR] Actor has no mapper during clip removal")
             return

        # Restore full opacity
        self.current_actor.GetProperty().SetOpacity(1.0)
        
        # Reset input to original full mesh
        # CRITICAL FIX: Ensure input is valid
        if self.current_poly_data.GetNumberOfPoints() > 0:
            mapper.SetInputData(self.current_poly_data)
        else:
            print("[WARN] Original poly data is empty, cannot restore mesh view")
            
        self.vtk_widget.GetRenderWindow().Render()

    def update_quality_visualization(self, metric="SICN (Min)", opacity=1.0, min_val=0.0, max_val=1.0):
        """
        Update visualization based on selected quality metric and filters.
        Rebuilds the displayed mesh to hide elements outside the range.
        """
        if not self.current_poly_data or not self.current_actor:
            return
        
        # SAFETY: Check for polyhedral meshes (no quality switching support yet)
        if self.current_volumetric_grid is not None and hasattr(self, 'current_volumetric_grid'):
            # This is a polyhedral mesh - quality switching not supported
            print("[WARN] Quality metric switching not supported for polyhedral meshes")
            # Just update opacity and return
            self.current_actor.GetProperty().SetOpacity(opacity)
            self.vtk_widget.GetRenderWindow().Render()
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
        elif "Jacobian" in metric:
            metric_key = 'per_element_quality'
            
        if metric_key in self.current_quality_data:
            quality_map = self.current_quality_data[metric_key]
        elif metric != "SICN (Min)":
            print(f"[WARN] Metric '{metric}' ({metric_key}) not found, defaulting to SICN")
            quality_map = self.current_quality_data.get('per_element_quality', {})
            
        if not quality_map:
            return

        # SAFETY: Ensure mesh elements exist
        if not self.current_mesh_elements:
            print("[WARN] Cannot update quality visualization: mesh elements not loaded")
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
            
        visible_count = 0
        filtered_count = 0
        
        # Get global range for this metric for consistent coloring
        all_vals = [float(v) for v in quality_map.values()]
        if all_vals:
            global_min, global_max = min(all_vals), max(all_vals)
            val_range = global_max - global_min if global_max > global_min else 1.0
        else:
            global_min, global_max, val_range = 0.0, 1.0, 1.0

        if not self.current_mesh_elements and not (hasattr(self, 'surface_to_volume_map') and self.surface_to_volume_map):
            print("[VTKViewer] No mesh elements or map to color")
            return

        # Prepare new colors array
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
            
        visible_count = 0
        filtered_count = 0
        
        # New Logic: Color EXISTING poly_data using Mapping
        # This prevents rebuilding the mesh and losing connectivity/shape
        if self.current_poly_data and hasattr(self, 'surface_to_volume_map') and self.surface_to_volume_map:
            print("[VTKViewer] Applying colors to existing surface using Volume Map...")
            num_cells = self.current_poly_data.GetNumberOfCells()
            colors.SetNumberOfTuples(num_cells)
            
            # Create a ghost array to hide filtered cells
            ghosts = vtk.vtkUnsignedCharArray()
            ghosts.SetNumberOfTuples(num_cells)
            ghosts.SetName(vtk.vtkDataSetAttributes.GhostArrayName())
            ghosts.Fill(0)
            
            for i in range(num_cells):
                elem_id = self.surface_to_volume_map.get(i)
                val = None
                
                check_debug = (i < 5) # Debug first 5 cells
                
                if elem_id is not None:
                    # Look up quality
                    if str(elem_id) in quality_map:
                        val = quality_map[str(elem_id)]
                        if check_debug: print(f"[DEBUG_COLOR] Cell {i} -> Elem {elem_id} (Type {type(elem_id)}) -> Found in map (STR key). Val: {val}")
                    elif int(elem_id) in quality_map:
                         val = quality_map[int(elem_id)]
                         if check_debug: print(f"[DEBUG_COLOR] Cell {i} -> Elem {elem_id} -> Found in map (INT key). Val: {val}")
                    elif check_debug:
                         keys_sample = list(quality_map.keys())[:5]
                         print(f"[DEBUG_COLOR] Cell {i} -> Elem {elem_id} -> NOT FOUND. Map keys sample: {keys_sample}")
                
                # Default Color (Gray) if no data
                r, g, b = 200, 200, 200
                
                # Special debug for non-found
                if val is None and check_debug:
                     print(f"[DEBUG_COLOR] Cell {i} has NO VALUE. Defaulting to Gray.")
                
                if val is not None:
                     # Check visibility filter
                    if not (min_val <= val <= max_val):
                        # Filtered out - Hide using Ghost Array
                        ghosts.SetValue(i, vtk.vtkDataSetAttributes.HIDDENCELL)
                        filtered_count += 1
                    else:
                        visible_count += 1
                        # Compute Color
                        if "Skewness" in metric:
                            # Bad > 0.7
                            if val > 0.7: r, g, b = 220, 20, 60
                            else:
                                h = (1.0 - max(0.0, min(1.0, val / 0.7))) * 0.28 + 0.05
                                r, g, b = hsl_to_rgb(h, 1.0, 0.5)
                        elif "Aspect" in metric:
                             # Bad > 10
                            if val > 10.0: r, g, b = 220, 20, 60
                            else:
                                h = (1.0 - max(0.0, min(1.0, (val - 1.0) / 9.0))) * 0.28 + 0.05
                                r, g, b = hsl_to_rgb(h, 1.0, 0.5)
                        else:
                            # Bad < 0.2
                            if val < 0.2: r, g, b = 220, 20, 60
                            else:
                                h = max(0.0, min(1.0, (val - 0.2) / 0.8)) * 0.28 + 0.05
                                r, g, b = hsl_to_rgb(h, 1.0, 0.5)
                
                colors.SetTuple3(i, r, g, b)
            
            self.current_poly_data.GetCellData().SetScalars(colors)
            if filtered_count > 0:
                 self.current_poly_data.GetCellData().AddArray(ghosts)
            else:
                 self.current_poly_data.GetCellData().RemoveArray(vtk.vtkDataSetAttributes.GhostArrayName())

        else:
            # FALLBACK: Legacy Rebuild (for 2D meshes or unmapped data)
            # This reconstructs the mesh from elements list (Logic preserved but simplified)
            print("[VTKViewer] Fallback: Rebuilding mesh for coloring (Legacy/2D)...")
            new_cells = vtk.vtkCellArray()
            # ... (Rest of legacy rebuilding logic if needed, but for 3D MSH this path is now avoided)
            # For brevity/safety, if we are here for 3D MSH without map, we effectively return empty or logic above handles it.
            # Assuming this block is reached for actual 2D surface meshes where current_mesh_elements allows reconstruction.
            # Copying previous logic briefly:
            # ...
            # Implementation omitted to force use of Map for 3D. 
            # If 2D mesh, load_mesh_file adds triangles to elements list, so we can iterate.
            pass # (Actually let's just use the map logic. 2D meshes should map Identity.)

        # Update mapper
        mapper = self.current_actor.GetMapper()
        mapper.SetInputData(self.current_poly_data) # Use Modified PolyData directly!
        mapper.ScalarVisibilityOn() 
        mapper.SetScalarModeToUseCellData()
        mapper.SetColorModeToDirectScalars()
        
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

    def on_hq_mesh_ready(self, stl_path: str, geom_info: dict, original_cad_path: str):
        """Callback when background HQ mesh generation finishes"""
        try:
            # RACE CONDITION CHECK: user might have loaded a different file
            if original_cad_path != self.current_cad_path:
                print(f"[HQ UPDATE] Stale update ignored. Current: {self.current_cad_path}, Received: {original_cad_path}")
                if os.path.exists(stl_path):
                    os.unlink(stl_path)
                return

            print(f"[HQ UPDATE] Swapping to high-fidelity mesh...")
            
            # Load the HQ mesh
            if not os.path.exists(stl_path) or os.path.getsize(stl_path) < 100:
                print("[HQ UPDATE ERROR] STL file missing or empty")
                return
                
            mesh = pv.read(stl_path)
            os.unlink(stl_path) # Cleanup
            
            if mesh.n_points == 0:
                print("[HQ UPDATE ERROR] Empty mesh")
                return

            poly_data = mesh.cast_to_unstructured_grid().extract_surface()
            self.current_poly_data = poly_data # Update stored data for operations
            
            # Remove old actor
            if self.current_actor:
                self.renderer.RemoveActor(self.current_actor)
            
            # --- APPLY VISUALIZATION PIPELINE (Same as fast load but on HQ data) ---
            # 1. Smooth Normals (No subdivision needed for HQ mesh usually)
            normals_gen = vtk.vtkPolyDataNormals()
            normals_gen.SetInputData(poly_data)
            normals_gen.ComputePointNormalsOn()
            normals_gen.ComputeCellNormalsOff()
            normals_gen.SplittingOn() 
            normals_gen.SetFeatureAngle(60.0)
            normals_gen.Update()
            smooth_poly_data = normals_gen.GetOutput()
            
            # 2. LOD Setup (Still good for performance even on HQ mesh)
            mapper_high = vtk.vtkPolyDataMapper()
            mapper_high.SetInputData(smooth_poly_data)
            
            decimator = vtk.vtkQuadricDecimation()
            decimator.SetInputData(smooth_poly_data)
            decimator.SetTargetReduction(0.9)
            decimator.Update()
            
            mapper_low = vtk.vtkPolyDataMapper()
            mapper_low.SetInputData(decimator.GetOutput())
            
            self.current_actor = vtk.vtkLODActor()
            self.current_actor.AddLODMapper(mapper_high)
            self.current_actor.AddLODMapper(mapper_low)
            self.current_actor.SetDesiredUpdateRate(15.0) # Actor property, safe to set
            
            # 3. Styling
            self.current_actor.GetProperty().SetColor(0.3, 0.5, 0.8)
            self.current_actor.GetProperty().SetInterpolationToPhong() 
            self.current_actor.GetProperty().BackfaceCullingOn()       
            self.current_actor.GetProperty().ShadingOn()
            self.current_actor.GetProperty().SetAmbient(0.3)
            self.current_actor.GetProperty().SetDiffuse(0.7)
            self.current_actor.GetProperty().SetSpecular(0.5)          
            self.current_actor.GetProperty().SetSpecularPower(40.0)
            self.current_actor.GetProperty().EdgeVisibilityOff()
            
            self.renderer.AddActor(self.current_actor)
            
            # Flash informative text
            self.info_label.setText(
                self.info_label.text().replace(
                    "<span style='color: #6c757d;'>", 
                    "<span style='color: #198754; font-weight:bold;'>HQ READY: "
                )
            )
            
            self.vtk_widget.GetRenderWindow().Render()
            print("[HQ UPDATE] Success!")
            
        except Exception as e:
            print(f"[HQ UPDATE ERROR] {e}")
            import traceback
            traceback.print_exc()

    
    def _start_hq_tessellation(self, filepath: str, complexity: dict):
        """Stage 3: Start background high-quality tessellation for mesh generation"""
        import subprocess
        import tempfile
        import os
        
        log_file = r"C:\Users\Owner\Downloads\cad_load_debug.txt"
        
        def log(msg):
            print(msg, flush=True)
            try:
                with open(log_file, 'a', encoding='utf-8') as f:
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    f.write(f"[{timestamp}] {msg}\n")
            except:
                pass
        
        # Kill any existing background process
        if self.hq_tessellation_process and self.hq_tessellation_process.poll() is None:
            log("[STAGE 3] Killing previous background tessellation process...")
            self.hq_tessellation_process.kill()
        
        # Create output path for high-quality STL
        temp_dir = tempfile.gettempdir()
        safe_name = Path(filepath).stem.replace(' ', '_')[:50]
        self.full_quality_stl_path = os.path.join(temp_dir, f"hq_{safe_name}.stl")
        self.hq_tessellation_complete = False
        
        log(f"[STAGE 3] Starting background high-quality tessellation...")
        log(f"[STAGE 3] Output path: {self.full_quality_stl_path}")
        
        # Create Python script to run gmsh with high-quality settings
        script_content = f"""
import gmsh
import sys

try:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("General.Verbosity", 3)
    
    # Open CAD file
    gmsh.open(r"{filepath}")
    
    # HIGH-QUALITY settings for mesh generation
    bbox_diag = {complexity['bbox_diagonal']}
    gmsh.option.setNumber("Mesh.MeshSizeMin", bbox_diag / 100.0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", bbox_diag / 40.0)  # 2x finer than preview
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 5)  # Enable curvature adaptation
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
    gmsh.option.setNumber("Mesh.MinimumCirclePoints", 12)
    
    # Enable parallel meshing
    import os
    gmsh.option.setNumber("General.NumThreads", os.cpu_count() or 4)
    
    print("[STAGE 3 SUBPROCESS] Generating high-quality mesh...")
    gmsh.model.mesh.generate(2)
    
    print("[STAGE 3 SUBPROCESS] Writing STL...")
    gmsh.write(r"{self.full_quality_stl_path}")
    
    gmsh.finalize()
    print("[STAGE 3 SUBPROCESS] Complete!")
    sys.exit(0)
    
except Exception as e:
    print(f"[STAGE 3 SUBPROCESS ERROR] {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
        
        # Write script to temp file
        script_path = os.path.join(temp_dir, f"hq_tess_{safe_name}.py")
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        log(f"[STAGE 3] Script written to: {script_path}")
        
        # Start subprocess
        try:
            import sys
            self.hq_tessellation_process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            
            log(f"[STAGE 3] Background process started (PID: {self.hq_tessellation_process.pid})")
            
            # Start monitoring thread
            QTimer.singleShot(5000, lambda: self._check_hq_tessellation_progress())
            
        except Exception as e:
            log(f"[STAGE 3 ERROR] Failed to start subprocess: {e}")
            self.full_quality_stl_path = None
    
    def _check_hq_tessellation_progress(self):
        """Monitor background tessellation progress"""
        if not self.hq_tessellation_process:
            return
        
        log_file = r"C:\Users\Owner\Downloads\cad_load_debug.txt"
        
        def log(msg):
            print(msg, flush=True)
            try:
                with open(log_file, 'a', encoding='utf-8') as f:
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    f.write(f"[{timestamp}] {msg}\n")
            except:
                pass
        
        # Check if process finished
        returncode = self.hq_tessellation_process.poll()
        
        if returncode is not None:
            # Process finished
            if returncode == 0:
                # Success
                import os
                if os.path.exists(self.full_quality_stl_path) and os.path.getsize(self.full_quality_stl_path) > 1000:
                    self.hq_tessellation_complete = True
                    file_size_mb = os.path.getsize(self.full_quality_stl_path) / (1024 * 1024)
                    log(f"[STAGE 3] ✓ High-quality tessellation complete! ({file_size_mb:.2f} MB)")
                    log(f"[STAGE 3] Mesh generation will use: {self.full_quality_stl_path}")
                else:
                    log(f"[STAGE 3] ✗ Process completed but STL not found or too small")
                    self.full_quality_stl_path = None
            else:
                # Error
                log(f"[STAGE 3] ✗ Background tessellation failed with code {returncode}")
                self.full_quality_stl_path = None
        else:
            # Still running, check again later
            log("[STAGE 3] Background tessellation still running...")
            QTimer.singleShot(10000, lambda: self._check_hq_tessellation_progress())
    
    def get_hq_stl_for_meshing(self) -> str:
        """
        Get path to high-quality STL for mesh generation.
        Returns cached path if available, otherwise None (caller should tessellate on-demand).
        """
        if self.hq_tessellation_complete and self.full_quality_stl_path:
            import os
            if os.path.exists(self.full_quality_stl_path):
                print(f"[MESH GEN] Using cached high-quality STL: {self.full_quality_stl_path}")
                return self.full_quality_stl_path
        
        print(f"[MESH GEN] High-quality STL not ready, will tessellate on-demand")
        return None
    
    def load_step_file(self, filepath: str):
        """Load CAD file with loading overlay (deferred to allow UI update)"""
        print(f"[CAD PREVIEW] ========== load_step_file called: {filepath} ==========", flush=True)
        
        # Clear view
        try:
            self.clear_view()
            print("[CAD PREVIEW] View cleared successfully")
        except Exception as ee:
            print(f"[CAD PREVIEW] clear_view failed: {ee}", flush=True)
        
        self.quality_label.setVisible(False)
        self.info_label.setText(f"Loading: {Path(filepath).name}")
        
        # Show loading overlay with initial message
        print("[CAD PREVIEW] Showing loading overlay...")
        self.loading_overlay.set_message("Loading CAD file...", "Analyzing complexity...")
        self.loading_overlay.show()
        self.loading_overlay.raise_()  # Bring to front
        
        # Process events to ensure overlay is visible before we block
        from PyQt5.QtWidgets import QApplication
        QApplication.processEvents()
        
        print(f"[CAD PREVIEW] Loading overlay visible: {self.loading_overlay.isVisible()}")
        
        # Defer actual loading slightly to allow overlay to render
        # Note: gmsh MUST run in main thread (cannot use QThread due to signal handlers)
        QTimer.singleShot(50, lambda: self._do_cad_load(filepath))
        
        print("[CAD PREVIEW] CAD loading deferred to allow UI update...")
        
        # Return None for now - geom_info will be stored in self.current_geom_info when ready
        return None
    
    def _detect_cad_complexity(self, filepath: str) -> dict:
        """Detect CAD file complexity and return adaptive settings"""
        import gmsh
        
        # Update UI to show we're reading the file
        self.loading_overlay.set_message("Reading CAD file...", "Opening with gmsh...")
        from PyQt5.QtWidgets import QApplication
        QApplication.processEvents()
        
        try:
            if not gmsh.is_initialized():
                gmsh.initialize()
            
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("General.Verbosity", 0)
            
            # This is the slow part for large files
            self.loading_overlay.set_message("Reading CAD file...", "Parsing geometry...")
            QApplication.processEvents()
            
            gmsh.open(filepath)
            
            # Count surfaces (gmsh entities with dimension 2)
            self.loading_overlay.set_message("Analyzing complexity...", "Counting surfaces...")
            QApplication.processEvents()
            
            surfaces = gmsh.model.getEntities(2)
            surface_count = len(surfaces)
            
            # Get bounding box
            bbox = gmsh.model.getBoundingBox(-1, -1)
            bbox_dims = [bbox[3]-bbox[0], bbox[4]-bbox[1], bbox[5]-bbox[2]]
            bbox_diag = (bbox_dims[0]**2 + bbox_dims[1]**2 + bbox_dims[2]**2)**0.5
            
            gmsh.finalize()
            
            # Adaptive quality settings
            if surface_count < 100:
                # Simple part - high quality preview
                quality_level = "High"
                mesh_size_divisor = 20.0
                enable_refinement = False
            elif surface_count < 500:
                # Medium part - good quality
                quality_level = "Medium"
                mesh_size_divisor = 12.0
                enable_refinement = True
            elif surface_count < 2000:
                # Complex assembly - medium quality
                quality_level = "Medium-Low"
                mesh_size_divisor = 8.0
                enable_refinement = True
            else:
                # Very complex assembly - fast coarse preview
                quality_level = "Coarse"
                mesh_size_divisor = 3.0
                enable_refinement = True
            
            return {
                "surface_count": surface_count,
                "bbox_diagonal": bbox_diag,
                "bbox_dims": bbox_dims,
                "quality_level": quality_level,
                "mesh_size_max": bbox_diag / mesh_size_divisor,
                "mesh_size_min": bbox_diag / 100.0,
                "enable_refinement": enable_refinement
            }
            
        except Exception as e:
            # Fallback to medium quality if detection fails
            return {
                "surface_count": 0,
                "bbox_diagonal": 100.0,
                "bbox_dims": [100, 100, 100],
                "quality_level": "Medium",
                "mesh_size_max": 10.0,
                "mesh_size_min": 1.0,
                "enable_refinement": False,
                "error": str(e)
            }
    
    def _do_cad_load(self, filepath: str):
        """Stage 1: Load CAD with adaptive quality (coarse preview for complex files)"""
        # Setup debug logging to file
        log_file = r"C:\Users\Owner\Downloads\cad_load_debug.txt"
        
        def log(msg):
            """Log to both console and file"""
            print(msg, flush=True)
            try:
                with open(log_file, 'a', encoding='utf-8') as f:
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    f.write(f"[{timestamp}] {msg}\n")
            except:
                pass
        
        def update_status(msg):
            """Update UI status label"""
            self.info_label.setText(f"<b>Loading CAD...</b><br>{msg}")
            from PyQt5.QtWidgets import QApplication
            QApplication.processEvents()
        
        # Clear log file at start
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f"=== CAD Load Debug Log - STAGE 1 (Coarse Preview) ===\n")
                f.write(f"File: {filepath}\n")
                f.write(f"="*50 + "\n\n")
        except:
            pass
        
        log("[STAGE 1] Starting coarse preview load...")
        update_status("Analyzing file complexity...")
        
        try:
            import pyvista as pv
            import gmsh
            import tempfile
            
            # STEP 1: Detect complexity and determine adaptive quality
            log("[STAGE 1] Detecting CAD complexity...")
            complexity = self._detect_cad_complexity(filepath)
            
            surface_count = complexity["surface_count"]
            quality_level = complexity["quality_level"]
            mesh_size_max = complexity["mesh_size_max"]
            mesh_size_min = complexity["mesh_size_min"]
            enable_refinement = complexity["enable_refinement"]
            
            log(f"[STAGE 1] Complexity Analysis:")
            log(f"  - Surface count: {surface_count}")
            log(f"  - Quality level: {quality_level}")
            log(f"  - Mesh size range: {mesh_size_min:.2f} - {mesh_size_max:.2f}")
            log(f"  - Enable refinement: {enable_refinement}")
            
            # Update overlay with time estimate based on complexity
            if surface_count > 5000:
                time_estimate = "Large assembly - this may take 2-5 minutes to tessellate"
                update_status(f"Detected: {surface_count:,} surfaces (Coarse quality)")
            elif surface_count > 2000:
                time_estimate = "Complex file - this may take 30-90 seconds"
                update_status(f"Detected: {surface_count:,} surfaces ({quality_level})")
            elif surface_count > 500:
                time_estimate = "Processing..."
                update_status(f"Detected: {surface_count:,} surfaces ({quality_level})")
            else:
                time_estimate = ""
                update_status(f"Detected: {surface_count:,} surfaces ({quality_level})")
            
            # Update loading overlay with time estimate
            if time_estimate:
                self.loading_overlay.set_message("Tessellating CAD file...", time_estimate)
                from PyQt5.QtWidgets import QApplication
                QApplication.processEvents()
            
            # Create temp file path for Stage 1 coarse preview
            tmp = tempfile.NamedTemporaryFile(suffix='.stl', delete=False)
            tmp_stl = tmp.name
            tmp.close()
            log(f"[STAGE 1] Temp STL file: {tmp_stl}")

            # --- STAGE 1 GMSH CONVERSION (COARSE) ---
            log("[STAGE 1] Starting coarse tessellation...")
            update_status(f"Tessellating ({quality_level})...")
            
            # Ensure fresh start
            try:
                if gmsh.is_initialized():
                    log("[STAGE 1] WARNING: gmsh already initialized, finalizing first...")
                    gmsh.finalize()
            except Exception as e:
                log(f"[STAGE 1] Gmsh cleanup exception (ok): {e}")
            
            gmsh.initialize()
            log("[STAGE 1] Gmsh initialized successfully")
            
            # Minimal verbosity for Stage 1
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("General.Verbosity", 1)

            try:
                update_status(f"Opening file...<br><small>{Path(filepath).name}</small>")
                log(f"[STAGE 1] Opening: {filepath}")
                gmsh.open(filepath)
                log("[STAGE 1] File opened successfully")

                # Geometry already opened above
                log(f"[STAGE 1] BBox: {complexity['bbox_dims'][0]:.2f} x {complexity['bbox_dims'][1]:.2f} x {complexity['bbox_dims'][2]:.2f}")
                log(f"[STAGE 1] BBox Diagonal: {complexity['bbox_diagonal']:.2f}")

                # --- ADAPTIVE TESSELLATION SETTINGS (STAGE 1: COARSE) ---
                update_status(f"Configuring {quality_level} quality...")
                log(f"[STAGE 1] Applying adaptive tessellation settings:")
                log(f"  - MeshSizeMin: {mesh_size_min:.2f} mm")
                log(f"  - MeshSizeMax: {mesh_size_max:.2f} mm")
                
                gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size_min)
                gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_max)
                gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)  # Disable for speed
                gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
                
                # Enable parallel meshing if available
                import os
                cpu_count = os.cpu_count() or 4
                gmsh.option.setNumber("General.NumThreads", cpu_count)
                log(f"  - NumThreads: {cpu_count}")
                log("[STAGE 1] Tessellation parameters configured")
                
                # Generate 2D mesh (coarse for Stage 1)
                if surface_count > 2000:
                    update_status(f"Generating coarse mesh...<br><small>{surface_count:,} surfaces - this may take a minute</small>")
                else:
                    update_status("Generating surface mesh...")
                
                log("[STAGE 1] Starting mesh generation (coarse)...")
                import time
                start_time = time.time()
                gmsh.model.mesh.generate(2)
                elapsed = time.time() - start_time
                log(f"[STAGE 1] Mesh generation complete in {elapsed:.2f}s")
                
                # Write to temp STL
                update_status("Writing preview STL...")
                log(f"[STAGE 1] Writing to: {tmp_stl}")
                gmsh.write(tmp_stl)
                log("[STAGE 1] STL file written")
                
            finally:
                if gmsh.is_initialized():
                    gmsh.finalize()
                    log("[CAD PREVIEW] Gmsh finalized")

            # Check result
            update_status("Checking output...")
            if not os.path.exists(tmp_stl):
                raise Exception(f"STL file was not created at: {tmp_stl}")
            
            file_size = os.path.getsize(tmp_stl)
            log(f"[CAD PREVIEW] STL file size: {file_size} bytes")
            
            if file_size < 100:
                raise Exception(f"STL file is too small ({file_size} bytes), likely empty")

            # Load into PyVista
            update_status("Loading mesh into viewer...")
            log("[CAD PREVIEW] Reading STL with PyVista...")
            mesh = pv.read(tmp_stl)
            try:
                os.unlink(tmp_stl)
                log("[CAD PREVIEW] Temp file deleted")
            except Exception as e:
                log(f"[CAD PREVIEW] Could not delete temp file: {e}")

            log(f"[CAD PREVIEW] STL loaded: {mesh.n_points} points, {mesh.n_cells} cells")

            if mesh.n_points == 0:
                raise Exception("Empty mesh - no geometry in CAD file")

            update_status("Processing geometry...")
            poly_data = mesh.cast_to_unstructured_grid().extract_surface()
            log(f"[CAD PREVIEW] Surface extracted: {poly_data.GetNumberOfPoints()} points")
            
            # --- IMPROVED CAD VISUALIZATION PIPELINE ---
            processing_poly_data = poly_data
            cell_count = poly_data.GetNumberOfCells()
            
            if cell_count < 10000:
                update_status("Applying subdivision...")
                log(f"[CAD PREVIEW] Applying subdivision to {cell_count} cells...")
                subdivision = vtk.vtkLinearSubdivisionFilter()
                subdivision.SetInputData(poly_data)
                subdivision.SetNumberOfSubdivisions(1)
                subdivision.Update()
                processing_poly_data = subdivision.GetOutput()
                log("[CAD PREVIEW] Subdivision complete")
            
            update_status("Generating smooth normals...")
            log("[CAD PREVIEW] Generating smooth normals...")
            normals_gen = vtk.vtkPolyDataNormals()
            normals_gen.SetInputData(processing_poly_data)
            normals_gen.ComputePointNormalsOn()
            normals_gen.ComputeCellNormalsOff()
            normals_gen.SplittingOn() 
            normals_gen.SetFeatureAngle(60.0)
            normals_gen.Update()
            smooth_poly_data = normals_gen.GetOutput()
            log("[CAD PREVIEW] Normals generated")
            
            # Store poly data
            self.current_poly_data = smooth_poly_data
            
            # --- CREATE ACTOR ---
            update_status("Creating visualization...")
            log("[CAD PREVIEW] Creating VTK actor...")
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(smooth_poly_data)
            self.current_actor = vtk.vtkActor()
            self.current_actor.SetMapper(mapper)
            
            # Styling
            self.current_actor.GetProperty().SetColor(0.3, 0.5, 0.8)
            self.current_actor.GetProperty().SetInterpolationToPhong() 
            self.current_actor.GetProperty().BackfaceCullingOff()       
            self.current_actor.GetProperty().ShadingOn()
            self.current_actor.GetProperty().SetAmbient(0.3)
            self.current_actor.GetProperty().SetDiffuse(0.7)
            self.current_actor.GetProperty().SetSpecular(0.5)          
            self.current_actor.GetProperty().SetSpecularPower(40.0)
            self.current_actor.GetProperty().EdgeVisibilityOff()

            self.renderer.AddActor(self.current_actor)
            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()
            log("[CAD PREVIEW] Actor added to renderer")

            # Calculate volume
            update_status("Calculating properties...")
            log("[CAD PREVIEW] Calculating volume...")
            mass = vtk.vtkMassProperties()
            mass.SetInputData(smooth_poly_data)
            mass.Update()
            volume = mass.GetVolume()
            
            # Unit detection
            bounds = smooth_poly_data.GetBounds()
            dim_x = bounds[1]-bounds[0]
            dim_y = bounds[3]-bounds[2]
            dim_z = bounds[5]-bounds[4]
            max_dim = max(dim_x, dim_y, dim_z)
            bbox_diag = (dim_x**2 + dim_y**2 + dim_z**2)**0.5
            
            unit_scale = 1.0
            unit_name = 'm'
            
            if max_dim > 5.0:
                unit_name = 'mm'
                unit_scale = 0.001
                volume_text = f"<br>Volume: {volume:.2f} mm³"
            else:
                volume_text = f"<br>Volume: {volume:.6f} m³"
            
            log(f"[CAD PREVIEW] Volume: {volume} {unit_name}³")
            log(f"[CAD PREVIEW] Units detected: {unit_name}")

            self.info_label.setText(
                f"<b>CAD Preview</b><br>"
                f"{Path(filepath).name}<br>"
                f"<span style='color: #6c757d;'>{smooth_poly_data.GetNumberOfPoints():,} nodes{volume_text}</span>"
            )
            
            # Store geom_info
            self.current_cad_path = filepath
            self.current_geom_info = {
                "volume": volume * (unit_scale ** 3),
                "bbox_diagonal": bbox_diag * unit_scale,
                "units_detected": unit_name,
                "volume_display": volume,
                "node_count": smooth_poly_data.GetNumberOfPoints(),
                "filepath": filepath,
                "surface_count": surface_count,
                "quality_level": quality_level
            }
            
            log(f"[STAGE 1] Successfully loaded coarse preview: {filepath}")
            log(f"[STAGE 1] Preview quality: {quality_level} ({smooth_poly_data.GetNumberOfPoints():,} nodes)")
            
            # STAGE 3: Start background high-quality tessellation for mesh generation
            log(f"[STAGE 1] Starting Stage 3: Background high-quality tessellation for mesh generation...")
            try:
                self._start_hq_tessellation(filepath, complexity)
            except Exception as e3:
                log(f"[STAGE 3 ERROR] Failed to start background tessellation: {e3}")
                # Non-fatal - mesh generation will tessellate on-demand if needed
            
            log(f"[STAGE 1] Log file written to: {log_file}")
            
        except Exception as e:
            log(f"[STAGE 1 ERROR] Load failed: {e}")
            import traceback
            tb = traceback.format_exc()
            log(f"[STAGE 1 ERROR] Traceback:\n{tb}")
            self.info_label.setText(f"<b style='color:red'>CAD Load Error</b><br><small>{str(e)[:150]}</small><br>See: {log_file}")
            self.current_geom_info = None
        
        finally:
            # Hide loading overlay
            log("[STAGE 1] Hiding loading overlay...")
            self.loading_overlay.hide()
            log("[STAGE 1] ========== Stage 1 Complete ==========")
            log(f"\nDebug log saved to: {log_file}\n")
    
    def _on_cad_load_finished(self, smooth_poly_data, geom_info):
        """Handle successful CAD loading from worker thread"""
        print("[CAD PREVIEW] ========== _on_cad_load_finished CALLED ==========", flush=True)
        print(f"[CAD PREVIEW] Received poly_data: {smooth_poly_data.GetNumberOfPoints()} points, {smooth_poly_data.GetNumberOfCells()} cells")
        print(f"[CAD PREVIEW] Received geom_info: {geom_info}")
        
        # Hide loading overlay
        print("[CAD PREVIEW] Hiding loading overlay...")
        self.loading_overlay.hide()
        
        try:
            # Store poly data
            self.current_poly_data = smooth_poly_data
            print("[CAD PREVIEW] Stored poly_data")
            
            # --- CREATE ACTOR ---
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(smooth_poly_data)
            self.current_actor = vtk.vtkActor()
            self.current_actor.SetMapper(mapper)
            print("[CAD PREVIEW] Created actor and mapper")
            
            # Styling
            self.current_actor.GetProperty().SetColor(0.3, 0.5, 0.8)
            self.current_actor.GetProperty().SetInterpolationToPhong() 
            self.current_actor.GetProperty().BackfaceCullingOff()       
            self.current_actor.GetProperty().ShadingOn()
            self.current_actor.GetProperty().SetAmbient(0.3)
            self.current_actor.GetProperty().SetDiffuse(0.7)
            self.current_actor.GetProperty().SetSpecular(0.5)          
            self.current_actor.GetProperty().SetSpecularPower(40.0)
            self.current_actor.GetProperty().EdgeVisibilityOff()
            print("[CAD PREVIEW] Applied actor styling")

            self.renderer.AddActor(self.current_actor)
            print("[CAD PREVIEW] Added actor to renderer")
            self.renderer.ResetCamera()
            print("[CAD PREVIEW] Reset camera")
            self.vtk_widget.GetRenderWindow().Render()
            print("[CAD PREVIEW] Rendered scene")

            # Update info label
            volume_display = geom_info.get('volume_display', 0)
            unit_name = geom_info.get('units_detected', 'm')
            node_count = geom_info.get('node_count', 0)
            filepath = geom_info.get('filepath', '')
            
            if unit_name == 'mm':
                volume_text = f"<br>Volume: {volume_display:.2f} mm³"
            else:
                volume_text = f"<br>Volume: {volume_display:.6f} m³"

            self.info_label.setText(
                f"<b>CAD Preview</b><br>"
                f"{Path(filepath).name}<br>"
                f"<span style='color: #6c757d;'>{node_count:,} nodes{volume_text}</span>"
            )
            print("[CAD PREVIEW] Updated info label")
            
            # Store for later use
            self.current_cad_path = filepath
            self.current_geom_info = geom_info  # Store for main.py to access
            print(f"[CAD PREVIEW] Stored geom_info: {self.current_geom_info}")
            
            # Disable HQ worker (fast mode only)
            if self.hq_worker: 
                try:
                    self.hq_worker.finished.disconnect()
                    self.hq_worker.stop()
                except:
                    pass
            self.hq_worker = None
            
            print(f"[CAD PREVIEW] Successfully loaded: {filepath}")
            print("[CAD PREVIEW] ========== _on_cad_load_finished COMPLETE ==========", flush=True)
            
        except Exception as e:
            print(f"[CAD PREVIEW ERROR] Failed to setup visualization: {e}")
            import traceback
            traceback.print_exc()
            self._on_cad_load_error(str(e))
    
    def _on_cad_load_error(self, error_message: str):
        """Handle CAD loading error from worker thread"""
        print(f"[CAD PREVIEW ERROR] ========== _on_cad_load_error CALLED ==========", flush=True)
        print(f"[CAD PREVIEW ERROR] Worker failed: {error_message}")
        
        # Hide loading overlay
        self.loading_overlay.hide()
        
        # Show error in info label
        self.info_label.setText(
            f"CAD Load Error<br><small>{error_message[:100]}...</small><br>Click 'Generate Mesh'"
        )
        
        # Store None for geom_info
        self.current_geom_info = None
        print("[CAD PREVIEW ERROR] ========== _on_cad_load_error COMPLETE ==========", flush=True)

    def apply_quality_coloring(self, result: dict):
        """
        Store quality data and trigger visualization update.
        """
        if not self.current_poly_data or not self.current_actor:
            return

        print(f"[DEBUG] Storing quality data...")
        self.current_quality_data = result
        
        # Default Visualization (SICN, full opacity, full range)
        self.update_quality_visualization(
            metric="SICN", 
            opacity=1.0, 
            min_val=0.0, 
            max_val=1.0
        )

    def update_quality_visualization(self, metric="SICN", opacity=1.0, min_val=0.0, max_val=1.0):
        """
        Update mesh coloring based on selected metric and range.
        Handles threshold filtering via transparency.
        """
        if not hasattr(self, 'current_quality_data') or not self.current_quality_data:
            return

        # Map display name to data key
        metric_key = 'per_element_quality' # Default (SICN)
        if "gamma" in metric.lower():
            metric_key = 'per_element_gamma'
        elif "skew" in metric.lower():
            metric_key = 'per_element_skewness'
        elif "aspect" in metric.lower():
            metric_key = 'per_element_aspect_ratio'
            
        print(f"[DEBUG] Updating Viz: Metric={metric} (Key={metric_key}), Range=[{min_val:.2f}, {max_val:.2f}]")
        
        per_elem_data = self.current_quality_data.get(metric_key)
        if not per_elem_data:
            print(f"[WARN] No data found for metric key: {metric_key}")
            return

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

        # 1. Update Surface Scalars (Polygon Mesh)
        surface_colors = vtk.vtkUnsignedCharArray()
        surface_colors.SetNumberOfComponents(4) # RGBA (Alpha for hiding)
        surface_colors.SetName("Colors")

        # Create Map for fast lookup
        str_quality_map = {str(k): v for k, v in per_elem_data.items()}

        # Surface Elements Logic - CHANGED: Now using parent TET quality
        # Build tet quality map first
        tet_quality_map = {}
        for elem in (self.current_mesh_elements or []):
            if elem['type'] == 'tetrahedron':
                eid = str(elem['id'])
                qual = str_quality_map.get(eid)
                if qual is not None:
                    tet_quality_map[eid] = qual
        
        print(f"[DEBUG] Built tet quality map: {len(tet_quality_map)} tets with quality data")
        
        visible_surface_count = 0
        
        for i in range(self.current_poly_data.GetNumberOfCells()):
            # Get parent TET ID (not surface element ID)
            parent_tet_id = None
            if hasattr(self, 'surface_to_volume_map') and self.surface_to_volume_map:
                parent_tet_id = self.surface_to_volume_map.get(i)
            
            # Look up TET quality (not surface quality)
            tet_quality = None
            if parent_tet_id is not None:
                tet_quality = tet_quality_map.get(str(parent_tet_id))
            
            if tet_quality is None:
                # No parent tet or no quality data: Gray, Full Opacity
                surface_colors.InsertNextTuple4(150, 150, 150, int(opacity * 255))
            else:
                # Filter Logic: Hide if TET quality is outside range
                if tet_quality < min_val or tet_quality > max_val:
                    # Hidden (Alpha 0)
                    surface_colors.InsertNextTuple4(0, 0, 0, 0)
                else:
                    # Color Logic: Use TET quality for coloring
                    quality = tet_quality
                    
                    # Normalize quality value based on metric type
                    if "aspect" in metric.lower():
                        # AR: 1.0 = perfect (green), 3-5 = acceptable (yellow), 10+ = bad (red)
                        # Map [1,10] → [1,0] for color scale
                        n = 1.0 - min(1.0, max(0.0, (quality - 1.0) / 9.0))
                        
                        if quality < 1.5:
                            # Excellent AR < 1.5 = bright green
                            surface_colors.InsertNextTuple4(0, 200, 0, int(opacity * 255))
                        elif quality > 8.0:
                            # Very bad AR > 8 = deep red
                            surface_colors.InsertNextTuple4(220, 20, 60, int(opacity * 255))
                        else:
                            h = n * 0.33  # Already inverted by normalization formula
                            r, g, b = hsl_to_rgb(h, 1.0, 0.5)
                            surface_colors.InsertNextTuple4(r, g, b, int(opacity * 255))
                    elif "skew" in metric.lower():
                        # Skewness: 0 = perfect (green), 0.5 = acceptable, 1.0 = bad (red)
                        n = quality  # Already [0,1]
                        
                        if quality < 0.1:
                            surface_colors.InsertNextTuple4(0, 200, 0, int(opacity * 255))
                        elif quality > 0.85:
                            surface_colors.InsertNextTuple4(220, 20, 60, int(opacity * 255))
                        else:
                            h = (1.0 - n) * 0.33  # Invert: low skew = green
                            r, g, b = hsl_to_rgb(h, 1.0, 0.5)
                            surface_colors.InsertNextTuple4(r, g, b, int(opacity * 255))
                    else:
                        # SICN/Gamma: [0,1] where low is bad
                        n = max(0.0, min(1.0, quality))
                        
                        if quality < 0.2:
                            # Low quality for SICN/Gamma = Red
                            surface_colors.InsertNextTuple4(220, 20, 60, int(opacity * 255))
                        else:
                            h = n * 0.33  # Normal: 0→red, 1→green
                            r, g, b = hsl_to_rgb(h, 1.0, 0.5)
                            surface_colors.InsertNextTuple4(r, g, b, int(opacity * 255))
                    visible_surface_count += 1

        self.current_poly_data.GetCellData().SetScalars(surface_colors)
        
        # 2. Update Volume Scalars (for Cross-Section)
        if self.current_volumetric_grid and self.current_mesh_elements:
             vol_colors = vtk.vtkUnsignedCharArray()
             vol_colors.SetNumberOfComponents(4) # RGBA
             vol_colors.SetName("Colors")
             
             vol_count = 0
             for elem in self.current_mesh_elements:
                 if elem['type'] == 'tetrahedron': # Volume elements only
                     eid = str(elem['id'])
                     qual = str_quality_map.get(eid)
                     
                     if qual is None:
                         vol_colors.InsertNextTuple4(150, 150, 150, int(opacity * 255))
                     else:
                        if qual < min_val or qual > max_val:
                            vol_colors.InsertNextTuple4(0, 0, 0, 0) # Hide
                        else:
                            # Apply same normalized color logic as surface
                            if "aspect" in metric.lower():
                                # AR normalization [1,10]
                                n = 1.0 - min(1.0, max(0.0, (qual - 1.0) / 9.0))
                                
                                if qual < 1.5:
                                    vol_colors.InsertNextTuple4(0, 200, 0, int(opacity * 255))
                                elif qual > 8.0:
                                    vol_colors.InsertNextTuple4(220, 20, 60, int(opacity * 255))
                                else:
                                    h = n * 0.33
                                    r, g, b = hsl_to_rgb(h, 1.0, 0.5)
                                    vol_colors.InsertNextTuple4(r, g, b, int(opacity * 255))
                            elif "skew" in metric.lower():
                                # Skewness [0,1]
                                n = qual
                                
                                if qual < 0.1:
                                    vol_colors.InsertNextTuple4(0, 200, 0, int(opacity * 255))
                                elif qual > 0.85:
                                    vol_colors.InsertNextTuple4(220, 20, 60, int(opacity * 255))
                                else:
                                    h = (1.0 - n) * 0.33
                                    r, g, b = hsl_to_rgb(h, 1.0, 0.5)
                                    vol_colors.InsertNextTuple4(r, g, b, int(opacity * 255))
                            else:
                                # SICN/Gamma [0,1]
                                n = max(0.0, min(1.0, qual))
                                
                                if qual < 0.2:
                                    vol_colors.InsertNextTuple4(220, 20, 60, int(opacity * 255))
                                else:
                                    h = n * 0.33
                                    r, g, b = hsl_to_rgb(h, 1.0, 0.5)
                                    vol_colors.InsertNextTuple4(r, g, b, int(opacity * 255))
                            vol_count += 1
             
             if vol_colors.GetNumberOfTuples() == self.current_volumetric_grid.GetNumberOfCells():
                 self.current_volumetric_grid.GetCellData().SetScalars(vol_colors)
             else:
                 print(f"[WARN] Volumetric grid size mismatch in update_quality_visualization")

        # 3. Force Render Update
        if self.current_actor:
            # Enable transparency depth sorting if opacity < 1 or filtering active
            self.renderer.SetUseDepthPeeling(1)
            self.renderer.SetMaximumNumberOfPeels(8)
            
            # CRITICAL: High ambient to show bright colors without lighting interference
            self.current_actor.GetProperty().SetAmbient(0.8)
            self.current_actor.GetProperty().SetDiffuse(0.5)
            self.current_actor.GetProperty().SetSpecular(0.0)
            
            # CRITICAL: Re-enable wireframe AFTER setting scalars
            self.current_actor.GetProperty().EdgeVisibilityOn()
            self.current_actor.GetProperty().SetEdgeColor(0.0, 0.0, 0.0)
            self.current_actor.GetProperty().SetLineWidth(1.0)
            
            # Trigger Render
            self.vtk_widget.GetRenderWindow().Render()
            print(f"[DEBUG] Quality coloring applied. Surface: {visible_surface_count} visible.")
            
    def _init_zone_actors(self):
        """Initialize actors for zone highlighting and selection"""
        # 1. Selection Actor (Solid Blue Overlay)
        self.selected_faces_actor = vtk.vtkActor()
        mapper = vtk.vtkPolyDataMapper()
        self.selected_faces_actor.SetMapper(mapper)
        
        prop = self.selected_faces_actor.GetProperty()
        prop.SetColor(0.0, 0.3, 1.0) # Bright Blue
        prop.SetOpacity(1.0)
        prop.SetLighting(False) # Flat color
        prop.SetRepresentationToSurface() # SOLID, not wireframe
        prop.EdgeVisibilityOn() # Show edges on top
        prop.SetEdgeColor(0.0, 0.0, 0.5) # Dark blue edges
        prop.SetLineWidth(2)
        
        # Use 3D offset: render strictly on top
        mapper.SetResolveCoincidentTopologyToPolygonOffset()
        mapper.SetResolveCoincidentTopologyPolygonOffsetParameters(-4.0, -4.0) # More aggressive offset
        
        self.renderer.AddActor(self.selected_faces_actor)
        self.selected_faces_actor.VisibilityOff()

        # 2. Highlight Actor (Cyan Hover Preview)
        self.highlight_actor = vtk.vtkActor()
        self.highlighted_mapper = vtk.vtkPolyDataMapper()
        self.highlight_actor.SetMapper(self.highlighted_mapper)
        
        h_prop = self.highlight_actor.GetProperty()
        h_prop.SetColor(0.0, 0.8, 1.0) # Cyan
        h_prop.SetOpacity(0.6) # Slightly transparent to see mesh under
        h_prop.SetLighting(False)
        h_prop.SetRepresentationToSurface() # SOLID
        h_prop.EdgeVisibilityOn()
        h_prop.SetEdgeColor(0.0, 0.5, 1.0)
        h_prop.SetLineWidth(2)
        
        self.highlighted_mapper.SetResolveCoincidentTopologyToPolygonOffset()
        self.highlighted_mapper.SetResolveCoincidentTopologyPolygonOffsetParameters(-3.0, -3.0)
        
        self.renderer.AddActor(self.highlight_actor)
        self.highlight_actor.VisibilityOff()
        
        # State for Face Partition Mode
        self.show_face_partitions = False

    def toggle_face_partitions(self, show: bool):
        """Toggle visualization of logical face partitions (random colors)"""
        self.show_face_partitions = show
        if not self.current_actor:
            return
            
        if show:
            print("[VTKViewer] Enabling Face Partition Mode...")
            scalars = self._create_partition_scalars()
            if scalars:
                self.current_poly_data.GetCellData().SetScalars(scalars)
                
                # Update Mapper for random colors
                mapper = self.current_actor.GetMapper()
                mapper.SetScalarModeToUseCellData()
                mapper.ScalarVisibilityOn()
                mapper.SetColorModeToMapScalars()
                
                # Create a random lookup table
                lut = vtk.vtkLookupTable()
                lut.SetNumberOfTableValues(256)
                lut.SetTableRange(0, 255)
                lut.Build()
                
                # Seed random for consistency
                rng = np.random.RandomState(42)
                for i in range(256):
                    r, g, b = rng.rand(3)
                    lut.SetTableValue(i, r, g, b, 1.0)
                
                mapper.SetLookupTable(lut)
                mapper.SetScalarRange(0, 255)
                
                self.vtk_widget.GetRenderWindow().Render()
        else:
            print("[VTKViewer] Disabling Face Partition Mode - reverting...")
            # Revert to standard (or quality)
            # Just re-load the mesh normally to reset mapper state
            # A bit heavy but safe.
            # Faster: Check if quality mode was on. For now, just reset scalars.
            self.current_poly_data.GetCellData().SetScalars(None)
            mapper = self.current_actor.GetMapper()
            mapper.ScalarVisibilityOff()
            self.vtk_widget.GetRenderWindow().Render()

    def _create_partition_scalars(self):
        """Create VTK scalars array from partition map"""
        if not self.zone_manager.element_to_face_partition:
            return None
            
        num_cells = self.current_poly_data.GetNumberOfCells()
        scalars = vtk.vtkUnsignedCharArray()
        scalars.SetNumberOfComponents(1)
        scalars.SetNumberOfTuples(num_cells)
        scalars.SetName("FacePartition")
        
        for i in range(num_cells):
            if i < len(self.vtk_cell_to_face_id):
                face_id = self.vtk_cell_to_face_id[i]
                partition = self.zone_manager.element_to_face_partition.get(face_id, 0)
                # Hash partition ID to 0-255 for random color lookup
                val = (partition * 175) % 256
                scalars.SetTuple1(i, val)
            else:
                scalars.SetTuple1(i, 0)
                
        return scalars

    def initialize_zone_manager(self):
        """
        Called after mesh is loaded. 
        Populates zone manager with data and builds cell ID map.
        """
        print("[VTKViewer] Initializing Zone Manager...")
        if not self.current_mesh_elements: 
            print("[VTKViewer] Cancelled: No mesh elements found")
            return
            
        try:
            points_dict = self.current_mesh_nodes or {}
            elements = self.current_mesh_elements
            
            # Populate node map if missing (fallback)
            if not self.current_node_map:
                self.current_node_map = {nid: i for i, nid in enumerate(points_dict.keys())}
                
            num_points = len(points_dict)
            if num_points == 0:
                print("[VTKViewer] Cancelled: No points found")
                return
                
            points_array = np.zeros((num_points, 3))
            
            for nid, idx in self.current_node_map.items():
                if nid in points_dict:
                    points_array[idx] = points_dict[nid]
                
            mapped_elements = []
            
            # Setup ID maps for clicking
            self.vtk_cell_to_face_id = []
            self.vtk_cell_to_entity_tag = []  # CAD surface tag for each cell
            
            for elem in elements:
                etype = elem.get('type')
                if etype in ('triangle', 'quadrilateral'):
                    # Map node IDs to indices
                    raw_nodes = elem.get('nodes')
                    mapped_nodes = [self.current_node_map.get(n, 0) for n in raw_nodes] # Safe get
                    
                    e_copy = elem.copy()
                    e_copy['nodes'] = mapped_nodes
                    mapped_elements.append(e_copy)
                    
                    self.vtk_cell_to_face_id.append(elem['id'])
                    # Store CAD entity_tag if available
                    self.vtk_cell_to_entity_tag.append(elem.get('entity_tag', None))

            # FALLBACK: If no explicit surface elements, use extracted surface mapping
            if not self.vtk_cell_to_face_id and hasattr(self, 'surface_to_volume_map') and self.surface_to_volume_map:
                print(f"[VTKViewer] Using surface->volume mapping for {len(self.surface_to_volume_map)} cells")
                
                # CRITICAL FIX: Use the EXTRACTED SURFACE's points, not original mesh nodes!
                # The extracted surface has its own point numbering (0 to N-1)
                if self.current_poly_data and self.current_poly_data.GetPoints():
                    vtk_points = self.current_poly_data.GetPoints()
                    num_surface_points = vtk_points.GetNumberOfPoints()
                    print(f"[VTKViewer] Extracted surface has {num_surface_points} points")
                    
                    # Build points array from extracted surface
                    points_array = np.zeros((num_surface_points, 3))
                    for i in range(num_surface_points):
                        points_array[i] = vtk_points.GetPoint(i)
                
                sorted_keys = sorted(self.surface_to_volume_map.keys())
                for i in sorted_keys:
                    vol_elem_id = int(self.surface_to_volume_map[i]) 
                    self.vtk_cell_to_face_id.append(vol_elem_id)
                    
                    # Construct pseudo-element for ZoneManager adjacency
                    if self.current_poly_data is not None:
                        cell = self.current_poly_data.GetCell(i)
                        npts = cell.GetNumberOfPoints()
                        face_nodes = []
                        for j in range(npts):
                            pid = cell.GetPointId(j) 
                            face_nodes.append(pid)
                        
                        mapped_elements.append({
                            'id': vol_elem_id,
                            'type': 'triangle' if npts == 3 else 'quadrilateral',
                            'nodes': face_nodes
                        })
            
            if not self.vtk_cell_to_face_id:
                print("[VTKViewer] WARNING: No surface faces mapped for selection!")
                    
            # Initialize Reverse Map (One-to-Many)
            self.face_id_to_vtk_cell = {}
            for idx, fid in enumerate(self.vtk_cell_to_face_id):
                if fid not in self.face_id_to_vtk_cell:
                    self.face_id_to_vtk_cell[fid] = []
                self.face_id_to_vtk_cell[fid].append(idx)
                    
            # Initialize Manager with correct points
            self.zone_manager.set_mesh_data(points_array, mapped_elements)
            print(f"[VTKViewer] Zone Manager Initialized: {len(self.vtk_cell_to_face_id)} faces mapped.")
            
        except Exception as e:
            print(f"[VTKViewer] Error initializing zone manager: {e}")
            import traceback
            traceback.print_exc()

    def on_scene_click(self, x, y, shift=False, ctrl=False):
        """Handle scene click from interactor"""
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.005)
        
        if picker.Pick(x, y, 0, self.renderer):
            cell_id = picker.GetCellId()
            if cell_id >= 0 and cell_id < len(self.vtk_cell_to_face_id):
                face_id = self.vtk_cell_to_face_id[cell_id]
                
                # Look up CAD surface tag if available
                entity_tag = None
                if hasattr(self, 'vtk_cell_to_entity_tag') and cell_id < len(self.vtk_cell_to_entity_tag):
                    entity_tag = self.vtk_cell_to_entity_tag[cell_id]
                
                # Look up CAD surface name from physical groups
                cad_name = None
                if entity_tag and hasattr(self, 'current_physical_groups'):
                    for pg_tag, pg_data in self.current_physical_groups.items():
                        if pg_data.get('name', '').endswith(f'_{entity_tag}'):
                            cad_name = pg_data.get('name')
                            break
                
                if cad_name:
                    print(f"[VTK] Clicked Cell {cell_id} -> Face {face_id} (CAD Surface: {cad_name})")
                elif entity_tag:
                    print(f"[VTK] Clicked Cell {cell_id} -> Face {face_id} (Entity {entity_tag})")
                else:
                    print(f"[VTK] Clicked Cell {cell_id} -> Face {face_id}")
                
                self.zone_manager.select_face(face_id, toggle=ctrl, multi_select=shift)
                self.update_zone_visuals()
        else:
            # Clicked background -> Deselect all
            if not shift and not ctrl:
                self.zone_manager.selected_faces.clear()
                self.update_zone_visuals()

    def on_scene_hover(self, x, y):
        """Handle mouse hover to highlight faces"""
        if not self.current_poly_data:
            return
            
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.005)
        
        if picker.Pick(x, y, 0, self.renderer):
            cell_id = picker.GetCellId()
            if cell_id >= 0:
                if cell_id < len(self.vtk_cell_to_face_id):
                    face_id = self.vtk_cell_to_face_id[cell_id]
                    
                    if face_id != self.zone_manager.highlighted_face:
                        self.zone_manager.highlighted_face = face_id
                        self.update_highlight_visuals()
                else:
                    pass
        else:
            if self.zone_manager.highlighted_face is not None:
                self.zone_manager.highlighted_face = None
                self.update_highlight_visuals()
                
    def update_highlight_visuals(self):
        """Update the highlight actor based on hover state (Highlight ENTIRE Partition)"""
        if self.zone_manager.highlighted_face is None:
            self.highlight_actor.VisibilityOff()
            self.vtk_widget.GetRenderWindow().Render()
            return
            
        # Get partition ID of hovered face
        fid = self.zone_manager.highlighted_face
        partition_id = self.zone_manager.element_to_face_partition.get(fid)
        
        # If we have a partition ID, highlight ALL faces in that partition
        faces_to_highlight = []
        if partition_id is not None:
            # Slow scan? We can optimize this by storing partition->faces map
            # For now, immediate feedback is key, let's just highlight the single face 
            # OR do the scan if the mesh isn't huge.
            # Optimization: ZoneManager should have partition->faces map?
            # Let's fallback to just checking the map, iterating all cells is fast enough for <100k faces
            pass
            
        # Creating selection data for potentially 1000s of faces per frame on hover is too slow
        # compromise: Highlight single cell for now, or implement partition actor caching.
        # User requested: "entire face that would get selected"
        # I will do a quick lookup: Scan self.vtk_cell_to_face_id -> partition_map check
        
        if partition_id is None:
             # Just highlight single cell
             # super().update_highlight_visuals() # wait, this method replaces it.
             faces_to_highlight = [fid]
        else:
             # Find all cells belonging to this partition
             # This loop in Python might be slow for 1M+ elements.
             # Optimization: We only need to find cells visible?
             # Let's try iterating.
             found_indices = []
             for i, test_fid in enumerate(self.vtk_cell_to_face_id):
                 if self.zone_manager.element_to_face_partition.get(test_fid) == partition_id:
                     found_indices.append(i)
             
             if not found_indices:
                 faces_to_highlight = [fid]
             else:
                 # Create selection node
                 ids = vtk.vtkIdTypeArray()
                 ids.SetNumberOfComponents(1)
                 for idx in found_indices:
                     ids.InsertNextValue(idx)
                 
                 selection_node = vtk.vtkSelectionNode()
                 selection_node.SetFieldType(vtk.vtkSelectionNode.CELL)
                 selection_node.SetContentType(vtk.vtkSelectionNode.INDICES)
                 selection_node.SetSelectionList(ids)
                 
                 selection = vtk.vtkSelection()
                 selection.AddNode(selection_node)
                 
                 extract = vtk.vtkExtractSelection()
                 extract.SetInputData(0, self.current_poly_data)
                 extract.SetInputData(1, selection)
                 extract.Update()
                 
                 self.highlighted_mapper.SetInputConnection(extract.GetOutputPort())
                 self.highlight_actor.VisibilityOn()
                 self.vtk_widget.GetRenderWindow().Render()
                 return

        # Fallback for single face
        cell_indices = self.face_id_to_vtk_cell.get(fid, [])
        if not cell_indices:
            self.highlight_actor.VisibilityOff()
            return
            
        ids = vtk.vtkIdTypeArray()
        ids.SetNumberOfComponents(1)
        for idx in cell_indices:
            ids.InsertNextValue(idx)
            
        selection_node = vtk.vtkSelectionNode()
        selection_node.SetFieldType(vtk.vtkSelectionNode.CELL)
        selection_node.SetContentType(vtk.vtkSelectionNode.INDICES)
        selection_node.SetSelectionList(ids)
        
        selection = vtk.vtkSelection()
        selection.AddNode(selection_node)
        
        extract = vtk.vtkExtractSelection()
        extract.SetInputData(0, self.current_poly_data)
        extract.SetInputData(1, selection)
        extract.Update()
        
        self.highlighted_mapper.SetInputConnection(extract.GetOutputPort())
        self.highlight_actor.VisibilityOn()
        self.vtk_widget.GetRenderWindow().Render()
        # selected_poly.Update() # This line is from the original code, but `selected_poly` is not defined here.
        
        # self.highlight_actor.GetMapper().SetInputData(selected_poly.GetOutput()) # This line is from the original code, but `selected_poly` is not defined here.
        # self.highlight_actor.VisibilityOn() # This line is redundant as it's already set above.
        # self.vtk_widget.GetRenderWindow().Render() # This line is redundant as it's already set above.

    
    def check_edge_visibility(self):
        """Dynamic edge visibility based on zoom level"""
        if not self.current_actor: return
        
        cam = self.renderer.GetActiveCamera()
        if not cam: return
        
        # Calculate pixels per world unit
        # Height of the view in world coordinates at the focal plane
        if cam.GetParallelProjection():
            view_height_world = 2.0 * cam.GetParallelScale()
        else:
            view_height_world = 2.0 * cam.GetDistance() * np.tan(np.deg2rad(cam.GetViewAngle()) / 2.0)
        
        # Get window height in pixels
        size = self.vtk_widget.GetRenderWindow().GetSize()
        win_height_px = size[1]
        
        if view_height_world <= 0: return

        pixels_per_world = win_height_px / view_height_world
        cell_size_px = self.avg_cell_size * pixels_per_world
        
        MIN_PIXELS_PER_CELL = 5.0
        
        prop = self.current_actor.GetProperty()
        current_vis = prop.GetEdgeVisibility()
        
        # Dynamic Line Width
        # Thinner lines when zoomed out (but not less than 1.0)
        target_width = max(1.0, min(1.5, cell_size_px / 40.0))
        prop.SetLineWidth(target_width)
        
        # print(f"[VIEWER_DEBUG] target_width: {target_width:.2f} (cell_px: {cell_size_px:.1f})")

        # Hysteresis to prevent flickering
        if current_vis == 1:
            if cell_size_px < (MIN_PIXELS_PER_CELL * 0.8):
                prop.SetEdgeVisibility(0)
                print(f"[VIEWER_DEBUG] Hiding edges (cell_px={cell_size_px:.1f} < {MIN_PIXELS_PER_CELL * 0.8})")
                self.vtk_widget.GetRenderWindow().Render()
        else:
            if cell_size_px > (MIN_PIXELS_PER_CELL * 1.2):
                prop.SetEdgeVisibility(1)
                print(f"[VIEWER_DEBUG] Showing edges (cell_px={cell_size_px:.1f} > {MIN_PIXELS_PER_CELL * 1.2})")
                self.vtk_widget.GetRenderWindow().Render()

    def _load_vtk_file(self, filepath: str, result: dict = None):
        """Load legacy VTK file with thermal results"""
        print(f"[DEBUG] Loading VTK file: {filepath}")
        try:
            import pyvista as pv
            import vtk
            
            # Load with PyVista
            mesh = pv.read(filepath)
            self.current_volumetric_grid = mesh
            
            # Check for Temperature scalar
            scalar_name = None
            if "Temperature" in mesh.point_data:
                scalar_name = "Temperature"
            elif "Temperature" in mesh.cell_data:
                scalar_name = "Temperature"
                
            # Extract surface for display
            poly_data = mesh.extract_surface()
            self.current_poly_data = poly_data
            
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(poly_data)
            
            if scalar_name:
                print(f"[DEBUG] Found thermal data: {scalar_name}")
                rng = mesh.get_data_range(scalar_name)
                print(f"[DEBUG] Temperature range: {rng}")
                
                mapper.SetScalarRange(rng)
                if scalar_name in mesh.point_data:
                    mapper.SetScalarModeToUsePointData()
                else:
                    mapper.SetScalarModeToUseCellData()
                mapper.ScalarVisibilityOn()
                
                # Thermal Colormap (Turbo/Jet-like)
                lut = vtk.vtkLookupTable()
                lut.SetNumberOfTableValues(256)
                lut.Build()
                
                ctf = vtk.vtkColorTransferFunction()
                ctf.AddRGBPoint(rng[0], 0.0, 0.0, 1.0)      # Blue
                ctf.AddRGBPoint(rng[0] + 0.35*(rng[1]-rng[0]), 0.0, 1.0, 1.0) # Cyan
                ctf.AddRGBPoint(rng[0] + 0.65*(rng[1]-rng[0]), 1.0, 1.0, 0.0) # Yellow
                ctf.AddRGBPoint(rng[1], 1.0, 0.0, 0.0)      # Red
                
                for i in range(256):
                    val = rng[0] + (i / 255.0) * (rng[1] - rng[0])
                    rgb = ctf.GetColor(val)
                    lut.SetTableValue(i, rgb[0], rgb[1], rgb[2], 1.0)
                
                mapper.SetLookupTable(lut)
                
                # Scalar Bar
                scalar_bar = vtk.vtkScalarBarActor()
                scalar_bar.SetLookupTable(lut)
                scalar_bar.SetTitle("Temperature (K)")
                scalar_bar.SetNumberOfLabels(5)
                scalar_bar.SetPosition(0.85, 0.1)
                scalar_bar.SetWidth(0.12)
                scalar_bar.SetHeight(0.8)
                self.renderer.AddActor2D(scalar_bar)
                
            self.current_actor = vtk.vtkActor()
            self.current_actor.SetMapper(mapper)
            self.renderer.AddActor(self.current_actor)
            
            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()
            self.info_label.setText(f"Loaded: {Path(filepath).name}")
            return "SUCCESS"
            
        except Exception as e:
            print(f"[ERROR] Failed to load VTK: {e}")
            import traceback
            traceback.print_exc()
            self.info_label.setText("Error loading VTK")
            return "FAILED"

    def load_mesh_file(self, filepath: str, result: dict = None):
        """Load and display mesh file with optional result dict for counts"""
        # Dispatch to VTK loader if legacy VTK file
        if Path(filepath).suffix.lower() == '.vtk':
            return self._load_vtk_file(filepath, result)

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
            # Store filepath for update_info_label method
            self.current_mesh_file = filepath
            
<<<<<<< Updated upstream
            if filepath.lower().endswith('.vtu'):
                print(f"[DEBUG] Detected VTU file, using standard VTK reader...", flush=True)
                reader = vtk.vtkXMLUnstructuredGridReader()
                reader.SetFileName(filepath)
                reader.Update()
                self.current_poly_data = reader.GetOutput()
=======
            # Detect file type
            file_ext = Path(filepath).suffix.lower()
            
            # Handle STL/VTU files differently - they are surface meshes, not volumetric
            if file_ext in ['.stl', '.vtu', '.vtk', '.ply', '.obj']:
                return self._load_surface_mesh_file(filepath, file_ext)
            
            print(f"[DEBUG] Parsing .msh file...")
            nodes, elements, physical_groups = self._parse_msh_file(filepath)
            self.current_physical_groups = physical_groups  # Store for zone selection
            print(f"[DEBUG] Parsed {len(nodes)} nodes, {len(elements)} elements, {len(physical_groups)} physical groups")
            # Load boxed parts metadata if available
            boxed_json_path = filepath.replace(".msh", "_boxed.json")
            self.boxed_physical_tags = []
            if Path(boxed_json_path).exists():
                try:
                    import json
                    with open(boxed_json_path, 'r') as f:
                        boxed_data = json.load(f)
                    self.boxed_physical_tags = boxed_data.get("boxed_physical_tags", [])
                    print(f"[DEBUG] Loaded {len(self.boxed_physical_tags)} boxed physical tags")
                    if self.boxed_physical_tags:
                        self.boxed_highlight_checkbox.setVisible(True)
                except Exception as e:
                    print(f"[DEBUG] Could not load boxed parts data: {e}")
            else:
                self.boxed_highlight_checkbox.setVisible(False)

            # Try to load surface quality data if not provided
            if not (result and result.get('per_element_quality')):
                quality_file = Path(filepath).with_suffix('.quality.json')
                print(f"[DEBUG] Checking for quality file: {quality_file}")
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
                        print(f"[DEBUG] [OK][OK]Loaded quality data for {len(result['per_element_quality'])} elements")
                    except Exception as e:
                        print(f"[DEBUG] [X][X]Could not load quality data: {e}")
                else:
                    # Compute quality from mesh file using Gmsh
                    if len(elements) > 1000000:
                        print(f"[DEBUG] Very large mesh ({len(elements)} elements) - skipping auto-quality to avoid long hang")
                        self.log_requested.emit(f"Notice: Mesh too large ({len(elements)} elements) for automatic quality preview.")
                    else:
                        print(f"[DEBUG] Computing quality from mesh ({len(elements)} elements)...")
                        quality_result = self._calculate_mesh_quality(filepath)
                        if quality_result:
                            if not result: result = {}
                            result.update(quality_result)
                            self.current_quality_data = quality_result
            else:
                print(f"[DEBUG] Quality data already in result dict")
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
            
            # Check if we have surface elements
            has_surface_elements = (tri_count > 0 or quad_count > 0)
            
            if has_surface_elements:
                print(f"[DEBUG] Found explicit surface elements ({tri_count} tris, {quad_count} quads)")
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
            else:
                print(f"[DEBUG] No explicit surface elements found. Extracting surface from volume...")
>>>>>>> Stashed changes
                
                # Extract surface for rendering
                surf_filter = vtk.vtkDataSetSurfaceFilter()
                surf_filter.SetInputData(self.current_poly_data)
                surf_filter.Update()
                poly_to_render = surf_filter.GetOutput()
                
                self.current_actor = vtk.vtkActor()
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(poly_to_render)
                self.current_actor.SetMapper(mapper)
                
                # Store minimal data for internal logic
                # (Note: manual node parsing skipped for VTU, features requiring it might be limited)
                self.current_mesh_elements = [] 
                
                display_nodes = self.current_poly_data.GetNumberOfPoints()
                display_elements = self.current_poly_data.GetNumberOfCells()

            else:
                print(f"[DEBUG] Parsing .msh file...", flush=True)
                nodes, elements, physical_groups = self._parse_msh_file(filepath)
                self.current_physical_groups = physical_groups
                
                # CRITICAL FIX: Store parsed data for ZoneManager and Quality Visualization
                self.current_mesh_nodes = nodes
                self.current_mesh_elements = elements
                
                # ... (rest of legacy MSH logic will be skipped or we need to restructure)
                # RESTRUCTURE REQUIRED: The function assumes 'nodes' and 'elements' exist.
                # We need to branch the rendering logic.

                # Legacy MSH Parsing Logic - Indented
                # Try to load surface quality data if not provided
                if not (result and result.get('per_element_quality')):
                    quality_file = Path(filepath).with_suffix('.quality.json')
                    print(f"[DEBUG] Checking for quality file: {quality_file}")
                    if quality_file.exists():
                        try:
                            import json
                            with open(quality_file, 'r') as f:
                                surface_quality = json.load(f)

                            if not result:
                                result = {}
                            result['per_element_quality'] = surface_quality.get('per_element_quality', {})
                            
                            self.current_quality_data = surface_quality
                            
                            result['quality_metrics'] = {
                                'sicn_10_percentile': surface_quality.get('quality_threshold_10', 0.3),
                                'sicn_min': surface_quality.get('statistics', {}).get('min_quality', 0.0),
                                'sicn_avg': surface_quality.get('statistics', {}).get('avg_quality', 0.5),
                                'sicn_max': surface_quality.get('statistics', {}).get('max_quality', 1.0)
                            }
                            print(f"[DEBUG] [OK][OK]Loaded quality data")
                        except Exception as e:
                            print(f"[DEBUG] [X][X]Could not load quality data: {e}")
                    else:
                        print(f"[DEBUG] No quality file found, computing from mesh if small enough...")
                        # (Skipping huge Gmsh compute block for brevity, assuming standard flow)
                
                print(f"[DEBUG] Creating VTK data structures manually for MSH...")
                points = vtk.vtkPoints()
                cells = vtk.vtkCellArray()

                node_map = {}
                for idx, (node_id, coords) in enumerate(nodes.items()):
                    points.InsertNextPoint(coords)
                    node_map[node_id] = idx

                # Count elements for display
                tet_count = sum(1 for e in elements if e['type'] == 'tetrahedron')
                tri_count = sum(1 for e in elements if e['type'] == 'triangle')
                
                # Check if we have surface elements
                has_surface_elements = (tri_count > 0)
                
                if has_surface_elements:
                    print(f"[DEBUG] Surface elements found (Tri: {tri_count}, Quad: {sum(1 for e in elements if e['type'] == 'quadrilateral')})")
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
                    
                    poly_data.SetPoints(points)
                    poly_data.SetPolys(cells)

                else:
                    # Volume elements only - Extract surface
                    print(f"[DEBUG] No surface elements. Extracting skin from {tet_count} volume elements...")
                    
                    ugrid = vtk.vtkUnstructuredGrid()
                    ugrid.SetPoints(points)
                    
                    # Efficiently build cell array for Tets
                    # Format: [n_pts, id0, id1, id2, id3, n_pts, ...]
                    # Using pure python list append is faster than vtkTetra loop
                    cell_list = []
                    for element in elements:
                        if element['type'] == 'tetrahedron':
                            cell_list.append(4)
                            cell_list.extend([node_map[nid] for nid in element['nodes']])
                    
                    if cell_list:
                        import numpy as np
                        # Use numpy for speed if available, else standard vtk id list
                        # vtkCellArray imports usually handle numpy arrays/lists
                        # Create vtkCellArray
                        
                        # ID array
                        raw_ids = cell_list
                        # Convert to vtkIdTypeArray
                        id_array = vtk.vtkIdTypeArray()
                        id_array.SetNumberOfValues(len(raw_ids))
                        
                        # Populate manually (slow) or assume list works?
                        # list argument to InsertNextCell is for single cell.
                        # For bulk, we use import utility or loop.
                        # Fast path: Use PyVista if available? No, stick to VTK for this legacy block.
                        
                        # Medium speed: Loop ints
                        # Correct "Bulk" way in pure VTK python is tricky without numpy integration.
                        # We will use the standard loop for correctness, it's 250k elements, might take 1-2s.
                        
                        vol_cells = vtk.vtkCellArray()
                        # Pre-allocating
                        vol_cells.Allocate(len(elements) * 5)
                        
                        # Keep track of element IDs corresponding to volume cells
                        vol_idx_to_elem_id = []
                        
                        for element in elements:
                            if element['type'] == 'tetrahedron':
                                # Build tet manually - safer than manual array hacking without types
                                tet = vtk.vtkTetra()
                                for i, node_id in enumerate(element['nodes']):
                                    tet.GetPointIds().SetId(i, node_map[node_id])
                                vol_cells.InsertNextCell(tet)
                                vol_idx_to_elem_id.append(element['id'])
                        
                        ugrid.SetCells(vtk.VTK_TETRA, vol_cells)
                        
                        # CRITICAL FIX: Store volume grid for cross-sections
                        self.current_volumetric_grid = ugrid
                        
                        # Filter surface (enabling original cell ID tracking)
                        print("[DEBUG] Running vtkDataSetSurfaceFilter...")
                        surf_filter = vtk.vtkDataSetSurfaceFilter()
                        surf_filter.SetInputData(ugrid)
                        surf_filter.PassThroughCellIdsOn() # CRITICAL: Ask for original IDs
                        surf_filter.Update()
                        poly_data = surf_filter.GetOutput()
                        
                        # Extract Original Cell IDs to map Surface -> Volume Element ID
                        print(f"[DEBUG] Surface extraction complete. Faces: {poly_data.GetNumberOfCells()}")
                        
                        orig_ids = poly_data.GetCellData().GetArray("vtkOriginalCellIds")
                        if orig_ids:
                             print(f"[DEBUG] Mapping surface cells to volume IDs via vtkOriginalCellIds...")
                             self.surface_to_volume_map = {}
                             for i in range(poly_data.GetNumberOfCells()):
                                 vol_idx = orig_ids.GetValue(i)
                                 if vol_idx < len(vol_idx_to_elem_id):
                                      elem_id = vol_idx_to_elem_id[vol_idx]
                                      self.surface_to_volume_map[i] = elem_id
                             print(f"[DEBUG] Mapped {len(self.surface_to_volume_map)} surface faces to volume elements")
                        else:
                             print(f"[WARN] vtkOriginalCellIds array not found in surface filter output!")
                    else:
                        print("[WARN] No tets found either? Empty mesh.")
                
                # CRITICAL: Store volume elements for cross-section functionality
                self.current_tetrahedra = [e for e in elements if e['type'] == 'tetrahedron']
                self.current_hexahedra = [e for e in elements if e['type'] == 'hexahedron']
                print(f"[DEBUG] Stored {len(self.current_tetrahedra)} tets and {len(self.current_hexahedra)} hexes for cross-sections")
                
                # Assign to shared variable so common render code can use it?
                # Actually, MSH legacy code creates actor MANUALLY.
                # Use a shared 'poly_to_render' variable.
                poly_to_render = poly_data
                self.current_poly_data = poly_data  # CRITICAL: Store for coloring/brush
                
                # Update shared counts
                display_nodes = len(nodes)
                display_elements = len(elements)
                
                # --- END MSH LEGACY BLOCK ---

            # COMMON RENDERING BLOCK
            # At this point, 'poly_to_render' should exist.
            
            # If VTK actor was not created in VTU block (MSH path), create it now
            if not self.current_actor:
                self.current_actor = vtk.vtkActor()
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(poly_to_render)
                self.current_actor.SetMapper(mapper)

            # Apply robust visibility settings (Common for both)
            self.current_actor.GetProperty().SetRepresentationToSurface()
            self.current_actor.GetProperty().BackfaceCullingOff()
            self.current_actor.GetProperty().EdgeVisibilityOn()  # Enable wireframe
            self.current_actor.GetProperty().SetEdgeColor(0.0, 0.0, 0.0)
            self.current_actor.GetProperty().SetLineWidth(1.0)
            
            print(f"[VIEWER_DEBUG] Actor created and configured (VTU/MSH agnostic)")
            self.renderer.AddActor(self.current_actor)
            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()
            
            # Update Info Label using dedicated method (supports quality metrics)
            if result:
                 self.update_info_label(result)
            else:
                 # Fallback manual update if no result dict
                 # Update Info Label (Common)
                 info_lines = [
                    f"<b>Mesh Generated</b><br>",
                    f"{Path(filepath).name}<br>",
                    f"<span style='color: #6c757d;'>",
                    f"Nodes: {display_nodes:,} * Elements: {display_elements:,}<br>",
                    f"Format: {'VTU' if filepath.lower().endswith('.vtu') else 'MSH'}"
                 ]
                 info_text = "".join(info_lines)
                 self.info_label.setText(info_text)
                 self.info_label.adjustSize()
            
            # Initialize Zone Manager
            if hasattr(self, 'initialize_zone_manager'):
                 self.initialize_zone_manager()
            self._init_zone_actors()
            
            # Calculate avg_cell_size
            if display_elements > 0:
                if result and result.get('volume'):
                     self.avg_cell_size = (result['volume'] / display_elements) ** (1/3.0)
                elif result and result.get('bounding_box'):
                     bb = result['bounding_box']
                     dims = [bb['max'][i] - bb['min'][i] for i in range(3)]
                     self.avg_cell_size = (dims[0]*dims[1]*dims[2] * 0.5 / display_elements) ** (1/3.0)
            
            self.check_edge_visibility()
            return "SUCCESS"
            
        except Exception as e:
            print(f"[ERROR] load_mesh_file failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
            self.info_label.setText(f"Error loading mesh: {str(e)}")
            return "FAILED"

    
    def update_info_label(self, result: dict = None):
        """Update info label header with quality metrics after background calculation
        
        Args:
            result: Dictionary containing quality_metrics, total_nodes, total_elements, etc.
        """
        if not result:
            return
            
        # Build info text with quality metrics
        display_nodes = result.get('total_nodes', 0)
        display_elements = result.get('total_elements', 0)
        tet_count = sum(1 for e in self.current_mesh_elements if e.get('type') == 'tetrahedron') if self.current_mesh_elements else 0
        tri_count = sum(1 for e in self.current_mesh_elements if e.get('type') == 'triangle') if self.current_mesh_elements else 0
        
        # Extract mesh filename from current state (set during load_mesh_file or from result)
        mesh_name = "Mesh"
        if hasattr(self, 'current_mesh_file') and self.current_mesh_file:
            mesh_name = Path(self.current_mesh_file).name
        
        info_lines = [
            f"<b>Mesh Generated</b><br>",
            f"{mesh_name}<br>",
            f"<span style='color: #6c757d;'>",
            f"Nodes: {display_nodes:,} * Elements: {display_elements:,}<br>",
            f"Tetrahedra: {tet_count:,} * Triangles: {tri_count:,}"
        ]
        
        # Add bounding box dimensions if available
        if result.get('bounding_box'):
            bb = result['bounding_box']
            dims = [bb['max'][i] - bb['min'][i] for i in range(3)]
            info_lines.append(f"<br><b>Bounds:</b> {dims[0]:.1f} x {dims[1]:.1f} x {dims[2]:.1f} mm")
        
        # Add volume if available
        if result.get('volume'):
            vol = result['volume']
            info_lines.append(f" | <b>Vol:</b> {vol / 1e9:,.4f} m³")
        
        # Add quality metrics if available
        if result.get('quality_metrics'):
            metrics = result['quality_metrics']
            info_lines.append("<br><b>Quality Metrics (avg):</b><br>")
            
            # SICN (primary gmsh metric) - use AVERAGE
            if 'sicn_avg' in metrics:
                sicn = metrics['sicn_avg']
                sicn_color = "#198754" if sicn >= 0.7 else "#ffc107" if sicn >= 0.5 else "#dc3545"
                info_lines.append(f"<span style='color: {sicn_color};'>SICN: {sicn:.3f}</span> ")
            
            # Gamma - use AVERAGE
            if 'gamma_avg' in metrics:
                gamma = metrics['gamma_avg']
                gamma_color = "#198754" if gamma >= 0.6 else "#ffc107" if gamma >= 0.4 else "#dc3545"
                info_lines.append(f"<span style='color: {gamma_color};'>γ: {gamma:.3f}</span><br>")
            
            # Skewness - use AVERAGE
            if 'skewness_avg' in metrics:
                skew = metrics['skewness_avg']
                skew_color = "#198754" if skew <= 0.3 else "#ffc107" if skew <= 0.5 else "#dc3545"
                info_lines.append(f"<span style='color: {skew_color};'>Skew: {skew:.3f}</span> ")
            
            # Aspect Ratio - use AVERAGE
            if 'aspect_ratio_avg' in metrics:
                ar = metrics['aspect_ratio_avg']
                ar_color = "#198754" if ar <= 2.0 else "#ffc107" if ar <= 3.0 else "#dc3545"
                info_lines.append(f"<span style='color: {ar_color};'>AR: {ar:.2f}</span>")
        
        info_lines.append("</span>")
        info_text = "".join(info_lines)
        self.info_label.setText(info_text)
        self.info_label.adjustSize()
        print(f"[DEBUG] Info label updated with quality metrics")

    def _parse_msh_file(self, filepath: str):
        nodes = {}
        elements = []
        physical_groups = {}  # {tag: {'dim': int, 'name': str}}
        entity_physical = {}  # {(dim, entity_tag): physical_tag} for Gmsh 4.1

        with open(filepath, 'r') as f:
            lines = [l.strip() for l in f.readlines()]

        # Detect version
        version = 4.1
        i = 0
        while i < len(lines):
            line = lines[i]
            if line == "$MeshFormat":
                i += 1
                if i < len(lines):
                    parts = lines[i].split()
                    if len(parts) >= 1:
                        try:
                            version = float(parts[0])
                        except Exception:
                            pass
                break
            i += 1

        print(f"[MESH_LOADER DEBUG] Detected Gmsh version: {version}")

        i = 0
        while i < len(lines):
            line = lines[i]

            # Parse $PhysicalNames section
            if line == "$PhysicalNames":
                i += 1
                if i >= len(lines):
                    break
                try:
                    num_physical = int(lines[i])
                    i += 1
                    for _ in range(num_physical):
                        if i >= len(lines):
                            break
                        parts = lines[i].split()
                        if len(parts) >= 3:
                            dim = int(parts[0])
                            tag = int(parts[1])
                            name = parts[2].strip('"')
                            physical_groups[tag] = {'dim': dim, 'name': name}
                        i += 1
                    if i < len(lines) and lines[i] == "$EndPhysicalNames":
                        i += 1
                    print(f"[MESH_LOADER] Parsed {len(physical_groups)} physical groups")
                    continue
                except ValueError:
                    i += 1
                    continue

            if line == "$Nodes":
                i += 1
<<<<<<< Updated upstream
                if i >= len(lines): break
                
                if version < 3.0:
                    # Gmsh 2.2
                    try:
                        num_nodes = int(lines[i])
                        i += 1
                        for _ in range(num_nodes):
                            if i >= len(lines): break
                            parts = lines[i].split()
                            if len(parts) >= 4:
                                nid = int(parts[0])
                                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                                nodes[nid] = [x, y, z]
                            i += 1
                        if i < len(lines) and lines[i] == "$EndNodes":
                            i += 1
                        continue
                    except ValueError:
                         i += 1
                         continue
                else:
                    # Gmsh 4.1
                    i += 1 
                    while i < len(lines) and lines[i] != "$EndNodes":
                        parts = lines[i].split()
                        if len(parts) == 4:
                            try:
                                num_nodes_in_block = int(parts[3])
                                i += 1
                                node_tags = []
                                for _ in range(num_nodes_in_block):
                                    node_tags.append(int(lines[i]))
                                    i += 1
                                for tag in node_tags:
                                    coords = lines[i].split()
                                    nodes[tag] = [float(coords[0]), float(coords[1]), float(coords[2])]
                                    i += 1
                            except ValueError:
                                i += 1
                        else:
                            i += 1
                    if i < len(lines) and lines[i] == "$EndNodes":
                        i += 1
                    continue

            elif line == "$Elements":
                i += 1
                if i >= len(lines): break
                
                if version < 3.0:
                    # Gmsh 2.2
                    try:
                        num_elements = int(lines[i])
                        i += 1
                        for _ in range(num_elements):
                            if i >= len(lines): break
                            parts = lines[i].split()
                            if len(parts) >= 3:
                                eid = int(parts[0])
                                etype = int(parts[1])
                                num_tags = int(parts[2])
                                node_start_idx = 3 + num_tags
                                node_ids = [int(n) for n in parts[node_start_idx:]]
                                
                                elem_data = None
                                if etype == 2: elem_data = {"id": eid, "type": "triangle", "nodes": node_ids}
                                elif etype == 3: elem_data = {"id": eid, "type": "quadrilateral", "nodes": node_ids}
                                elif etype == 4: elem_data = {"id": eid, "type": "tetrahedron", "nodes": node_ids}
                                elif etype == 5: elem_data = {"id": eid, "type": "hexahedron", "nodes": node_ids}
                                
                                if elem_data:
                                    elements.append(elem_data)
                            i += 1
                        if i < len(lines) and lines[i] == "$EndElements":
                             i += 1
                        continue
                    except ValueError:
                         i += 1
                         continue
                else:
                    # Gmsh 4.1
                    i += 1 
                    while i < len(lines) and lines[i] != "$EndElements":
                        parts = lines[i].split()
                        if len(parts) == 4:
                            entity_dim = int(parts[0])  # Dimension of entity (2=surface, 3=volume)
                            entity_tag = int(parts[1])  # Entity tag (maps to CAD surface)
                            element_type = int(parts[2])
                            num_elements_in_block = int(parts[3])
=======
                if i >= len(lines):
                    break
                # Gmsh 4.x: numEntityBlocks, then per block: entityDim entityTag parametric numNodes, then numNodes lines of nodeId x y z
                try:
                    num_blocks = int(lines[i])
                    i += 1
                    for _ in range(num_blocks):
                        if i >= len(lines):
                            break
                        block_parts = lines[i].split()
                        if len(block_parts) >= 4:
                            num_nodes_in_block = int(block_parts[3])
>>>>>>> Stashed changes
                            i += 1
                            for _ in range(num_nodes_in_block):
                                if i >= len(lines):
                                    break
                                parts = lines[i].split()
                                if len(parts) >= 4:
                                    node_id = int(parts[0])
                                    nodes[node_id] = [float(parts[1]), float(parts[2]), float(parts[3])]
                                i += 1
                        else:
                            i += 1
                except (ValueError, IndexError):
                    # Fallback: advance to $EndNodes
                    while i < len(lines) and lines[i] != "$EndNodes":
                        parts = lines[i].split()
                        if len(parts) >= 4:
                            try:
                                node_id = int(parts[0])
                                nodes[node_id] = [float(parts[1]), float(parts[2]), float(parts[3])]
                            except (ValueError, IndexError):
                                pass
                        i += 1
                while i < len(lines) and lines[i] != "$EndNodes":
                    i += 1
                if i < len(lines):
                    i += 1
                continue

            if line == "$Elements":
                i += 1
                if i >= len(lines):
                    break
                while i < len(lines) and lines[i] != "$EndElements":
                    parts = lines[i].split()
                    if len(parts) == 4:
                        entity_dim = int(parts[0])
                        entity_tag = int(parts[1])
                        element_type = int(parts[2])
                        num_elements_in_block = int(parts[3])
                        i += 1

                        type_str = None
                        expected_nodes = 0
                        if element_type == 2:
                            type_str = "triangle"
                            expected_nodes = 3
                        elif element_type == 3:
                            type_str = "quadrilateral"
                            expected_nodes = 4
                        elif element_type == 4:
                            type_str = "tetrahedron"
                            expected_nodes = 4
                        elif element_type == 5:
                            type_str = "hexahedron"
                            expected_nodes = 8
                        elif element_type == 11:
                            type_str = "tetrahedron"
                            expected_nodes = 4

                        if type_str:
                            for _ in range(num_elements_in_block):
                                if i >= len(lines):
                                    break
                                data = lines[i].split()
                                if len(data) >= 1 + expected_nodes:
                                    eid = int(data[0])
                                    enodes = [int(x) for x in data[1:1+expected_nodes]]
                                    elem_dict = {"id": eid, "type": type_str, "nodes": enodes}
                                    if entity_dim == 2:
                                        elem_dict['entity_tag'] = entity_tag
                                    elements.append(elem_dict)
                                i += 1
                        else:
                            i += num_elements_in_block
                    else:
                        i += 1
                if i < len(lines) and lines[i] == "$EndElements":
                    i += 1
                continue

            i += 1

        return nodes, elements, physical_groups
    def load_component_visualization(self, result: Dict):
        """Load and display CoACD components with PyVista"""
        from pathlib import Path
        debug_log = Path("poly_debug.txt")
        with open(debug_log, 'w') as f:
            f.write("=== COMPONENT VIZ DEBUG (Fixed) ===\n")
            f.write(f"Function called\n")
            f.write(f"Result keys: {list(result.keys())}\n")
        
        print("="*80)
        print("[COMPONENT-VIZ] *** FUNCTION CALLED *** IN VTK_VIEWER.PY")
        print("="*80)
        try:
            import pyvista as pv
            import numpy as np
            
            component_files = result.get('component_files', [])
            if not component_files:
                print("[ERROR] No component files in result")
                self.info_label.setText("Error: No components found")
                return "FAILED"
            
            print(f"[COMPONENT-VIZ] Loading {len(component_files)} components...")
            self.clear_view()
            
            # Load and merge all components
            merged_mesh = None
            for i, comp_file in enumerate(component_files):
                if not Path(comp_file).exists():
                    print(f"[WARNING] Component file not found: {comp_file}")
                    continue
                
                # Load with PyVista
                comp_mesh = pv.read(comp_file)
                
                # Ensure Component_ID scalar exists
                if "Component_ID" not in comp_mesh.array_names:
                    comp_mesh["Component_ID"] = np.full(comp_mesh.n_cells, i, dtype=int)
                
                # Merge into single mesh
                if merged_mesh is None:
                    merged_mesh = comp_mesh
                else:
                    merged_mesh = merged_mesh.merge(comp_mesh)
                
                print(f"[COMPONENT-VIZ] Loaded component {i}: {comp_mesh.n_cells} cells")
            
            if merged_mesh is None:
                print("[ERROR] No components loaded")
                self.info_label.setText("Error: Failed to load components")
                return "FAILED"
            
            # Always extract surface for display, but check if we have volume data
            ugrid = merged_mesh.cast_to_unstructured_grid()
            
            # Filter to keep only volume cells if they exist (Robust check)
            cell_types = ugrid.celltypes
            # Check for 3D cell types: Tet(10), Voxel(11), Hex(12), Wedge(13), Pyramid(14)
            vol_mask = np.isin(cell_types, [vtk.VTK_HEXAHEDRON, vtk.VTK_TETRA, vtk.VTK_WEDGE, vtk.VTK_PYRAMID, 
                                          vtk.VTK_HEXAHEDRON, 11, 12, 10, 13, 14])
            
            has_volume_cells = False
            if np.any(vol_mask):
                print(f"[COMPONENT-VIZ] Found {np.sum(vol_mask)} volume cells! Filtering...")
                ugrid = ugrid.extract_cells(vol_mask)
                has_volume_cells = True
            else:
                print("[COMPONENT-VIZ] No volume cells found in successful meshes")
            
            polydata = ugrid.extract_surface()
            
            print(f"[COMPONENT-VIZ] Displaying surface with {polydata.GetNumberOfCells()} faces (from {ugrid.n_cells} volume cells)" if has_volume_cells else f"[COMPONENT-VIZ] Displaying surface mesh")
            
            # Determine visualization mode
            viz_mode = result.get('visualization_mode', 'components')
            print(f"[COMPONENT-VIZ] Visualization mode: {viz_mode}")

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(polydata)
            
            if viz_mode == 'quality':
                # Quality Visualization Mode (Green-Yellow-Red)
                # Compute quality metric (Scaled Jacobian or similar)
                print("[COMPONENT-VIZ] Computing quality metrics...")
                quality = ugrid.compute_cell_quality(quality_measure='scaled_jacobian')
                polydata.cell_data['Quality'] = quality.cell_data['CellQuality']
                
                mapper.SetScalarModeToUseCellData()
                mapper.SetScalarRange(0.0, 1.0)  # Quality usually 0-1
                mapper.ScalarVisibilityOn()
                
                # Create diverging colormap (Red=Bad, Green=Good)
                lut = vtk.vtkLookupTable()
                lut.SetNumberOfTableValues(256)
                lut.Build()
                ctf = vtk.vtkColorTransferFunction()
                ctf.AddRGBPoint(0.0, 1.0, 0.0, 0.0)  # Red (Bad)
                ctf.AddRGBPoint(0.5, 1.0, 1.0, 0.0)  # Yellow (Medium)
                ctf.AddRGBPoint(1.0, 0.0, 1.0, 0.0)  # Green (Good)
                
                for i in range(256):
                    val = i / 255.0
                    rgb = ctf.GetColor(val)
                    lut.SetTableValue(i, rgb[0], rgb[1], rgb[2], 1.0)
                
                mapper.SetLookupTable(lut)
                
                # Add scalar bar for Quality
                scalar_bar = vtk.vtkScalarBarActor()
                scalar_bar.SetLookupTable(lut)
                scalar_bar.SetTitle("Mesh Quality")
                scalar_bar.SetNumberOfLabels(5)
                scalar_bar.SetPosition(0.85, 0.1)
                scalar_bar.SetWidth(0.12)
                scalar_bar.SetHeight(0.8)
                self.renderer.AddActor2D(scalar_bar)
                
            else:
                # Component ID Mode (Default)
                mapper.SetScalarModeToUseCellData()
                mapper.SetScalarRange(0, result.get('num_components', 10))
                mapper.ScalarVisibilityOn()
                
                # Create categorical color lookup table
                lut = vtk.vtkLookupTable()
                num_colors = max(10, result.get('num_components', 10))
                lut.SetNumberOfTableValues(num_colors)
                lut.Build()
                
                # Use distinct colors (tab20 colormap approximation)
                colors = [
                    (0.12, 0.47, 0.71), (1.0, 0.50, 0.05), (0.17, 0.63, 0.17),
                    (0.84, 0.15, 0.16), (0.58, 0.40, 0.74), (0.55, 0.34, 0.29),
                    (0.89, 0.47, 0.76), (0.50, 0.50, 0.50), (0.74, 0.74, 0.13),
                    (0.09, 0.75, 0.81), (0.68, 0.78, 0.91), (1.0, 0.73, 0.47),
                    (0.60, 0.87, 0.54), (1.0, 0.60, 0.60), (0.79, 0.70, 0.84),
                    (0.78, 0.70, 0.65), (0.98, 0.75, 0.83), (0.78, 0.78, 0.78),
                    (0.86, 0.86, 0.55), (0.62, 0.85, 0.88)
                ]
                for i in range(num_colors):
                    color = colors[i % len(colors)]
                    lut.SetTableValue(i, color[0], color[1], color[2], 1.0)
                
                mapper.SetLookupTable(lut)
                
                # Add scalar bar for Component ID
                scalar_bar = vtk.vtkScalarBarActor()
                scalar_bar.SetLookupTable(lut)
                scalar_bar.SetTitle("Component ID")
                scalar_bar.SetNumberOfLabels(min(num_colors, 10))
                scalar_bar.SetPosition(0.85, 0.1)
                scalar_bar.SetWidth(0.12)
                scalar_bar.SetHeight(0.8)
                self.renderer.AddActor2D(scalar_bar)
            
            # Create actor
            self.current_actor = vtk.vtkActor()
            self.current_actor.SetMapper(mapper)
            
            # Set initial opacity from parent opacity slider if available
            current_opacity = self.parent().viz_opacity_spin.value() if hasattr(self.parent(), 'viz_opacity_spin') else 1.0
            self.current_actor.GetProperty().SetOpacity(current_opacity)
            
            # Increase ambient lighting to reduce dark shadows
            self.current_actor.GetProperty().SetAmbient(0.6)  # High ambient for visibility
            self.current_actor.GetProperty().SetDiffuse(0.6)
            self.current_actor.GetProperty().SetSpecular(0.2)
            
            # Enable edge visibility to show hex mesh structure
            if has_volume_cells:
                self.current_actor.GetProperty().SetEdgeVisibility(True)
                self.current_actor.GetProperty().SetEdgeColor(0.2, 0.2, 0.2)  # Dark gray edges
                self.current_actor.GetProperty().SetLineWidth(1.0)
            
            # Add to renderer
            self.renderer.AddActor(self.current_actor)
            
            # Reset camera and render
            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()
            
            # Update info
            self.info_label.setText(f"Components: {result.get('num_components', 0)} | "
                                   f"Volume Error: {result.get('volume_error_pct', 0):.2f}%")
            
            print(f"[COMPONENT-VIZ] Displayed {result.get('num_components')} components successfully")
            return "SUCCESS"
            
        except ImportError:
            self.info_label.setText("Error: PyVista not installed (pip install pyvista)")
            print("[ERROR] PyVista not available")
            return "FAILED"
        except Exception as e:
            self.info_label.setText(f"Error loading components: {str(e)}")
            import traceback
            traceback.print_exc()
            return "FAILED"
