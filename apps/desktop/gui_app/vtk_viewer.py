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
    QPushButton, QCheckBox, QComboBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from .interactor import CustomInteractorStyle
from .utils import hsl_to_rgb


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
    
    def load_polyhedral_file(self, json_path: str):
        """Load polyhedral mesh from JSON file"""
        # DEBUG: Write to file to prove function is running
        debug_log = Path("C:/Users/Owner/Downloads/MeshPackageLean/poly_debug.txt")
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
            
            # 4. Manual Surface Extraction
            # VTK's vtkDataSetMapper cannot render VTK_POLYHEDRON cells.
            # Extract all faces from the JSON and create a surface mesh.
            print("[POLY-VIZ] Manually extracting faces for visualization...")
            
            all_faces = []
            for elem in elements:
                if elem['type'] != 'polyhedron':
                    continue
                for face in elem['faces']:
                    try:
                        vtk_face = [node_map[nid] for nid in face]
                        all_faces.append(vtk_face)
                    except KeyError:
                        continue
            
            # Create PolyData from extracted faces
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(points)
            
            polys = vtk.vtkCellArray()
            for face_indices in all_faces:
                polys.InsertNextCell(len(face_indices))
                for idx in face_indices:
                    polys.InsertCellPoint(idx)
            
            polydata.SetPolys(polys)
            
            with open(debug_log, 'a') as f:
                f.write(f"=== MANUAL SURFACE ===\n")
                f.write(f"Faces extracted: {len(all_faces)}\n")
                f.write(f"PolyData cells: {polydata.GetNumberOfCells()}\n")
            
            print(f"[POLY-VIZ] Extracted {polydata.GetNumberOfCells()} face polygons")
            
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
            
            self.info_label.setText(f"Polyhedral Mesh: {ugrid.GetNumberOfCells()} cells")
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
        # If we have a volumetric grid (Polyhedra), use vtkCutter
        if self.current_volumetric_grid and self.current_volumetric_grid.GetNumberOfCells() > 0:
            # Check if we have polyhedra
            cell_type = self.current_volumetric_grid.GetCellType(0)
            if cell_type == vtk.VTK_POLYHEDRON:
                plane = vtk.vtkPlane()
                plane.SetOrigin(plane_origin)
                plane.SetNormal(plane_normal)
                
                cutter = vtk.vtkCutter()
                cutter.SetInputData(self.current_volumetric_grid)
                cutter.SetCutFunction(plane)
                cutter.Update()
                return cutter.GetOutput()
        
        # Fallback to manual intersection for standard meshes (Tets/Hexes)
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
                    # Compute quality from mesh file using Gmsh
                    print(f"[DEBUG] No quality file found, computing quality from mesh using Gmsh...")
                    try:
                        import gmsh
                        gmsh.initialize()
                        gmsh.option.setNumber("General.Terminal", 0)
                        gmsh.merge(str(filepath))
                        
                        per_element_quality = {}
                        per_element_gamma = {}
                        per_element_skewness = {}
                        per_element_aspect_ratio = {}
                        
                        # Get surface elements (triangles)
                        tri_types, tri_tags, tri_nodes = gmsh.model.mesh.getElements(2)
                        all_qualities = []
                        
                        for elem_type, tags in zip(tri_types, tri_tags):
                            if elem_type in [2, 9]:  # Linear & Quadratic Triangles
                                try:
                                    # Extract SICN quality for surface triangles
                                    sicn_vals = gmsh.model.mesh.getElementQualities(tags.tolist(), "minSICN")
                                    gamma_vals = gmsh.model.mesh.getElementQualities(tags.tolist(), "gamma")
                                    
                                    for tag, sicn, gamma in zip(tags, sicn_vals, gamma_vals):
                                        tag_int = int(tag)
                                        per_element_quality[tag_int] = float(sicn)
                                        per_element_gamma[tag_int] = float(gamma)
                                        per_element_skewness[tag_int] = 1.0 - float(sicn)
                                        per_element_aspect_ratio[tag_int] = 1.0 / float(sicn) if sicn > 0 else 100.0
                                        all_qualities.append(sicn)
                                    
                                    print(f"[DEBUG] Computed quality for {len(tags)} surface triangles (type {elem_type})")
                                except Exception as e:
                                    print(f"[DEBUG] Error computing surface triangle qualities: {e}")
                        
                        gmsh.finalize()
                        
                        if per_element_quality:
                            # Calculate statistics
                            sorted_q = sorted(all_qualities)
                            idx_10 = max(0, int(len(sorted_q) * 0.10))
                            
                            avg_gamma = sum(per_element_gamma.values()) / len(per_element_gamma) if per_element_gamma else 0
                            avg_skewness = sum(per_element_skewness.values()) / len(per_element_skewness) if per_element_skewness else 0
                            avg_aspect_ratio = sum(per_element_aspect_ratio.values()) / len(per_element_aspect_ratio) if per_element_aspect_ratio else 1.0
                            
                            if not result:
                                result = {}
                            result['per_element_quality'] = per_element_quality
                            result['per_element_gamma'] = per_element_gamma
                            result['per_element_skewness'] = per_element_skewness
                            result['per_element_aspect_ratio'] = per_element_aspect_ratio
                            result['quality_metrics'] = {
                                'sicn_min': min(all_qualities),
                                'sicn_avg': sum(all_qualities) / len(all_qualities),
                                'sicn_max': max(all_qualities),
                                'sicn_10_percentile': sorted_q[idx_10],
                                'gamma_avg': avg_gamma,
                                'avg_skewness': avg_skewness,
                                'avg_aspect_ratio': avg_aspect_ratio,
                            }
                            
                            self.current_quality_data = result
                            
                            print(f"[DEBUG] [OK][OK] Computed quality for {len(per_element_quality)} surface elements")
                            print(f"[DEBUG] Quality range: {min(all_qualities):.3f} to {max(all_qualities):.3f}")
                            print(f"[DEBUG] Gamma avg: {avg_gamma:.3f}, Skew avg: {avg_skewness:.3f}, AR avg: {avg_aspect_ratio:.2f}")
                        
                    except Exception as e:
                        print(f"[DEBUG] Failed to compute quality from mesh: {e}")
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
    def load_component_visualization(self, result: Dict):
        """Load and display CoACD components with PyVista"""
        from pathlib import Path
        debug_log = Path("C:/Users/Owner/Downloads/MeshPackageLean/component_debug.txt")
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
