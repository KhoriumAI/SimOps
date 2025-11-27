"""
VTK 3D Viewer Coordinator
=========================

Main viewer class that coordinates rendering, interaction, and delegates
specific visualization tasks to specialized modules.
"""

import vtk
from PyQt5.QtWidgets import QFrame, QVBoxLayout, QLabel, QSizePolicy, QPushButton, QHBoxLayout
from PyQt5.QtCore import Qt
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from ..interactor import CustomInteractorStyle

# Import strategies
from .quality_renderer import QualityRenderer
from .cross_section import CrossSectionHandler
from .mesh_loader import MeshLoader
from .polyhedral_viz import PolyhedralVisualizer
from .component_viz import ComponentVisualizer

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

        self._setup_lighting()
        self._setup_interactor(parent)
        self._setup_axes()
        self._setup_overlays()
        
        # State
        self.current_actor = None
        self.current_poly_data = None
        self.current_volumetric_grid = None
        self.current_mesh_nodes = {}
        self.current_mesh_elements = []
        self.current_node_map = {}
        self.current_tetrahedra = []
        self.current_hexahedra = []
        self.current_quality_data = {}
        
        # Initialize modules
        self.quality_renderer = QualityRenderer(self)
        self.cross_section_handler = CrossSectionHandler(self)
        self.mesh_loader = MeshLoader(self)
        self.polyhedral_viz = PolyhedralVisualizer(self)
        self.component_viz = ComponentVisualizer(self)
        
        self.vtk_widget.Initialize()
        self.vtk_widget.Start()

    def _setup_lighting(self):
        self.renderer.SetAmbient(0.4, 0.4, 0.4)
        
        light1 = vtk.vtkLight()
        light1.SetPosition(1, 1, 1)
        light1.SetColor(1.0, 1.0, 1.0)
        light1.SetIntensity(0.6)
        self.renderer.AddLight(light1)
        
        light2 = vtk.vtkLight()
        light2.SetPosition(-0.5, 1, 0.5)
        light2.SetColor(1.0, 1.0, 1.0)
        light2.SetIntensity(0.3)
        self.renderer.AddLight(light2)
        
        light3 = vtk.vtkLight()
        light3.SetPosition(0, 0.5, -1)
        light3.SetColor(1.0, 1.0, 1.0)
        light3.SetIntensity(0.2)
        self.renderer.AddLight(light3)

    def _setup_interactor(self, parent):
        self.interactor_style = CustomInteractorStyle(parent=parent)
        self.vtk_widget.GetRenderWindow().GetInteractor().SetInteractorStyle(self.interactor_style)

    def _setup_axes(self):
        self.axes_widget = vtk.vtkOrientationMarkerWidget()
        self.axes_actor = vtk.vtkAxesActor()
        self.axes_actor.SetShaftTypeToCylinder()
        self.axes_widget.SetOrientationMarker(self.axes_actor)
        self.axes_widget.SetInteractor(self.vtk_widget.GetRenderWindow().GetInteractor())
        self.axes_widget.SetViewport(0, 0, 0.15, 0.15)
        self.axes_widget.SetEnabled(1)
        self.axes_widget.InteractiveOff()

    def _setup_overlays(self):
        # Info label
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
        self.info_label.setFixedWidth(450)
        self.info_label.move(15, 15)
        
        # Quality label
        self.quality_label = QLabel("", self)
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
        self.quality_label.setVisible(False)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.quality_label.isVisible():
            self.quality_label.move(
                self.width() - self.quality_label.width() - 15,
                15
            )

    def clear_view(self):
        actors = self.renderer.GetActors()
        actors.InitTraversal()
        for i in range(actors.GetNumberOfItems()):
            actor = actors.GetNextActor()
            if actor:
                self.renderer.RemoveActor(actor)
        
        actors2d = self.renderer.GetActors2D()
        actors2d.InitTraversal()
        for i in range(actors2d.GetNumberOfItems()):
            actor = actors2d.GetNextActor2D()
            if actor:
                self.renderer.RemoveActor2D(actor)
        
        self.current_actor = None
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

    # Delegate methods
    def load_mesh_file(self, mesh_path, result=None):
        return self.mesh_loader.load_mesh_file(mesh_path, result)

    def load_step_file(self, filepath):
        return self.mesh_loader.load_step_file(filepath)

    def load_polyhedral_file(self, json_path):
        return self.polyhedral_viz.load_file(json_path)

    def load_component_visualization(self, result):
        return self.component_viz.load_visualization(result)

    def show_quality_report(self, metrics):
        self.quality_renderer.show_report(metrics)
    
    def set_clipping(self, enabled, axis='x', offset=0.0):
        self.cross_section_handler.set_clipping(enabled, axis, offset)
        
    def set_cross_section_mode(self, mode):
        self.cross_section_handler.set_cross_section_mode(mode)
        
    def set_cross_section_element_mode(self, mode):
        self.cross_section_handler.set_cross_section_element_mode(mode)
