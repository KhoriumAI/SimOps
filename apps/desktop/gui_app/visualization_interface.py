"""
Visualization Interface: Mesh Rendering Abstraction
====================================================

Abstract interface between mesh data and rendering system.
Isolates VTK dependencies behind a clean API.

Usage:
    from apps.desktop.gui_app.visualization_interface import IVisualizationAdapter
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np


@dataclass
class RenderableMesh:
    """Format-agnostic mesh data for rendering."""
    vertices: np.ndarray                    # (N, 3)
    cells: np.ndarray                       # Cell connectivity
    cell_type: str                          # "tet4", "tet10", "hex8", "poly"
    quality_values: Optional[np.ndarray] = None  # Per-cell quality
    boundary_faces: Optional[np.ndarray] = None  # For surface rendering


@dataclass
class ViewState:
    """Camera and display state."""
    camera_position: Tuple[float, float, float] = (0, 0, 1)
    camera_focal: Tuple[float, float, float] = (0, 0, 0)
    clipping_enabled: bool = False
    clipping_axis: str = 'x'
    clipping_offset: float = 0.0


class IVisualizationAdapter(ABC):
    """
    Abstract interface for mesh visualization.
    
    Implementations:
    - VTKVisualizationAdapter: Uses VTK (heavy)
    - MockVisualizationAdapter: Logs calls (for testing)
    """
    
    @abstractmethod
    def load_mesh(self, mesh: RenderableMesh):
        """Load mesh data for rendering."""
        pass
    
    @abstractmethod
    def set_quality_coloring(self, enabled: bool, metric: str = "sicn"):
        """Toggle quality-based coloring."""
        pass
    
    @abstractmethod
    def set_cross_section(self, enabled: bool, axis: str, offset: float):
        """Configure cross-section clipping plane."""
        pass
    
    @abstractmethod
    def highlight_faces(self, face_ids: List[int], color: Tuple[float, float, float]):
        """Highlight selected faces."""
        pass
    
    @abstractmethod
    def get_view_state(self) -> ViewState:
        """Get current camera/display configuration."""
        pass
    
    @abstractmethod
    def set_view_state(self, state: ViewState):
        """Restore camera/display configuration."""
        pass
    
    @abstractmethod
    def clear(self):
        """Clear all rendered content."""
        pass


class MockVisualizationAdapter(IVisualizationAdapter):
    """Mock adapter that logs calls for testing."""
    
    def __init__(self):
        self._loaded_mesh = None
        self._view_state = ViewState()
        self._quality_enabled = False
    
    def load_mesh(self, mesh: RenderableMesh):
        print(f"[MockViz] Loading mesh: {len(mesh.vertices)} verts, type={mesh.cell_type}")
        self._loaded_mesh = mesh
    
    def set_quality_coloring(self, enabled: bool, metric: str = "sicn"):
        print(f"[MockViz] Quality coloring: enabled={enabled}, metric={metric}")
        self._quality_enabled = enabled
    
    def set_cross_section(self, enabled: bool, axis: str, offset: float):
        print(f"[MockViz] Cross-section: enabled={enabled}, axis={axis}, offset={offset}")
    
    def highlight_faces(self, face_ids: List[int], color: Tuple[float, float, float]):
        print(f"[MockViz] Highlighting {len(face_ids)} faces")
    
    def get_view_state(self) -> ViewState:
        return self._view_state
    
    def set_view_state(self, state: ViewState):
        self._view_state = state
    
    def clear(self):
        print("[MockViz] Cleared")
        self._loaded_mesh = None
