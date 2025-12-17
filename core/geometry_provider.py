"""
Geometry Provider: CAD/Geometry Operations Interface
=====================================================

Abstract interface for geometry loading and manipulation.
Isolates Gmsh/OpenCascade dependencies behind a clean API.

Usage:
    from core.geometry_provider import IGeometryProvider, get_geometry_provider
    
    provider = get_geometry_provider()
    info = provider.load_file("model.step")
    surface = provider.extract_surface_mesh(element_size=1.0)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class BoundingBox:
    """Axis-aligned bounding box."""
    min_corner: np.ndarray  # (3,) xyz
    max_corner: np.ndarray  # (3,) xyz
    
    @property
    def diagonal(self) -> float:
        return float(np.linalg.norm(self.max_corner - self.min_corner))
    
    @property
    def center(self) -> np.ndarray:
        return (self.min_corner + self.max_corner) / 2


@dataclass
class GeometryInfo:
    """Summary of loaded geometry."""
    num_volumes: int
    num_surfaces: int
    num_edges: int
    bounding_box: BoundingBox
    has_shells: bool = False  # True if contains open shells (not solid)
    volume_tags: Optional[List[int]] = None


@dataclass
class SurfaceMesh:
    """Discrete surface mesh representation."""
    vertices: np.ndarray   # (N, 3) float
    triangles: np.ndarray  # (M, 3) int indices
    normals: Optional[np.ndarray] = None  # (N, 3) per-vertex normals


class IGeometryProvider(ABC):
    """
    Abstract interface for CAD geometry operations.
    
    Implementations:
    - GmshGeometryProvider: Uses Gmsh SDK (heavy)
    - MockGeometryProvider: Returns unit cube (for testing)
    """
    
    @abstractmethod
    def load_file(self, filepath: str) -> GeometryInfo:
        """
        Load CAD file (STEP/IGES/STL).
        
        Args:
            filepath: Path to geometry file
            
        Returns:
            GeometryInfo with volume/surface counts and bbox
        """
        pass
    
    @abstractmethod
    def extract_surface_mesh(self, element_size: float) -> SurfaceMesh:
        """
        Generate triangular surface mesh.
        
        Args:
            element_size: Target edge length
            
        Returns:
            SurfaceMesh with vertices and triangles
        """
        pass
    
    @abstractmethod
    def heal_geometry(self) -> bool:
        """
        Attempt to repair geometry issues.
        
        Returns:
            True if healing succeeded
        """
        pass
    
    @abstractmethod
    def cleanup(self):
        """Release resources (finalize Gmsh, etc.)."""
        pass


class MockGeometryProvider(IGeometryProvider):
    """Mock provider returning unit cube for testing."""
    
    def load_file(self, filepath: str) -> GeometryInfo:
        return GeometryInfo(
            num_volumes=1,
            num_surfaces=6,
            num_edges=12,
            bounding_box=BoundingBox(
                min_corner=np.array([0.0, 0.0, 0.0]),
                max_corner=np.array([1.0, 1.0, 1.0])
            )
        )
    
    def extract_surface_mesh(self, element_size: float) -> SurfaceMesh:
        # Unit cube vertices
        v = np.array([
            [0,0,0], [1,0,0], [1,1,0], [0,1,0],  # bottom
            [0,0,1], [1,0,1], [1,1,1], [0,1,1]   # top
        ], dtype=float)
        # 12 triangles (2 per face)
        t = np.array([
            [0,1,2], [0,2,3],  # bottom
            [4,6,5], [4,7,6],  # top
            [0,4,5], [0,5,1],  # front
            [2,6,7], [2,7,3],  # back
            [0,3,7], [0,7,4],  # left
            [1,5,6], [1,6,2]   # right
        ], dtype=int)
        return SurfaceMesh(vertices=v, triangles=t)
    
    def heal_geometry(self) -> bool:
        return True
    
    def cleanup(self):
        pass


# Provider registry
_provider_instance: Optional[IGeometryProvider] = None


def get_geometry_provider() -> IGeometryProvider:
    """Get current geometry provider (mock by default)."""
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = MockGeometryProvider()
    return _provider_instance


def register_geometry_provider(provider: IGeometryProvider):
    """Register a real provider implementation."""
    global _provider_instance
    _provider_instance = provider
