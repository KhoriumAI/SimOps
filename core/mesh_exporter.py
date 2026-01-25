"""
Mesh Exporter Interface: Format-Agnostic Export
================================================

Abstract interface for mesh export to various formats.
Enables pluggable exporters without changing caller code.

Usage:
    from core.mesh_exporter import IExporter, get_exporter
    
    exporter = get_exporter(".msh")
    exporter.export(mesh, "output.msh")
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


@dataclass
class ExportableMesh:
    """Format-agnostic mesh representation for export."""
    points: np.ndarray                    # (N, 3)
    cells: np.ndarray                     # Connectivity
    cell_type: str                        # "tet4", "tet10", "hex8"
    boundary_zones: Optional[Dict[str, np.ndarray]] = None  # zone_name → face indices
    zone_types: Optional[Dict[str, str]] = None  # zone_name → "wall"/"inlet"
    cell_zone_name: str = "fluid"


class IExporter(ABC):
    """Abstract interface for mesh export."""
    
    @abstractmethod
    def export(self, mesh: ExportableMesh, filepath: str) -> bool:
        """
        Export mesh to file.
        
        Returns:
            True on success
        """
        pass
    
    @abstractmethod
    def get_extensions(self) -> List[str]:
        """Return supported file extensions (e.g., ['.msh', '.cas'])."""
        pass


# Exporter registry
_exporters: Dict[str, IExporter] = {}


def register_exporter(extension: str, exporter: IExporter):
    """Register an exporter for a file extension."""
    _exporters[extension.lower()] = exporter


def get_exporter(extension: str) -> Optional[IExporter]:
    """Get exporter for file extension, or None if not registered."""
    return _exporters.get(extension.lower())


def get_supported_formats() -> List[str]:
    """Return list of registered export formats."""
    return list(_exporters.keys())
