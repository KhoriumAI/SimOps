"""
Geometry Cleanup and Defeaturing Module
========================================

DEPRECATED: This module has been moved to core/cad_cleaning/geometry_cleanup.py
This file is kept for backwards compatibility. Please update imports to:
    from core.cad_cleaning import GeometryCleanup

Handles problematic CAD geometry that causes meshing failures:
- Removes or merges small edges and curves
- Identifies and handles sharp features
- Detects infinitely thin surfaces
- Provides geometry repair utilities

Uses Gmsh's OpenCASCADE kernel capabilities.
"""

import warnings
warnings.warn(
    "core.geometry_cleanup is deprecated. Use core.cad_cleaning.GeometryCleanup instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export for backwards compatibility
from core.cad_cleaning.geometry_cleanup import GeometryCleanup

__all__ = ['GeometryCleanup']
