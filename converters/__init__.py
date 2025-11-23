"""
Mesh Conversion Module

Provides post-processing converters for tetrahedral meshes:
- Polyhedral mesh generation (node-based dual/agglomeration)
- Hexahedral mesh generation (THex splitting)
"""

from .poly_hex_converter import TetToPolyConverter, TetToHexConverter, PolyCell, PolyFace

__all__ = ['TetToPolyConverter', 'TetToHexConverter', 'PolyCell', 'PolyFace']
