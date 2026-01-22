"""
Color Mapping Utilities for Thermal Visualization
=================================================

Predefined color maps optimized for temperature gradient visualization.
"""

from typing import Dict, List
import numpy as np

# Predefined color maps for thermal visualization
THERMAL_COLORMAPS = {
    'viridis': 'viridis',      # Perceptually uniform, good for general use
    'plasma': 'plasma',        # High contrast, good for hotspots
    'inferno': 'inferno',      # Dark background, excellent for thermal
    'coolwarm': 'coolwarm',    # Diverging, good for deviation from reference
    'jet': 'jet',              # Classic (not recommended, but included)
    'turbo': 'turbo',          # Modern alternative to jet
    'hot': 'hot',              # Black-red-yellow-white, traditional thermal
}

# Default colormap for temperature visualization
DEFAULT_COLORMAP = 'inferno'

class ColorMapRegistry:
    """Registry for managing color maps"""
    
    def __init__(self):
        self._colormaps = THERMAL_COLORMAPS.copy()
    
    def register(self, name: str, cmap: str):
        """Register a new color map"""
        self._colormaps[name] = cmap
    
    def get(self, name: str) -> str:
        """Get color map by name, fallback to default if not found"""
        return self._colormaps.get(name, DEFAULT_COLORMAP)
    
    def list_available(self) -> List[str]:
        """List all available color maps"""
        return list(self._colormaps.keys())
    
    def get_default(self) -> str:
        """Get default color map"""
        return DEFAULT_COLORMAP


def get_colormap(name: str = None) -> str:
    """
    Get a color map for thermal visualization.
    
    Args:
        name: Color map name. If None, returns default.
        
    Returns:
        Color map identifier compatible with PyVista/Matplotlib
    """
    if name is None:
        return DEFAULT_COLORMAP
    
    registry = ColorMapRegistry()
    return registry.get(name)


def create_temperature_range(min_temp: float, max_temp: float, n_levels: int = 10) -> np.ndarray:
    """
    Create evenly spaced temperature levels for scalar bar.
    
    Args:
        min_temp: Minimum temperature
        max_temp: Maximum temperature
        n_levels: Number of discrete levels
        
    Returns:
        Array of temperature values
    """
    return np.linspace(min_temp, max_temp, n_levels)


def get_recommended_colormap(physics_type: str = 'thermal') -> str:
    """
    Get recommended color map based on physics type.
    
    Args:
        physics_type: Type of physics ('thermal', 'structural', 'cfd', etc.)
        
    Returns:
        Recommended color map name
    """
    recommendations = {
        'thermal': 'inferno',
        'structural': 'viridis',
        'cfd': 'coolwarm',
        'pressure': 'plasma',
        'velocity': 'turbo',
    }
    
    return recommendations.get(physics_type, DEFAULT_COLORMAP)
