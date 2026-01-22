"""
Camera Utilities for Multi-Angle Thermal Visualization
=======================================================

Provides standard camera positions and configurations for consistent
thermal rendering across SimOps reports.
"""

import numpy as np
from typing import Dict, Tuple, Optional

# Standard view configurations for thermal visualizations
STANDARD_VIEWS = {
    "isometric": {
        "position": (1, 1, 1),
        "focal_point": (0, 0, 0),
        "up": (0, 0, 1),
        "parallel_projection": False,
        "description": "3D perspective view showing all three axes"
    },
    "top": {
        "position": (0, 0, 1),
        "focal_point": (0, 0, 0),
        "up": (0, 1, 0),
        "parallel_projection": True,
        "description": "Orthographic view from above (XY plane)"
    },
    "front": {
        "position": (0, -1, 0),
        "focal_point": (0, 0, 0),
        "up": (0, 0, 1),
        "parallel_projection": True,
        "description": "Orthographic front view (XZ plane)"
    },
    "section": {
        "normal": (1, 0, 0),  # X-axis cut plane
        "origin": (0, 0, 0),
        "focal_point": (0, 0, 0),
        "up": (0, 0, 1),
        "parallel_projection": True,
        "description": "Cross-section cut along X-axis"
    }
}


def setup_camera(plotter, view_name: str, mesh_bounds: Tuple[float, ...], zoom: float = 1.5):
    """
    Configure PyVista plotter camera for a standard view.
    
    Args:
        plotter: PyVista Plotter instance
        view_name: Name of view ('isometric', 'top', 'front', 'section')
        mesh_bounds: Tuple of (xmin, xmax, ymin, ymax, zmin, zmax)
        zoom: Zoom factor (>1.0 = closer, <1.0 = farther)
        
    Returns:
        Configured plotter
    """
    if view_name not in STANDARD_VIEWS:
        raise ValueError(f"Unknown view: {view_name}. Available: {list(STANDARD_VIEWS.keys())}")
    
    view_config = STANDARD_VIEWS[view_name]
    
    # Calculate mesh dimensions and center
    xmin, xmax, ymin, ymax, zmin, zmax = mesh_bounds
    center = np.array([
        (xmin + xmax) / 2,
        (ymin + ymax) / 2,
        (zmin + zmax) / 2
    ])
    
    # Calculate characteristic length (diagonal)
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    char_length = np.sqrt(dx**2 + dy**2 + dz**2)
    
    # Set focal point to mesh center to ensure flexible centering
    plotter.camera.focal_point = center
    
    # Set camera position relative to center
    if "position" in view_config:
        # Normalize position vector and scale by characteristic length
        pos = np.array(view_config["position"])
        pos_normalized = pos / np.linalg.norm(pos)
        
        if view_config["parallel_projection"]:
            # For orthographic views, position far away
            camera_distance = char_length * 10
        else:
            # For perspective (isometric), closer for better depth
            camera_distance = char_length * 3.0 / zoom
        
        # Position is Center + Offset
        plotter.camera.position = center + (pos_normalized * camera_distance)
    
    # Set up vector
    plotter.camera.up = view_config["up"]
    
    # Set projection type
    plotter.camera.parallel_projection = view_config["parallel_projection"]
    
    # For orthographic views, set parallel scale
    if view_config["parallel_projection"]:
        # Parallel scale is half the viewport height
        # We want to fit the mesh with some padding
        max_extent = max(dx, dy, dz)
        plotter.camera.parallel_scale = max_extent / (2 * zoom)
    
    return plotter


def get_view_title(view_name: str, job_name: str) -> str:
    """
    Generate a descriptive title for a view.
    
    Args:
        view_name: Name of view
        job_name: Job/component name
        
    Returns:
        Formatted title string
    """
    view_config = STANDARD_VIEWS.get(view_name, {})
    description = view_config.get("description", view_name.capitalize())
    
    return f"{job_name} - {description}"
