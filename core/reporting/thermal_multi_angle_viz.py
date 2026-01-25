"""
Thermal Multi-Angle Visualization
==================================

Generates professional thermal field visualizations from CalculiX/OpenFOAM VTU files
with multiple camera angles for comprehensive thermal analysis.
"""

import numpy as np
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Try to import PyVista, fall back to matplotlib if unavailable
PYVISTA_AVAILABLE = False
FORCE_MATPLOTLIB = True  # Force matplotlib on Python 3.14+ due to PyVista incompatibility

try:
    import sys
    if sys.version_info >= (3, 14):
        FORCE_MATPLOTLIB = True
        logger.info("Python 3.14+ detected - forcing matplotlib fallback")
    else:
        import pyvista as pv
        from core.reporting.camera_utils import setup_camera, get_view_title, STANDARD_VIEWS
        pv.OFF_SCREEN = True
        PYVISTA_AVAILABLE = True
        logger.info("PyVista available for 3D visualization")
except (ImportError, AttributeError) as e:
    logger.warning(f"PyVista unavailable ({e}), will use matplotlib fallback")
    PYVISTA_AVAILABLE = False


def generate_thermal_views(
    vtu_path: str,
    output_dir: Path,
    job_name: str,
    views: List[str] = None,
    window_size: tuple = (1920, 1080),
    dpi: int = 300,
    colormap: str = 'coolwarm'
) -> List[str]:
    """
    Generate thermal field visualizations from multiple camera angles.

    Args:
        vtu_path: Path to VTU file with temperature data
        output_dir: Directory to save images
        job_name: Job name for file naming
        views: List of views to generate ['isometric', 'top', 'front', 'section']
        window_size: Render window size (width, height) for high quality output
        dpi: Target DPI for print quality (300 recommended)
        colormap: Matplotlib colormap name (default: 'coolwarm')

    Returns:
        List of paths to generated PNG images
    """
    if views is None:
        views = ['isometric', 'top', 'front', 'section']

    # Use matplotlib fallback if PyVista unavailable or forced
    if not PYVISTA_AVAILABLE or FORCE_MATPLOTLIB:
        logger.info("Using matplotlib 3D fallback for visualization")
        try:
            from core.reporting.thermal_viz_matplotlib3d import generate_thermal_views_matplotlib
            return generate_thermal_views_matplotlib(vtu_path, output_dir, job_name, views, colormap)
        except Exception as e:
            logger.error(f"Matplotlib fallback also failed: {e}")
            return []

    # Use PyVista for high-quality rendering
    try:
        # Read VTU file
        mesh = pv.read(vtu_path)
        logger.info(f"Loaded thermal mesh: {mesh.n_points} points, {mesh.n_cells} cells")

        # Debug: List available arrays
        logger.info(f"Point data: {list(mesh.point_data.keys())}")
        logger.info(f"Cell data: {list(mesh.cell_data.keys())}")
    except Exception as e:
        # PyVista failed during read - fall back to matplotlib
        logger.warning(f"PyVista mesh reading failed: {e}")
        logger.info("Using matplotlib 3D fallback for visualization")
        try:
            from core.reporting.thermal_viz_matplotlib3d import generate_thermal_views_matplotlib
            return generate_thermal_views_matplotlib(vtu_path, output_dir, job_name, views, colormap)
        except Exception as e2:
            logger.error(f"Matplotlib fallback also failed: {e2}")
            return []

    # Continue with PyVista rendering
    try:
        
        # Find temperature field (try common naming conventions)
        temp_field = None
        for field_name in ['T', 'temperature', 'Temperature', 'NT11', 'TEMPERATURE']:
            if field_name in mesh.point_data:
                temp_field = field_name
                break
            elif field_name in mesh.cell_data:
                # Convert cell data to point data
                mesh = mesh.cell_data_to_point_data()
                if field_name in mesh.point_data:
                    temp_field = field_name
                    break
        
        if not temp_field:
            logger.error(f"No temperature field found. Available: {list(mesh.point_data.keys())}")
            return []
        
        logger.info(f"Using temperature field: '{temp_field}'")
        
        # Get temperature stats
        temps = mesh.point_data[temp_field]
        temp_min, temp_max = temps.min(), temps.max()
        logger.info(f"Temperature range: {temp_min:.2f} - {temp_max:.2f} K")
        
        # Convert to Celsius for display (assuming input is Kelvin)
        temps_celsius = temps - 273.15
        mesh['temperature_celsius'] = temps_celsius
        
        temp_c_min, temp_c_max = temps_celsius.min(), temps_celsius.max()
        logger.info(f"Temperature range (C): {temp_c_min:.2f} - {temp_c_max:.2f} °C")
        
        # Get mesh bounds
        bounds = mesh.bounds
        
        # Generate images for each view
        image_paths = []
        
        for view_name in views:
            output_path = output_dir / f"{job_name}_thermal_{view_name}.png"
            
            try:
                plotter = pv.Plotter(off_screen=True, window_size=window_size)
                plotter.set_background('white')
                
                # Prepare mesh for this view
                view_mesh = mesh.copy()
                
                # For cross-section view, create a slice
                if view_name == 'section':
                    view_config = STANDARD_VIEWS['section']
                    normal = view_config['normal']
                    origin = view_config['origin']
                    
                    # Create slice
                    view_mesh = mesh.slice(normal=normal, origin=origin)
                    
                    if view_mesh.n_points == 0:
                        logger.warning(f"Empty slice for {view_name}, skipping")
                        continue
                    
                    logger.info(f"Created cross-section: {view_mesh.n_points} points")
                
                # Add mesh with temperature coloring
                if 'temperature_celsius' in view_mesh.point_data:
                    plotter.add_mesh(
                        view_mesh,
                        scalars='temperature_celsius',
                        cmap=colormap,  # Blue (cold) to Red (hot)
                        show_edges=True,
                        edge_color='black',
                        edge_opacity=0.1,
                        line_width=0.5,
                        smooth_shading=True,
                        show_scalar_bar=True,
                        scalar_bar_args={
                            'title': 'Temperature (°C)',
                            'vertical': True,
                            'position_x': 0.85,
                            'position_y': 0.15,
                            'width': 0.08,
                            'height': 0.7,
                            'title_font_size': 16,
                            'label_font_size': 14,
                            'color': 'black',
                            'n_labels': 5,
                            'fmt': '%.1f'
                        }
                    )
                else:
                    # Fallback: use original field
                    plotter.add_mesh(
                        view_mesh,
                        scalars=temp_field,
                        cmap=colormap,
                        show_edges=True,
                        edge_opacity=0.1,
                        show_scalar_bar=True
                    )
                
                # Setup camera for this view
                setup_camera(plotter, view_name, bounds, zoom=1.5)
                
                # Add title
                title = get_view_title(view_name, job_name)
                plotter.add_text(
                    title,
                    position='upper_left',
                    font_size=18,
                    color='black',
                    font='arial'
                )
                
                # Save high-quality image
                plotter.screenshot(str(output_path))
                plotter.close()

                logger.info(f"Saved {view_name} view to {output_path}")
                image_paths.append(str(output_path))

            except Exception as e:
                logger.error(f"Failed to generate {view_name} view with PyVista: {e}")
                # Try matplotlib fallback for this specific view
                try:
                    from core.reporting.thermal_viz_matplotlib3d import generate_matplotlib_3d_view
                    logger.info(f"Attempting matplotlib fallback for {view_name} view...")
                    result = generate_matplotlib_3d_view(
                        str(vtu_path),
                        str(output_path),
                        view_name=view_name,
                        colormap=colormap
                    )
                    if result:
                        image_paths.append(str(output_path))
                except Exception as e2:
                    logger.error(f"Matplotlib fallback also failed for {view_name}: {e2}")
                continue
        
        return image_paths
        
    except Exception as e:
        logger.exception(f"Failed to generate thermal views: {e}")
        return []


def create_thermal_slice(
    mesh: pv.DataSet,
    normal: tuple = (1, 0, 0),
    origin: tuple = (0, 0, 0)
) -> pv.DataSet:
    """
    Create a cross-section slice through the mesh.
    
    Args:
        mesh: PyVista mesh with temperature data
        normal: Normal vector of slice plane
        origin: Point on slice plane
        
    Returns:
        Sliced mesh
    """
    return mesh.slice(normal=normal, origin=origin)
