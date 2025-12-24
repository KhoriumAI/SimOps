"""
Multi-Angle CFD Streamline Visualization
=========================================

Generates professional streamline visualizations from OpenFOAM VTU files
with multiple camera angles for comprehensive flow field analysis.
"""

import pyvista as pv
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

# Set rendering to headless for Docker
pv.OFF_SCREEN = True

def generate_multi_angle_streamlines(
    vtk_path: str,
    output_dir: Path,
    job_name: str,
    angles: List[str] = None
) -> List[str]:
    """
    Generate streamline visualizations from multiple camera angles.
    
    Args:
        vtk_path: Path to OpenFOAM VTU file
        output_dir: Directory to save images
        job_name: Job name for file naming
        angles: List of angles to generate ['iso', 'front', 'side', 'top']
        
    Returns:
        List of paths to generated images
    """
    if angles is None:
        angles = ['iso', 'front', 'side', 'top']
    
    try:
        # Read OpenFOAM VTU
        mesh = pv.read(vtk_path)
        logger.info(f"Loaded mesh with {mesh.n_points} points, {mesh.n_cells} cells")
        
        # Debug: List available arrays
        logger.info(f"Point data arrays: {list(mesh.point_data.keys())}")
        logger.info(f"Cell data arrays: {list(mesh.cell_data.keys())}")
        
        # Find velocity field (OpenFOAM uses 'U')
        velocity_field = None
        for field_name in ['U', 'velocity', 'Velocity', 'VELOCITY']:
            if field_name in mesh.point_data:
                velocity_field = field_name
                break
            elif field_name in mesh.cell_data:
                mesh = mesh.cell_data_to_point_data()
                if field_name in mesh.point_data:
                    velocity_field = field_name
                    break
        
        if not velocity_field:
            logger.error(f"No velocity field found. Available: {list(mesh.point_data.keys())}")
            return []
        
        logger.info(f"Using velocity field: '{velocity_field}'")
        
        # Calculate velocity magnitude for coloring
        U = mesh.point_data[velocity_field]
        mesh['velocity_magnitude'] = np.linalg.norm(U, axis=1)
        
        # Get geometry bounds
        bounds = mesh.bounds
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        center = mesh.center
        
        logger.info(f"Mesh bounds: X[{xmin:.2f}, {xmax:.2f}] Y[{ymin:.2f}, {ymax:.2f}] Z[{zmin:.2f}, {zmax:.2f}]")
        
        # Generate streamlines from multiple seed points
        # Place seeds near inlet (assuming z-min is inlet based on config)
        seed_points = []
        n_seeds = 30
        
        # Create grid of seed points at inlet
        for i in range(n_seeds):
            theta = 2 * np.pi * i / n_seeds
            r_frac = 0.5 + 0.4 * (i % 5) / 5  # Varying radii
            r = r_frac * min(xmax - xmin, ymax - ymin) / 2
            x = center[0] + r * np.cos(theta)
            y = center[1] + r * np.sin(theta)
            seed_points.append([x, y, zmin + 0.01*(zmax-zmin)])
        
        seeds = pv.PolyData(seed_points)
        
        # Generate streamlines
        streamlines = mesh.streamlines_from_source(
            seeds,
            vectors=velocity_field,
            integration_direction='forward',
            max_time=100.0,
            initial_step_length=0.01,
            max_step_length=0.1
        )
        
        if streamlines.n_points == 0:
            logger.warning("No streamlines generated, using direct mesh visualization")
            return _generate_mesh_fallback(mesh, output_dir, job_name, angles)
        
        logger.info(f"Generated {streamlines.n_lines} streamlines with {streamlines.n_points} points")
        
        # Create tube representation for better visibility
        tubes = streamlines.tube(radius=(xmax-xmin)*0.002)
        
        # Ensure magnitude is available on tubes
        if 'velocity_magnitude' not in tubes.point_data and velocity_field in tubes.point_data:
            U_tubes = tubes.point_data[velocity_field]
            tubes['velocity_magnitude'] = np.linalg.norm(U_tubes, axis=1)
        
        # Generate views for each requested angle
        image_paths = []
        
        for angle in angles:
            output_path = output_dir / f"{job_name}_streamlines_{angle}.png"
            
            plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
            plotter.set_background('white')
            
            # Add the actual geometry as semi-transparent surface for context
            # This shows what the flow is interacting with
            plotter.add_mesh(
                mesh,
                color='lightblue',
                opacity=0.4,
                show_edges=True,
                edge_color='darkblue',
                line_width=1
            )
            
            # Add streamline tubes colored by velocity
            if 'velocity_magnitude' in tubes.point_data:
                plotter.add_mesh(
                    tubes,
                    scalars='velocity_magnitude',
                    cmap='jet',
                    opacity=0.95,
                    show_scalar_bar=True,
                    scalar_bar_args={'title': 'Velocity (m/s)', 'vertical': True}
                )
            else:
                plotter.add_mesh(tubes, color='blue', opacity=0.8)
            
            # Set camera position based on angle
            if angle == 'iso':
                plotter.view_isometric()
                title = f"{job_name} - Isometric View"
            elif angle == 'front':
                plotter.view_yz()
                title = f"{job_name} - Front View (YZ)"
            elif angle == 'side':
                plotter.view_xz()
                title = f"{job_name} - Side View (XZ)"
            elif angle == 'top':
                plotter.view_xy()
                title = f"{job_name} - Top View (XY)"
            else:
                plotter.view_isometric()
                title = f"{job_name} - {angle.capitalize()}"
            
            plotter.add_text(title, position='upper_left', font_size=14, color='black')
            plotter.camera.zoom(1.2)
            
            # Save screenshot
            plotter.screenshot(str(output_path))
            plotter.close()
            
            logger.info(f"Saved {angle} view to {output_path}")
            image_paths.append(str(output_path))
        
        return image_paths
        
    except Exception as e:
        logger.exception(f"Failed to generate multi-angle streamlines: {e}")
        return []


def _generate_mesh_fallback(mesh: pv.DataSet, output_dir: Path, job_name: str, angles: List[str]) -> List[str]:
    """Generate mesh-only visualizations when streamlines fail"""
    logger.info("Generating mesh-only fallback visualizations")
    image_paths = []
    
    for angle in angles:
        try:
            output_path = output_dir / f"{job_name}_mesh_{angle}.png"
            
            plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
            plotter.set_background('white')
            
            # Show mesh with edges
            plotter.add_mesh(mesh, color='lightblue', opacity=0.6, show_edges=True, edge_color='gray')
            
            # Set camera
            if angle == 'iso':
                plotter.view_isometric()
            elif angle == 'front':
                plotter.view_yz()
            elif angle == 'side':
                plotter.view_xz()
            elif angle == 'top':
                plotter.view_xy()
            
            plotter.add_text(f"{job_name} - Mesh ({angle.capitalize()})", 
                           position='upper_left', font_size=14, color='black')
            plotter.camera.zoom(1.2)
            
            plotter.screenshot(str(output_path))
            plotter.close()
            
            image_paths.append(str(output_path))
            logger.info(f"Saved mesh fallback for {angle}")
            
        except Exception as e:
            logger.error(f"Failed to generate mesh fallback for {angle}: {e}")
    
    return image_paths
