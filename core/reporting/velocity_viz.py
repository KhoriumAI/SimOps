
import pyvista as pv
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def generate_mesh_viz(vtk_path: str, output_path: str, title: str = "Mesh Visualization"):
    """
    Generate valid Mesh-Only visualization (Wireframe + Surface)
    Used when solver fails or data is missing.
    """
    try:
        mesh = pv.read(vtk_path)
        
        plotter = pv.Plotter(off_screen=True)
        plotter.set_background('white')
        
        # Add wireframe
        plotter.add_mesh(mesh, style='wireframe', color='black', opacity=0.3)
        # Add surface with distinctive color indicating "No Data" or just geometry
        plotter.add_mesh(mesh, color='lightgray', opacity=0.5, show_edges=True)
        
        plotter.add_text(title, font_size=12, color='black')
        
        # Ensure consistent view
        plotter.view_xy()
        plotter.enable_parallel_projection()
        
        plotter.screenshot(output_path)
        logger.info(f"Saved mesh-only viz to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to generate mesh viz: {e}")
        return False

def generate_velocity_streamlines(vtk_path: str, output_path: str, title: str = "Velocity Streamlines"):
    """
    Generate 2D velocity streamline visualization.
    """
    try:
        mesh = pv.read(vtk_path)
        
        # OpenFOAM often outputs Cell Data. Streamlines need Point Data.
        if 'U' in mesh.cell_data and 'U' not in mesh.point_data:
            mesh = mesh.cell_data_to_point_data()
        
        # Check if U exists
        if 'U' not in mesh.point_data:
            logger.error("No velocity field 'U' in VTK file")
            # Fallback to mesh viz? Or return False so caller handles it?
            # Let's try to fallback internally? No, explicit control in worker is better.
            return False
            
        # For 2D / 1-cell thick mesh, slice it
        bounds = mesh.bounds
        z_mid = (bounds[4] + bounds[5]) / 2.0
        
        # Slice
        single_plane = mesh.slice(normal=[0,0,1], origin=[0,0,z_mid])
        
        # Streamlines
        # Source line at inlet
        xmin, xmax, ymin, ymax, _, _ = bounds
        
        # Source: Line at x = xmin + eps
        
        streamlines = single_plane.streamlines(
            vectors='U', 
            source_center=((xmin + 0.01), 0, z_mid),
            source_radius=(ymax - ymin)/2 * 0.9,
            n_points=50,
            integration_direction='forward'
        )
        
        # Calculate Magnitude for plotting
        streamlines['magU'] = np.linalg.norm(streamlines['U'], axis=1)
        
        # Plot
        plotter = pv.Plotter(off_screen=True)
        plotter.set_background('white')
        
        plotter.add_mesh(single_plane, color='white', opacity=0.1, style='wireframe')
        
        # Add Cylinder (mask?) - It's a hole, so mesh respects it.
        
        # Color streamlines by Velocity Magnitude
        # Note: tube filter propagates point data usually.
        tubes = streamlines.tube(radius=0.0005)
        
        # Robustness Check
        if 'magU' not in tubes.point_data:
            # Try to recover from U if present
            if 'U' in tubes.point_data:
                tubes['magU'] = np.linalg.norm(tubes['U'], axis=1)
            elif 'U' in tubes.cell_data:
                 tubes = tubes.cell_data_to_point_data()
                 if 'U' in tubes.point_data:
                     tubes['magU'] = np.linalg.norm(tubes['U'], axis=1)
        
        if 'magU' in tubes.point_data:
            plotter.add_mesh(tubes, scalars='magU', cmap='jet', lighting=False)
        else:
             # Fallback solid color
             logger.warning("Could not find velocity magnitude for coloring. Using solid color.")
             plotter.add_mesh(tubes, color='blue', lighting=False)
        
        plotter.add_text(title, font_size=12, color='black')
        plotter.view_xy()
        plotter.enable_parallel_projection()
        
        plotter.screenshot(output_path)
        logger.info(f"Saved streamline viz to {output_path}")
        return True
        
    except Exception as e:
        logger.exception(f"Failed to generate visualization: {e}")
        return False
