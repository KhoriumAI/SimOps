"""
Velocity Visualization Module
=============================

Provides fallback 2D/3D velocity field visualization using PyVista/Matplotlib.
This is Tier 2 in the visualization pipeline (after ParaView).
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Try to import visualization libraries
try:
    import pyvista as pv
    HAVE_PYVISTA = True
except ImportError:
    HAVE_PYVISTA = False
    logger.warning("PyVista not available - velocity visualization will be limited")

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    HAVE_MATPLOTLIB = True
except ImportError:
    HAVE_MATPLOTLIB = False
    logger.warning("Matplotlib not available")


def generate_velocity_streamlines(
    vtk_path: str, 
    output_path: str, 
    title: str = "Velocity Streamlines"
) -> bool:
    """
    Generate velocity streamline visualization using PyVista.
    
    This is a fallback visualization method when ParaView is not available.
    
    Args:
        vtk_path: Path to VTK/VTU file with velocity field
        output_path: Output PNG path
        title: Title for the plot
        
    Returns:
        True if successful, False otherwise
    """
    if not HAVE_PYVISTA:
        logger.error("PyVista not available for velocity visualization")
        return False
    
    try:
        mesh = pv.read(vtk_path)
        
        # Debug: Log available arrays
        logger.info(f"[VTK Debug] Cell data arrays: {list(mesh.cell_data.keys())}")
        logger.info(f"[VTK Debug] Point data arrays: {list(mesh.point_data.keys())}")
        logger.info(f"[VTK Debug] Mesh points: {mesh.n_points}, cells: {mesh.n_cells}")
        
        # OpenFOAM often outputs Cell Data. Streamlines need Point Data.
        if 'U' in mesh.cell_data and 'U' not in mesh.point_data:
            logger.info("[VTK] Converting cell data to point data")
            mesh = mesh.cell_data_to_point_data()
        
        # Check if U exists
        if 'U' not in mesh.point_data:
            logger.error("No velocity field 'U' in VTK file")
            logger.info(f"Available point data: {list(mesh.point_data.keys())}")
            logger.info(f"Available cell data: {list(mesh.cell_data.keys())}")
            return False
        
        # Get velocity data
        velocity = mesh.point_data['U']
        
        # Calculate velocity magnitude for coloring
        if velocity.ndim == 2 and velocity.shape[1] == 3:
            vel_mag = np.linalg.norm(velocity, axis=1)
        else:
            vel_mag = np.abs(velocity)
        
        mesh.point_data['Velocity Magnitude'] = vel_mag
        
        logger.info(f"[VTK] Velocity range: {vel_mag.min():.4f} - {vel_mag.max():.4f} m/s")
        
        # Check for valid velocity data
        if vel_mag.max() < 1e-10:
            logger.warning("[VTK] Velocity field is essentially zero - this may be a failed simulation")
        
        # Setup plotter
        pl = pv.Plotter(off_screen=True)
        pl.set_background('white')
        
        # Get mesh bounds for streamline seeding
        bounds = mesh.bounds
        center = [
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2,
        ]
        
        # Create seed line for streamlines (along inlet direction)
        x_range = bounds[1] - bounds[0]
        y_range = bounds[3] - bounds[2]
        z_range = bounds[5] - bounds[4]
        
        # Seed from one side
        x_start = bounds[0] + x_range * 0.1  # 10% from min X
        
        num_seeds = 15
        seed_points = np.zeros((num_seeds * num_seeds, 3))
        idx = 0
        for i in range(num_seeds):
            for j in range(num_seeds):
                seed_points[idx, 0] = x_start
                seed_points[idx, 1] = bounds[2] + y_range * (0.1 + 0.8 * i / (num_seeds - 1))
                seed_points[idx, 2] = bounds[4] + z_range * (0.1 + 0.8 * j / (num_seeds - 1))
                idx += 1
        
        seed_source = pv.PolyData(seed_points)
        
        try:
            # Generate streamlines
            streamlines = mesh.streamlines_from_source(
                seed_source,
                vectors='U',
                max_time=1000.0,
                integration_direction='both',
            )
            
            if streamlines.n_points == 0:
                logger.warning("[VTK] No streamlines generated - trying slice visualization instead")
                return _generate_velocity_slice_fallback(mesh, output_path, title)
            
            # Add streamlines colored by velocity magnitude
            pl.add_mesh(
                streamlines,
                scalars='Velocity Magnitude',
                cmap='turbo',
                line_width=2,
                show_scalar_bar=True,
                scalar_bar_args={'title': 'Velocity (m/s)'}
            )
            
        except Exception as e:
            logger.warning(f"[VTK] Streamline generation failed: {e}")
            return _generate_velocity_slice_fallback(mesh, output_path, title)
        
        # Add mesh outline for context
        pl.add_mesh(mesh.outline(), color='gray', line_width=1)
        
        # Camera setup
        pl.view_isometric()
        pl.camera.zoom(1.2)
        
        # Add title
        pl.add_text(title, position='upper_left', font_size=14, color='black')
        
        # Save
        pl.screenshot(output_path)
        pl.close()
        
        logger.info(f"[VTK] Saved velocity visualization to: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Velocity visualization failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def _generate_velocity_slice_fallback(
    mesh: 'pv.DataSet',
    output_path: str,
    title: str
) -> bool:
    """
    Generate a 2D velocity slice when streamlines fail.
    
    This is a fallback within the fallback - used when streamline
    generation produces empty results.
    """
    try:
        # Get mesh center
        bounds = mesh.bounds
        center = [
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2,
        ]
        
        # Create a slice through the center (XY plane at Z-mid)
        sliced = mesh.slice(normal='z', origin=center)
        
        if sliced.n_points == 0:
            # Try XZ plane
            sliced = mesh.slice(normal='y', origin=center)
        
        if sliced.n_points == 0:
            logger.error("[VTK] Could not create any valid slices")
            return False
        
        # Setup plotter
        pl = pv.Plotter(off_screen=True)
        pl.set_background('white')
        
        # Plot the slice with velocity magnitude
        if 'Velocity Magnitude' in sliced.point_data:
            scalars = 'Velocity Magnitude'
        elif 'U' in sliced.point_data:
            # Calculate magnitude from U
            U = sliced.point_data['U']
            sliced.point_data['Velocity Magnitude'] = np.linalg.norm(U, axis=1)
            scalars = 'Velocity Magnitude'
        else:
            scalars = None
        
        pl.add_mesh(
            sliced,
            scalars=scalars,
            cmap='turbo',
            show_scalar_bar=True if scalars else False,
            scalar_bar_args={'title': 'Velocity (m/s)'} if scalars else {}
        )
        
        pl.view_xy()
        pl.add_text(f"{title} (Slice View)", position='upper_left', font_size=12, color='black')
        
        pl.screenshot(output_path)
        pl.close()
        
        logger.info(f"[VTK] Saved velocity slice to: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Velocity slice fallback failed: {e}")
        return False


def generate_mesh_viz(
    vtk_path: str, 
    output_path: str, 
    title: str = "Mesh Visualization"
) -> bool:
    """
    Generate simple mesh visualization (last resort fallback).
    
    This is Tier 3 - used when velocity field is missing or invalid.
    Shows the mesh geometry without any field coloring.
    
    Args:
        vtk_path: Path to VTK/VTU file
        output_path: Output PNG path
        title: Title for the plot
        
    Returns:
        True if successful, False otherwise
    """
    if not HAVE_PYVISTA:
        logger.error("PyVista not available for mesh visualization")
        return False
    
    try:
        mesh = pv.read(vtk_path)
        
        logger.info(f"[Mesh Viz] Loaded mesh with {mesh.n_points} points, {mesh.n_cells} cells")
        
        if mesh.n_points == 0:
            logger.error("[Mesh Viz] Empty mesh - cannot visualize")
            return False
        
        # Setup plotter
        pl = pv.Plotter(off_screen=True)
        pl.set_background('white')
        
        # Add mesh surface with edges
        pl.add_mesh(
            mesh,
            color='lightblue',
            show_edges=True,
            edge_color='gray',
            opacity=0.8
        )
        
        # Camera setup
        pl.view_isometric()
        pl.camera.zoom(1.2)
        
        # Add title
        pl.add_text(title, position='upper_left', font_size=14, color='black')
        
        # Add bounds annotation
        bounds = mesh.bounds
        dims = f"X: {bounds[1]-bounds[0]:.2f}, Y: {bounds[3]-bounds[2]:.2f}, Z: {bounds[5]-bounds[4]:.2f}"
        pl.add_text(dims, position='lower_left', font_size=10, color='gray')
        
        # Save
        pl.screenshot(output_path)
        pl.close()
        
        logger.info(f"[Mesh Viz] Saved mesh visualization to: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Mesh visualization failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def generate_pressure_contour(
    vtk_path: str,
    output_path: str,
    title: str = "Pressure Distribution"
) -> bool:
    """
    Generate pressure contour visualization.
    
    Args:
        vtk_path: Path to VTK/VTU file with pressure field
        output_path: Output PNG path
        title: Title for the plot
        
    Returns:
        True if successful, False otherwise
    """
    if not HAVE_PYVISTA:
        return False
    
    try:
        mesh = pv.read(vtk_path)
        
        # Check for pressure field
        if 'p' not in mesh.point_data and 'p' not in mesh.cell_data:
            logger.error("No pressure field 'p' in VTK file")
            return False
        
        # Convert cell data if needed
        if 'p' in mesh.cell_data and 'p' not in mesh.point_data:
            mesh = mesh.cell_data_to_point_data()
        
        pressure = mesh.point_data['p']
        logger.info(f"[Pressure] Range: {pressure.min():.4f} - {pressure.max():.4f} Pa")
        
        # Setup plotter
        pl = pv.Plotter(off_screen=True)
        pl.set_background('white')
        
        # Add surface mesh colored by pressure
        pl.add_mesh(
            mesh,
            scalars='p',
            cmap='coolwarm',
            show_scalar_bar=True,
            scalar_bar_args={'title': 'Pressure (Pa)'}
        )
        
        pl.view_isometric()
        pl.camera.zoom(1.2)
        pl.add_text(title, position='upper_left', font_size=14, color='black')
        
        pl.screenshot(output_path)
        pl.close()
        
        logger.info(f"[Pressure] Saved visualization to: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Pressure visualization failed: {e}")
        return False
