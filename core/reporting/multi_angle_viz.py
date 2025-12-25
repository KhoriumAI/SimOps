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
        # Place seeds at inlet (X-min plane) for +X velocity flow
        # USER REQUEST: Zoom in closer and make streamlines dense just around the cube coordinates.
        # We assume the object is at (0,0,0).
        
        # Dimensions estimation (Object is usually much smaller than tunnel)
        # We'll assume object is roughly within 20% of the minor dimension of the tunnel
        # but we want to seed slightly wider to show flow going *around* it.
        tunnel_min_dim = min(ymax - ymin, zmax - zmin)
        # USER REQUEST: Seed closer to cube. Reduced from 0.15 to 0.08 (8% radius).
        target_radius = tunnel_min_dim * 0.08 
        if target_radius < 0.1: target_radius = 50.0 # Fallback/Sensitivity for absolute units?? 
        # Actually, let's stick to relative for generalization. 
        # Better heuristic: The object is usually "small" relative to domain.
        # For the "cube" case, domain is large.
        
        # New High-Density Localized Seeding
        # Center of seeding at (xmin, 0, 0) assuming object at (0,0,0)
        # Width/Height of seed window = target_radius * 2.5 (to cover object + immediate surroundings)
        
        seed_points = []
        seed_window_y = [-target_radius * 1.5, target_radius * 1.5]
        seed_window_z = [-target_radius * 1.5, target_radius * 1.5]
        
        # Grid seeding was too periodic. User requested random seeding.
        # USER REQUEST: Ensure streamlines appear on BOTH sides of the object.
        # Using stratified sampling to guarantee symmetric coverage.
        n_seeds = 80  # Increased for better coverage
        
        # Calculate enclosure dimensions
        L_x = xmax - xmin
        L_y = ymax - ymin
        L_z = zmax - zmin
        enclosure_diagonal = np.sqrt(L_x**2 + L_y**2 + L_z**2)
        
        logger.info(f"Seeding Window Y: {seed_window_y}, Z: {seed_window_z}")
        
        # Create random seed points at inlet (X-min plane)
        x_seed = xmin + 0.02 * L_x
        
        # STRATIFIED RANDOM SAMPLING: Divide into 4 quadrants to ensure coverage on all sides
        rng = np.random.default_rng(42)
        seeds_per_quadrant = n_seeds // 4
        
        quadrants = [
            (seed_window_y[0], 0, seed_window_z[0], 0),        # Y-, Z-
            (0, seed_window_y[1], seed_window_z[0], 0),        # Y+, Z-
            (seed_window_y[0], 0, 0, seed_window_z[1]),        # Y-, Z+
            (0, seed_window_y[1], 0, seed_window_z[1]),        # Y+, Z+
        ]
        
        for y_min, y_max, z_min, z_max in quadrants:
            y_coords = rng.uniform(y_min, y_max, seeds_per_quadrant)
            z_coords = rng.uniform(z_min, z_max, seeds_per_quadrant)
            for i in range(seeds_per_quadrant):
                seed_points.append([x_seed, y_coords[i], z_coords[i]])
        
        seeds = pv.PolyData(seed_points)
        logger.info(f"Created {len(seed_points)} stratified seed points at X={x_seed:.2f}")
        
        # Calculate streamline length
        # Since we are zoomed in, we still need them long enough to pass the object
        # but we definitely need high step count for detailed curve around object
        max_streamline_length = enclosure_diagonal * 2.0
        
        # Run streamlines
        # Note: We explicitly set max_time to a large value to avoid premature termination,
        # even if PyVista warns it is deprecated (some versions might still enforce a default).
        streamlines = mesh.streamlines_from_source(
            seeds,
            vectors=velocity_field,
            integration_direction='forward',
            max_steps=20000,           # Increased from 5000
            max_time=10000.0,          # Large value to ensure time isn't the limiter
            initial_step_length=L_x * 0.001,
            max_step_length=L_x * 0.05, # Increased max step
            terminal_speed=1e-10       # Lowered terminal speed
        )
        
        if streamlines.n_points == 0:
            logger.warning("No streamlines generated, using direct mesh visualization")
            return _generate_mesh_fallback(mesh, output_dir, job_name, angles)
        
        logger.info(f"Generated {streamlines.n_lines} streamlines with {streamlines.n_points} points")
        
        # USER SUGGESTION: Filter out streamlines with <10% velocity variation
        # These represent undisturbed far-field flow and just clutter the visualization.
        if velocity_field in streamlines.point_data and streamlines.n_lines > 0:
            try:
                U_stream = streamlines.point_data[velocity_field]
                vel_mag = np.linalg.norm(U_stream, axis=1) if U_stream.ndim == 2 else U_stream
                streamlines['velocity_magnitude'] = vel_mag
                
                # Extract individual streamlines and filter
                lines = streamlines.lines
                filtered_cells = []
                i = 0
                while i < len(lines):
                    n_pts = lines[i]
                    cell_indices = lines[i+1:i+1+n_pts]
                    
                    if len(cell_indices) > 1:
                        cell_vel = vel_mag[cell_indices]
                        vel_min, vel_max = cell_vel.min(), cell_vel.max()
                        # Keep if variation > 10%
                        if vel_min > 0 and (vel_max - vel_min) / vel_min > 0.10:
                            filtered_cells.append((n_pts, cell_indices))
                    
                    i += n_pts + 1
                
                if len(filtered_cells) > 0 and len(filtered_cells) < streamlines.n_lines:
                    # Rebuild filtered streamlines
                    keep_pts = set()
                    for n_pts, indices in filtered_cells:
                        keep_pts.update(indices)
                    keep_pts = sorted(keep_pts)
                    
                    old_to_new = {old: new for new, old in enumerate(keep_pts)}
                    new_points = streamlines.points[keep_pts]
                    new_lines = []
                    for n_pts, indices in filtered_cells:
                        new_lines.append(n_pts)
                        new_lines.extend([old_to_new[idx] for idx in indices])
                    
                    streamlines = pv.PolyData(new_points, lines=new_lines)
                    # Re-copy velocity data
                    streamlines['velocity_magnitude'] = vel_mag[keep_pts]
                    if U_stream.ndim == 2:
                        streamlines[velocity_field] = U_stream[keep_pts]
                    
                    logger.info(f"Filtered to {streamlines.n_lines} streamlines with >10% velocity variation")
            except Exception as e:
                logger.warning(f"Velocity filtering failed: {e}, using unfiltered streamlines")
        
        # Scale tube radius down significantly since we have many more of them now
        # and we are zooming in.
        # Reduced further (0.01 -> 0.005) per user request.
        # USER REQUEST (Latest): "A bit thicker". Increased back to 0.008.
        tube_radius = target_radius * 0.008 
        tubes = streamlines.tube(radius=tube_radius)
        
        # Ensure magnitude is available on tubes
        if 'velocity_magnitude' not in tubes.point_data and velocity_field in tubes.point_data:
            U_tubes = tubes.point_data[velocity_field]
            tubes['velocity_magnitude'] = np.linalg.norm(U_tubes, axis=1)
        
        # DOPPLER COLORING: Add individual velocity components for planar views
        # This disambiguates streamlines pointing at/away from camera (orthographic collapse fix)
        if velocity_field in tubes.point_data:
            U_tubes = tubes.point_data[velocity_field]
            if U_tubes.ndim == 2 and U_tubes.shape[1] >= 3:
                tubes['velocity_x'] = U_tubes[:, 0]  # For Front view (YZ plane)
                tubes['velocity_y'] = U_tubes[:, 1]  # For Side view (XZ plane)
                tubes['velocity_z'] = U_tubes[:, 2]  # For Top view (XY plane)
        
        # Generate views for each requested angle
        image_paths = []
        
        for angle in angles:
            output_path = output_dir / f"{job_name}_streamlines_{angle}.png"
            
            plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
            plotter.set_background('white')
            
            # Calculate focus region (Object Origin)
            # We assume the object is near (0,0,0) as per standard meshing alignment.
            # The mesh center is usually far downstream due to the wake region.
            # Focus on a region around (0,0,0) OR xmin if (0,0,0) is far?
            # Safe bet: Focus on (0,0,0) but ensure we capture a reasonable volume.
            
            # Dimensions of the "Object of Interest"
            # We used target_radius for seeding, let's use it for view bounds too.
            # Only go slightly larger than seeding window to keep focus tight
            view_scale = target_radius * 2.0
            
            # Define bounds centered at Origin (0,0,0)
            focus_bounds = [
                -view_scale, view_scale,
                -view_scale, view_scale,
                -view_scale, view_scale
            ]
            
            # Clip mesh to focus region for context display (The Cutaway)
            # But for the "Outer Box", we just want the outline.
            
            # 1. Add Domain Outline (Wireframe) - Shows the full tunnel extent
            plotter.add_mesh(
                mesh.outline(),
                color='black',
                opacity=0.1, # Fainter outline
                line_width=1
            )
            
            # 2. Add the Object (The Hole)
            # Accessing the inner surface is tricky with just VTU.
            # Strategy: Clip the mesh to the focus region and show it semi-transparent.
            # This allows seeing the object "void" if we are close enough.
            focus_box = pv.Box(bounds=focus_bounds)
            try:
                # We show a localized cutaway of the mesh around the object
                clipped_mesh = mesh.clip_box(focus_bounds, invert=False)
                # Show this cutaway very disjointly so it doesn't block view
                plotter.add_mesh(
                    clipped_mesh,
                    color='grey',
                    opacity=0.1,  # Very transparent to see streamlines inside
                    show_edges=True,
                    edge_color='black',
                    line_width=0.5
                )
            except:
                pass 
            
            # Clip tubes to a reasonable region so they don't clutter the view 
            # if they go off to infinity, but we want to see the wake.
            # Extend +X bound deeper downstream.
            # USER REQUEST: Crop them off before they hit the color bar.
            # Reduced X-max from view_scale * 6.0 to 1.5 to clip earlier.
            viz_bounds = [
                -view_scale * 2.0, view_scale * 1.5, # Reduced downstream extent
                -view_scale * 1.5, view_scale * 1.5,
                -view_scale * 1.5, view_scale * 1.5
            ]
            try:
                clipped_tubes = tubes.clip_box(viz_bounds, invert=False) if tubes.n_points > 0 else tubes
            except:
                clipped_tubes = tubes
            
            # Add streamline tubes colored by velocity
            if 'velocity_magnitude' in clipped_tubes.point_data:
                # For Planar views, we want a "True Cross Section".
                # This means we should only show streamlines that exist within a thin slice
                # of the plane we are looking at.
                
                view_tubes = clipped_tubes
                # USER FIX: Slice was too thin (10%), missing most streamlines.
                # Increased to 50% for better coverage while still showing cross-section.
                slice_thick = view_scale * 0.5
                
                if angle == 'top': # XY plane, looking down Z
                    # Clip Z to a slab around 0
                    try:
                        view_tubes = view_tubes.clip_box(
                            [-1e9, 1e9, -1e9, 1e9, -slice_thick, slice_thick], invert=False
                        )
                    except: pass
                elif angle == 'side': # XZ plane, looking along Y
                    # Clip Y to a thin slab around 0
                    try:
                        view_tubes = view_tubes.clip_box(
                            [-1e9, 1e9, -slice_thick, slice_thick, -1e9, 1e9], invert=False
                        )
                    except: pass
                
                if view_tubes.n_points > 0:
                    # DOPPLER COLORING: Color by the velocity component normal to the view plane
                    # This disambiguates "dots" - red = towards camera, blue = away
                    if angle == 'front' and 'velocity_x' in view_tubes.point_data:
                        scalar_name = 'velocity_x'
                        scalar_title = 'Velocity X (m/s)\n→ Towards you'
                    elif angle == 'side' and 'velocity_y' in view_tubes.point_data:
                        scalar_name = 'velocity_y'
                        scalar_title = 'Velocity Y (m/s)\n→ Towards you'
                    elif angle == 'top' and 'velocity_z' in view_tubes.point_data:
                        scalar_name = 'velocity_z'
                        scalar_title = 'Velocity Z (m/s)\n→ Towards you'
                    else:
                        scalar_name = 'velocity_magnitude'
                        scalar_title = 'Velocity (m/s)'
                    
                    # Use diverging colormap for directional components (RdBu_r: red=positive, blue=negative)
                    cmap = 'RdBu_r' if scalar_name != 'velocity_magnitude' else 'jet'
                    
                    plotter.add_mesh(
                        view_tubes,
                        scalars=scalar_name,
                        cmap=cmap,
                        opacity=0.95,
                        show_scalar_bar=True,
                        scalar_bar_args={'title': scalar_title, 'vertical': True}
                    )
            elif clipped_tubes.n_points > 0:
                plotter.add_mesh(clipped_tubes, color='blue', opacity=0.8)
            
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
            
            # Camera setup: Explicitly control centering and zoom
            # 'reset_camera' can sometimes introduce offsets based on bounds aspect ratio.
            # We want to force focus on (0,0,0).
            
            plotter.camera.focal_point = (0.0, 0.0, 0.0)
            
            # The view_scale determines the region of interest radius.
            # We want to zoom such that this region fills the view.
            # zoom(1.5) -> Magnify by 1.5x -> Viewport height shows 1/1.5 of the original height.
            # Original height relative to bounds [-s, s] is 2s.
            # ParallelScale is half-height of the view.
            
            current_scale = view_scale  # The "radius" of our interest bounds
            # Zoom 1.5 means we want to see LESS (magnify).
            # Effective parallel scale = current_scale / 1.5
            target_parallel_scale = current_scale / 1.5
            
            if angle == 'iso':
                # Perspective projection for 3D look
                plotter.camera.parallel_projection = False
                # Position equidistant for isometric
                # Zooming in/out in perspective is changing distance or view angle.
                # We'll use distance.
                iso_dist = view_scale * 3.0 # Base distance
                # Apply zoom: Closer distance = Higher zoom
                iso_dist /= 1.5 
                
                # Normalize vector (1,1,1) -> length sqrt(3)
                # We want position at distance iso_dist along (1,1,1)
                pos = np.array([1.0, 1.0, 1.0])
                pos = pos / np.linalg.norm(pos) * iso_dist
                plotter.camera.position = pos
                plotter.camera.up = (0, 0, 1)
                
            elif angle == 'front': # YZ plane, Look X+ (or X-)
                 plotter.camera.parallel_projection = True
                 plotter.camera.parallel_scale = target_parallel_scale
                 plotter.camera.position = (current_scale * 10, 0, 0) # Far out on X axis
                 plotter.camera.up = (0, 0, 1) # Z up
                 
            elif angle == 'side': # XZ plane, Look Y+ (or Y-)
                 plotter.camera.parallel_projection = True
                 plotter.camera.parallel_scale = target_parallel_scale
                 plotter.camera.position = (0, current_scale * 10, 0) # Far out on Y axis
                 plotter.camera.up = (0, 0, 1) # Z up
                 
            elif angle == 'top': # XY plane, Look Z+ (or Z-)
                 plotter.camera.parallel_projection = True
                 plotter.camera.parallel_scale = target_parallel_scale
                 plotter.camera.position = (0, 0, current_scale * 10) # Far out on Z axis
                 plotter.camera.up = (0, 1, 0) # Y up
            
            # Ensure text is re-added or adjusted if needed, but 'add_text' handles overlay.
            # No need to call reset_camera or zoom anymore since we set params explicitly.
            
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
