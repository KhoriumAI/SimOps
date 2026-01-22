"""
Interactive 3D Thermal Results Viewer
=====================================

Allows interactive inspection of simulation results using PyVista.
Supports interactive clipping/slicing to investigate internal temperature gradients.

Usage:
    python tools/visualize_interactive.py --case ./thermal_runs/cases/case_low_power_natural
    python tools/visualize_interactive.py --mock  # Run with mock data for testing
"""

import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import pyvista as pv

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def create_mock_heatsink():
    """Create a mock heatsink mesh with realistic temperature gradients"""
    # Geometry
    base = pv.Cube(center=(0,0,0), x_length=0.1, y_length=0.05, z_length=0.01)
    
    # Add fins
    fins = []
    for x_offset in [-0.03, 0.0, 0.03]:
        fin = pv.Cube(center=(x_offset, 0, 0.015), x_length=0.01, y_length=0.05, z_length=0.02)
        fins.append(fin)
    
    # Combine geometry
    heatsink = base
    for fin in fins:
        heatsink += fin
        
    # Generate Physics-like Temperature Field
    # Source at center bottom (hot spot)
    centers = heatsink.cell_centers().points
    x, y, z = centers[:, 0], centers[:, 1], centers[:, 2]
    
    # Gaussian heat source centered at (0,0,0)
    # Decay faster to reach ambient at edges
    # Ambient = 27C, Max Source = 85C (Delta = 58C)
    
    # Distance from center
    r2 = x**2 + y**2 + (z + 0.005)**2 
    
    # Temperature function: T = T_amb + dT * exp(-r^2 / sigma^2)
    # Adjusted sigma to ensure edges cool down significantly
    sigma_sq = 0.002  # Narrower spread
    
    T_ambient = 27.0
    T_max = 85.0
    dT = T_max - T_ambient
    
    temps = T_ambient + dT * np.exp(-r2 / sigma_sq)
    
    # Add some convection cooling effect (cooler at top of fins)
    # Reduce temp based on Z height (z goes -0.005 to 0.025)
    z_factor = (z + 0.005) / 0.03  # 0 to 1
    temps -= 10 * z_factor * (temps - T_ambient) / dT  # Cool top more where T is high
    
    heatsink['Temperature'] = temps
    
    return heatsink

def load_case_data(case_dir: Path):
    """Load OpenFOAM case data"""
    case_dir = Path(case_dir)
    foam_file = case_dir / "case.foam"
    
    # Create empty .foam if missing (to trigger reader)
    if not foam_file.exists():
        with open(foam_file, 'w') as f: pass
        
    try:
        reader = pv.POpenFOAMReader(str(foam_file))
        logger.info(f"Available time steps: {reader.time_values}")
        
        # Load latest time
        reader.set_active_time_value(reader.time_values[-1])
        mesh = reader.read()
        
        # OpenFOAM reader often returns MultiBlock. Combine or extract relevant block.
        if isinstance(mesh, pv.MultiBlock):
            # Try to combine or pick internal mesh
            # Usually block 0 is internalMesh if using standard reader
            logger.info("Dataset is MultiBlock, combining...")
            flat_mesh = pv.UnstructuredGrid()
            for block in mesh:
                if isinstance(block, pv.UnstructuredGrid):
                    flat_mesh = flat_mesh.merge(block)
            mesh = flat_mesh
            
        return mesh
        
    except Exception as e:
        logger.error(f"Failed to load case data: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Interactive 3D Thermal Viewer')
    parser.add_argument('--case', help='Path to OpenFOAM case directory')
    parser.add_argument('--mock', action='store_true', help='Use mock data')
    args = parser.parse_args()
    
    mesh = None
    title = ""
    
    if args.mock:
        logger.info("Generating mock data...")
        mesh = create_mock_heatsink()
        title = "Mock Thermal Result"
    elif args.case:
        logger.info(f"Loading case: {args.case}")
        mesh = load_case_data(args.case)
        title = f"Case: {Path(args.case).name}"
    else:
        logger.error("Must specify --case or --mock")
        sys.exit(1)
        
    if mesh is None:
        logger.error("No mesh loaded.")
        sys.exit(1)
        
    # Setup Plotter
    p = pv.Plotter(notebook=False)
    p.add_text(title, position='upper_left', font_size=12)
    
    # Add mesh with clipping capabilities
    # We add a 'capped' look by showing the plane widget but filling the mesh
    
    logger.info("Controls:")
    logger.info(" - Click and drag to rotate")
    logger.info(" - 'W' for wireframe, 'S' for surface")
    logger.info(" - Drag the arrow/frame to clip the mesh")
    
    # Add the volume with clipping
    p.add_mesh_clip_plane(
        mesh, 
        scalars='Temperature',
        cmap='inferno',
        show_edges=False,
        crinkle=False, # Smooth cut
        scalar_bar_args={'title': 'Temperature (°C)'}
    )
    
    # Add axes
    p.add_axes()
    
    # Show
    p.show()

if __name__ == '__main__':
    main()
