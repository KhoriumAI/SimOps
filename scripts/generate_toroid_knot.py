"""
Generate a Toroidal Knot using Gmsh and export to STEP format
"""
import gmsh
import numpy as np
from pathlib import Path

def create_toroid_knot_gmsh(output_path, p=2, q=3, R=10, r=3, tube_radius=1.5, num_points=128):
    """
    Create a (p,q)-torus knot using Gmsh
    """
    gmsh.initialize()
    gmsh.model.add("ToroidKnot")
    
    # Generate knot curve points
    t_values = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    points = []
    
    for i, t in enumerate(t_values):
        # Parametric equations for (p,q)-torus knot
        x = (R + r * np.cos(q * t)) * np.cos(p * t)
        y = (R + r * np.cos(q * t)) * np.sin(p * t)
        z = r * np.sin(q * t)
        
        point_tag = gmsh.model.occ.addPoint(x, y, z)
        points.append(point_tag)
    
    # Create spline through points (closed loop)
    spline = gmsh.model.occ.addBSpline(points + [points[0]])
    
    # Create wire from spline
    wire = gmsh.model.occ.addWire([spline])
    
    # Create circular disk to sweep along the path
    # Place disk at first point, perpendicular to curve
    first_point_coords = [
        (R + r * np.cos(0)) * np.cos(0),
        (R + r * np.cos(0)) * np.sin(0),
        r * np.sin(0)
    ]
    
    # Create a small disk
    disk = gmsh.model.occ.addDisk(first_point_coords[0], first_point_coords[1], first_point_coords[2], 
                                   tube_radius, tube_radius)
    
    # Create pipe by sweeping disk along wire
    # Note: Gmsh pipe is addPipe(wireDimTag, [(dim, tag)])
    try:
        pipe_dimtags = gmsh.model.occ.addPipe([(2, disk)], wire)
        gmsh.model.occ.synchronize()
        
        # Write STEP file
        gmsh.model.occ.synchronize()
        gmsh.write(str(output_path))
        
        print(f"✓ Toroid Knot saved to: {output_path}")
        
    except Exception as e:
        print(f"Error creating pipe: {e}")
        print("Falling back to simpler torus shape...")
        
        # Fallback: Create a simple torus
        gmsh.clear()
        gmsh.model.add("SimpleTorus")
        torus = gmsh.model.occ.addTorus(0, 0, 0, R, r)
        gmsh.model.occ.synchronize()
        gmsh.write(str(output_path))
        print(f"✓ Simple Torus saved to: {output_path}")
    
    gmsh.finalize()

if __name__ == "__main__":
    output_file = Path("c:/Users/markm/Downloads/Simops/cad_files/ToroidKnot.step")
    print(f"Generating Toroidal Knot (2,3) with Gmsh...")
    print(f"Output: {output_file}")
    
    create_toroid_knot_gmsh(output_file, p=2, q=3, R=10, r=3, tube_radius=1.5, num_points=64)
