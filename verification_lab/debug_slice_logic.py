
import sys
import os
import json
import numpy as np

# Add backend to path (parent of verification_lab is root, then down to backend)
# verification_lab is at root level?
# MeshPackageLean/verification_lab
# MeshPackageLean/backend
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
backend_dir = os.path.join(root_dir, 'backend')
sys.path.append(backend_dir)

from slicing import generate_slice_mesh

def test_simple_tet_slice():
    print("Testing Simple Tet Slice...")
    # Tet vertices: 
    # 0: (0,0,0)
    # 1: (1,0,0)
    # 2: (0,1,0)
    # 3: (0,0,1)
    nodes = {
        0: [0.0, 0.0, 0.0],
        1: [1.0, 0.0, 0.0],
        2: [0.0, 1.0, 0.0],
        3: [0.0, 0.0, 1.0]
    }
    
    # Element: type 4 (Tet4), tag 1, nodes [0, 1, 2, 3]
    elements = [
        (4, 1, [0, 1, 2, 3])
    ]
    
    # Plane: x = 0.5 implies Normal=(1,0,0), Origin=(0.5, 0, 0)
    plane_origin = [0.5, 0.0, 0.0]
    plane_normal = [1.0, 0.0, 0.0]
    
    quality_map = {"1": 1.0} # Perfect quality
    
    result = generate_slice_mesh(nodes, elements, quality_map, plane_origin, plane_normal)
    
    vertices = result['vertices']
    print(f"Generated {len(vertices)//3} vertices")
    
    # Check if all vertices are on the plane (x=0.5)
    all_on_plane = True
    for i in range(0, len(vertices), 3):
        x, y, z = vertices[i], vertices[i+1], vertices[i+2]
        print(f"Pt {i//3}: ({x:.4f}, {y:.4f}, {z:.4f})")
        if abs(x - 0.5) > 1e-6:
            all_on_plane = False
            print(f"  ERROR: Point not on plane x=0.5!")
            
    if all_on_plane:
        print("SUCCESS: All points lie on the cutting plane.")
    else:
        print("FAILURE: Some points are NOT on the cutting plane.")
        
    # Check expected intersection points
    # We expect intersections on edges connected to node 1 (at x=1)
    # Edge 0-1: (0,0,0)-(1,0,0) -> (0.5, 0, 0)
    # Edge 1-2: (1,0,0)-(0,1,0) -> (0.5, 0.5, 0)
    # Edge 1-3: (1,0,0)-(0,0,1) -> (0.5, 0, 0.5)
    
    expected_points = [
        [0.5, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5]
    ]
    
    # Verify we found points close to these
    found_count = 0
    for exp in expected_points:
        found = False
        for i in range(0, len(vertices), 3):
            pt = [vertices[i], vertices[i+1], vertices[i+2]]
            dist = np.linalg.norm(np.array(pt) - np.array(exp))
            if dist < 1e-4:
                found = True
                break
        if found:
            found_count += 1
        else:
            print(f"  MISSING expected point: {exp}")
            
    if found_count == 3:
        print("SUCCESS: All expected intersection points found.")
    else:
        print(f"FAILURE: Only found {found_count}/3 expected points.")

if __name__ == "__main__":
    test_simple_tet_slice()
