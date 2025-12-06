import sys
import numpy as np
import trimesh
import argparse
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from strategies.conformal_hex_glue import ConformalHexGenerator

def create_simple_grid(nx=2, ny=2, nz=2):
    """Create a simple structured hex grid."""
    x = np.linspace(0, 1, nx+1)
    y = np.linspace(0, 1, ny+1)
    z = np.linspace(0, 1, nz+1)
    
    vertices = []
    for k in range(nz+1):
        for j in range(ny+1):
            for i in range(nx+1):
                vertices.append([x[i], y[j], z[k]])
    vertices = np.array(vertices)
    
    hexes = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                n0 = i + j * (nx + 1) + k * (nx + 1) * (ny + 1)
                n1 = n0 + 1
                n2 = n0 + (nx + 1) + 1
                n3 = n0 + (nx + 1)
                n4 = n0 + (nx + 1) * (ny + 1)
                n5 = n4 + 1
                n6 = n4 + (nx + 1) + 1
                n7 = n4 + (nx + 1)
                hexes.append([n0, n1, n2, n3, n4, n5, n6, n7])
    hexes = np.array(hexes, dtype=np.int32)
    return vertices, hexes

def test_checkpoint_1():
    """Checkpoint 1: Boundary Node Detection"""
    print("\n=== Checkpoint 1: Boundary Node Detection ===")
    
    # Create 2x2x2 grid (8 elements)
    # 3x3x3 nodes = 27 nodes
    # Interior node should be at index 13 (center)
    vertices, hexes = create_simple_grid(2, 2, 2)
    
    gen = ConformalHexGenerator(None, verbose=False)
    boundary_nodes = gen._identify_boundary_nodes(hexes)
    
    print(f"Total nodes: {len(vertices)}")
    print(f"Identified boundary nodes: {len(boundary_nodes)}")
    
    # Expected: 27 nodes total, 1 interior node. 26 boundary nodes.
    # Interior node is at index: 1 + 1*(3) + 1*(9) = 13 (using i,j,k=1,1,1 and stride 3,9)
    # Let's verify index 13 is NOT in boundary_nodes
    
    is_13_boundary = 13 in boundary_nodes
    print(f"Is central node (13) on boundary? {is_13_boundary}")
    
    if len(boundary_nodes) == 26 and not is_13_boundary:
        print("PASS: Correctly identified boundary nodes.")
        return True
    else:
        print("FAIL: Incorrect boundary node count or identification.")
        return False

def test_checkpoint_2():
    """Checkpoint 2: Surface Projection"""
    print("\n=== Checkpoint 2: Surface Projection ===")
    
    # Create a cylinder mesh using trimesh
    cylinder = trimesh.creation.cylinder(radius=1.0, height=2.0, sections=32)
    # Vertex projection requires dense vertices. Subdivide to add vertices along the side.
    for _ in range(3):
        cylinder = cylinder.subdivide()
        
    print(f"Test cylinder vertices: {len(cylinder.vertices)}")
    
    # Create a test vertex outside the cylinder
    # Point at (1.2, 0, 0) should project to (1.0, 0, 0)
    # Point at (0, 1.2, 0) should project to (0, 1.0, 0)
    vertices = np.array([
        [1.2, 0.0, 0.0],  # Case A
        [0.0, 1.2, 0.0],  # Case B
        [0.5, 0.5, 3.0]   # Case C (Top cap projection?)
    ])
    
    indices = np.array([0, 1, 2])
    
    gen = ConformalHexGenerator(None, verbose=False)
    projected = gen._project_boundary_nodes(vertices, indices, cylinder)
    
    print("Original -> Projected:")
    for v, p in zip(vertices, projected):
        print(f"  {v} -> {p}")
        
    # Check Case A
    dist_a = np.linalg.norm(projected[0] - np.array([1.0, 0.0, 0.0]))
    # Check Case B
    dist_b = np.linalg.norm(projected[1] - np.array([0.0, 1.0, 0.0]))
    
    if dist_a < 0.01 and dist_b < 0.01:
        print("PASS: Projection logic works.")
        return True
    else:
        print(f"FAIL: Projection distances too large: {dist_a}, {dist_b}")
        return False

def test_checkpoint_3():
    """Checkpoint 3: Interior Smoothing"""
    print("\n=== Checkpoint 3: Interior Smoothing ===")
    
    # Create 2x2x2 grid (3x3x3 nodes)
    vertices, hexes = create_simple_grid(2, 2, 2)
    # Vertices range from 0 to 1
    
    # Move central node (index 13) close to a corner to simulate distortion
    # Center should be at (0.5, 0.5, 0.5)
    center_idx = 13
    original_pos = vertices[center_idx].copy()
    print(f"Original center: {original_pos}")
    
    vertices[center_idx] = [0.1, 0.1, 0.1] # move towards corner
    print(f"Distorted center: {vertices[center_idx]}")
    
    gen = ConformalHexGenerator(None, verbose=False)
    
    # Fix all nodes except 13
    fixed_mask = np.ones(len(vertices), dtype=bool)
    fixed_mask[center_idx] = False
    
    smoothed = gen._smooth_interior_nodes(vertices, hexes, fixed_mask, iterations=10)
    
    final_pos = smoothed[center_idx]
    print(f"Smoothed center: {final_pos}")
    
    # Should move back towards 0.5, 0.5, 0.5 (average of neighbors)
    # Neighbors of 13 are at 0.5 distance along axes (0,0.5,0), (1,0.5,0) etc?
    # Actually in 2x2x2 grid, neighbors of center are 6 face centers? 
    # Wait, connectivity of center node in 2x2x2 grid:
    # It's connected to 6 neighbors (Left, Right, Front, Back, Top, Bottom)
    # Their coords are (0,0.5,0.5), (1,0.5,0.5), etc.
    # Average of neighbors is exactly (0.5, 0.5, 0.5).
    
    dist_from_target = np.linalg.norm(final_pos - original_pos)
    
    if dist_from_target < 0.1:
        print("PASS: Smoothing restored node position.")
        return True
    else:
        print(f"FAIL: Smoothing did not converge close enough. Dist: {dist_from_target}")
        return False

def test_checkpoint_4():
    """Checkpoint 4: Integration (Mock)"""
    print("\n=== Checkpoint 4: Integration Test ===")
    # We can't easily run full integration without a STEP file.
    # We will simulate the generator flow.
    
    # Create a box mesh 
    vertices, hexes = create_simple_grid(4, 4, 4)
    # Scale to match cylinder
    vertices = vertices * 2.0 - 1.0 # -1 to 1
    
    # Target: Cylinder radius 1.5 (Box is size 2, so corners stick out, faces are inside)
    # Actually box from -1 to 1 fits inside radius 1.5 cylinder? No.
    # Box corners are at sqrt(1+1) = 1.414. Fits inside 1.5.
    # Let's make target cylinder smaller, radius 0.8, so box boundaries project INWARD.
    
    cylinder = trimesh.creation.cylinder(radius=0.8, height=2.0)
    
    gen = ConformalHexGenerator(None, verbose=True)
    
    # Manually run pipeline
    print("Running project_and_smooth...")
    new_verts = gen._project_and_smooth(vertices, hexes, cylinder)
    
    # Check if boundary nodes are at radius 0.8
    boundary_nodes = gen._identify_boundary_nodes(hexes)
    b_verts = new_verts[boundary_nodes]
    radii = np.linalg.norm(b_verts[:, :2], axis=1) # XY radius
    
    mean_radius = np.mean(radii)
    print(f"Mean boundary radius: {mean_radius:.3f} (Target 0.8)")
    
    if abs(mean_radius - 0.8) < 0.05:
        print("PASS: Mock integration successful.")
        return True
    else:
        print("FAIL: Boundary did not conform to cylinder.")
        return False

def test_checkpoint_5():
    """Checkpoint 5: Volumetric Filtering"""
    print("\n=== Checkpoint 5: Volumetric Filtering ===")
    
    from strategies.conformal_hex_glue import ChunkInfo
    
    # Create a simple convex chunk: Triangle Prism
    # Vertices: (0,0,0), (1,0,0), (0,1,0) base, extruded to z=1
    verts = np.array([
        [0,0,0], [1,0,0], [0,1,0],
        [0,0,1], [1,0,1], [0,1,1]
    ], dtype=np.float64)
    
    # Faces (indices) - simplified, just need logic to work
    # But filtering needs VALID faces with normals.
    # Let's use 4 faces of a Tetrahedron for simplicity?
    # Tet: (0,0,0), (1,0,0), (0,1,0), (0,0,1)
    # Bounding box is (0,0,0) to (1,1,1). Volume 1/6.
    # Box volume 1.
    # Filter should remove > 50% of hexes.
    
    tet_verts = np.array([
        [0,0,0], [1,0,0], [0,1,0], [0,0,1]
    ], dtype=np.float64)
    
    # Faces (OUTWARD winding order is crucial)
    # 0, 2, 1 (Bottom, z=0) -> N=(0,0,-1)
    # 0, 1, 3 (Front, y=0) -> N=(0,-1,0)
    # 0, 3, 2 (Left, x=0) -> N=(-1,0,0)
    # 1, 2, 3 (Diagonal) -> N=(1,1,1) normalized
    
    tet_faces = np.array([
        [0, 2, 1],
        [0, 1, 3],
        [0, 3, 2],
        [1, 2, 3]
    ], dtype=np.int32)
    
    chunk = ChunkInfo(0, tet_verts, tet_faces, np.mean(tet_verts, axis=0), 0.16)
    
    dim = 10
    grid_verts, hexes = create_simple_grid(dim, dim, dim)
    # Scale to 0..1? create_simple_grid does 0..1 by default.
    
    gen = ConformalHexGenerator(None, verbose=False)
    
    print(f"Total hexes in box: {len(hexes)}")
    
    filtered = gen._filter_hexes_inside_chunk(hexes, grid_verts, chunk)
    
    print(f"Filtered hexes: {len(filtered)}")
    
    # Volume of tet is 1/6 = 0.166
    # Ratio should be around 0.166
    ratio = len(filtered) / len(hexes)
    print(f"Ratio: {ratio:.3f} (Expected ~0.167)")
    
    if 0.1 < ratio < 0.25:
        print("PASS: Filtering kept reasonable subset of hexes.")
        return True
    else:
        print("FAIL: Ratio outside expected range.")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=int, default=0)
    args = parser.parse_args()
    
    if args.checkpoint in [0, 1]:
        if not test_checkpoint_1(): sys.exit(1)
    if args.checkpoint in [0, 2]:
        if not test_checkpoint_2(): sys.exit(1)
    if args.checkpoint in [0, 3]:
        if not test_checkpoint_3(): sys.exit(1)
    if args.checkpoint in [0, 4]:
        if not test_checkpoint_4(): sys.exit(1)
    if args.checkpoint in [0, 5]:
        if not test_checkpoint_5(): sys.exit(1)
        
    print("\nALL TESTS PASSED")
