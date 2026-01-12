
import sys
import os
from pathlib import Path
import numpy as np

# Add backend to path
sys.path.append(os.path.abspath("backend"))

from slicing import parse_msh_for_slicing, generate_slice_mesh

def test_slicing():
    # Adjust path to where I am running
    msh_path = Path(r"c:\Users\markm\Downloads\MeshPackageLean\apps\cli\generated_meshes\core_sample_mesh.msh")
    
    if not msh_path.exists():
        print(f"Error: {msh_path} not found")
        return

    print(f"Parsing {msh_path}...")
    nodes, elements = parse_msh_for_slicing(str(msh_path))
    
    print(f"Nodes: {len(nodes)}")
    print(f"Volume Elements: {len(elements)}")
    
    # Check element types
    if elements:
        types = set(e[0] for e in elements)
        print(f"Element types found: {types}")
    
    if not elements:
        print("Error: No volume elements found!")
        return

    # Calculate bounds
    pts = np.array(list(nodes.values()))
    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    size = bbox_max - bbox_min
    
    print(f"Bounds: {bbox_min} to {bbox_max}")
    print(f"Center: {center}")
    print(f"Size: {size}")

    # Define plane (X axis, 50% offset)
    axis = 'x'
    offset_percent = 50
    plane_origin = center.tolist()
    plane_normal = [1, 0, 0] # X axis
    # Offset from max X (frontend sends offset from Max in api_server.py logic)
    # api_server.py: plane_origin[0] = bbox_max[0] - (offset_percent / 100.0) * size[0]
    plane_origin[0] = bbox_max[0] - (offset_percent / 100.0) * size[0]
    
    print(f"Cutting at plane origin: {plane_origin}, normal: {plane_normal}")
    
    # Fake quality map
    quality_map = {}
    
    # Generate slice
    print("Generating slice...")
    slice_data = generate_slice_mesh(nodes, elements, quality_map, plane_origin, plane_normal)
    
    print("Slice Result:")
    vertices = slice_data.get('vertices', [])
    indices = slice_data.get('indices', [])
    colors = slice_data.get('colors', [])
    
    print(f"  Vertices: {len(vertices)} ({len(vertices)//3} points)")
    print(f"  Indices: {len(indices)} ({len(indices)//3} triangles)")
    print(f"  Colors: {len(colors)}")
    
    if len(indices) == 0:
        print("FAILURE: No triangles generated!")

if __name__ == "__main__":
    test_slicing()
