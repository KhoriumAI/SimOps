
import sys
import os
import json
from pathlib import Path
import meshio

# Add project root to path
sys.path.append('c:/Users/markm/Downloads/MeshPackageLean')

from backend.api_server import parse_msh_file

def test_parse():
    msh_path = r"C:\Users\markm\Downloads\MeshPackageLean\apps\cli\generated_meshes\50197f09-5ff8-442c-858f-2b81269c2c47_ge_jet_engine_bracket_benchmark_Medium_fast_tet.msh"
    
    print(f"Testing parse for: {msh_path}")
    
    # 1. Direct meshio inspection
    mesh = meshio.read(msh_path)
    print(f"Meshio loaded. Points: {len(mesh.points)}, Cell blocks: {len(mesh.cells)}")
    for i, block in enumerate(mesh.cells):
        print(f"  Block {i}: type={block.type}, count={len(block.data)}")
    
    print(f"Cell data keys: {mesh.cell_data.keys()}")
    if 'gmsh:element_id' in mesh.cell_data:
        ids = mesh.cell_data['gmsh:element_id']
        print(f"Found gmsh:element_id! Blocks with IDs: {len(ids)}")
        for i, block_ids in enumerate(ids):
            print(f"  Block {i} IDs sample: {block_ids[:5].tolist() if len(block_ids) > 0 else 'EMPTY'}")
    else:
        print("MISSING gmsh:element_id in meshio data")

    # 2. Run the main parse
    data = parse_msh_file(msh_path)
    
    # Check mapping success
    colors = data.get('colors', [])
    num_faces = len(data.get('vertices', [])) // 9
    blue_count = 0
    for i in range(0, len(colors), 3):
        r, g, b = colors[i], colors[i+1], colors[i+2]
        if abs(r - 0.29) < 0.01 and abs(g - 0.56) < 0.01 and abs(b - 0.89) < 0.01:
            blue_count += 1
    
    print(f"\nFinal Mapping Report:")
    print(f"  Total boundary faces extracted: {num_faces}")
    print(f"  Faces with MISSING quality (Blue): {blue_count}")
    print(f"  Faces with SUCCESSFUL quality: {num_faces - blue_count}")

if __name__ == "__main__":
    test_parse()
