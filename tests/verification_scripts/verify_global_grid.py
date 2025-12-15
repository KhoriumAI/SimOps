import numpy as np
import trimesh
import tempfile
import os
import sys

# Ensure project root is in path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from strategies.conformal_hex_glue import ConformalHexGenerator, AdjacencyGraph

def test_global_grid():
    print("=== Testing Global Hex Grid Generation ===")
    
    # 1. Create a simple Box
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    # Add some subdivision to the validation mesh? No, let the generator handle it.
    
    print(f"Created Box: {len(mesh.vertices)} vertices")
    
    # 2. Setup AdjacencyGraph (still required by constructor, though unused for grid gen)
    # We can pass empty chunks since generate() doesn't use graph.chunks anymore in new flow
    graph = AdjacencyGraph(verbose=True) 
    
    # 3. Create Generator
    generator = ConformalHexGenerator(graph, verbose=True)
    
    # 4. Generate
    print("Running generate()...")
    try:
        verts, hexes = generator.generate(divisions=4, reference_surface=mesh)
    except Exception as e:
        print(f"FAIL: Generation raised exception: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"Generated: {len(verts)} vertices, {len(hexes)} hexes")
    
    if len(hexes) == 0:
        print("FAIL: No hexes generated")
        return
        
    # 5. Check Volume
    # We recycled compute_hex_mesh_volume? No, that was a standalone function.
    # Let's direct import if possible, or reimplement quick check.
    from strategies.conformal_hex_glue import compute_hex_mesh_volume
    
    vol = compute_hex_mesh_volume(verts, hexes)
    print(f"Hex Volume: {vol:.4f} (Expected ~1.0)")
    
    if abs(vol - 1.0) < 0.05:
        print("PASS: Volume is accurate")
    else:
        print("FAIL: Volume mismatch")

if __name__ == "__main__":
    test_global_grid()
