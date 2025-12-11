import numpy as np
import trimesh
import tempfile
import os
from strategies.conformal_hex_glue import generate_conformal_hex_mesh

def test_coarse_cube():
    print("=== Verifying Coarse STL Fix ===")
    
    # 1. Create a simple Box (coarse, 8 vertices)
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    print(f"Created Box: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # 2. Save to temp STL
    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp:
        mesh.export(tmp.name)
        stl_path = tmp.name
    
    print(f"Saved temp STL to: {stl_path}")
    
    try:
        # 3. Generate Hex Mesh (Mock CoACD input - just the box vertices/faces)
        # We need to simulate CoACD input parts
        # CoACD returns list of (verts, faces)
        coacd_parts = [(mesh.vertices, mesh.faces)]
        
        # Run generator
        print("Running generate_conformal_hex_mesh...")
        result = generate_conformal_hex_mesh(
            coacd_parts=coacd_parts,
            divisions=4,
            epsilon=0.01,
            reference_stl=stl_path,
            verbose=True
        )
        
        # 4. Check Results
        if not result['success']:
            print("FAILED: Generation failed")
            print(result.get('error'))
            return

        vol_data = result.get('volume', {})
        diff_pct = vol_data.get('volume_diff_pct', 100.0)
        hex_vol = vol_data.get('hex_volume', 0.0)
        stl_vol = vol_data.get('stl_volume', 0.0)
        
        print(f"Hex Volume: {hex_vol:.4f}")
        print(f"STL Volume: {stl_vol:.4f}")
        print(f"Difference: {diff_pct:.2f}%")
        
        if diff_pct < 5.0:
            print("PASS: Volume difference is minimal (< 5%)")
        else:
            print("FAIL: Volume difference is too high (> 5%)")
            
        # Check if densification happened (from logs)
        
    finally:
        if os.path.exists(stl_path):
            os.remove(stl_path)

if __name__ == "__main__":
    test_coarse_cube()
