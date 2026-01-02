
import numpy as np
import trimesh
from pathlib import Path
import sys
import shutil

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from strategies.openfoam_hex import create_snappy_case

def create_disjoint_stl(filename):
    """Creates an STL with two disjoint cubes."""
    
    # Cube 1: centered at (0,0,0), size 10
    cube1 = trimesh.creation.box(extents=(10, 10, 10))
    cube1.apply_translation([0, 0, 0])
    
    # Cube 2: centered at (50,0,0), size 10
    cube2 = trimesh.creation.box(extents=(10, 10, 10))
    cube2.apply_translation([50, 0, 0])
    
    # Combine (but keep disjoint)
    mesh = trimesh.util.concatenate([cube1, cube2])
    
    mesh.export(filename)
    print(f"Created disjoint STL at {filename} with 2 connected components.")
    return filename

def verify_multi_region():
    print("\nStarting Multi-Region Verification...")
    
    stl_path = Path("temp_disjoint.stl")
    create_disjoint_stl(stl_path)
    
    output_dir = Path("temp_case_multi")
    if output_dir.exists(): shutil.rmtree(output_dir)
    output_dir.mkdir()
    
    try:
        # Run configuration generation
        # This will call create_snappy_case -> generate the dict
        # We need to spy on the generated dict
        
        # We Mock the actual meshing call to avoid needing Docker/OpenFOAM
        # But we need to inspect the generated 'system/snappyHexMeshDict'
        
        # We can call create_snappy_case directly if we mock the runner? 
        # Actually create_snappy_case is what we modified.
        # But it returns the path to the case directory.
        
        # NOTE: create_snappy_case is an internal helper called by generate_openfoam_hex_mesh
        # Let's import it directly (done above).
        
        print("Running create_snappy_case...")
        
        create_snappy_case(
            case_dir=output_dir.resolve(),
            stl_path=str(stl_path.resolve()),
            cell_size=2.0,
            mesh_scope='Internal',
            layers=0
        )
        
        # Check the dict
        dict_path = output_dir / "system" / "snappyHexMeshDict"
        if not dict_path.exists():
            print("FAIL: snappyHexMeshDict not created")
            return
            
        content = dict_path.read_text()
        
        print("\nChecking snappyHexMeshDict content...")
        
        if "locationsInMesh" in content:
            print("PASS: Found 'locationsInMesh' keyword.")
        else:
            print("FAIL: 'locationsInMesh' keyword missing (expected for multi-body).")
            
        if "volume_0" in content and "volume_1" in content:
             print("PASS: Found entries for volume_0 and volume_1")
        else:
             print("FAIL: Missing volume definitions for bodies.")
             print("Content snippets:")
             print(content[content.find("location"):content.find("allowFreeStanding")])
             
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if stl_path.exists(): stl_path.unlink()
        if output_dir.exists(): shutil.rmtree(output_dir)

if __name__ == "__main__":
    verify_multi_region()
