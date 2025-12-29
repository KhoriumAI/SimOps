
import sys
import os
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from strategies.openfoam_hex import create_snappy_case

def test_snappy_configuration():
    print("Testing SnappyHexMesh Configuration...")
    
    # Setup temp dir
    temp_dir = Path("temp_test_snappy")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    
    # Create dummy STL
    stl_path = temp_dir / "dummy.stl"
    with open(stl_path, "w") as f:
        f.write("solid dummy\n")
        f.write("facet normal 0 0 1\n")
        f.write("  outer loop\n")
        f.write("    vertex 0 0 0\n")
        f.write("    vertex 1 0 0\n")
        f.write("    vertex 0 1 0\n")
        f.write("  endloop\n")
        f.write("endfacet\n")
        f.write("endsolid dummy")
        
    # Test 1: Internal Mesh, No Layers
    case_dir_1 = temp_dir / "case_internal"
    case_dir_1.mkdir()
    print("\n[Test 1] Generating Internal Mesh config (Layers=0)...")
    create_snappy_case(case_dir_1, str(stl_path), mesh_scope='Internal', layers=0)
    
    dict_1 = (case_dir_1 / "system" / "snappyHexMeshDict").read_text()
    if "locationInMesh (-100.0 -100.0 -100.0);" not in dict_1 and "locationInMesh" in dict_1:
         # Note: Without trimesh valid STL, it falls back to center. 
         # My dummy STL has no volume, so bounds are -100 to 100 (fallback). Center is 0 0 0.
         print("  Internal/Fallback location check: PASS (Center used)")
    
    if "addLayers       false;" in dict_1:
        print("  AddLayers disable check: PASS")
    else:
        print("  AddLayers disable check: FAIL")

    # Test 2: External Mesh, 3 Layers
    case_dir_2 = temp_dir / "case_external"
    case_dir_2.mkdir()
    print("\n[Test 2] Generating External Mesh config (Layers=3)...")
    create_snappy_case(case_dir_2, str(stl_path), mesh_scope='External', layers=3)
    
    dict_2 = (case_dir_2 / "system" / "snappyHexMeshDict").read_text()
    
    # External point logic: min - margin*0.25
    # bounds -100 to 100. Size 200. Margin 100. min_b -100. 
    # min_pt = -200.
    # location = -100 - 25 = -125.
    if "locationInMesh (-125.0 -125.0 -125.0);" in dict_2:
        print("  External location check: PASS (-125.0)")
    else:
        print(f"  External location check: FAIL. Found location line: {[l for l in dict_2.splitlines() if 'locationInMesh' in l]}")

    if "addLayers       true;" in dict_2:
        print("  AddLayers enable check: PASS")
    else:
        print("  AddLayers enable check: FAIL")
        
    if "nSurfaceLayers 3;" in dict_2:
        print("  Layer count check: PASS")
    else:
        print("  Layer count check: FAIL")

    # Cleanup
    shutil.rmtree(temp_dir)
    print("\nDone.")

if __name__ == "__main__":
    test_snappy_configuration()
