
import sys
import os
from pathlib import Path

# Add apps/cli to path
sys.path.append(str(Path(__file__).parent / "apps" / "cli"))

from mesh_worker_subprocess import generate_hex_testing_visualization

def test_hex_test_integration():
    cad_file = "C:/Users/Owner/Downloads/MeshTest/cad_files/Cube.step"
    if not Path(cad_file).exists():
        print(f"CAD file not found: {cad_file}")
        # Try to find any step file
        cad_files = list(Path("C:/Users/Owner/Downloads/MeshTest/cad_files").glob("*.step"))
        if cad_files:
            cad_file = str(cad_files[0])
            print(f"Using alternative CAD file: {cad_file}")
        else:
            print("No STEP files found.")
            return

    print(f"Testing Hex Testing Integration on {cad_file}...")
    
    result = generate_hex_testing_visualization(cad_file, quality_params={'save_stl': True})
    
    print("\nResult:")
    print(f"Success: {result['success']}")
    print(f"Message: {result.get('message')}")
    print(f"Component Files: {len(result.get('component_files', []))}")
    
    for f in result.get('component_files', []):
        print(f"  - {Path(f).name}")

if __name__ == "__main__":
    test_hex_test_integration()
