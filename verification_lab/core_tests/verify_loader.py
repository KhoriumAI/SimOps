import sys
import os
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Mock viewer
class MockViewer:
    pass

from apps.desktop.gui_app.vtk.mesh_loader import MeshLoader

def test_loader():
    loader = MeshLoader(MockViewer())
    
    # Path to the generated mesh from the user log
    mesh_path = current_dir / "apps" / "cli" / "generated_meshes" / "Cube_conformal_hex.msh"
    
    if not mesh_path.exists():
        print(f"File not found: {mesh_path}")
        # Try finding any .msh file
        msh_files = list((current_dir / "apps" / "cli" / "generated_meshes").glob("*.msh"))
        if msh_files:
            mesh_path = msh_files[0]
            print(f"Using alternative file: {mesh_path}")
        else:
            return

    print(f"Testing parsing of: {mesh_path}")
    try:
        nodes, elements = loader._parse_msh_file(str(mesh_path))
        print(f"SUCCESS: Parsed {len(nodes)} nodes, {len(elements)} elements")
        
        # Check node coords
        first_node = list(nodes.values())[0]
        print(f"Sample node: {first_node}")
        
        # Check elements
        hexes = [e for e in elements if e['type'] == 'hexahedron']
        quads = [e for e in elements if e['type'] == 'quadrilateral']
        print(f"Hex count: {len(hexes)}")
        print(f"Quad count: {len(quads)}")
        
        if len(hexes) > 0 and isinstance(first_node[0], float):
            print("Validation PASSED (Nodes are floats, Elements found)")
            return True
        else:
             print("Validation FAILED (Empty or bad types)")
             return False

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_loader()
