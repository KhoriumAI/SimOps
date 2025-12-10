
import pyvista as pv
from pathlib import Path
import sys

# Path to the generated mesh
mesh_path = Path("C:/Users/Owner/Downloads/MeshPackageLean/apps/cli/generated_meshes/model_openfoam_hex.vtk")

print(f"Checking: {mesh_path}")
if not mesh_path.exists():
    print("ERROR: File does not exist!")
    sys.exit(1)

try:
    print(f"File size: {mesh_path.stat().st_size} bytes")
    
    # Try reading as text to see header
    with open(mesh_path, 'r') as f:
        header = [next(f) for _ in range(5)]
    print("\nFile Header:")
    print("".join(header))
    
    # Try reading with PyVista
    print("\nCreating PyVista Reader...")
    mesh = pv.read(str(mesh_path))
    print(f"Successfully loaded mesh!")
    print(f"Points: {mesh.n_points}")
    print(f"Cells: {mesh.n_cells}")
    print(f"Bounds: {mesh.bounds}")
    
    if mesh.n_points == 0:
        print("ERROR: Mesh has 0 points!")
    else:
        print("SUCCESS: Mesh appears valid.")
        
except Exception as e:
    print(f"ERROR loading mesh: {e}")
    import traceback
    traceback.print_exc()
