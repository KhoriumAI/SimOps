
import pyvista as pv
from pathlib import Path
import sys

# Path from the user's log
mesh_path = Path("C:/Users/Owner/Downloads/MeshPackageLean/apps/cli/generated_meshes/model_openfoam_hex.vtu")

print(f"Checking: {mesh_path}")
if not mesh_path.exists():
    print("ERROR: File does not exist!")
    # Check if .vtk exists instead
    vtk_path = mesh_path.with_suffix(".vtk")
    if vtk_path.exists():
        print(f"Found .vtk instead: {vtk_path}")
        mesh_path = vtk_path
    else:
        sys.exit(1)

try:
    print(f"File size: {mesh_path.stat().st_size} bytes")
    
    # Try reading with PyVista
    print("\nLoading with PyVista...")
    mesh = pv.read(str(mesh_path))
    print(f"Type: {type(mesh)}")
    print(f"Points: {mesh.n_points}")
    print(f"Cells: {mesh.n_cells}")
    print(f"Bounds: {mesh.bounds}")
    
    if mesh.n_points > 0:
        print("\nExtracting surface...")
        surf = mesh.extract_surface()
        print(f"Surface Points: {surf.n_points}")
        print(f"Surface Cells: {surf.n_cells}")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
