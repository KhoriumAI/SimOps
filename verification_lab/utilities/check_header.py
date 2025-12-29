
from pathlib import Path

mesh_path = Path("C:/Users/Owner/Downloads/MeshPackageLean/apps/cli/generated_meshes/model_openfoam_hex.vtu")

if mesh_path.exists():
    print(f"Reading header of {mesh_path.name}...")
    try:
        with open(mesh_path, 'r') as f:
            for i in range(10):
                line = f.readline()
                if not line: break
                print(f"{i+1}: {line.strip()}")
    except Exception as e:
        print(f"Error reading file: {e}")
else:
    print("File not found.")
