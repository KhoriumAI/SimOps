from pathlib import Path
import os
import sys

# Adjust path to match user's venv
venv_path = Path(r"C:\Users\markm\Downloads\MeshPackageLean\venv\Lib\site-packages")
vtkmodules = venv_path / "vtkmodules"
base_path = vtkmodules

print(f"Testing glob in: {base_path}")
print(f"Exists: {base_path.exists()}")

if not base_path.exists():
    sys.exit(1)

vtk_module_name = "vtkCommonCore"
pattern = f"{vtk_module_name}[.-]*"
print(f"Pattern: {pattern}")

print("Starting glob...")
try:
    for f in base_path.glob(pattern):
        print(f"Match: {f.name}")
    print("Glob finished successfully.")
except Exception as e:
    print(f"Glob failed: {e}")
