path = r"C:\Users\markm\Downloads\MeshPackageLean\venv\Lib\site-packages\vtkmodules\__init__.py"
print(f"Reading {path}")
with open(path, 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if 25 <= i <= 55:
        print(f"{i+1}: {repr(line)}")
