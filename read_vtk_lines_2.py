path = r"C:\Users\markm\Downloads\MeshPackageLean\venv\Lib\site-packages\vtkmodules\__init__.py"
with open(path, 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if 20 <= i <= 45:
        print(f"{i+1}: {repr(line)}")
