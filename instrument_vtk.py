import os

path = r"C:\Users\markm\Downloads\MeshPackageLean\venv\Lib\site-packages\vtkmodules\__init__.py"
print(f"Reading {path}")

with open(path, 'r') as f:
    lines = f.readlines()

modified = False
for i, line in enumerate(lines):
    if "for f in Path(base_path).glob" in line:
        print(f"Found glob loop at line {i+1}")
        # Insert print before this line
        lines.insert(i, "    print(f'DEBUG: globbing {base_path} for {vtk_module_name}', flush=True)\n")
        modified = True
        break

if modified:
    print("Writing modified file...")
    with open(path, 'w') as f:
        f.writelines(lines)
    print("Done.")
else:
    print("Could not find glob loop.")
