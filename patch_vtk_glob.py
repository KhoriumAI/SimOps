import os

path = r"C:\Users\markm\Downloads\MeshPackageLean\venv\Lib\site-packages\vtkmodules\__init__.py"
print(f"Patching {path}")

with open(path, 'r') as f:
    lines = f.readlines()

new_lines = []
found = False
for line in lines:
    if "for f in Path(base_path).glob" in line:
        found = True
        indent = line[:line.find("for")]
        print(f"Replacing glob loop at line...")
        # Add import os if not present? It's safe to add inside function or top.
        # But 'import os' inside loop is fine.
        new_code = [
            f"{indent}# PATCHED BY ANTIGRAVITY\n",
            f"{indent}import os\n",
            f"{indent}print(f'DEBUG: Scanning {{base_path}} for {{vtk_module_name}}', flush=True)\n",
            f"{indent}try:\n",
            f"{indent}    filenames = os.listdir(base_path)\n",
            f"{indent}except Exception as e:\n",
            f"{indent}    print(f'DEBUG: listdir failed: {{e}}')\n",
            f"{indent}    filenames = []\n",
            f"{indent}for filename in filenames:\n",
            f"{indent}    if filename.startswith(vtk_module_name) and (len(filename) > len(vtk_module_name)) and (filename[len(vtk_module_name)] in '.-'):\n",
            f"{indent}        f = Path(base_path) / filename\n"
        ]
        new_lines.extend(new_code)
    else:
        new_lines.append(line)

if found:
    print("Writing patched file...")
    with open(path, 'w') as f:
        f.writelines(new_lines)
    print("Done.")
else:
    print("Could not find glob loop.")
