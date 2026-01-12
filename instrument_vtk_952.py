import os

path = r"C:\Users\markm\Downloads\MeshPackageLean\venv\Lib\site-packages\vtkmodules\__init__.py"
print(f"Fixing UnboundLocalError in {path}")

with open(path, 'r') as f:
    lines = f.readlines()

clean_lines = []
for line in lines:
    # Indent lines that are incorrectly outside the if block
    if "resolved_file = f.resolve()" in line:
        clean_lines.append("    " + line)
    elif "if resolved_file.is_file()" in line:
        clean_lines.append("    " + line)
    # The return statement also needs to be indented
    elif "return str(resolved_file)" in line:
        clean_lines.append("    " + line)
    else:
        clean_lines.append(line)

print("Writing fixed file...")
with open(path, 'w') as f:
    f.writelines(clean_lines)
print("Done.")
