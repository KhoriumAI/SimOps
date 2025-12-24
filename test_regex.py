
from pathlib import Path
import re

mesh_path = Path("C:/Users/Owner/Downloads/MeshPackageLean/apps/cli/generated_meshes/model_openfoam_hex.vtu")
content = mesh_path.read_text()

print(f"File content length: {len(content)}")
msg = "Searching..."

p_match = re.search(r'''NumberOfPoints=['"](\d+)['"]''', content)
c_match = re.search(r'''NumberOfCells=['"](\d+)['"]''', content)

nodes = int(p_match.group(1)) if p_match else 0
cells = int(c_match.group(1)) if c_match else 0

print(f"Nodes found: {nodes}")
print(f"Cells found: {cells}")

if nodes > 0 and cells > 0:
    print("REGEX SUCCESS")
else:
    print("REGEX FAILED")
