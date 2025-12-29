import gmsh
import sys

# Check the first mesh file to see if it has volume elements
mesh_file = "temp_stls/volume_meshes/vol_1.msh"

gmsh.initialize()
gmsh.open(mesh_file)

# Get element types
element_types = gmsh.model.mesh.getElementTypes()

print(f"Inspecting: {mesh_file}")
print(f"Element types present: {element_types}")
print()

for elem_type in element_types:
    name = gmsh.model.mesh.getElementProperties(elem_type)[0]
    dim = gmsh.model.mesh.getElementProperties(elem_type)[1]
    count = len(gmsh.model.mesh.getElementsByType(elem_type)[0])
    print(f"  Type {elem_type}: {name} (dim={dim}) - count: {count}")

# Check if we have any 3D elements (tetrahedra, hexahedra, etc.)
has_volume = any(gmsh.model.mesh.getElementProperties(et)[1] == 3 for et in element_types)

print()
if has_volume:
    print("[OK] This is a VOLUME mesh (has 3D elements)")
else:
    print("[X] This is a SURFACE mesh (no 3D elements)")

gmsh.finalize()
