#!/usr/bin/env python3
"""Create a fresh cylinder directly in gmsh and mesh it"""
import gmsh

gmsh.initialize()
gmsh.model.add("cylinder")

# Create cylinder: radius=50mm, height=50mm
gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, 50, 50)  # x,y,z, dx,dy,dz, radius
gmsh.model.occ.synchronize()

print("Created cylinder: r=50mm, h=50mm")

# Set mesh sizes
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 10)  # 10mm
print(f"Set mesh size: max=10mm")

# Generate
print("\nGenerating mesh...")
import time
start = time.time()
gmsh.model.mesh.generate(3)
elapsed = time.time() - start

nodes = gmsh.model.mesh.getNodes()
elem_3d = gmsh.model.mesh.getElements(3)

print(f"\n[OK] SUCCESS in {elapsed:.2f}s")
print(f"  Nodes: {len(nodes[0])}")
print(f"  3D elements: {sum(len(tags) for tags in elem_3d[1])}")

gmsh.write("fresh_cylinder.msh")
print("\nWrote: fresh_cylinder.msh")

gmsh.finalize()
