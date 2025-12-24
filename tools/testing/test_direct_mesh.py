#!/usr/bin/env python3
"""
Direct test - bypass all the framework to see if gmsh itself works
"""
import gmsh
import sys

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)

# Load
gmsh.open("CAD_files/Cylinder.step")

# Set mesh sizes DIRECTLY
print("\n" + "="*60)
print("SETTING MESH SIZES")
print("="*60)
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.01)  # 10mm in meters

# Verify
cl_min = gmsh.option.getNumber("Mesh.CharacteristicLengthMin")
cl_max = gmsh.option.getNumber("Mesh.CharacteristicLengthMax")
print(f"Verified: cl_min = {cl_min*1000:.2f}mm, cl_max = {cl_max*1000:.2f}mm")

# Disable curvature override
gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
print("Disabled curvature-based sizing")

# Set algorithm
gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
print("Algorithm: Frontal-Delaunay")

print("\n" + "="*60)
print("GENERATING MESH")
print("="*60)

# Generate
gmsh.model.mesh.generate(3)

# Count
nodes = gmsh.model.mesh.getNodes()
elem_2d = gmsh.model.mesh.getElements(2)
elem_3d = gmsh.model.mesh.getElements(3)

print(f"\nResults:")
print(f"  Nodes: {len(nodes[0])}")
print(f"  2D elements: {sum(len(tags) for tags in elem_2d[1])}")
print(f"  3D elements: {sum(len(tags) for tags in elem_3d[1])}")

gmsh.write("test_direct.msh")
print("\nWrote: test_direct.msh")

gmsh.finalize()
