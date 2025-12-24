#!/usr/bin/env python3
"""Test mesh field with 2D generation"""
import gmsh
import time

def log(msg):
    print(msg, flush=True)

log("Test: Mesh field with 2D generation on Cube")
log("="*60)
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)
gmsh.open("CAD_files/Cube.step")

# Create uniform background mesh field
log("Creating uniform background mesh field...")
field_tag = gmsh.model.mesh.field.add("MathEval")
gmsh.model.mesh.field.setString(field_tag, "F", "100")  # 100mm everywhere
gmsh.model.mesh.field.setAsBackgroundMesh(field_tag)

# Disable other mesh size sources
gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
log("Disabled other mesh size sources")

# Set algorithm
gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
log("Algorithm: Frontal-Delaunay")

# Generate 1D
log("\nGenerating 1D mesh...")
start = time.time()
gmsh.model.mesh.generate(1)
elapsed = time.time() - start
nodes = gmsh.model.mesh.getNodes()
log(f"[OK] 1D done in {elapsed:.3f}s, {len(nodes[0])} nodes")

# Generate 2D
log("\nGenerating 2D mesh...")
start = time.time()
gmsh.model.mesh.generate(2)
elapsed = time.time() - start
elem_2d = gmsh.model.mesh.getElements(2)
num_2d = sum(len(tags) for tags in elem_2d[1])
log(f"[OK] 2D done in {elapsed:.3f}s, {num_2d} elements")

gmsh.write("test_field_2d.msh")
log("\nWrote: test_field_2d.msh")

gmsh.finalize()

log("\n" + "="*60)
log("[OK] SUCCESS - 2D meshing works with mesh field!")
