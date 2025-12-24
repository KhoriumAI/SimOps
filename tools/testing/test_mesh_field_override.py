#!/usr/bin/env python3
"""Test using a mesh field to override STEP entity prescriptions"""
import gmsh

def log(msg):
    print(msg, flush=True)

log("Test: Using mesh field to override STEP entity sizes")
log("="*60)
gmsh.initialize()
gmsh.open("CAD_files/Cube.step")

# Create a uniform background mesh field
log("Creating uniform background mesh field...")
field_tag = gmsh.model.mesh.field.add("MathEval")
gmsh.model.mesh.field.setString(field_tag, "F", "100")  # 100mm everywhere
gmsh.model.mesh.field.setAsBackgroundMesh(field_tag)
log(f"Set field {field_tag} as background mesh with 100mm size")

# Disable other mesh size sources
gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
log("Disabled all other mesh size sources")

# Generate 1D
log("\nGenerating 1D mesh...")
gmsh.model.mesh.generate(1)
nodes = gmsh.model.mesh.getNodes()
log(f"Nodes: {len(nodes[0])}")

gmsh.finalize()

log("\n" + "="*60)
if len(nodes[0]) < 1000:
    log("[OK] Mesh field WORKS! Reasonable node count.")
else:
    log("[X] Mesh field doesn't help - still too many nodes")
