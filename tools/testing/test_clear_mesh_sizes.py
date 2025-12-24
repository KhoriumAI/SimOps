#!/usr/bin/env python3
"""Test clearing mesh sizes from STEP import"""
import gmsh
import time

def log(msg):
    print(msg, flush=True)

log("Test 1: WITHOUT clearing mesh sizes from STEP")
log("="*60)
gmsh.initialize()
gmsh.open("CAD_files/Cube.step")
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.1)  # 100mm
gmsh.model.mesh.generate(1)
nodes1 = gmsh.model.mesh.getNodes()
log(f"Nodes: {len(nodes1[0])}")
gmsh.finalize()

log("\nTest 2: WITH clearing mesh sizes from STEP")
log("="*60)
gmsh.initialize()
gmsh.open("CAD_files/Cube.step")

# CLEAR all mesh size prescriptions from geometry
log("Clearing mesh sizes on all entities...")
for dim in [0, 1, 2, 3]:
    entities = gmsh.model.getEntities(dim)
    for entity in entities:
        try:
            gmsh.model.mesh.setSize(entity, 0)  # Clear
        except:
            pass

# Now set global size
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.1)  # 100mm
log("Set CharacteristicLengthMax = 100mm")

gmsh.model.mesh.generate(1)
nodes2 = gmsh.model.mesh.getNodes()
log(f"Nodes: {len(nodes2[0])}")
gmsh.finalize()

log("\n" + "="*60)
log(f"Result: {len(nodes1[0])} nodes WITHOUT clear, {len(nodes2[0])} nodes WITH clear")
if len(nodes2[0]) < len(nodes1[0]):
    log("[OK] Clearing mesh sizes HELPS!")
else:
    log("[X] Clearing mesh sizes doesn't help")
