#!/usr/bin/env python3
"""Check tree mesh orientation to debug rotation issue"""

import vtk
import gmsh

# Load tree mesh
mesh_file = "/Users/animeneko/Downloads/Mesh Animation/Tree3_surface.msh"

print(f"Loading: {mesh_file}")
gmsh.initialize()
gmsh.open(mesh_file)

# Get all nodes
node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
print(f"\nTotal nodes: {len(node_tags)}")

# Reshape coordinates
coords = node_coords.reshape(-1, 3)

# Calculate bounds
min_x, max_x = coords[:, 0].min(), coords[:, 0].max()
min_y, max_y = coords[:, 1].min(), coords[:, 1].max()
min_z, max_z = coords[:, 2].min(), coords[:, 2].max()

width_x = max_x - min_x
width_y = max_y - min_y
width_z = max_z - min_z

print(f"\nBounds:")
print(f"  X: {min_x:.2f} to {max_x:.2f} (width: {width_x:.2f})")
print(f"  Y: {min_y:.2f} to {max_y:.2f} (height: {width_y:.2f})")
print(f"  Z: {min_z:.2f} to {max_z:.2f} (depth: {width_z:.2f})")

print(f"\nDimensions sorted:")
dims = [("X", width_x), ("Y", width_y), ("Z", width_z)]
dims.sort(key=lambda x: x[1], reverse=True)
for axis, width in dims:
    print(f"  {axis}: {width:.2f}")

print(f"\nAnalysis:")
if width_y > width_x and width_y > width_z:
    print("  Tree trunk is along Y-axis (correct - already upright)")
elif width_x > width_y and width_x > width_z:
    print("  Tree trunk is along X-axis (lying on side)")
    print("  Need to rotate 90° around Z-axis to stand upright")
elif width_z > width_y and width_z > width_x:
    print("  Tree trunk is along Z-axis (lying on side)")
    print("  Need to rotate 90° around X-axis to stand upright")

gmsh.finalize()
