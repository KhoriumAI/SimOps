#!/usr/bin/env python3
"""
Test VTK quality coloring on cylinder mesh
"""
import json
import vtk
from pathlib import Path

mesh_file = "cad_files/Cylinder_surface.msh"
quality_file = "cad_files/Cylinder_surface.quality.json"

# Load quality data
print(f"Loading quality data from {quality_file}")
with open(quality_file) as f:
    quality_data = json.load(f)

per_element_quality = quality_data['per_element_quality']
threshold = quality_data['quality_threshold_10']

print(f"Quality threshold (10th percentile): {threshold:.3f}")
print(f"Quality data entries: {len(per_element_quality)}")

# Parse mesh file (simple parser)
def parse_msh_file(filepath):
    nodes = {}
    elements = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    in_nodes = False
    in_elements = False

    for i, line in enumerate(lines):
        line = line.strip()

        if line == "$Nodes":
            in_nodes = True
            continue
        elif line == "$EndNodes":
            in_nodes = False
            continue
        elif line == "$Elements":
            in_elements = True
            continue
        elif line == "$EndElements":
            in_elements = False
            continue

        if in_nodes:
            parts = line.split()
            if len(parts) == 4:
                try:
                    node_id = int(parts[0])
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    nodes[node_id] = (x, y, z)
                except:
                    pass

        if in_elements:
            parts = line.split()
            if len(parts) >= 7:
                try:
                    elem_id = int(parts[0])
                    elem_type = int(parts[1])

                    # Triangle (type 2)
                    if elem_type == 2:
                        num_tags = int(parts[2])
                        node_start = 3 + num_tags
                        element_nodes = [int(parts[j]) for j in range(node_start, len(parts))]
                        elements.append({
                            'id': elem_id,
                            'type': 'triangle',
                            'nodes': element_nodes
                        })
                except:
                    pass

    return nodes, elements

print(f"\nParsing mesh file: {mesh_file}")
nodes, elements = parse_msh_file(mesh_file)
print(f"Nodes: {len(nodes)}")
print(f"Elements: {len(elements)}")

# Create VTK structures
points = vtk.vtkPoints()
cells = vtk.vtkCellArray()

node_map = {}
for idx, (node_id, coords) in enumerate(nodes.items()):
    points.InsertNextPoint(coords)
    node_map[node_id] = idx

# Add triangles
for element in elements:
    if element['type'] == 'triangle':
        tri = vtk.vtkTriangle()
        for i, node_id in enumerate(element['nodes']):
            tri.GetPointIds().SetId(i, node_map[node_id])
        cells.InsertNextCell(tri)

poly_data = vtk.vtkPolyData()
poly_data.SetPoints(points)
poly_data.SetPolys(cells)

print(f"\nVTK PolyData created: {poly_data.GetNumberOfCells()} cells")

# Add colors
colors = vtk.vtkUnsignedCharArray()
colors.SetNumberOfComponents(3)
colors.SetName("Colors")

color_counts = {'red': 0, 'orange': 0, 'yellow': 0, 'yellow_green': 0, 'green': 0, 'default': 0}

for element in elements:
    if element['type'] == 'triangle':
        elem_id = element['id']
        quality = per_element_quality.get(str(elem_id), None)

        if quality is None:
            colors.InsertNextTuple3(51, 179, 102)  # Default green
            color_counts['default'] += 1
        elif quality <= threshold:
            colors.InsertNextTuple3(255, 0, 0)  # RED
            color_counts['red'] += 1
        elif quality < 0.3:
            colors.InsertNextTuple3(255, 128, 0)  # Orange
            color_counts['orange'] += 1
        elif quality < 0.5:
            colors.InsertNextTuple3(255, 255, 0)  # Yellow
            color_counts['yellow'] += 1
        elif quality < 0.7:
            colors.InsertNextTuple3(128, 255, 0)  # Yellow-green
            color_counts['yellow_green'] += 1
        else:
            colors.InsertNextTuple3(51, 179, 102)  # Green
            color_counts['green'] += 1

# Attach colors to polydata
poly_data.GetCellData().SetScalars(colors)

print(f"\nColors applied:")
print(f"  Red (worst 10%): {color_counts['red']}")
print(f"  Orange: {color_counts['orange']}")
print(f"  Yellow: {color_counts['yellow']}")
print(f"  Yellow-green: {color_counts['yellow_green']}")
print(f"  Green: {color_counts['green']}")
print(f"  Default: {color_counts['default']}")

# Create mapper and actor
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputData(poly_data)
mapper.SetScalarModeToUseCellData()
mapper.ScalarVisibilityOn()

actor = vtk.vtkActor()
actor.SetMapper(mapper)

# Create renderer
renderer = vtk.vtkRenderer()
renderer.SetBackground(0.1, 0.1, 0.1)
renderer.AddActor(actor)

# Create render window
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window.SetSize(800, 600)

# Create interactor
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)

# Set camera
renderer.ResetCamera()

print("\n[OK] Rendering window with quality-colored mesh")
print("  RED triangles = worst 10% quality")
print("  Rotate with left mouse, zoom with right mouse")
print("  Close window to exit")

# Start
interactor.Initialize()
render_window.Render()
interactor.Start()
