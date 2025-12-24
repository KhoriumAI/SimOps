#!/usr/bin/env python3
"""
Test VTK color visualization directly
"""
import vtk
import sys
from pathlib import Path

# Use the existing cube mesh
mesh_file = Path("generated_meshes/Cube_mesh.msh")

if not mesh_file.exists():
    print(f"Mesh file not found: {mesh_file}")
    sys.exit(1)

# Parse mesh file
from parse_gmsh_mesh import parse_gmsh_mesh
nodes, elements = parse_gmsh_mesh(str(mesh_file))

print(f"Loaded {len(nodes)} nodes, {len(elements)} elements")

# Extract per-element quality
from extract_element_quality import extract_per_element_quality
quality_data = extract_per_element_quality(str(mesh_file))
per_elem_quality = quality_data.get('per_element_quality', {})
threshold = quality_data.get('sicn_10_percentile', 0.3)

print(f"Quality data: {len(per_elem_quality)} triangles")
print(f"Threshold: {threshold:.3f}")

# Create VTK structures
points = vtk.vtkPoints()
node_map = {}
for i, (node_id, coords) in enumerate(nodes.items()):
    points.InsertNextPoint(coords)
    node_map[node_id] = i

cells = vtk.vtkCellArray()
colors = vtk.vtkUnsignedCharArray()
colors.SetNumberOfComponents(3)
colors.SetName("Colors")

color_counts = {'red': 0, 'orange': 0, 'yellow': 0, 'yellow_green': 0, 'green': 0}

# Add triangles with colors
for element in elements:
    if element['type'] == 'triangle':
        tri = vtk.vtkTriangle()
        for i, node_id in enumerate(element['nodes'][:3]):
            tri.GetPointIds().SetId(i, node_map[node_id])
        cells.InsertNextCell(tri)

        # Assign color based on quality
        elem_id = element['id']
        quality = per_elem_quality.get(elem_id, 1.0)

        if quality <= threshold:
            colors.InsertNextTuple3(255, 0, 0)  # RED
            color_counts['red'] += 1
        elif quality < 0.3:
            colors.InsertNextTuple3(255, 128, 0)  # ORANGE
            color_counts['orange'] += 1
        elif quality < 0.5:
            colors.InsertNextTuple3(255, 255, 0)  # YELLOW
            color_counts['yellow'] += 1
        elif quality < 0.7:
            colors.InsertNextTuple3(128, 255, 0)  # YELLOW-GREEN
            color_counts['yellow_green'] += 1
        else:
            colors.InsertNextTuple3(51, 179, 102)  # GREEN
            color_counts['green'] += 1

print(f"Color distribution: {color_counts}")

# Create polydata
poly_data = vtk.vtkPolyData()
poly_data.SetPoints(points)
poly_data.SetPolys(cells)
poly_data.GetCellData().SetScalars(colors)

print(f"PolyData has {poly_data.GetNumberOfCells()} cells")
print(f"PolyData has scalars: {poly_data.GetCellData().GetScalars() is not None}")

# Create mapper with scalar coloring enabled
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputData(poly_data)
mapper.SetScalarModeToUseCellData()
mapper.ScalarVisibilityOn()

print(f"Mapper scalar visibility: {mapper.GetScalarVisibility()}")
print(f"Mapper scalar mode: {mapper.GetScalarMode()}")

# Create actor
actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().EdgeVisibilityOn()
actor.GetProperty().SetEdgeColor(0, 0, 0)

# Create renderer and window
renderer = vtk.vtkRenderer()
renderer.AddActor(actor)
renderer.SetBackground(0.1, 0.1, 0.1)
renderer.ResetCamera()

render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window.SetSize(800, 600)
render_window.SetWindowName("VTK Color Test - Should show colored triangles")

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)

print("\n" + "="*60)
print("LAUNCHING VTK VIEWER")
print("="*60)
print("Expected: Mesh with colored triangles (red for worst quality)")
print(f"Red triangles: {color_counts['red']}")
print(f"Green triangles: {color_counts['green']}")
print("\nPress 'q' to quit")
print("="*60)

render_window.Render()
interactor.Start()
