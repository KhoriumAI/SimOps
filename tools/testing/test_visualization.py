#!/usr/bin/env python3
"""
Quick test script to verify mesh visualization works
Tests VTK rendering separately from the GUI
"""

import sys
from pathlib import Path
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel

def parse_msh_file(filepath: str):
    """Quick .msh parser"""
    nodes = {}
    elements = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line == "$Nodes":
            i += 2
            while lines[i].strip() != "$EndNodes":
                parts = lines[i].strip().split()
                if len(parts) == 4:
                    num_nodes = int(parts[3])
                    i += 1
                    node_tags = []
                    for _ in range(num_nodes):
                        node_tags.append(int(lines[i].strip()))
                        i += 1
                    for tag in node_tags:
                        coords = lines[i].strip().split()
                        nodes[tag] = [float(coords[0]), float(coords[1]), float(coords[2])]
                        i += 1
                else:
                    i += 1

        elif line == "$Elements":
            i += 2
            while lines[i].strip() != "$EndElements":
                parts = lines[i].strip().split()
                if len(parts) == 4:
                    element_type = int(parts[2])
                    num_elements = int(parts[3])
                    i += 1

                    for _ in range(num_elements):
                        elem_parts = lines[i].strip().split()
                        elem_nodes = [int(x) for x in elem_parts[1:]]

                        # Only care about tets (4 nodes) and triangles (3 nodes)
                        if element_type == 4 and len(elem_nodes) == 4:  # tetrahedron
                            elements.append({'type': 'tetrahedron', 'nodes': elem_nodes[:4]})
                        elif element_type == 2 and len(elem_nodes) == 3:  # triangle
                            elements.append({'type': 'triangle', 'nodes': elem_nodes[:3]})
                        elif element_type == 11 and len(elem_nodes) == 10:  # 10-node tet
                            elements.append({'type': 'tetrahedron', 'nodes': elem_nodes[:4]})
                        elif element_type == 9 and len(elem_nodes) == 6:  # 6-node triangle
                            elements.append({'type': 'triangle', 'nodes': elem_nodes[:3]})

                        i += 1
                else:
                    i += 1
        else:
            i += 1

    return nodes, elements


class MeshViewer(QMainWindow):
    def __init__(self, mesh_file):
        super().__init__()
        self.mesh_file = mesh_file
        self.setWindowTitle("Mesh Visualization Test")
        self.setGeometry(100, 100, 1200, 800)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Info label
        self.info_label = QLabel("Loading mesh...")
        layout.addWidget(self.info_label)

        # VTK widget
        self.vtk_widget = QVTKRenderWindowInteractor(central)
        layout.addWidget(self.vtk_widget)

        # Reload button
        reload_btn = QPushButton("Reload Mesh")
        reload_btn.clicked.connect(self.load_mesh)
        layout.addWidget(reload_btn)

        # Setup VTK
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.95, 0.95, 0.95)
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)

        # Load mesh
        self.load_mesh()

        self.vtk_widget.Initialize()
        self.vtk_widget.Start()

    def load_mesh(self):
        print("\n" + "=" * 60)
        print("LOADING MESH")
        print("=" * 60)
        print(f"File: {self.mesh_file}")
        print(f"Exists: {Path(self.mesh_file).exists()}")

        try:
            # Clear previous
            self.renderer.RemoveAllViewProps()

            # Parse .msh
            print("Parsing .msh file...")
            nodes, elements = parse_msh_file(self.mesh_file)
            print(f"[OK] Parsed {len(nodes)} nodes, {len(elements)} elements")

            # Create VTK structures
            print("Creating VTK data structures...")
            points = vtk.vtkPoints()
            cells = vtk.vtkCellArray()

            node_map = {}
            for idx, (node_id, coords) in enumerate(nodes.items()):
                points.InsertNextPoint(coords)
                node_map[node_id] = idx

            print(f"[OK] Added {points.GetNumberOfPoints()} points")

            tet_count = 0
            tri_count = 0

            for element in elements:
                if element['type'] == 'tetrahedron':
                    tet = vtk.vtkTetra()
                    for i, node_id in enumerate(element['nodes']):
                        tet.GetPointIds().SetId(i, node_map[node_id])
                    cells.InsertNextCell(tet)
                    tet_count += 1
                elif element['type'] == 'triangle':
                    tri = vtk.vtkTriangle()
                    for i, node_id in enumerate(element['nodes']):
                        tri.GetPointIds().SetId(i, node_map[node_id])
                    cells.InsertNextCell(tri)
                    tri_count += 1

            print(f"[OK] Created {tet_count} tets, {tri_count} triangles")

            # Create mesh
            mesh = vtk.vtkUnstructuredGrid()
            mesh.SetPoints(points)
            mesh.SetCells(vtk.VTK_TETRA if tet_count > 0 else vtk.VTK_TRIANGLE, cells)

            print(f"[OK] UnstructuredGrid: {mesh.GetNumberOfCells()} cells")

            # Extract surface
            surface_filter = vtk.vtkGeometryFilter()
            surface_filter.SetInputData(mesh)
            surface_filter.Update()

            poly_output = surface_filter.GetOutput()
            print(f"[OK] Geometry filter (outer surface only): {poly_output.GetNumberOfCells()} surface cells")

            # Create mapper and actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(surface_filter.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.2, 0.7, 0.4)
            actor.GetProperty().EdgeVisibilityOn()
            actor.GetProperty().SetEdgeColor(0.1, 0.1, 0.1)
            actor.GetProperty().SetLineWidth(1)

            print("[OK] Actor created")

            # Add to renderer
            self.renderer.AddActor(actor)
            print(f"[OK] Added actor to renderer (total actors: {self.renderer.GetActors().GetNumberOfItems()})")

            # Reset camera and render
            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()

            print("[OK] RENDER COMPLETE!")
            print("=" * 60)

            self.info_label.setText(
                f"[OK] Mesh loaded successfully\n"
                f"Nodes: {len(nodes):,} * Elements: {len(elements):,}\n"
                f"Tets: {tet_count:,} * Triangles: {tri_count:,}"
            )

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"\n[X] {error_msg}")
            import traceback
            traceback.print_exc()
            print("=" * 60)
            self.info_label.setText(error_msg)


def main():
    # Find a mesh file to test
    project_root = Path(__file__).parent
    mesh_folder = project_root / "generated_meshes"

    if not mesh_folder.exists():
        print("Error: generated_meshes/ folder not found")
        print("Generate a mesh first!")
        sys.exit(1)

    # Find first .msh file
    mesh_files = list(mesh_folder.glob("*.msh"))
    if not mesh_files:
        print("Error: No .msh files found in generated_meshes/")
        print("Generate a mesh first!")
        sys.exit(1)

    mesh_file = str(mesh_files[0])
    print(f"Testing with: {mesh_file}")

    app = QApplication(sys.argv)
    viewer = MeshViewer(mesh_file)
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
