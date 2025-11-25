# Polyhedral Mesh Visualization Design

## Overview
This document outlines the plan to extend the `MeshPackageLean` GUI to support **Polyhedral (Dual) Meshes**. The current viewer supports Tetrahedra (4-node) and Hexahedra (8-node), but Polyhedra have an arbitrary number of faces and vertices, requiring `vtkPolyhedron` support.

## 1. Data Structure Requirements

### Current Structure (Limited)
- **Nodes**: `{id: (x, y, z)}`
- **Elements**: `[{'id': 1, 'type': 'tetrahedron', 'nodes': [1, 2, 3, 4]}]`

### New Polyhedral Structure
Polyhedra require explicit face definitions, not just a list of nodes.
- **Nodes**: Same `{id: (x, y, z)}`
- **Elements**:
  ```python
  {
      'id': 100,
      'type': 'polyhedron',
      'faces': [
          [1, 2, 3],      # Face 0 (triangle)
          [1, 4, 2],      # Face 1
          [2, 4, 3],      # Face 2
          [3, 4, 1]       # Face 3
      ]
  }
  ```

## 2. VTK Implementation Strategy

### `vtkPolyhedron`
We must use `vtkUnstructuredGrid` with cell type `vtk.VTK_POLYHEDRON`.

**Construction Logic:**
```python
# VTK requires a specific 1D array format for polyhedra:
# [numFaces, numFace0Pts, id0_0, id0_1..., numFace1Pts, id1_0...]

face_stream = [num_faces]
for face in faces:
    face_stream.append(len(face))
    face_stream.extend(face_node_indices)

ugrid.InsertNextCell(vtk.VTK_POLYHEDRON, len(face_stream), face_stream)
```

### Cross-Section Slicing
The current manual intersection logic (`_intersect_element_with_plane`) is hardcoded for Tets/Hexes.
**Proposed Change:** Use `vtkCutter` for polyhedra. It is robust and handles arbitrary cell types natively.

```python
cutter = vtk.vtkCutter()
cutter.SetInputData(ugrid)
cutter.SetCutFunction(plane)
cutter.Update()
slice_polydata = cutter.GetOutput()
```

## 3. Parsing Strategy (MSH 4.1 / Custom)

The current simple parser in `utils.py` is insufficient. We need a robust parser for polyhedral data.
**Recommendation:** Use `meshio` library if possible, or implement a dedicated parser for the specific output format of the polyhedral strategy.

## 4. Proposed Class Structure

We will create a `PolyhedralVTKViewer` that extends or replaces the current `VTK3DViewer`.

### File: `apps/desktop/gui_app/polyhedral_viewer_prototype.py`

```python
class PolyhedralVTKViewer(VTK3DViewer):
    def load_polyhedral_mesh(self, nodes, polyhedra):
        # Build vtkUnstructuredGrid with VTK_POLYHEDRON
        ...
        
    def _generate_cross_section_mesh(self, origin, normal):
        # Override to use vtkCutter
        ...
```

## 5. Implementation Steps

1.  **Create Prototype**: `polyhedral_viewer_prototype.py` (Done in this task).
2.  **Update Parser**: Modify `utils.py` or add `polyhedral_parser.py` to read the dual mesh format.
3.  **Integrate**: Update `main.py` to use the new viewer capabilities when "Polyhedral" strategy is selected.
