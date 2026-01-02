# ADR 0008: Progressive CAD Rendering Strategy

*Status:* Proposed
*Date:* 2025-12-31
*Tags:* #ui, #ux, #geometry, #rendering

## 1. Context & Problem Statement
Large CAD files can take significant time (10-60s) to tessellate into a 3D surface mesh. During this time, the user only sees a loading spinner, which can feel like the application has hung (even with progress bars).

* **The Goal:** Provide immediate visual feedback by rendering the wireframe (CAD edges) as soon as the file is synchronized, then overlaying the surface mesh once tessellation completes.

## 2. Technical Decision
We will upgrade the `PreviewWorker` to support a **Two-Stage Emission** flow:

1. **Stage 1 (Wireframe):**
   - After `gmsh.model.occ.synchronize()`, we trigger `gmsh.model.mesh.generate(1)` (1D discretization).
   - Export the 1D mesh to a temporary `.vtk` file (often an `UnstructuredGrid`).
   - Emit a `lines_ready` signal to the GUI with the line mesh.
   - The GUI renders these lines using `vtkDataSetMapper` (which supports unstructured grids).

2. **Stage 2 (Surface):**
   - Proceed with `gmsh.model.mesh.generate(2)` (2D tessellation).
   - Export to `.stl` (standard surface format).
   - Emit the final `finished` signal.
   - The GUI replaces/overlays the lines with the surfaced model.

**Sequence requirement:**
- `STATUS: Reading CAD file` (Before synchronization)
- `LINES_READY_MARKER` (After 1D mesh)
- `STATUS: Tessellating surfaces` (Before 2D mesh)

## 3. Performance Trade-offs
* **Compute Cost:** `generate(1)` is extremely fast compared to `generate(2)`. The overhead is negligible (~100-500ms).
* **Memory Cost:** Storing two temporary meshes briefly.

## 4. Verification Plan
- Load a complex STEP file and verify that lines appear within 1-2 seconds, followed by the surface several seconds later.
- Verify status messages are in the correct order.
