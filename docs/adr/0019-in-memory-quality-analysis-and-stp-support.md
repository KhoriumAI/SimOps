# ADR 0019: In-Memory Meshing Quality Analysis and .stp Extension Support

*Status:* Accepted
*Date:* 2026-01-08
*Tags:* #meshing, #gmsh, #performance, #bug-fix, #cfd-quality

## 1. Context & Problem Statement
The meshing pipeline was failing when processing STEP files with the `.stp` extension, particularly in the "Fast Tet" (HXT) strategy used during batch uploads. 

* *The Constraint:* The `mesh_worker_subprocess.py` was crashing due to a `NameError` (missing `gmsh` import) and a `RuntimeError` caused by attempting to re-initialize and dual-finalize the Gmsh library during the CFD quality analysis phase.
* *The Goal:* Support both `.step` and `.stp` extensions seamlessly and ensure the meshing worker is robust against library re-initialization crashes.

## 2. Technical Decision
We implemented a fix in `apps/cli/mesh_worker_subprocess.py` to stabilize the meshing pipeline.

* *Mechanism:* 
    1. **Global Import:** Added `import gmsh` to the global scope of the worker script.
    2. **In-Memory Analysis:** Switched from `analyze_mesh_file()` (which re-opens the file in a new Gmsh session) to `analyze_current_mesh()` (which uses the active in-memory session).
    3. **Resilient Finalization:** Wrapped `gmsh.finalize()` in a try-except block to gracefully handle cases where the session might have already been closed by a sub-module.
* *Dependencies:* Requires `core.cfd_quality.CFDQualityAnalyzer` to support the `analyze_current_mesh` method.

## 3. Mathematical & Physical Implications
* *Conservation:* No change to mesh data; this only affects the diagnostic/quality gathering phase.
* *Stability:* Significantly improves software stability by preventing SIGSEGV/Runtime errors during the final stage of mesh generation.

## 4. Performance Trade-offs
* *Compute Cost:* **Reduced Overhead.** By using `analyze_current_mesh()`, we skip the redundant disk I/O and mesh parsing step required to re-read the `.msh` file from disk.
* *Memory Cost:* Minimal; leverages the memory already allocated by Gmsh for the current model.

## 5. Verification Plan
* *Sanity Check:* Run `mesh_worker_subprocess.py` directly with a `.stp` file.
* *Regression:* Verified that both `.step` and `.stp` files generate previews and complete meshing successfully using the "Fast Tet" strategy.
* *Integration:* Confirmed with `scripts/debug/verify_mesh_worker_stp.py` and `scripts/debug/verify_preview_stp.py`.
