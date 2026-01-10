# ADR 0018: VTK Import Optimization and Geometry Guardian Behavior

*Status:* Accepted
*Date:* 2026-01-08
*Tags:* #performance, #geometry, #guardian, #optimization

## 1. Context & Problem Statement
The meshing process experienced significant startup delays (15+ seconds) due to VTK's internal module scanning during import. Additionally, the Geometry Guardian was halting the meshing process when encountering "TERMINAL" (unrepairable) geometry, even though the underlying meshing engine (Gmsh) could often successfully mesh such files.

* *The Constraint:* VTK's lazy-loading mechanism scans the filesystem for 150+ submodules at import time, causing unacceptable latency in worker processes.
* *The Goal:* Reduce worker startup time to <1s and prevent the Guardian from being a hard blocker for non-watertight geometry.

## 2. Technical Decision
We implemented a two-pronged optimization:

* *Mechanism 1 (Lazy Loading):* Replaced all enthusiastic top-level imports of `vtk` and `pyvista` with local, lazy imports within the specific methods that require them.
* *Mechanism 2 (Trimesh-First Inspection):* Refactored `TopologyInspector` to prioritize `trimesh` for manifoldness checks. Trimesh is significantly lighter and does not trigger VTK scanning. VTK is now only used as a fallback if `trimesh` is unavailable.
* *Mechanism 3 (Guardian Behavior):* Modified `BaseMeshGenerator` to treat "TERMINAL" status as a warning rather than a termination condition. The mesher now proceeds with the original input file after logging a warning.

## 3. Mathematical & Physical Implications
* *Validity:* Proceeding with non-watertight geometry relies on Gmsh's robustness. If the input is truly unmeshable, Gmsh will still return an error, but we avoid premature rejection.
* *Quality:* Mesh quality metrics are still calculated after generation. If the "TERMINAL" input leads to poor quality, it will be reflected in the final report.

## 4. Performance Trade-offs
* *Compute Cost:* Negligible. The check is now faster due to Trimesh.
* *Memory Cost:* Reduced baseline memory footprint as VTK modules are not loaded unless a fallback is needed.
* *Latency:* Startup latency reduced from ~15s to <1s.

## 5. Verification Plan
* *Sanity Check:* Run `scripts/run_local_modal.py` and verify workers start without "Scanning vtkmodules" logs.
* *Regression:* Verify that a known "broken" file still produces a warning but attempts meshing.
