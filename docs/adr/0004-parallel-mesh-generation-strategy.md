# ADR 0004: Parallel Volume-by-Volume Meshing Strategy
*Status:* Accepted
*Date:* 2025-12-28
*Tags:* #meshing, #parallelism, #performance, #robustness

## 1. Context & Problem Statement
Large CAD assemblies with hundreds of volumes often cause Gmsh to hang or crash when processed as a single monolithic entity. A single problematic volume can stall the entire meshing process.

* *The Constraint:* Gmsh's internal state can become corrupted or inefficient when handling extremely large, complex assemblies in a single thread.
* *The Goal:* Create a resilient meshing pipeline that can handle complex assemblies by isolating failures and utilizing multi-core hardware.

## 2. Technical Decision
*Mechanism:* Implementing a "Persistent Workers with Timeouts" architecture (`Fast Parallel Mesh`).
* *Isolation:* Each worker process loads the CAD, isolates EXACTLY ONE volume, meshes it, and exports a standalone `.msh` file.
* *Watchdog:* A strict 30s (configurable) timeout per volume using `func_timeout`.
* *Persistence:* Successful meshes are cached; only failed or missing volumes are re-processed.
* *Merging:* Final assembly is created by `gmsh.merge()`ing individual volume meshes.

*Dependencies:* `multiprocessing`, `func_timeout`.

## 3. Mathematical & Physical Implications
* *Conservation:* Yes. Volume conservation is maintained as each volume is meshed independently.
* *Stability:* Increases system stability by preventing a single "bad apple" volume from crashing the entire process.
* *Geometric Constraints:* Relies on volumes being properly distinct in the STEP file. Intersecting volumes may lead to non-conformal interfaces unless pre-processed.

## 4. Performance Trade-offs
* *Compute Cost:* Increases overhead due to repeated CAD loading in each worker. However, total wall time is significantly reduced on multi-core systems.
* *Memory Cost:* Higher peak memory usage as multiple processes load CAD data simultaneously.

## 5. Verification Plan
* *Sanity Check:* Verify `generated_meshes/merged.msh` contains the expected number of physical groups.
* *Regression:* Compare total volume of the merged mesh against the CAD original.
