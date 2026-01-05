# Modal Parallel Assembly Meshing
*Status:* Accepted
*Date:* 2026-01-04
*Tags:* #meshing, #cloud, #performance, #modal

## 1. Context & Problem Statement
The user needs to mesh large CAD assemblies (150+ volumes). The current single-worker approach (running locally or on a single EC2/Modal instance) is too slow because it meshes volumes sequentially or with limited local parallelism (capped at ~6 workers to prevent instability).

* *The Constraint:* Meshing 150+ volumes sequentially takes too long.
* *The Goal:* Utilize massive parallelism to mesh all volumes simultaneously.
* *Requirement:* Solver requires volumes to be separate (for TIE constraints), but the Viewer requires a single file with all meshes.

## 2. Technical Decision
We implemented a "fan-out" strategy using Modal's `.map()` capability to run `mesh_single_volume_task` containers in parallel.

* *Mechanism:* 
    1. `generate_mesh` (Orchestrator) analyzes the STEP file.
    2. If > 3 volumes, it launches N parallel Modal tasks, each handling one volume tag.
    3. Each task runs `isolation_worker_script.py` (our proven surgical logic) in a fresh container.
    4. Orchestrator gathers all fragment `.msh` files.
    5. Orchestrator merges them using `gmsh.merge()` *without* `removeDuplicateNodes()`.
    6. Nodes/Elements are renumbered to ensure global uniqueness.

* *Dependencies:* Modal Python SDK, Gmsh.

## 3. Mathematical & Physical Implications
* *Connectivity:* The resulting mesh is *non-conformal* at interfaces (duplicate nodes exist). This is intentional for solver contact pairing (TIE).
* *Validity:* Renumbering ensures no ID collisions.

## 4. Performance Trade-offs
* *Compute Cost:* Increases linearly with volume count (N containers), but wall-clock time is drastically reduced (approaching `max(single_volume_time)` + overhead).
* *Overhead:* Downloading/Uploading fragments to S3 adds latency, but for large jobs, this is negligible compared to compute time.

## 5. Verification Plan
* *Sanity Check:* Mesh a multi-volume assembly and check that all volumes appear in the viewer.
* *Solver Check:* Verify that the solver can read the individual volume blocks.
