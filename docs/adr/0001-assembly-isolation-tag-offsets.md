# Assembly Meshing with Surgical Isolation and Tag Offsets
*Status:* Accepted
*Date:* 2026-01-03
*Tags:* #geometry, #parallelization, #optimization, #assembly-meshing

## 1. Context & Problem Statement
We needed to generate high-quality tetrahedral meshes for complex assemblies (e.g., heater cores with 16+ volumes). Beause the system was previously failing to mesh the entire assembly at once (hanging or crashing), we implemented a "Surgical Isolation" strategy which meshes each volume in a separate subprocess.

* *The Constraint:* When merging these isolated valid meshes back into a single assembly, the resulting mesh was consistently **inverted** (negative SICN), even though the individual parts were perfect.
* *The Goal:* Produce a correct, valid assembly mesh (positive SICN) from the isolated parts without topology corruption.

## 2. Technical Decision
We identified that the inversion was caused by **Node ID Collisions**. every isolated worker (running Gmsh in a fresh process) generated nodes starting at ID 1. When `gmsh.merge()` combined them, it could not distinguish between "Volume 1, Node 5" and "Volume 2, Node 5", creating a twisted, invalid topology.

### The Fix: Unique Tag Offsets
We enforced unique numeric ranges for every volume's entities by setting `Mesh.FirstNodeTag` and `Mesh.FirstElementTag` in the worker process before meshing.

* *Mechanism:* 
  `offset = volume_tag * 1,000,000`
  `gmsh.option.setNumber("Mesh.FirstNodeTag", offset)`
  `gmsh.option.setNumber("Mesh.FirstElementTag", offset)`
  
* *Result:* 
  Volume 1 uses IDs 1,000,000+
  Volume 2 uses IDs 2,000,000+
  ...
  This guarantees zero collisions during merge.

### Secondary Decision: Dynamic Meshing Strategy
* *Mechanism:* The isolation worker now accepts a `--strategy` argument. It defaults to the fast **HXT** algorithm (`tet_hxt_optimized`) but can fallback to **Delaunay** (`Mesh.Algorithm3D=1`) if requested by the user or if HXT fails.

### Secondary Decision: Non-Conformal Assembly Merging
* *The Problem:* Even with unique tags, `gmsh.model.mesh.removeDuplicateNodes()` was forcing nodes to merge at interfaces between volumes. Since the volumes are meshed in isolation (non-conformal), their surface triangulations do not match. Merging these mismatched nodes twisted elements, causing inversion (negative SICN).
* *The Fix:* We explicitly **DISABLED** `removeDuplicateNodes()` for surgical assemblies.
* *Result:* The final assembly is a collection of valid, touching, but **discontinuous** (non-conformal) meshes.
* *Trade-off:* 
  - **Pros:** Guarantees valid geometry (Positive SICN).
  - **Cons:** Solvers must use contact interfaces (standard for assemblies) as nodes are not shared across boundaries.

## 3. Mathematical & Physical Implications
* *Validity:* Crucial. Without unique tags, the mesh topology is mathematically invalid (negative Jacobian/SICN) and unusable for FEA/CFD.
* *Stability:* The offset of 1M allows for volumes with up to 1M nodes/elements. If a single part exceeds this, collisions resume. (Current typical parts are <100k nodes).
* *Conformality:* The mesh is now non-conformal. Loads will not transfer across boundaries without explicit contact definitions in the solver.

## 4. Performance Trade-offs
* *Compute Cost:* Negligible. Setting the offset is instantaneous.
* *Memory Cost:* Negligible.
* *Parallelism:* Allows fully parallel generation (limited only by CPU cores) since each worker is independent and collision-free.

## 5. Verification Plan
* *Sanity Check:* Verify that `[Worker Vn]` logs show "Applied Tag Offset: N,000,000".
* *Metric:* Final assembly mesh must have min SICN > 0. (Achieved: Min SICN ~0.17 for core sample).
