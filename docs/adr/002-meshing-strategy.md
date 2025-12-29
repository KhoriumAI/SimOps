# ADR 002: Meshing Strategy (GMSH + Observer Mode)
*Status:* Accepted
*Date:* 2025-12-28
*Tags:* #meshing, #geometry, #reliability

## 1. Context & Problem Statement
Meshing complex CAD assemblies often leads to "hangs" or geometry errors (e.g., during `removeDuplicateNodes()`). Attempting to fix bad meshes automatically frequently causes the process to stall indefinitely.
* *The Constraint:* The pipeline must be robust enough to handle dirty CAD without infinite loops.
* *The Goal:* Move from "active cleaning" to "observational gating" for increased reliability.

## 2. Technical Decision
Adopt the HXT algorithm for 3D meshing and implement internal "Observer Mode" quality gates.
* *Mechanism:*
    * Use `General.NumThreads` for parallel HXT meshing.
    * Use `gmsh.model.mesh.field` for curvature-adaptive refinement.
    * **Observer Mode:** Instead of calling expensive cleanup functions, the system checks mesh quality (Gamma, Density) and immediately rejects/fails the mesh if it doesn't meet minimum thresholds, preventing hangs.
* *Dependencies:* Requires GMSH 4.11+ with HXT support.

## 3. Mathematical & Physical Implications
* *Conservation:* Preserves volume identity by avoiding aggressive merge/cleanup that collapses small gaps.
* *Stability:* Ensures the solver doesn't receive highly distorted elements that cause divergence.
* *Geometric Constraints:* Large assemblies require `strategies/assembly_strategy.py` to manage multi-volume tagging.

## 4. Performance Trade-offs
* *Compute Cost:* Faster average meshing time due to removal of slow cleanup steps.
* *Memory Cost:* Higher peak memory during HXT parallel execution.

## 5. Verification Plan
* *Sanity Check:* Mesh `Airfoil.step` and verify it completes in < 30 seconds.
* *Regression:* Run `reproduce_hang.py` to confirm the hang is resolved by skipping problematic cleanup.
