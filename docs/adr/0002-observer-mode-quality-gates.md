# ADR 0002: Implementation of "Observer Mode" (No-Cleanup Quality Gates)
*Status:* Accepted
*Date:* 2025-12-28
*Tags:* #meshing, #robustness, #assembly, #quality-gate

## 1. Context & Problem Statement
Large CAD assemblies (8,000+ surfaces) frequently encountered "Computation Hangs" during the splitting phase. The root cause was identified as `gmsh.model.mesh.removeDuplicateNodes()`. On "sliver" geometry or slightly overlapping faces, this function enters exponential calculation loops or infinite cycles trying to reconcile floating-point proximity between thousands of nodes.

Legacy logic tried to "Fix" the mesh before checking quality, which led to silent stalls that bypassed manager timeouts.

## 2. Technical Decision
*Mechanism:* 
1. **Abolition of Cleanup:** Removed all calls to `gmsh.model.mesh.removeDuplicateNodes()` in the `_parallel_split_worker`.
2. **Observer Mode (Judge, Don't Touch):** The pipeline now operates as a pure observer. It generates the mesh and immediately evaluates it against strict quality gates.
3. **Fatal Rejection Gate:** 
   - `FATAL_GAMMA_FLOOR = 0.001`: Any part with a minimum element gamma below this value is immediately rejected and boxed. This prevents "Zero Area" elements from crashing downstream octree merges.
4. **Heuristic Quality Gate:**
   - Rejected if `minGamma < 0.05` AND `avgGamma < 0.40`. This filters out "messy" meshes that would likely cause solver divergence.
5. **Real-time Heartbeat:** Enabled `flush=True` on all worker print statements to ensure terminal buffering doesn't hide the exact volume causing a bottleneck.

*Dependencies:* Relies on `gmsh.model.mesh.getElementQualities(..., "gamma")` for evaluation.

## 3. Mathematical & Physical Implications
*Stability:* Eliminates the primary source of process-level hangs in the assembly pipeline.
*Accuracy:* Rejection of elements with $\gamma < 10^{-3}$ significantly improves the robustness of the Octree-based mesh merger and prevents "Null bucket" errors.
*Coverage:* Some parts that might have been "saved" by node merging will now be boxed. However, this is a preferred trade-off for predictable execution timelines.

## 4. Performance Trade-offs
*Compute Cost:* Reduced. Skipping the node merging step saves significant CPU time on complex parts.
*Memory Cost:* Reduced. Node merging is a RAM-intensive operation on high-vertex models.
*Reliability:* Significantly increased. The pipeline is now deterministic; it either finishes fast or hits the timeout gate gracefully.

## 5. Verification Plan
*Sanity Check:* Run `scripts/parallel_safe_splitter.py` on the target heater board assembly.
*Verification:* Confirm that volumes 1, 2, and 54 are rejected for `Fatal Gamma` without hanging the worker process.
*Integration:* Verify that `strategies/assembly_strategy.py` identifies these failures and proceeds to box them automatically.
