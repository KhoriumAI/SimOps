# ADR 0001: Implementation of V6 "Immortal" Splitter
*Status:* Accepted
*Date:* 2025-12-28
*Tags:* #meshing, #geometry, #robustness, #parallel-processing

## 1. Context & Problem Statement
The meshing pipeline faced two critical failure modes in complex electronics assemblies:
1. **Geometric Interference:** Small quality failures on large structural parts (e.g., PCB brackets > 20mm) triggered auto-boxing. These bounding boxes often intersected other components, short-circuiting simulations.
2. **Symmetry Crashes:** Mirrored parts (mathematically identical but reflected) often caused Gmsh to hit numerical singularities or infinite loops, leading to process hangs.
3. **Cleaning Hangs:** The `removeDuplicateNodes()` call, intended to fix overlapping faces, frequently hung on high-vertex assemblies.

## 2. Technical Decision
*Mechanism:* 
1. **Stochastic Geometry:** Enabled `Mesh.RandomFactor = 1e-8` to apply a sub-atomic random perturbation to node coordinates, breaking mathematical symmetry in mirrored parts.
2. **Diplomatic Immunity:** Increased the "too big to fail" threshold from 20mm to 50mm. 
3. **No-Box Policy:** For parts > 50mm, quality rejection is disabled. Distorted meshes are preserved over bounding boxes unless the mesh is truly empty.
4. **Relaxed Tolerance:** Increased `Mesh.ToleranceInitial` to `1e-3` to improve surface meshing success rate on faceted CAD.
5. **Fail-Fast Splitter:** Removed the hanging `removeDuplicateNodes()` utility in favor of preserving the raw mesh.

*Dependencies:* Requires `gmsh` 4.10+ for consistent `RandomFactor` behavior across platforms.

## 3. Mathematical & Physical Implications
*Conservation:* Conservative? Yes. The perturbation is small enough ($10^{-8}$ mm) that it does not affect physical results or volume conservation.
*Stability:* Increases stability by preventing division-by-zero or zero-volume-element errors in strictly symmetric cases.
*Geometric Constraints:* Large parts may contain low-quality elements ($\gamma < 10^{-3}$), which solvers must handle (CalculiX/OpenFOAM have internal limiters).

## 4. Performance Trade-offs
*Compute Cost:* Negligible. Stochastic perturbation is done during node creation.
*Memory Cost:* None.
*Triage Cost:* Reduces "Simulation-Ending" intersection errors at the cost of potentially poorer local mesh quality on complex structural components.

## 5. Verification Plan
*Sanity Check:* Run `scan_assembly_metrics.py` to verify threshold logic.
*Regression:* Verify the "Mirrored Part" (Vol 7/Vol 8) meshes successfully without timing out.
*Integration:* Check that `_boxed.json` identifies the correct (small) parts as boxed.
