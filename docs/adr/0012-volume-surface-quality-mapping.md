# ADR 0012: Robust Volume-to-Surface Quality Mapping
*Status:* Accepted
*Date:* 2026-01-02
*Tags:* #visualization, #quality, #frontend, #backend, #meshing

## 1. Context & Problem Statement
The web frontend visualizes mesh quality by coloring the 2D surface triangles. However, the critical quality metrics (SICN, Gamma, etc.) are calculated for the 3D volume elements (tetrahedrons). 

* *The Constraint:* When a user selects a metric other than SICN, the mesh appeared gray because the backend was failing to map 3D volume quality values to the 2D surface elements rendered by the browser. 
* *The Goal:* Achieve parity between the on-prem GUI and the web version by ensuring every surface triangle reflects the quality of its adjacent volume element for all metrics.

## 2. Technical Decision
We implemented a **Robust Node-Set Intersection Heuristic** for quality mapping.

* *Mechanism:* 
    1. During mesh post-processing, we build a `node_to_element` map for all volume elements.
    2. For every surface triangle, we extract its corner nodes.
    3. We perform a set intersection of the volume elements attached to those specific corner nodes.
    4. The resulting intersection identifies the exactly adjacent volume element(s).
    5. The "worst" quality value (min/max depending on metric) from the adjacent volume is assigned to the surface triangle.
* *Dependencies:* Requires the `gmsh` API and `numpy` for efficient array handling during the reload/mapping phase.

## 3. Mathematical & Physical Implications
* *Stability:* The heuristic is stable for standard manifold meshes. Non-manifold edges may lead to assignment from multiple adjacent volumes.
* *Geometric Constraints:* Elements must share all three corner nodes with a volume element to be considered "adjacent" in this strict intersection model.

## 4. Performance Trade-offs
* *Compute Cost:* Increases the post-meshing "extraction" phase by ~1-2 seconds for meshes with >1M elements due to map building and set operations.
* *Memory Cost:* Temporary memory spike in the `mesh_worker_subprocess` to hold the node-to-volume mapping.

## 5. Verification Plan
* *Sanity Check:* Select "Skewness" in the web UI; verify that "bad" (red) elements are visible in areas of high curvature or tight gaps.
* *Regression:* Verify that SICN coloring remains identical to previous builds.
