# ADR 0006: Feature Parity for Mesh Generation Settings

## Status
Accepted

## Context
The desktop application provided more granular control over mesh generation than the initial web frontend. To achieve feature parity, the web frontend requires additional UI elements for:
- Minimum Element Size
- Element Order (Linear vs. Quadratic)
- ANSYS Export Modes (CFD/FEA)
- Optional Target Element Count (ability to "remove" the target constraint)

The backend meshing worker already had partial support for these parameters, but they were not effectively wired to the frontend, and some naming mismatches existed.

## Decision
1.  **Frontend (App.jsx)**:
    - Added state variables and UI controls for `minElementSize`, `elementOrder`, and `ansysMode`.
    - Added a toggle for `useTargetElements`. When disabled, `target_elements` is passed as `null`.
    - Unified naming convention to match backend expectations (`max_size_mm`, `min_size_mm`).

2.  **Backend (mesh_worker_subprocess.py)**:
    - Implemented safe handling for `target_elements: null`.
    - Verified that `element_order` (1 for Tet4, 2 for Tet10) and `ansys_mode` are correctly propagated to the meshing strategies.

3.  **Strategy Logic (exhaustive_strategy.py)**:
    - Updated the heuristic sizing logic to skip target-based calculations if `target_elements` is not provided, allowing purely constraint-based meshing.

## Consequences
- The web frontend now provides equivalent control to the desktop application for these critical parameters.
- Meshing becomes more predictable when users specify absolute size constraints without an implicit target count.
- ANSYS-compatible files (.bdf) are now generated via the web interface.
