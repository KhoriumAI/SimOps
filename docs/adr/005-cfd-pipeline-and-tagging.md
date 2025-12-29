# 005. CFD Pipeline & Advanced Tagging Logic

**Date**: 2025-12-28  
**Status**: Adopted  

## Context
To support Computational Fluid Dynamics (CFD) validation, specifically the Poiseuille Flow benchmark, we needed to extend the SimOps pipeline to support:
1.  **Internal Flow**: Disable the default "Virtual Wind Tunnel" which wraps geometry in a large air box.
2.  **Flow Direction**: Support vector-based inlet velocity (e.g., flow along Z-axis) instead of hardcoded X-axis default.
3.  **Boundary Tagging**: Explicitly define "inlet", "outlet", and "wall" boundaries on arbitrary geometry, overriding the default "Golden Template" (HeatSource/Adiabatic) logic used for thermal analysis.

## Decision
We implemented the following changes to the core architecture:

### 1. Schema Extensions
- **Virtual Wind Tunnel**: Added `virtual_wind_tunnel` (bool) to `PhysicsConfig`. This allows explicit control over domain generation. Defaults to auto-detection (enabled if velocity > 0), but can be forced `False` for internal flow.
- **Inlet Velocity Vector**: Updated `inlet_velocity` in `PhysicsConfig` to accept `Union[float, List[float]]`, allowing `[0, 0, 1]` for Z-aligned flows.

### 2. Custom Tagging Rules
We integrated a robust custom tagging system into `CFDMeshStrategy`.
- **Selector Logic**: Improved `z_min` and `z_max` selectors to strictly check for **flatness** (planar faces). A face is selected only if *both* its minimum and maximum Z coordinates are within tolerance of the target plane. This prevents vertical side walls from being incorrectly tagged as caps.
- **Rules Engine**: The strategy now accepts a list of `TaggingRule` objects. If present, it bypasses the default "Golden Template" logic and applies rules in order:
    1.  `z_min` / `z_max` (Planar Caps)
    2.  `all_remaining` (Walls)

### 3. Verification
Verified via `tests/cfd_validation/01_poiseuille_flow.py` using a Cylinder geometry.
- **Outcome**: Successfully tagged Inlet (1 surface), Outlet (1 surface), and Wall (2 surfaces).
- **Solver**: OpenFOAM environment is currently a prerequisite. The pipeline generates valid meshes and physical groups ready for `gmshToFoam`.

## Consequences
- **Positive**: Enables accurate setup for internal flow problems (pipes, ducts) and complex boundary conditions.
- **Negative**: Requires correctly configured JSON sidecar with tagging rules for custom geometries.
- **Constraint**: Users must have OpenFOAM installed and accessible via `bash` for the solver stage to execute.
