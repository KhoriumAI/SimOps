================================================================================
GOLDEN CHT CASE - README
================================================================================
OpenFOAM chtMultiRegionFoam Master Template for Conjugate Heat Transfer
================================================================================

OVERVIEW
--------
This is a battle-tested OpenFOAM case template for multi-region conjugate heat
transfer simulations. It prioritizes STABILITY over perfect accuracy, designed 
to handle "dirty" supply chain CAD without crashing during meshing.

STATUS: Template structure validated. Meshing pipeline (blockMesh + snappyHexMesh
        + splitMeshRegions) runs successfully. Solver requires OpenFOAM-specific
        thermophysics configuration for your installation version.

VALIDATED REGIONS
-----------------
  - region1        : Fluid domain (28,800 cells, mesh OK)
  - solid_heatsink : Aluminum heatsink solid (34,000 cells, mesh OK)

MESH QUALITY METRICS (from checkMesh)
-------------------------------------
  - Max non-orthogonality: 25.24° (OK < 65°)
  - Max skewness: 0.33 (OK < 4)
  - Max aspect ratio: 1.0 (excellent)
  - All mesh checks: PASSED


================================================================================
VARIABLE DOCUMENTATION (For Python Script Serialization - Oscar's Work)
================================================================================

The following dictionary keys are meant to be modified by the Python script:

SYSTEM/CONTROLDICT
------------------
  endTime         : Number of iterations (default: 100)
  writeInterval   : Results output frequency (default: 10)

SYSTEM/BLOCKMESHDICT
--------------------
  vertices        : Domain bounding box corners [mm]
  blocks          : Background mesh cell counts (20 20 10)

SYSTEM/SNAPPYHEXMESHDICT
------------------------
  locationInMesh  : Point guaranteed in fluid region (METERS, e.g., (0 0 0.025))
  refinementSurfaces.*.level : (min max) refinement levels
  snapControls.nRelaxIter : Relaxation iterations (default: 5, increase for dirty CAD)

0/REGION1/T (Fluid)
-------------------
  internalField   : Initial temperature [K] (default: 300)
  inlet.value     : Inlet temperature [K] (default: 300)

0/REGION1/U (Fluid)  
-------------------
  inlet.value     : Inlet velocity [m/s] (default: (1 0 0))

0/SOLID_HEATSINK/T
------------------
  internalField   : Initial heatsink temperature [K] (default: 350)

CONSTANT/REGION1/THERMOPHYSICALPROPERTIES (Fluid)
-------------------------------------------------
  thermoType.transport  : "sutherland" for OpenFOAM 13
  mixture.transport.As  : Sutherland constant
  mixture.transport.Ts  : Sutherland temperature [K]

CONSTANT/SOLID_HEATSINK/THERMOPHYSICALPROPERTIES
------------------------------------------------
  thermoType.transport  : "constIsoSolid" for OpenFOAM 13
  thermoType.thermo     : "eConst" for OpenFOAM 13  
  thermoType.energy     : "sensibleInternalEnergy" for solids
  mixture.transport.kappa : Thermal conductivity [W/m/K]


================================================================================
USAGE INSTRUCTIONS
================================================================================

1. GENERATE TEST GEOMETRY
   python generate_test_geometry.py
   
   Creates: constant/triSurface/heatsink.stl, constant/triSurface/chip.stl

2. RUN MESHING PIPELINE (in WSL/Linux with OpenFOAM sourced)
   blockMesh                           # Create background mesh
   snappyHexMesh -overwrite            # Refine around geometry
   splitMeshRegions -cellZones -overwrite  # Create multi-region

3. RUN SOLVER (OpenFOAM 13+ uses foamMultiRun)
   foamMultiRun                        # Or: chtMultiRegionFoam (legacy)

4. POST-PROCESS
   foamToVTK -latestTime               # Convert to VTK for ParaView


================================================================================
STABILITY FEATURES
================================================================================

SCHEMES (system/fvSchemes):
  - First-order upwind for all convective terms (div(phi,*))
  - Limited corrected Laplacian schemes

SOLVER (system/fvSolution):
  - Conservative under-relaxation (U=0.5, h=0.5)
  - GAMG preconditioner for pressure
  - Momentum predictor disabled

MESHING (system/snappyHexMeshDict):
  - nRelaxIter = 5 (higher than default 3)
  - Relaxed quality thresholds (maxNonOrtho=70, maxBoundarySkewness=25)
  - Implicit feature snapping (more tolerant of dirty geometry)
  - No boundary layers (disabled for stability)


================================================================================
OPENFOAM VERSION NOTES
================================================================================
This template was created for OpenFOAM (Foundation) version 13.

Key differences from older versions:
  1. chtMultiRegionFoam is replaced by foamMultiRun
  2. Solid thermo: use "constIsoSolid" transport, "eConst" thermo
  3. regionProperties format: regions ( fluid (region1) solid (solid_heatsink) )
  
For ESI-OpenCFD (v2406+), the format may differ slightly.


================================================================================
VERSION INFO
================================================================================
Created: 2026-01-13
OpenFOAM Version: 13 (Foundation, cfd.direct)
Template Version: 1.0
Validation: Mesh OK (blockMesh + snappyHexMesh + splitMeshRegions + checkMesh)

================================================================================
