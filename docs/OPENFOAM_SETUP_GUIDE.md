# OpenFOAM CHT Case Structure Guide

This document details the OpenFOAM case structure used by the Sequential Thermal Job Runner for conjugate heat transfer (CHT) simulations of electronics cooling.

## Directory Structure

```
case_<setup_name>/
├── 0/                          # Initial conditions (time=0)
│   ├── region1/                # Fluid region fields
│   │   ├── T                   # Temperature [K]
│   │   ├── U                   # Velocity [m/s]
│   │   ├── p                   # Pressure [Pa]
│   │   └── p_rgh               # Pressure (hydrostatic removed)
│   └── solid_heatsink/         # Solid region fields
│       └── T                   # Temperature [K]
│
├── constant/                   # Physical properties
│   ├── regionProperties        # Region definitions
│   ├── triSurface/            # STL geometry files
│   ├── region1/               # Fluid properties
│   │   └── thermophysicalProperties
│   └── solid_heatsink/        # Solid properties
│       └── thermophysicalProperties
│
└── system/                     # Solver settings
    ├── controlDict             # Simulation control
    ├── fvSchemes               # Discretization schemes
    ├── fvSolution              # Solver algorithms
    ├── blockMeshDict           # Background mesh
    ├── snappyHexMeshDict       # Refinement around geometry
    ├── region1/               # Region-specific settings
    └── solid_heatsink/
```

---

## Key Configuration Files

### 1. controlDict (Simulation Control)

```c
application     chtMultiRegionFoam;
startTime       0;
endTime         200;           // Number of iterations
deltaT          1;
writeInterval   20;            // Output frequency
```

### 2. Boundary Conditions

**Fluid Region (0/region1/T)**
```c
inlet   { type fixedValue; value uniform 300; }  // 27°C inlet
outlet  { type inletOutlet; inletValue uniform 300; }
region1_to_solid_heatsink {
    type compressible::turbulentTemperatureCoupledBaffleMixed;
    Tnbr T;
    kappaMethod fluidThermo;
}
```

**Velocity (0/region1/U)**
```c
inlet   { type fixedValue; value uniform (1 0 0); }  // 1 m/s
outlet  { type inletOutlet; inletValue uniform (0 0 0); }
```

### 3. Material Properties

**Fluid (Air)**
| Property | Value | Units |
|----------|-------|-------|
| Molecular weight | 28.9 | kg/kmol |
| Cp | 1005 | J/kg/K |
| Transport | Sutherland | - |

**Solid (Aluminum 6061)**
| Property | Value | Units |
|----------|-------|-------|
| Thermal conductivity | 167 | W/m/K |
| Density | 2700 | kg/m³ |
| Specific heat | 896 | J/kg/K |

---

## Running the Simulation

### Prerequisites
- OpenFOAM 13 (Foundation version) in WSL or Linux
- STL geometry files in `constant/triSurface/`

### Commands
```bash
# 1. Generate background mesh
blockMesh

# 2. Refine around geometry
snappyHexMesh -overwrite

# 3. Split into regions
splitMeshRegions -cellZones -overwrite

# 4. Run solver
chtMultiRegionFoam   # or: foamMultiRun (OpenFOAM 13+)

# 5. Convert to VTK for visualization
foamToVTK -latestTime
```

---

## Three Standard Setups

| Setup | Power | Velocity | Use Case |
|-------|-------|----------|----------|
| `low_power_natural` | 5W | 0.1 m/s | IoT devices, natural convection |
| `medium_power_forced` | 25W | 1.0 m/s | Standard PCB with chassis fan |
| `high_power_active` | 50W | 3.0 m/s | High-performance with active cooling |

---

## Validation Criteria

| Check | Pass Condition |
|-------|----------------|
| Max temperature | < 150°C (electronics limit) |
| Convergence | Residual < 1e-4 |
| Outliers | Within 3σ of mean |
| Physical sanity | T_min < T_avg < T_max |

---

## Mesh Quality Thresholds

From `snappyHexMeshDict`:
- Max non-orthogonality: 70° (relaxed for dirty CAD)
- Max skewness: 25
- Min Jacobian: 1e-15 (very permissive)

These are intentionally relaxed to handle real-world CAD files.
