# ADR-005: Golden CHT Case OpenFOAM Configuration

## Status
**In Progress** - Meshing validated, solver thermophysics debugging active

## Context
We need a "Master Template" OpenFOAM case for chtMultiRegionFoam that handles dirty supply chain CAD without crashing. This ADR documents configuration decisions and what works/doesn't work.

---

## Architecture Decisions

### 1. Directory Structure
**Decision**: Use standard OpenFOAM multi-region structure with explicit region subdirectories.

```
Golden_CHT_Case/
├── 0/                    # Time zero boundary conditions
│   ├── region1/          # Fluid region
│   └── solid_heatsink/   # Solid region
├── constant/
│   ├── regionProperties  # Defines region types
│   ├── region1/          # Fluid constants
│   └── solid_heatsink/   # Solid constants  
└── system/
    ├── controlDict       # Global settings
    ├── region1/          # Fluid schemes/solution
    └── solid_heatsink/   # Solid schemes/solution
```

**Status**: ✅ Works - splitMeshRegions creates this structure automatically

---

### 2. snappyHexMeshDict Settings for Dirty CAD

**Decision**: Prioritize stability with relaxed quality thresholds.

| Parameter | Default | Our Value | Rationale |
|-----------|---------|-----------|-----------|
| `nRelaxIter` | 3 | **5** | More relaxation for poor geometry |
| `maxNonOrtho` | 65 | **70** | Allow slightly worse cells to avoid crashes |
| `maxBoundarySkewness` | 20 | **25** | More tolerance at boundaries |
| `implicitFeatureSnap` | false | **true** | Better for dirty STL without edges |
| `addLayers` | true | **false** | Disabled for stability |

**Status**: ✅ Works - snappyHexMesh completes with 62,800 cells, 0 errors

---

### 3. fvSchemes - First-Order Upwind

**Decision**: Use first-order upwind for all convective terms to maximize stability.

```cpp
divSchemes
{
    div(phi,U)      Gauss upwind;   // NOT linearUpwind or QUICK
    div(phi,h)      Gauss upwind;
    div(phi,K)      Gauss upwind;
}
```

**Status**: ✅ Works - These schemes are universally stable

---

### 4. OpenFOAM 13 regionProperties Format

**Decision**: Use the Foundation v13 list syntax.

```cpp
// WORKS (OpenFOAM 13 Foundation)
regions
(
    fluid       (region1)
    solid       (solid_heatsink)
);

// DOES NOT WORK (older format)
regions
(
    fluid { region1; }
    solid { solid_heatsink; }
);
```

**Status**: ✅ Works - Parser accepts list format

---

### 5. Solid Thermophysics (OpenFOAM 13)

**Decision**: OpenFOAM 13 changed solid thermo API significantly.

#### What DOESN'T Work:
```cpp
// OLD FORMAT - FAILS
thermoType
{
    transport       constIso;    // Error: "Unknown transport type"
    thermo          hConst;      // Error: "Unknown thermo type hConst"
}
```

#### What SHOULD Work:
```cpp
// OpenFOAM 13 FORMAT
thermoType
{
    type            heSolidThermo;
    mixture         pureMixture;
    transport       constIsoSolid;  // Changed from constIso
    thermo          eConst;         // Changed from hConst
    equationOfState rhoConst;
    specie          specie;
    energy          sensibleInternalEnergy;  // Changed from sensibleEnthalpy
}
```

**Status**: ⚠️ Partially works - Parser accepts syntax, but solver has file I/O issues

---

### 6. Fluid Thermophysics (OpenFOAM 13)

**Decision**: OpenFOAM 13 may have changed const transport as well.

#### Attempting:
```cpp
thermoType
{
    type            heRhoThermo;
    mixture         pureMixture;
    transport       sutherland;     // More standard than "const"
    thermo          hConst;
    equationOfState perfectGas;
    specie          specie;
    energy          sensibleEnthalpy;
}
```

**Status**: 🔄 Testing in progress

---

## Validation Results

### Meshing Pipeline
| Step | Status | Result |
|------|--------|--------|
| blockMesh | ✅ Pass | 4,000 cells |
| snappyHexMesh | ✅ Pass | 62,800 cells, 0 errors |
| splitMeshRegions | ✅ Pass | 2 regions created |
| checkMesh (region1) | ✅ Pass | "Mesh OK" |
| checkMesh (solid_heatsink) | ✅ Pass | "Mesh OK" |

### Solver Execution
| Solver | Status | Notes |
|--------|--------|-------|
| chtMultiRegionFoam | ❌ Redirects to foamMultiRun | OF13 change |
| foamMultiRun | ⚠️ Fails on thermophysics | Debugging |

---

## Lessons Learned

1. **OpenFOAM 13 broke backward compatibility** for thermophysics
2. **snappyHexMesh implicit feature snapping** is more robust for dirty CAD than explicit
3. **nRelaxIter=5** is necessary for geometries with sharp edges/poor triangulation
4. **First-order upwind** never fails, use for template
5. **Region naming**: splitMeshRegions uses cellZone names, so snappyHexMeshDict must define cellZones

---

## Next Steps

1. Find working OpenFOAM 13 tutorial thermophysics example
2. Test with laplacianFoam first (simpler, single region)
3. If OF13 incompatible, document required changes for Oscar's scripts
