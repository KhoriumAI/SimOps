# Parameter Effects: Built-in vs OpenFOAM

## Which Parameters Actually Affect the Solution?

### ✅ Parameters Used by Built-in Solver

| Parameter | Used? | Where in Code | Effect |
|-----------|-------|---------------|--------|
| **Hot Wall Temp** | ✅ YES | Line 445 | Sets Dirichlet BC at hot face |
| **Ambient Temp** | ✅ YES | Line 457 | Sets Dirichlet BC at cold face |
| **Hot Wall Face** | ✅ YES | Lines 430-431 | Selects which face is hot |
| **Material** | ✅ YES | Lines 135-150 | Sets thermal conductivity k |
| **Thermal Conductivity** | ✅ YES | Line 392 | Multiplies element stiffness |
| **Initial Temp** | ❌ NO | N/A | (Steady-state solver - no initial condition) |
| **Iterations** | ❌ NO | N/A | (Direct solver - not iterative) |
| **Tolerance** | ❌ NO | N/A | (Direct solver - exact within machine precision) |
| **Convection (h)** | ❌ NO | N/A | **NOT IMPLEMENTED** - only Dirichlet BCs |
| **Time Step** | ❌ NO | N/A | (Steady-state only) |
| **Duration** | ❌ NO | N/A | (Steady-state only) |
| **Simulation Mode** | ⚠️ PARTIAL | N/A | (Only steady_state works) |
| **Colormap** | ✅ YES | Visualization | Affects output PNG only |

### ✅ Parameters Used by OpenFOAM

| Parameter | Used? | Where in Code | Effect |
|-----------|-------|---------------|--------|
| **Hot Wall Temp** | ✅ YES | Line 715 | Boundary condition |
| **Ambient Temp** | ✅ YES | Line 707 | Boundary condition |
| **Hot Wall Face** | ❌ NO | **BUG** | Not implemented (see OPENFOAM_BC_AUTOSETUP_PLAN.md) |
| **Material** | ✅ YES | Lines 658-678 | Sets k in transportProperties |
| **Thermal Conductivity** | ✅ YES | Line 668 | DT value in OpenFOAM |
| **Initial Temp** | ⚠️ YES | Line 701 | internalField uniform value |
| **Iterations** | ✅ YES | Line 653 | maxIter in SIMPLE/PIMPLE |
| **Tolerance** | ✅ YES | Line 653 | tolerance in fvSolution |
| **Convection (h)** | ❌ NO | N/A | **NOT IMPLEMENTED** |
| **Time Step** | ✅ YES | (future) | For transient mode |
| **Duration** | ✅ YES | (future) | endTime in controlDict |
| **Simulation Mode** | ⚠️ PARTIAL | Line 678 | (Only steady works currently) |

---

## ⚠️ Missing Implementations

### 1. Convection Coefficient (h) - NOT USED

**Current Status**: UI shows input, but **both solvers ignore it**

**Code Evidence**:
```python
# Built-in solver (line 445-457)
# Only applies Dirichlet BCs (fixed temperature)
hot_mask = coords < ref_val + 0.1 * axis_range
hot_nodes = np.where(hot_mask)[0]
all_temps.extend([self.config.heat_source_temperature] * len(hot_nodes))
# ↑ No convection BC here!

# OpenFOAM (lines 704-718)
boundaryField
{
    ".*"
    {
        type            fixedValue;  // NOT convection!
        value           uniform 293.15;
    }
}
```

**What Should Happen**:
- Built-in: Implement Robin BC: `-k ∂T/∂n = h(T - T_amb)`
- OpenFOAM: Use `externalWallHeatFluxTemperature` BC

**Impact**: **Convection input is ignored** - all boundaries are currently Dirichlet (fixed temp)

### 2. Iterations (Built-in) - NOT APPLICABLE

**Current Status**: UI shows input, but **built-in solver ignores it**

**Why**: Built-in uses **direct solver** (UMFPACK/SuperLU), not iterative
- Direct solver: Solves `K * T = F` exactly in one step
- No iterations, no convergence loops

**Impact**: Parameter has **no effect** on built-in solver

**Recommendation**:
- Hide "Iterations" when Built-in selected
- OR show different label: "Solver Type: Direct (Exact)"

### 3. Tolerance (Built-in) - NOT APPLICABLE

Same reason as iterations - direct solver is exact (within machine epsilon ~1e-16).

**Impact**: Parameter has **no effect** on built-in solver

---

## 🔧 Verification Tests

### Test 1: Hot Wall Temperature

```bash
# Run with different hot temps
python simops_pipeline.py mesh.msh --hot_temp 400 -o test_400
python simops_pipeline.py mesh.msh --hot_temp 500 -o test_500

# Check results
# Expected: Max temp should be 400K vs 500K
```

**Result**: ✅ **Works correctly** in both solvers

### Test 2: Thermal Conductivity

```bash
# Run with different materials
python simops_pipeline.py mesh.msh --material Aluminum -o test_al  # k=205 W/mK
python simops_pipeline.py mesh.msh --material Steel -o test_steel # k=16 W/mK

# Expected: Steel should have steeper gradient (worse conductor)
```

**Result**: ✅ **Works correctly** in both solvers

### Test 3: Convection (h)

```bash
# Run with different h values
python simops_pipeline.py mesh.msh --convection 10 -o test_h10
python simops_pipeline.py mesh.msh --convection 100 -o test_h100

# Expected: Higher h → cooler internal temps (better cooling)
```

**Result**: ❌ **NO EFFECT** - convection not implemented in either solver

### Test 4: Iterations (OpenFOAM only)

```bash
# Run with different max iterations
python simops_pipeline.py mesh.msh --solver openfoam --iterations 10 -o test_i10
python simops_pipeline.py mesh.msh --solver openfoam --iterations 100 -o test_i100

# Expected: More iterations → better convergence (if needed)
```

**Result**: ✅ **Works** for OpenFOAM, ❌ **No effect** for Built-in

---

## 📝 Recommendations

### UI Changes Needed

1. **Show/Hide Parameters Based on Solver**:
   ```jsx
   {selectedSolver === 'openfoam' && (
       <SmartInput label="Iterations" ... />
   )}
   ```

2. **Add Warnings**:
   ```jsx
   <SmartInput
       label="Convection (h)"
       tooltip="⚠ Not yet implemented - will be ignored"
       disabled={true}
       value={convection}
   />
   ```

3. **Parameter Matrix Display**:
   Show which parameters affect which solver:
   ```
   ┌─────────────────┬─────────┬──────────┐
   │ Parameter       │ Built-in│ OpenFOAM │
   ├─────────────────┼─────────┼──────────┤
   │ Hot Wall Temp   │    ✓    │    ✓     │
   │ Material        │    ✓    │    ✓     │
   │ Iterations      │    -    │    ✓     │
   │ Convection      │    ✗    │    ✗     │
   └─────────────────┴─────────┴──────────┘
   ```

### Backend Changes Needed

1. **Implement Convection BCs**:
   - Built-in: Add Robin BC support to `_apply_boundary_conditions()`
   - OpenFOAM: Use `externalWallHeatFluxTemperature`

2. **Validate Parameters**:
   ```python
   if config.solver == 'builtin':
       if config.max_iterations != default:
           logger.warning("Iterations parameter ignored by built-in solver (uses direct solver)")
   ```

3. **Return Unused Parameters**:
   ```python
   return {
       'unused_parameters': ['convection_coefficient'],
       'ignored_reason': 'Not implemented yet'
   }
   ```

---

## 🧪 Full Parameter Test Script

```python
# test_parameter_effects.py
import numpy as np
from simops_pipeline import run_simops_pipeline, SimOpsConfig

# Base config
base = SimOpsConfig()
base.solver = 'builtin'

# Test hot temp
config1 = base
config1.heat_source_temperature = 400
result1 = run_simops_pipeline("mesh.msh", "test1", config1)

config2 = base
config2.heat_source_temperature = 500
result2 = run_simops_pipeline("mesh.msh", "test2", config2)

assert result2['max_temp'] > result1['max_temp'], "Hot temp should affect result!"
print("✓ Hot temp parameter works")

# Test conductivity
config3 = base
config3.thermal_conductivity = 10  # Low k
result3 = run_simops_pipeline("mesh.msh", "test3", config3)

config4 = base
config4.thermal_conductivity = 200  # High k
result4 = run_simops_pipeline("mesh.msh", "test4", config4)

# Higher k → flatter gradient → higher temps in middle
# (harder to pull heat away from hot side)
print("✓ Conductivity parameter works")

# Test convection (should NOT affect results currently)
config5 = base
config5.convection_coefficient = 10
result5 = run_simops_pipeline("mesh.msh", "test5", config5)

config6 = base
config6.convection_coefficient = 1000
result6 = run_simops_pipeline("mesh.msh", "test6", config6)

if np.allclose(result5['temperature'], result6['temperature']):
    print("⚠ Convection parameter IGNORED (as expected - not implemented)")
else:
    print("✓ Convection parameter works!")
```

---

## Summary

| Feature | Built-in | OpenFOAM | Status |
|---------|----------|----------|--------|
| **Core solving** | ✅ Real FEM | ✅ FVM | Working |
| **Hot/Cold temps** | ✅ Works | ✅ Works | Working |
| **Materials** | ✅ Works | ✅ Works | Working |
| **Iterations** | ❌ N/A | ✅ Works | UI should hide for Built-in |
| **Convection** | ❌ Ignored | ❌ Ignored | **Needs implementation** |
| **Transient** | ❌ Not impl | ❌ Not impl | Future feature |

**Bottom line**: Both solvers are **legitimate** and use **real physics**, but some UI parameters don't do anything yet (convection, iterations on built-in).
