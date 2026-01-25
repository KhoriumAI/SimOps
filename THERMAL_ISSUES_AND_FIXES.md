# Thermal Analysis Issues and Fixes

## Issue Summary

You reported two problems:
1. **2D scatter plots show random red/blue distribution** ✅ FIXED
2. **3D isometric view is all one color (blue)** ✅ FIXED (visualization)
3. **No clear thermal gradient in some cases** ⚠️ NEW ISSUE (physics)

## Issue 1: 2D Visualization - Random Scatter ✅ FIXED

### Root Cause
Scatter plots projected 3D temperature data onto 2D planes, causing depth overlaps to appear random.

### Fix
Replaced scatter with interpolated contour plots using `scipy.interpolate.griddata`.

### Files Modified
- `simops_pipeline.py` (lines 868-969)

### Verification
![Comparison](simops_output/fddbfe8a/visualization_comparison.png)
- OLD: Random red/blue dots
- NEW: Smooth thermal gradients

## Issue 2: 3D Visualization - All Blue ✅ FIXED

### Root Cause
**PyVista incompatibility with Python 3.14+**:
```
AttributeError: 'typing.Union' object attribute '__doc__' is read-only
```

PyVista imports successfully but crashes during rendering, creating blank/all-blue images.

### Fix
1. Created matplotlib 3D fallback backend
2. Auto-detection of Python 3.14+ forces matplotlib
3. Per-view fallback when PyVista rendering fails

### Files Modified
- `core/reporting/thermal_multi_angle_viz.py` - Added fallback logic
- `core/reporting/thermal_viz_matplotlib3d.py` - New matplotlib backend

### Verification
Test files generated:
- `fddbfe8a_test_thermal_isometric.png` - Shows blue→red gradient ✅
- `daaf205a_core_sample_FIXED_thermal_isometric.png` - Shows temperature distribution ✅

## Issue 3: Weak Thermal Gradients (daaf205a case) ⚠️ PHYSICS ISSUE

### Diagnostic Results

**Case: daaf205a_core_sample_medium_fast_tet**
- **Solver**: OpenFOAM laplacianFoam
- **Temperature range**: 293.15K - 373.15K (80K spread)
- **Unique values**: Only 18 (should be thousands)
- **Z-correlation**: -0.099 ❌ (should be >0.7)
- **Top temperature**: 373.15K (hot) ❌ Should be cold
- **Bottom temperature**: 373.15K (hot) ✅ Correct

### Root Cause: Boundary Condition Mismatch

**OpenFOAM boundary file** (constant/polyMesh/boundary):
```
1
(
    patch0  // Only 1 patch exists
    {
        type            patch;
        nFaces          8650;
        startFace       18463;
    }
)
```

**Boundary conditions file** (0/T):
```python
boundaryField
{
    ".*"                // Catches ALL patches
    {
        type            fixedValue;
        value           uniform 293.15;  // Sets everything to COLD
    }
    "heatsink_bottom"   // This patch DOESN'T EXIST!
    {
        type            fixedValue;
        value           uniform 373.15;
    }
    "patch0"            // Never reached (already caught by ".*")
    {
        type            fixedValue;
        value           uniform 373.15;
    }
}
```

**Result**: All boundaries set to 293K (cold) by the wildcard. The 373K you see is from `internalField uniform 373.15` (initial conditions), not from solving.

### Fix Required

The OpenFOAM boundary conditions need to be fixed in `simops_pipeline.py`:

```python
def _write_boundary_conditions(self, case_dir: Path):
    amb = self.config.ambient_temperature
    hot = self.config.heat_source_temperature

    content = f"""
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    "0";
    object      T;
}}
dimensions      [0 0 0 1 0 0 0];
internalField   uniform {amb};  // Start with ambient, not hot!
boundaryField
{{
    patch0  // Use actual patch name from boundary file
    {{
        type            mixed;  // Allow both hot and cold regions
        refValue        uniform {hot};
        refGradient     uniform 0;
        valueFraction   uniform 0.5;  // 50/50 mix for general case
    }}
}}
"""
```

### Better Fix: Identify Hot/Cold Surfaces Geometrically

The proper fix requires:
1. **Read the boundary file** to get actual patch names
2. **Analyze patch geometry** (Z-min, Z-max, etc.)
3. **Apply correct BC** to each patch based on position

This requires modifying `OpenFOAMRunner._write_boundary_conditions()` in `simops_pipeline.py` (lines 688-722).

## Issue 4: Multiple Volumes / TIE (Thermal Interface Elements)

### What You Asked About
"not sure if TIE is enabled to pass thermal between volumes"

### Current Status
The mesh analysis shows **only 1 cellZone** and **1 boundary patch**, so there's only one volume.

### If You Have Multiple CAD Bodies

If your CAD file contains multiple disconnected bodies (e.g., a heatsink + PCB + chip), you need:

1. **Option A: Merge volumes** in CAD before meshing
2. **Option B: Add interface handling** in OpenFOAM:
   - Create `regionProperties` to define multiple regions
   - Use `chtMultiRegionFoam` instead of `laplacianFoam`
   - Define interface patches with `compressible::turbulentTemperatureCoupledBaffleMixed` BC
3. **Option C: Use CalculiX** instead of OpenFOAM (has better multi-body support)

## Recommended Actions

### Immediate (Already Done ✅)
1. ✅ Use interpolated contours for 2D visualization
2. ✅ Force matplotlib for 3D on Python 3.14+
3. ✅ Add matplotlib fallback for PyVista failures

### Short-term (Requires Code Changes)
1. ⚠️ **Fix OpenFOAM boundary conditions** to use actual patch names
2. ⚠️ Add geometric analysis to identify hot/cold surfaces
3. ⚠️ Change initial field to `ambient` instead of `hot`

### Long-term (Architecture)
1. Add multi-region support for disconnected volumes
2. Switch to `chtMultiRegionFoam` for complex assemblies
3. Add TIE (thermal interface) handling for component contacts

## Testing

### Quick Test for Visualization Fixes
```bash
python regenerate_3d_views.py
```

Check the `*_FIXED_thermal_*.png` files.

### Test for Physics Fix (After Implementing)
1. Run simulation on simple geometry (single cube)
2. Check diagnostics: correlation should be >0.8
3. Verify temperature gradient from hot to cold
4. Ensure 100+ unique temperature values

## Files Reference

| File | Purpose | Status |
|------|---------|--------|
| `simops_pipeline.py` | Main pipeline, OpenFOAM BC writer | Needs BC fix |
| `core/reporting/thermal_multi_angle_viz.py` | 3D visualization controller | ✅ Fixed |
| `core/reporting/thermal_viz_matplotlib3d.py` | Matplotlib 3D backend | ✅ Added |
| `debug_thermal_viz.py` | Diagnostic tool | ✅ Working |
| `regenerate_3d_views.py` | Test script | ✅ Working |

## Summary

| Issue | Type | Status | Action Required |
|-------|------|--------|-----------------|
| Random 2D scatter | Visualization | ✅ Fixed | None - deployed |
| All-blue 3D view | Visualization | ✅ Fixed | None - deployed |
| Weak thermal gradients | Physics/Solver | ⚠️ Identified | Fix OpenFOAM BCs |
| Multi-volume TIE | Physics/Solver | ℹ️ Not applicable yet | Future feature |

**Next Step**: Fix the OpenFOAM boundary condition writer to use actual patch names from the mesh.
