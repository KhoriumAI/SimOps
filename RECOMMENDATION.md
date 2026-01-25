# Recommendation: Switch to Built-in Solver

## Problem Summary

**OpenFOAM Issue**: Your mesh has only 1 boundary patch, but OpenFOAM needs separate hot/cold patches to establish thermal gradient.

**Result**:
- Built-in solver: ✅ 62% nodes with gradient (300K→800K)
- OpenFOAM solver: ❌ 0% nodes with gradient (99.6% at 373K)

## Immediate Fix: Use `solver: "builtin"`

### Option 1: Config File
```json
{
  "solver": "builtin",
  "heat_source_temperature": 800.0,
  "ambient_temperature": 300.0
}
```

### Option 2: Command Line
```bash
python simops_pipeline.py model.step --solver builtin
```

### Option 3: Desktop App
Set solver to "Built-in" in settings (not "OpenFOAM")

## Why Built-in Works Better

| Feature | Built-in Solver | OpenFOAM |
|---------|----------------|----------|
| **Boundary handling** | Applies BCs by geometry (Z-coordinate) | Needs separate patches in mesh |
| **Your results** | ✅ Proper gradient | ❌ Nearly uniform |
| **Speed** | Very fast (~0.02s) | Slower (~1.0s) |
| **Mesh requirements** | Any mesh works | Needs multiple boundary patches |

## Long-term Fix for OpenFOAM (Future)

To use OpenFOAM with proper gradients, the mesh needs multiple physical groups:
1. Create "hot_face" physical surface in CAD/mesh generator
2. Create "cold_face" physical surface
3. gmshToFoam will create separate patches
4. OpenFOAM can apply different BCs to each

**This requires changes to meshing strategy**, not just BC writing.

## Test Now

Run a quick test with built-in solver:

```bash
cd c:\Users\markm\Downloads\Simops

# Create test config
cat > test_config.json << EOF
{
  "solver": "builtin",
  "heat_source_temperature": 373.15,
  "ambient_temperature": 293.15,
  "hot_wall_face": "z_min"
}
EOF

# Run simulation (replace with your mesh file)
python simops_pipeline.py your_mesh.msh -o test_builtin --config "$(cat test_config.json)"

# Check results
python debug_thermal_viz.py test_builtin/thermal_result.vtk
```

**Expected**: Z-correlation >0.8, many intermediate temperature values

## Summary

✅ **Use built-in solver** - it works correctly with your mesh
❌ **Don't use OpenFOAM** - until mesh has multiple boundary patches

The visualization fixes (contours + 3D) work with both solvers!
