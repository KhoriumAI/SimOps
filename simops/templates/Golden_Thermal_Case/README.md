# Golden Thermal Case (Scrappy Edition)

**"Code that isn't running in front of a user today is worthless."**

This is the simplified, robust, and verified template for Thermal CHT simulations using **OpenFOAM ESI v2312**.

## 🚀 Quick Start
```bash
# 1. Generate Geometry & Mesh (Gmsh)
python generate_tet_mesh.py

# 2. Convert to OpenFOAM & Split Regions
# Must be run in OpenFOAM environment (WSL/Docker)
gmshToFoam test_geom.msh
splitMeshRegions -cellZones -overwrite

# 3. Setup Case (Properties, BCs, Control)
python setup_case.py

# 4. Run Steady-State Solver
chtMultiRegionSimpleFoam
```

## 📐 Architecture
- **Mesh**: Unstructured Tetrahedral (Gmsh). direct `gmshToFoam` conversion.
- **Regions**:
  - `solid_chip`: Geometric source (top 350K).
  - `solid_heatsink`: Geometric sink (bottom 300K).
- **Physics**:
  - **Coupling**: `turbulentTemperatureCoupledBaffleMixed`
  - **Thermo**: `heSolidThermo`, `hConst`, `constIso` (isotropic constant).
  - **Solver**: `chtMultiRegionSimpleFoam` (Steady-State PBiCGStab / DIC).

## ⚠️ Key Settings (ESI v2312)
- **Transport**: Use `constIso` for simple solids (not `constant`).
- **Preconditioner**: Use `DIC` (Diagonal Incomplete Cholesky) for symmetric solid matrices (`h`).
- **Equation of State**: `rhoConst`.
