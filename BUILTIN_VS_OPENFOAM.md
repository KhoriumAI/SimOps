# Builtin Solver vs OpenFOAM - Simple Explanation

## TL;DR

**Builtin Solver:**
- ✅ Works immediately, no installation needed
- ✅ Fast setup (Python + Gmsh only)
- ⚠️ Basic heat conduction only
- 📦 Good for: Quick thermal checks, prototyping, development

**OpenFOAM Solver:**
- ⚠️ Requires WSL (Windows) or Linux + OpenFOAM installation
- ⚠️ Slower setup (needs template files, case setup)
- ✅ Full CFD capabilities (fluid flow + heat transfer)
- 🎯 Good for: Production analysis, realistic cooling scenarios, publication-quality results

---

## What is the Builtin Solver?

The **builtin solver** is a custom Python-based thermal solver built into SimOps.

### How It Works

1. **Loads your mesh** using Gmsh (reads .msh files)
2. **Builds a stiffness matrix** using finite element assembly
   - For each tetrahedral element, calculates heat conduction contribution
   - Uses vectorized NumPy operations for speed
3. **Solves the heat equation:** `∇·(k∇T) = 0`
   - Where k = thermal conductivity, T = temperature
   - Uses SciPy sparse matrix solver (`spsolve`)
4. **Applies boundary conditions:**
   - Hot surface: Fixed temperature (heat source)
   - Cold surface: Ambient temperature
   - Other surfaces: Natural convection

### What It Can Do

- ✅ Steady-state heat conduction in solids
- ✅ Material thermal properties (Aluminum, Copper, Steel, etc.)
- ✅ Temperature distribution visualization
- ✅ Handles tetrahedral meshes
- ✅ Fast (~5-30 seconds depending on mesh size)

### What It CANNOT Do

- ❌ Fluid flow (air movement around parts)
- ❌ Convective cooling (moving air carrying heat away)
- ❌ Multi-region coupling (fluid + solid interaction)
- ❌ Turbulence modeling
- ❌ Transient/time-dependent analysis (well, it tries but it's simplified)

### Code Location

**File:** [simops_pipeline.py:78-428](simops_pipeline.py#L78-L428)

**Key Method:** `ThermalSolver.solve(mesh_file)`

**Algorithm:**
```python
# Simplified pseudocode
for each tetrahedral element:
    compute_element_stiffness_matrix()
    add_to_global_matrix()

apply_boundary_conditions()
temperature = solve_linear_system(K * T = F)
```

---

## What is OpenFOAM?

**OpenFOAM** (Open Field Operation and Manipulation) is an industry-standard open-source CFD (Computational Fluid Dynamics) suite.

### How It Works

1. **Copies a case template** from `simops/templates/Golden_Thermal_Case/`
2. **Parameterizes the case** with your settings:
   - Power input (Watts) → heat flux on chip
   - Inlet air temperature and velocity
   - Material properties for solid regions
3. **Runs `chtMultiRegionFoam`** solver via WSL:
   - Solves coupled heat transfer between solid (chip/heatsink) and fluid (air)
   - Iterates to convergence (typically 200-500 iterations)
4. **Exports results** to VTK format for visualization

### What It Can Do

- ✅ **Conjugate Heat Transfer (CHT):** Solid + fluid coupled analysis
- ✅ **Fluid flow:** Models air movement around parts
- ✅ **Forced convection:** Fans, airflow velocities
- ✅ **Natural convection:** Buoyancy-driven cooling
- ✅ **Turbulence:** Laminar and turbulent flow regimes
- ✅ **Multi-region:** Different materials (chip, heatsink, PCB)
- ✅ **Transient:** Time-dependent heating/cooling cycles

### What It Requires

**On Windows:**
- WSL2 (Windows Subsystem for Linux)
- OpenFOAM installed in WSL (typically version 2312)
- Template files must be accessible from WSL paths

**On Linux:**
- OpenFOAM installed (`apt install openfoam2312` or compile from source)

### Code Location

**Orchestrator:** [simops_pipeline.py:807-865](simops_pipeline.py#L807-L865)

**Runner:** [tools/thermal_job_runner.py:368-468](tools/thermal_job_runner.py#L368-L468)

**Case Generator:** [tools/thermal_job_runner.py:178-350](tools/thermal_job_runner.py#L178-L350)

---

## Side-by-Side Comparison

| Feature | Builtin Solver | OpenFOAM |
|---------|----------------|----------|
| **Installation** | Python + pip install | WSL + OpenFOAM (complex) |
| **Speed** | 5-30 seconds | 2-10 minutes |
| **Accuracy** | Good for conduction | Excellent for CHT |
| **Physics** | Heat conduction only | Full CFD + heat transfer |
| **Air cooling** | Approximated (convection coeff) | Simulated (actual flow) |
| **Mesh size** | 10K-100K elements | 50K-500K+ elements |
| **Use case** | Quick checks, dev work | Production, validation |
| **Results** | Temperature field | Temp + velocity + pressure |

---

## When to Use Which?

### Use **Builtin** When:

1. **Rapid prototyping** - "Does this heatsink design work?"
2. **Development testing** - Testing the pipeline without OpenFOAM
3. **Small geometries** - Simple parts with < 50K elements
4. **Conduction-dominated** - Solid materials with minimal airflow
5. **No WSL available** - Running on systems without Linux/WSL

### Use **OpenFOAM** When:

1. **Production analysis** - Final design validation
2. **Active cooling** - Fans, forced airflow, heat sinks with fins
3. **Multi-physics** - Need fluid dynamics + heat transfer coupling
4. **Publication quality** - Results for papers, reports, datasheets
5. **Complex scenarios** - Electronics enclosures, PCB assemblies

---

## Example: Cooling a 50W Chip

### Builtin Solver Approach:

```python
config = {
    "solver": "builtin",
    "heat_source_power": 50.0,          # 50W chip
    "ambient_temperature": 293.15,       # 20°C ambient
    "convection_coefficient": 20.0,      # Guess at air cooling
    "material": "Aluminum"               # Heatsink material
}
```

**Result:** "The chip will reach ~75°C"

**Assumption:** You guessed the convection coefficient (h = 20 W/m²K). But is that right? Depends on airflow!

### OpenFOAM Approach:

```python
config = {
    "solver": "openfoam",
    "heat_source_power": 50.0,          # 50W chip
    "inlet_temp_k": 293.15,              # 20°C inlet air
    "air_velocity_ms": 2.0,              # 2 m/s fan speed
    "material": "Aluminum"               # Heatsink material
}
```

**Result:** "The chip reaches 68°C with this airflow pattern [shows velocity field]"

**Advantage:** Actually simulates the air movement, no guessing h-value. Shows hot spots, flow stagnation zones.

---

## Technical Details: What the Builtin Solver Actually Does

### Step 1: Load Mesh
Uses Gmsh Python API to read the `.msh` file and extract:
- Node coordinates (x, y, z positions)
- Element connectivity (which nodes form each tetrahedron)

### Step 2: Assemble Global Stiffness Matrix
For each tetrahedral element:

```python
def compute_element_stiffness(node_coords, k_thermal):
    # Calculate element volume
    V = element_volume(node_coords)

    # Shape function derivatives (∂N/∂x, ∂N/∂y, ∂N/∂z)
    dN = shape_function_derivatives(node_coords)

    # Element stiffness: K_e = k * V * (B^T * B)
    K_e = k_thermal * V * (dN @ dN.T)

    return K_e
```

Assemble into sparse global matrix: **K** (size: N_nodes × N_nodes)

### Step 3: Apply Boundary Conditions

Detect surfaces and apply:
- **Hot surface:** T = T_hot (fixed temperature)
- **Cold surface:** T = T_ambient
- **Other surfaces:** Natural convection (simplified as Robin BC)

Modifies the system: **K @ T = F**

### Step 4: Solve Linear System

```python
from scipy.sparse.linalg import spsolve
temperature = spsolve(K, F)  # Solve K*T = F
```

Uses direct sparse solver (UMFPACK/SuperLU) - very fast for moderate sizes.

### Step 5: Post-Process

- Extract min/max temperatures
- Export to VTK for visualization
- Generate PNG temperature contour plot

---

## Why OpenFOAM is More Complex

OpenFOAM doesn't just solve one equation - it solves **multiple coupled equations**:

1. **Navier-Stokes** (fluid momentum): ∂u/∂t + (u·∇)u = -∇p + ν∇²u
2. **Continuity** (mass conservation): ∇·u = 0
3. **Energy** (heat transfer): ∂T/∂t + u·∇T = α∇²T
4. **Solid conduction:** ∂T/∂t = α_solid∇²T

These are **coupled** at the fluid-solid interface:
- Temperature must be continuous
- Heat flux must be continuous: k_fluid(∂T/∂n) = k_solid(∂T/∂n)

**Iterative Solution:**
1. Solve momentum → get velocity field
2. Solve energy in fluid → get fluid temperature
3. Solve energy in solid → get solid temperature
4. Check if interface conditions satisfied? No → iterate again
5. Repeat until converged (residuals < tolerance)

This is why OpenFOAM takes 200-500 iterations and several minutes.

---

## Summary for Your Project

**Current State:**
- Frontend automatically selects builtin if OpenFOAM not available ✅
- Backend checks OpenFOAM on startup ✅
- Integration tests validate both solvers ✅

**Recommendation:**
- **Default to builtin** for fast iteration during development
- **Switch to OpenFOAM** for final validation or when you need CFD
- Use `solver: "builtin"` in your config unless you specifically need fluid flow analysis

**User Impact:**
- Users can now run thermal simulations immediately without installing WSL/OpenFOAM
- When OpenFOAM is available, they get more accurate results with airflow
- Console messages clearly indicate which solver is being used
