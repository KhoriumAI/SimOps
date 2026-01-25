# CalculiX Simulation Quality & Validation Roadmap

**Objective**: Elevate SimOps thermal analysis from "Debugging/Toy" to "Production-Grade Engineering" using CalculiX, ensuring accuracy and reliability without falling back to Python solvers.

## Executive Summary
We will execute four parallel tracks to harden the physics engine and validate results.

---

## Track 1: Advanced Physics (Beyond Conduction)
**Goal**: Move from simple conduction to realistic thermal environments.
*   **1.1 Convection & Radiation**: Implement `*FILM` and `*RADIATE` boundary conditions in `CalculiXAdapter`.
*   **1.2 Transient Analysis**: Enable time-dependent heat transfer (`*HEAT TRANSFER, DIRECT`) to capture warm-up curves and thermal shock.
*   **1.3 Nonlinearity**: Enable `*STEP, NLGEOM` to handle large deformations or contact if we expand to thermo-mechanical.

## Track 2: Material Fidelity & Library
**Goal**: Replace "Generic Aluminum" with accurate, temperature-dependent materials.
*   **2.1 Temperature-Dependent Props**: Support `*CONDUCTIVITY` and `*SPECIFIC HEAT` as tables vs temperature.
*   **2.2 Material Library System**: specific JSON/YAML definitions for Al6061, Copper C110, SS304, etc.
*   **2.3 Anisotropy**: Support for PCBs (orthotropic interactions) which conduct well in-plane but poorly thru-plane.

## Track 3: Automated V&V (Verification & Validation)
**Goal**: "Trust but Verify" - Automated checks that physics are correct.
*   **3.1 Analytical Benchmarks**: A suite of unit tests (e.g., "1D Rod Conduction", "Sphere Cooling") where the exact answer is known mathmatically.
*   **3.2 Energy Balance Check**: Post-process `.dat` files to ensure Heat In = Heat Out + Delta Internal Energy.
*   **3.3 NAFEMS Benchmarks**: Run standard industry thermal benchmarks (e.g., T1, T2) as part of CI/CD.

## Track 4: Mesh Independence & Convergence
**Goal**: Ensure results are driven by physics, not mesh size.
*   **4.1 Grid Convergence Index (GCI)**: Automatically run simulation at 3 mesh densities to error-bar the result.
*   **4.2 Second-Order Elements**: Force `C3D10` (Tet10) instead of `C3D4` (Tet4) for quadratic interpolation accuracy (crucial for curves).
*   **4.3 Residual Monitoring**: Parse CalculiX stdout to ensure thermal residuals drop below `1e-6`.

---

## Implementation Priority
1.  **Track 3 (V&V)** - Immediate. Build the ruler before building the house.
2.  **Track 4 (Mesh)** - High. Switch to Tet10 immediately for accuracy.
3.  **Track 1 (Physics)** - Medium. Add Convection next.
4.  **Track 2 (Materials)** - Ongoing.
