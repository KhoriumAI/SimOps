# ADR 0003: Standardization on mm-tonne-s Unit System
*Status:* Accepted
*Date:* 2025-12-28
*Tags:* #physics, #units, #simulation, #calculix, #openfoam

## 1. Context & Problem Statement
Inconsistent unit scaling across different stages of the simulation pipeline (CAD import, meshing, solving, and reporting) led to non-physical results. Specifically, convection coefficients and material properties were being scaled improperly, causing benchmark failures.

* *The Constraint:* CAD files are typically in mm, but solvers like CalculiX and OpenFOAM often expect consistent SI units or a specific derived set.
* *The Goal:* Establish a single, "absolute" unit system across the entire pipeline to eliminate redundant and conflicting scaling factors.

## 2. Technical Decision
*Mechanism:* Adoption of the **mm-tonne-s** (millimeter, tonne, second) unit system.
* *Length:* mm
* *Mass:* tonne (1000 kg)
* *Time:* s
* *Force:* N ($1\text{ N} = 1\text{ tonne} \cdot 1\text{ mm/s}^2$)
* *Pressure/Stress:* MPa ($1\text{ MPa} = 1\text{ N/mm}^2$)
* *Energy:* mJ ($1\text{ mJ} = 1\text{ N} \cdot \text{mm}$)
* *Power:* mW ($1\text{ mW} = 1\text{ mJ/s}$)
* *Density:* $\text{tonne/mm}^3$ (e.g., Steel $\approx 7.8 \times 10^{-9}$)

*Dependencies:* Requires `CalculiXAdapter` to strictly enforce these conversions for material properties ($h$, $k$, $C_p$, $\rho$).

## 3. Mathematical & Physical Implications
* *Conservation:* Yes. Dimensional consistency is maintained.
* *Stability:* Solvers are numerically more stable when variables (like coordinates and stresses) are closer to order 1-100, which this system facilitates for typical mechanical parts.
* *Geometric Constraints:* None, but requires all input CAD to be treated as mm.

## 4. Performance Trade-offs
* *Compute Cost:* Zero. Initial scaling is a one-time operation during INP generation.
* *Memory Cost:* Zero.

## 5. Verification Plan
* *Sanity Check:* Verify that a $25.0 \text{ W/m}^2\text{K}$ convection coefficient equates to $0.025$ in the INP.
* *Regression:* Run thermal benchmark with known temperature distribution and verify results within 1%.
