# ADR 001: Adoption of mm-tonne-s Unit System
*Status:* Accepted
*Date:* 2025-12-28
*Tags:* #physics, #simulation, #units

## 1. Context & Problem Statement
SimOps handles mechanical and thermal simulations where CAD geometry is typically provided in millimeters. Standard SI units (meters) lead to extremely small values for dimensions and potentially large values for stress, leading to numerical precision issues or inconvenient manual conversions.
* *The Constraint:* The solver (CalculiX) expects consistent units. Mixing meters and millimeters results in incorrect physical behavior.
* *The Goal:* Establish a consistent unit system that works natively with CAD (mm) and provides physically realistic outputs (MPa, Watts).

## 2. Technical Decision
We adopt the **mm-tonne-s (mTs)** unit system across the entire pipeline.
* *Mechanism:*
    * Length: mm
    * Mass: tonne (1,000 kg)
    * Time: s
    * Force: Newton (N)
    * Stress/Elastic Modulus: MPa ($N/mm^2$)
    * Density: $tonne/mm^3$ (e.g., Steel $\approx 7.85 \times 10^{-9}$)
    * Thermal Conductivity: $W/(mm \cdot K)$ (e.g., $50 W/mK \rightarrow 0.05 W/mmK$)
    * Specific Heat: $J/(tonne \cdot K)$ (e.g., $450 J/kgK \rightarrow 4.5 \times 10^5 J/tonneK$)
* *Dependencies:* Material libraries must be hydrated with values converted to mTs before INP generation.

## 3. Mathematical & Physical Implications
* *Conservation:* Yes.
* *Stability:* Improved numerical stability for small-scale mechanical components.
* *Geometric Constraints:* CAD imports must be verified as mm-scaled.

## 4. Performance Trade-offs
* *Compute Cost:* Negligible (scaling operations are trivial).
* *Memory Cost:* None.

## 5. Verification Plan
* *Sanity Check:* Compare a simple cantilever beam deflection against analytical solutions in MPa.
* *Regression:* Verify `thermal_verification_fix.py` passes with mTs scaling.
