# ADR 002: Physics-Based Reporting Guardrails
*Status:* Accepted
*Date:* 2025-12-28
*Tags:* #reporting #physics #validation #post-processing

## 1. Context & Problem Statement
SimOps simulations can occasionally converge mathematically but produce results that are physically nonsensical (e.g., thermal cryogenic errors or structural scaling artifacts). Users need immediate, high-confidence feedback on the validity of the results within the generated PDF reports.
* *The Constraint:* Manual inspection of residuals and stress magnitudes is time-consuming and prone to human error.
* *The Goal:* Automate the "sanity check" process by embedding physics-based pass/fail criteria directly into the simulation reports.

## 2. Technical Decision
Implemented automated validation checks for structural and thermal physics.
* *Mechanism:* 
    - **Structural:** Automated safety factor calculation ($yield\_strength / max\_stress$) and displacement-to-part-size ratio check ($max\_disp < 0.1\% \text{ characteristic size}$).
    - **Thermal:** Range checks ($T \in [0, 1000]K$) and gradient checks ($\Delta T < 500K$).
* *Dependencies:* `reportlab` for PDF generation, `NumPy` for post-simulation metrics analysis.

## 3. Mathematical & Physical Implications
* *Conservation:* Not applicable.
* *Stability:* Improves the perceived stability of the platform by flagging numerical "explosions" as warnings or failures rather than presenting them as legitimate data.
* *Geometric Constraints:* Displacement checks require an accurate `characteristic_size_mm` estimate for the component.

## 4. Performance Trade-offs
* *Compute Cost:* Negligible. Checks are performed on scalar result values.
* *Memory Cost:* No change.

## 5. Verification Plan
* *Sanity Check:* Verified that the Cube test case passes with ✓ PASS indicators for both stress and displacement.
* *Regression:* Verified that reports are generated for batch runs without crashing.
