# ADR 003: Simulation Validation (Traffic Light System)
*Status:* Accepted
*Date:* 2025-12-28
*Tags:* #ci, #validation, #automation

## 1. Context & Problem Statement
Manual inspection of hundreds of simulation results is unscaleable. We need a way to automatically flag "junk" runs while highlighting successful ones for review.
* *The Constraint:* Physical errors (explosions, unrealistic temperatures) must be auto-detected.
* *The Goal:* Create a three-tier validation system: Red (Hard Fail), Yellow (Visual Review), Green (Pass).

## 2. Technical Decision
Implement `validate_batch.py` as a post-processing guardrail.
* *Mechanism:*
    * **Red Filter (Auto-Fail):** Scans logs for "bounding", "fatal", "NAN", or $dt < 1e-5$. Flags values like $T > 5000K$ or $U > Mach 1$.
    * **Yellow Filter (Visual Audit):** Generates a contact sheet (HTML) of critical thumbnails (velocity, stress, residuals).
    * **Green (Passed):** Results that pass all physics checks and are visually approved.
* *Dependencies:* Requires `result.json` and standardized visualization outputs.

## 3. Mathematical & Physical Implications
* *Conservation:* Ensures mass/energy conservation by checking residual convergence trends.
* *Stability:* Flags numerically unstable runs before they are used for downstream analysis.

## 4. Performance Trade-offs
* *Compute Cost:* Fast log parsing; image extraction adds slight overhead.
* *Memory Cost:* Low.

## 5. Verification Plan
* *Sanity Check:* Run `validate_batch.py` on a known failed case and ensure it's moved to `_FAILED`.
* *Regression:* Verify the summary HTML correctly displays all thumbnails for a successful sweep.
