# ADR 001: Robust FRD Parser for Windows CalculiX
*Status:* Accepted
*Date:* 2025-12-28
*Tags:* #numerics #calculix #parsing #bugfix

## 1. Context & Problem Statement
Structural simulation results were displaying a consistent $10^5$ scaling error (e.g., 95mm displacement instead of 3nm). This led to unphysical "quiet failures" where simulations converged but produced garbage data.
* *The Constraint:* The Windows binary of CalculiX 2.23 outputs `.frd` files using 13-character widths for negative numbers with 3-digit exponents (e.g., `-1.2345E+004`), which differs from the standard 12-char fixed-width expectation. 
* *The Goal:* Establish a parsing mechanism that reliably handles non-standard column widths and prevents the merger of Node IDs with scientific values.

## 2. Technical Decision
Adopted a robust regex-based parser with explicit column slicing.
* *Mechanism:* Slicing the input line at `line[13:]` to strictly isolate the value columns from the Node ID column, followed by a specific regex `r'[-+]?\d*\.\d+[Ee][-+]+\d{3}'` to capture 3-digit exponents without greedily consuming adjacent data.
* *Dependencies:* `re` module in Python.

## 3. Mathematical & Physical Implications
* *Conservation:* Not applicable (post-processing only).
* *Stability:* Resolves catastrophic numerical interpretation errors ($10^{50}$ magnitude spikes) caused by misaligned column reading.
* *Geometric Constraints:* Robust across all element types (Tet4, Tet10, C3D8) supported by CalculiX.

## 4. Performance Trade-offs
* *Compute Cost:* Negligible increase compared to fixed-width slicing; regex overhead is minimal on typical FEA mesh sizes (~10k-100k nodes).
* *Memory Cost:* No change.

## 5. Verification Plan
* *Sanity Check:* Standard Cantilever Beam benchmark (Steel, 4000N load) verified to match analytical results.
* *Regression:* `scripts/run_batch_verification.py` confirmed consistent results across Cube, Cylinder, and L-bracket geometries.
