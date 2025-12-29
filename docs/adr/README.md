# Architectural Decision Records (ADRs)

This folder contains records of significant technical decisions made during the development of SimOps.

## ADR Template

To create a new ADR, use the following template:

# [Short Title, e.g., Implementation of Roe Solver / Switch to Voronoi Dual]
*Status:* [Proposed | Accepted | Deprecated]
*Date:* YYYY-MM-DD
*Tags:* [e.g., #numerics, #geometry, #optimization, #flux-scheme]

## 1. Context & Problem Statement
The mathematical or architectural constraint driving this decision.
* *The Constraint:* [e.g., The current central difference scheme creates spurious oscillations at shock waves (Gibbs phenomenon).]
* *The Goal:* [e.g., We need a Total Variation Diminishing (TVD) scheme to handle discontinuities.]

## 2. Technical Decision
The specific algorithm or library adopted.
* *Mechanism:* [e.g., Implementing a MUSCL reconstruction with a Minmod limiter.]
* *Dependencies:* [e.g., Requires calculating gradient vectors at cell centers.]

## 3. Mathematical & Physical Implications
Crucial for validity.
* *Conservation:* [e.g., Strictly conservative? Yes/No.]
* *Stability:* [e.g., Reduces max stable CFL from 1.0 to 0.8.]
* *Geometric Constraints:* [e.g., Requires mesh orthogonality > 0.7 or gradients become inaccurate.]

## 4. Performance Trade-offs
* *Compute Cost:* [e.g., Increases flux calculation time by 2x due to reconstruction step.]
* *Memory Cost:* [e.g., Needs to store gradient tensors for every cell.]

## 5. Verification Plan
* *Sanity Check:* [e.g., Sod Shock Tube benchmark.]
* *Regression:* [e.g., Compare residuals against the previous version on the standard nozzle test case.]
