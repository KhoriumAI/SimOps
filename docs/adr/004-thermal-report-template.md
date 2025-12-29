# ADR 004: Specialized Thermal Report Template
*Status:* Accepted
*Date:* 2025-12-28
*Tags:* #reporting #thermal #pdf #validation

## 1. Context & Problem Statement
The generic PDF report generator lacked specific fields for thermal physics (e.g., Min/Max temperatures in Kelvin, material properties like Thermal Conductivity). Users needed a professional, physics-aware document for thermal simulation results.
* *The Constraint:* Reusing the structural report logic for thermal resulted in missing or incorrectly labeled fields.
* *The Goal:* Create a dedicated `ThermalPDFReportGenerator` that aligns with CalculiX thermal output and provides specific validation for thermal gradients and ranges.

## 2. Technical Decision
Developed and integrated `ThermalPDFReportGenerator` into the core simulation pipeline.
* *Mechanism:* 
    - Dedicated class in `core/reporting/thermal_report.py`.
    - Integrated into `simops_worker.py`'s `generate_report` function.
    - Added explicit Kelvin-to-Celsius conversions and pass/fail logic for thermal limits. 
    - Included material property table for Al6061 (Thermal Conductivity, Specific Heat, Density).
* *Dependencies:* `reportlab` for PDF layout.

## 3. Mathematical & Physical Implications
* *Conservation:* Not applicable.
* *Stability:* Flags unphysical results (e.g., $T < 0K$ or $\Delta T > 500K$) as PASS/FAIL/WARN.
* *Geometric Constraints:* None beyond standard component geometry.

## 4. Performance Trade-offs
* *Compute Cost:* Negligible.
* *Memory Cost:* No change.

## 5. Verification Plan
* *Sanity Check:* Regenerated Cube thermal job and verified `Cube_thermal_report.pdf` contains correct units (K and °C) and validation status.
* *Regression:* Confirmed `simops_worker.py` correctly dispatches to the new generator based on simulation type.
