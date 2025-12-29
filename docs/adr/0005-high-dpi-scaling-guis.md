# ADR 0005: Resolution of High-DPI Scaling Artifacts in GUIs
*Status:* Accepted
*Date:* 2025-12-28
*Tags:* #gui, #windows, #ux, #gmsh, #high-dpi

## 1. Context & Problem Statement
When dragging a GUI window from a high-DPI display (e.g., laptop at 150%) to a standard-DPI display (e.g., monitor at 100%), "black bar" artifacts or ghosting appear. 

* *The Constraint:* Windows "DPI Virtualization" lies to the application about screen resolution to prevent legacy apps from appearing too small. During a scaling factor change (drag event), the OS blits a stretched bitmap before the application can redraw.
* *The Goal:* Force the application to be DPI-aware and eliminate rendering artifacts during mixed-DPI transitions.

## 2. Technical Decision
*Mechanism:* Force "Application" level DPI awareness to prevent OS-level bitmap stretching.
1. **Code Fix (Gmsh):** Set `General.HighResolutionGraphics` to 1 and manually handle font scaling if necessary.
2. **OS Fix:** Override High-DPI scaling behavior in Windows Properties for the Python executable (set to "Application").

*Dependencies:* Requires `gmsh` configuration before initialization.

## 3. Mathematical & Physical Implications
* *Stability:* Improves UI stability and rendering correctness across different monitor configurations.
* *Geometric Constraints:* None, though fonts may require manual bias on 4K displays.

## 4. Performance Trade-offs
* *Compute Cost:* Negligible. Forces immediate `paintEvent` triggers upon resize/drag.
* *Visual Quality:* High. Prevents the "dead pixel" / "black bar" effect.

## 5. Verification Plan
* *Sanity Check:* Drag the application window between a laptop and external monitor. Verify no black bars persist after the jump.
* *Regression:* Check text readability on 4K displays; ensure `General.FontSize` is readable.
