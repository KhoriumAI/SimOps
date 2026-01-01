# ADR 0007: Robust CAD Loading Strategy (Preview Fallback)

*Status:* Accepted
*Date:* 2025-12-31
*Tags:* #geometry, #gmsh, #robustness, #cad

## 1. Context & Problem Statement
The current CAD preview mechanism uses Gmsh's OpenCASCADE (OCC) kernel to load STEP files and generate a coarse tessellation. Certain malformed or complex CAD files cause the OCC kernel to hang or crash during wire repair operations (`occ.synchronize`). 

* **The Constraint:** Gmsh's OCC healing/repair options, while helpful for most files, can enter infinite loops or trigger fatal errors on geometry with self-intersecting wires or degenerate surfaces.
* **The Goal:** Ensure the 3D viewer remains responsive and provides a valid (even if un-healed) preview for all files, avoiding application-wide hangs.

## 2. Technical Decision
We implement a **Three-Stage Fallback Strategy** in the `PreviewWorker` subprocess.

1. **Stage 1 (Standard):** Attempt load with OCC Healing enabled (`OCCFixSmallEdges`, `OCCAutoFix`, etc.) using `gmsh.open()`.
2. **Stage 2 (Raw Load):** If Stage 1 fails, perform a **Full Gmsh Reset** (`finalize` -> `initialize`) and retry with all healing options **disabled** using `gmsh.open()`.
3. **Stage 3 (Deep Import):** If Stage 2 resulted in zero entities (common with malformed files) or fails, perform another reset and use `gmsh.model.occ.importShapes()` directly. This bypasses high-level wrappers and is more robust for fragmented assembly files.

**Subprocess Isolation & Timeout:**
The entire process is isolated in a subprocess with a strict **60-second timeout**. We use **non-blocking I/O (threaded queue)** to read the subprocess output, ensuring the timeout check is never blocked by a pending `readline()` call from a hung subprocess.

## 3. Mathematical & Physical Implications
* **Conservation:** Not applicable (visualization only).
* **Stability:** Significantly increases UI stability by isolating geometry engine crashes.
* **Geometric Constraints:** Stage 2 (Raw Load) might display visual artifacts (missing surfaces, disjointed edges) because it bypasses healing. This is acceptable for a "Preview" phase.

## 4. Performance Trade-offs
* **Compute Cost:** Failed attempts add ~2-5 seconds overhead before falling back to the next stage.
* **Memory Cost:** Minor; isolated to the short-lived subprocess.

## 5. Verification Plan
* **Sanity Check:** Use the problematic file `00010009_..._step_000 (1).step` which previously caused hangs.
* **Regression:** Ensure clean STEP files still use Stage 1 and look correct.
