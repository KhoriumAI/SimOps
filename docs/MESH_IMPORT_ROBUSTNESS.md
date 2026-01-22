# Robust Mesh Import Pipeline

## Overview

The SimOps CFD pipeline now includes a robust mesh import system that automatically handles .msh files from any source (GMSH GUI, external tools, different versions) and ensures compatibility with OpenFOAM's `gmshToFoam` converter.

## The Problem: GMSH Format Compatibility

### Why Native GMSH Files Fail with OpenFOAM

OpenFOAM's `gmshToFoam` converter has **strict format requirements** that often conflict with modern GMSH defaults:

| Issue | Description | Impact |
|-------|-------------|--------|
| **MSH Format Version** | GMSH 4.x (default since 2019) uses MSH 4.0/4.1 format | `gmshToFoam` fails with "bad stream" or "unknown element" errors |
| **Binary vs ASCII** | Binary format is more compact but harder to debug | Parse errors are cryptic and hard to diagnose |
| **Element Types** | Modern GMSH supports high-order elements | `gmshToFoam` only supports first-order elements (tets, hexes, prisms) |
| **Physical Groups** | GMSH requires explicit physical group tagging | Missing tags → unassigned faces → simulation failure |

### Recommended Format: MSH 2.2 ASCII

The most reliable format for `gmshToFoam` is **MSH 2.2 ASCII**:
- Widely supported across OpenFOAM versions (v7-v13, ESI, Foundation)
- Human-readable for debugging
- Compatible with first-order mesh elements
- Established since GMSH 2.5 (2011)

## The Solution: 4-Layer Import Pipeline

The new `_import_mesh_robust()` method implements a multi-layer approach:

```
┌─────────────────────────────────────────┐
│  1. FORMAT DETECTION                    │
│  → Parse .msh header                    │
│  → Extract version, format type         │
│  → Assess gmshToFoam compatibility      │
└─────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│  2. AUTO-CONVERSION (if needed)         │
│  → Detect MSH 4.x format                │
│  → Convert to MSH 2.2 using gmsh CLI    │
│  → Fallback to original if fails        │
└─────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│  3. PRIMARY: gmshToFoam                 │
│  → Native OpenFOAM converter            │
│  → Parse log for diagnostics            │
│  → Extract cell count, errors, warnings │
└─────────────────────────────────────────┘
                 ↓ (if fails)
┌─────────────────────────────────────────┐
│  4. FALLBACK: Fluent Route              │
│  → GMSH → Fluent format                 │
│  → fluent3DMeshToFoam converter         │
│  → More robust for complex meshes       │
└─────────────────────────────────────────┘
```

## Key Components

### 1. `detect_msh_format()` - Format Detection
**File**: [core/solvers/msh_utils.py](../core/solvers/msh_utils.py)

Parses the `$MeshFormat` header to extract:
```
$MeshFormat
4.1 0 8        ← version format_type size_type
$EndMeshFormat
```

Returns:
```python
{
    "version": "4.1",
    "format": "ASCII",  # or "Binary"
    "gmshToFoam_compatible": False,
    "warning": "MSH 4.1 often fails with gmshToFoam - recommend conversion to 2.2"
}
```

### 2. `convert_msh_format()` - Auto-Conversion
**File**: [core/solvers/msh_utils.py](../core/solvers/msh_utils.py)

Uses GMSH CLI to convert formats:
```bash
gmsh input.msh -format msh22 -o output.msh -v 0
```

Handles:
- WSL path translation on Windows
- Timeout protection (60s limit)
- Error message extraction

### 3. `parse_gmshToFoam_log()` - Error Diagnostics
**File**: [core/solvers/msh_utils.py](../core/solvers/msh_utils.py)

Extracts actionable information from `log.gmshToFoam`:

| Pattern | Meaning | Action |
|---------|---------|--------|
| `Unknown element type` | Unsupported mesh elements | Convert to first-order elements |
| `Bad input stream` | Format incompatibility | Use auto-conversion or Fluent route |
| `duplicate patch names` | Boundary naming conflict | Fix physical groups in GMSH |
| `total: N` | Cell count | Success metric |

### 4. `convert_via_fluent()` - Fallback Route
**File**: [core/solvers/msh_utils.py](../core/solvers/msh_utils.py)

Two-step conversion for problematic meshes:
1. GMSH → Fluent format (`.msh`)
2. `fluent3DMeshToFoam` → OpenFOAM polyMesh

Why it's more robust:
- Fluent format is simpler and more standardized
- `fluent3DMeshToFoam` has better error tolerance
- Handles high-order elements by downgrading

## Usage

### For Users Importing External Meshes

The pipeline is **fully automatic**. Just provide any `.msh` file:

```python
from core.solvers.cfd_solver import CFDSolver

solver = CFDSolver(use_wsl=True)
results = solver.run(
    mesh_file=Path("external_mesh.msh"),  # Any MSH version
    output_dir=Path("./outputs"),
    config={"inlet_velocity": [10, 0, 0]}
)
```

The solver will:
1. Detect if format is incompatible
2. Auto-convert to MSH 2.2 if needed
3. Log all conversions and warnings
4. Fall back to Fluent route if necessary

### Log Output Example

```
[Mesh Import] Importing external_mesh.msh...
[MSH Detection] Version: 4.1, Format: ASCII, Compatible: False
[Mesh Import] Detected MSH 4.1 (ASCII)
[Mesh Import] MSH 4.1 is not optimal for gmshToFoam
[Mesh Import] Auto-converting to MSH 2.2 for compatibility...
[MSH Conversion] Converting external_mesh.msh to MSH 2.2...
[MSH Conversion] Successfully converted to mesh_converted.msh
[Mesh Import] Format is now compatible with gmshToFoam
[Mesh Import] Converting mesh with gmshToFoam...
[Mesh Import] Successfully imported 45328 cells
[Mesh Import] ✓ gmshToFoam conversion successful
```

### Handling Failures

If both routes fail, you'll get a comprehensive error report:

```
Mesh import failed via all routes:
1. gmshToFoam failed: Unknown element type
2. Fluent fallback failed: Fluent format export unsupported

Detected Format: MSH 4.1 (Binary)

Troubleshooting:
- Ensure mesh file is valid (not corrupted)
- Check that mesh contains 3D volume elements (not just surfaces)
- Verify physical groups are properly defined in GMSH
- Try exporting mesh as MSH 2.2 ASCII format

Log file: ./outputs/mesh_case/log.gmshToFoam
```

## Common Scenarios

### Scenario 1: GMSH 4.x from GUI
**Problem**: User exports from GMSH 4.11 GUI with default settings (MSH 4.1)

**Solution**: Auto-converts to MSH 2.2 before gmshToFoam

### Scenario 2: Binary Format
**Problem**: Binary `.msh` is more compact but harder to debug

**Solution**:
1. Pipeline attempts conversion
2. If fails, uses Fluent route
3. Logs suggest using ASCII for better debugging

### Scenario 3: High-Order Elements
**Problem**: Mesh contains second-order tets or hexes

**Solution**: Fluent route automatically downgrades to first-order

### Scenario 4: Missing Physical Groups
**Problem**: Mesh has no boundary tags

**Solution**:
- Parser detects missing patches
- Logs actionable error: "Required patch 'inlet' not found"
- User can fix in GMSH and re-import

## Performance Impact

| Operation | Time (typical) | Notes |
|-----------|---------------|-------|
| Format detection | ~1ms | Header parse only |
| MSH 4.1 → 2.2 conversion | 0.5-5s | Depends on mesh size |
| gmshToFoam (direct) | 2-30s | Primary method |
| Fluent route (fallback) | 5-60s | Slower but more robust |

**Recommendation**: For production pipelines, pre-convert meshes to MSH 2.2 to avoid runtime overhead.

## Integration with Existing Code

### Modified Files

1. **[core/solvers/cfd_solver.py](../core/solvers/cfd_solver.py)**
   - Added `_import_mesh_robust()` method
   - Imports utilities from `msh_utils`
   - Replaces direct `gmshToFoam` call

2. **[core/solvers/msh_utils.py](../core/solvers/msh_utils.py)** (NEW)
   - Format detection logic
   - Conversion utilities
   - Log parsers
   - Fallback routes

### Backward Compatibility

The changes are **fully backward compatible**:
- MSH 2.2 meshes pass through unchanged
- No config changes required
- Existing workflows continue to work

## Testing

### Manual Testing

Test with different mesh formats:

```bash
# Test MSH 2.2 (should pass directly)
gmsh -format msh22 -o test_22.msh test.geo

# Test MSH 4.1 (should auto-convert)
gmsh -format msh41 -o test_41.msh test.geo

# Test binary (should handle gracefully)
gmsh -format msh22 -bin -o test_bin.msh test.geo
```

### Validation Checklist

- [ ] MSH 2.2 ASCII imports without conversion
- [ ] MSH 4.1 ASCII auto-converts to 2.2
- [ ] Binary formats handled (convert or fallback)
- [ ] Invalid files produce clear error messages
- [ ] Fluent fallback works when gmshToFoam fails
- [ ] Log parsing extracts errors/warnings correctly

## Future Improvements

1. **Format Detection Cache**: Cache format info to avoid re-detection
2. **Parallel Conversion**: Try both routes simultaneously for speed
3. **Native Python Parser**: Replace gmsh CLI dependency with native parser
4. **Format Auto-Selection**: Let user specify preferred conversion route

## References

- [GMSH File Format Specification](http://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format)
- [OpenFOAM gmshToFoam Documentation](https://www.openfoam.com/documentation/guides/latest/doc/guide-applications-utilities-mesh-conversion-gmshtofoam.html)
- [OpenFOAM Issue Tracker - MSH 4.x Compatibility](https://bugs.openfoam.org/view.php?id=3234)

## Summary

The robust mesh import pipeline ensures that SimOps can handle `.msh` files from any source without manual intervention. The multi-layer approach (detection → conversion → primary → fallback) provides both performance (fast path for compatible formats) and reliability (fallback for edge cases).

**Key Benefit**: Users can now import meshes from GMSH GUI, external tools, or legacy projects without worrying about format compatibility - the system handles it automatically.
