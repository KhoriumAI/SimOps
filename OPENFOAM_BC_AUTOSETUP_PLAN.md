# OpenFOAM Boundary Condition Auto-Setup Implementation Plan

## Executive Summary

**Goal**: Automatically detect mesh geometry and apply correct hot/cold boundary conditions to OpenFOAM thermal simulations, eliminating the current single-patch limitation.

**Current Problem**:
- gmshToFoam creates only 1 boundary patch from most meshes
- Cannot establish thermal gradient with single boundary
- Built-in solver works (uses geometric detection), OpenFOAM fails

**Solution Architecture**:
1. Read OpenFOAM mesh and analyze geometry of each boundary patch
2. Classify patches as hot/cold based on user config (`hot_wall_face`)
3. Write boundary conditions with patch-specific temperatures
4. Support multiple patches with intelligent defaults

---

## Current State Analysis

### What Works ✅
- **Built-in solver**: Uses `_apply_boundary_conditions()` to geometrically identify hot/cold regions based on Z-coordinates (or X/Y based on config)
- **Mesh reading**: gmshToFoam successfully converts Gmsh → OpenFOAM
- **Solver execution**: laplacianFoam runs without errors

### What's Broken ❌
- **Single patch limitation**: Most meshes produce 1 patch (entire external surface)
- **Hard-coded patch names**: Code looks for "heatsink_bottom" that doesn't exist
- **No geometric analysis**: Doesn't analyze where patches are located
- **Wildcard BCs**: Using ".*" catches all patches before specific ones

### File Locations
- BC writer: `simops_pipeline.py` lines 688-765
- Built-in solver geometric BC: `simops_pipeline.py` lines 399-460
- OpenFOAM runner: `simops_pipeline.py` lines 501-806

---

## Implementation Plan

### Phase 1: Patch Geometry Analysis (Foundation)

**File**: New file `core/openfoam/boundary_analyzer.py`

**Purpose**: Read OpenFOAM mesh and determine geometric properties of each patch

**Implementation**:

```python
class BoundaryPatchAnalyzer:
    """
    Analyze OpenFOAM polyMesh boundary patches to determine:
    - Spatial location (min/max X/Y/Z)
    - Surface area
    - Surface normal direction
    - Centroid
    """

    def __init__(self, case_dir: Path):
        self.case_dir = case_dir
        self.patches = {}

    def analyze_patches(self) -> Dict[str, PatchGeometry]:
        """
        Read OpenFOAM polyMesh and analyze each boundary patch

        Returns:
            Dict mapping patch name to geometric properties
        """
        # Read boundary file to get patch names and face ranges
        boundary_data = self._read_boundary_file()

        # Read points file to get node coordinates
        points = self._read_points_file()

        # Read faces file to get face connectivity
        faces = self._read_faces_file()

        # Read owner file to map faces to cells
        owner = self._read_owner_file()

        # For each patch, analyze its geometry
        for patch_name, patch_info in boundary_data.items():
            start_face = patch_info['startFace']
            n_faces = patch_info['nFaces']

            # Get faces belonging to this patch
            patch_faces = faces[start_face:start_face + n_faces]

            # Extract points from faces
            patch_points = self._extract_patch_points(patch_faces, points)

            # Compute geometric properties
            geometry = self._compute_geometry(patch_points, patch_faces)

            self.patches[patch_name] = PatchGeometry(
                name=patch_name,
                centroid=geometry['centroid'],
                bounds=geometry['bounds'],
                area=geometry['area'],
                normal=geometry['normal'],
                face_count=n_faces
            )

        return self.patches

    def _compute_geometry(self, points: np.ndarray, faces: List) -> Dict:
        """Compute centroid, bounds, area, normal from patch points/faces"""

        centroid = np.mean(points, axis=0)

        bounds = {
            'x': (points[:, 0].min(), points[:, 0].max()),
            'y': (points[:, 1].min(), points[:, 1].max()),
            'z': (points[:, 2].min(), points[:, 2].max())
        }

        # Compute average surface normal using cross products
        normals = []
        total_area = 0.0

        for face in faces:
            if len(face) >= 3:
                # Triangle approximation for area/normal
                p0, p1, p2 = points[face[0]], points[face[1]], points[face[2]]
                v1 = p1 - p0
                v2 = p2 - p0
                n = np.cross(v1, v2)
                area = 0.5 * np.linalg.norm(n)

                if area > 1e-12:
                    normals.append(n / (2 * area))  # Normalized
                    total_area += area

        avg_normal = np.mean(normals, axis=0)
        avg_normal = avg_normal / np.linalg.norm(avg_normal)

        return {
            'centroid': centroid,
            'bounds': bounds,
            'area': total_area,
            'normal': avg_normal
        }
```

**Deliverables**:
- [x] Create `core/openfoam/` directory
- [ ] Implement `boundary_analyzer.py` with patch reading
- [ ] Parse OpenFOAM binary/ASCII formats for points, faces, owner
- [ ] Compute patch geometry (centroid, bounds, normal, area)
- [ ] Unit tests with known meshes

**Timeline**: 2-3 days

---

### Phase 2: Boundary Condition Assignment (Intelligence)

**File**: New file `core/openfoam/bc_strategy.py`

**Purpose**: Given patch geometries and user config, assign appropriate BCs

**Implementation**:

```python
class BCAssignmentStrategy:
    """
    Determine which patches should be hot/cold based on:
    1. User config (hot_wall_face: z_min, z_max, x_min, etc.)
    2. Patch geometry analysis
    3. Thermal problem physics
    """

    def __init__(self, config: SimOpsConfig):
        self.config = config

    def assign_boundary_conditions(
        self,
        patches: Dict[str, PatchGeometry]
    ) -> Dict[str, BCSpec]:
        """
        Assign temperature BCs to each patch

        Args:
            patches: Patch geometries from BoundaryPatchAnalyzer

        Returns:
            Dict mapping patch name to BC specification
        """
        bc_assignments = {}

        # Parse user config for hot face direction
        hot_face_dir = self._parse_hot_face_config()

        # Strategy 1: Single patch case
        if len(patches) == 1:
            # Apply gradient across the single patch based on geometry
            return self._handle_single_patch(patches, hot_face_dir)

        # Strategy 2: Multi-patch case
        # Classify each patch as hot/cold/adiabatic based on location
        for patch_name, geom in patches.items():
            bc_type = self._classify_patch(geom, hot_face_dir)

            if bc_type == 'hot':
                bc_assignments[patch_name] = BCSpec(
                    type='fixedValue',
                    value=self.config.heat_source_temperature
                )
            elif bc_type == 'cold':
                bc_assignments[patch_name] = BCSpec(
                    type='fixedValue',
                    value=self.config.ambient_temperature
                )
            else:  # adiabatic
                bc_assignments[patch_name] = BCSpec(
                    type='zeroGradient'
                )

        return bc_assignments

    def _classify_patch(
        self,
        geom: PatchGeometry,
        hot_face_dir: Dict
    ) -> str:
        """
        Classify a patch as 'hot', 'cold', or 'adiabatic'
        based on its location relative to hot_face_dir
        """
        axis = hot_face_dir['axis']  # 'x', 'y', or 'z'
        is_min = hot_face_dir['is_min']  # True for x_min, False for x_max

        # Get patch location along the specified axis
        bounds = geom.bounds[axis]
        patch_center = geom.centroid[{'x': 0, 'y': 1, 'z': 2}[axis]]

        # Get overall mesh bounds (need to compute this from all patches)
        # For now, use patch bounds

        # Classify based on position
        # If patch is at the hot face side, it's hot
        # If patch is at the opposite side, it's cold
        # Otherwise, adiabatic (no heat flux)

        # TODO: Implement robust classification logic
        # considering tolerances and multiple patches

        return 'adiabatic'  # Default

    def _handle_single_patch(
        self,
        patches: Dict[str, PatchGeometry],
        hot_face_dir: Dict
    ) -> Dict[str, BCSpec]:
        """
        Special handling for single-patch case

        Option A: Use mixed BC with spatial variation
        Option B: Split patch into regions (requires mesh modification)
        Option C: Use funkySetFields or groovyBC for spatially varying BC
        """
        patch_name = list(patches.keys())[0]

        # For now, use 'mixed' BC with average temperature
        # This won't create strong gradient but better than uniform
        T_avg = (self.config.heat_source_temperature +
                 self.config.ambient_temperature) / 2

        return {
            patch_name: BCSpec(
                type='mixed',
                refValue=self.config.heat_source_temperature,
                refGradient=0.0,
                valueFraction=0.5,
                comments='Single patch - using mixed BC'
            )
        }
```

**Advanced Features**:
- [ ] Support for multiple hot/cold regions
- [ ] Automatic adiabatic wall detection (side walls)
- [ ] Convective BC option for ambient surfaces
- [ ] Contact resistance between patches (TIE)

**Deliverables**:
- [ ] Implement `bc_strategy.py` with classification logic
- [ ] Handle single-patch case (mixed BC or warn user)
- [ ] Handle multi-patch case with geometric classification
- [ ] Configuration options for BC strategy (conservative/aggressive)
- [ ] Unit tests with various geometries

**Timeline**: 3-4 days

---

### Phase 3: Integration with OpenFOAMRunner (Deployment)

**File**: Modify `simops_pipeline.py` OpenFOAMRunner class

**Changes Required**:

```python
class OpenFOAMRunner:
    def solve(self, mesh_file: str, output_dir: str) -> Dict:
        # ... existing setup code ...

        # Convert mesh
        self._run_foam_command(f"gmshToFoam {foam_mesh} -case {foam_case}")

        # NEW: Analyze boundary patches
        from core.openfoam.boundary_analyzer import BoundaryPatchAnalyzer
        from core.openfoam.bc_strategy import BCAssignmentStrategy

        analyzer = BoundaryPatchAnalyzer(case_dir)
        patches = analyzer.analyze_patches()

        self._log(f"  Analyzed {len(patches)} boundary patches:")
        for name, geom in patches.items():
            self._log(f"    - {name}: centroid={geom.centroid}, area={geom.area:.2f}")

        # NEW: Assign BCs intelligently
        bc_strategy = BCAssignmentStrategy(self.config)
        bc_assignments = bc_strategy.assign_boundary_conditions(patches)

        self._log("  Assigned boundary conditions:")
        for name, bc in bc_assignments.items():
            self._log(f"    - {name}: {bc.type} = {bc.value}K")

        # Write BCs
        self._write_boundary_conditions_v2(case_dir, bc_assignments)

        # ... continue with solver run ...

    def _write_boundary_conditions_v2(
        self,
        case_dir: Path,
        bc_assignments: Dict[str, BCSpec]
    ):
        """
        Write OpenFOAM boundary conditions file based on analyzed patches
        """
        amb = self.config.ambient_temperature

        # Build boundary field entries from assignments
        boundary_entries = []

        for patch_name, bc_spec in bc_assignments.items():
            if bc_spec.type == 'fixedValue':
                boundary_entries.append(f'''    {patch_name}
    {{
        type            fixedValue;
        value           uniform {bc_spec.value};
    }}''')
            elif bc_spec.type == 'mixed':
                boundary_entries.append(f'''    {patch_name}
    {{
        type            mixed;
        refValue        uniform {bc_spec.refValue};
        refGradient     uniform {bc_spec.refGradient};
        valueFraction   uniform {bc_spec.valueFraction};
    }}''')
            elif bc_spec.type == 'zeroGradient':
                boundary_entries.append(f'''    {patch_name}
    {{
        type            zeroGradient;
    }}''')

        content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    "0";
    object      T;
}}
dimensions      [0 0 0 1 0 0 0];
internalField   uniform {amb};
boundaryField
{{
{chr(10).join(boundary_entries)}
}}
"""
        with open(case_dir / "0" / "T", "wb") as f:
            f.write(content.strip().replace("\r\n", "\n").encode("utf-8") + b"\n")
```

**Deliverables**:
- [ ] Integrate analyzer and strategy into OpenFOAMRunner
- [ ] Update `_write_boundary_conditions()` to use new system
- [ ] Add logging for patch analysis and BC assignment
- [ ] Backward compatibility check
- [ ] Integration tests with real meshes

**Timeline**: 2 days

---

### Phase 4: Mesh Enhancement (Optional)

**Problem**: Many CAD exports produce single-surface meshes

**Solution**: Automatically split boundary into multiple patches

**File**: New file `core/openfoam/patch_splitter.py`

**Implementation**:

```python
class BoundaryPatchSplitter:
    """
    Split a single OpenFOAM boundary patch into multiple patches
    based on geometric criteria (e.g., split by Z-coordinate)
    """

    def split_patch_by_geometry(
        self,
        case_dir: Path,
        patch_name: str,
        split_axis: str = 'z',
        n_regions: int = 2
    ) -> List[str]:
        """
        Split a patch into N regions along specified axis

        Returns:
            List of new patch names created
        """
        # Read mesh
        # Identify faces in patch
        # Group faces by coordinate ranges
        # Create new patch entries in boundary file
        # Update face labels
        # Write modified mesh

        pass
```

**Use Case**:
```python
# If analyzer detects single patch
if len(patches) == 1:
    splitter = BoundaryPatchSplitter()
    new_patches = splitter.split_patch_by_geometry(
        case_dir,
        'patch0',
        split_axis='z',
        n_regions=2
    )
    # Re-run analysis on split patches
```

**Deliverables**:
- [ ] Implement patch splitting algorithm
- [ ] Modify OpenFOAM boundary file in-place
- [ ] Update face ownership
- [ ] Test with various geometries
- [ ] Add to strategy as fallback option

**Timeline**: 4-5 days (complex)

---

## Configuration Interface

### User-Facing Config Options

Add to `SimOpsConfig`:

```python
@dataclass
class SimOpsConfig:
    # ... existing fields ...

    # OpenFOAM BC Strategy
    openfoam_bc_strategy: str = "auto"  # "auto", "manual", "split_patch"
    openfoam_bc_tolerance: float = 0.1  # Fraction of dimension for patch classification
    openfoam_adiabatic_sides: bool = True  # Make side walls adiabatic
    openfoam_convection_bc: bool = False  # Use convection instead of fixed temp for cold
    openfoam_split_single_patch: bool = False  # Auto-split single patches
```

### Example Configs

**Simple - Auto Everything**:
```json
{
  "solver": "openfoam",
  "hot_wall_face": "z_min",
  "heat_source_temperature": 373.15,
  "ambient_temperature": 293.15,
  "openfoam_bc_strategy": "auto"
}
```

**Advanced - Custom Control**:
```json
{
  "solver": "openfoam",
  "hot_wall_face": "z_min",
  "openfoam_bc_strategy": "split_patch",
  "openfoam_split_single_patch": true,
  "openfoam_adiabatic_sides": true,
  "openfoam_convection_bc": true,
  "convection_coefficient": 20.0
}
```

---

## Testing Strategy

### Unit Tests
- [ ] Test patch geometry parser with known OpenFOAM meshes
- [ ] Test BC classification logic with synthetic geometries
- [ ] Test BC file writer with various patch configurations

### Integration Tests
- [ ] Test with cube mesh (6 patches - one per face)
- [ ] Test with cylinder mesh (3 patches - top/bottom/side)
- [ ] Test with single-patch mesh (worst case)
- [ ] Test with complex assembly mesh (10+ patches)

### Validation Tests
- [ ] Compare results with built-in solver (should match)
- [ ] Compare with manual OpenFOAM setup (benchmark)
- [ ] Verify temperature gradients are physical
- [ ] Check solver convergence and stability

---

## Fallback Strategies

### If Patch Analysis Fails
1. **Fallback to built-in solver**: Warn user, use Python solver instead
2. **Use conservative BCs**: Apply mixed BC to all patches
3. **Prompt user**: Ask for manual patch assignment

### If Single Patch Detected
1. **Warn user**: Log that gradient may be weak
2. **Use mixed BC**: Better than uniform
3. **Suggest mesh improvement**: Recommend creating physical groups in CAD

### If Geometric Classification Uncertain
1. **Use safe defaults**: Adiabatic for ambiguous patches
2. **Log warnings**: Inform user of assumptions made
3. **Provide override option**: Allow manual BC specification

---

## Dependencies

### New Python Packages
- None (use existing: numpy, pathlib)

### OpenFOAM Requirements
- OpenFOAM 10+ or ESI OpenFOAM 2312+
- Standard solvers: laplacianFoam
- Optional: groovyBC for spatially varying BCs

### Code Structure
```
core/
├── openfoam/
│   ├── __init__.py
│   ├── boundary_analyzer.py      # Phase 1
│   ├── bc_strategy.py             # Phase 2
│   ├── patch_splitter.py          # Phase 4 (optional)
│   └── foam_io.py                 # Helper functions for reading OpenFOAM files
├── solvers/
│   └── (existing files)
└── (existing structure)
```

---

## Performance Considerations

### Expected Performance
- Patch analysis: <0.5s for typical meshes (<100K faces)
- BC assignment: <0.1s (mostly computation)
- Total overhead: <1s added to OpenFOAM workflow

### Optimization Opportunities
- Cache analyzed geometries for repeated runs
- Parallelize face processing for large meshes
- Use binary OpenFOAM format for faster I/O

---

## Future Enhancements

### Multi-Region Support (Conjugate Heat Transfer)
- Detect disconnected volumes in mesh
- Set up `chtMultiRegionFoam` automatically
- Handle solid-solid interfaces (TIE)
- Handle solid-fluid interfaces

### Contact Resistance
- Detect interfaces between volumes
- Apply contact resistance BCs
- Support thermal gap models

### Convective BCs
- Auto-detect external surfaces
- Apply natural/forced convection
- Estimate heat transfer coefficients

### Radiation
- Detect facing surfaces
- Set up view factors
- Apply radiation BCs

---

## Migration Path

### For Existing Users

**Phase 1 Release** (Geometric Analysis):
- No breaking changes
- New feature: Auto-detect patches
- Fallback to current behavior if analysis fails

**Phase 2 Release** (Intelligent BCs):
- Automatic BC assignment
- Config flag to use old behavior: `openfoam_bc_strategy: "legacy"`
- Migration guide for custom BC users

**Phase 3 Release** (Patch Splitting):
- Optional feature: `openfoam_split_single_patch: false` by default
- Opt-in for users with single-patch meshes

### Documentation Updates
- [ ] User guide: "How OpenFOAM BCs are assigned"
- [ ] Developer docs: Architecture of BC system
- [ ] Examples: Various mesh types and configs
- [ ] Troubleshooting: Common issues and fixes

---

## Success Metrics

### Technical Metrics
- ✅ >95% of meshes correctly classified
- ✅ Temperature gradients match built-in solver (±5%)
- ✅ No solver failures due to bad BCs
- ✅ <1s overhead for BC setup

### User Metrics
- ✅ Users can run OpenFOAM without manual BC setup
- ✅ Results match expectations (hot at source, cold at sink)
- ✅ Reduced support tickets about "no gradient" issues
- ✅ Positive feedback on auto-detection feature

---

## Timeline Summary

| Phase | Feature | Duration | Dependencies |
|-------|---------|----------|--------------|
| 1 | Patch Geometry Analysis | 2-3 days | None |
| 2 | BC Assignment Strategy | 3-4 days | Phase 1 |
| 3 | Integration | 2 days | Phase 1, 2 |
| 4 | Patch Splitting (Optional) | 4-5 days | Phase 1, 2, 3 |
| Testing | All phases | 2-3 days | All |
| **Total** | **Core features (1-3)** | **~2 weeks** | - |
| **Total** | **With optional (1-4)** | **~3 weeks** | - |

---

## Risk Assessment

### High Risk
- **OpenFOAM binary format parsing**: Complex, version-dependent
  - *Mitigation*: Use OpenFOAM Python bindings if available, or require ASCII format

### Medium Risk
- **Geometric classification edge cases**: Ambiguous geometries
  - *Mitigation*: Conservative defaults, user overrides

### Low Risk
- **Performance on large meshes**: >1M faces
  - *Mitigation*: Optimize, add caching, warn on huge meshes

---

## References

### OpenFOAM Documentation
- [Boundary Conditions Guide](https://www.openfoam.com/documentation/guides/latest/doc/guide-bcs.html)
- [polyMesh Format](https://www.openfoam.com/documentation/guides/latest/doc/guide-mesh-formats.html)
- [groovyBC for Spatially Varying BCs](https://openfoamwiki.net/index.php/Contrib/groovyBC)

### Similar Implementations
- ParaView patch extraction
- PyFoam utilities
- cfMesh boundary layer tools

---

## Approval & Next Steps

### Required Approvals
- [ ] Architecture review
- [ ] UX review (config interface)
- [ ] Security review (file parsing)

### Ready to Start?
1. Create feature branch: `feature/openfoam-bc-autosetup`
2. Implement Phase 1 (analyzer)
3. PR with unit tests
4. Iterate on Phases 2-3

### Questions / Discussion
- Preferred BC strategy for single-patch case?
- Should we require physical groups in CAD for best results?
- Support for non-Cartesian geometries (cylinders, spheres)?
