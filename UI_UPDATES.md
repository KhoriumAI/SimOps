# UI Updates Summary

## Changes Made

### 1. Solver Selection Button ✅

**Location**: `simops-frontend/src/App.jsx`

**Changes**:
- Added new state variable `selectedSolver` (line 50)
- Added solver selection UI in left panel (lines 590-614)
- Updated simulation config to use user-selected solver instead of auto-detection (line 299)
- Added warning log when OpenFOAM is selected but not available (line 305)

**Features**:
- **Two-button toggle**: Built-in vs OpenFOAM
- **Visual indicator**: Shows ⚠ warning when OpenFOAM is not available
- **Tooltip help**: Explains what each solver does
- **Always functional**: Both buttons work even when OpenFOAM isn't installed

**UI Location**:
```
Left Panel
└── Physics Setup
    └── Solver & Output
        └── [Solver Engine]  ← NEW!
            ├── [Built-in] button
            └── [OpenFOAM ⚠] button
```

### 2. Z-Up Orientation ✅

**Location**:
- `simops-frontend/src/components/ResultViewer.jsx` (lines 223-232, 267-273)
- `simops-frontend/src/components/Viz.jsx` (lines 8-9)

**Changes**:
- **Camera**: Added `up={[0, 0, 1]}` to PerspectiveCamera
- **OrbitControls**: Added `up={[0, 0, 1]}` parameter
- **Gizmo**: Updated axis labels for clarity (X, Y, Z)

**Before**: Y-up (Y is vertical, typical game/animation convention)
**After**: Z-up (Z is vertical, typical CAD/engineering convention)

**Effect**:
- Models now load with Z-axis pointing up
- Rotation feels more natural for CAD models
- Matches industry-standard tools (SolidWorks, CATIA, etc.)

---

## Testing Checklist

### Solver Selection
- [ ] Click "Built-in" button → should highlight, solver should be 'builtin'
- [ ] Click "OpenFOAM" button → should highlight, solver should be 'openfoam'
- [ ] Run simulation with Built-in → should work (always available)
- [ ] Run simulation with OpenFOAM when not installed → should show warning
- [ ] Run simulation with OpenFOAM when installed → should use OpenFOAM
- [ ] Check console logs → should say "Solver: Builtin (Python)" or "Solver: OpenFOAM"

### Z-Up Orientation
- [ ] Load a mesh → should appear with Z-axis pointing up
- [ ] Check axis gizmo (bottom-right) → Z should point up (blue)
- [ ] Orbit around → rotation should feel natural for CAD
- [ ] Compare with old behavior → Z was previously sideways/horizontal

---

## Code Changes Detail

### File: `simops-frontend/src/App.jsx`

#### Change 1: Add state for solver selection
```diff
  const [colormap, setColormap] = useState('jet')
+ const [selectedSolver, setSelectedSolver] = useState('builtin')
```

#### Change 2: Update config to use selected solver
```diff
  const config = {
      // ... other config ...
-     solver: openfoamAvailable ? 'openfoam' : 'builtin'
+     solver: selectedSolver
  }

- const solverName = openfoamAvailable ? 'OpenFOAM' : 'Builtin (Python)'
+ const solverName = selectedSolver === 'openfoam' ? 'OpenFOAM' : 'Builtin (Python)'
  addLog(`Solver: ${solverName}`, 'info')

- if (!openfoamAvailable) {
-     addLog("   (OpenFOAM not available - using fast Python solver)", 'info')
- }
+ if (selectedSolver === 'openfoam' && !openfoamAvailable) {
+     addLog("   Warning: OpenFOAM selected but not available - this may fail", 'error')
+ }
```

#### Change 3: Add UI button in left panel
```jsx
<div className="flex flex-col gap-1 mb-2">
    <label className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium flex items-center gap-1.5">
        Solver Engine
        <div className="relative group/tip cursor-help">
            <HelpCircle className="w-2.5 h-2.5 opacity-40 hover:opacity-80 transition-opacity" />
            <div className="...">
                Built-in: Fast Python solver (always available).
                OpenFOAM: Advanced CFD solver (requires WSL + OpenFOAM installation).
            </div>
        </div>
    </label>
    <div className="flex bg-muted/30 rounded p-0.5 border border-border">
        <button
            onClick={() => setSelectedSolver('builtin')}
            className={`... ${selectedSolver === 'builtin' ? 'bg-primary ...' : '...'}`}
        >
            Built-in
        </button>
        <button
            onClick={() => setSelectedSolver('openfoam')}
            className={`... ${selectedSolver === 'openfoam' ? 'bg-primary ...' : '...'}`}
            title={!openfoamAvailable ? 'OpenFOAM not detected - install WSL + OpenFOAM to enable' : 'Use OpenFOAM solver'}
        >
            OpenFOAM {!openfoamAvailable && '⚠'}
        </button>
    </div>
</div>
```

### File: `simops-frontend/src/components/ResultViewer.jsx`

#### Change 1: Set camera to Z-up
```diff
- <PerspectiveCamera makeDefault position={[50, 50, 50]} fov={45} />
+ <PerspectiveCamera makeDefault position={[50, 50, 50]} fov={45} up={[0, 0, 1]} />
```

#### Change 2: Set OrbitControls to Z-up
```diff
  <OrbitControls
      makeDefault
      enableDamping={false}
      rotateSpeed={0.8}
      zoomSpeed={1.2}
+     up={[0, 0, 1]}
  />
```

#### Change 3: Update gizmo labels (optional clarity)
```diff
  <GizmoHelper alignment="bottom-right" margin={[80, 200]}>
-     <GizmoViewport axisColors={['#ef4444', '#22c55e', '#3b82f6']} labelColor="white" />
+     <GizmoViewport
+         axisColors={['#ef4444', '#22c55e', '#3b82f6']}
+         labelColor="white"
+         axisHeadScale={0.75}
+         labels={['X', 'Y', 'Z']}
+     />
  </GizmoHelper>
```

### File: `simops-frontend/src/components/Viz.jsx`

#### Change: Set camera and controls to Z-up
```diff
- <Canvas shadows dpr={[1, 2]}>
-     <OrbitControls makeDefault />
+ <Canvas shadows dpr={[1, 2]} camera={{ position: [5, 5, 5], up: [0, 0, 1] }}>
+     <OrbitControls makeDefault up={[0, 0, 1]} />
```

---

## User-Facing Changes

### Before
1. **Solver**: Automatically chosen (OpenFOAM if available, otherwise Built-in)
   - User had no control
   - Unclear which solver was running

2. **Orientation**: Y-up (standard game/animation)
   - Z-axis was horizontal
   - Rotation felt "sideways" for CAD models

### After
1. **Solver**: User manually selects
   - Clear toggle button in UI
   - See which solver will run before executing
   - Warning if OpenFOAM selected but not available

2. **Orientation**: Z-up (standard CAD/engineering)
   - Z-axis is vertical (up)
   - Natural rotation for engineering models
   - Matches industry tools

---

## Screenshots of UI Changes

### Solver Selection Button (NEW)
```
┌─────────────────────────────────┐
│ SOLVER ENGINE                   │
│ ┌──────────────┬───────────────┐│
│ │  BUILT-IN ✓  │   OPENFOAM ⚠  ││
│ └──────────────┴───────────────┘│
└─────────────────────────────────┘
   ^Active          ^Not available
```

When OpenFOAM IS available:
```
┌─────────────────────────────────┐
│ SOLVER ENGINE                   │
│ ┌──────────────┬───────────────┐│
│ │   BUILT-IN   │  OPENFOAM ✓   ││
│ └──────────────┴───────────────┘│
└─────────────────────────────────┘
                     ^Active
```

### 3D Viewer Orientation

**Before (Y-up)**:
```
        Y (up)
        |
        |
        +---- X
       /
      Z
```

**After (Z-up)**:
```
        Z (up)
        |
        |
        +---- X
       /
      Y
```

---

## Backend Impact

### No backend changes required
- Backend already supports both solvers via `config.solver` parameter
- Built-in solver always works
- OpenFOAM solver requires WSL + OpenFOAM installation (unchanged)

### API Request Example
```json
{
  "filename": "model.msh",
  "config": {
    "solver": "builtin",  // or "openfoam" - now user-controlled!
    "heat_source_temperature": 373.15,
    "ambient_temperature": 293.15,
    // ... other params
  }
}
```

---

## Known Issues / Future Enhancements

### Current Limitations
1. **OpenFOAM BC issue**: As documented in `OPENFOAM_BC_AUTOSETUP_PLAN.md`, OpenFOAM currently has boundary condition issues with single-patch meshes
   - Recommendation: Use Built-in solver until OpenFOAM BC auto-setup is implemented
   - Built-in solver works correctly with all meshes

2. **No solver performance comparison**: User can't see speed/quality difference before running
   - Future: Add estimated solve time or benchmark data

3. **No mid-simulation solver switch**: Can't change solver while simulation is running
   - This is expected behavior (solver is locked once started)

### Future Enhancements
- [ ] Add "Recommended" badge to Built-in solver
- [ ] Show solver capabilities comparison table
- [ ] Add "Why is OpenFOAM disabled?" help link
- [ ] Remember user's last selected solver (localStorage)
- [ ] Add solver benchmark/timing display after completion

---

## Deployment Notes

### Frontend Build
```bash
cd simops-frontend
npm install  # Only if new dependencies (none added)
npm run build
```

### No Docker Rebuild Required
- Pure frontend changes
- No backend modifications
- No new npm packages

### Testing
1. Start backend: `python app.py`
2. Start frontend: `cd simops-frontend && npm run dev`
3. Open browser: `http://localhost:5173`
4. Test solver selection and 3D orientation

---

## Rollback Plan

If issues arise, revert these files:
```bash
git checkout HEAD -- simops-frontend/src/App.jsx
git checkout HEAD -- simops-frontend/src/components/ResultViewer.jsx
git checkout HEAD -- simops-frontend/src/components/Viz.jsx
```

Or restore from this summary:
- Remove `selectedSolver` state
- Restore original config: `solver: openfoamAvailable ? 'openfoam' : 'builtin'`
- Remove solver selection UI block
- Remove `up={[0, 0, 1]}` from Camera and OrbitControls

---

## Related Documents

- [OPENFOAM_BC_AUTOSETUP_PLAN.md](OPENFOAM_BC_AUTOSETUP_PLAN.md) - Plan for fixing OpenFOAM boundary conditions
- [THERMAL_ISSUES_AND_FIXES.md](THERMAL_ISSUES_AND_FIXES.md) - Thermal visualization fixes
- [RECOMMENDATION.md](RECOMMENDATION.md) - Why to use Built-in solver (for now)
- [DEPLOYMENT_SUMMARY.md](DEPLOYMENT_SUMMARY.md) - Overall thermal fixes deployment

---

## Success Criteria

✅ **User can manually switch between Built-in and OpenFOAM solvers**
✅ **Models load in Z-up orientation (engineering standard)**
✅ **UI is intuitive and self-explanatory**
✅ **Warning shown when OpenFOAM selected but unavailable**
✅ **No breaking changes to existing functionality**
✅ **No backend modifications required**
