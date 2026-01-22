# H2: Gmsh Mesh Format Incompatibility

## Mission
Investigate mesh format requirements for gmshToFoam and ensure the pipeline generates compatible meshes.

## Context
gmshToFoam error log showed:
```
Unhandled element 15 at line 320 in/on physical region ID: 0
Perhaps you created a higher order mesh?
```
Element type 15 is a 1-node point element in MSH format. OpenFOAM may not support this.

## Files to Analyze (Read-Only)
- `simops_pipeline.py` lines 1100-1150 (mesh handling in run_simops_pipeline)
- `strategies/pymesh_strategy.py` (likely the file referred to as `cfd_mesh_strategy.py`)

## Your Task
1. Document which MSH element types are supported by gmshToFoam
2. Create a mesh conversion script that strips unsupported elements
3. Test with a known-good mesh file

## Verification Requirements
Create `verify_task.py` that:
- Loads a test mesh and validates element types
- Runs gmshToFoam on the converted mesh (via WSL)
- Confirms polyMesh directory is created

## Output Location
All files in: `AI_Agent_Projects/OpenFOAM_Debug/task_H2_mesh_format/`
