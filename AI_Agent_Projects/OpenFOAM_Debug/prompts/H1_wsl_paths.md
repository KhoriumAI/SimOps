# H1: WSL Path Escaping Bug

## Mission
Investigate and fix potential Windows-to-WSL path conversion issues in `OpenFOAMRunner._run_wsl_command()`.

## Context
The current implementation uses `wslpath` but the command failed with:
```
wslpath: C:UsersmarkmDownloadsSimopsLoft_mesh_v2.msh
```
This indicates the backslashes are being stripped before reaching `wslpath`.

## Files to Analyze (Read-Only)
- `simops_pipeline.py` lines 520-580 (OpenFOAMRunner.solve and to_wsl function)

## Your Task
1. Create a test script in your task folder that isolates the path conversion issue
2. Create a fixed version of the `to_wsl()` function as a shadow copy
3. Verify the fix works with a simple WSL command test

## Verification Requirements
Create `verify_task.py` that:
- Tests path conversion for paths with spaces, special chars, and different drive letters
- Confirms the converted path is accessible from WSL

## Output Location
All files in: `AI_Agent_Projects/OpenFOAM_Debug/task_H1_wsl_paths/`
