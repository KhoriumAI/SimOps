# H3: Exception Swallowing in Pipeline

## Mission
Audit the error handling in `OpenFOAMRunner` and `run_simops_pipeline` to ensure failures are properly reported.

## Context
The E2E test shows:
- Simulation "completed" in 2s
- Elements: 0, Nodes: 0
- Converged: False
- But no error was raised!

This suggests exceptions are being caught and placeholder data returned.

## Files to Analyze (Read-Only)
- `simops_pipeline.py` lines 500-740 (OpenFOAMRunner class)
- `simops_pipeline.py` lines 1150-1180 (solver dispatch in run_simops_pipeline)

## Your Task
1. Trace the error handling flow from WSL command failure to API response
2. Identify where failures are silently converted to "success" with placeholder data
3. Create a shadow copy with explicit error propagation

## Verification Requirements
Create `verify_task.py` that:
- Simulates a WSL command failure
- Confirms the error propagates to the API response
- Verifies the API returns HTTP 500, not 200 with empty data

## Output Location
All files in: `AI_Agent_Projects/OpenFOAM_Debug/task_H3_exception_handling/`
