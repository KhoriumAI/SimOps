# E2E Test vs UI Flow Discrepancy Analysis

## Problem Statement
- **E2E tests pass** but show the simulation runs successfully
- **UI simulation fails** with "NetworkError when attempting to fetch resource"
- Users experience a different code path than what the tests validate

## Root Cause Analysis

### 1. E2E Tests Are Using Mocks
**Location:** `tests/e2e/test_e2e_integration.py`

The e2e tests instantiate `E2EPipelineRunner(use_mocks=True)` at line 255, 278, and 292.

**What this means:**
- The tests bypass the actual HTTP API layer entirely
- They call Python functions directly: `_run_solving()`, `_run_meshing()`, etc.
- Solver execution returns synthetic mock data (lines 143-179)
- Never actually tries to:
  - Import `OpenFOAMRunner`
  - Copy the `Golden_Thermal_Case` template
  - Run real OpenFOAM commands
  - Make HTTP requests to the backend

**Code Evidence:**
```python
# Line 138 in test_e2e_integration.py
def _run_solving(self, test_case: E2ETestCase, config) -> StageResult:
    """Run solver stage (mocked for fast tests)."""
    # Mock solver execution
    # Extract expected temperature from test case criteria
    ...
    return StageResult(
        stage=PipelineStage.SOLVING,
        passed=True,
        ...
        metadata={
            "converged": True,
            "iterations": random.randint(20, 100),
            ...
        }
    )
```

### 2. UI Flow Uses Real HTTP API
**Location:** `simops-frontend/src/App.jsx`

The UI makes an actual HTTP POST request to `/api/simulate` (line 204):

```javascript
const simRes = await fetch('http://localhost:8000/api/simulate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        filename: uploadedFilename,
        config: config  // includes solver: 'openfoam'
    })
})
const simData = await simRes.json()  // Line 212 - PROBLEM!
```

**The Bug:**
The frontend doesn't check if the HTTP request succeeded before trying to parse JSON. If the backend returns an error (4xx/5xx), trying to parse the error response as JSON can fail and throw "NetworkError when attempting to fetch resource".

**Correct pattern:**
```javascript
const simRes = await fetch(...)
if (!simRes.ok) {
    throw new Error(`HTTP error! status: ${simRes.status}`)
}
const simData = await simRes.json()
```

### 3. Backend OpenFOAM Path
**Location:** `simops-backend/api_server.py` → `simops_pipeline.py` → `tools/thermal_job_runner.py`

When `config.solver = "openfoam"`, the backend:

1. Calls `run_simops_pipeline()` (line 235 in api_server.py)
2. Pipeline checks `if config.solver == "openfoam"` (line 807 in simops_pipeline.py)
3. Imports `OpenFOAMRunner` from `tools.thermal_job_runner` (line 811)
4. Creates `ThermalSetup` and runs the simulation
5. **CaseGenerator tries to copy template** (line 209 in thermal_job_runner.py):
   ```python
   shutil.copytree(self.TEMPLATE_PATH, case_dir)
   ```

### 4. What Could Fail
Based on testing (`test_backend_import.py`), we verified:
- ✓ All imports work correctly
- ✓ Template exists at `simops/templates/Golden_Thermal_Case`
- ✓ Config can be created
- ✓ OpenFOAMRunner can be instantiated

**Remaining possibilities:**
1. **Missing mesh file**: If `uploadedFilename` doesn't exist in the upload folder
2. **Template copy failure**: Permissions issue or locked files
3. **OpenFOAM not available**: When not in dry_run mode
4. **Mesh conversion failure**: gmshToFoam or mesh import issues
5. **Backend exception**: Any unhandled exception returns 500, which frontend mishandles

## The Disconnect

| E2E Tests | UI Flow |
|-----------|---------|
| Direct Python function calls | HTTP API requests |
| Mocked solver results | Real solver execution |
| No file I/O | Actual file operations |
| No OpenFOAM dependencies | Requires OpenFOAM/WSL |
| Always succeeds | Can fail in many ways |
| Tests pass quickly | Real simulation is slow |

## Recommendations

### Immediate Fixes

#### 1. Fix Frontend Error Handling
**File:** `simops-frontend/src/App.jsx` line 204-223

```javascript
const simRes = await fetch('http://localhost:8000/api/simulate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        filename: uploadedFilename,
        config: config
    })
})

// Check response status BEFORE parsing JSON
if (!simRes.ok) {
    let errorMsg = `Simulation failed with status ${simRes.status}`
    try {
        const errorData = await simRes.json()
        errorMsg = errorData.error || errorMsg
    } catch {
        const text = await simRes.text()
        errorMsg = text || errorMsg
    }
    throw new Error(errorMsg)
}

const simData = await simRes.json()
```

#### 2. Add Integration Tests That Use Real API
Create `tests/e2e/test_api_integration.py`:
```python
def test_api_simulate_endpoint():
    """Test the actual /api/simulate endpoint"""
    # Start backend server
    # Upload real mesh file
    # POST to /api/simulate with openfoam config
    # Assert response is 200 or contains specific error
```

#### 3. Add Diagnostics Endpoint
**File:** `simops-backend/api_server.py`

Already exists at `/api/diagnostics` but could be enhanced to check:
- OpenFOAM availability
- Template directory existence
- WSL connectivity
- Upload folder permissions

### Long-term Improvements

1. **Separate test environments:**
   - Unit tests (mocked)
   - Integration tests (real API, mocked external deps)
   - E2E tests (full stack with real OpenFOAM)

2. **Better error propagation:**
   - Backend should return structured errors
   - Frontend should display backend errors properly
   - Add error codes for different failure modes

3. **Test matrix:**
   ```
   Test Type    | Mocked | API | OpenFOAM | Expected
   -------------|--------|-----|----------|----------
   Unit         | Yes    | No  | No       | Fast
   Integration  | Some   | Yes | No       | Medium
   E2E          | No     | Yes | Yes      | Slow
   ```

## Next Steps

1. Run `test_api_flow.py` with backend running to see the actual error
2. Fix the frontend JSON parsing bug
3. Add proper error handling throughout the stack
4. Create real integration tests that use the HTTP API

## Test Scripts Created

- `test_backend_import.py` - Validates all imports work
- `test_api_flow.py` - Simulates the exact frontend HTTP request
- Run these to diagnose the actual failure mode
