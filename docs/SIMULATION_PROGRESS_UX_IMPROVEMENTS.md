# Simulation Progress & Console Visibility UX Improvements

## Problem Summary

### Issue 1: Missing Step 4 Clarity
The frontend displays steps 1-3 immediately, then makes a blocking API call to `/api/simulate`. During this call (which can take significant time for meshing with boundary layers, solving, and visualization), the user sees no progress updates in the console. Only after the API returns do Steps 4-5 appear.

**Current flow:**
- **Step 1:** Configuring solver (immediate)
- **Step 2:** Initializing mesh (immediate)
- **Step 3:** Setting boundary conditions (immediate)
- *[Long blocking API call - no visibility]*
- **Step 4:** Solving equations (after API returns)
- **Step 5:** Generating visualization (after API returns)

**Backend reality:**
The backend (`simops_pipeline.py`) actually performs extensive work during the API call:
- Mesh generation with 5 sub-steps (geometry analysis, boundary layers, volume meshing, optimization)
- Thermal solving with iterative convergence
- Visualization generation (PNG, VTK, PDF)

None of this progress is visible to the user during execution.

### Issue 2: Console Visibility During Loading
When `isProcessing=true`, the `ResultViewer` component shows a loading overlay with:
- `z-50` positioning (above console at `z-20`)
- `bg-background/80` (80% opacity dark overlay)
- Covers the entire viewport from top to `consoleOffset`

This darkens the viewport and obscures the 3D view, making the application feel less responsive even though the console is technically visible.

## Root Cause Analysis

### Backend Architecture
**File:** `simops-backend/api_server.py` (line 197-255)
- The `/api/simulate` endpoint calls `run_simops_pipeline()` **synchronously**
- This is a blocking call that returns only when the entire pipeline completes
- No intermediate status updates are sent during execution

**File:** `simops_pipeline.py` (line 703-961)
- The pipeline executes 5 major steps with many sub-steps
- Uses print statements for logging (not visible to frontend)
- No callback mechanism for progress reporting

### Frontend Architecture
**File:** `simops-frontend/src/App.jsx` (line 265-324)
- Logs steps 1-3 synchronously before the API call
- Makes blocking fetch to `/api/simulate`
- Logs steps 4-5 only after the API returns with results

**File:** `simops-frontend/src/components/ResultViewer.jsx` (line 168-179)
- Loading overlay uses `bg-background/80` which significantly darkens the viewport
- Positioned at `z-50` (above everything)
- Shows generic "PROCESSING GEOMETRY..." message

## Proposed Solutions

### Solution 1: Polling-Based Progress (Quick - 2-3 hours)
**Complexity:** Low | **Time:** 2-3 hours | **Dependencies:** None

#### Overview
Create a job queue system where simulations run asynchronously in background threads. Frontend polls backend every 500ms for progress updates.

#### Backend Changes

**New File:** `simops-backend/job_manager.py`
```python
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Callable
import threading
import time

class JobStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class JobProgress:
    job_id: str
    status: JobStatus
    stage: str  # e.g., "mesh_analyzing", "solving_iteration"
    stage_name: str  # Human-readable: "Analyzing geometry"
    progress_pct: float  # 0-100
    data: Dict  # Extra data (iteration count, node count, etc.)
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

class JobManager:
    def __init__(self):
        self.jobs: Dict[str, JobProgress] = {}

    def create_job(self, job_id: str) -> JobProgress:
        job = JobProgress(
            job_id=job_id,
            status=JobStatus.QUEUED,
            stage="queued",
            stage_name="Queued",
            progress_pct=0,
            data={},
            started_at=time.time()
        )
        self.jobs[job_id] = job
        return job

    def update_progress(self, job_id: str, stage: str, stage_name: str,
                       progress_pct: float, data: Dict = None):
        if job_id in self.jobs:
            self.jobs[job_id].stage = stage
            self.jobs[job_id].stage_name = stage_name
            self.jobs[job_id].progress_pct = progress_pct
            self.jobs[job_id].data = data or {}
            self.jobs[job_id].status = JobStatus.RUNNING

    def complete_job(self, job_id: str, result: Dict):
        if job_id in self.jobs:
            self.jobs[job_id].status = JobStatus.COMPLETED
            self.jobs[job_id].progress_pct = 100
            self.jobs[job_id].completed_at = time.time()
            self.jobs[job_id].data['result'] = result

    def fail_job(self, job_id: str, error: str):
        if job_id in self.jobs:
            self.jobs[job_id].status = JobStatus.FAILED
            self.jobs[job_id].error = error
            self.jobs[job_id].completed_at = time.time()

    def get_job(self, job_id: str) -> Optional[JobProgress]:
        return self.jobs.get(job_id)

# Global singleton
job_manager = JobManager()
```

**Modify:** `simops-backend/api_server.py`

1. Import job manager:
```python
from job_manager import job_manager, JobStatus
import threading
```

2. Modify `/api/simulate` endpoint (line 197):
```python
@app.route('/api/simulate', methods=['POST'])
def trigger_simulation():
    # ... existing validation code ...

    # Create job
    job_id = str(uuid.uuid4())[:8]
    job_manager.create_job(job_id)

    # Run pipeline in background thread
    def run_job():
        def progress_callback(stage, stage_name, progress_pct, data=None):
            job_manager.update_progress(job_id, stage, stage_name, progress_pct, data)

        try:
            results_metadata = run_simops_pipeline(
                cad_file=str(input_path),
                output_dir=str(job_output_dir),
                config=pipeline_config,
                verbose=True,
                progress_callback=progress_callback
            )
            job_manager.complete_job(job_id, results_metadata)
        except Exception as e:
            job_manager.fail_job(job_id, str(e))

    thread = threading.Thread(target=run_job, daemon=True)
    thread.start()

    # Return job ID immediately
    return jsonify({
        'status': 'accepted',
        'job_id': job_id
    }), 202
```

3. Add status polling endpoint:
```python
@app.route('/api/job/<job_id>/status', methods=['GET'])
def get_job_status(job_id):
    job = job_manager.get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    return jsonify({
        'status': job.status.value,
        'stage': job.stage,
        'stage_name': job.stage_name,
        'progress_pct': job.progress_pct,
        'data': job.data,
        'error': job.error
    })
```

4. Add result endpoint:
```python
@app.route('/api/job/<job_id>/result', methods=['GET'])
def get_job_result(job_id):
    job = job_manager.get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    if job.status != JobStatus.COMPLETED:
        return jsonify({
            'error': 'Job not completed',
            'status': job.status.value
        }), 400

    result = job.data.get('result', {})
    return jsonify({
        'status': 'success',
        'results': result
    })
```

**Modify:** `simops_pipeline.py`

1. Add progress_callback parameter (line 703):
```python
def run_simops_pipeline(
    cad_file: str,
    output_dir: str = "simops_output",
    config: Optional[SimOpsConfig] = None,
    verbose: bool = True,
    progress_callback: Optional[Callable] = None
) -> Dict:
```

2. Insert 8 progress callback points:

**After geometry analysis:**
```python
if progress_callback:
    progress_callback(
        'mesh_analyzing',
        'Analyzing CAD geometry',
        10,
        {'num_nodes': mesh_stats.get('num_nodes', 0)}
    )
```

**Before boundary layer generation:**
```python
if progress_callback:
    progress_callback(
        'mesh_boundary_layers',
        'Generating boundary layers',
        25,
        {}
    )
```

**After mesh generation:**
```python
if progress_callback:
    progress_callback(
        'mesh_complete',
        'Mesh generation complete',
        40,
        mesh_stats
    )
```

**Before solving:**
```python
if progress_callback:
    progress_callback(
        'solving_start',
        'Setting up thermal solver',
        45,
        {}
    )
```

**During solving (modify solver init):**
```python
solver = ThermalSolver(
    verbose=verbose,
    progress_callback=lambda iter_data: progress_callback(
        'solving_iteration',
        f'Solving equations',
        45 + (iter_data['iter'] / iter_data['max_iter']) * 40,
        iter_data
    ) if progress_callback else None
)
```

**After solving:**
```python
if progress_callback:
    progress_callback(
        'solving_complete',
        'Thermal solution complete',
        85,
        {'max_temp': results.get('max_temperature_C')}
    )
```

**During visualization:**
```python
if progress_callback:
    progress_callback(
        'generating_viz',
        'Generating visualizations',
        90,
        {}
    )
```

**Before final return:**
```python
if progress_callback:
    progress_callback(
        'finalizing',
        'Finalizing results',
        95,
        {}
    )
```

**Modify:** `core/solvers/cfd_solver.py`

1. Update `ThermalSolver.__init__` to accept progress callback:
```python
def __init__(self, verbose=True, progress_callback=None):
    self.verbose = verbose
    self.progress_callback = progress_callback
```

2. Report progress during iterations (in solve method):
```python
# Inside iteration loop
if self.progress_callback and iteration % 10 == 0:
    self.progress_callback({
        'iter': iteration,
        'max_iter': max_iterations,
        'residual': residual
    })
```

#### Frontend Changes

**Modify:** `simops-frontend/src/App.jsx`

1. Add new state (line 20):
```javascript
const [currentJobId, setCurrentJobId] = useState(null)
const [jobProgress, setJobProgress] = useState(null)
const pollingIntervalRef = useRef(null)
```

2. Replace `runSimulation` function (line 229):
```javascript
const runSimulation = async () => {
    setIsProcessing(true)
    setStatusMessage("Starting simulation...")
    addLog(`Starting simulation: ${power}W heat source, ${ambientTemp}K ambient`, 'info')

    try {
        const config = {
            heat_source_power: power,
            ambient_temperature: ambientTemp,
            initial_temperature: initialTemp,
            convection_coefficient: convection,
            material: material,
            simulation_type: simMode,
            time_step: timestep,
            duration: duration,
            max_iterations: iterations,
            tolerance: tolerance,
            write_interval: writeInterval,
            colormap: colormap,
            solver: 'builtin'
        }

        if (!openfoamAvailable) {
            addLog("   (OpenFOAM not available - using fast Python solver)", 'info')
        }

        // Step 1: Start simulation job
        const startRes = await fetch('http://localhost:8000/api/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                filename: uploadedFilename,
                config: config
            })
        })

        if (!startRes.ok) {
            const errorData = await startRes.json()
            throw new Error(errorData.error || 'Failed to start simulation')
        }

        const { job_id } = await startRes.json()
        setCurrentJobId(job_id)

        // Step 2: Poll for progress
        let lastStage = ''

        pollingIntervalRef.current = setInterval(async () => {
            try {
                const statusRes = await fetch(`http://localhost:8000/api/job/${job_id}/status`)
                const status = await statusRes.json()

                setJobProgress(status)
                setStatusMessage(status.stage_name || 'Processing...')

                // Log new stages
                if (status.stage !== lastStage) {
                    lastStage = status.stage
                    logStageUpdate(status)
                }

                // Check if complete
                if (status.status === 'completed') {
                    clearInterval(pollingIntervalRef.current)
                    await handleJobComplete(job_id)
                } else if (status.status === 'failed') {
                    clearInterval(pollingIntervalRef.current)
                    throw new Error(status.error || 'Simulation failed')
                }
            } catch (err) {
                console.error('Polling error:', err)
            }
        }, 500)  // Poll every 500ms

    } catch (err) {
        addLog(`Error: ${err.message}`, 'error')
        setIsProcessing(false)
        if (pollingIntervalRef.current) {
            clearInterval(pollingIntervalRef.current)
        }
    }
}
```

3. Add helper functions:
```javascript
const logStageUpdate = (status) => {
    const stageLogMap = {
        'mesh_analyzing': () => {
            addLog(`Step 1: Analyzing CAD geometry`, 'step')
            if (status.data?.num_nodes) {
                addLog(`   Found ${status.data.num_nodes.toLocaleString()} nodes`, 'info')
            }
        },
        'mesh_boundary_layers': () => {
            addLog(`Step 2: Generating boundary layers`, 'step')
            addLog(`   This may take a moment for complex geometries...`, 'info')
        },
        'mesh_complete': () => {
            addLog(`Step 3: Mesh generation complete`, 'success')
            if (status.data?.num_elements) {
                addLog(`   Total elements: ${status.data.num_elements.toLocaleString()}`, 'info')
            }
        },
        'solving_start': () => {
            addLog(`Step 4: Setting up thermal solver`, 'step')
        },
        'solving_iteration': () => {
            if (!window.__solving_logged) {
                addLog(`Step 5: Solving thermal equations`, 'step')
                window.__solving_logged = true
            }
            if (status.data?.iter && status.data?.max_iter) {
                const pct = ((status.data.iter / status.data.max_iter) * 100).toFixed(0)
                setStatusMessage(`Solving: iteration ${status.data.iter}/${status.data.max_iter} (${pct}%)`)
            }
        },
        'solving_complete': () => {
            addLog(`Step 6: Thermal solution converged`, 'success')
            window.__solving_logged = false
            if (status.data?.max_temp) {
                addLog(`   Max Temperature: ${status.data.max_temp.toFixed(1)}°C`, 'success')
            }
        },
        'generating_viz': () => {
            addLog(`Step 7: Generating visualizations`, 'step')
        },
        'finalizing': () => {
            addLog(`Step 8: Finalizing results`, 'step')
        }
    }

    const logFn = stageLogMap[status.stage]
    if (logFn) logFn()
}

const handleJobComplete = async (job_id) => {
    try {
        const resultRes = await fetch(`http://localhost:8000/api/job/${job_id}/result`)
        const resultData = await resultRes.json()

        setSimResults(resultData.results)
        addLog(`Simulation completed successfully!`, 'success')

        if (resultData.results?.pdf_url) {
            const pdfUrl = `http://localhost:8000${resultData.results.pdf_url}`
            addLog(`Opening PDF report...`, 'info')
            window.open(pdfUrl, '_blank')
        }

        setIsProcessing(false)
        setStatusMessage("Complete")
    } catch (err) {
        addLog(`Error retrieving results: ${err.message}`, 'error')
        setIsProcessing(false)
    }
}
```

4. Cleanup on unmount:
```javascript
useEffect(() => {
    return () => {
        if (pollingIntervalRef.current) {
            clearInterval(pollingIntervalRef.current)
        }
    }
}, [])
```

**Modify:** `simops-frontend/src/components/ResultViewer.jsx`

Remove loading overlay, add corner spinner:

1. Delete lines 166-180 (the full-screen loading overlay)

2. Add corner loading indicator after line 165:
```javascript
{/* Corner Loading Indicator - No Overlay */}
{isLoading && (
    <div className="absolute top-4 right-4 z-50 flex items-center gap-2 bg-card/95 backdrop-blur border border-primary/30 rounded-lg px-3 py-2 shadow-xl">
        <Loader2 className="w-4 h-4 animate-spin text-primary" />
        <span className="text-xs font-mono text-foreground">
            {loadingMessage || 'PROCESSING...'}
        </span>
    </div>
)}
```

#### Trade-offs

**Pros:**
- Simple to implement (no WebSocket libraries)
- Works behind firewalls/proxies (standard HTTP)
- Easy to debug with browser DevTools
- No new dependencies
- Backward compatible

**Cons:**
- 500ms latency on updates
- More HTTP requests (not a problem for localhost)
- Polling continues even when browser tab is inactive
- No ability to cancel running simulations

---

### Solution 2: Server-Sent Events (SSE) (Moderate - 4-6 hours)
**Complexity:** Medium | **Time:** 4-6 hours | **Dependencies:** None

#### Overview
Backend streams progress updates to frontend via SSE. Frontend receives real-time updates without polling overhead.

#### Implementation Notes

**Backend:**
- Add `/api/simulate/stream` endpoint using Flask's `Response` with `text/event-stream` mimetype
- Stream progress updates as SSE messages
- Frontend uses `EventSource` API to receive updates

**Frontend:**
- Replace polling with `EventSource` connection
- Listen for `message` events
- Handle automatic reconnection

**Trade-offs:**
- **Pros:** True real-time updates (no 500ms delay), efficient (single connection), standard HTTP
- **Cons:** GET-only (must encode config in URL), unidirectional (can't cancel), some proxies block SSE, browser limit (~6 connections per domain)

---

### Solution 3: WebSocket-Based (Advanced - 8-12 hours)
**Complexity:** High | **Time:** 8-12 hours | **Dependencies:** Flask-SocketIO, socket.io-client

#### Overview
Full bidirectional communication allowing real-time updates AND user control (pause/cancel simulations).

#### Implementation Notes

**Backend:**
- Install Flask-SocketIO
- Add WebSocket handlers for `start_simulation`, `cancel_simulation`
- Emit progress events via `socketio.emit()`

**Frontend:**
- Install socket.io-client
- Connect to WebSocket server
- Listen for progress events
- Send cancel/pause commands

**Trade-offs:**
- **Pros:** True bidirectional, can cancel/pause, most responsive, can show live plots, multiple clients can monitor same job
- **Cons:** Most complex, new dependencies, requires WebSocket support (blocked by some proxies), more difficult to debug

---

## Recommended Implementation Path

### Phase 1: Quick Win (30 minutes)
**UI-only improvements - no backend changes:**

1. **Better step descriptions** in `App.jsx`:
```javascript
// Current misleading steps:
Step 1: Configuring solver
Step 2: Initializing mesh
Step 3: Setting boundary conditions

// More accurate steps:
Step 1: Uploading geometry to server
Step 2: Preparing simulation environment
Step 3: Waiting for mesh generation and solve...
Step 4: [Appears after solve] Results ready
```

2. **Remove viewport overlay** in `ResultViewer.jsx`:
   - Replace full-screen overlay with corner spinner
   - Keep console fully visible

3. **Add estimated time message**:
```javascript
addLog('   Meshing and solving may take 30-60 seconds...', 'info')
```

### Phase 2: Polling System (2-3 hours)
Implement Solution 1 as documented above.

### Phase 3: Real-time Streaming (4-6 hours)
Upgrade to SSE (Solution 2) for smoother updates.

### Phase 4: Advanced Features (8-12 hours)
Full WebSocket implementation with cancellation and live monitoring.

---

## Verification Checklist

### Backend Tests
- [ ] Job manager creates jobs correctly
- [ ] Status endpoint returns current progress
- [ ] Result endpoint returns final results
- [ ] Background threads don't block API server
- [ ] Error handling works (failed mesh, solver crash)
- [ ] Multiple concurrent jobs are tracked correctly

### Frontend Tests
- [ ] Polling starts when simulation begins
- [ ] Console shows 8 distinct progress steps
- [ ] Status message updates during solving iterations
- [ ] Polling stops when simulation completes
- [ ] Results display correctly
- [ ] PDF auto-download still works
- [ ] No console overlay during processing
- [ ] Corner spinner appears/disappears correctly

### Integration Tests
- [ ] Run complex geometry (slow mesh generation)
- [ ] Run simple geometry (fast execution)
- [ ] Test with convergence failures
- [ ] Test with invalid CAD files
- [ ] Verify console remains readable throughout
- [ ] Test console collapse/expand during simulation

---

## Files to Modify

### Backend (Python)
1. **`simops-backend/job_manager.py`** (NEW) - Job tracking system
2. **`simops-backend/api_server.py`** (MODIFY) - Add 3 endpoints, background execution
3. **`simops_pipeline.py`** (MODIFY) - Add progress_callback param, 8 callback points
4. **`core/solvers/cfd_solver.py`** (MODIFY) - Iteration progress reporting

### Frontend (React)
5. **`simops-frontend/src/App.jsx`** (MODIFY) - Polling logic, detailed step logging
6. **`simops-frontend/src/components/ResultViewer.jsx`** (MODIFY) - Remove overlay, add corner spinner

---

## Estimated Timeline

| Phase | Duration | Deliverables |
|-------|----------|-------------|
| Phase 1 (UI-only) | 30 min | Better step messages, no overlay |
| Phase 2 (Polling) | 2-3 hours | Full backend progress tracking |
| Phase 3 (SSE) | 4-6 hours | Real-time streaming updates |
| Phase 4 (WebSocket) | 8-12 hours | Bidirectional control, cancellation |

## Notes

- Start with Phase 1 for immediate improvement with minimal effort
- Phase 2 (Polling) provides the best balance of effort vs. benefit
- Phases 3-4 are optional enhancements for production-grade UX
- All phases are backward compatible with existing code
