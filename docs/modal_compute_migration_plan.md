# Modal Compute Migration - Implementation Plan

Migrate meshing compute workloads from AWS EC2 to Modal serverless GPU workers while keeping AWS as the orchestrator and database layer.

## User Review Required

> [!IMPORTANT]
> **Architecture Change**: This migration replaces the current subprocess-based meshing on EC2 with Modal serverless GPU workers. AWS will only handle API orchestration, authentication, and database operations.

> [!IMPORTANT]
> **Deployment Strategy**: Based on your input, the plan uses a simplified approach:
> - **Dev Environment**: Test thoroughly with `USE_MODAL_COMPUTE=true`
> - **Production**: Once validated, flip feature flag to 100% Modal traffic instantly
> - **Rollback**: Instant rollback to EC2 via feature flag if issues arise
> 
> This eliminates the gradual percentage-based rollout complexity.

> [!CAUTION]
> **Preview Meshing I/O Consideration**: The spike confirms 0.9s meshing time on Modal T4 GPU. However, you mentioned "if the I/O is fast" for preview meshing. We need to measure S3 I/O latency for small files to confirm preview generation should also move to Modal vs staying on EC2.

---

## Proposed Changes

### Modal Service Layer

#### [NEW] [mesh_service.py](file:///c:/Users/markm/Downloads/MeshPackageLean/backend/modal_service.py)

**New Modal service module** that will be deployed to Modal Cloud. This encapsulates all GPU meshing logic.

**Based on Modal spike** (`modal-spike.py`) but porting the **full exhaustive meshing strategy** from `apps/cli/mesh_worker_subprocess.py` and `strategies/exhaustive_strategy.py`.

**Key components**:
- **`generate_mesh(bucket, key, quality_params)`**: Main meshing function
  - Downloads CAD file from S3 to `/tmp/input.step`
  - Runs **exhaustive strategy testing** (not just trimesh - full Gmsh pipeline)
  - Uploads result files (.msh, _result.json, .quality.json) back to S3
  - Returns result metadata (strategy, score, quality_metrics, processing_time)
  
- **`generate_preview_mesh(bucket, key)`**: Fast preview generation (ported from `parse_step_file_for_preview`)
  - Downloads CAD file from S3
  - Generates triangulated preview using Gmsh
  - Uploads preview JSON to S3
  - Returns preview metadata

**Modal Configuration** (from spike):
```python
image = (
    modal.Image.debian_slim()
    .pip_install("numpy", "trimesh", "boto3", "gmsh", "cascadio", "scipy")
    .apt_install("libgl1-mesa-glx", "libglu1-mesa", "libxrender1")
)

@app.function(
    image=image,
    gpu="T4",
    timeout=600,
    secrets=[aws_secret]
)
```

**Additional dependencies needed** (for full meshing pipeline):
- Gmsh Python API (already included)
- NumPy, SciPy (already included)
- Any isolation worker dependencies from `core/isolation_worker_script.py`

**Error Handling**:
- S3 download failures → immediate failure with descriptive error
- Meshing failures → retry with fallback strategies (same as EC2)
- Upload failures → retry with exponential backoff
- Timeout errors → log and return partial results if available

---

### AWS Backend Integration

#### [MODIFY] [api_server.py](file:///c:/Users/markm/Downloads/MeshPackageLean/backend/api_server.py)

**Changes to `run_mesh_generation()` function** (lines 146-384):

**Current flow**:
```python
# Downloads S3 file to local temp
# Spawns subprocess: mesh_worker_subprocess.py
# Streams logs from subprocess stdout
# Parses result JSON and updates database
```

**New flow**:
```python
# 1. Prepare S3 paths (no download needed)
# 2. Invoke Modal function: modal.Function.lookup("khorium-production", "generate_mesh")
# 3. Submit job asynchronously with .spawn()
# 4. Store Modal job ID in database
# 5. Poll for completion or use webhook callback
# 6. Parse result from S3 and update database
```

**Specific changes**:
1. **Import Modal client**: Add `import modal` at top
2. **Replace subprocess invocation**: Replace `subprocess.Popen(...)` with:
   ```python
   mesher = modal.Function.lookup("khorium-production", "generate_mesh")
   job = mesher.spawn(bucket_name, s3_key, quality_params)
   ```
3. **Track Modal job**: Store `job.object_id` in `MeshResult.modal_job_id`
4. **Implement polling**: Create background thread or use Modal webhooks to track completion
5. **Result parsing**: Update to read result files directly from S3 instead of local filesystem

**Changes to `upload_cad_file()` function** (lines 474-627):

**Preview generation**:
- Replace `parse_step_file_for_preview()` subprocess with Modal invocation:
  ```python
  preview_fn = modal.Function.lookup("khorium-production", "generate_preview_mesh")
  preview_job = preview_fn.spawn(bucket_name, s3_key)
  ```

---

#### [MODIFY] [models.py](file:///c:/Users/markm/Downloads/MeshPackageLean/backend/models.py)

**Add Modal job tracking to `MeshResult` model** (around line 111):

```python
class MeshResult(db.Model):
    # ... existing fields ...
    job_id = db.Column(db.String(50), nullable=True)  # Existing: MSH-0101-ABCD
    
    # NEW: Modal compute tracking
    modal_job_id = db.Column(db.String(100), nullable=True, index=True)  # Modal job.object_id
    modal_status = db.Column(db.String(20), nullable=True)  # pending, running, completed, failed
    modal_started_at = db.Column(db.DateTime, nullable=True)
    modal_completed_at = db.Column(db.DateTime, nullable=True)
```

**Database migration**:
- Create migration script to add new columns
- Ensure backward compatibility (nullable fields)

---

### Modal Deployment & Configuration

#### [NEW] [modal_deploy.py](file:///c:/Users/markm/Downloads/MeshPackageLean/scripts/modal_deploy.py)

**Deployment script** to simplify Modal deployments.

**Functionality**:
- Validates Modal CLI is installed (`modal --version`)
- Checks AWS secrets are configured in Modal
- Deploys `mesh_service.py` to Modal Cloud
- Prints deployment URL and function endpoints
- Optionally runs smoke test

**Usage**:
```powershell
python scripts/modal_deploy.py --environment production
```

---

#### [NEW] [modal_client.py](file:///c:/Users/markm/Downloads/MeshPackageLean/backend/modal_client.py)

**Wrapper module** for cleaner Modal integration in `api_server.py`.

**Provides**:
- `submit_mesh_job(bucket, key, quality_params)` → Returns job ID
- `submit_preview_job(bucket, key)` → Returns job ID
- `get_job_status(job_id)` → Returns status object
- `wait_for_completion(job_id, timeout)` → Blocks until complete
- Error handling and retry logic
- Logging integration

**Benefits**:
- Cleaner separation of concerns
- Easier to mock/test
- Centralized Modal configuration

---

### Infrastructure & Configuration

#### [MODIFY] [requirements.txt](file:///c:/Users/markm/Downloads/MeshPackageLean/requirements.txt)

Add Modal client library:
```
modal>=2024.1.0
```

---

#### [MODIFY] [backend/.env](file:///c:/Users/markm/Downloads/MeshPackageLean/backend/.env)

Add Modal configuration:
```bash
# Modal Configuration
USE_MODAL_COMPUTE=false  # Feature flag: false=EC2, true=Modal
MODAL_APP_NAME=khorium-production
MODAL_MESH_FUNCTION=generate_mesh
MODAL_PREVIEW_FUNCTION=generate_preview_mesh
MODAL_API_TOKEN=<secret>  # For API auth
```

**Feature flag strategy**:
- Dev environment: `USE_MODAL_COMPUTE=true` (test thoroughly)
- Production: Once dev validated, flip to `USE_MODAL_COMPUTE=true` (100% traffic instantly)
- Rollback: Flip back to `false` if critical issues detected

---

#### [NEW] [.github/workflows/deploy-modal.yml](file:///c:/Users/markm/Downloads/MeshPackageLean/.github/workflows/deploy-modal.yml)

**GitHub Actions workflow** to auto-deploy Modal service on push to main.

**Steps**:
1. Checkout code
2. Install Modal CLI
3. Authenticate with Modal using GitHub Secret
4. Deploy `mesh_service.py`
5. Run smoke test (simple cube meshing)
6. Notify on Slack if deployment fails

---

### Documentation & ADR

#### [NEW] [docs/adr/0018-modal-compute-migration.md](file:///c:/Users/markm/Downloads/MeshPackageLean/docs/adr/0018-modal-compute-migration.md)

**Architecture Decision Record** documenting the migration.

**Sections**:
1. **Context**: 5-minute EC2 cold start vs 15-second Modal burst
2. **Technical Decision**: Use Modal T4 GPU workers for all meshing
3. **Performance Implications**: 
   - 20x faster cold start
   - 0.9s meshing time (vs ~5-10s on EC2)
   - Auto-scaling (no ASG management)
4. **Cost Trade-offs**: 
   - Modal: Pay-per-second GPU usage
   - EC2: 24/7 instance costs (even when idle)
5. **Rollback Plan**: Feature flag to disable Modal instantly

---

#### [MODIFY] [DEPLOYMENT.md](file:///c:/Users/markm/Downloads/MeshPackageLean/DEPLOYMENT.md)

Add Modal deployment section:

```markdown
## Modal Compute Deployment

### Initial Setup
1. Install Modal CLI: `pip install modal`
2. Authenticate: `modal token new`
3. Create AWS secret: `modal secret create my-aws-secret`

### Deploy Service
```powershell
modal deploy backend/modal_service.py
```

### Verify Deployment
```powershell
modal app list  # Should show khorium-production
modal app logs khorium-production  # View recent logs
```

### Rollback
Set `USE_MODAL_COMPUTE=false` in `.env` and restart gunicorn
```

---

## Verification Plan

### Automated Tests

**Unit Tests** (`tests/test_modal_service.py`):
- Test Modal function signatures
- Mock S3 I/O and verify error handling
- Test quality parameter parsing
- Verify result JSON structure

**Integration Tests** (`tests/test_modal_integration.py`):
- Deploy Modal service to staging
- Submit real meshing job via AWS backend
- Verify S3 file upload/download
- Validate mesh quality metrics match EC2 baseline

**Smoke Test** (in CI/CD):
```python
# Simple cube meshing on Modal
result = modal.Function.lookup("khorium-production", "generate_mesh").call(
    bucket="test-bucket",
    key="test-files/simple_cube.step",
    quality_params={"target_elements": 1000}
)
assert result["success"] == True
assert result["element_count"] > 0
```

### Manual Verification

1. **Dev Environment Testing**:
   - Enable `USE_MODAL_COMPUTE=true` on dev
   - Upload test CAD file
   - Generate mesh and verify viewer display
   - Check logs for Modal job ID

2. **Staging Testing**:
   - Deploy to staging environment
   - Test with 5-10 different CAD files (STEP, STL)
   - Verify preview generation works
   - Test batch processing (if applicable)

3. **Performance Benchmarking** (in dev environment):
   - Measure end-to-end latency:
     - Upload → Modal invocation delay
     - S3 download time on Modal
     - Meshing time (target: <2s)
     - S3 upload time
     - Total time vs EC2 baseline
   - Test concurrent requests (10+ simultaneous jobs)
   - Monitor Modal GPU utilization

4. **Production Deployment**:
   - Flip `USE_MODAL_COMPUTE=true` in production `.env`
   - Restart gunicorn
   - Monitor error rates, latency, and user feedback for first 24 hours
   - Rollback instantly if error rate >5%

### Success Criteria

- ✅ Modal meshing time < 2 seconds (target: 0.9s)
- ✅ End-to-end latency < 20 seconds (vs 5 min EC2 cold start)
- ✅ Quality metrics identical to EC2 baseline (±5%)
- ✅ Zero mesh file corruption or S3 I/O errors
- ✅ Auto-scaling handles 10+ concurrent jobs
- ✅ Cost reduction of ≥40% vs 24/7 EC2 instance

### Rollback Triggers

- Modal error rate > 5%
- Mesh quality degradation > 10% vs EC2
- S3 I/O failures > 2%
- User-reported meshing failures spike
- Cost exceeds EC2 baseline by 20%

### Rollback Procedure

1. Set `USE_MODAL_COMPUTE=false` in production `.env`
2. Restart gunicorn: `sudo systemctl restart gunicorn`
3. Verify EC2 meshing works: Test with sample file
4. Investigate Modal issues in logs

---

## Migration Timeline

| Phase | Duration | Milestone |
|-------|----------|-----------|
| **1. Planning & Design** | ✅ Complete | Implementation plan approved |
| **2. Modal Service Development** | 2 days | `mesh_service.py` complete, unit tests pass |
| **3. AWS Backend Integration** | 1 day | `api_server.py` updated, feature flag working |
| **4. Dev Environment Testing** | 2 days | All tests pass, dev environment verified |
| **5. Production Deployment** | 1 day | Flip feature flag, monitor for 24h |
| **6. EC2 Decommissioning** | 1 day | Remove EC2 dependencies (if successful) |

**Total**: ~1 week end-to-end

---

## Open Questions

1. **Modal API Authentication**: Does the backend need API tokens, or is `Function.lookup()` sufficient with just the Modal CLI auth?
   
2. **Webhook vs Polling**: Should we use Modal webhooks for job completion, or poll `get_job_status()` from a background thread?

3. **Preview I/O Latency**: What's the actual S3 I/O overhead for small files? If >2 seconds, should previews stay on EC2?

4. **Batch Processing**: Does the batch processing feature (`BatchJob` model) also need to migrate to Modal, or is it currently using the same `mesh_worker_subprocess.py`?

5. **Log Streaming**: The current implementation streams logs via `subprocess.stdout`. How do we replicate real-time log streaming from Modal jobs?
