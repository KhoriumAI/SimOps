# AWS Migration & Feature Parity Plan (MeshGen Focus)

## Executive Summary
This document outlines the strategy to migrate the **MeshGen** capabilities (Pre-warmed Mesh Generation, GPU Acceleration, Gmsh Integration) to the "Website" architecture (AWS-ready React + Python API). The goal is to bring the robust, high-performance meshing features from the local GUI (`launch_gui.py`) to the web platform.

## 1. Architecture Analysis

### Current "On-Prem" State (MeshGen)
- **Entry Point**: `launch_gui.py` (PyQt5) -> `gui_app.py`.
- **Compute Engine**: `core/mesh_worker_pool.py` managing `apps/cli/mesh_worker_daemon.py`.
- **Key Feature**: **Pre-warming**. The daemon loads heavy libraries (gmsh, numpy, cupy) *before* the user requests a mesh, providing instant feedback.
- **Capabilities**: Gmsh meshing (CPU) + Experimental GPU Meshing (`core/gpu_mesher.py`).

### Current "Website" State (`backend/api_server.py`)
- **Compute**: Spawns a new `subprocess` for *every* request (`mesh_worker_subprocess.py`).
- **Latency**: High startup cost (3-5s overhead per request) due to cold imports.
- **State**: In-memory.
- **Interface**: REST API + React Frontend.

### Target AWS Architecture
- **Frontend**: React (S3 + CloudFront).
- **API Tier**: Python Flask/FastAPI (EC2/Fargate).
- **Worker Tier**: **Long-running Worker Pool** (Dockerized `mesh_worker_daemon`) to maintain pre-warmed state.
- **Queue**: Redis (for job distribution) + Direct Daemon Communication (for warming).

## 2. Gap Analysis

| Feature | On-Prem (MeshGen) | Website (Current) | Gap / Action Required |
| :--- | :--- | :--- | :--- |
| **Execution Model** | **Pre-warmed Daemon** (Instant) | **Cold Subprocess** (Slow) | **CRITICAL**: Port `MeshWorkerPool` logic to the cloud backend. |
| **GPU Support** | Checks for CuPy/CUDA | CPU Only | **Infrastructure**: Update Dockerfile to support NVIDIA runtime (optional for Phase 1). |
| **Job Management** | Direct Pipe (Stdin/Stdout) | `subprocess.Popen` (One-shot) | **Architecture**: Move from "One-shot" to "Persistent Worker" model. |
| **State Management** | Local Filesystem | In-Memory | **Persistence**: Add Database (PostgreSQL) and S3 support. |

## 3. Migration Roadmap

### Phase 1: Local Cloud-Native Environment (Containerization)
**Goal**: Dockerize the MeshGen engine while preserving the "Pre-warming" capability.
1.  **Refactor `mesh_worker_daemon.py`**: Ensure it can run in a Docker container and accept commands via a robust channel (Redis or ZeroMQ) instead of just stdin/stdout. *Note: Stdin/out works for local subprocess, but Redis is better for distributed.* 
    *   *Correction*: For Phase 1, we can keep using `MeshWorkerPool` *inside* the API container for simplicity, or separate it. Let's separate it.
2.  **Create `Dockerfile.meshgen`**:
    *   Base: Python 3.10-slim (or nvidia/cuda for GPU later).
    *   Deps: `gmsh`, `numpy`, `scipy`.
    *   Entrypoint: A modified `worker_service.py` that listens to Redis.
3.  **Create `Dockerfile.api`**: Flask API.
4.  **`docker-compose.yml`**: API + Redis + MeshWorker.

### Phase 2: Backend & Queue Integration
**Goal**: Connect the API to the persistent worker.
1.  **Job Queue**: Implement `RedisQueue` in the API to push jobs.
2.  **Worker Service**: Create a wrapper around `mesh_worker_daemon.py` that pops jobs from Redis and feeds them to the pre-warmed internal daemon (or embeds the daemon logic directly).
    *   *Strategy*: The "Pre-warming" happens once when the container starts. The container *is* the daemon.
3.  **Persistence**: Add `models.py` (SQLAlchemy) for tracking Projects/Meshes.

### Phase 3: S3 & Data Handling
**Goal**: Cloud-ready file storage.
1.  **S3 Integration**: Abstract file path logic. The Worker downloads CAD from S3, meshes to local temp, uploads usage-ready `.msh` and `.vtk` to S3.
2.  **API Updates**: `POST /upload` -> S3. `GET /download` -> S3 presigned URL.

### Phase 4: Frontend Integration
**Goal**: Verify the "Instant" feel on the web.
1.  **Status Polling/WebSockets**: Since meshing is fast (sub-second with pre-warming), simple polling or WebSockets is needed for real-time feedback.
2.  **Visualization**: Ensure the API returns the optimized JSON/VTK data expected by the React Three.js viewer.

## 4. Immediate Next Steps (Verification Plan)
1.  **Containerize**: Create `Dockerfile.meshgen` that runs `mesh_worker_daemon.py` and waits for input.
2.  **Orchestrate**: Setup `docker-compose.cloud.yml` (API + Redis + Worker).
3.  **Verify**: Prove that the second mesh request is significantly faster than the first (verifying pre-warming works in Docker).
