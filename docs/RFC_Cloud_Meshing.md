# RFC: Cloud Meshing Architecture - SnappyHexMesh/OpenFOAM Port

**Date Created:** 2026-01-05  
**Deadline:** 2026-01-07  
**Author:** Cloud Meshing Team

---

## Executive Summary

This RFC documents the architectural plan for porting SnappyHexMesh (OpenFOAM's hex-dominant mesher) to cloud infrastructure. After evaluating Modal.com serverless and raw AWS EC2 options, we recommend **Modal with a pre-built OpenFOAM Docker image** as the primary approach, with AWS EC2 as a fallback for edge cases.

---

## Background & Context

### Current State
The codebase has two meshing paths:
1. **Gmsh-based meshing** - Already ported to Modal via `modal_service.py`, running on T4 GPU with ~0.9s mesh generation
2. **OpenFOAM/SnappyHexMesh** - Currently in `openfoam_hex.py`, requiring WSL2 on Windows or native Linux

### Problem Statement
Running OpenFOAM on serverless infrastructure involves unknowns:
- **Dependency management**: OpenFOAM has complex system dependencies (MPI, OpenGL, etc.)
- **Container sizes**: Images range 640MB-1.38GB compressed
- **File I/O latency**: Mesh files can be 100MB+ requiring efficient S3 strategies
- **Sandbox compatibility**: OpenFOAM may assume filesystem behaviors not present in gVisor

---

## Feasibility Analysis

### 1. Docker Image Strategy

| Option | Image Size | Cold Start | Recommendation |
|--------|-----------|------------|----------------|
| **`opencfd/openfoam2406-run`** | ~640 MB | 10-20s | ✅ **Recommended** |
| `opencfd/openfoam2406-dev` | ~1.38 GB | 20-40s | ❌ Unnecessary for meshing |
| Custom Dockerfile (slim) | ~500 MB | 8-15s | ⚠️ Maintenance overhead |

**Recommendation**: Use the official `opencfd/openfoam2406-run` image directly. Modal can pull from Docker Hub and cache the image. The "run" variant includes all meshing utilities including `blockMesh`, `snappyHexMesh`, and `foamMeshToFluent` without development tools.

### 2. Modal Compatibility

Modal's architecture is well-suited for OpenFOAM:

| Feature | Modal Support | Notes |
|---------|--------------|-------|
| Docker images from registry | ✅ Yes | `modal.Image.from_registry("opencfd/openfoam2406-run")` |
| Large image handling | ✅ Lazy loading | Files loaded on-demand, not all 640MB upfront |
| Filesystem access | ✅ `/tmp` writable | OpenFOAM case directories work in `/tmp` |
| MPI parallelism | ⚠️ Single-node | Multi-node MPI not supported; single-node parallelism OK |
| GPU acceleration | ❌ Not applicable | SnappyHexMesh is CPU-bound |

**Key Finding**: Modal's lazy-loading filesystem means only the OpenFOAM binaries and libraries needed for `snappyHexMesh` are loaded, not the full 640MB image. This should result in 10-20 second cold starts for the first job, with near-instant warm starts.

### 3. AWS EC2 Comparison

| Metric | Modal | AWS EC2 |
|--------|-------|---------|
| Cold start | 10-20s (first job) | 3-5 min (instance launch) |
| Warm start | <1s | <1s (if instance running) |
| Idle cost | $0 | ~$150/month (c5.xlarge) |
| Scaling | Auto (0→N) | Manual ASG config |
| Maintenance | Low (serverless) | High (patching, AMIs) |
| Complex jobs | Single-node only | Multi-node MPI possible |

**Recommendation**: Use **Modal** for 95% of meshing jobs. Reserve EC2 for edge cases requiring multi-node MPI or specialized hardware.

---

## I/O Strategy

### Mesh File Size Analysis

| Geometry Complexity | Typical Mesh Size | Transfer Time (100 Mbps) |
|---------------------|-------------------|--------------------------|
| Simple (1k cells) | 50 KB | <1s |
| Medium (100k cells) | 5 MB | <1s |
| Complex (1M cells) | 50 MB | 4-5s |
| Very Complex (10M+) | 500 MB+ | 40-50s |

### Proposed I/O Architecture

```
┌─────────────┐         ┌─────────────────┐         ┌──────────────┐
│   Client    │         │      AWS S3     │         │  Modal Worker │
│  (Browser)  │         │  (Mesh Storage) │         │  (OpenFOAM)  │
└──────┬──────┘         └────────┬────────┘         └──────┬───────┘
       │                         │                         │
       │  1. Upload CAD/STL      │                         │
       │ ──────────────────────► │                         │
       │    (Presigned URL)      │                         │
       │                         │   2. Download CAD       │
       │                         │ ◄───────────────────────│
       │                         │   (boto3.download_file) │
       │                         │                         │
       │                         │        3. Run OpenFOAM  │
       │                         │        blockMesh →      │
       │                         │        snappyHexMesh →  │
       │                         │        foamMeshToFluent │
       │                         │                         │
       │                         │   4. Upload .msh + logs │
       │                         │ ◄───────────────────────│
       │                         │   (multipart upload)    │
       │                         │                         │
       │  5. Download mesh       │                         │
       │ ◄────────────────────── │                         │
       │  (Presigned URL)        │                         │
       └─────────────────────────┴─────────────────────────┘
```

### Key I/O Optimizations

1. **Presigned URLs**: Client uploads/downloads directly to S3, bypassing API server
2. **Multipart Upload**: For meshes >100MB, use S3 multipart for reliability
3. **Streaming Logs**: OpenFOAM stdout piped to CloudWatch or returned incrementally via webhooks
4. **Compressed Output**: Generate `.msh.gz` to reduce transfer times by 60-80%
5. **Progress Callbacks**: Frontend polls `/job/{id}/status` for real-time progress

---

## Proof of Concept

### Immediate Validation Goal

Run `blockMesh` on Modal and retrieve the OpenFOAM header string to confirm:
1. Docker image pulls correctly
2. OpenFOAM environment sources properly
3. Filesystem operations work in gVisor sandbox
4. Logs can be captured and returned

### Test Service: `openfoam_poc_service.py`

A minimal Modal function that:
1. Initializes OpenFOAM environment
2. Creates a simple box case (`blockMeshDict`)
3. Runs `blockMesh`
4. Returns stdout with OpenFOAM header

### Expected Output

```
/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.com
    \\  /    A nd           | Version:  v2406
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/
Build  : _openfoam-com Build System
Exec   : blockMesh
```

---

## Decision Summary

| Question | Recommendation |
|----------|----------------|
| **Modal vs AWS EC2?** | Modal (primary), EC2 (fallback for MPI) |
| **Pre-built or custom Docker?** | Pre-built `opencfd/openfoam2406-run` |
| **I/O architecture?** | Client ↔ S3 ↔ Modal with presigned URLs |
| **First milestone?** | `blockMesh` PoC with log capture |

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| gVisor sandbox incompatibility | Medium | High | Test blockMesh first, fallback to EC2 |
| Cold start >30s | Low | Medium | Modal lazy-loading; use warm pool |
| Large mesh I/O timeout | Medium | Medium | Chunked multipart + presigned URLs |
| MPI parallelism needed | Low | Low | Most jobs single-node; EC2 fallback |

---

## Timeline

| Milestone | Target Date | Status |
|-----------|-------------|--------|
| RFC completed | 2026-01-07 | ✅ Complete |
| blockMesh PoC | 2026-01-07 | 🟡 In Progress |
| snappyHexMesh integration | 2026-01-10 | ⏳ Planned |
| Production deployment | 2026-01-14 | ⏳ Planned |

---

## Appendix: Related Files

- `backend/modal_service.py` - Existing Gmsh Modal service
- `strategies/openfoam_hex.py` - Current WSL-based OpenFOAM integration
- `docs/modal_compute_migration_plan.md` - Prior Modal migration documentation
