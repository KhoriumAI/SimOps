# Compute Backend Abstraction Layer

*Status:* Accepted
*Date:* 2026-01-03
*Tags:* #architecture, #compute, #modal.com, #performance

## 1. Context & Problem Statement

The mesh generation system has two compute paths for preview generation:
1. **AWS EC2 Local** - GMSH runs directly on the EC2 instance
2. **SSH Tunnel to Threadripper** - Requests forwarded via SSH tunnel to a powerful local workstation

**The Problems:**
- The compute path selection is hardcoded in `api_server.py`
- No easy way to benchmark or compare backend performance
- Future migration to Modal.com serverless compute requires architectural changes
- Large CAD files may benefit from more powerful local hardware

**The Goal:** Create a pluggable compute backend abstraction that:
- Allows easy switching between compute providers
- Enables benchmarking across backends
- Prepares the architecture for Modal.com integration

## 2. Technical Decision

Implemented a **Strategy Pattern** for compute backends with the following hierarchy:

```
ComputeBackend (ABC)
├── LocalGMSHBackend      # Runs GMSH on current machine
├── HTTPRemoteBackend     # Generic HTTP endpoint
│   └── SSHTunnelBackend  # Specialized for SSH tunnel (localhost:8080)
└── FallbackBackend       # Tries multiple backends in order
```

**Key Files:**
- `backend/compute_backend.py` - Backend implementations
- `backend/config.py` - Configuration for backend selection
- `scripts/benchmark_compute.py` - Performance testing script

**Configuration:**
```bash
# Environment variables
COMPUTE_BACKEND=auto|local|ssh_tunnel|remote_http
SSH_TUNNEL_PORT=8080
REMOTE_COMPUTE_URL=http://localhost:8080
```

## 3. Future Modal.com Integration

The abstraction enables easy Modal.com integration:

```python
class ModalComputeBackend(HTTPRemoteBackend):
    """Modal.com serverless compute"""
    
    def __init__(self):
        self.modal_app = modal.App("mesh-preview")
        
    def generate_preview(self, cad_file_path: str, timeout: int = 120) -> Dict:
        # Upload to Modal volume
        # Invoke Modal function
        # Return result
        pass
```

Migration path:
1. Deploy GMSH container to Modal.com
2. Add `ModalComputeBackend` to `compute_backend.py`
3. Set `COMPUTE_BACKEND=modal` or update auto-selection logic

## 4. Performance Trade-offs

| Backend | Latency | Throughput | Cost | Reliability |
|---------|---------|------------|------|-------------|
| Local GMSH | Low | Limited by EC2 | EC2 cost | High |
| SSH Tunnel | Network RTT | Limited by local PC | Local PC + EC2 | Requires tunnel |
| Modal.com | Cold start + Network | Auto-scaling | Per-second billing | High (managed) |

## 5. Verification Plan

### Benchmark Script
```bash
python scripts/benchmark_compute.py
```

Tests 4 CAD files of varying sizes:
- Cube.step (8 KB)
- tesla_valve_benchmark_single.step (400 KB)
- 00010009_*.step (1.7 MB)
- ChamboRegina.step (3.7 MB)

Outputs:
- Timing comparison table
- Recommendation for default backend
- JSON results file (`benchmark_results.json`)

### Decision Criteria
If local compute is consistently within 2 seconds of SSH tunnel performance:
- Switch default to `COMPUTE_BACKEND=local`
- Eliminates SSH tunnel dependency
- Simplifies deployment
