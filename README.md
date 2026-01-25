# SimOps - Thermal Analysis Appliance

A Docker-based thermal simulation service. Drop a STEP file in, get temperature results out.

## What It Does

SimOps is a comprehensive thermal simulation appliance that works in two modes:

1.  **GUI Application**: Interactive mesh generation, visualization, and cloud strategy testing.
2.  **Docker Service**: Watch-folder automation for volume processing.

## Features

- **Exhaustive Strategy Testing**: Automatically tries multiple meshing algorithms to find the best result
- **Quality-Driven**: Optimizes for mesh quality metrics (SICN, aspect ratio, skewness)
- **PyQt5 GUI**: Interactive mesh generation and visualization
- **CAD Support**: Import STEP files for mesh generation
- **VTK Visualization**: Real-time 3D mesh preview with quality coloring
- **Paintbrush Refinement**: Selectively refine specific regions
- **Multiple Algorithms**: Tetrahedral, hexahedral, hybrid meshing strategies
- **HPC Profiling**: Detailed compute latency and sub-process timing logs for Threadripper/EPYC systems
- **Docker Automation**: Watches a folder, auto-meshes, runs thermal analysis, and outputs reports.

## Installation

```bash
# Create conda environment (for GUI/Local)
conda create -n meshing python=3.11
conda activate meshing

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### GUI Application
```bash
cd apps/desktop/gui_app
python main.py
```

### Docker Service (Watch Folder)
```bash
docker-compose up -d
# Drop a STEP file into ./input/
```

### Local Cloud Development
Run code on Modal.com directly from your local machine without pushing to git:
```bash
modal run scripts/run_local_modal.py --input-file path/to/model.step
```
See [docs/local_development.md](docs/local_development.md) for details.

## DevOps Checklist

Before committing, pushing, or deploying, run the unified devops checklist:

```powershell
# Run all checks (recommended before commits)
.\DEVOPS_CHECKLIST.ps1

# Run only pre-commit checks (type safety, schema sync, env vars)
.\DEVOPS_CHECKLIST.ps1 -PreCommit

# Run only pre-push checks (happy path validation)
.\DEVOPS_CHECKLIST.ps1 -PrePush

# Run post-deployment verification
.\DEVOPS_CHECKLIST.ps1 -PostPush -Url https://api.khorium.ai
```

The script consolidates all devops checks from `CONTRIBUTING.md`, `DEPLOYMENT.md`, and validation guides into one executable checklist. See [`DEVOPS_CHECKLIST.ps1`](DEVOPS_CHECKLIST.ps1) for details.

## Requirements

- Python 3.11+
- gmsh, PyQt5, VTK, pyvista, numpy
- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- 4GB RAM minimum

## Quick Start

```bash
# Start the appliance
docker-compose up -d

# Drop a STEP file into the input folder
cp model.step ./input/

# Results appear in ./output/ within 1-5 minutes
```

Monitor jobs at http://localhost:9181

## Output Files

For input `model.step`, you get:

- `model_HighFi_CFD.msh` - Generated mesh
- `model_temperature.png` - Temperature visualization
- `model_thermal.vtk` - VTK file for ParaView
- `model_report.pdf` - PDF summary
- `model_result.json` - Metadata

## Docker Commands

Start:
```bash
docker-compose up -d
```

Stop:
```bash
docker-compose down
```

Rebuild after code changes:
```bash
docker-compose up -d --build
```

View logs:
```bash
docker-compose logs -f worker
docker-compose logs -f watcher
```

Check status:
```bash
docker-compose ps
```

Restart a service:
```bash
docker-compose restart watcher
docker-compose restart worker
```

Scale workers:
```bash
docker-compose up -d --scale worker=8
```

## Configuration

Copy `.env.example` to `.env` and edit as needed.

Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| POLL_INTERVAL | 2 | Folder check interval (seconds) |
| WORKER_TTL | 600 | Job timeout (seconds) |

## Running Without Docker

```bash
pip install -r requirements-worker.txt
python simops_worker.py model.step -o ./output
```

## Troubleshooting

**File not detected**: Delete the file, wait 5 seconds, copy it back. Or restart the watcher:
```bash
docker-compose restart watcher
```

**No containers running**: 
```bash
docker-compose down
docker-compose up -d
```

**Dashboard stuck on loading**:
```bash
docker-compose restart dashboard
```
>>>>>>> origin/main

## Project Structure

```
### Source Code (MeshPackageLean)
```
MeshPackageLean/
├── apps/
│   ├── desktop/     # PyQt5 GUI application
│   ├── web/         # Web app entry point
│   └── cli/         # Command-line tools
├── backend/         # Flask API Server
├── core/            # Core meshing engine
├── strategies/      # Meshing strategy implementations
└── converters/      # Mesh format converters
```

### Docker Services (SimOps)
```
simops/
├── docker-compose.yml      # Orchestration
├── watcher.py              # Folder monitoring
├── simops_worker.py        # Simulation engine
├── input/                  # Watch folder
└── output/                 # Results
```

## Services (Docker Mode)

| Service | Port | Purpose |
|---------|------|---------|
| redis | 6379 | Job queue |
| watcher | - | Monitors input folder |
| worker (x4) | - | Runs simulations |
| dashboard | 9181 | Web UI for job status |
