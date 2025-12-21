# SimOps - Thermal Analysis Appliance

A Docker-based thermal simulation service. Drop a STEP file in, get temperature results out.

## What It Does

SimOps watches a folder for CAD files. When you drop a STEP file in, it automatically:
1. Generates a finite element mesh
2. Runs steady-state thermal analysis
3. Outputs temperature visualizations, VTK files, and PDF reports

No configuration required. Failed meshes are automatically retried with different strategies.

## Requirements

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

## Project Structure

```
simops/
├── docker-compose.yml      # Container orchestration
├── Dockerfile.worker       # Worker image
├── Dockerfile.watcher      # Watcher image
├── watcher.py              # Folder monitoring
├── simops_worker.py        # Simulation engine
├── input/                  # Drop files here
├── output/                 # Results appear here
└── core/
    └── strategies/
        └── cfd_strategy.py # Mesh generation
```

## Services

| Service | Port | Purpose |
|---------|------|---------|
| redis | 6379 | Job queue |
| watcher | - | Monitors input folder |
| worker (x4) | - | Runs simulations |
| dashboard | 9181 | Web UI for job status |
