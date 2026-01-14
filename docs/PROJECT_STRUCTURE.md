# MeshPackageLean - Project Structure

## Overview

This document describes the organized project structure for the **MeshPackageLean** project.

---

## Directory Structure

```
MeshPackageLean/
├── core/                      # Core mesh generation engine and logic
│   ├── advanced_geometry.py
│   ├── mesh_generator.py
│   ├── quality.py
│   └── ...
│
├── strategies/                # Meshing strategies and algorithms
│   ├── exhaustive_strategy.py
│   ├── anisotropic_geometry.py
│   └── ...
│
├── apps/                      # Applications
│   ├── desktop/
│   │   └── gui_app/           # Main Desktop GUI
│   │       └── main.py        # Entry point
│   ├── web/                   # Web app entry point
│   └── cli/                   # Command-line tools
│       └── mesh_worker_subprocess.py
│
├── backend/                   # Flask API Server
│   ├── api_server.py          # Main Flask app
│   ├── models.py              # SQLAlchemy models
│   ├── storage.py             # S3/Local storage abstraction
│   ├── modal_service.py       # Modal.com compute backend
│   └── ...
│
├── scripts/                   # Utility, Debug, and Infrastructure Scripts
│   ├── debug/                 # One-off utility and debug scripts
│   │   ├── check_db.py
│   │   ├── inspect_last_project.py
│   │   └── legacy/            # Archived redundant root files
│   ├── infra/                 # Deployment and infrastructure tools
│   │   ├── deploy.ps1
│   │   ├── deploy_env_to_dev.sh
│   │   └── ...
│   ├── run_mesher.py
│   ├── run_local_modal.py
│   └── ...
│
├── config/                    # Configuration Files
│   └── aws/                   # AWS CloudFront, SSM, and WAF configs
│       ├── dev_cf_config.json
│       ├── staging_cf_config.json
│       └── ...
│
├── metadata/                  # Deployment Metadata
│   └── deployment/            # Historical deployment logs and outputs
│       └── promotion_log_*.log
│
├── samples/                   # Sample Data for Testing
│   └── Airfoil_surface.msh
│
├── tools/                     # Testing and Analysis Tools
│   ├── testing/
│   └── visualization/
│
├── docs/                      # Documentation
│   ├── PROJECT_STRUCTURE.md   # This file
│   ├── adr/                   # Architecture Decision Records
│   └── ...
│
├── web-frontend/              # React/Vite Web Interface
│
├── cad_files/                 # Input CAD files
│
├── README.md                  # Main README
├── CHANGELOG.md               # Version history
└── requirements.txt           # Python dependencies
```

---

## Applications

### Desktop GUI

Located in `apps/desktop/gui_app/`.

**Entry Point**: `main.py`

```bash
python apps/desktop/gui_app/main.py
```

**Features**:
- Interactive CAD loading
- Real-time 3D visualization (PyVista)
- Mesh quality analysis
- One-click mesh generation

### Web Interface

**Frontend**: `web-frontend/` (React/Vite)
**Backend**: `backend/` (Flask)

**Start Backend**:
```bash
python backend/api_server.py
```

### CLI Analysis

Scripts located in `scripts/` provide command-line utilities for meshing and fixing files.

**Run Mesher**:
```bash
python scripts/run_mesher.py input.step
```

**Voxel Repair**:
```bash
python scripts/voxel_repair_tool.py input.stl output.stl
```

---

## Development

### Python Requirements

Standardized on **Python 3.11**.

```bash
pip install -r requirements.txt
```

### Testing

Run verification scripts in `tools/testing/`:

```bash
python tools/testing/verify_mesh_quality.py
```

---

**Last Updated**: January 8, 2026
