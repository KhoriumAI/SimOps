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
│   ├── web-frontend/          # React/Vite Web App
│   └── cli/                   # Command-line tools
│
├── backend/                   # Flask API Server
│   ├── api_server.py
│   └── ...
│
├── scripts/                   # Utility and Maintenance Scripts
│   ├── run_mesher.py
│   ├── voxel_repair_tool.py   # Consolidated tool
│   └── ...
│
├── tools/                     # Testing and Analysis Tools
│   ├── testing/
│   └── visualization/
│
├── docs/                      # Documentation
│   ├── PROJECT_STRUCTURE.md
│   └── ...
│
├── generated_meshes/          # Output directory
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

**Last Updated**: December 27, 2025
