# Apps Directory

This directory contains various application interfaces for the Khorium MeshGen project.

## Subdirectories

- **cli/**: Command-line interfaces for batch processing and integration scripts.
  - `mesh_worker.py`: Worker process for mesh generation.
  - `mesh_worker_subprocess.py`: Version optimized for running as a subprocess to handle crashes gracefully.
- **desktop/**: Desktop GUI applications based on PyVista and alternative UI frameworks.
  - `gui_final.py`: The primary desktop interface.
- **monitor/**: Tools for monitoring mesh generation jobs and performance.
- **web/**: Streamlit-based web interfaces (alternative to the React-based `web-frontend`).

## Usage

Each application folder typically has its own requirements and entry point. For the main desktop GUI, use:
```bash
python apps/desktop/gui_final.py
```
