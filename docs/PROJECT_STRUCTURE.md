# MeshTest - Project Structure

## Overview

This document describes the organized project structure for the Khorium MeshGen project. All files have been reorganized into logical categories for better maintainability.

---

## Directory Structure

```
MeshTest/
├── core/                      # Core mesh generation engine
│   ├── exhaustive_generator.py
│   ├── advanced_geometry.py
│   ├── anisotropic_meshing.py
│   ├── quality.py
│   └── ...
│
├── strategies/                # Meshing strategies
│   ├── adaptive/
│   │   └── intelligent_sizing.py
│   └── ...
│
├── apps/                      # Applications (NEW)
│   ├── desktop/              # Desktop GUI applications
│   │   ├── gui_final.py
│   │   ├── gui_modern.py
│   │   └── ...
│   ├── web/                  # Web applications
│   │   ├── app_fixed.py     # Fixed version (recommended)
│   │   ├── app_improved.py  # Plotly version
│   │   ├── .streamlit/      # Streamlit config
│   │   └── requirements_web.txt
│   └── cli/                  # Command-line interfaces
│       ├── mesh_worker.py
│       └── mesh_worker_subprocess.py
│
├── tools/                    # Utility tools (NEW)
│   ├── testing/             # Test scripts
│   │   ├── test_airfoil.py
│   │   ├── test_cylinder.py
│   │   ├── test_phase2_phase3.py
│   │   └── ...
│   ├── benchmarking/        # Benchmark scripts
│   │   ├── benchmark_batch_meshing.py
│   │   ├── benchmark_quality_performance.py
│   │   └── ...
│   ├── analysis/            # Analysis utilities
│   │   ├── compute_geometric_deviation.py
│   │   ├── parse_msh41.py
│   │   ├── batch_mesh.py
│   │   ├── gmsh_to_ansys_converter.py
│   │   └── ...
│   └── visualization/       # Visualization tools
│       └── mesh_showcase.py
│
├── docs/                    # Documentation (NEW)
│   ├── bugfixes/           # Bug fix documentation
│   │   ├── ANISOTROPY_FINAL_FIX_NOV8.md
│   │   ├── COMPREHENSIVE_ANISOTROPIC_FIX.md
│   │   ├── INTELLIGENT_SIZING_BUG_FIX.md
│   │   ├── MESH_QUALITY_FIXES_NOV8.md
│   │   ├── THREADING_FIX_SUMMARY.md
│   │   └── ...
│   ├── features/           # Feature implementation docs
│   │   ├── AUTOMATIC_HEALING_IMPLEMENTATION.md
│   │   ├── INTELLIGENT_STRATEGY_SELECTION.md
│   │   ├── PHASE2_PHASE3_IMPLEMENTATION.md
│   │   ├── PDE_ADAPTIVE_REFINEMENT_PLAN.md
│   │   └── ...
│   ├── guides/             # User guides
│   │   ├── README_GUI.md
│   │   ├── GREENSCREEN_GUIDE.md
│   │   ├── MONTAGE_GUIDE.md
│   │   └── ...
│   ├── web/                # Web app documentation
│   │   ├── WEB_APP_FIXES_NOV8.md
│   │   ├── QUICK_START_WEB.md
│   │   ├── README_WEB_FIXES.md
│   │   └── VERCEL_DEPLOYMENT_GUIDE.md
│   └── benchmarks/         # Benchmark reports
│       ├── BENCHMARK_RESULTS_NOV8.md
│       └── ...
│
├── cad_files/              # Test CAD geometries
│   ├── Cube.step
│   ├── Cylinder.step
│   ├── Airfoil.step
│   └── ...
│
├── generated_meshes/       # Output mesh files
├── benchmark_results/      # Benchmark data
├── examples/              # Example scripts
├── experiments/           # Experimental features
│
├── README.md             # Main README
├── requirements.txt      # Python dependencies
└── PROJECT_STRUCTURE.md  # This file
```

---

## Applications

### Desktop GUI

Located in `apps/desktop/`

**Recommended**: `gui_final.py` - Most stable desktop GUI

```bash
python apps/desktop/gui_final.py
```

**Features**:
- Interactive CAD file loading
- Real-time mesh generation
- 3D visualization with PyVista
- Quality analysis
- Mesh export

### Web Interface

Located in `apps/web/`

**Recommended**: `app_fixed.py` - Threading-safe web interface

```bash
cd apps/web
streamlit run app_fixed.py
```

**Features**:
- Browser-based 3D visualization (Plotly)
- Works on Vercel/Heroku/AWS
- No OpenGL dependencies
- Real-time progress tracking
- Mobile-responsive

**Versions**:
- `app_fixed.py` - Threading-safe (use this!)
- `app_improved.py` - Plotly version (has Gmsh threading issue)

### CLI Tools

Located in `apps/cli/`

**Mesh Worker**: Batch mesh generation

```bash
python apps/cli/mesh_worker.py input.step output.msh
```

---

## Tools

### Testing

Located in `tools/testing/`

Run individual tests:
```bash
python tools/testing/test_airfoil.py
python tools/testing/test_cylinder.py
python tools/testing/test_phase2_phase3.py
```

### Benchmarking

Located in `tools/benchmarking/`

**Batch Benchmark**:
```bash
python tools/benchmarking/benchmark_batch_meshing.py
```

Benchmarks multiple geometries and generates reports.

### Analysis

Located in `tools/analysis/`

**Geometric Deviation Analysis**:
```bash
python tools/analysis/compute_geometric_deviation.py mesh.msh
```

**MSH Parser**:
```bash
python tools/analysis/parse_msh41.py mesh.msh
```

**Gmsh to ANSYS Converter**:
```bash
python tools/analysis/gmsh_to_ansys_converter.py input.msh output.cdb
```

**Batch Processing**:
```bash
python tools/analysis/batch_mesh.py --input-dir cad_files/ --output-dir generated_meshes/
```

### Visualization

Located in `tools/visualization/`

**Mesh Showcase Generator**:
```bash
python tools/visualization/mesh_showcase.py mesh.msh
```

Creates animated mesh visualizations with quality heatmaps.

---

## Documentation

### Bug Fixes

Located in `docs/bugfixes/`

Documents all bug fixes including:
- Anisotropic meshing fixes
- Intelligent sizing bug fixes
- Threading fixes
- Camera/shadow fixes
- Quality metric fixes

**Key Files**:
- [THREADING_FIX_SUMMARY.md](docs/bugfixes/THREADING_FIX_SUMMARY.md) - Gmsh threading fix
- [ANISOTROPY_FINAL_FIX_NOV8.md](docs/bugfixes/ANISOTROPY_FINAL_FIX_NOV8.md) - Anisotropic meshing
- [MESH_QUALITY_FIXES_NOV8.md](docs/bugfixes/MESH_QUALITY_FIXES_NOV8.md) - Quality improvements

### Features

Located in `docs/features/`

Implementation documentation for:
- Automatic geometry healing
- Intelligent strategy selection
- Phase 2/3 advanced features
- PDE-based adaptive refinement

**Key Files**:
- [AUTOMATIC_HEALING_IMPLEMENTATION.md](docs/features/AUTOMATIC_HEALING_IMPLEMENTATION.md)
- [PHASE2_PHASE3_IMPLEMENTATION.md](docs/features/PHASE2_PHASE3_IMPLEMENTATION.md)

### Guides

Located in `docs/guides/`

User guides and tutorials:
- GUI usage guide
- Greenscreen/montage creation
- Mesh showcase generation

**Key Files**:
- [README_GUI.md](docs/guides/README_GUI.md) - Desktop GUI guide
- [MONTAGE_GUIDE.md](docs/guides/MONTAGE_GUIDE.md) - Creating mesh montages

### Web Documentation

Located in `docs/web/`

Web application documentation:
- Web app fixes
- Quick start guide
- Vercel deployment

**Key Files**:
- [WEB_APP_FIXES_NOV8.md](docs/web/WEB_APP_FIXES_NOV8.md) - Technical details
- [QUICK_START_WEB.md](docs/web/QUICK_START_WEB.md) - Quick start
- [VERCEL_DEPLOYMENT_GUIDE.md](docs/web/VERCEL_DEPLOYMENT_GUIDE.md) - Deployment

### Benchmarks

Located in `docs/benchmarks/`

Benchmark results and performance reports.

---

## Core Modules

Located in `core/`

### Mesh Generation

- **exhaustive_generator.py** - Main mesh generation engine with automatic healing
- **advanced_geometry.py** - Advanced geometric analysis and healing
- **anisotropic_meshing.py** - Anisotropic mesh strategies

### Quality Analysis

- **quality.py** - `MeshQualityAnalyzer` for SICN, aspect ratio, skewness, etc.

---

## Strategies

Located in `strategies/`

### Adaptive Strategies

- **intelligent_sizing.py** - Intelligent mesh sizing based on curvature and geometry

---

## Quick Start Guide

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For web interface
pip install -r apps/web/requirements_web.txt
```

### 2. Desktop GUI

```bash
python apps/desktop/gui_final.py
```

### 3. Web Interface

```bash
cd apps/web
streamlit run app_fixed.py
```

### 4. CLI Mesh Generation

```bash
python apps/cli/mesh_worker.py cad_files/Cube.step output.msh
```

### 5. Run Tests

```bash
python tools/testing/test_airfoil.py
```

### 6. Run Benchmarks

```bash
python tools/benchmarking/benchmark_batch_meshing.py
```

---

## Development Workflow

### Testing New Features

1. Create test in `tools/testing/`
2. Implement feature in `core/` or `strategies/`
3. Run test to verify
4. Update documentation in `docs/features/`

### Fixing Bugs

1. Create bug report in `docs/bugfixes/`
2. Implement fix in relevant module
3. Add regression test in `tools/testing/`
4. Update fix documentation

### Adding Applications

1. Create new app in `apps/desktop/`, `apps/web/`, or `apps/cli/`
2. Add documentation in `docs/guides/`
3. Update this file

---

## File Naming Conventions

### Documentation

- **Bugs**: `*_FIX*.md`, `*_BUGFIX*.md`
- **Features**: `*_IMPLEMENTATION.md`, `*_PLAN.md`, `*_ROADMAP.md`
- **Guides**: `*_GUIDE.md`, `README_*.md`
- **Reports**: `BENCHMARK_*.md`, `*_REPORT.md`, `*_SUMMARY.md`

### Code

- **Tests**: `test_*.py`
- **Benchmarks**: `benchmark_*.py`
- **GUIs**: `gui_*.py`
- **Apps**: `app_*.py`
- **Workers**: `*_worker*.py`

---

## Dependencies

### Core Dependencies

See [requirements.txt](requirements.txt)

```
numpy
scipy
gmsh
pyvista
vtk
meshio
```

### Web Dependencies

See [apps/web/requirements_web.txt](apps/web/requirements_web.txt)

```
streamlit
plotly
meshio
numpy
gmsh
```

---

## Key Changes from Previous Structure

### Before (Disorganized)
```
MeshTest/
├── test_airfoil.py
├── test_cylinder.py
├── gui_final.py
├── app_fixed.py
├── ANISOTROPY_FINAL_FIX_NOV8.md
├── THREADING_FIX_SUMMARY.md
├── ... 100+ files in root ...
```

**Problem**: Hard to find files, cluttered root directory

### After (Organized)
```
MeshTest/
├── apps/          # Applications
├── tools/         # Utilities
├── docs/          # Documentation
├── core/          # Engine
└── strategies/    # Algorithms
```

**Benefit**: Clear organization, easy navigation

---

## Common Tasks

### Generate a Mesh (Desktop)
```bash
python apps/desktop/gui_final.py
# Load CAD → Configure → Generate → Export
```

### Generate a Mesh (Web)
```bash
cd apps/web
streamlit run app_fixed.py
# Upload CAD → Configure → Generate → Download
```

### Generate a Mesh (CLI)
```bash
python apps/cli/mesh_worker.py input.step output.msh \
  --size 3.0 --quality high
```

### Run All Tests
```bash
for test in tools/testing/test_*.py; do
  python "$test"
done
```

### Benchmark Multiple Geometries
```bash
python tools/benchmarking/benchmark_batch_meshing.py
```

### Create Mesh Visualization
```bash
python tools/visualization/mesh_showcase.py generated_meshes/Airfoil_mesh.msh
```

### Convert to ANSYS Format
```bash
python tools/analysis/gmsh_to_ansys_converter.py mesh.msh mesh.cdb
```

---

## Migration Guide

If you have scripts that reference old file locations, update them:

### Old → New Mappings

| Old Location | New Location |
|-------------|-------------|
| `test_*.py` | `tools/testing/test_*.py` |
| `benchmark_*.py` | `tools/benchmarking/benchmark_*.py` |
| `gui_*.py` | `apps/desktop/gui_*.py` |
| `app*.py` | `apps/web/app*.py` |
| `mesh_worker*.py` | `apps/cli/mesh_worker*.py` |
| `*_FIX*.md` | `docs/bugfixes/*_FIX*.md` |
| `*_GUIDE.md` | `docs/guides/*_GUIDE.md` |
| `WEB_*.md` | `docs/web/WEB_*.md` |

### Update Import Paths

If you have Python imports that reference moved modules:

```python
# OLD (won't work)
from mesh_worker import generate_mesh

# NEW (correct)
import sys
sys.path.append('apps/cli')
from mesh_worker import generate_mesh
```

Or add the project root to PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:/Users/animeneko/Downloads/MeshTest"
```

---

## Contributing

### Adding New Features

1. Implement in `core/` or `strategies/`
2. Add tests in `tools/testing/`
3. Document in `docs/features/`
4. Update this file if adding new directories

### Fixing Bugs

1. Fix code in relevant module
2. Add regression test in `tools/testing/`
3. Document in `docs/bugfixes/`

### Adding Documentation

- Bug fixes: `docs/bugfixes/`
- Features: `docs/features/`
- Guides: `docs/guides/`
- Web: `docs/web/`
- Benchmarks: `docs/benchmarks/`

---

## Support

### Documentation

- **Main README**: [README.md](README.md)
- **This Guide**: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- **Bug Fixes**: [docs/bugfixes/](docs/bugfixes/)
- **Features**: [docs/features/](docs/features/)
- **Guides**: [docs/guides/](docs/guides/)
- **Web**: [docs/web/](docs/web/)

### Test Files

- Test CAD files: `cad_files/`
- Example meshes: `generated_meshes/`

---

## Summary

### New Structure Benefits:

1. ✅ **Clear Organization** - Files grouped by function
2. ✅ **Easy Navigation** - Find what you need quickly
3. ✅ **Logical Categories** - Apps, tools, docs separated
4. ✅ **Maintainable** - Easy to add new files in right place
5. ✅ **Scalable** - Structure supports project growth

### Key Directories:

- **apps/** - Run applications (desktop, web, CLI)
- **tools/** - Use utilities (testing, benchmarking, analysis, visualization)
- **docs/** - Read documentation (bugfixes, features, guides, web)
- **core/** - Core mesh generation engine
- **strategies/** - Meshing strategies and algorithms

---

**Date**: November 8, 2025
**Status**: ✅ COMPLETE - Project reorganized
**Next**: Use new structure for all development

---

## Quick Reference Card

```bash
# Desktop GUI
python apps/desktop/gui_final.py

# Web Interface (Fixed Version)
streamlit run apps/web/app_fixed.py

# CLI Mesh Generation
python apps/cli/mesh_worker.py input.step output.msh

# Run Tests
python tools/testing/test_airfoil.py

# Run Benchmarks
python tools/benchmarking/benchmark_batch_meshing.py

# Analyze Mesh
python tools/analysis/compute_geometric_deviation.py mesh.msh

# Create Visualization
python tools/visualization/mesh_showcase.py mesh.msh

# Documentation
open docs/web/QUICK_START_WEB.md
```

---

End of PROJECT_STRUCTURE.md
