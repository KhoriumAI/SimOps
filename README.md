# MeshPackageLean

Advanced 3D mesh generation toolkit with intelligent strategy selection and quality optimization.

## Features

- **Exhaustive Strategy Testing**: Automatically tries multiple meshing algorithms to find the best result
- **Quality-Driven**: Optimizes for mesh quality metrics (SICN, aspect ratio, skewness)
- **PyQt5 GUI**: Interactive mesh generation and visualization
- **CAD Support**: Import STEP files for mesh generation
- **VTK Visualization**: Real-time 3D mesh preview with quality coloring
- **Paintbrush Refinement**: Selectively refine specific regions
- **Multiple Algorithms**: Tetrahedral, hexahedral, hybrid meshing strategies

## Installation

```bash
# Create conda environment
conda create -n meshing python=3.12
conda activate meshing

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### GUI Application
```bash
cd apps/desktop
python gui_final.py
```

### Command Line
```bash
python batch_mesh.py path/to/model.step
```

## Requirements

- Python 3.12+
- gmsh
- PyQt5
- VTK
- pyvista
- numpy

## Project Structure

```
MeshPackageLean/
├── apps/
│   ├── desktop/     # PyQt5 GUI application
│   └── cli/         # Command-line tools
├── core/            # Core meshing functionality
├── strategies/      # Meshing strategy implementations
├── converters/      # Mesh format converters
└── tools/           # Utilities and testing
```

## License

MIT License

## Authors

Mesh generation toolkit for research and engineering applications.
