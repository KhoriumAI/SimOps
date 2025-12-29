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
├── skills/          # Reusable problem-solving and validation scripts
└── tools/           # Utilities and testing
```

## Coding & Architectural Guidelines

To maintain consistency and technical rigor, follow these rules during development:

- **Documentation Template (ADRs):** All major architectural or mathematical decisions (e.g., scheme changes, new solver implementations) must be documented in `docs/adr` using the template below.
- **Folder READMEs:** Every new directory must include a `README.md` explaining its purpose within the SimOps ecosystem.
- **Khorium Skills:** Code that solves generalized problems or provides a reusable "toolbox" should be commented with its purpose and added to the `khorium_skills/` directory.
- **Decision Reference:** Consult `docs/adr/` frequently when stuck or when questioning specific implementation details.

### Decision Record Template
Use this exact template for all new ADRs in `docs/adr/`.

# [Short Title, e.g., Implementation of Roe Solver / Switch to Voronoi Dual]
*Status:* [Proposed | Accepted | Deprecated]
*Date:* YYYY-MM-DD
*Tags:* [e.g., #numerics, #geometry, #optimization, #flux-scheme]

## 1. Context & Problem Statement
The mathematical or architectural constraint driving this decision.
* *The Constraint:* [e.g., The current central difference scheme creates spurious oscillations at shock waves (Gibbs phenomenon).]
* *The Goal:* [e.g., We need a Total Variation Diminishing (TVD) scheme to handle discontinuities.]

## 2. Technical Decision
The specific algorithm or library adopted.
* *Mechanism:* [e.g., Implementing a MUSCL reconstruction with a Minmod limiter.]
* *Dependencies:* [e.g., Requires calculating gradient vectors at cell centers.]

## 3. Mathematical & Physical Implications
Crucial for validity.
* *Conservation:* [e.g., Strictly conservative? Yes/No.]
* *Stability:* [e.g., Reduces max stable CFL from 1.0 to 0.8.]
* *Geometric Constraints:* [e.g., Requires mesh orthogonality > 0.7 or gradients become inaccurate.]

## 4. Performance Trade-offs
* *Compute Cost:* [e.g., Increases flux calculation time by 2x due to reconstruction step.]
* *Memory Cost:* [e.g., Needs to store gradient tensors for every cell.]

## 5. Verification Plan
* *Sanity Check:* [e.g., Sod Shock Tube benchmark.]
* *Regression:* [e.g., Compare residuals against the previous version on the standard nozzle test case.]

## License

MIT License

## Authors

Mesh generation toolkit for research and engineering applications.
