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

### Command Line
```bash
python scripts/run_mesher.py path/to/model.step
```

## Requirements

- Python 3.11+
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

## Authors

Mesh generation toolkit for research and engineering applications.

## How to Code

To maintain project consistency and knowledge retention, please follow these guidelines:

### 1. Document Major Decisions (ADRs)
When making architectural or mathematical decisions, create a new record in `docs/adr/` using the following template:

```markdown
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
```

### 2. Generalize and Share Skills
When solving a problem that could apply broadly, contribute to the `khorium_skills/toolbox/`:
- Add a comment at the top explaining the generalized utility.
- Reference existing skills when stuck.

### 3. Folder Documentation
Always include a `README.md` inside any new folder to document its purpose.

### 4. Continuous Reference
Reference the `docs/adr/` folder frequently to understand past decisions and maintain alignment with the project's evolution.
