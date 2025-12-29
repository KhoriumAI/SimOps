# CFD Validation Sandbox

## Purpose
This directory contains scripts and artifacts for validating the Computational Fluid Dynamics (CFD) pipeline in SimOps, specifically interacting with OpenFOAM and ensuring physical correctness of fluid flows.

## Contents
- **Scripts**: Python scripts to run benchmarks (e.g., Poiseuille flow, lid-driven cavity).
- **Configs**: JSON configurations for specific CFD test cases.
- **Docs**: Specific verification notes for CFD.


## Prerequisites
- **OpenFOAM**: Must be installed and accessible via `bash` (WSL on Windows). The pipeline executes `gmshToFoam` and standard OpenFOAM solvers.
- **Project Root**: Scripts should be run from the project root.

## Usage
### Run Poiseuille Flow Benchmark
```bash
python tests/cfd_validation/01_poiseuille_flow.py
```

## Expected Behavior
- **Success**: Generates mesh, runs `simpleFoam`, and validates pressure drop (< 10% error).
- **Partial Success (Solver Missing)**: Generates detailed mesh `output/Cylinder_HighFi_Layered.msh` with correct 'inlet', 'outlet', 'wall' tags, but fails at solver stage. 

## Isolation
All temporary output files (logs, case directories, meshes) should be directed to the `output/` directory or a local `temp/` subdirectory within this folder, which is git-ignored. **Do not generate files in the project root.**
