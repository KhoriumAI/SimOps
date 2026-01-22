# OpenFOAM Validator Tool

A standalone utility to verify that an OpenFOAM case directory is correctly structured and that the execution environment is ready.

## Quick Start

Run from the repository root:

```powershell
python -m tools.openfoam_validator.main path/to/your/case
```

## Requirements

### 1. Python 3
The script requires Python 3.7+ with no external dependencies (uses standard libraries only).

### 2. OpenFOAM Installation
To pass the **Environment Check**, OpenFOAM must be installed and accessible.

#### On Windows (via WSL - Recommended)
The tool automatically detects OpenFOAM installed within WSL.
1.  Enable WSL and install a distribution (e.g., Ubuntu).
2.  Inside WSL, install OpenFOAM (e.g., [OpenFOAM-com](https://www.openfoam.com/download/install-windows.php) or [OpenFOAM-org](https://openfoam.org/download/windows/)).
3.  **Ensure it is in your path**: If you need to source `etc/bashrc` to use OpenFOAM, add the sourcing command to your `~/.bashrc` file in WSL.

#### On Linux (Native)
Simply ensure OpenFOAM commands like `simpleFoam` or `foamList` are in your `PATH`.

## Validation Checks
The tool performs checks in the following order:

1.  **Structure Check**: 
    - Verifies the existence of `0/`, `constant/`, and `system/` directories.
    - Verifies default dictionary files: `controlDict`, `fvSchemes`, `fvSolution`.
2.  **Environment Check**: 
    - Checks for `foamList` or `simpleFoam` via native shell or WSL.
    - Identifies if commands can be executed.
3.  **Dry Run (mesh check)**:
    - Runs `checkMesh` to ensure the mesh is valid and readable by OpenFOAM.
4.  **Solver Analysis**:
    - Parses `system/controlDict` to determine which solver (e.g., `buoyantSimpleFoam`) the case is intended for.

## Troubleshooting

### Error: "OpenFOAM executables (foamList/simpleFoam) not found"
- **WSL Users**: Open your WSL terminal and run `simpleFoam -help`. If it works there but not in the validator, ensure your `~/.bashrc` correctly sources the OpenFOAM environment variables.
- **Native Users**: Ensure your OpenFOAM installation directory is added to your system `PATH`.
