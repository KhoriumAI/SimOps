# Tests Directory

This directory contains automated and manual tests for verifying the mesh generation engine, simulation physics, and API functionality.

## Subdirectories

- **cfd_validation/**: Tests specifically for CFD-related meshing and boundary conditions.
- **physics/**: Physical validation tests for thermal and structural solvers.
- **verification_scripts/**: A collection of scripts used to verify specific features or bug fixes.

## Running Tests

Most tests can be run individually using Python:
```bash
python tests/verification_scripts/my_test_script.py
```

For regression testing, it is recommended to use the scripts in the `tools/testing/` directory (if applicable) or run the batch verification script in the `scripts/` folder.
