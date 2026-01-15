# Scripts Directory

One-off or recurring scripts for maintenance, setup, and batch verification.

## Key Scripts

- `run_batch_verification.py`: Runs a suite of verification tests across multiple geometries.
- `setup_sweep.py`: Prepares configuration sweeps for sensitivity studies.
- `verify_si_physics.py`: Specialized script for verifying the correctness of physical units (SI system) in simulations.
- `cleanup_test_files.py`: Utility to remove temporary meshes and log files.

## Guidelines

Scripts should be kept self-contained when possible. For more complex tools used by the whole team, consider moving them to the `tools/` directory.
