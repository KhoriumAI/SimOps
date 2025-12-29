# Verification Lab

This directory is the central hub for all testing, verification, and diagnostic scripts.

## Directory Structure

To keep the codebase clean and maintainable, please place scripts into the following subfolders:

- `core_tests/`: Logic tests for core meshing components and algorithms.
- `geometry/`: CAD/STL geometry validation, diagnostics, and triage scripts.
- `integration/`: End-to-end integration tests and full pipeline verifications.
- `export_validation/`: Verification of output formats (MSH, VTK, VTU, Fluent, etc.).
- `gpu_benchmarks/`: Performance testing and benchmarks for GPU-accelerated meshing.
- `utilities/`: Miscellaneous check scripts, header verifiers, and small helper tools.
- `test_data/`: Small sample files used specifically for verification tests.

## Rules
1. **NO CLUTTER**: Do not place scripts directly in the `verification_lab` root.
2. **SUBFOLDERS ONLY**: Create a new subfolder if a set of scripts doesn't fit into the existing categories.
3. **DOCUMENTATION**: If adding a complex test suite, include a local README within the subfolder.

---
*Maintained by the Khorium AI Team*
