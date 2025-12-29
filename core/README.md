# Core Library

This directory contains the central business logic and algorithms of the Khorium Mesh Package.

## Key Modules

- **mesh_generator.py**: The base class and orchestrator for mesh generation.
- **geometry_cleanup.py**: Utilities for analyzing and repairing invalid CAD geometry.
- **quality.py**: Centralized mesh quality metric calculations (SICN, Gamma, etc.).
- **gpu_mesher.py**: Interface for GPU-accelerated meshing operations.
- **cad_cleaning/**: Specialized algorithms for deep CAD repair and defeaturing.

## Purpose
This code is environment-agnostic and used by the Desktop App, Web App, and CLI workers.
