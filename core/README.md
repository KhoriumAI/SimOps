# Core Directory

This is the heart of the MeshGen engine, containing the fundamental logic for geometry processing, mesh generation, and quality analysis.

## Key Components

### Geometry & Healing
- `geometry_healer.py`: Handles hole filling, facet repair, and manifold checks.
- `advanced_geometry.py`: Advanced topological analysis and geometric data structures.
- `geometry_cleanup.py`: Utilities for pre-processing CAD files.

### Mesh Generation
- `mesh_generator.py`: Base classes and orchestration for mesh generation.
- `gpu_mesher.py`: Interface for GPU-accelerated meshing logic.
- `anisotropic_meshing.py`: Support for directional mesh refining.
- `curvature_adaptive.py`: Logic for sizing based on surface curvature.

### Quality & Validation
- `quality.py`: Main suite of quality metrics (SICN, Aspect Ratio, Skewness).
- `mesh_quality_validator.py`: Automated validation of generated meshes.
- `hex_quality_analyzer.py`: Specific metrics for hexahedral and hex-dominant meshes.

### System & Orchestration
- `config.py`: Central configuration management.
- `mesh_worker_pool.py`: Managing parallel worker processes.
- `orchestration/`: Coordination of complex meshing pipelines.

## Development

When adding new core functionality, ensure you follow the structure in `docs/templates/DOCUMENTATION_TEMPLATE.md` and add corresponding tests in the `tests/` directory.
