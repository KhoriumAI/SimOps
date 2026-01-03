# Meshing Strategies

This directory implements specific mesh generation algorithms and strategies.

## Key Strategies

- **exhaustive_strategy.py**: An automatic optimizer that runs multiple strategies and picks the best result based on quality scores.
- **tetgen_strategy.py**: Wrapper for the TetGen library (alternative to Gmsh).
- **pymesh_strategy.py**: Integration with PyMesh for mesh repair and generation.
- **paintbrush_strategy.py**: Logic for localized refinement using "painted" regions.
- **surgical_**.py**: Components of the Surgical Isolation Strategy for handling complex assemblies part-by-part.

## Usage
These strategies are typically instantiated and invoked by the `BaseMeshGenerator` in `core/mesh_generator.py`.
