# Strategies Directory

Contains various meshing algorithms and strategy selector logic. Strategies define *how* a mesh is generated using the underlying core tools.

## Available Strategies

- `exhaustive_strategy.py`: Detailed, high-quality search through multiple parameter sets.
- `hex_dominant_strategy.py`: Specialized path for hex-dominant meshing.
- `hxt_strategy.py`: Integration with HXT (High Performance Tetrahedral Meshing).
- `tetgen_strategy.py`: Support for TetGen-based meshing.
- `adaptive_strategy.py`: Dynamically adjusts mesh density based on features.
- `intelligent_strategy_selector.py`: AI-driven or rule-based selection of the best strategy for a given geometry.

## Writing a New Strategy

New strategies should inherit from the base strategy class defined in `core/mesh_generator.py` (when applicable) and provide a `generate()` method. Use `docs/templates/DOCUMENTATION_TEMPLATE.md` to document any new strategy additions.
