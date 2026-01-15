# Tools Directory

Utility scripts for analysis, benchmarking, and visualization that are not part of the core engine or frontend applications.

## Subdirectories

- **analysis/**: Scripts for parsing mesh formats (MSH, CFD data) and computing deviations.
- **benchmarking/**: Suites for comparing mesh generation performance and quality across different versions.
- **visualization/**: Specialized tools for creating mesh showcases, heatmaps, and high-quality renders.

## Common Tasks

- **Benchmark Run**:
  ```bash
  python tools/benchmarking/benchmark_batch_meshing.py
  ```
- **Mesh Analysis**:
  ```bash
  python tools/analysis/compute_geometric_deviation.py path/to/mesh.msh
  ```
