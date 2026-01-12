# Fast Mode (Beta) Documentation

Fast Mode is a set of performance optimizations designed to reduce wait times during CAD visualization and mesh generation. It encompasses two primary systems: **Preview Acceleration** and **High-Performance Meshing (HXT)**.

## 1. Preview Acceleration (Standard)
Optimized preview generation is now the **standard behavior** for all CAD file uploads. The system prioritizes visualization speed by skipping expensive topological healing and using faster compute backends by default.

### How it works:
- **Remote Compute Backend**: The triangulation task is offloaded to a high-performance compute node (via SSH tunnel or Modal) when available.
- **Skipping Topological Checks**: The system disables `Geometry.OCCAutoFix` and uses lenient tolerances (`1e-2`) to avoid the 15-30s delay associated with "healing" CAD geometry that is only being viewed.
- **Lazy Loading**: STL/STEP files are loaded with specialized "fast-path" settings.
- **Average Performance**: 2-8 seconds for complex assemblies.

---

## 2. Meshing Strategy: Tetrahedral (HXT)
When selecting a meshing strategy, **Tetrahedral (HXT)** is the recommended "Fast Mode" for 3D generation.

### Why HXT is faster:
- **Native Parallelism**: Unlike standard Delaunay or Frontal algorithms which are primarily single-threaded during the 3D phase, HXT (High-performance Tetrahedral) is a modern, internally multi-threaded algorithm in Gmsh.
- **Multiprocessing (Surgical Loop)**: For assemblies, the system spawns multiple HXT workers in parallel. Each volume is meshed in its own process, taking full advantage of all available CPU cores.
- **Robustness (Less Retries)**: HXT is significantly more tolerant of "dirty" CAD geometry (gaps, self-intersections). It often succeeds where other algorithms fail, avoiding the need for expensive "Exhaustive Mode" retries.
- **Optimized Defaults**: Fast Mode HXT disables slow secondary optimizations (like Netgen) while maintaining high standard quality metrics, ensuring a fast path to a solver-ready mesh.

---

## Summary Comparison

| Feature | Standard Mode | Fast Mode (HXT/Beta) |
| :--- | :--- | :--- |
| **Preview** | Local Gmsh processing | Remote SSH/Modal Backend |
| **3D Algorithm** | Delaunay (Sequential) | HXT (Parallel) |
| **Assembly Handling** | Sequential Volume Meshing | Parallel Surgical Isolation |
| **CAD Tolerance** | Low (requires precise geometry) | High (handles imperfect CAD) |
| **CPU Utilization** | ~10-15% (Single Core) | ~80-100% (All Cores) |
