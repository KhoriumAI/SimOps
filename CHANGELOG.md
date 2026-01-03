# Changelog

All notable changes to the **MeshPackageLean** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Changed
- **Tech Debt Cleanup**: Removed ~4GB of temporary mesh files and large binary artifacts (`.stl`, `.msh` > 100MB).
- **Consolidation**: Unified `voxel_repair_tool` (v2-v7) into a single `voxel_repair_tool.py`.
- **Restructuring**: Moved root-level debugging and verification scripts to `scripts/` and `tools/`.
- **Documentation**: Updated `README.md` to reflect accurate Python 3.11 requirement and new directory structure.
- **Entry Points**: Standardized Desktop GUI entry point to `apps/desktop/gui_app/main.py`.

### Added
- **Quality Metrics**: Added support for Gamma, Skewness, Aspect Ratio, and Minimum Angle in the web visualizer.
- **Deployment Automation**: Integrated automatic CloudFront cache invalidation and expanded backend watch paths.

### Fixed
- **Mesh Quality Visualization**: Resolved issue where only SICN displayed correctly. Implemented robust volume-to-surface quality mapping (node-set intersection heuristic) to ensure surface elements reflect 3D tet quality.
- **API Data Consistency**: Fixed `api_server.py` to merge per-element quality data from worker subprocess results before serving to frontend.

## [1.0.0] - 2025-12-27
### Added
- Core meshing engine with multiple strategies (Exhaustive, Anisotropic, etc.).
- Desktop GUI (PyQt5) for interactive meshing.
- Web frontend (React/Vite) and Flask backend API.
- Support for STEP, STL, and mesh quality analysis.
