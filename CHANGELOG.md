# Changelog

All notable changes to the **MeshPackageLean** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Changed
- **Guardian Behavior**: Modified Geometry Guardian to issue a warning instead of stopping the process when encountering unrepairable (TERMINAL) geometry. This allows Gmsh to attempt meshing on problematic CAD files which often results in successful watertight meshes despite initial health checks.
- **Topology Inspection**: Refactored `TopologyInspector` to use `trimesh` as the primary engine for manifoldness checks, significantly reducing overhead compared to the previous VTK-heavy approach.
- **Ansys Export Compatibility**: Fixed boundary layer naming conventions during export to ensure seamless import into Ansys Fluent and Mechanical.
- **GPU Meshing**: Temporarily disabled GPU-based meshing strategies (HighSpeed GPU) from the GUI selection to prevent failed runs while the environment is being stabilized.
- **Tech Debt Cleanup**: Removed ~4GB of temporary mesh files and large binary artifacts (`.stl`, `.msh` > 100MB).
- **Consolidation**: Unified `voxel_repair_tool` (v2-v7) into a single `voxel_repair_tool.py`.
- **Restructuring**: Moved root-level debugging and verification scripts to `scripts/` and `tools/`.
- **Documentation**: Updated `README.md` to reflect accurate Python 3.11 requirement and new directory structure.
- **Entry Points**: Standardized Desktop GUI entry point to `apps/desktop/gui_app/main.py`.

### Added
- **Usage Tracking**: Implemented comprehensive job usage logging and daily quotas to prevent compute resource abuse.
- **Integrity Guardrails**: Integrated `mypy` type-checking and Alembic schema synchronization checks into the PR pre-flight workflow.
- **Admin Analytics**: Developed a dedicated admin dashboard for monitoring job statistics and user activity.
- **Performance Optimization**: Implemented lazy-loading for VTK and PyVista across the core meshing engine and strategies, reducing worker startup time from 15+ seconds to under 1 second by avoiding unnecessary module scanning.
- **Cloud Meshing (Modal)**: Initial deployment of cloud compute backend for off-device high-performance meshing.
- **Quality Metrics**: Added support for Gamma, Skewness, Aspect Ratio, and Minimum Angle in the web visualizer, including visual heat maps and histograms.
- **Curvature Adaptive Meshing**: Added a specific toggle to dynamically adjust mesh sizing based on surface curvature.
- **Smart Feedback**: Integrated automated Job ID attachment to Slack feedback submissions for faster debugging.
- **Deployment Automation**: Integrated automatic CloudFront cache invalidation and expanded backend watch paths.
- **Environment Management**: Migrated to GitHub Secrets-driven `.env` generation for secure production configuration.

### Fixed
- **Improved Face Selection**: Fixed a regression where clicking a single face would select the entire assembly.
- **Fast Mode Rendering**: Resolved issues where HXT ("Fast Mode") meshes would fail to render in the 3D viewer after completion.
- **VTK Verbosity**: Suppressed verbose "Scanning vtkmodules" debug logs that cluttered the console output during startup.
- **Database Schema Mismatch**: Resolved `UndefinedColumn` error (missing `job_id`) by performing a comprehensive schema update on the AWS RDS PostgreSQL instance. Verified that `mesh_results`, `projects`, and `users` tables are now in sync with the current Python models.
- **Registration 504 Error**: Resolved backend service hangs caused by a missing RDS Security Group (blocking connectivity) and a Python `IndentationError` in the log streaming loop.
- **Login Persistence**: Transitioned Dev backend from volatile SQLite file to persistent PostgreSQL RDS database to prevent data loss during deployments.
- **Database Connectivity**: Added mandatory `sslmode=require` and connection timeouts to robustly handle AWS RDS connections.
- **Promotion Script**: Patched `scripts/promote_to_staging.py` to automatically copy Security Groups from source to target DB instances.
- **Terminal Handling**: Fixed Terminal UI glitches in the web frontend for smoother log rendering.
- **GUI Dependencies**: Resolved startup crashes related to `qtrangeslider` and `psutil`.

### Discovered
- **Infrastructure Misconfiguration**: CloudFront distribution E352AHA7L040MU routes frontend to DEV S3 bucket but API calls to STAGING ALB, creating a mixed environment. Documented in ADR-0013.

## [1.0.0] - 2025-12-27
### Added
- Core meshing engine with multiple strategies (Exhaustive, Anisotropic, etc.).
- Desktop GUI (PyQt5) for interactive meshing.
- Web frontend (React/Vite) and Flask backend API.
- Support for STEP, STL, and mesh quality analysis.
