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
- **Environment Management**: Migrated to GitHub Secrets-driven `.env` generation for secure production configuration.

### Fixed
- **Database Schema Mismatch**: Resolved `UndefinedColumn` error (missing `job_id`) by performing a comprehensive schema update on the AWS RDS PostgreSQL instance. Verified that `mesh_results`, `projects`, and `users` tables are now in sync with the current Python models.
- **Registration 504 Error**: Resolved backend service hangs caused by a missing RDS Security Group (blocking connectivity) and a Python `IndentationError` in the log streaming loop.
- **Login Persistence**: Transitioned Dev backend from volatile SQLite file to persistent PostgreSQL RDS database to prevent data loss during deployments.
- **Database Connectivity**: Added mandatory `sslmode=require` and connection timeouts to robustly handle AWS RDS connections.
- **Promotion Script**: Patched `scripts/promote_to_staging.py` to automatically copy Security Groups from source to target DB instances.
- **Terminal Handling**: Fixed Terminal UI glitches in the web frontend for smoother log rendering.

### Discovered
- **Infrastructure Misconfiguration**: CloudFront distribution E352AHA7L040MU routes frontend to DEV S3 bucket but API calls to STAGING ALB, creating a mixed environment. Documented in ADR-0013.

## [1.0.0] - 2025-12-27
### Added
- Core meshing engine with multiple strategies (Exhaustive, Anisotropic, etc.).
- Desktop GUI (PyQt5) for interactive meshing.
- Web frontend (React/Vite) and Flask backend API.
- Support for STEP, STL, and mesh quality analysis.
