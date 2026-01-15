# SimOps Offline Installer

Professional air-gap Docker Compose installer for SimOps simulation platform.

## Overview

This installer allows you to deploy SimOps on machines without internet access. It packages all Docker images into portable `.tar` files that can be loaded on any machine with Docker Desktop installed.

## Requirements

- **Docker Desktop** (Windows/Mac) or **Docker Engine** (Linux)
- **8GB+ RAM** recommended
- **20GB+ disk space** for images and data

## Quick Start

### For Developers (Building the Package)

Run on a machine with internet access:

```powershell
cd c:\Users\markm\Downloads\Simops
.\scripts\build_offline_package.ps1
```

This creates an `installer/` folder containing:
- `images/` - Docker image tar files (~5-10GB)
- `docker-compose-offline.yml` - Service configuration
- `install.bat` / `install.ps1` - Installation scripts
- `.env.template` - Configuration template
- `update.bat` - For applying future updates

### For End Users (Installing Offline)

1. Copy the entire `installer/` folder to the target machine
2. Install Docker Desktop if not already installed
3. Run `install.bat` (double-click) or `install.ps1` (PowerShell)
4. Open http://localhost:8080 in your browser

## Services

| Service | Port | Description |
|---------|------|-------------|
| Frontend | 8080 | Web application |
| Backend API | 5000 | REST API server |
| Redis | 6379 | Job queue |
| Dashboard | 9181 | Job monitoring |

## Configuration

Edit `.env` before starting to customize:

```ini
# Ports
FRONTEND_PORT=8080
BACKEND_PORT=5000

# Worker scaling
WORKER_REPLICAS=2
WORKER_MEM_LIMIT=4G

# Security
JWT_SECRET_KEY=your-secure-key-here
```

## Updating

To apply updates:

1. Get the new image `.tar` files
2. Replace files in `images/` folder
3. Run `update.bat`

Your data (uploads, configurations) is preserved during updates.

## Troubleshooting

### Services won't start
```bash
docker-compose -f docker-compose-offline.yml logs
```

### Out of disk space
```bash
docker system prune -a
```

### Reset everything
```bash
docker-compose -f docker-compose-offline.yml down -v
docker-compose -f docker-compose-offline.yml up -d
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Browser (:8080)                     │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│              Frontend (Nginx + React)                   │
│                   - Static files                        │
│                   - API proxy                           │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│              Backend (Flask + Gunicorn)                 │
│                   - REST API                            │
│                   - WebSocket                           │
│                   - SQLite DB                           │
└───────────┬─────────────────────────────┬───────────────┘
            │                             │
┌───────────▼───────────┐     ┌───────────▼───────────────┐
│        Redis          │     │     Worker (x N)          │
│    - Job Queue        │◄────┤   - OpenFOAM              │
│    - Pub/Sub          │     │   - CalculiX              │
└───────────────────────┘     │   - Gmsh                  │
                              └───────────────────────────┘
```

## License

MIT License - SimOps Team
