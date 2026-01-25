# SimOps Online Installation Guide

This guide explains how to install SimOps using the "Online" method, which includes automatic updates.

## Prerequisites
- **Docker Desktop** installed and running.
- **Internet connection** (for pulling images).

## Quick Start (For Users)

1.  **Download** the following files to a folder on your computer (e.g., `SimOps/`):
    - `docker-compose-online.yml`
    - `scripts/install_online.bat`
    - `.env.template` (rename to `.env` if you need to customize ports)

2.  **Run** the installer:
    - Double-click `install_online.bat`.

3.  **Access** the application:
    - Web App: http://localhost:8080
    - Dashboard: http://localhost:9181

## Revisions & Updates

This installation uses **Watchtower** to keep your application up to date automatically.

- **Check Frequency**: Every hour.
- **Mechanism**: 
    1.  The system checks GitHub Container Registry for a newer version of `simops-backend`, `simops-frontend`, or `simops-worker`.
    2.  If found, it pulls the new image.
    3.  It restarts the service seamlessly.

## Troubleshooting

### "Headless" Mode (Server)
If you are running on a server without a GUI, you can run the following command directly:

```bash
docker-compose -f docker-compose-online.yml up -d
```

### Force an Update Immediately
To force a check for updates right now, run:

```bash
docker-compose -f docker-compose-online.yml pull
docker-compose -f docker-compose-online.yml up -d
```
