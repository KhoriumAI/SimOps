# SimOps Online Installation Guide

This guide explains how to deploy, distribute, and install SimOps using the auto-updating GHCR system.

## 1. For Developers (How to Release)

### Prerequisites
- Ensure the repository visibility allows Package storage (Public repos are free, Private require checks).
- The **Actions** permissions must allow Read/Write access to Packages.

### Triggering a Release
Simply push to the `main` branch. The GitHub Action `.github/workflows/publish.yml` will automatically:
1. Build `watcher` and `worker` Docker images.
2. Push them to `ghcr.io/khoriumai/simops-watcher:latest` (and worker).

### Verifying the Build
Go to your GitHub Repository -> **Actions** tab to see the build progress. Once green, the packages will appear in the repository **Packages** sidebar.

---

## 2. For Users (How to Install)

### Prerequisites
- Docker Desktop installed and running.
- A GitHub account and Personal Access Token (PAT).

### Generating a PAT
1. Go to GitHub Settings -> Developer settings -> Personal access tokens -> Tokens (classic).
2. Generate new token.
3. Select scope: `read:packages`.
4. Save the token.

### Installation
1. Download the distribution files (or clone the repo):
   - `docker-compose-online.yml`
   - `scripts/install_online.bat`
2. Double-click `scripts/install_online.bat`.
3. When prompted, enter your GitHub Username and the PAT you generated.

### Updates
The system includes **Watchtower**.
- It checks for updates every hour.
- If the Developer pushes a new image to `main`, your instance will automatically download it and restart within an hour.
- No manual action required.

---

## 3. Testing Locally
To test the "Online" mode on your dev machine:
1. Push your latest changes to GitHub.
2. Wait for Action to complete.
3. Run `scripts/install_online.bat` in a **new folder** (copy the script and yaml there) to simulate a fresh client install.
