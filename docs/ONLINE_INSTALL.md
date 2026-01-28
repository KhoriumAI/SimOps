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
- **Docker Desktop** installed and running.
- **Python 3.8+** (used by the installer; [python.org](https://www.python.org/downloads/) or `brew install python` on Mac).
- A **GitHub account** and **Personal Access Token (PAT)** (for pulling images from GHCR).

### Generating a PAT
1. Go to GitHub **Settings** → **Developer settings** → **Personal access tokens** → **Tokens (classic)**.
2. Generate a new token.
3. Select scope: **`read:packages`**.
4. Save the token.

### Installation

1. Download the distribution files (or clone the repo):
   - `docker-compose-online.yml`
   - `install_online.py` (required)
   - **Windows:** `install_online.bat`
   - **Mac / Linux:** `install_online.sh`

2. Run the installer:
   - **Windows:** Double‑click `install_online.bat`, or run `install_online.bat` from a terminal.
   - **Mac / Linux:** In a terminal, from the project directory:
     ```bash
     chmod +x install_online.sh
     ./install_online.sh
     ```
   - **Any OS:** Run `python install_online.py` (or `python3 install_online.py`) from the project directory.

3. When prompted:
   - **Pull/refresh images?** Choose **Y** (default) to get the latest SimOps frontend from the registry, or **n** to use local images.
   - If you choose to pull, enter your **GitHub username** and **PAT** with `read:packages` scope when prompted.

### OpenFOAM (optional)

The installer checks for **OpenFOAM**. SimOps works without it (built‑in solver only). OpenFOAM enables advanced CFD and hex meshing.

- If OpenFOAM is **not** detected, the installer will ask: **Install OpenFOAM? [y/N]**
  - **No (default):** Skip; continue with SimOps only.
  - **Yes:** The installer will attempt to install OpenFOAM:
    - **Mac:** `brew install openfoam` (requires [Homebrew](https://brew.sh)). You may need to add `source $(brew --prefix openfoam)/etc/bashrc` to your shell profile.
    - **Linux:** Adds the OpenFOAM repo and installs `openfoam2406-default` via `apt` (Debian/Ubuntu).
    - **Windows:** Prints step‑by‑step instructions for installing OpenFOAM inside **WSL2** (Ubuntu).

### Updates
The system includes **Watchtower**.
- It checks for updates every hour.
- If the developer pushes a new image to `main`, your instance will automatically download it and restart within an hour.
- No manual action required.

---

## 3. Testing Locally
To test the "Online" mode on your dev machine:
1. Push your latest changes to GitHub.
2. Wait for the Action to complete.
3. Run `install_online.bat` (Windows) or `./install_online.sh` (Mac/Linux) in a **new folder** (copy the script, `install_online.py`, and `docker-compose-online.yml` there) to simulate a fresh client install.
