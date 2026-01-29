#!/usr/bin/env python3
"""
SimOps Fast Installer — Minimal installer that just pulls and starts.
No checks, no prompts, no OpenFOAM detection. Just pull images and start.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Project root = directory containing this script
ROOT = Path(__file__).resolve().parent
COMPOSE_FILE = ROOT / "docker-compose-online.yml"


def find_free_port(start: int) -> int:
    import socket
    for p in range(start, start + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", p))
                return p
        except OSError:
            continue
    return start


def docker_compose_cmd() -> list[str]:
    """Prefer 'docker compose' (v2), fall back to 'docker-compose'."""
    if subprocess.run(["docker", "compose", "version"], capture_output=True).returncode == 0:
        return ["docker", "compose"]
    import shutil
    if shutil.which("docker-compose"):
        return ["docker-compose"]
    return []


def check_docker_running() -> bool:
    """Check if Docker daemon is running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return False


def main() -> int:
    print("SimOps Fast Installer")
    print("=" * 60)
    print("Pulling images and starting services...")
    print("")

    # Check Docker is running
    print("[0/2] Checking Docker...")
    if not check_docker_running():
        print("[ERROR] Docker Desktop is not running.")
        print("")
        print("Please:")
        print("  1. Start Docker Desktop")
        print("  2. Wait for it to fully start (whale icon in system tray)")
        print("  3. Run this installer again")
        print("")
        input("Press Enter to close...")
        return 1
    print("  Docker is running.")
    print("")

    if not COMPOSE_FILE.is_file():
        print(f"[ERROR] docker-compose-online.yml not found in {ROOT}")
        input("Press Enter to close...")
        return 1

    compose = docker_compose_cmd()
    if not compose:
        print("[ERROR] docker compose / docker-compose not found.")
        input("Press Enter to close...")
        return 1

    # Find ports
    frontend_port = find_free_port(3010)
    api_port = find_free_port(8010)
    print(f"Using ports - Frontend: {frontend_port}  API: {api_port}")
    print("")

    env = os.environ.copy()
    env["FRONTEND_PORT"] = str(frontend_port)
    env["API_PORT"] = str(api_port)

    # Stop any existing containers
    print("[1/3] Stopping existing containers...")
    subprocess.run(
        compose + ["-f", str(COMPOSE_FILE), "down", "--remove-orphans"],
        cwd=ROOT,
        capture_output=True,
        timeout=60,
    )
    print("  Done.")
    print("")

    # Pull and start
    print("[2/3] Pulling images...")
    r = subprocess.run(
        compose + ["-f", str(COMPOSE_FILE), "pull"],
        env=env,
        cwd=ROOT,
        timeout=600,
    )
    if r.returncode != 0:
        print("[WARNING] Pull failed. Continuing with local images.")
        print("")
    else:
        print("  Images pulled successfully.")
        print("")

    print("[3/3] Starting services...")
    r = subprocess.run(
        compose + ["-f", str(COMPOSE_FILE), "up", "-d"],
        env=env,
        cwd=ROOT,
        timeout=120,
    )
    if r.returncode != 0:
        print("[ERROR] Failed to start services.")
        print("")
        print("Troubleshooting:")
        print("  - Make sure Docker Desktop is running")
        print("  - Check if ports 3010 and 8010 are already in use")
        print("  - Try: docker compose -f docker-compose-online.yml logs")
        print("")
        input("Press Enter to close...")
        return 1
    
    print("  Services started.")
    print("")

    print("")
    print("Installation complete.")
    print(f"  SimOps Workbench:  http://localhost:{frontend_port}")
    print(f"  API:               http://localhost:{api_port}")
    print("")
    print("Services are starting...")
    print("(Wait a few seconds for containers to be ready)")
    print("")
    input("Press Enter to close...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
