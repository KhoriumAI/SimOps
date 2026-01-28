#!/usr/bin/env python3
"""
SimOps Online Installer — cross-platform (Windows, macOS, Linux).

- Checks Docker, finds free ports, optionally pulls images from GHCR (with auth),
  starts docker-compose-online.
- Prompts whether to pull/refresh images (default: yes) to ensure SimOps frontend.
- Optionally checks for OpenFOAM; if missing, offers to install (Mac: Homebrew,
  Linux: apt; Windows: instructions for WSL).
- Waits for backend API to be ready before opening the browser.
"""

from __future__ import annotations

import getpass
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Project root = directory containing this script
ROOT = Path(__file__).resolve().parent
COMPOSE_FILE = ROOT / "docker-compose-online.yml"
LOGS_DIR = ROOT / "logs"
LOG_FILE = None  # set at startup


def log(msg: str, also_console: bool = True) -> None:
    line = msg.rstrip()
    if LOG_FILE:
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass
    if also_console:
        print(line)


def log_err(msg: str) -> None:
    log(f"[ERROR] {msg}")


def log_info(msg: str) -> None:
    log(f"[INFO] {msg}")


def log_debug(msg: str) -> None:
    log(f"[DEBUG] {msg}", also_console=False)


def run(cmd: list[str], capture: bool = True, timeout: int = 300) -> subprocess.CompletedProcess:
    log_debug(f"Running: {' '.join(cmd)}")
    return subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        timeout=timeout,
        cwd=ROOT,
    )


def find_free_port(start: int) -> int:
    for p in range(start, start + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", p))
                return p
        except OSError:
            continue
    return start


def compose_down() -> bool:
    """Stop and remove existing SimOps containers. Returns True if down ran OK."""
    compose = docker_compose_cmd()
    if not compose:
        return True
    r = subprocess.run(
        compose + ["-f", str(COMPOSE_FILE), "down", "--remove-orphans"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    log_debug(f"compose down exit {r.returncode}")
    return r.returncode == 0


def _parse_docker_port_output(stdout: str) -> int | None:
    """Parse '80/tcp -> 0.0.0.0:3000' style output. Returns port or None."""
    for line in stdout.strip().splitlines():
        if "->" in line:
            part = line.split("->")[-1].strip()
            if ":" in part:
                port_str = part.rsplit(":", 1)[-1].strip()
                try:
                    return int(port_str)
                except ValueError:
                    pass
    return None


def frontend_port_from_docker() -> int | None:
    """Get actual frontend host port from running container. None if not running."""
    r = subprocess.run(
        ["docker", "port", "simops-frontend-container"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=5,
    )
    if r.returncode != 0 or not r.stdout:
        return None
    return _parse_docker_port_output(r.stdout)


def check_docker() -> bool:
    """Ensure Docker is installed and in PATH."""
    if not shutil.which("docker"):
        log_err("Docker is not installed or not in PATH.")
        if platform.system() == "Darwin":
            log("Install Docker Desktop for Mac: https://docs.docker.com/desktop/install/mac-install/")
        elif platform.system() == "Windows":
            log("Install Docker Desktop for Windows: https://docs.docker.com/desktop/install/windows-install/")
        else:
            log("Install Docker: https://docs.docker.com/engine/install/")
        return False
    try:
        run(["docker", "--version"])
        return True
    except Exception as e:
        log_err(f"Docker check failed: {e}")
        return False


def docker_compose_cmd() -> list[str]:
    """Prefer 'docker compose' (v2), fall back to 'docker-compose'."""
    if run(["docker", "compose", "version"], capture=True).returncode == 0:
        return ["docker", "compose"]
    if shutil.which("docker-compose"):
        return ["docker-compose"]
    return []


def check_openfoam() -> bool:
    """True if OpenFOAM (cfMesh or snappy) is available."""
    sys_path = list(sys.path)
    try:
        sys.path.insert(0, str(ROOT))
        from strategies.openfoam_hex import check_any_openfoam_available

        return check_any_openfoam_available()
    except Exception:
        pass
    finally:
        sys.path[:] = sys_path

    # Fallback: Mac Homebrew OpenFOAM
    if platform.system() == "Darwin" and shutil.which("brew"):
        try:
            prefix = subprocess.run(
                ["brew", "--prefix", "openfoam"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=ROOT,
            )
            if prefix.returncode != 0:
                return False
            base = prefix.stdout.strip()
            if not base:
                return False
            rc = (Path(base) / "etc" / "bashrc").resolve()
            if not rc.is_file():
                return False
            r = subprocess.run(
                ["bash", "-c", f"source '{rc}' 2>/dev/null; which snappyHexMesh"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=ROOT,
            )
            return r.returncode == 0
        except Exception:
            pass
    return False


def install_openfoam_mac() -> bool:
    """Install OpenFOAM on macOS via Homebrew."""
    if not shutil.which("brew"):
        log_err("Homebrew not found. Install from https://brew.sh")
        return False
    log_info("Installing OpenFOAM via Homebrew (this may take a few minutes)...")
    try:
        r = run(["brew", "install", "openfoam"], capture=False, timeout=1200)
        if r.returncode != 0:
            log_err("Homebrew install failed.")
            return False
        log_info("OpenFOAM installed. Add to your shell profile:")
        log("    source $(brew --prefix openfoam)/etc/bashrc")
        return True
    except subprocess.TimeoutExpired:
        log_err("OpenFOAM install timed out.")
        return False
    except Exception as e:
        log_err(f"OpenFOAM install failed: {e}")
        return False


def install_openfoam_linux() -> bool:
    """Install OpenFOAM on Linux (Debian/Ubuntu) via official repo."""
    if not shutil.which("curl") or not shutil.which("apt-get"):
        log_err("curl and apt-get required. Use your distro's package manager to install OpenFOAM.")
        return False
    log_info("Adding OpenFOAM repo and installing (may prompt for sudo)...")
    try:
        r = run(
            ["bash", "-c", "curl -s https://dl.openfoam.com/add-debian-repo.sh | sudo bash"],
            capture=False,
            timeout=120,
        )
        if r.returncode != 0:
            log_err("Failed to add OpenFOAM repo.")
            return False
        r = run(
            ["sudo", "apt-get", "update"],
            capture=False,
            timeout=120,
        )
        if r.returncode != 0:
            log_err("apt-get update failed.")
            return False
        r = run(
            ["sudo", "apt-get", "install", "-y", "openfoam2406-default"],
            capture=False,
            timeout=600,
        )
        if r.returncode != 0:
            log_err("OpenFOAM package install failed.")
            return False
        log_info("OpenFOAM installed. Source the environment in your shell:")
        log("    source /usr/lib/openfoam/openfoam2406/etc/bashrc")
        return True
    except subprocess.TimeoutExpired:
        log_err("OpenFOAM install timed out.")
        return False
    except Exception as e:
        log_err(f"OpenFOAM install failed: {e}")
        return False


def install_openfoam_windows() -> bool:
    """Print instructions for installing OpenFOAM on Windows (WSL)."""
    log("OpenFOAM on Windows runs inside WSL (Windows Subsystem for Linux).")
    log("")
    log("1. Install WSL2 and Ubuntu:  wsl --install")
    log("2. Open Ubuntu, then run:")
    log("   curl -s https://dl.openfoam.com/add-debian-repo.sh | sudo bash")
    log("   sudo apt-get update && sudo apt-get install -y openfoam2406-default")
    log("   echo 'source /usr/lib/openfoam/openfoam2406/etc/bashrc' >> ~/.bashrc")
    log("")
    log("3. Restart the SimOps installer or your shell, then re-run this script.")
    return False


def offer_install_openfoam() -> None:
    """If OpenFOAM not found, prompt and optionally install."""
    if check_openfoam():
        log_info("OpenFOAM detected.")
        return
    log("")
    log("OpenFOAM not detected. SimOps can run without it (built-in solver only).")
    log("OpenFOAM enables advanced CFD/hex meshing. Install it? [y/N]: ", also_console=True)
    try:
        raw = input().strip().lower()
    except EOFError:
        raw = "n"
    if raw not in ("y", "yes"):
        log_info("Skipping OpenFOAM install.")
        return
    log("")
    if platform.system() == "Darwin":
        install_openfoam_mac()
    elif platform.system() == "Windows":
        install_openfoam_windows()
    else:
        install_openfoam_linux()
    log("")


def ghcr_login() -> bool:
    """Prompt for GitHub PAT and run docker login ghcr.io. Returns True on success."""
    log_info("Not authenticated to GitHub Container Registry.")
    log("You need a Personal Access Token with read:packages scope.")
    try:
        user = input("GitHub username: ").strip()
        if not user:
            log_err("Username required. Skipping pull.")
            return False
        token = getpass.getpass("GitHub PAT (read:packages): ").strip()
        if not token:
            log_err("Token required. Skipping pull.")
            return False
    except EOFError:
        log_err("No input. Skipping pull.")
        return False

    tmp = ROOT / "logs" / f".token_{os.getpid()}"
    try:
        tmp.write_text(token, encoding="utf-8")
        with open(tmp, "r", encoding="utf-8") as f:
            r = subprocess.run(
                ["docker", "login", "ghcr.io", "-u", user, "--password-stdin"],
                stdin=f,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=ROOT,
            )
    except Exception as e:
        log_err(f"Login failed: {e}")
        return False
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass

    if r.returncode != 0:
        log_err("Authentication failed. Skipping pull.")
        return False
    log_info("Authenticated to GHCR.")
    return True


def check_ghcr_auth() -> bool:
    """True if we can access GHCR (e.g. manifest inspect succeeds)."""
    r = run(
        ["docker", "manifest", "inspect", "ghcr.io/khoriumai/simops-frontend:latest"],
        timeout=15,
    )
    return r.returncode == 0


def check_backend_ready(api_port: int, frontend_port: int, max_attempts: int = 60) -> bool:
    """Check if backend API is ready via both direct and frontend proxy. Returns True when ready."""
    import urllib.request
    import urllib.error
    
    # Check backend directly
    backend_url = f"http://localhost:{api_port}/api/health"
    # Check backend via frontend proxy (how the browser accesses it)
    proxy_url = f"http://localhost:{frontend_port}/api/health"
    # Check frontend is serving content
    frontend_url = f"http://localhost:{frontend_port}/"
    
    backend_ready = False
    proxy_ready = False
    frontend_serving = False
    
    for i in range(max_attempts):
        # Check frontend is serving HTML
        if not frontend_serving:
            try:
                with urllib.request.urlopen(frontend_url, timeout=2) as response:
                    if response.status == 200:
                        content = response.read(100).decode('utf-8', errors='ignore')
                        if '<html' in content.lower() or '<!doctype' in content.lower():
                            frontend_serving = True
                            log_debug(f"Frontend serving content (attempt {i+1})")
            except (urllib.error.URLError, OSError):
                pass
        
        # Check direct backend
        if not backend_ready:
            try:
                with urllib.request.urlopen(backend_url, timeout=2) as response:
                    if response.status == 200:
                        backend_ready = True
                        log_debug(f"Backend direct check passed (attempt {i+1})")
            except (urllib.error.URLError, OSError):
                pass
        
        # Check via frontend proxy (how browser accesses it)
        if not proxy_ready:
            try:
                with urllib.request.urlopen(proxy_url, timeout=2) as response:
                    if response.status == 200:
                        proxy_ready = True
                        log_debug(f"Backend proxy check passed (attempt {i+1})")
            except (urllib.error.URLError, OSError):
                pass
        
        # All must be ready
        if backend_ready and proxy_ready and frontend_serving:
            return True
        
        if i < max_attempts - 1:
            _sleep(1)
    
    # Log what failed for debugging
    if not frontend_serving:
        log_debug("Readiness check: frontend not serving HTML")
    if not backend_ready:
        log_debug("Readiness check: direct API /api/health failed")
    if not proxy_ready:
        log_debug("Readiness check: /api/health via frontend proxy failed")
    return False


def main() -> int:
    global LOG_FILE
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE = LOGS_DIR / f"install_{ts}.log"

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("SimOps Online Installer - Install Log\n")
        f.write(f"Started: {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"Log file: {LOG_FILE}\n")
        f.write("-" * 80 + "\n\n")

    log("SimOps Online Installer")
    log("=" * 60)
    log("This script installs SimOps using the latest images from GitHub Registry.")
    log("")
    log(f"Output is logged to: {LOG_FILE}")
    log("")

    # Docker
    log("[1/6] Checking Docker...")
    if not check_docker():
        _pause()
        return 1
    log_info("Docker found.")
    log("")

    # OpenFOAM check + optional install
    log("[2/6] Checking OpenFOAM...")
    offer_install_openfoam()
    log("")

    # Compose file and docker compose
    if not COMPOSE_FILE.is_file():
        log_err(f"docker-compose-online.yml not found in {ROOT}")
        _pause()
        return 1
    compose = docker_compose_cmd()
    if not compose:
        log_err("docker compose / docker-compose not found.")
        _pause()
        return 1

    # Stop existing SimOps so we don't reuse old containers or hit port conflicts
    log("[3/6] Stopping any existing SimOps containers...")
    compose_down()
    log_info("Ready.")
    log("")

    # Ports (check after down; avoid 3000/8000 so we don't collide with MeshGen/Vite dev)
    log("[4/6] Finding available ports...")
    frontend_port = find_free_port(3010)
    api_port = find_free_port(8010)
    log(f"  Frontend: {frontend_port}  API: {api_port}")
    if frontend_port == 3010:
        log_info("Using 3010+ to avoid conflict with local MeshGen/Vite on 3000.")
    log("")

    # Pull (ask user if they want to refresh images)
    log("[5/6] Checking for image updates...")
    env = os.environ.copy()
    env["FRONTEND_PORT"] = str(frontend_port)
    env["API_PORT"] = str(api_port)

    log("Pull/refresh images from registry? [Y/n]: ", also_console=True)
    try:
        raw = input().strip().lower()
    except EOFError:
        raw = "y"
    
    should_pull = raw not in ("n", "no")
    
    if should_pull:
        log_info("Pulling images from registry...")
        if not check_ghcr_auth() and not ghcr_login():
            log_err("Cannot pull without GHCR auth. Continuing with local images.")
            log("If you see MeshGen instead of SimOps, the local image may be wrong.")
        else:
            r = subprocess.run(
                compose + ["-f", str(COMPOSE_FILE), "pull"],
                env=env,
                cwd=ROOT,
                timeout=600,
            )
            if r.returncode != 0:
                log("")
                log_err("Pull failed. Continuing with local images.")
                log("If you see MeshGen instead of SimOps, the local image may be wrong.")
            else:
                log_info("Pulled SimOps images from registry.")
    else:
        log_info("Skipping pull. Using local images.")
    log("")

    # Up
    log("[6/6] Starting SimOps services...")
    r = subprocess.run(
        compose + ["-f", str(COMPOSE_FILE), "up", "-d"],
        env=env,
        cwd=ROOT,
        timeout=120,
    )
    if r.returncode != 0:
        log_err("Failed to start services. Check Docker and ports.")
        _pause()
        return 1

    # Verify frontend is running and resolve actual host port (Docker may differ on WSL2)
    actual_port = None
    for _ in range(5):
        actual_port = frontend_port_from_docker()
        if actual_port is not None:
            break
        _sleep(1)
    if actual_port is None:
        log_err("Frontend container did not start or port mapping unavailable.")
        log("Check: docker ps -a  and  docker compose -f docker-compose-online.yml logs frontend")
        _pause()
        return 1
    log_info("Services started.")
    log("")

    # Wait for backend to be ready (both direct and via frontend proxy) before opening browser
    log("Waiting for backend API and frontend to be ready...")
    if check_backend_ready(api_port, actual_port):
        log_info("Backend and frontend are ready.")
        # Give frontend JavaScript time to complete its own readiness check in the browser
        log("Giving frontend JavaScript time to initialize...")
        _sleep(5)
    else:
        log("Services not fully ready yet, but continuing...")
        log("If the UI shows 'backend not ready': re-run, choose Y to pull images, then try again.")
    log("")

    # Done — use actual port from Docker, not our pre-up guess
    log("Installation complete.")
    workbench_url = f"http://localhost:{actual_port}"
    log(f"  SimOps Workbench:  {workbench_url}")
    log(f"  API:               http://localhost:{api_port}")
    log("")
    if actual_port != frontend_port:
        log_info(f"Frontend bound to port {actual_port} (requested {frontend_port}).")
    log("  (Port 3010+ avoids MeshGen/Vite dev on 3000.)")
    log("")

    _sleep(2)
    log("Opening SimOps Workbench in browser...")
    _open_browser(workbench_url)
    log("")
    log(f"Full log: {LOG_FILE}")
    log("")
    log("If you see MeshGen: re-run and choose to pull images, or try Ctrl+Shift+R.")
    _pause()
    return 0


def _sleep(secs: int) -> None:
    try:
        import time

        time.sleep(secs)
    except Exception:
        pass


def _open_browser(url: str) -> None:
    try:
        if platform.system() == "Darwin":
            subprocess.run(["open", url], check=False, timeout=5)
        elif platform.system() == "Windows":
            os.startfile(url)  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", url], check=False, timeout=5)
    except Exception as e:
        log_debug(f"Could not open browser: {e}")
        log(f"Open in browser: {url}")


def _pause() -> None:
    log("")
    log("Press Enter to close...")
    try:
        input()
    except EOFError:
        pass


if __name__ == "__main__":
    sys.exit(main())
