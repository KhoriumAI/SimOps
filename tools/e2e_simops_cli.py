#!/usr/bin/env python3
"""
E2E SimOps CLI – mimics user flow: upload -> Execute Solve -> viewer.

1. POST /api/vendor/upload (files) -> saved_as, preview_url
2. POST /api/simulate { filename: saved_as, config } -> job_id
3. GET /api/job/<job_id> until status success -> results.vtk_url, etc.
4. GET /api/job/<job_id>/vtk -> 200 and VTK body

Usage:
    API_URL=http://localhost:8010 python tools/e2e_simops_cli.py [mesh.msh]
    python tools/e2e_simops_cli.py --start-api [mesh.msh]   # start API locally, then run E2E

Default API_URL: http://localhost:8000 (docker-compose-online). Use 8010+ if using install_online.
"""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
CANDIDATES = [
    "Loft_mesh.msh",
    "Loft_mesh_v2.msh",
    "verification_lab/Cube_test.msh",
    "cad_files/120mm_fan_asm.msh",
    "input/Loft.step",
    "samples/cube.step",
]
if sys.platform != "win32":
    CANDIDATES.extend(["ExampleOF_Cases/Cube_medium_fast_tet.msh"])


def log(msg: str) -> None:
    print(f"  [E2E] {msg}")


def _free_port(start: int = 5099) -> int:
    for p in range(start, start + 50):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("", p))
            s.close()
            return p
        except OSError:
            continue
    return start


def main(api_url: str | None = None) -> int:
    url = api_url or API_URL
    mesh_path = None
    args = [a for a in sys.argv[1:] if a != "--start-api"]
    if args:
        p = Path(args[0])
        if p.exists():
            mesh_path = p
    if not mesh_path:
        for name in CANDIDATES:
            p = Path(name)
            if p.exists():
                mesh_path = p
                break
    if not mesh_path:
        log("No mesh file found. Usage: python e2e_simops_cli.py [path/to/file.msh]")
        return 1

    log(f"Mesh: {mesh_path}")
    log(f"API:  {url}")

    # 1. Health
    try:
        r = requests.get(f"{url}/api/health", timeout=5)
        r.raise_for_status()
        log("Health OK")
    except Exception as e:
        log(f"Health FAIL: {e}")
        return 1

    # 2. Vendor upload
    log("Uploading...")
    try:
        with open(mesh_path, "rb") as f:
            files = {"files": (mesh_path.name, f, "application/octet-stream")}
            r = requests.post(f"{url}/api/vendor/upload", files=files, timeout=60)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log(f"Upload FAIL: {e}")
        resp = getattr(e, "response", None)
        if resp is not None:
            try:
                log(f"  Body: {resp.text[:500]}")
            except Exception:
                pass
        if "404" in str(e) and "vendor/upload" in str(e):
            log("  Tip: Use --start-api to run API from this repo (avoids old Docker images).")
        return 1

    saved_as = data.get("saved_as")
    if not saved_as:
        log("Upload OK but no saved_as in response")
        return 1
    log(f"Upload OK: {saved_as}")

    preview_url = data.get("preview_url")
    if preview_url:
        log(f"Preview URL: {preview_url}")
    else:
        log("No preview_url (optional)")
    # .msh must have preview (meshio path); otherwise we fall back to gmsh and fail in Docker
    if mesh_path.suffix.lower() == ".msh" and not preview_url:
        err = data.get("stl_generation_error") or "unknown"
        log(f"FAIL: .msh upload must return preview_url (meshio). Got: {err}")
        return 1

    # 2b. Viewer check: preview URL must serve VTK for .msh
    if mesh_path.suffix.lower() == ".msh" and preview_url:
        full_preview = f"{url}{preview_url}" if preview_url.startswith("/") else preview_url
        try:
            r = requests.get(full_preview, timeout=15)
            r.raise_for_status()
            buf = r.text[:300]
            if "# vtk DataFile" not in buf and "vertices" not in buf.lower():
                log(f"FAIL: preview URL did not return VTK/JSON: {buf[:200]}")
                return 1
            log("Preview fetch OK (viewer can load)")
        except Exception as e:
            log(f"FAIL: preview fetch: {e}")
            return 1

    # 3. Simulate (builtin)
    config = {
        "solver": "builtin",
        "heat_source_temperature": 373.15,
        "ambient_temperature": 293.15,
        "material": "Aluminum",
        "max_iterations": 2000,
        "tolerance": 1e-8,
    }
    payload = {"filename": saved_as, "config": config}
    log("Execute Solve (POST /api/simulate)...")
    try:
        r = requests.post(
            f"{url}/api/simulate",
            json=payload,
            timeout=30,
        )
        r.raise_for_status()
        sim = r.json()
    except Exception as e:
        log(f"Simulate FAIL: {e}")
        resp = getattr(e, "response", None)
        if resp is not None:
            try:
                log(f"  Body: {getattr(resp, 'text', '')[:500]}")
            except Exception:
                pass
        return 1

    if sim.get("status") != "started" or not sim.get("job_id"):
        log(f"Simulate returned: {sim}")
        return 1
    job_id = sim["job_id"]
    log(f"Job started: {job_id}")

    # 4. Poll /api/job/<id>
    log("Polling /api/job until success...")
    for _ in range(300):
        try:
            r = requests.get(f"{url}/api/job/{job_id}", timeout=10)
            r.raise_for_status()
            j = r.json()
        except Exception as e:
            log(f"Poll FAIL: {e}")
            return 1

        st = j.get("status")
        if st == "success":
            log("Job success")
            results = j.get("results") or {}
            break
        if st == "failed":
            log(f"Job failed: {j.get('error', 'unknown')}")
            return 1
        time.sleep(0.5)
    else:
        log("Poll timeout")
        return 1

    vtk_url = results.get("vtk_url")
    if not vtk_url:
        log("No vtk_url in results")
        return 1
    log(f"VTK URL: {vtk_url}")

    # 5. Fetch VTK (viewer would load this)
    full_vtk = f"{url}{vtk_url}" if vtk_url.startswith("/") else vtk_url
    try:
        r = requests.get(full_vtk, timeout=10)
        r.raise_for_status()
        body = r.text
    except Exception as e:
        log(f"VTK fetch FAIL: {e}")
        return 1

    if "# vtk DataFile" not in body[:200]:
        log("VTK response does not look like VTK")
        return 1
    log("VTK fetch OK (viewer could load this)")

    log("")
    log("E2E OK: upload -> simulate -> poll -> VTK")
    return 0


if __name__ == "__main__":
    do_start_api = "--start-api" in sys.argv
    api_url = None
    proc = None

    if do_start_api:
        root = Path(__file__).resolve().parent.parent
        port = _free_port()
        api_url = f"http://127.0.0.1:{port}"
        log(f"Starting API on port {port} (--start-api)...")
        env = os.environ.copy()
        env["PORT"] = str(port)
        env["FLASK_DEBUG"] = "0"
        proc = subprocess.Popen(
            [sys.executable, "backend/api_server.py"],
            cwd=root,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        for _ in range(60):
            try:
                r = requests.get(f"{api_url}/api/health", timeout=2)
                if r.status_code == 200:
                    log("API ready.")
                    break
            except Exception:
                pass
            time.sleep(0.5)
        else:
            log("API failed to become ready.")
            if proc:
                proc.terminate()
                proc.wait(timeout=5)
            sys.exit(1)

    try:
        code = main(api_url)
    finally:
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    sys.exit(code)
