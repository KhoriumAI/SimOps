#!/usr/bin/env python3
"""
Standalone test for mesh_worker_subprocess.py
Mimics exactly how gui_final.py calls the worker.
"""
import sys
import os
import json
import subprocess
import tempfile
import time
from pathlib import Path

# Add MeshPackageLean to path
sys.path.insert(0, str(Path(__file__).parent))

def test_worker():
    print("="*60)
    print("TESTING MESH WORKER SUBPROCESS")
    print("="*60)
    
    # cad_file = "cad_files/Cube.step"
    cad_file = str(Path("cad_files/Cube.step"))
    if not os.path.exists(cad_file):
        print(f"[X] File not found: {cad_file}")
        sys.exit(1)
        
    worker_script = Path("simops_worker.py")
    if not worker_script.exists():
        print(f"[X] Worker script not found: {worker_script}")
        sys.exit(1)
        
    # Quality params (mimic GUI defaults)
    quality_params = {
        "quality_preset": "Medium",
        "max_size_mm": 100,
        "curvature_adaptive": False,
        "mesh_strategy": "Tetrahedral (Delaunay)"
    }
    
    cmd = [sys.executable, str(worker_script), cad_file]
    # cmd.extend(["--quality-params", json.dumps(quality_params)])
    
    print(f"Executing command: {' '.join(cmd)}")
    
    start_time = time.time()
    
    # Use NamedTemporaryFile just like GUI should
    with tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8', delete=False) as log_file:
        log_path = log_file.name
        
    print(f"Log file: {log_path}")
    
    try:
        with open(log_path, 'w') as w_file:
            print("Starting subprocess...")
            process = subprocess.Popen(
                cmd,
                stdout=w_file,
                stderr=subprocess.STDOUT,
                text=True
            )
            print(f"Subprocess started with PID: {process.pid}")
            
            with open(log_path, 'r') as r_file:
                while process.poll() is None:
                    line = r_file.readline()
                    
                    if not line:
                        time.sleep(0.05)
                        # Check for timeout
                        if time.time() - start_time > 120:
                            print("\n[X] TIMEOUT: Process took too long")
                            process.kill()
                            break
                        continue
                        
                    print(f"[WORKER] {line.strip()}")
                    
                # Final read
                for line in r_file:
                    print(f"[WORKER] {line.strip()}")
            
            process.wait()
            print(f"\nProcess finished with return code: {process.returncode}")
            
            if process.returncode == 0:
                print("[OK] SUCCESS")
            else:
                print("[X] FAILED")
    finally:
        if os.path.exists(log_path):
            os.unlink(log_path)

if __name__ == "__main__":
    test_worker()
