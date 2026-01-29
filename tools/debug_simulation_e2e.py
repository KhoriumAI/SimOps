"""
Debug E2E Simulation Script
===========================
This script simulates a complete user session to debug simulation issues.
It verifies:
1. Backend connectivity
2. File upload mechanisms
3. OpenFOAM simulation execution (with new hot_wall_face config)
4. Result correctness (element count > 1, convergence, temperature range)

Usage:
    python tools/debug_simulation_e2e.py
"""
import requests
import json
import time
import sys
from pathlib import Path

# Configuration
API_URL = "http://localhost:8000"  # Default backend port (matches restored api_server.py)
# Try to find a valid mesh file
CANDIDATE_FILES = [
    "Loft_mesh_v2.msh",
    "Loft_mesh.msh",
    "cad_files/test.msh",
    "ExampleOF_Cases/Cube_medium_fast_tet.msh"
]

def print_section(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def main():
    print_section("SIMOPS DEBUG E2E - STARTING")
    
    # 1. Health Check
    # -------------------------------------------------------------------------
    print("\n[Step 1] Checking Backend Health...")
    try:
        resp = requests.get(f"{API_URL}/api/health", timeout=5)
        if resp.status_code == 200:
            print(f"  [OK] Backend healthy: {resp.json()}")
        else:
            print(f"  [FAIL] Backend returned {resp.status_code}")
            return
    except Exception as e:
        print(f"  [FAIL] Could not connect to {API_URL}: {e}")
        print("  Please ensure 'python simops-backend/api_server.py' is running.")
        return

    # 2. File Upload
    # -------------------------------------------------------------------------
    print("\n[Step 2] Uploading Test Mesh...")
    mesh_path = None
    for f in CANDIDATE_FILES:
        p = Path(f)
        if p.exists():
            mesh_path = p
            break
            
    if not mesh_path:
        print("  [WARN] No test mesh found. Creating a dummy mock file for API testing...")
        mesh_path = Path("mock_test.msh")
        mesh_path.write_text("Mock mesh data")
        
    print(f"  Using file: {mesh_path}")
    
    upload_filename = None
    try:
        with open(mesh_path, 'rb') as f:
            files = {'files': (mesh_path.name, f, 'application/octet-stream')}
            resp = requests.post(f"{API_URL}/api/upload", files=files)
            
        if resp.status_code == 200:
            data = resp.json()
            upload_filename = data['saved_as']
            print(f"  [OK] Uploaded as: {upload_filename}")
        else:
            print(f"  [FAIL] Upload failed: {resp.text}")
            return
    except Exception as e:
        print(f"  [FAIL] Upload error: {e}")
        return

    # 3. Trigger Simulation (OpenFOAM)
    # -------------------------------------------------------------------------
    print("\n[Step 3] Triggering OpenFOAM Simulation...")
    
    # New Config Structure
    config = {
        "solver": "openfoam",
        "hot_wall_face": "z_min",          # New feature test
        "heat_source_temperature": 373.15, # 100°C
        "ambient_temperature": 293.15,     # 20°C
        "material": "Aluminum",
        "convection_coefficient": 20.0
    }
    
    payload = {
        "filename": upload_filename,
        "config": config
    }
    
    print(f"  Config: {json.dumps(config, indent=2)}")
    
    start_time = time.time()
    try:
        # Note: Current API implementation appears synchronous for simplicity
        # If it becomes async, this script needs polling logic.
        resp = requests.post(f"{API_URL}/api/simulate", json=payload, timeout=600)
        
        if resp.status_code != 200:
            print(f"  [FAIL] Simulation failed: {resp.status_code}")
            print(f"  Response: {resp.text}")
            return
            
        data = resp.json()
        job_id = data.get('job_id')
        results = data.get('results', {})
        
        elapsed = time.time() - start_time
        print(f"  [OK] Simulation completed in {elapsed:.2f}s")
        print(f"  Job ID: {job_id}")
        
    except requests.exceptions.Timeout:
        print("  [FAIL] Simulation timed out (server took too long)")
        return
    except Exception as e:
        print(f"  [FAIL] Error triggering simulation: {e}")
        return

    # 4. Result Validation
    # -------------------------------------------------------------------------
    print_section("RESULT VALIDATION")
    
    # Check Element Count (The Key Bug)
    num_elements = results.get('num_elements', 0)
    print(f"\n1. Mesh Stats:")
    print(f"   Elements: {num_elements}")
    print(f"   Nodes:    {results.get('num_nodes', 0)}")
    
    if num_elements <= 1:
        print("   [FAIL] Element count is suspiciously low (<=1). The '1 element' bug persists.")
    else:
        print("   [PASS] Element count looks realistic.")
        
    # Check Convergence
    converged = results.get('converged', False)
    status_str = "CONVERGED" if converged else "FAILED"
    print(f"\n2. Solver Status: {status_str}")
    
    if not converged:
        print("   [WARN] Solver did not converge.")
    else:
        print("   [PASS] Solver converged successfully.")
        
    # Check Physics (Temperatures)
    min_temp = results.get('min_temp', 0)
    max_temp = results.get('max_temp', 0)
    print(f"\n3. Physics Check:")
    print(f"   Min Temp: {min_temp:.2f} K")
    print(f"   Max Temp: {max_temp:.2f} K")
    
    expected_max = config['heat_source_temperature']
    expected_min = config['ambient_temperature']
    
    # Allow some numerical error
    if abs(max_temp - expected_max) < 5.0 and abs(min_temp - expected_min) < 5.0:
        print("   [PASS] Temperatures match boundary conditions.")
    else:
        print(f"   [WARN] Temperatures deviate from expected BCs ({expected_min}-{expected_max}).")
        
    # Check Files
    print(f"\n4. Output Files:")
    print(f"   PNG: {results.get('png_url')}")
    print(f"   VTK: {results.get('vtk_url')}")
    
    print_section("DEBUG SUMMARY")
    if num_elements > 1 and converged:
        print("SUCCESS: Simulation ran properly with OpenFOAM.")
    else:
        print("FAILURE: Simulation ran but results indicate issues.")

if __name__ == "__main__":
    main()
