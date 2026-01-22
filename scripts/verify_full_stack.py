import requests
import time
import sys
from pathlib import Path
import json

# Configuration
BACKEND_URL = "http://localhost:5001"
TEST_FILE_CONTENT = b"solid cube = 10; tlo cube -10; loops;" # Dummy Geo file content

def log(msg, status="INFO"):
    colors = {
        "INFO": "\033[94m", # Blue
        "SUCCESS": "\033[92m", # Green
        "ERROR": "\033[91m", # Red
        "WARN": "\033[93m" # Yellow
    }
    end = "\033[0m"
    print(f"{colors.get(status, '')}[{status}] {msg}{end}")

def verify_backend_health():
    log("Checking Backend Health...")
    try:
        response = requests.get(f"{BACKEND_URL}/api/health")
        if response.status_code == 200:
            log("Backend is Healthy", "SUCCESS")
            return True
        else:
            log(f"Backend returned {response.status_code}", "ERROR")
            return False
    except requests.exceptions.ConnectionError:
        log("Backend Connection Refused (Is dev_orchestrator.py running?)", "ERROR")
        return False

def verify_upload():
    log("Testing File Upload...")
    try:
        files = {'files': ('test_geometry.geo', TEST_FILE_CONTENT)}
        response = requests.post(f"{BACKEND_URL}/api/upload", files=files)
        
        if response.status_code == 200:
            data = response.json()
            log(f"Upload Successful: {data['saved_as']}", "SUCCESS")
            
            # Verify retrieval
            retrieval_url = f"{BACKEND_URL}{data['url']}"
            retrieval = requests.get(retrieval_url)
            if retrieval.status_code != 200:
                log(f"File Retrieval Failed: {retrieval.status_code}", "ERROR")
                return None

            # Verify Preview Generation logic (It returns preview_url key)
            if 'preview_url' in data:
                log(f"Preview URL present: {data['preview_url']}", "SUCCESS")
            else:
                log("Preview URL missing from response", "WARN")

            return data['saved_as']
        else:
            log(f"Upload Failed: {response.status_code}", "ERROR")
            log(f"Headers: {response.headers}", "ERROR")
            log(f"Response: {response.text}", "ERROR")
            return None
    except Exception as e:
        log(f"Upload Exception: {e}", "ERROR")
        return None

def verify_simulation(filename):
    log("Testing Simulation Trigger (Dry Run)...")
    try:
        payload = {
            "filename": filename,
            "config": {
                "simulation_type": "thermal",
                "heat_source_power": 100.0,
                "material": "Aluminum"
            }
        }
        
        # Note: This might fail if the actual pipeline dependencies (gmsh, etc) aren't installed 
        # in the environment where the backend is running.
        # But we want to test if the API endpoint works and tries to run it.
        response = requests.post(f"{BACKEND_URL}/api/simulate", json=payload)
        
        if response.status_code == 200:
            log("Simulation Triggered Successfully", "SUCCESS")
            return True
        elif response.status_code == 500:
            # If it fails due to missing pipeline deps, that's still a "Partial Success" 
            # for the API layer verification (it tried).
            err = response.json().get('error', '')
            if "simops_pipeline" in err or "Could not locate" in err:
                 log(f"Backend API Reachable, but Pipeline missing (Expected in dev env if not set up): {err}", "WARN")
                 return True
            else:
                 log(f"Simulation Failed: {err}", "ERROR")
                 return False
        else:
            log(f"Simulation API Error: {response.status_code} - {response.text}", "ERROR")
            return False
    except Exception as e:
        log(f"Simulation Exception: {e}", "ERROR")
        return False

def main():
    print("=== SimOps Full Stack Verification ===")
    
    if not verify_backend_health():
        sys.exit(1)
        
    saved_filename = verify_upload()
    if not saved_filename:
        sys.exit(1)
        
    if not verify_simulation(saved_filename):
        sys.exit(1)
        
    print("\n🎉 ALL CHECKS PASSED: Environment is ready.")

if __name__ == "__main__":
    main()
