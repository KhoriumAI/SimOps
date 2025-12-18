
import sys
import os
from pathlib import Path

print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")

# Add project root to path (simulating mesh_worker_subprocess.py)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
print(f"Project Root: {project_root}")

def check_import(module_name):
    print(f"Testing import: {module_name}...", end=" ", flush=True)
    try:
        __import__(module_name)
        print("OK")
        return True
    except Exception as e:
        print(f"FAILED")
        print(f"  Error: {e}")
        return False

# Test Core Dependencies
print("\n--- Core Dependencies ---")
check_import("numpy")
check_import("gmsh")

# Test Project Modules (simulating what mesh_worker_subprocess imports)
print("\n--- Project Modules ---")
check_import("strategies.exhaustive_strategy")
check_import("strategies.hex_dominant_strategy")
check_import("strategies.conformal_hex_glue")
check_import("strategies.polyhedral_strategy")
check_import("strategies.openfoam_hex")
check_import("core.config")

print("\nDone.")
