
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, ".")
sys.path.insert(0, "./core")

try:
    print("Testing imports...")
    from core.solvers.calculix_wrapper import CalculiXWrapper
    from core.solvers.openfoam_wrapper import OpenFOAMWrapper
    from simops_pipeline import CalculiXSolver, SimOpsConfig
    print("[OK] Imports successful")
    
    # Test instantiation
    print("Testing instantiation...")
    cw = CalculiXWrapper(verbose=True)
    ofw = OpenFOAMWrapper(verbose=True)
    cs = CalculiXSolver(SimOpsConfig())
    print("[OK] Instantiation successful")
    
    # Test availability check (safe to run)
    print(f"CalculiX Available: {cw.is_available()}")
    print(f"OpenFOAM Available: {ofw.is_available()}")
    
    print("\n[SUCCESS] Solver integration code is valid.")
    
except Exception as e:
    print(f"\n[ERROR] Verification failed: {e}")
    sys.exit(1)
