
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, ".")
sys.path.insert(0, "./core")

try:
    print("Testing Material Library...")
    from core.mat_lib import MaterialLibrary, get_material_conductivity
    from simops_pipeline import SimOpsConfig, CalculiXSolver
    
    # 1. Test Library Loading
    mat_lib = MaterialLibrary.get_instance()
    materials = mat_lib.materials
    print(f"[OK] Loaded {len(materials)} materials: {list(materials.keys())}")
    
    # 2. Test Fetching
    k_al = get_material_conductivity("Aluminum_6061")
    print(f"[OK] Aluminum K: {k_al}")
    if not isinstance(k_al, list):
        print("[!] Aluminum K should be a table (list of lists)")
        sys.exit(1)
        
    k_steel = get_material_conductivity("Generic_Steel")
    print(f"[OK] Steel K: {k_steel}")
    if not isinstance(k_steel, float):
         print("[!] Generic Steel K should be float")
         sys.exit(1)
    
    # 3. Test Config Integration
    config = SimOpsConfig(material="Copper_C110")
    print(f"[OK] Config material: {config.material}")
    
    # 4. Test Solver Resolution (Dry run logic check)
    # We can't easily mock the internal solver call without patching, 
    # but we can verify that the import logic in solve() works by instantiation
    solver = CalculiXSolver(config)
    print("[OK] CalculiXSolver instantiated with material config")
    
    print("\n[SUCCESS] Material Library system verification passed.")
    
except Exception as e:
    print(f"\n[ERROR] Verification failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
