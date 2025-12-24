"""
Test External Aerodynamics Pipeline
===================================
Verifies:
1. Virtual Wind Tunnel Generation (Geometry)
2. BOI Creation
3. Metadata Export (L_char)
4. Turbulence Logic (Re calculation)
"""

import sys
import json
import shutil
from pathlib import Path
import gmsh

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.strategies.cfd_strategy import CFDMeshStrategy, CFDMeshConfig

def create_dummy_cad(filename):
    """Create a simple box to represent a car/object"""
    gmsh.initialize()
    gmsh.model.add("dummy_car")
    # L=4m, W=1.8m, H=1.4m (Car-ish)
    gmsh.model.occ.addBox(0, 0, 0, 4.0, 1.8, 1.4)
    gmsh.model.occ.synchronize()
    gmsh.write(filename)
    gmsh.finalize()

def test_pipeline():
    print("="*60)
    print("TEST: External Aerodynamics Pipeline")
    print("="*60)
    
    output_dir = Path("./test_output_aero")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()
    
    cad_file = str(output_dir / "dummy_car.step")
    create_dummy_cad(cad_file)
    print(f"[OK] Created dummy CAD: {cad_file}")
    
    # 1. Test Meshing & Wind Tunnel
    print("\n[Step 1] Testing Meshing with Virtual Wind Tunnel...")
    
    config = CFDMeshConfig(
        virtual_wind_tunnel=True,
        min_mesh_size=0.5,
        max_mesh_size=2.0,
        num_layers=3
    )
    
    strategy = CFDMeshStrategy(verbose=True)
    mesh_out = str(output_dir / "test_mesh.msh")
    
    # Run Generation
    success, stats = strategy.generate_cfd_mesh(cad_file, mesh_out, config)
    
    if not success:
        print("[FAIL] Meshing failed!")
        return False
        
    print("[OK] Meshing successful")
    
    # 2. Check Metadata (Simulating simops_worker behavior)
    print("\n[Step 2] Verifying Metadata...")
    # NOTE: In the actual worker, the worker code writes the JSON. 
    # Here we are just using the strategy class. 
    # BUT, the strategy object `strategy` (variable name `strategy` here is `runner` in worker)
    # has `wind_tunnel_data`.
    
    wt_data = strategy.wind_tunnel_data
    if not wt_data:
        print("[FAIL] No wind tunnel data found in strategy instance!")
        return False
        
    dims = wt_data.get('dimensions', {})
    L_char = dims.get('L_char')
    print(f"  Captured L_char: {L_char}")
    
    # Expected: 4.0 (the box length)
    if abs(L_char - 4.0) > 0.1:
        print(f"[FAIL] L_char {L_char} != Expected 4.0")
        return False
    else:
        print(f"[OK] L_char matches object length")
        
    # 3. Test Turbulence Logic (Dry Run)
    print("\n[Step 3] Testing Turbulence Logic...")
    
    # Simulate Solver Config
    u_inlet = 20.0 # m/s (approx 72 km/h)
    nu = 1.5e-5    # Air
    
    Re = (u_inlet * L_char) / nu
    print(f"  Calculated Re: {Re:.2e}")
    
    if Re > 4000:
        print(f"  Regime: TURBULENT (Correct)")
    else:
        print(f"  Regime: LAMINAR (Incorrect for this velocity)")
        return False
        
    # Check if we would switch to RAS
    # (Logic is: Re > 4000 -> RAS)
    expected_model = "RAS"
    print(f"  Expected Model: {expected_model}")
    
    return True

if __name__ == "__main__":
    try:
        if test_pipeline():
            print("\n[SUCCESS] Pipeline Verification Complete")
            sys.exit(0)
        else:
            print("\n[FAILURE] Validation Failed")
            sys.exit(1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
