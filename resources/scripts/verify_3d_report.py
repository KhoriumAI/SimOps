
import sys
import os
from pathlib import Path
import json

# Add current dir to path
sys.path.append(os.getcwd())

from simops_worker import run_simulation

# Paths
cad_file = r"c:\Users\Owner\Downloads\Simops\temp_test\Verify_CFD.step"
sidecar_file = r"c:\Users\Owner\Downloads\Simops\temp_test\Verify_CFD.json"
output_dir = r"c:\Users\Owner\Downloads\Simops\output"

# Ensure output dir exists
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Create a sidecar for the test if it doesn't exist
sidecar_data = {
    "version": "1.0",
    "physics": {
        "simulation_type": "cfd",
        "inlet_velocity": 45.0,
        "kinematic_viscosity": 1.5e-5,
        "material": "Air"
    }
}
with open(sidecar_file, 'w') as f:
    json.dump(sidecar_data, f, indent=4)


print("="*60)
print("MANUAL VERIFICATION: Professional Report & Physics Fixes")
print("Target: Cylinder.step (Forced Convection)")
print("="*60)

try:
    # Run simulation
    result = run_simulation(cad_file, output_dir)
    print("\n[SUCCESS] Simulation and Report Generation complete!")
    
    dispatch_dir_str = getattr(result, 'dispatch_dir', None)
    if dispatch_dir_str:
        print(f"Result details in: {dispatch_dir_str}")
        dispatch_dir = Path(dispatch_dir_str)
        if dispatch_dir.exists():
            files = list(dispatch_dir.glob("*"))
            print("\nGenerated Artifacts:")
            for f in files:
                print(f" - {f.name}")
except Exception as e:
    print(f"\n[FAILURE] Simulation failed: {e}")
    import traceback
    traceback.print_exc()
