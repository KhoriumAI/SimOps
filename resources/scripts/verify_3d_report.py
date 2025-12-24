
import sys
import os
from pathlib import Path
import json

# Add current dir to path
sys.path.append(os.getcwd())

from simops_worker import run_simulation

# Paths
cad_file = r"c:\Users\Owner\Downloads\Simops\input\Cylinder_TrackA.step"
sidecar_file = r"c:\Users\Owner\Downloads\Simops\input\Cylinder_TrackA.json"
output_dir = r"c:\Users\Owner\Downloads\Simops\output"

# Ensure output dir exists
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Create a sidecar for the test if it doesn't exist
sidecar_data = {
    "version": "1.0",
    "physics": {
        "material": "Aluminum 6061",
        "heat_source_temperature": 400.0,
        "ambient_temperature": 300.0,
        "convection_coeff": 150.0,
        "unit_scaling": 1.0
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
