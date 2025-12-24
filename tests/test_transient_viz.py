
import sys
import shutil
import json
import os
from pathlib import Path

sys.path.insert(0, "/app")
# simops_worker imports might fail if dependencies missing, but container has them.
from simops_worker import run_simulation

def test_viz():
    print("Starting Transient Viz Test...")
    
    # Inputs
    # Assuming /app/core/tests/fixtures/Cube.step existed? 
    # Or just use the one we used for pipeline test.
    # Where was Cube.step?
    # In Step 1335: /app/cad_files/Cube.step
    cad_src = Path("/app/cad_files/Cube.step")
    
    test_dir = Path("/output/test_viz")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True)
    
    cad_dst = test_dir / "Cube_Trans.step"
    
    if not cad_src.exists():
        print(f"Source CAD not found: {cad_src}")
        # Try to find any step file
        # Fallback to creating a dummy file? No, need geometry.
        # Assuming Cube.step exists as per previous context.
        return
        
    shutil.copy(cad_src, cad_dst)
    
    # Config
    config = {
        "physics": {
            "transient": True,
            "duration": 5.0,
            "time_step": 1.0, # Explicitly set step size? Adapter calculates it if missing (duration/steps).
            # Let's set steps=5
            "steps": 5, 
            "material": "Aluminum_6061", # Use library
            "heat_load_watts": 1000.0,
            "convection_coeff": 10.0
        }
    }
    
    with open(cad_dst.with_suffix('.json'), 'w') as f:
        json.dump(config, f)
        
    # Run
    # run_simulation determines generic name from filename
    try:
        res = run_simulation(str(cad_dst), str(test_dir))
        print("Simulation completed.")
    except Exception as e:
        print(f"Simulation execution failed: {e}")
        import traceback
        traceback.print_exc()
        
    # Validation
    base = "Cube_Trans"
    png = test_dir / f"{base}_temperature.png"
    ts = test_dir / f"{base}_transient.png"
    pdf = test_dir / f"{base}_report.pdf"
    
    print(f"PNG Exists: {png.exists()}")
    print(f"Transient Plot Exists: {ts.exists()}")
    print(f"PDF Exists: {pdf.exists()}")
    
    if ts.exists():
        print("SUCCESS: Transient visualization generated.")
    else:
        print("FAILURE: Transient visualization missing.")

if __name__ == "__main__":
    test_viz()
