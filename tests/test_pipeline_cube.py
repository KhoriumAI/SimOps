
import sys
import os
from pathlib import Path

# Add app path for imports
sys.path.insert(0, "/app")

from simops_worker import run_simulation, SimulationResult

def test_pipeline():
    cad_file = "/app/cad_files/Cube.step"
    output_dir = "/output/cube_test"
    
    print(f"Running pipeline on {cad_file}...")
    
    try:
        result = run_simulation(cad_file, output_dir)
        
        if result.success:
            print("[PASS] Simulation Pipeline Success")
            print(f"Strategy: {result.strategy_name}")
            print(f"Elements: {result.num_elements}")
            print(f"Temp Range: {result.min_temp:.1f}K - {result.max_temp:.1f}K")
            sys.exit(0)
        else:
            print("[FAIL] Simulation Pipeline Failed")
            print(result.error)
            sys.exit(1)
            
    except Exception as e:
        print(f"[CRITICAL] Pipeline Exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_pipeline()
