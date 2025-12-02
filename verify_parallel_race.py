import sys
import os
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategies.exhaustive_strategy import ExhaustiveMeshGenerator
from core.config import Config

def verify_parallel_race():
    print("="*60)
    print("VERIFYING PARALLEL MESHING RACE CONDITION")
    print("="*60)
    
    # Setup paths
    base_dir = Path(__file__).parent
    cad_file = base_dir / "cad_files" / "Cube.step"
    output_file = base_dir / "test_output" / "parallel_race_result.msh"
    
    # Ensure output dir exists
    output_file.parent.mkdir(exist_ok=True)
    
    if not cad_file.exists():
        print(f"[!] Error: CAD file not found: {cad_file}")
        return
        
    print(f"Input: {cad_file}")
    print(f"Output: {output_file}")
    
    # Initialize generator
    config = Config()
    generator = ExhaustiveMeshGenerator(config)
    
    # Run generation
    start_time = time.time()
    success = generator.generate_mesh(str(cad_file), str(output_file))
    end_time = time.time()
    
    duration = end_time - start_time
    
    print("\n" + "="*60)
    print(f"RESULT: {'SUCCESS' if success else 'FAILURE'}")
    print(f"Duration: {duration:.2f} seconds")
    print("="*60)
    
    if success:
        print(f"Mesh saved to: {output_file}")
        if output_file.exists():
            print(f"File size: {output_file.stat().st_size} bytes")
        else:
            print("[!] Output file missing!")
            
    # Check logs for parallel execution indicators
    # (We can't easily check stdout here as it's printed, but the user will see it)
    print("\nCheck the logs above for:")
    print("1. 'EXHAUSTIVE MESH GENERATION STRATEGY (PARALLEL RACE)'")
    print("2. 'Starting parallel pool with X workers'")
    print("3. 'WINNER FOUND'")
    print("4. 'Terminating other strategies...'")

if __name__ == "__main__":
    # Fix for multiprocessing on Windows
    import multiprocessing
    multiprocessing.freeze_support()
    verify_parallel_race()
