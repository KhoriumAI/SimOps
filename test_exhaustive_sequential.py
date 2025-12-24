#!/usr/bin/env python3
"""
Test ExhaustiveMeshGenerator in sequential mode.
Verifies that the strategy logic itself doesn't hang.
"""
import sys
import os
import signal
import time
from pathlib import Path

# Add MeshPackageLean to path
sys.path.insert(0, str(Path(__file__).parent))

from core.config import Config
from strategies.exhaustive_strategy import ExhaustiveMeshGenerator

def timeout_handler(signum, frame):
    print(f"\n[X] TIMEOUT after {TIMEOUT}s - Process hung!")
    raise TimeoutError("Operation timed out")

# Set timeout
TIMEOUT = 60  # 60 seconds max
signal.signal(signal.SIGALRM, timeout_handler)

try:
    print("="*60)
    print("EXHAUSTIVE STRATEGY SEQUENTIAL TEST")
    print("="*60)
    
    # cad_file = "cad_files/Cube.step"
    cad_file = "cad_files/Loft.step"
    if not os.path.exists(cad_file):
        print(f"[X] File not found: {cad_file}")
        sys.exit(1)
        
    # Create config
    config = Config()
    
    # Initialize generator in SEQUENTIAL mode
    print("\nInitializing ExhaustiveMeshGenerator (use_parallel=False)...")
    generator = ExhaustiveMeshGenerator(config, use_parallel=False)
    print("[OK] Generator initialized")
    
    # Run generation
    print(f"\nGenerating mesh for {cad_file}...")
    output_file = "output/meshes/test_exhaustive.msh"
    os.makedirs("output/meshes", exist_ok=True)
    
    signal.alarm(TIMEOUT)
    start_time = time.time()
    
    result = generator.generate_mesh(cad_file, output_file)
    
    elapsed = time.time() - start_time
    signal.alarm(0)
    
    if result.success:
        print(f"\n[OK] SUCCESS in {elapsed:.2f}s")
        print(f"  Score: {result.score:.2f}")
        print(f"  Strategy: {result.strategy_name}")
        print(f"  Output: {result.output_file}")
    else:
        print(f"\n[X] FAILED: {result.message}")
        sys.exit(1)
        
    print("\n" + "="*60)
    print("[OK][OK][OK] TEST COMPLETED SUCCESSFULLY")
    print("="*60)
    sys.exit(0)

except TimeoutError as e:
    print(f"\n[X][X][X] TIMEOUT ERROR: {e}")
    sys.exit(1)
except Exception as e:
    print(f"\n[X][X][X] ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
