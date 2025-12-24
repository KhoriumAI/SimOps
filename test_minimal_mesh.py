#!/usr/bin/env python3
"""
Minimal mesh generation test to identify hanging points.
Tests each phase with timeouts and cleanup.
"""
import sys
import os
import signal
import time
from pathlib import Path

# Add MeshPackageLean to path
sys.path.insert(0, str(Path(__file__).parent))

def timeout_handler(signum, frame):
    print(f"\n[X] TIMEOUT after {TIMEOUT}s - Process hung!")
    raise TimeoutError("Operation timed out")

# Set timeout for entire script
TIMEOUT = 30  # 30 seconds max
signal.signal(signal.SIGALRM, timeout_handler)

try:
    print("="*60)
    print("MINIMAL MESH GENERATION TEST")
    print("Testing each phase with timeouts")
    print("="*60)
    
    # Phase 1: Import gmsh
    print("\n[Phase 1] Importing gmsh...")
    signal.alarm(5)
    import gmsh
    signal.alarm(0)
    print("[OK] gmsh imported")
    
    # Phase 2: Initialize gmsh
    print("\n[Phase 2] Initializing gmsh...")
    signal.alarm(5)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("General.NumThreads", 1)  # Single-threaded
    signal.alarm(0)
    print("[OK] gmsh initialized")
    
    # Phase 3: Load CAD file
    print("\n[Phase 3] Loading CAD file...")
    # cad_file = "cad_files/Cube.step"
    cad_file = "cad_files/Loft.step"
    if not os.path.exists(cad_file):
        print(f"[X] File not found: {cad_file}")
        sys.exit(1)
    
    signal.alarm(10)
    gmsh.model.occ.importShapes(cad_file)
    gmsh.model.occ.synchronize()
    signal.alarm(0)
    print(f"[OK] Loaded {cad_file}")
    
    # Phase 4: Set mesh parameters
    print("\n[Phase 4] Setting mesh parameters...")
    signal.alarm(5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 1.0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 5.0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    signal.alarm(0)
    print("[OK] Parameters set")
    
    # Phase 5: Generate 2D mesh (CRITICAL - this often hangs)
    print("\n[Phase 5] Generating 2D mesh...")
    print("  (This is where hangs often occur)")
    signal.alarm(15)
    start_time = time.time()
    gmsh.model.mesh.generate(2)
    elapsed = time.time() - start_time
    signal.alarm(0)
    print(f"[OK] 2D mesh generated in {elapsed:.2f}s")
    
    # Phase 6: Generate 3D mesh
    print("\n[Phase 6] Generating 3D mesh...")
    signal.alarm(15)
    start_time = time.time()
    gmsh.model.mesh.generate(3)
    elapsed = time.time() - start_time
    signal.alarm(0)
    print(f"[OK] 3D mesh generated in {elapsed:.2f}s")
    
    # Phase 7: Save mesh
    print("\n[Phase 7] Saving mesh...")
    output_file = "output/meshes/test_minimal.msh"
    os.makedirs("output/meshes", exist_ok=True)
    signal.alarm(5)
    gmsh.write(output_file)
    signal.alarm(0)
    print(f"[OK] Saved to {output_file}")
    
    # Phase 8: Cleanup
    print("\n[Phase 8] Cleaning up...")
    signal.alarm(5)
    gmsh.finalize()
    signal.alarm(0)
    print("[OK] gmsh finalized")
    
    print("\n" + "="*60)
    print("[OK][OK][OK] ALL PHASES COMPLETED SUCCESSFULLY")
    print("="*60)
    sys.exit(0)
    
except TimeoutError as e:
    print(f"\n[X][X][X] TIMEOUT ERROR: {e}")
    print("Process hung at current phase - forcing cleanup")
    try:
        if gmsh.isInitialized():
            gmsh.finalize()
    except:
        pass
    sys.exit(1)
    
except Exception as e:
    print(f"\n[X][X][X] ERROR: {e}")
    import traceback
    traceback.print_exc()
    try:
        if gmsh.isInitialized():
            gmsh.finalize()
    except:
        pass
    sys.exit(1)
    
finally:
    # Ensure cleanup
    signal.alarm(0)
    try:
        if 'gmsh' in dir() and gmsh.isInitialized():
            gmsh.finalize()
    except:
        pass
