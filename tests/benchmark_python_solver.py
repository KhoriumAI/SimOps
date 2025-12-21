
import os
import sys
import time
import gmsh
import numpy as np
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from simops_pipeline import ThermalSolver, SimOpsConfig

def generate_box_mesh(filename, size=0.5):
    gmsh.initialize()
    gmsh.model.add("box")
    gmsh.model.occ.addBox(0,0,0, 1,1,1)
    gmsh.model.occ.synchronize()
    
    # Mesh size
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", size)
    
    gmsh.model.mesh.generate(3)
    gmsh.write(filename)
    gmsh.finalize()

def run_benchmark():
    print("="*60)
    print("PYTHON SOLVER BENCHMARK")
    print("="*60)
    
    mesh_file = "temp_bench.msh"
    
    # Generate a moderately dense mesh (~20k-50k nodes ideally for test, but let's start small)
    # size=0.1 -> ~5k nodes
    # size=0.05 -> ~40k nodes
    print("Generating mesh...")
    generate_box_mesh(mesh_file, size=0.08) # ~10k nodes
    
    config = SimOpsConfig(
        heat_source_temperature=373.0,
        ambient_temperature=293.0,
        thermal_conductivity=200.0,
    )
    
    solver = ThermalSolver(config, verbose=True)
    
    print("\nStarting Solve...")
    t0 = time.time()
    result = solver.solve(mesh_file)
    dt = time.time() - t0
    
    print(f"\nTotal Time: {dt:.4f}s")
    print(f"Solve Time: {result['solve_time']:.4f}s")
    print(f"Stats: {result['min_temp']:.1f}K - {result['max_temp']:.1f}K")
    
    if os.path.exists(mesh_file):
        os.remove(mesh_file)
        
    if dt < 2.0:
        print("\n[PASS] Performance is optimal (<2s)")
    else:
        print("\n[WARN] Performance is sluggish (>2s)")

if __name__ == "__main__":
    run_benchmark()
