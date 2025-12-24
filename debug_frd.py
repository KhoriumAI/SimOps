
from pathlib import Path
import os

def check_frd(path):
    print(f"Checking {path}")
    if not path.exists():
        print("  File not found")
        return
        
    with open(path, 'r') as f:
        lines = f.readlines()
        
    print(f"  Total lines: {len(lines)}")
    
    steps = 0
    for line in lines:
        if "100CL" in line:
            print(f"  Found Step Header: {line.strip()}")
            steps += 1
            
    print(f"  Total Steps Found: {steps}")
    
    # Print last 20 lines
    print("  Last 20 lines:")
    for line in lines[-20:]:
        print("    " + line.strip())

base = Path("validation_results")
check_frd(base / "VolumetricSourceBlockCase/mesh.frd")
check_frd(base / "SurfaceFluxSlabCase/mesh.frd")
