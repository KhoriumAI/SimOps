import trimesh
import numpy as np
import time
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(PROJECT_ROOT, "robust_soup.stl")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "fixed_soup.stl")

def fast_repair_tool(input_path, output_path):
    start_time = time.time()
    print(f"[FastRepair] Loading {input_path}...")
    
    # 1. Load without processing (Speed up)
    # process=False prevents slow merging on load
    soup = trimesh.load(input_path, force='mesh', process=False)
    
    # 2. Explicitly merge vertices (The #1 fix for "open edges")
    # This snaps vertices that are within 1e-5 distance (floating point errors)
    soup.merge_vertices(merge_tex=True, merge_norm=True)
    
    # 3. Split into bodies
    print("[FastRepair] Splitting bodies...")
    bodies = soup.split(only_watertight=False)
    print(f"[FastRepair] Analyzed {len(bodies)} volumes.")

    repaired_bodies = []
    
    for i, body in enumerate(bodies):
        if body.is_watertight:
            repaired_bodies.append(body)
            continue
            
        # --- THE SPEED FIX ---
        # Don't try loop-finding repair. It hangs.
        # Check if it's just inverted normals first (cheap)
        try:
            trimesh.repair.fix_winding(body)
            trimesh.repair.fix_inversion(body)
        except:
            pass
            
        if body.is_watertight:
            repaired_bodies.append(body)
            print(f"   Shape {i}: Fixed via winding/inversion flip.")
            continue
            
        # If still bad, use CONVEX HULL. 
        # It creates a watertight "wrapper" instantly.
        # This saves the volume existence for the simulation.
        try:
            hull = body.convex_hull
            repaired_bodies.append(hull)
            print(f"   Shape {i}: Was leaky. Replaced with CONVEX HULL (Watertight).")
        except Exception as e:
            # Degenerate geometry (< 4 points, coplanar, etc.)
            # Skip it - it has no volume anyway
            print(f"   Shape {i}: SKIPPED (degenerate: {e})")

    # 4. Export
    print("[FastRepair] Merging and saving...")
    final_mesh = trimesh.util.concatenate(repaired_bodies)
    final_mesh.export(output_path)
    
    elapsed = time.time() - start_time
    return f"SUCCESS: Processed {len(bodies)} volumes in {elapsed:.2f} seconds. Output: {output_path}"

if __name__ == "__main__":
    result = fast_repair_tool(INPUT_PATH, OUTPUT_PATH)
    print(result)
