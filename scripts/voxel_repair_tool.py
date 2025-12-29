import trimesh
import numpy as np
import time
import os

def repair_tool_v7_hybrid(input_path, output_path):
    start_time = time.time()
    print(f"[HybridFix] Loading {input_path}...")
    
    if not os.path.exists(input_path):
        return f"FAILURE: Input file {input_path} does not exist."

    # 1. WELD (Crucial for the "Exploded" mesh)
    try:
        scene_or_mesh = trimesh.load(input_path, force='mesh') 
        if isinstance(scene_or_mesh, trimesh.Scene):
             soup = scene_or_mesh.dump(concatenate=True)
        else:
             soup = scene_or_mesh
    except Exception as e:
        return f"FAILURE: Could not load mesh: {e}"

    print(f"   Shape Loading: {len(soup.faces)} faces.")
    print(f"   -> Pre-Weld Faces: {len(soup.faces)}")
    soup.merge_vertices(merge_tex=True, merge_norm=True)
    print(f"   -> Post-Weld Faces: {len(soup.faces)}")
    
    bodies = soup.split(only_watertight=False)
    print(f"[HybridFix] Identified {len(bodies)} distinct regions after welding.")

    repaired_bodies = []
    
    # CONFIGURATION
    SMALL_PART_THRESHOLD = 500.0  # mm^3. Below this, we assume it's a simple pin/chip.
    VOXEL_PITCH_FINE = 0.05       # 50 microns for large complex parts
    MAX_VOXELS = 40_000_000      # RAM Safety
    
    hull_count = 0
    voxel_count = 0
    skipped_dust = 0
    watertight_count = 0
    
    for i, body in enumerate(bodies):
        # A. Dust Filter
        if len(body.faces) < 20: 
            skipped_dust += 1
            continue 

        if body.is_watertight:
            repaired_bodies.append(body)
            watertight_count += 1
            continue
            
        # B. Analyze Geometry
        # We need an estimate. Hull volume is a safe upper bound.
        try:
            hull = body.convex_hull
            vol_approx = hull.volume
        except:
            vol_approx = 0 # Point cloud/Line junk
            skipped_dust += 1
            continue
        
        # --- PATH 1: SMALL PARTS -> CONVEX HULL ---
        if vol_approx < SMALL_PART_THRESHOLD:
            # Pins, resistors, capacitors are usually convex.
            # Hulling them guarantees safety and negligible volume error.
            repaired_bodies.append(hull)
            hull_count += 1
            # print(f"   Shape {i} (Small): Hulled.")
            
        # --- PATH 2: LARGE PARTS -> VOXEL REMESH ---
        else:
            # The Board, Heatsink, etc.
            # We need to preserve concavities (holes, cutouts).
            print(f"   Shape {i} (Large, Vol ~{vol_approx:.1f}mm^3): Voxelizing...")
            try:
                dims = body.extents
                # RAM Safety Check
                estimated_voxels = (dims[0] * dims[1] * dims[2]) / (VOXEL_PITCH_FINE ** 3)
                
                target_pitch = VOXEL_PITCH_FINE
                if estimated_voxels > MAX_VOXELS:
                    target_pitch = ( (dims[0] * dims[1] * dims[2]) / MAX_VOXELS ) ** (1/3)
                    print(f"      [Warning] Resolving pitch to {target_pitch:.4f}mm for RAM safety.")

                voxel_grid = body.voxelized(pitch=target_pitch)
                remesh = voxel_grid.marching_cubes
                trimesh.repair.fix_inversion(remesh)
                trimesh.repair.fix_normals(remesh)
                
                if remesh.is_watertight:
                    repaired_bodies.append(remesh)
                    voxel_count += 1
                    print(f"      -> SUCCESS (Faces: {len(remesh.faces)})")
                else:
                    # Fallback
                    repaired_bodies.append(hull)
                    hull_count += 1
                    print(f"      -> Voxel failed watertightness. Fallback to Hull.")
            except Exception as e:
                 print(f"      -> ERROR: {e}. Fallback to Hull.")
                 repaired_bodies.append(hull)
                 hull_count += 1

    # Export
    print(f"[HybridFix] Concatenating {len(repaired_bodies)} bodies...")
    if repaired_bodies:
        final_mesh = trimesh.util.concatenate(repaired_bodies)
        print(f"[HybridFix] Exporting to {output_path}...")
        final_mesh.export(output_path)
    else:
        return "FAILURE: No bodies preserved."
    
    elapsed = time.time() - start_time
    return (f"SUCCESS: Hybrid Fix.\n"
            f"   - {watertight_count} Already Watertight\n"
            f"   - {hull_count} Hulled (Small/Fallback)\n"
            f"   - {voxel_count} Voxelized (Large)\n"
            f"   - {skipped_dust} Dust pieces ignored\n"
            f"   - Output: {output_path}\n"
            f"   - Time: {elapsed:.2f}s")

if __name__ == "__main__":
    import sys
    input_file = "robust_soup.stl"
    output_file = "fixed_soup.stl"
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
        
    result = repair_tool_v7_hybrid(input_file, output_file)
    print(result)
