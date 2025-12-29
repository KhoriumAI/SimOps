import gmsh
import trimesh
import numpy as np
import os
import sys
import time

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
STEP_FILE = os.path.join(PROJECT_ROOT, "cad_files", "00-CM240411_BIANCA-C00_HEATER_BOAR_0704A1.STEP")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "temp_stls", "plan_d_hybrid")
KNOWN_GOOD_DIR = os.path.join(PROJECT_ROOT, "temp_stls") 

def log(msg):
    print(msg, flush=True)

def get_all_volume_tags():
    """Reads the STEP file once to get the list of volume tags."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.occ.importShapes(STEP_FILE)
        gmsh.model.occ.synchronize()
        vols = gmsh.model.getEntities(dim=3)
        tags = [t for _, t in vols]
    except:
        tags = []
    gmsh.finalize()
    return tags

def process_single_volume(target_tag):
    """
    Loads STEP, deletes everything but target_tag, meshes, exports raw STL.
    Returns: Path to raw STL, or None if failed.
    """
    temp_raw = os.path.join(OUTPUT_DIR, f"temp_raw_{target_tag}.stl")
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1) # See logs for this part
    gmsh.model.add(f"Iso_{target_tag}")
    
    try:
        gmsh.model.occ.importShapes(STEP_FILE)
        gmsh.model.occ.synchronize()
        
        # Verify tag exists (IDs might shift if STEP structure is weird, 
        # but usually Gmsh preserves IDs if file is constant)
        # However, to be safe, we should rely on index? 
        # Tags are usually persistent for same file.
        
        all_vols = gmsh.model.getEntities(3)
        # Find the entity with our tag
        target_exists = False
        to_remove = []
        for dim, tag in all_vols:
            if tag == target_tag:
                target_exists = True
            else:
                to_remove.append((dim, tag))
                
        if not target_exists:
            log(f"   [Error] Volume {target_tag} not found in fresh load.")
            gmsh.finalize()
            return None
            
        # REMOVE others
        if to_remove:
            gmsh.model.occ.remove(to_remove, recursive=True)
            gmsh.model.occ.synchronize()
            
        # MESH 2D
        gmsh.option.setNumber("Mesh.Algorithm", 6) # Frontal Delaunay
        gmsh.model.mesh.generate(2)
        
        gmsh.write(temp_raw)
        gmsh.finalize()
        return temp_raw
        
    except Exception as e:
        log(f"   [Error] Isolating {target_tag} failed: {e}")
        gmsh.finalize()
        return None

def voxel_smash_robust():
    log(f"--- INITIATING PLAN D: ROBUST VOXEL SMASH ---")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Enumerate
    log("Enumerating volumes...")
    vol_tags = get_all_volume_tags()
    log(f"Found {len(vol_tags)} volumes.")
    
    success_count = 0
    smash_count = 0

    for tag in vol_tags:
        vol_name = f"vol_{tag}.stl"
        target_path = os.path.join(OUTPUT_DIR, vol_name)
        
        # --- CHECK ALREADY DONE ---
        # If the file exists in OUTPUT_DIR with reasonable size, skip.
        if os.path.exists(target_path):
            # log(f"[Skip] Volume {tag} already processed.")
            # Assume it's good (either preserved or smashed in previous run)
            # Count it?
            if os.path.getsize(target_path) > 100:
                success_count += 1 # Rough counting
                continue

        # --- PATH A: THE "GOOD" PARTS ---
        good_path = os.path.join(KNOWN_GOOD_DIR, vol_name)
        if os.path.exists(good_path) and os.path.isfile(good_path):
             try:
                mesh = trimesh.load(good_path)
                if mesh.is_watertight:
                    mesh.export(target_path)
                    log(f"[Good] Volume {tag} preserved.")
                    success_count += 1
                    continue
             except:
                pass

        # --- PATH B: THE "BAD" PARTS (ISOLATE & SMASH) ---
        log(f"[Bad] Volume {tag} dirty. ISOLATING & SMASHING...")
        
        # 1. Isolate and Mesh (Robust)
        raw_stl_path = process_single_volume(tag)
        
        if not raw_stl_path or not os.path.exists(raw_stl_path):
            log(f"   [Fail] Could not extract surface for {tag}.")
            continue
            
        # 2. Voxelize
        try:
            raw_mesh = trimesh.load(raw_stl_path)
            bounds = raw_mesh.extents
            if bounds is None or np.all(bounds == 0):
                 log(f"   [Skip] Zero extent.")
                 continue

            min_feature = np.min(bounds)
            pitch = max(min_feature / 5.0, 0.05) 
            
            # Throttling
            MAX_VOXELS = 15_000_000 # Reduced slightly for safety
            vol_approx = bounds[0]*bounds[1]*bounds[2]
            if vol_approx / (pitch**3) > MAX_VOXELS:
                pitch = (vol_approx / MAX_VOXELS)**(1/3)
                log(f"   -> Throttling to {pitch:.4f}mm")

            voxel_grid = raw_mesh.voxelized(pitch=pitch)
            smashed_mesh = voxel_grid.marching_cubes
            
            trimesh.repair.fix_inversion(smashed_mesh)
            trimesh.repair.fix_normals(smashed_mesh)
            
            smashed_mesh.export(target_path)
            log(f"[Smashed] Volume {tag} done.")
            smash_count += 1
            
        except Exception as e:
            log(f"   [Fail] Voxelization error {tag}: {e}")
        
        # Cleanup
        if os.path.exists(raw_stl_path):
            os.remove(raw_stl_path)

    log(f"--- PLAN D COMPLETE ---")

if __name__ == "__main__":
    voxel_smash_robust()
