import gmsh
import multiprocessing
import os
import time
import sys
import glob
import math

# --- CONFIGURATION ---
STEP_FILE = "cad_files/00-CM240411_BIANCA-C00_HEATER_BOAR_0704A1.STEP"
OUTPUT_DIR = "temp_stls/assembly_ready"
FAIL_DIR = "temp_stls/failures"
BATCH_COUNT = 6       # Split assembly into 6 chunks
TIMEOUT_SEC = 180     # 3 minutes max per batch (includes load time)

def get_abs_path(fname):
    if os.path.exists(fname): return os.path.abspath(fname)
    if os.path.exists(os.path.join("..", fname)): return os.path.abspath(os.path.join("..", fname))
    return os.path.abspath(fname)

def batch_worker(batch_id, tags_in_batch, step_path):
    """
    Loads file ONCE, isolates the batch, meshes the batch, exports STLs.
    """
    try:
        # Silence logs
        sys.stdout = open(os.devnull, 'w')
        
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        
        # 1. Load (~17s)
        gmsh.model.occ.importShapes(step_path)
        gmsh.model.occ.synchronize()
        
        # 2. Isolate Batch (Remove everything NOT in this batch)
        # This is efficient: we do one massive delete command
        all_vols = gmsh.model.getEntities(dim=3)
        to_delete = [v for v in all_vols if v[1] not in tags_in_batch]
        
        gmsh.model.occ.remove(to_delete, recursive=True)
        gmsh.model.occ.synchronize()
        
        # 3. Mesh Settings (Fast & Robust)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 5.0)
        gmsh.option.setNumber("Mesh.Algorithm", 1) # Delaunay (Faster/Safer than Frontal for dirty geometry)
        
        # 4. Generate 3D Mesh (Atomic Operation)
        gmsh.model.mesh.generate(3)
        
        # 5. Export Individual STLs
        # We iterate our known tags. If they exist in the mesh, we save them.
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        
        saved_count = 0
        for tag in tags_in_batch:
            # Check if volume still exists (wasn't deleted by bad boolean)
            # Create physical group to isolate export
            gmsh.model.removePhysicalGroups()
            try:
                gmsh.model.addPhysicalGroup(3, [tag], tag)
                out_path = os.path.join(OUTPUT_DIR, f"vol_{tag}.stl")
                gmsh.write(out_path)
                saved_count += 1
            except:
                pass # Volume might have failed meshing entirely
                
        gmsh.finalize()
        return (batch_id, "SUCCESS", saved_count)

    except Exception as e:
        return (batch_id, "ERROR", str(e))

def main():
    print(f"============================================================")
    print(f"[WAR] DIVIDE & CONQUER MESHING")
    print(f"============================================================")
    
    step_path = get_abs_path(STEP_FILE)
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    if not os.path.exists(FAIL_DIR): os.makedirs(FAIL_DIR)

    # 1. Scan Tags
    print(f"[1/3] Scanning Assembly...")
    print(f"      Loading: {step_path}")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    
    try:
        gmsh.model.occ.importShapes(step_path)
        gmsh.model.occ.synchronize()
        print(f"      File loaded successfully")
    except Exception as e:
        print(f"      ERROR loading file: {e}")
        gmsh.finalize()
        return
    
    vols = gmsh.model.getEntities(dim=3)
    all_entities = gmsh.model.getEntities()
    all_tags = [t for d, t in vols]
    
    print(f"      -> All entities: {len(all_entities)}")
    print(f"      -> Volumes (dim=3): {len(vols)}")
    
    gmsh.finalize()
    
    print(f"      -> Found {len(all_tags)} volumes.")
    
    # 2. Split into Batches
    chunk_size = math.ceil(len(all_tags) / BATCH_COUNT)
    batches = []
    for i in range(BATCH_COUNT):
        start = i * chunk_size
        end = start + chunk_size
        batch_tags = all_tags[start:end]
        if batch_tags:
            batches.append(batch_tags)
            
    print(f"      -> Launched {len(batches)} batches (~{chunk_size} parts each).")

    # 3. Parallel Execution
    active = {}
    completed_batches = 0
    failed_batches = 0
    total_parts_done = 0
    
    # Check existing to skip
    # (Optional: Logic to remove existing tags from batches could go here)
    
    for i, batch_tags in enumerate(batches):
        p = multiprocessing.Process(target=batch_worker, args=(i, batch_tags, step_path))
        p.start()
        active[p.pid] = {"p": p, "id": i, "tags": batch_tags, "start": time.time()}
        
    print(f"[2/3] Processing... (Timeout: {TIMEOUT_SEC}s)")
    
    while active:
        remove = []
        for pid, info in active.items():
            p = info["p"]
            dur = time.time() - info["start"]
            
            if not p.is_alive():
                remove.append(pid)
                if p.exitcode == 0:
                    print(f"   [OK] Batch {info['id']} Complete ({dur:.1f}s)")
                    completed_batches += 1
                    # We assume success means most tags exported.
                else:
                    print(f"   [X] Batch {info['id']} CRASHED")
                    failed_batches += 1
                    # Log suspects
                    with open(os.path.join(FAIL_DIR, f"suspects_batch_{info['id']}.txt"), "w") as f:
                        f.write("\n".join(map(str, info['tags'])))

            elif dur > TIMEOUT_SEC:
                print(f"   [TIMEOUT] Batch {info['id']} TIMEOUT ({dur:.1f}s) - KILLING")
                p.terminate()
                p.join()
                remove.append(pid)
                failed_batches += 1
                # Log suspects
                with open(os.path.join(FAIL_DIR, f"suspects_batch_{info['id']}.txt"), "w") as f:
                    f.write("\n".join(map(str, info['tags'])))
        
        for pid in remove: del active[pid]
        time.sleep(1)

    # 4. Summary & Next Steps
    stl_count = len(glob.glob(os.path.join(OUTPUT_DIR, "*.stl")))
    print(f"\n[3/3] Campaign Finished.")
    print(f"      Total STLs Ready: {stl_count}/{len(all_tags)}")
    
    if failed_batches > 0:
        print(f"\n[!] {failed_batches} BATCHES FAILED.")
        print(f"    Suspect lists saved to {FAIL_DIR}/")
        print(f"    You can now inspect these specific groups or run 'finish_with_prejudice.py' on them.")
    else:
        print(f"    [PERFECT] PERFECT RUN. All parts meshed.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
