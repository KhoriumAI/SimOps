import gmsh
import multiprocessing
import os
import time
import sys
import glob
import math

# --- CONFIGURATION ---
TARGET_FILENAME = "00-CM240411_BIANCA-C00_HEATER_BOAR_0704A1.STEP"
OUTPUT_DIR = "temp_stls/assembly_ready"
FAIL_DIR = "temp_stls/failures"
BATCH_COUNT = 6       # 6 Parallel workers
TIMEOUT_SEC = 240     # 4 minutes max per batch

def find_step_file():
    """Auto-detects the STEP file"""
    search_paths = [
        TARGET_FILENAME,
        os.path.join("cad_files", TARGET_FILENAME),
        os.path.join("..", TARGET_FILENAME),
        os.path.join("..", "cad_files", TARGET_FILENAME)
    ]
    for p in search_paths:
        if os.path.exists(p):
            return os.path.abspath(p)
    return None

def batch_worker(batch_id, tags_in_batch, step_path):
    """
    Worker: Loads Master -> Isolates Batch -> Meshes -> Exports
    """
    try:
        sys.stdout = open(os.devnull, 'w')
        
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        
        # 1. Load Master (~17s)
        # Using importShapes for manipulation capability
        gmsh.model.occ.importShapes(step_path)
        gmsh.model.occ.synchronize()
        
        # 2. Isolate: Delete everything NOT in this batch
        all_vols = gmsh.model.getEntities(dim=3)
        to_delete = [v for v in all_vols if v[1] not in tags_in_batch]
        
        gmsh.model.occ.remove(to_delete, recursive=True)
        gmsh.model.occ.synchronize()
        
        # 3. Robust Mesh Settings
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 5.0)
        gmsh.option.setNumber("Mesh.Algorithm", 1) # Delaunay
        
        # 4. Mesh
        gmsh.model.mesh.generate(3)
        
        # 5. Export
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        saved_count = 0
        for tag in tags_in_batch:
            gmsh.model.removePhysicalGroups()
            try:
                gmsh.model.addPhysicalGroup(3, [tag], tag)
                out_path = os.path.join(OUTPUT_DIR, f"vol_{tag}.stl")
                gmsh.write(out_path)
                saved_count += 1
            except:
                pass 
                
        gmsh.finalize()
        return (batch_id, "SUCCESS", saved_count)

    except Exception as e:
        return (batch_id, "ERROR", str(e))

def main():
    print(f"============================================================")
    print(f"[>>] SEPARATE & MESH V2 (Robust Scan)")
    print(f"============================================================")
    
    # 1. Find File
    step_path = find_step_file()
    if not step_path:
        print(f"[X] FATAL: Could not find '{TARGET_FILENAME}'.")
        return
    
    print(f"[FILE] Found Source: {step_path}")
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    if not os.path.exists(FAIL_DIR): os.makedirs(FAIL_DIR)

    # 2. Robust Scan (Using gmsh.open instead of importShapes)
    print(f"[1/3] Scanning Assembly Tags...")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        # gmsh.open is safer for just reading tags
        gmsh.open(step_path) 
        vols = gmsh.model.getEntities(dim=3)
        all_tags = [t for d, t in vols]
    except Exception as e:
        print(f"[X] Error reading file: {e}")
        return
    gmsh.finalize()
    
    if not all_tags:
        print("[X] Error: File loaded but contained 0 volumes. (Try killing python processes)")
        return

    print(f"      -> Found {len(all_tags)} volumes.")
    
    # 3. Create Batches
    chunk_size = math.ceil(len(all_tags) / BATCH_COUNT)
    batches = []
    for i in range(BATCH_COUNT):
        start = i * chunk_size
        end = start + chunk_size
        batch_tags = all_tags[start:end]
        if batch_tags:
            batches.append(batch_tags)
            
    print(f"      -> Split into {len(batches)} batches of ~{chunk_size} parts.")

    # 4. Launch Workers
    print(f"[2/3] Processing Batches (Parallel)...")
    active = {}
    
    for i, batch_tags in enumerate(batches):
        p = multiprocessing.Process(target=batch_worker, args=(i, batch_tags, step_path))
        p.start()
        active[p.pid] = {"p": p, "id": i, "tags": batch_tags, "start": time.time()}
        
    # 5. Monitor
    failed_batches = 0
    while active:
        remove = []
        for pid, info in active.items():
            p = info["p"]
            dur = time.time() - info["start"]
            
            if not p.is_alive():
                remove.append(pid)
                if p.exitcode == 0:
                    print(f"   [OK] Batch {info['id']} Finished ({dur:.1f}s)")
                else:
                    print(f"   [X] Batch {info['id']} CRASHED")
                    failed_batches += 1
            
            elif dur > TIMEOUT_SEC:
                print(f"   [TIMEOUT] Batch {info['id']} TIMEOUT ({dur:.1f}s) - Killing")
                p.terminate()
                remove.append(pid)
                failed_batches += 1
                
                with open(os.path.join(FAIL_DIR, f"suspects_batch_{info['id']}.txt"), "w") as f:
                    f.write("\n".join(map(str, info['tags'])))
        
        for pid in remove: del active[pid]
        time.sleep(1)

    print(f"\n[3/3] Job Complete. Check {OUTPUT_DIR} for results.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
