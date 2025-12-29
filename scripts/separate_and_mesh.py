import gmsh
import multiprocessing
import os
import time
import sys
import glob
import math

# --- CONFIGURATION ---
# The filename you are looking for
TARGET_FILENAME = "00-CM240411_BIANCA-C00_HEATER_BOAR_0704A1.STEP"
OUTPUT_DIR = "temp_stls/assembly_ready"
FAIL_DIR = "temp_stls/failures"
BATCH_COUNT = 6       # 6 Parallel workers
TIMEOUT_SEC = 240     # 4 minutes max per batch (generous for heavy parts)

def find_step_file():
    """Auto-detects the STEP file in current or common subdirs"""
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
    Loads Master -> Isolates Batch -> Meshes -> Exports Individual STLs
    """
    try:
        sys.stdout = open(os.devnull, 'w') # Silence stdout
        
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        
        # 1. Load Master (~17s)
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
        gmsh.option.setNumber("Mesh.Algorithm", 1) # Delaunay (Fast/Robust)
        
        # 4. Mesh the Batch
        gmsh.model.mesh.generate(3)
        
        # 5. Export Individual STLs
        # We loop through the tags we EXPECT. If they exist, we save them.
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        
        saved_count = 0
        for tag in tags_in_batch:
            # We must use Physical Groups to tell Gmsh "Save ONLY this tag"
            gmsh.model.removePhysicalGroups()
            
            # Check if volume exists (it might have been lost if boolean failed)
            # A quick way is to try adding it.
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
    print(f"[>>] SEPARATE & MESH (Auto-Path Fix)")
    print(f"============================================================")
    
    # 1. Find the File
    step_path = find_step_file()
    if not step_path:
        print(f"[X] FATAL: Could not find '{TARGET_FILENAME}' in current folder or subfolders.")
        print(f"   Please make sure you are in the 'meshpackagelean' directory.")
        return
    
    print(f"[FILE] Found Source: {step_path}")
    
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    if not os.path.exists(FAIL_DIR): os.makedirs(FAIL_DIR)

    # 2. Scan for Tags
    print(f"[1/3] Scanning Assembly Tags...")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.occ.importShapes(step_path)
    vols = gmsh.model.getEntities(dim=3)
    all_tags = [t for d, t in vols]
    gmsh.finalize()
    
    if not all_tags:
        print("[X] Error: File loaded but contained 0 volumes. Is the STEP file corrupted?")
        return

    print(f"      -> Found {len(all_tags)} volumes to process.")
    
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
    
    # Check for existing files to skip? 
    # Optional: For now, we overwrite to ensure we get fresh, clean meshes.
    
    for i, batch_tags in enumerate(batches):
        p = multiprocessing.Process(target=batch_worker, args=(i, batch_tags, step_path))
        p.start()
        active[p.pid] = {"p": p, "id": i, "tags": batch_tags, "start": time.time()}
        
    # 5. Monitor
    failed_batches = 0
    total_stls = 0
    
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
                p.join()
                remove.append(pid)
                failed_batches += 1
                
                # Save suspect list
                with open(os.path.join(FAIL_DIR, f"suspects_batch_{info['id']}.txt"), "w") as f:
                    f.write("\n".join(map(str, info['tags'])))
        
        for pid in remove: del active[pid]
        time.sleep(1)

    # 6. Verify Output
    stl_files = glob.glob(os.path.join(OUTPUT_DIR, "vol_*.stl"))
    print(f"\n[3/3] Job Complete.")
    print(f"      Total STLs Generated: {len(stl_files)} / {len(all_tags)}")
    
    if len(stl_files) == len(all_tags):
        print("      [SUCCESS] SUCCESS: Full assembly separated and meshed.")
    else:
        print(f"      [!] MISSING: {len(all_tags) - len(stl_files)} parts.")
        print(f"      Check {FAIL_DIR} for the suspect lists of the failed batches.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
