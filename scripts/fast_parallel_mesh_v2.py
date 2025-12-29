"""
FAST PARALLEL MESH V2 (Manager-Enforced Timeout)

STRATEGY:
- Manager process spawns workers using multiprocessing.Process
- Manager monitors wall-clock time of each active worker
- If a worker exceeds timeout, Manager terminates() it (Hard Kill)
- Output: .msh files with unique physical groups (Volume_{tag})
- Final Step: Merges all successful meshes

Prerequisites: None (Standard Lib only)
"""

import gmsh
import multiprocessing
import time
import os
import sys
import shutil
from pathlib import Path

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
STEP_FILE = str(PROJECT_ROOT / "cad_files" / "00-CM240411_BIANCA-C00_HEATER_BOAR_0704A1.STEP")
OUTPUT_DIR = str(PROJECT_ROOT / "temp_fast_mesh")
FAIL_DIR = str(PROJECT_ROOT / "temp_failures")
FINAL_MERGED = str(PROJECT_ROOT / "generated_meshes" / "heater_board_v2_merged.msh")

TIMEOUT_SEC = 30    # Kill worker if it takes longer than this
MAX_WORKERS = max(1, multiprocessing.cpu_count() - 2)

# --- WORKER FUNCTION (Runs in separate process) ---
def worker_task(vol_tag):
    """
    Loads CAD, isolates volume, meshes, and saves.
    Returns exit code 0 on success, 1 on error.
    """
    try:
        # 1. Initialize logic
        out_path = os.path.join(OUTPUT_DIR, f"vol_{vol_tag}.msh")
        
        # 2. Redirect stdout/stderr to suppress Gmsh spam
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 0)

        # 3. Fast Load
        gmsh.model.occ.importShapes(STEP_FILE)
        gmsh.model.occ.synchronize()

        # 4. Isolate Volume (Delete everything else)
        all_vols = gmsh.model.getEntities(dim=3)
        to_delete = [v for v in all_vols if v[1] != vol_tag]
        
        if len(to_delete) == len(all_vols):
            # The requested tag doesn't exist
            return 1
            
        gmsh.model.occ.remove(to_delete, recursive=True)
        gmsh.model.occ.synchronize()

        # 5. Mesh
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 5.0)
        gmsh.option.setNumber("Mesh.Algorithm", 6) # Frontal-Delaunay
        gmsh.option.setNumber("Mesh.Algorithm3D", 1) 
        
        gmsh.model.mesh.generate(3)
        
        # Check if nodes exist
        if gmsh.model.mesh.getNodes()[0].size == 0:
            return 1

        # 6. CRITICAL: Assign Physical Group
        # We must re-discover the volume tag because operations might change internal numbering
        vols = gmsh.model.getEntities(3)
        if vols:
            current_tag = vols[0][1]
            pg = gmsh.model.addPhysicalGroup(3, [current_tag])
            gmsh.model.setPhysicalName(3, pg, f"Volume_{vol_tag}")

        # 7. Export
        gmsh.option.setNumber("Mesh.SaveAll", 0) # Only save physical groups
        gmsh.write(out_path)
        gmsh.finalize()
        return 0

    except Exception:
        # In case of crash, we just return 1
        return 1

# --- MERGE FUNCTION ---
def merge_all_meshes():
    """Merge all successful meshes into one file"""
    print("\n" + "=" * 50)
    print("MERGING ALL VOLUMES...")
    print("=" * 50)
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("Merged_Assembly")
    
    msh_files = sorted(Path(OUTPUT_DIR).glob("vol_*.msh"))
    print(f"Found {len(msh_files)} mesh files to merge")
    
    for msh_file in msh_files:
        try:
            gmsh.merge(str(msh_file))
        except Exception as e:
            print(f"  [!] Failed to merge {msh_file.name}: {e}")
    
    # Verify physical groups
    phys_groups = gmsh.model.getPhysicalGroups(3)
    print(f"[OK] Merged mesh contains {len(phys_groups)} physical volume groups")
    
    os.makedirs(os.path.dirname(FINAL_MERGED), exist_ok=True)
    gmsh.option.setNumber("Mesh.SaveAll", 1)
    gmsh.write(FINAL_MERGED)
    gmsh.finalize()
    print(f"[OK] Final mesh saved: {FINAL_MERGED}")

# --- MAIN MANAGER ---
def main():
    print(f"============================================================")
    print(f"FAST PARALLEL MESH V2 (Manager-Enforced Timeout)")
    print(f"timeout={TIMEOUT_SEC}s, workers={MAX_WORKERS}")
    print(f"============================================================")

    # 1. Setup Dirs
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    
    if os.path.exists(FAIL_DIR): shutil.rmtree(FAIL_DIR)
    os.makedirs(FAIL_DIR)

    # 2. Quick Scan for Volume Tags
    print(f"Scanning: {os.path.basename(STEP_FILE)}...")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.occ.importShapes(STEP_FILE)
        all_vols = gmsh.model.getEntities(dim=3)
        all_tags = [tag for dim, tag in all_vols]
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load STEP file. {e}")
        return
    gmsh.finalize()
    
    print(f"Found {len(all_tags)} volumes. Processing...")

    # 3. Queue Management
    todo_queue = all_tags[:]
    active_procs = {} # Map: pid -> {process, tag, start_time}
    
    completed = 0
    errors = 0
    timeouts = 0
    skipped = 0
    
    total_count = len(all_tags)
    
    while todo_queue or active_procs:
        # A. Fill Slots
        while len(active_procs) < MAX_WORKERS and todo_queue:
            tag = todo_queue.pop(0)
            
            # Spawn
            p = multiprocessing.Process(target=worker_task, args=(tag,))
            p.start()
            active_procs[p.pid] = {
                "p": p,
                "tag": tag,
                "start": time.time()
            }
        
        # B. Monitor Active
        pids_to_remove = []
        
        for pid, info in active_procs.items():
            p = info["p"]
            tag = info["tag"]
            start_t = info["start"]
            duration = time.time() - start_t
            
            # Check 1: Finished?
            if not p.is_alive():
                pids_to_remove.append(pid)
                if p.exitcode == 0:
                    print(f"[{completed+skipped+errors+timeouts+1}/{total_count}] Vol {tag}: [OK] Success ({duration:.1f}s)")
                    completed += 1
                else:
                    print(f"[{completed+skipped+errors+timeouts+1}/{total_count}] Vol {tag}: [FAIL] Error/Crash ({duration:.1f}s)")
                    errors += 1
                continue
            
            # Check 2: Timeout?
            if duration > TIMEOUT_SEC:
                print(f"[{completed+skipped+errors+timeouts+1}/{total_count}] Vol {tag}: [TIMEOUT] Killing... ({duration:.1f}s)")
                p.terminate() # The Hard Kill
                p.join()      # Cleanup
                
                # Dump info
                with open(os.path.join(FAIL_DIR, f"timeout_vol_{tag}.txt"), "w") as f:
                    f.write(f"Volume {tag} timed out after {TIMEOUT_SEC} seconds.")
                
                pids_to_remove.append(pid)
                timeouts += 1
                
        # C. Cleanup
        for pid in pids_to_remove:
            del active_procs[pid]
            
        # Avoid CPU spin
        time.sleep(0.05)

    print("-" * 60)
    print(f"DONE. Success: {completed}, Timeouts: {timeouts}, Errors: {errors}")
    print(f"If timeouts > 0, check {FAIL_DIR}")
    
    # 4. Merge
    if completed > 0:
        merge_all_meshes()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
