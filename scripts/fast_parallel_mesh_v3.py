"""
FAST PARALLEL MESH V3 (Diagnostic Mode + Absolute Pathing)

FIXES:
- Absolute path resolution to prevent "File Not Found" silent failures
- Zombie protection with fresh file streams per worker
- Diagnostic logging to verify volume detection
- Manager-enforced timeout (Windows-safe)
- Outputs .msh files with unique physical groups for proper merging
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

# Use the filename exactly as it appears in your folder
STEP_FILENAME = "00-CM240411_BIANCA-C00_HEATER_BOAR_0704A1.STEP"
OUTPUT_DIR = str(PROJECT_ROOT / "temp_fast_mesh_v3")
FAIL_DIR = str(PROJECT_ROOT / "temp_failures_v3")
FINAL_MERGED = str(PROJECT_ROOT / "generated_meshes" / "heater_board_v3_merged.msh")

TIMEOUT_SEC = 30    # Kill worker if it takes longer than this
MAX_WORKERS = min(12, max(1, multiprocessing.cpu_count() - 2))  # Cap at 12

# --- WORKER FUNCTION ---
def worker_task(vol_tag, step_path_abs):
    """
    Independent worker: Loads CAD, isolates 1 volume, meshes, exports.
    Returns exit code 0 on success, 1 on error.
    """
    try:
        # Redirect stdout/stderr to avoid terminal spam
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        
        out_path = os.path.join(OUTPUT_DIR, f"vol_{vol_tag}.msh")
        
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 0)
        
        # Load File with absolute path
        gmsh.model.occ.importShapes(step_path_abs)
        gmsh.model.occ.synchronize()

        # Isolate
        all_vols = gmsh.model.getEntities(dim=3)
        to_delete = [v for v in all_vols if v[1] != vol_tag]
        
        if len(to_delete) == len(all_vols):
            # Tag not found (shouldn't happen)
            return 1
            
        gmsh.model.occ.remove(to_delete, recursive=True)
        gmsh.model.occ.synchronize()

        # Mesh Settings (Balanced for Speed/Quality)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 5.0)
        gmsh.option.setNumber("Mesh.Algorithm", 6) # Frontal-Delaunay
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        
        # Generate
        gmsh.model.mesh.generate(3)
        
        # Validate: Did we get nodes?
        if gmsh.model.mesh.getNodes()[0].size == 0:
            return 1 # Empty mesh

        # CRITICAL: Assign unique physical group
        vols = gmsh.model.getEntities(3)
        if vols:
            current_tag = vols[0][1]
            pg = gmsh.model.addPhysicalGroup(3, [current_tag])
            gmsh.model.setPhysicalName(3, pg, f"Volume_{vol_tag}")

        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.write(out_path)
        gmsh.finalize()
        return 0

    except:
        return 1


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
    # 1. Path Safety - Try multiple locations
    cad_dir = PROJECT_ROOT / "cad_files"
    possible_paths = [
        cad_dir / STEP_FILENAME,
        PROJECT_ROOT / STEP_FILENAME,
        Path(STEP_FILENAME),
    ]
    
    step_path_abs = None
    for p in possible_paths:
        if p.exists():
            step_path_abs = str(p.resolve())
            break
    
    print(f"============================================================")
    print(f"FAST PARALLEL MESH V3 (Diagnostic Mode)")
    print(f"Target: {step_path_abs}")
    print(f"Workers: {MAX_WORKERS}, Timeout: {TIMEOUT_SEC}s")
    print(f"============================================================")

    if step_path_abs is None or not os.path.exists(step_path_abs):
        print(f"FATAL: File not found. Tried:")
        for p in possible_paths:
            print(f"  - {p}")
        return

    # Setup Output
    if os.path.exists(OUTPUT_DIR): 
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    
    if os.path.exists(FAIL_DIR): 
        shutil.rmtree(FAIL_DIR)
    os.makedirs(FAIL_DIR)

    # 2. Diagnostic Load
    print(f"\nInspecting CAD geometry...")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1) # Show log this time
    gmsh.option.setNumber("General.Verbosity", 2)
    try:
        gmsh.model.occ.importShapes(step_path_abs)
        gmsh.model.occ.synchronize()
        
        solids = gmsh.model.getEntities(dim=3)
        shells = gmsh.model.getEntities(dim=2)
        
        print(f"   -> Found {len(solids)} Solids (Volumes)")
        print(f"   -> Found {len(shells)} Shells (Surfaces)")
        
        all_tags = [tag for dim, tag in solids]
    except Exception as e:
        print(f"FATAL: Gmsh Import Failed. {e}")
        gmsh.finalize()
        return
    gmsh.finalize()

    if len(all_tags) == 0:
        print("WARNING: No Solids found. If you see Shells, the CAD is surface-only.")
        return

    # 3. Execution Loop
    print(f"\nProcessing {len(all_tags)} volumes with {MAX_WORKERS} workers...")
    print("-" * 60)
    
    todo_queue = all_tags[:]
    active_procs = {} 
    
    completed = 0
    errors = 0
    timeouts = 0
    total = len(all_tags)

    while todo_queue or active_procs:
        # A. Fill Pool
        while len(active_procs) < MAX_WORKERS and todo_queue:
            tag = todo_queue.pop(0)
                
            p = multiprocessing.Process(target=worker_task, args=(tag, step_path_abs))
            p.start()
            active_procs[p.pid] = {"p": p, "tag": tag, "start": time.time()}

        # B. Monitor
        pids_to_remove = []
        for pid, info in active_procs.items():
            p = info["p"]
            tag = info["tag"]
            dur = time.time() - info["start"]
            
            if not p.is_alive():
                pids_to_remove.append(pid)
                if p.exitcode == 0:
                    print(f"[{completed+errors+timeouts+1}/{total}] Vol {tag}: [OK] ({dur:.1f}s)")
                    completed += 1
                else:
                    print(f"[{completed+errors+timeouts+1}/{total}] Vol {tag}: [FAIL]")
                    errors += 1
            
            elif dur > TIMEOUT_SEC:
                print(f"[{completed+errors+timeouts+1}/{total}] Vol {tag}: [TIMEOUT] Killing...")
                p.terminate()
                p.join()
                # Log failure
                with open(os.path.join(FAIL_DIR, f"timeout_vol_{tag}.txt"), "w") as f:
                    f.write(f"Timeout after {TIMEOUT_SEC}s")
                pids_to_remove.append(pid)
                timeouts += 1

        for pid in pids_to_remove:
            del active_procs[pid]
            
        time.sleep(0.05)

    print("=" * 60)
    print(f"COMPLETE. Success: {completed}, Errors: {errors}, Timeouts: {timeouts}")
    
    if timeouts > 0:
        print(f"Check {FAIL_DIR} for timeout logs")
    
    # 4. Merge
    if completed > 0:
        merge_all_meshes()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
