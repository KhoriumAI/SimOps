import gmsh
import multiprocessing
import time
import os
import sys
import glob

# --- CONFIGURATION ---
# Updated path to match known location
STEP_FILE = "cad_files/00-CM240411_BIANCA-C00_HEATER_BOAR_0704A1.STEP"
BREP_DIR = "temp_geometry"
STL_DIR = "temp_stls/assembly_ready"
FAIL_DIR = "temp_stls/failures"
MAX_WORKERS = 12  # Utilizing your i9 cores

def extract_worker(vol_tag, step_path):
    """Phase 1: Extract single volume to BREP"""
    try:
        # Suppress output
        sys.stdout = open(os.devnull, 'w')
        
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        
        # Load (~17-40s CPU intense)
        gmsh.model.occ.importShapes(step_path)
        gmsh.model.occ.synchronize()
        
        # Isolate
        all_vols = gmsh.model.getEntities(dim=3)
        to_delete = [v for v in all_vols if v[1] != vol_tag]
        if len(to_delete) == len(all_vols): return 1
        
        gmsh.model.occ.remove(to_delete, recursive=True)
        gmsh.model.occ.synchronize()
        
        # Export lightweight CAD
        gmsh.write(os.path.join(BREP_DIR, f"vol_{vol_tag}.brep"))
        gmsh.finalize()
        return 0
    except:
        return 1

def mesh_worker(brep_path):
    """Phase 2: Mesh tiny BREP to STL"""
    try:
        sys.stdout = open(os.devnull, 'w')
        vol_tag = os.path.basename(brep_path).split('_')[1].split('.')[0]
        
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        
        # Instant Load (<0.1s)
        gmsh.open(brep_path)
        
        # Mesh Settings
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 5.0)
        gmsh.option.setNumber("Mesh.Algorithm", 6) # Frontal-Delaunay
        
        gmsh.model.mesh.generate(3)
        
        # Validate
        if len(gmsh.model.mesh.getNodes()[0]) == 0: return 1
        
        gmsh.write(os.path.join(STL_DIR, f"vol_{vol_tag}.stl"))
        gmsh.finalize()
        return 0
    except:
        return 1

def run_phase(name, target_func, task_list, timeout, context=None):
    print(f"\n--- PHASE: {name} ({len(task_list)} tasks, {timeout}s timeout) ---")
    active = {}
    todo = task_list[:]
    completed = 0
    errors = 0
    timeouts = 0
    
    while todo or active:
        # Fill Pool
        while len(active) < MAX_WORKERS and todo:
            item = todo.pop(0)
            args = (item, context) if context else (item,)
            p = multiprocessing.Process(target=target_func, args=args)
            p.start()
            active[p.pid] = {"p": p, "item": item, "start": time.time()}
        
        # Monitor
        remove = []
        for pid, info in active.items():
            p = info["p"]
            dur = time.time() - info["start"]
            
            if not p.is_alive():
                remove.append(pid)
                if p.exitcode == 0:
                    completed += 1
                    # Optional: Print progress every 10 items to reduce spam
                    if completed % 10 == 0: print(f"   Progress: {completed}/{len(task_list)}...")
                else:
                    errors += 1
                    print(f"   Failed: {info['item']}")
            elif dur > timeout:
                p.terminate()
                p.join()
                remove.append(pid)
                timeouts += 1
                print(f"   Timeout: {info['item']}")
                
                # If meshing timed out, save the bad geometry for inspection
                if name == "MESHING" and isinstance(info['item'], str):
                    try:
                        import shutil
                        fail_target = os.path.join(FAIL_DIR, os.path.basename(info['item']))
                        shutil.copy(info['item'], fail_target)
                    except: pass

        for pid in remove: del active[pid]
        time.sleep(0.1)
        
    print(f"   -> Result: {completed} OK, {errors} Errors, {timeouts} Timeouts")

def main():
    # Setup
    # Resolve path relative to this script for robustness
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    
    # Check if STEP_FILE is absolute or relative
    if os.path.isabs(STEP_FILE):
        abs_step = STEP_FILE
    else:
        abs_step = os.path.join(PROJECT_ROOT, STEP_FILE)
        
    if not os.path.exists(abs_step):
        # Fallback to CWD relative (if running from root)
        cwd_path = os.path.abspath(STEP_FILE)
        if os.path.exists(cwd_path):
            abs_step = cwd_path
        else:
             print(f"CRITICAL: Could not find STEP file at {abs_step} or {cwd_path}")
             return

    # Update global dirs to be absolute paths to avoid confusion
    global BREP_DIR, STL_DIR, FAIL_DIR
    BREP_DIR = os.path.join(PROJECT_ROOT, BREP_DIR)
    STL_DIR = os.path.join(PROJECT_ROOT, STL_DIR)
    FAIL_DIR = os.path.join(PROJECT_ROOT, FAIL_DIR)
    
    for d in [BREP_DIR, STL_DIR, FAIL_DIR]:
        if not os.path.exists(d): os.makedirs(d)

    # 1. Get Tags
    print(f"ULTRA FAST PIPELINE | Workers: {MAX_WORKERS}")
    print(f"Source: {abs_step}")
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.occ.importShapes(abs_step)
        gmsh.model.occ.synchronize()
        vols = gmsh.model.getEntities(dim=3)
        tags = [t for d, t in vols]
    except Exception as e:
        print(f"CRITICAL: Could not parse STEP. {e}")
        return
    gmsh.finalize()
    
    print(f"Found {len(tags)} volumes.")

    # 2. PHASE 1: EXPLODE (Parallel Extraction)
    # Check what's already extracted to save time
    existing_breps = []
    if os.path.exists(BREP_DIR):
        existing_files = glob.glob(os.path.join(BREP_DIR, "*.brep"))
        for f in existing_files:
            try:
                tag = int(os.path.basename(f).split('_')[1].split('.')[0])
                existing_breps.append(tag)
            except: pass
            
    to_extract = [t for t in tags if t not in existing_breps]
    
    if to_extract:
        # 90s timeout to survive the CPU contention during load
        run_phase("EXPLOSION", extract_worker, to_extract, 90, abs_step)
    else:
        print("\n--- PHASE: EXPLOSION ---")
        print("   -> All BREPs already cached. Skipping.")

    # 3. PHASE 2: MESH (Parallel Meshing)
    brep_files = glob.glob(os.path.join(BREP_DIR, "*.brep"))
    
    # 15s timeout is plenty for a single pre-loaded part
    run_phase("MESHING", mesh_worker, brep_files, 15)

    print(f"\nPIPELINE COMPLETE.")
    print(f"Valid STLs: {len(glob.glob(os.path.join(STL_DIR, '*.stl')))}")
    print(f"Failures:   {len(glob.glob(os.path.join(FAIL_DIR, '*.brep')))}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
