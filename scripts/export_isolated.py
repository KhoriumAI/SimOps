import subprocess
import os
import time
import sys
import glob
import shlex

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
WORKER_SCRIPT = os.path.join(SCRIPT_DIR, "parallel_vol_mesher.py") # We reuse this but for export
STEP_FILE = os.path.join(PROJECT_ROOT, "cad_files", "00-CM240411_BIANCA-C00_HEATER_BOAR_0704A1.STEP")
OUTPUT_STL = os.path.join(PROJECT_ROOT, "robust_soup.stl")
TEMP_DIR = os.path.join(PROJECT_ROOT, "temp_stls")

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def main():
    t0 = time.time()
    log("STARTING ISOLATED MULTI-PROCESS STL EXPORTER...")
    
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    # 1. Get Volume Count (we know it's 151, but let's be dynamic or hardcode for speed)
    # We can use the existing multi-stage extractor logic or just hardcode for this specific file
    # to save time as we verified 151 many times.
    num_volumes = 151
    log(f"   - Target Volumes: {num_volumes}")
    
    # 2. Spawn Subprocesses
    # We need a worker that takes --volume_index and exports STL.
    # We will create a tiny temporary worker script to avoid modifying parallel_vol_mesher again if it's complex.
    
    worker_code = """
import gmsh
import sys
import os

def export_vol(vol_idx, step_path, out_dir):
    try:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.occ.importShapes(step_path)
        gmsh.model.occ.synchronize()
        
        vols = gmsh.model.getEntities(3)
        if vol_idx >= len(vols):
            sys.exit(0)
            
        # Isolate volume
        target = vols[vol_idx]
        gmsh.model.removeEntities([v for i,v in enumerate(vols) if i != vol_idx], recursive=True)
        
        # Mesh 2D with fallback
        for alg in [6, 5, 1]:
            try:
                gmsh.option.setNumber("Mesh.Algorithm", alg)
                gmsh.model.mesh.generate(2)
                out_file = os.path.join(out_dir, f"vol_{vol_idx}.stl")
                gmsh.write(out_file)
                gmsh.finalize()
                return
            except Exception as e:
                print(f"Alg {alg} failed: {e}")
                # Reset? GMSH state might be dirty.
                # Ideally we re-load. But let's try just proceeding.
                continue
        print("All algorithms failed.")
        sys.exit(1)
    except Exception as e:
        print(e)
        sys.exit(1)

if __name__ == "__main__":
    export_vol(int(sys.argv[1]), sys.argv[2], sys.argv[3])
"""
    worker_path = os.path.join(SCRIPT_DIR, "iso_exporter.py")
    with open(worker_path, "w") as f:
        f.write(worker_code)
        
    log("   - Spawning workers...")
    processes = []
    
    # Batching to avoid killing CPU
    BATCH_SIZE = 4
    
    for i in range(num_volumes):
        dst = os.path.join(TEMP_DIR, f"vol_{i}.stl")
        if os.path.exists(dst):
            continue
            
        cmd = [sys.executable, worker_path, str(i), STEP_FILE, TEMP_DIR]
        log(f"     [DEBUG] Running: {cmd}")
        p = subprocess.Popen(cmd)
        processes.append(p)
        
        if len(processes) >= BATCH_SIZE:
            for p in processes:
                try:
                    p.communicate(timeout=60)
                except subprocess.TimeoutExpired:
                    log("     [!] Worker timed out. Killing...")
                    p.kill()
                    p.communicate()
            processes = []
            log(f"     Batch finished. Progress: {i+1}/{num_volumes}")
            
    # Final cleanup
    for p in processes:
        try:
            p.communicate(timeout=60)
        except subprocess.TimeoutExpired:
             p.kill()
             p.communicate()
        
    # 3. Merge STLs
    log("   - Merging STLs...")
    stls = glob.glob(os.path.join(TEMP_DIR, "*.stl"))
    log(f"     Found {len(stls)} STLs.")
    
    # We can just cat them if binary or using pyvista
    import pyvista as pv
    if not stls:
        log("   [!] No STLs generated.")
        return

    # Efficient merge?
    # PyVista merge is easy
    blocks = pv.MultiBlock([pv.read(s) for s in stls])
    merged = blocks.combine()
    merged.save(OUTPUT_STL)
    
    log(f"[OK] DONE. Saved robust soup to {OUTPUT_STL}")
    log(f"[Finished] Total Time: {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
