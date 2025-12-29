import subprocess
import os
import sys
import time
import gmsh
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
STEP_FILE = os.path.join(PROJECT_ROOT, "cad_files", "00-CM240411_BIANCA-C00_HEATER_BOAR_0704A1.STEP")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "simulation_ready_defeatured.vtk")
WORKER_SCRIPT = os.path.join(SCRIPT_DIR, "surgical_worker.py")
TEMP_DIR = os.path.join(PROJECT_ROOT, "temp_defeatured")

TIMEOUT_MESH = 300 # 5 Minutes for Good Volume Mesh
TIMEOUT_BOX = 60   # 1 Minute for Box Fallback (Load Time included)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def get_tags():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.occ.importShapes(STEP_FILE)
        vols = gmsh.model.getEntities(3)
        tags = [t for _, t in vols]
    except:
        tags = []
    gmsh.finalize()
    return tags

def run_task(tag):
    vol_msh = os.path.join(TEMP_DIR, f"vol_{tag}.msh")
    if os.path.exists(vol_msh) and os.path.getsize(vol_msh) > 1000:
        return f"SKIP_{tag}"

    # Try 1: Faithful Mesh (Includes Canary Check)
    try:
        cmd = [sys.executable, WORKER_SCRIPT, "--tag", str(tag)]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_MESH)
        
        if res.returncode == 0 and "SUCCESS" in res.stdout:
            return f"MESH_{tag}" # Success
        elif "DELETE" in res.stdout:
            return f"DEL_{tag}"
        elif "BOXED" in res.stdout:
            return f"BOX_{tag}" # Worker decided to box it (1D check)
        elif res.returncode == 2:
             # Canary exit (2D timeout signal)
             log(f"Volume {tag} killed by Canary (Timeout). Boxing.")
        else:
            # Failed with code 1 or text
             pass # Fallthrough to box
            
    except subprocess.TimeoutExpired:
        log(f"TIMEOUT Volume {tag} (> {TIMEOUT_MESH}s). Fallback to BOX.")
    except Exception:
        pass 
        
    # Try 2: BOX Fallback
    try:
        cmd = [sys.executable, WORKER_SCRIPT, "--tag", str(tag), "--box"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_BOX)
        if res.returncode == 0:
            return f"BOX_{tag}"
    except:
        return f"CRITICAL_{tag}"
        
    return f"FAIL_{tag}"

def main():
    log("--- STARTING INVINCIBLE ORCHESTRATOR V3 (CANARY) ---")
    
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
        
    log("Scanning STEP...")
    tags = get_tags()
    log(f"Found {len(tags)} volumes.")
    
    MAX_WORKERS = 16 
    results = {}
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_tag = {executor.submit(run_task, tag): tag for tag in tags}
        
        for future in as_completed(future_to_tag):
            tag = future_to_tag[future]
            try:
                status = future.result()
                results[tag] = status
                if "DEL" in status:
                    log(f"Deleted {tag}")
                elif "BOX" in status:
                    log(f"Boxed {tag}")
                elif "MESH" in status:
                    log(f"Meshed {tag}")
                elif "SKIP" in status:
                    # log(f"Cached {tag}")
                    pass
                else:
                    log(f"FAILED {tag}")
            except Exception as e:
                log(f"Exception {tag}: {e}")

    log("Merging results...")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("Final_Defeatured")
    
    import glob
    msh_files = glob.glob(os.path.join(TEMP_DIR, "vol_*.msh"))
    log(f"Merging {len(msh_files)} components.")
    
    for m in msh_files:
        try: gmsh.merge(m)
        except: pass
        
    gmsh.write(OUTPUT_FILE)
    gmsh.finalize()
    log("--- DONE ---")

if __name__ == "__main__":
    main()
