import gmsh
import os
import time
import math
import multiprocessing
import numpy as np

# =================CONFIGURATION=================
INPUT_STEP = "cad_files/00-CM240411_BIANCA-C00_HEATER_BOAR_0704A1.STEP"
OUTPUT_DIR = "temp_stls"
NUM_WORKERS = 3

# === TRIAGE THRESHOLDS ===
DENSITY_THRESHOLD = 12.0  
MAX_DIAGONAL_FOR_BOXING = 50.0 

# QUALITY GATES (The "Bouncer")
# If Gamma < 0.001, it WILL crash the Octree. Box it.
FATAL_GAMMA_FLOOR = 0.001 
POOR_QUALITY_THRESHOLD = 0.05 
AVG_QUALITY_THRESHOLD = 0.40  
# ===============================================

def get_diagonal(bbox):
    dx = bbox[3] - bbox[0]
    dy = bbox[4] - bbox[1]
    dz = bbox[5] - bbox[2]
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def split_worker(worker_id, total_workers):
    # Initialize with safeguards
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0) 
    gmsh.option.setNumber("Mesh.Binary", 1)
    gmsh.option.setNumber("Mesh.MeshOnlyVisible", 1)
    gmsh.option.setNumber("General.NumThreads", 1) 

    try:
        gmsh.open(INPUT_STEP)
    except Exception as e:
        print(f"[Worker {worker_id}] Load Error: {e}", flush=True)
        return {"meshed": 0, "boxed": 0}

    volumes = gmsh.model.getEntities(3)
    gmsh.model.setVisibility(gmsh.model.getEntities(), 0, recursive=True)

    my_volumes = [v for i, v in enumerate(volumes) if i % total_workers == worker_id]
    
    print(f"[Worker {worker_id}] Ready to process {len(my_volumes)} volumes.", flush=True)
    
    local_stats = {"meshed": 0, "boxed": 0}

    for idx, (dim, vol_tag) in enumerate(my_volumes):
        # Progress heartbeat every 5 parts
        if idx % 5 == 0:
             print(f"[Worker {worker_id}] Processing Vol {vol_tag} ({idx+1}/{len(my_volumes)})...", flush=True)

        filename = f"vol_{vol_tag}.stl"
        filepath = os.path.join(OUTPUT_DIR, filename)

        # --- GATE 1: DENSITY ---
        bbox = gmsh.model.getBoundingBox(dim, vol_tag)
        diag = get_diagonal(bbox)
        surfaces = gmsh.model.getBoundary([(dim, vol_tag)], recursive=False)
        curves = gmsh.model.getBoundary([(dim, vol_tag)], recursive=True)
        density = (len(curves) + len(surfaces)) / (diag + 1e-9)
        
        if (density > DENSITY_THRESHOLD) and (diag < MAX_DIAGONAL_FOR_BOXING):
            with open(filepath + ".marker", "w") as f:
                f.write(f"BOX_ME\n{bbox}")
            local_stats["boxed"] += 1
            continue

        # --- EXECUTE MESH ---
        gmsh.model.setVisibility([(dim, vol_tag)], 1, recursive=True)
        
        # TIMEOUT PROTECTION (Manual check)
        # We can't interrupt C++, but we can log start/end
        t0 = time.time()
        gmsh.model.mesh.generate(2)
        
        # If meshing took > 5 seconds for one part, warn user
        if time.time() - t0 > 5.0:
            print(f"[Worker {worker_id}] WARNING: Vol {vol_tag} took {time.time()-t0:.1f}s to mesh 2D.", flush=True)

        # --- GATE 2: GAMMA CHECK (NO CLEANUP) ---
        _, elem_tags, _ = gmsh.model.mesh.getElements(2, -1) 
        
        quality_ok = True
        if len(elem_tags) > 0:
            # FIX: 'minGamma' is not a valid quality name. Using 'gamma' (lowercase) per core/quality.py
            quals = gmsh.model.mesh.getElementQualities(elem_tags[0], "gamma")
            
            if len(quals) > 0:
                min_q = np.min(quals)
                avg_q = np.mean(quals)
                
                if min_q < FATAL_GAMMA_FLOOR:
                    print(f"[Worker {worker_id}] REJECT Vol {vol_tag}: Fatal Gamma ({min_q:.1e})", flush=True)
                    quality_ok = False
                elif (min_q < POOR_QUALITY_THRESHOLD) and (avg_q < AVG_QUALITY_THRESHOLD):
                    print(f"[Worker {worker_id}] REJECT Vol {vol_tag}: Poor Quality (Min:{min_q:.3f}, Avg:{avg_q:.3f})", flush=True)
                    quality_ok = False
        else:
            quality_ok = False

        # --- EXPORT ---
        if quality_ok:
            surface_tags = [tag for (s_dim, tag) in surfaces]
            if surface_tags:
                gmsh.model.removePhysicalGroups()
                gmsh.model.addPhysicalGroup(2, surface_tags, 1)
                gmsh.write(filepath)
                local_stats["meshed"] += 1
        else:
            with open(filepath + ".marker", "w") as f:
                f.write(f"BOX_ME\n{bbox}")
            local_stats["boxed"] += 1
            
        gmsh.model.mesh.clear()
        gmsh.model.setVisibility([(dim, vol_tag)], 0, recursive=True)

    gmsh.finalize()
    print(f"[Worker {worker_id}] DONE.", flush=True)
    return local_stats

def main_safe():
    start_time = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"=================================================")
    print(f"Starting Safe Parallel Splitter (3 Workers)")
    print(f"Automatic Cleanup: DISABLED (Causes hangs)")
    print(f"Quality Logic: Strict Rejection Only")
    print(f"=================================================")

    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        args = [(i, NUM_WORKERS) for i in range(NUM_WORKERS)]
        results = pool.starmap(split_worker, args)
    
    total_meshed = sum(r['meshed'] for r in results)
    total_boxed = sum(r['boxed'] for r in results)
    
    elapsed = time.time() - start_time
    print(f"\n=================================================")
    print(f"Job Complete in {elapsed:.2f}s")
    print(f"Total Meshed: {total_meshed}")
    print(f"Total Boxed:  {total_boxed}")
    print(f"=================================================")

if __name__ == "__main__":
    multiprocessing.freeze_support() 
    main_safe()
