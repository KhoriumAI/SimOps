import gmsh
import multiprocessing
import os
import time
import sys
import glob

# --- CONFIGURATION ---
STEP_FILE = "cad_files/00-CM240411_BIANCA-C00_HEATER_BOAR_0704A1.STEP"
OUTPUT_DIR = "temp_stls/assembly_ready"
FAIL_DIR = "temp_stls/failures"
# Threshold: Parts with more faces than this are treated as "Heavy" (Surgical)
# Parts with fewer are "Light" (Bulk)
FACE_THRESHOLD = 200 
MAX_WORKERS = 4  # Reduced from 12 to 4 to prevent I/O choking

def get_abs_path(fname):
    if os.path.exists(fname): return os.path.abspath(fname)
    if os.path.exists(os.path.join("..", fname)): return os.path.abspath(os.path.join("..", fname))
    return os.path.abspath(fname)

def surgical_worker(tag, step_path):
    """Worker for HEAVY parts: Isolated load & mesh"""
    try:
        sys.stdout = open(os.devnull, 'w') # Silence
        out_path = os.path.join(OUTPUT_DIR, f"vol_{tag}.stl")
        
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        
        # Load (~17s)
        gmsh.model.occ.importShapes(step_path)
        gmsh.model.occ.synchronize()
        
        # Isolate
        all_vols = gmsh.model.getEntities(dim=3)
        to_del = [v for v in all_vols if v[1] != tag]
        gmsh.model.occ.remove(to_del, recursive=True)
        gmsh.model.occ.synchronize()
        
        # Mesh
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 5.0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.mesh.generate(3)
        
        gmsh.write(out_path)
        gmsh.finalize()
        return 0
    except:
        # Dump failure for inspection
        try:
            fail_path = os.path.join(FAIL_DIR, f"fail_vol_{tag}.brep")
            gmsh.write(fail_path)
        except: pass
        return 1

def main():
    print(f"============================================================")
    print(f"[+] SMART HYBRID MESH GENERATOR")
    print(f"============================================================")
    
    step_path = get_abs_path(STEP_FILE)
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    if not os.path.exists(FAIL_DIR): os.makedirs(FAIL_DIR)

    # 1. TRIAGE (Single Load)
    print(f"[1/4] Loading Master for Triage...")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.occ.importShapes(step_path)
    gmsh.model.occ.synchronize()
    
    all_vols = gmsh.model.getEntities(dim=3)
    light_tags = []
    heavy_tags = []
    
    print(f"      Analyzing {len(all_vols)} volumes...")
    for dim, tag in all_vols:
        # Count faces to determine complexity
        faces = gmsh.model.getAdjacencies(dim, tag)[1]
        if len(faces) > FACE_THRESHOLD:
            heavy_tags.append(tag)
        else:
            light_tags.append(tag)
            
    print(f"      -> {len(light_tags)} Light Parts (Bulk Mesh)")
    print(f"      -> {len(heavy_tags)} Heavy Parts (Surgical Mesh)")
    
    # 2. BULK PROCESSING (Light Parts)
    print(f"[2/4] Processing Light Parts (In-Memory)...")
    
    # Remove heavy parts from this session to speed up meshing
    if heavy_tags:
        heavy_pairs = [(3, t) for t in heavy_tags]
        gmsh.model.occ.remove(heavy_pairs, recursive=True)
        gmsh.model.occ.synchronize()
    
    # Mesh everything remaining
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 5.0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(3)
    
    # Export one by one using Physical Groups to isolate
    gmsh.option.setNumber("Mesh.SaveAll", 0) # Only save defined physical groups
    
    count = 0
    for tag in light_tags:
        # Define physical group for JUST this volume
        gmsh.model.removePhysicalGroups()
        gmsh.model.addPhysicalGroup(3, [tag], tag)
        
        out_name = os.path.join(OUTPUT_DIR, f"vol_{tag}.stl")
        gmsh.write(out_name)
        count += 1
        if count % 20 == 0: print(f"      Exported {count}/{len(light_tags)}...")
        
    gmsh.finalize()
    print(f"      [OK] Bulk Phase Complete.")

    # 3. SURGICAL PROCESSING (Heavy Parts)
    if heavy_tags:
        print(f"[3/4] Processing {len(heavy_tags)} Heavy Parts (Parallel Workers)...")
        # Check if already done
        todo = [t for t in heavy_tags if not os.path.exists(os.path.join(OUTPUT_DIR, f"vol_{t}.stl"))]
        
        if todo:
            pool = multiprocessing.Pool(MAX_WORKERS)
            # Use starmap to pass args
            tasks = [(t, step_path) for t in todo]
            
            results = pool.starmap(surgical_worker, tasks)
            pool.close()
            pool.join()
            
            failures = results.count(1)
            print(f"      [OK] Surgical Phase Complete. {failures} Failures.")
        else:
            print("      All heavy parts already exist. Skipping.")
    
    # 4. FINAL REPORT
    total_stls = len(glob.glob(os.path.join(OUTPUT_DIR, "*.stl")))
    print(f"============================================================")
    print(f"JOB DONE. Total STLs: {total_stls}/{len(all_vols)}")
    if total_stls < len(all_vols):
        print(f"Check {FAIL_DIR} for broken geometry.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
