import gmsh
import os
import sys
import time
import multiprocessing
from functools import partial

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
STEP_FILE = os.path.join(PROJECT_ROOT, "cad_files", "00-CM240411_BIANCA-C00_HEATER_BOAR_0704A1.STEP")
TEMP_DIR = os.path.join(PROJECT_ROOT, "temp_defeatured")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "simulation_ready_defeatured.vtk")
DELETE_KEYWORDS = ["SCREW", "WASHER", "NUT", "SPACER", "BOLT"]

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def get_b_rep_info():
    """Reads STEP, returns list of (tag, name, bbox)."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.occ.importShapes(STEP_FILE)
        gmsh.model.occ.synchronize()
        vols = gmsh.model.getEntities(dim=3)
        
        info_list = []
        for dim, tag in vols:
            name = gmsh.model.getEntityName(dim, tag)
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
            info_list.append((tag, name, (xmin, ymin, zmin, xmax, ymax, zmax)))
    except Exception:
        info_list = []
        
    gmsh.finalize()
    return info_list

def worker_process_volume(data):
    """Worker function for multiprocessing."""
    tag, name, bbox = data
    vol_msh = os.path.join(TEMP_DIR, f"vol_{tag}.msh")
    
    # 0. Check Exists
    if os.path.exists(vol_msh) and os.path.getsize(vol_msh) > 1000:
        return (tag, "EXIST", vol_msh)

    # 1. SMART DELETE
    # Case insensitive check
    upper_name = name.upper()
    for kw in DELETE_KEYWORDS:
        if kw in upper_name:
            return (tag, "DELETE", kw)

    # 2. ISOLATE & MESH
    try:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 0) # Silence
        gmsh.model.add(f"Proc_{tag}")

        try:
            # Load & Isolate
            gmsh.model.occ.importShapes(STEP_FILE)
            gmsh.model.occ.synchronize()
            
            all_vols = gmsh.model.getEntities(3)
            to_delete = [(d, t) for d, t in all_vols if t != tag]
            if len(to_delete) == len(all_vols):
                 gmsh.finalize()
                 return (tag, "ERROR", "Lost tag")
                 
            gmsh.model.occ.remove(to_delete, recursive=True)
            gmsh.model.occ.synchronize()
            
            # Mesh Settings
            gmsh.option.setNumber("Mesh.Algorithm3D", 1) # Delaunay
            gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5)
            gmsh.option.setNumber("Mesh.MeshSizeMax", 5.0)
            
            # Try Mesh
            gmsh.model.mesh.generate(3)
            
            # Verify nodes
            if gmsh.model.mesh.getNodes()[0].size == 0:
                 raise RuntimeError("Empty mesh")

            status = "KEEP"
            
        except Exception:
            # 3. DEFEATURE (Box)
            status = "REPLACE"
            gmsh.model.remove()
            gmsh.model.add(f"Box_{tag}")
            
            xmin, ymin, zmin, xmax, ymax, zmax = bbox
            dx, dy, dz = xmax-xmin, ymax-ymin, zmax-zmin
            
            # If flat (0 thickness), make small thickness
            if dx < 1e-6: dx = 0.1
            if dy < 1e-6: dy = 0.1
            if dz < 1e-6: dz = 0.1
            
            gmsh.model.occ.addBox(xmin, ymin, zmin, dx, dy, dz)
            gmsh.model.occ.synchronize()
            
            gmsh.option.setNumber("Mesh.Algorithm3D", 1)
            gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5)
            gmsh.option.setNumber("Mesh.MeshSizeMax", 5.0)
            gmsh.model.mesh.generate(3)

        # Write
        final_vols = gmsh.model.getEntities(3)
        if final_vols:
            vt = final_vols[0][1]
            p_grp = gmsh.model.addPhysicalGroup(3, [vt])
            gmsh.model.setPhysicalName(3, p_grp, f"Vol_{tag}")
            
        gmsh.write(vol_msh)
        gmsh.finalize()
        return (tag, status, vol_msh)
        
    except Exception as e:
        # Failsafe
        try: gmsh.finalize() 
        except: pass
        return (tag, "CRITICAL_FAIL", str(e))

def main():
    log("--- STARTING PARALLEL SURGICAL AMPUTATION ---")
    
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
        
    # Phase 1: Master Analysis
    log("Analyzing STEP Structure...")
    info_list = get_b_rep_info()
    log(f"Found {len(info_list)} volumes. Launching parallel workers...")
    
    # Phase 2: Parallel Execution
    # Use 75% of CPU cores or max 8
    num_workers = min(os.cpu_count(), 12)
    log(f"Spawning {num_workers} workers.")
    
    processed_paths = []
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Map returns in order
        results = pool.map(worker_process_volume, info_list)
        
        for tag, status, payload in results:
            if status == "EXIST":
                # log(f"[Skip] Volume {tag} cached.")
                processed_paths.append(payload)
            elif status == "DELETE":
                log(f"[DELETE] Volume {tag} ({payload})")
            elif status == "KEEP":
                log(f"[KEEP] Volume {tag} meshed.")
                processed_paths.append(payload)
            elif status == "REPLACE":
                log(f"[REPLACE] Volume {tag} became BOX.")
                processed_paths.append(payload)
            elif status == "ERROR":
                log(f"[ERROR] Volume {tag}: {payload}")
            elif status == "CRITICAL_FAIL":
                 log(f"[CRITICAL] Volume {tag}: {payload}")

    # Phase 3: Merge
    log(f"Stitching {len(processed_paths)} parts...")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("Frankenstein_Assembly")
    
    for msh in processed_paths:
        try:
            gmsh.merge(msh)
        except:
            pass
            
    log(f"Exporting to {OUTPUT_FILE}...")
    gmsh.write(OUTPUT_FILE)
    gmsh.finalize()
    log("--- MISSION COMPLETE ---")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
