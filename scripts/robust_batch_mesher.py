import gmsh
import os
import time
import multiprocessing
import glob
import sys

# =================CONFIGURATION=================
INPUT_DIR = "temp_stls"
OUTPUT_DIR = "temp_meshes"
NUM_WORKERS = 4  # Safe number for 32GB RAM
TIMEOUT_SECONDS = 30 # If a part takes >30s, kill it and box it.
# ===============================================

def get_stl_bounds(stl_path):
    """
    Robust binary STL bounds reader (Zero-Dependency)
    """
    import struct
    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = float('-inf')
    try:
        with open(stl_path, 'rb') as f:
            header = f.read(80)
            count_bytes = f.read(4)
            if len(count_bytes) < 4: return None
            count = struct.unpack('<I', count_bytes)[0]
            for _ in range(count):
                data = f.read(50)
                if len(data) < 50: break
                floats = struct.unpack('<12fH', data)
                for i in range(3, 12, 3):
                    x, y, z = floats[i], floats[i+1], floats[i+2]
                    min_x, max_x = min(min_x, x), max(max_x, x)
                    min_y, max_y = min(min_y, y), max(max_y, y)
                    min_z, max_z = min(min_z, z), max(max_z, z)
        return (min_x, min_y, min_z, max_x, max_y, max_z)
    except:
        return None

def mesh_worker(stl_path, output_path):
    """
    The Worker Process.
    Strictly uses Delaunay (Algo 1) and disables HXT (Algo 10).
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    
    # --- CRITICAL SETTINGS FOR STABILITY ---
    # 1 = Delaunay (Robust), 10 = HXT (Fragile/Fast)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1) 
    # Disable optimization (often triggers HXT crash on bad geo)
    gmsh.option.setNumber("Mesh.Optimize", 0) 
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)
    # ---------------------------------------

    try:
        gmsh.merge(stl_path)
        gmsh.model.mesh.generate(3)
        gmsh.write(output_path)
    except Exception:
        sys.exit(1) # Fail signal
    finally:
        gmsh.finalize()

def box_worker(stl_path, output_path):
    """Fallback: Creates a Bounding Box mesh"""
    bounds = get_stl_bounds(stl_path)
    if not bounds: return False
    
    min_x, min_y, min_z, max_x, max_y, max_z = bounds
    dx, dy, dz = max_x - min_x, max_y - min_y, max_z - min_z
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("box")
        gmsh.model.occ.addBox(min_x, min_y, min_z, dx, dy, dz)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(3)
        gmsh.write(output_path)
    finally:
        gmsh.finalize()

def process_part(stl_path):
    basename = os.path.basename(stl_path)
    name_no_ext = os.path.splitext(basename)[0]
    output_path = os.path.join(OUTPUT_DIR, name_no_ext + ".msh")
    
    # 0. Skip if done
    if os.path.exists(output_path):
        return "SKIP"

    # 1. Check for Pre-Box Marker (from Splitter)
    marker_path = stl_path + ".marker"
    if os.path.exists(marker_path):
        box_worker(stl_path, output_path)
        return "BOX_PRE"

    # 2. Try Robust Meshing (Subprocess with Timeout)
    p = multiprocessing.Process(target=mesh_worker, args=(stl_path, output_path))
    p.start()
    p.join(TIMEOUT_SECONDS)
    
    if p.is_alive():
        p.terminate() # KILL THE HANG
        p.join()
        # Fallback to Box
        box_worker(stl_path, output_path)
        return "BOX_TIMEOUT"
    
    if p.exitcode != 0:
        # Fallback to Box (Crashed)
        box_worker(stl_path, output_path)
        return "BOX_CRASH"

    return "MESHED"

def main_robust_mesher():
    start_time = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stl_files = glob.glob(os.path.join(INPUT_DIR, "*.stl"))
    
    print(f"=================================================")
    print(f"Starting Robust Batch Mesher")
    print(f"Strategy: Delaunay (Algo 1) | HXT Disabled")
    print(f"Fail-Safe: Any Crash/Hang -> Instant Bounding Box")
    print(f"Queue: {len(stl_files)} parts")
    print(f"=================================================")

    # Use Pool to limit concurrency
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        results = pool.map(process_part, stl_files)
    
    stats = {
        "MESHED": results.count("MESHED"),
        "BOX_PRE": results.count("BOX_PRE"),
        "BOX_CRASH": results.count("BOX_CRASH"),
        "BOX_TIMEOUT": results.count("BOX_TIMEOUT"),
        "SKIP": results.count("SKIP")
    }
    
    elapsed = time.time() - start_time
    print(f"\n=================================================")
    print(f"Job Complete in {elapsed:.2f}s")
    print(f"Meshed: {stats['MESHED']}")
    print(f"Boxed (Pre-marked): {stats['BOX_PRE']}")
    print(f"Boxed (Crashed):    {stats['BOX_CRASH']} (Vol 26 died here)")
    print(f"Boxed (Timeout):    {stats['BOX_TIMEOUT']}")
    print(f"=================================================")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main_robust_mesher()
