import gmsh
import multiprocessing
import os
import glob
import time
import sys

# --- CONFIGURATION ---
STL_INPUT_DIR = "temp_stls/assembly_ready"
OUTPUT_DIR = "temp_stls/volume_meshes"
FAIL_DIR = "temp_stls/failures"
MAX_WORKERS = 6
MIN_FILE_SIZE = 1024  # 1KB - filter out corrupt/empty files
TIMEOUT_SEC = 120  # 2 minutes per part (generous for complex geometry)

def volume_mesh_worker(stl_path):
    """
    Generates a volume mesh from an STL surface mesh.
    Returns: (filename, status, message)
    """
    try:
        # Silence output
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        
        filename = os.path.basename(stl_path)
        vol_tag = filename.replace("vol_", "").replace(".stl", "")
        
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        
        # 1. Load STL surface
        gmsh.merge(stl_path)
        
        # 2. Check if geometry is watertight (critical!)
        try:
            gmsh.model.mesh.classifySurfaces(angle=40 * 3.14159 / 180, boundary=True)
            gmsh.model.mesh.createGeometry()
        except Exception as e:
            gmsh.finalize()
            return (filename, "FAILED", f"Leaky geometry (not watertight): {e}")
        
        # 3. Create a volume from the closed surface
        # Get all surfaces
        surfaces = gmsh.model.getEntities(2)
        if not surfaces:
            gmsh.finalize()
            return (filename, "FAILED", "No surfaces found after classification")
        
        try:
            # Create a surface loop
            surface_tags = [s[1] for s in surfaces]
            sl = gmsh.model.geo.addSurfaceLoop(surface_tags)
            # Create volume from surface loop
            gmsh.model.geo.addVolume([sl])
            gmsh.model.geo.synchronize()
        except Exception as e:
            gmsh.finalize()
            return (filename, "FAILED", f"Failed to create volume: {e}")
        
        # 4. Set mesh parameters
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 5.0)
        gmsh.option.setNumber("Mesh.Algorithm", 1)  # Delaunay
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay 3D
        
        # 5. Generate volume mesh
        gmsh.model.mesh.generate(3)
        
        # 6. Export
        output_path = os.path.join(OUTPUT_DIR, f"vol_{vol_tag}.msh")
        gmsh.write(output_path)
        
        gmsh.finalize()
        return (filename, "SUCCESS", f"Meshed to {output_path}")
        
    except Exception as e:
        try:
            gmsh.finalize()
        except:
            pass
        return (filename, "FAILED", str(e))

def main():
    print("============================================================")
    print("[RESUME] Volume Mesh Generator (From STL Checkpoint)")
    print("============================================================")
    
    # 1. Create output directories
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if not os.path.exists(FAIL_DIR):
        os.makedirs(FAIL_DIR)
    
    # 2. Scan for valid STL files (recursive search)
    print(f"[1/4] Scanning for STL files (recursive)...")
    all_stls = glob.glob(os.path.join("temp_stls", "**", "*.stl"), recursive=True)
    
    print(f"      -> Found {len(all_stls)} total STL files")
    
    valid_stls = []
    corrupt_count = 0
    
    for stl_path in all_stls:
        file_size = os.path.getsize(stl_path)
        if file_size > MIN_FILE_SIZE:
            valid_stls.append(stl_path)
        else:
            corrupt_count += 1
            print(f"      [SKIP] Corrupt/empty: {os.path.basename(stl_path)} ({file_size} bytes)")
    
    print(f"      -> {len(valid_stls)} valid STLs (>{MIN_FILE_SIZE} bytes)")
    if corrupt_count > 0:
        print(f"      -> Skipped {corrupt_count} corrupt files")
    
    if not valid_stls:
        print("[X] No valid STL files found. Exiting.")
        return
    
    # 3. Check what's already been done
    existing_meshes = glob.glob(os.path.join(OUTPUT_DIR, "*.msh"))
    existing_tags = set()
    for mesh_path in existing_meshes:
        tag = os.path.basename(mesh_path).replace("vol_", "").replace(".msh", "")
        existing_tags.add(tag)
    
    todo_stls = []
    for stl_path in valid_stls:
        tag = os.path.basename(stl_path).replace("vol_", "").replace(".stl", "")
        if tag not in existing_tags:
            todo_stls.append(stl_path)
    
    if existing_tags:
        print(f"      -> {len(existing_tags)} already meshed, {len(todo_stls)} remaining")
    
    if not todo_stls:
        print("[OK] All STLs already have volume meshes!")
        return
    
    # 4. Process in parallel
    print(f"[2/4] Generating volume meshes ({MAX_WORKERS} workers)...")
    
    start_time = time.time()
    success_count = 0
    failed_count = 0
    
    # Use multiprocessing pool
    with multiprocessing.Pool(MAX_WORKERS) as pool:
        results = pool.map(volume_mesh_worker, todo_stls)
    
    # 5. Report results
    print(f"\n[3/4] Processing results...")
    for filename, status, message in results:
        if status == "SUCCESS":
            success_count += 1
            print(f"   [OK] {filename}")
        else:
            failed_count += 1
            print(f"   [X] {filename}: {message}")
            # Log failure
            with open(os.path.join(FAIL_DIR, f"failed_{filename}.txt"), "w") as f:
                f.write(message)
    
    elapsed = time.time() - start_time
    
    print(f"\n[4/4] Job Complete ({elapsed:.1f}s)")
    print(f"      Success: {success_count}")
    print(f"      Failed: {failed_count}")
    print(f"      Output: {OUTPUT_DIR}")
    
    if failed_count > 0:
        print(f"      Check {FAIL_DIR} for error details")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
