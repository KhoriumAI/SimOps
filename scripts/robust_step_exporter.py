import gmsh
import os
import time
import sys
import subprocess
import numpy as np

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
STEP_FILE = os.path.join(PROJECT_ROOT, "cad_files", "00-CM240411_BIANCA-C00_HEATER_BOAR_0704A1.STEP")
TEMP_DIR = os.path.join(PROJECT_ROOT, "temp_stls")
OUTPUT_SOUP = os.path.join(PROJECT_ROOT, "robust_soup.stl")

# Optimization
# We use Algorithm 5 (Delaunay) which is robust for complex CAD surfaces.
# We set a fine enough mesh size to capture pins but not so dense as to crawl.
MESH_SIZE = 1.0  # mm

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def extract_volume_isolated(vol_idx):
    """Worker function to mesh a single volume in a fresh process."""
    output_file = os.path.join(TEMP_DIR, f"vol_{vol_idx}.stl")
    
    if os.path.exists(output_file):
        log(f"   [Skip] vol_{vol_idx}.stl already exists.")
        return True

    # Isolation: Setup Gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    
    try:
        gmsh.model.occ.importShapes(STEP_FILE)
        gmsh.model.occ.synchronize()
        
        vols = gmsh.model.getEntities(3)
        if vol_idx >= len(vols):
            return False
            
        dim, tag = vols[vol_idx]
        
        # 1. DELETE EVERYTHING EXCEPT THE TARGET VOLUME
        # This ensures Gmsh doesn't accidentally mesh neighbors or gaps.
        all_entities = gmsh.model.getEntities()
        to_delete = [e for e in all_entities if e != (3, tag)]
        gmsh.model.occ.remove(to_delete, recursive=True)
        gmsh.model.occ.synchronize()
        
        # 2. MESHING SETTINGS
        # Use Delaunay for robustness
        gmsh.option.setNumber("Mesh.Algorithm", 5) 
        gmsh.option.setNumber("Mesh.MeshSizeMin", MESH_SIZE)
        gmsh.option.setNumber("Mesh.MeshSizeMax", MESH_SIZE * 5.0)
        
        # 3. GENERATE SURFACE MESH
        gmsh.model.mesh.generate(2)
        
        # 4. EXPORT
        if not os.path.exists(TEMP_DIR):
            os.makedirs(TEMP_DIR)
        gmsh.write(output_file)
        
        gmsh.finalize()
        return True
    except Exception as e:
        if gmsh.isInitialized():
            gmsh.finalize()
        return False

def main():
    if "--worker" in sys.argv:
        idx_arg = sys.argv[sys.argv.index("--worker") + 1]
        if extract_volume_isolated(int(idx_arg)):
            sys.exit(0)
        else:
            sys.exit(1)

    # MASTER MODE
    t0 = time.time()
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    
    # Get volume count
    gmsh.initialize()
    gmsh.model.occ.importShapes(STEP_FILE)
    gmsh.model.occ.synchronize()
    len_vols = len(gmsh.model.getEntities(3))
    gmsh.finalize()
    
    log(f"Starting isolated extraction of {len_vols} volumes...")
    
    max_parallel = 4
    active_procs = []
    
    for i in range(len_vols):
        # Throttle
        while len(active_procs) >= max_parallel:
            time.sleep(0.1)
            active_procs = [p for p in active_procs if p.poll() is None]
        
        cmd = [sys.executable, __file__, "--worker", str(i)]
        p = subprocess.Popen(cmd)
        active_procs.append(p)
        
        if i % 10 == 0:
            log(f"Queued volume {i}/{len_vols}...")

    # Wait for all
    for p in active_procs:
        p.wait()

    log("Extraction complete. Merging results...")
    # Clean up old soup if exists
    if os.path.exists(OUTPUT_SOUP):
        os.remove(OUTPUT_SOUP)
        
    import pyvista as pv
    stls = [os.path.join(TEMP_DIR, f) for f in os.listdir(TEMP_DIR) if f.endswith(".stl")]
    
    if not stls:
        log("FAILURE: No STL files generated.")
        return

    # Merge into a single soup for reference, but we keep the shells for shell_preserver
    # Actually, let's just create a combined VTK directly
    all_meshes = []
    for f in stls:
        try:
            m = pv.read(f)
            if m.n_cells > 0:
                all_meshes.append(m)
        except: pass
        
    if all_meshes:
        combined = pv.MultiBlock(all_meshes).combine()
        combined.save(os.path.join(PROJECT_ROOT, "robust_soup.vtk"))
        # Also export as STL for tools that need it
        combined.extract_surface().save(OUTPUT_SOUP)
        log(f"SUCCESS: Created robust assembly with {len(all_meshes)} volumes.")
        log(f"Total Volume: {combined.volume:.2f}")
    
    log(f"Total Time: {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
