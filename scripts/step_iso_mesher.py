import gmsh
import os
import time
import sys
import subprocess
import pyvista as pv

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
STEP_FILE = os.path.join(PROJECT_ROOT, "cad_files", "00-CM240411_BIANCA-C00_HEATER_BOAR_0704A1.STEP")
TEMP_DIR = os.path.join(PROJECT_ROOT, "temp_meshes")
OUTPUT_VTK = os.path.join(PROJECT_ROOT, "simulation_ready.vtk")

def log(msg):
    # Avoid unicode in logs for safety
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def mesh_volume_isolated(vol_idx):
    """Worker function to 3D mesh a single volume."""
    output_file = os.path.join(TEMP_DIR, f"vol_{vol_idx}.vtk")
    
    if os.path.exists(output_file):
        log(f"   [Skip] Volume {vol_idx} already exists.")
        return True
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    
    try:
        # Load output format as VTK
        # gmsh.option.setNumber("Mesh.FileFormat", 1) 
        
        # Import
        gmsh.model.occ.importShapes(STEP_FILE)
        gmsh.model.occ.synchronize()
        
        vols = gmsh.model.getEntities(3)
        if vol_idx >= len(vols):
            return False
            
        dim, tag = vols[vol_idx]
        
        # Isolate
        all_entities = gmsh.model.getEntities(3)
        to_delete = [e for e in all_entities if e != (3, tag)]
        gmsh.model.occ.remove(to_delete, recursive=True)
        gmsh.model.occ.synchronize()
        
        # Mesh Settings (Plan B: Independent)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 5.0)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1) # Delaunay
        
        # Generate 3D Mesh
        gmsh.model.mesh.generate(3)
        
        # Export
        if not os.path.exists(TEMP_DIR):
            os.makedirs(TEMP_DIR)
        gmsh.write(output_file)
        
        gmsh.finalize()
        return True
    except Exception as e:
        log(f"   [ERROR] Volume {vol_idx} failed: {e}")
        if gmsh.isInitialized():
            gmsh.finalize()
        return False

def main():
    if "--worker" in sys.argv:
        idx_arg = sys.argv[sys.argv.index("--worker") + 1]
        try:
            if mesh_volume_isolated(int(idx_arg)):
                sys.exit(0)
            else:
                sys.exit(1)
        except:
            sys.exit(1)

    # MASTER MODE
    t0 = time.time()
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    
    # Get count
    gmsh.initialize()
    gmsh.model.occ.importShapes(STEP_FILE)
    gmsh.model.occ.synchronize()
    len_vols = len(gmsh.model.getEntities(3))
    gmsh.finalize()
    
    log(f"Starting Independent 3D Meshing for {len_vols} volumes...")
    
    max_parallel = 4
    active_procs = []
    
    for i in range(len_vols):
        while len(active_procs) >= max_parallel:
            time.sleep(0.1)
            active_procs = [p for p in active_procs if p.poll() is None]
        
        cmd = [sys.executable, __file__, "--worker", str(i)]
        p = subprocess.Popen(cmd)
        active_procs.append(p)
        
        if i % 10 == 0:
            log(f"Queued volume {i}/{len_vols}...")

    for p in active_procs:
        p.wait()

    log("Meshing complete. Merging into final assembly...")
    
    files = [os.path.join(TEMP_DIR, f) for f in os.listdir(TEMP_DIR) if f.endswith(".vtk")]
    if not files:
        log("FAILURE: No meshes generated.")
        return

    # Merge using PyVista
    valid_meshes = []
    for f in files:
        try:
            m = pv.read(f)
            if m.n_cells > 0:
                valid_meshes.append(m)
        except Exception as e:
            log(f"Warning: Could not read {f}: {e}")

    if valid_meshes:
        assembly = pv.MultiBlock(valid_meshes).combine()
        assembly.save(OUTPUT_VTK)
        log(f"SUCCESS: Created {OUTPUT_VTK} with {len(valid_meshes)} volumes.")
        log(f"Total Volume: {assembly.volume:.2f}")
    
    log(f"Total Time: {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
