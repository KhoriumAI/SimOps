import gmsh
import sys
import os
import time

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
STEP_FILE = os.path.join(PROJECT_ROOT, "cad_files", "00-CM240411_BIANCA-C00_HEATER_BOAR_0704A1.STEP")
OUTPUT_VTK = os.path.join(PROJECT_ROOT, "simulation_ready.vtk")
OUTPUT_INP = os.path.join(PROJECT_ROOT, "simulation_ready.inp")

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def mesh_step_directly(step_file, output_file):
    log(f"--- STARTING DIRECT STEP MESHING: {step_file} ---")
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("RocketAssembly")

    # 1. LOAD STEP (Directly into OpenCascade Kernel)
    log("[1/5] Importing STEP file...")
    try:
        vols = gmsh.model.occ.importShapes(step_file)
        gmsh.model.occ.synchronize()
    except Exception as e:
        log(f"[X] FATAL: Could not import STEP. {e}")
        gmsh.finalize()
        return False

    raw_volumes = gmsh.model.getEntities(dim=3)
    log(f"[OK] Imported {len(raw_volumes)} volumes from source.")

    # 2. PLAN B: INDEPENDENT MESHING (Skip Fragment)
    log("[2/5] Skipping Boolean Fragment (Plan B: Independent Meshing)...")
    # try:
    #     gmsh.model.occ.fragment(raw_volumes, raw_volumes)
    #     gmsh.model.occ.synchronize()
    # except Exception as e:
    #     log(f"[X] FATAL: Boolean Fragment failed (Dirty CAD?). {e}")
    #     gmsh.finalize()
    #     return False
    
    post_frag_volumes = gmsh.model.getEntities(dim=3)
    log(f"[OK] Post-Fragment: {len(post_frag_volumes)} volumes.")

    # 3. MESH SETUP
    log("[3/5] Configuring Mesh options...")
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 5.0)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1) # 1=Delaunay (Robust)

    # 4. GENERATE MESH
    log("[4/5] Meshing Volume (This may take a few minutes)...")
    try:
        gmsh.model.mesh.generate(3)
    except Exception as e:
        log(f"[X] FATAL: Meshing failed. {e}")
        gmsh.finalize()
        return False

    # 5. EXPORT
    log(f"[5/5] Exporting to {output_file}...")
    gmsh.write(output_file)
    
    # Also export INP for CalculiX if needed
    inp_path = output_file.replace(".vtk", ".inp")
    gmsh.write(inp_path)
    log(f"   -> Also exported to {inp_path}")

    final_entities = gmsh.model.getEntities(dim=3)
    log(f"SUCCESS. Final Mesh contains {len(final_entities)} volumes.")
    gmsh.finalize()
    return True

if __name__ == "__main__":
    step = STEP_FILE
    output = OUTPUT_VTK
    
    if len(sys.argv) > 1:
        step = sys.argv[1]
    if len(sys.argv) > 2:
        output = sys.argv[2]
        
    success = mesh_step_directly(step, output)
    sys.exit(0 if success else 1)
