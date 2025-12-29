import gmsh
import sys
import os
import time

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
STEP_FILE = os.path.join(PROJECT_ROOT, "cad_files", "00-CM240411_BIANCA-C00_HEATER_BOAR_0704A1.STEP")
OUTPUT_VTK = os.path.join(PROJECT_ROOT, "simulation_ready_overlapping.vtk")
OUTPUT_INP = os.path.join(PROJECT_ROOT, "simulation_ready_overlapping.inp")

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def mesh_overlapping(step_file, output_file):
    log(f"--- STARTING PLAN C: OVERLAPPING MESH ---")
    log(f"Input: {step_file}")
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("RocketAssembly_Overlapping")

    # 1. LOAD STEP
    log("[1/4] Importing STEP file...")
    try:
        gmsh.model.occ.importShapes(step_file)
        gmsh.model.occ.synchronize()
    except Exception as e:
        log(f"FATAL: Could not import STEP. {e}")
        gmsh.finalize()
        return False

    raw_volumes = gmsh.model.getEntities(dim=3)
    log(f"SUCCESS: Imported {len(raw_volumes)} volumes.")

    # 2. NO BOOLEAN FRAGMENT (The breakdown of Plan A/B)
    log("[2/4] SKIPPING Boolean Fragment (Intentional).")
    log("      Nodes will NOT be conformal at interfaces.")
    log("      Overlaps WILL exist.")
    log("      Solver TIE constraints will handle connectivity.")

    # 3. MESH SETUP
    log("[3/4] Configuring Mesh options...")
    # Using settings proven in Core Sample 'default' strategy
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 5.0)
    
    # Use HXT (Parallel Delaunay) for speed if available, otherwise default
    gmsh.option.setNumber("Mesh.Algorithm3D", 1) # 1=Delaunay (Robust), 10=HXT (Fast/Fragile)
    # Important: Optimize netgen/hxt for robustness over quality
    gmsh.option.setNumber("Mesh.Optimize", 1) 

    # 4. GENERATE MESH
    log("[4/4] Meshing 151 Volumes (Parallel)...")
    t0 = time.time()
    try:
        gmsh.model.mesh.generate(3)
    except Exception as e:
        log(f"FATAL: Meshing failed. {e}")
        gmsh.finalize()
        return False
    
    duration = time.time() - t0
    log(f"Meshing completed in {duration:.2f}s.")

    # 5. EXPORT
    log(f"Exporting to {output_file}...")
    gmsh.write(output_file)
    
    # Also export INP for CalculiX/Abaqus solvers (TIE constraints use this format)
    inp_path = output_file.replace(".vtk", ".inp")
    gmsh.write(inp_path)
    log(f"Generated {inp_path}")

    # Validation
    # Check if we have 3D elements for all volumes?
    # Actually, we can just check total element count or just success.
    # Ideally confirm 151 volumes are present in the mesh.
    
    # Note: dim=3 entities still exist.
    final_entities = gmsh.model.getEntities(dim=3)
    log(f"FINAL CHECK: Mesh contains {len(final_entities)} volumes.")
    
    gmsh.finalize()
    return True

if __name__ == "__main__":
    success = mesh_overlapping(STEP_FILE, OUTPUT_VTK)
    sys.exit(0 if success else 1)
