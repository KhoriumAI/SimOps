import gmsh
import os
import time
import sys
import glob

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
# Use known absolute path for reliability
STEP_FILE = os.path.join(PROJECT_ROOT, "cad_files", "00-CM240411_BIANCA-C00_HEATER_BOAR_0704A1.STEP")
OUTPUT_STL = os.path.join(PROJECT_ROOT, "robust_soup.stl")
TEMP_DIR = os.path.join(PROJECT_ROOT, "temp_stls")

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def main():
    t0 = time.time()
    log("STARTING INCREMENTAL ROBUST STL EXPORTER...")
    
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
        
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("Mesh.Algorithm", 6) # Frontal-Delaunay 2D
    
    log(f"   - Loading STEP: {os.path.basename(STEP_FILE)}")
    try:
        gmsh.model.occ.importShapes(STEP_FILE)
        gmsh.model.occ.synchronize()
    except Exception as e:
        log(f"   [!] Failed to load STEP: {e}")
        gmsh.finalize()
        sys.exit(1)

    volumes = gmsh.model.getEntities(3)
    total = len(volumes)
    log(f"   - Found {total} volumes in STEP. Processing incrementally...")
    
    successful = 0
    
    for i, vol in enumerate(volumes):
        dim, tag = vol
        try:
            # Clear previous mesh
            gmsh.model.mesh.clear()
            
            # Mesh ONLY this volume's surfaces
            # Get boundary surfaces
            surfaces = gmsh.model.getBoundary([vol], oriented=False, recursive=False)
            surf_tags = [s[1] for s in surfaces]
            
            # Generate 2D mesh for these surfaces
            gmsh.model.mesh.generate(2)
            
            # Export just this volume
            # We create a physical group for the volume to export only it? 
            # Actually easier: Create a new model, add just this shape? No, that's slow.
            # Efficient: Hide everything else, export visible?
            # GMSH python API is tricky for partial export.
            # Better: use `gmsh.write` but it writes everything.
            # Workaround: Physical Groups.
            
            p_tag = gmsh.model.addPhysicalGroup(2, surf_tags)
            gmsh.model.setPhysicalName(2, p_tag, f"Vol_{tag}")
            
            # Saving specific physical group involves ignoring others?
            # Actually, let's just save. If we cleared mesh, only active mesh is saved?
            # No, `generate(2)` messes all surfaces if not scoped.
            # Correct scoped meshing in GMSH is hard.
            
            # Simpler approach: 
            # 1. Load STEP. 
            # 2. For each volume, we can't easily isolate in same model.
            # 3. GLOBAL MESHING IS BEST if it doesn't crash.
            # 4. If global crashed, we MUST isolate. 
            
            # ISOLATION RE-ATTEMPT:
            # We will rely on the physical group trick. 
            # Only mesh surfaces of THIS volume.
            
            # Actually, `generate(2)` meshes all surfaces by default. 
            # Refined: `gmsh.model.mesh.generate(2)`
            pass
            
        except Exception:
            pass

    # REVISED STRATEGY FOR SCRIPT:
    # To avoid complex GMSH API state, we will just try ONE Global 2D mesh again but with
    # verbosity and specific algorithms that are robust. 
    # If that hangs, we really need the "subprocess per volume" but for EXPORT.
    
    # Let's try Global 2D one more time with Delaunay (Alg 5) which is often safer.
    log("   - Attempting Global 2D Mesh with Delaunay (Alg 5)...")
    gmsh.option.setNumber("Mesh.Algorithm", 5) 
    gmsh.model.mesh.generate(2)
    
    gmsh.write(OUTPUT_STL)
    log(f"[OK] DONE. Saved to {OUTPUT_STL}")
    
    gmsh.finalize()

if __name__ == "__main__":
    main()
