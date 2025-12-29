import gmsh
import os
import time
import sys

# CONFIGURATION
# Resolve path relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
STEP_FILENAME = "00-CM240411_BIANCA-C00_HEATER_BOAR_0704A1.STEP"
STEP_FILE = os.path.join(PROJECT_ROOT, "cad_files", STEP_FILENAME)
DEBUG_DIR = "debug_geometry"

def triage_geometry():
    print(f"============================================================")
    print(f"GEOMETRY TRIAGE & EXPLOSION")
    print(f"============================================================")
    
    if not os.path.exists(DEBUG_DIR): os.makedirs(DEBUG_DIR)
    
    # 1. LOAD MASTER (ONCE)
    print(f"[1/3] Loading Master File (This may take 10-20s)...")
    t0 = time.time()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.occ.importShapes(STEP_FILE)
        gmsh.model.occ.synchronize()
    except Exception as e:
        print(f"FATAL: Could not load file. {e}")
        return
    load_time = time.time() - t0
    print(f"   -> Load Complete in {load_time:.2f}s")

    # 2. ANALYZE COMPLEXITY
    print(f"[2/3] Analyzing Volumes...")
    vols = gmsh.model.getEntities(dim=3)
    shells = gmsh.model.getEntities(dim=2)
    print(f"   -> Volumes: {len(vols)}")
    print(f"   -> Loose Surfaces (Shells): {len(shells)}")
    
    # Store complexity info (Tag -> Face Count)
    complexity = []
    
    for dim, tag in vols:
        # Get boundary surfaces for this volume
        # (Adjacency check is fast in memory)
        faces = gmsh.model.getAdjacencies(dim, tag)[1]
        complexity.append( (tag, len(faces)) )

    # Sort by complexity (Face count)
    complexity.sort(key=lambda x: x[1], reverse=True)
    
    print("\n   TOP 5 HEAVIEST PARTS (Likely Troublemakers):")
    for tag, faces in complexity[:5]:
        print(f"      - Vol {tag}: {faces} Faces")

    # 3. EXPLODE (SAVE INDIVIDUAL CAD FILES)
    print(f"\n[3/3] Exploding into {DEBUG_DIR}/...")
    
    # We use a trick: We don't delete. We use `write` with visibility filtering if possible,
    # OR we re-load per part if memory allows. 
    # Actually, the fastest robust way for "Dirty" files is to iterate and Save.
    # Since 'remove' is slow, we will try a "Recursive Copy" strategy if supported,
    # OR just re-import for the heavy ones.
    
    # Let's try the safest export method:
    # We iterate the complexity list.
    
    success = 0
    for i, (tag, faces) in enumerate(complexity):
        file_name = f"vol_{tag}.brep"
        target_path = os.path.join(DEBUG_DIR, file_name)
        
        # skip if exists
        if os.path.exists(target_path):
            success += 1
            continue

        try:
            # CLEAR & RE-LOAD for perfect isolation
            # This incurs the 10s load penalty per part, but GUARANTEES isolation.
            # Given your crashes, stability > speed for this triage step.
            gmsh.model.mesh.clear()
            gmsh.clear() 
            gmsh.model.occ.importShapes(STEP_FILE)
            gmsh.model.occ.synchronize()
            
            # Remove everything else
            all_vols = gmsh.model.getEntities(dim=3)
            to_remove = [v for v in all_vols if v[1] != tag]
            
            # If there are loose shells, remove them too to keep file clean
            all_shells = gmsh.model.getEntities(dim=2)
            
            # Execute Cut
            gmsh.model.occ.remove(to_remove, recursive=True)
            # (Optional: remove shells if they aren't part of the volume)
            # gmsh.model.occ.remove(all_shells, recursive=True) 
            
            gmsh.model.occ.synchronize()
            
            # Export B-Rep (CAD format, not Mesh)
            gmsh.write(target_path)
            
            print(f"   -> Exported Vol {tag} ({faces} faces)")
            success += 1
            
        except Exception as e:
            print(f"   FAILED to export Vol {tag}: {e}")

    gmsh.finalize()
    print(f"\nTriage Complete. Check {DEBUG_DIR}/ to visualize individual parts.")

if __name__ == "__main__":
    triage_geometry()
