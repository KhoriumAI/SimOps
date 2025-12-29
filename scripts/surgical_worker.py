import gmsh
import os
import sys
import argparse
import time
import threading

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
STEP_FILE = os.path.join(PROJECT_ROOT, "cad_files", "00-CM240411_BIANCA-C00_HEATER_BOAR_0704A1.STEP")
TEMP_DIR = os.path.join(PROJECT_ROOT, "temp_defeatured")

DELETE_KEYWORDS = ["SCREW", "WASHER", "NUT", "SPACER", "BOLT", "HEADER"]
MAX_EDGES_FOR_SMALL_PART = 500
SMALL_PART_DIAGONAL_MM = 10.0 

def log(msg):
    print(msg, flush=True)

def create_box_mesh(tag, bbox):
    # Clears model and meshes a box in its place
    # We assume GMSH is already initialized
    gmsh.model.remove()
    gmsh.model.add(f"Box_{tag}")
    xmin, ymin, zmin, xmax, ymax, zmax = bbox
    dx, dy, dz = xmax-xmin, ymax-ymin, zmax-zmin
    if dx < 1e-4: dx = 0.1
    if dy < 1e-4: dy = 0.1
    if dz < 1e-4: dz = 0.1
    
    gmsh.model.occ.addBox(xmin, ymin, zmin, dx, dy, dz)
    gmsh.model.occ.synchronize()
    
    gmsh.option.setNumber("Mesh.Algorithm3D", 1) 
    gmsh.model.mesh.generate(3)
    
    # Physical group
    vols = gmsh.model.getEntities(3)
    if vols:
        vt = vols[0][1]
        pg = gmsh.model.addPhysicalGroup(3, [vt])
        gmsh.model.setPhysicalName(3, pg, f"Vol_{tag}")

def run_canary_2d():
    try:
        gmsh.option.setNumber("Mesh.Algorithm", 6) # Frontal Delaunay
        gmsh.model.mesh.generate(2)
    except:
        pass

def process_single(tag, force_box=False):
    vol_msh = os.path.join(TEMP_DIR, f"vol_{tag}.msh")
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(f"Worker_{tag}")
    
    try:
        # Load & Isolate
        gmsh.model.occ.importShapes(STEP_FILE)
        gmsh.model.occ.synchronize()
        
        dim_tags = gmsh.model.getEntities(3)
        target_name = ""
        found = False
        to_delete = []
        
        for d, t in dim_tags:
            if t == tag:
                target_name = gmsh.model.getEntityName(d, t)
                found = True
            else:
                to_delete.append((d, t))
                
        if not found:
            log(f"SKIP_LOST: {tag}")
            gmsh.finalize()
            return
            
        # 1. KEYWORD CHECK
        uname = target_name.upper()
        for kw in DELETE_KEYWORDS:
            if kw in uname:
                log(f"DELETE: {tag} ({kw})")
                gmsh.finalize()
                return

        # Get BBox early
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(3, tag)
        diag = ((xmax-xmin)**2 + (ymax-ymin)**2 + (zmax-zmin)**2)**0.5

        if force_box:
            create_box_mesh(tag, (xmin, ymin, zmin, xmax, ymax, zmax))
            gmsh.write(vol_msh)
            log(f"BOXED: {tag}")
            gmsh.finalize()
            return

        # 2. CANARY CHECK A: 1D COMPLEXITY
        gmsh.model.occ.remove(to_delete, recursive=True)
        gmsh.model.occ.synchronize()
        
        edges = gmsh.model.getEntities(1)
        if (len(edges) > MAX_EDGES_FOR_SMALL_PART) and (diag < SMALL_PART_DIAGONAL_MM):
            log(f"CANARY_DIE: {tag} (Edges={len(edges)}). Swapping to BOX.")
            create_box_mesh(tag, (xmin, ymin, zmin, xmax, ymax, zmax))
            gmsh.write(vol_msh)
            log(f"BOXED: {tag}")
            gmsh.finalize()
            return

        # 3. CANARY CHECK B: 2D MESH (THREADED TIMEOUT)
        # We start 2D generation in a thread. If it takes > 5s, we kill and Box.
        
        canary_thread = threading.Thread(target=run_canary_2d)
        canary_thread.daemon = True # Kill when main dies
        canary_thread.start()
        canary_thread.join(timeout=5.0)
        
        if canary_thread.is_alive():
            log(f"CANARY_DIE: {tag} (2D Timeout). Swapping to BOX.")
            # We cannot easily stop the gmsh thread safely without crashing, 
            # so we might simply exit or try to proceed with box?
            # Actually, if GMSH is busy, we can't run create_box_mesh.
            # We must EXIT and let the Orchestrator handle the 'BOXing' via fallback!
            # The Orchestrator sees us exit with default code 0? No, we should error.
            # OR we signal Orchestrator to "BOX_ME".
            sys.exit(2) # Exit Code 2 = "Please Box Me"
        
        # 4. VOLUME MESH (Unlimited time, relative to orchestrator)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1) 
        gmsh.model.mesh.generate(3)
        
        if len(gmsh.model.mesh.getNodes()[0]) == 0:
            raise RuntimeError("Empty Mesh")
            
        # PHYSICAL GROUP 
        vols = gmsh.model.getEntities(3)
        if vols:
             vt = vols[0][1]
             pg = gmsh.model.addPhysicalGroup(3, [vt])
             gmsh.model.setPhysicalName(3, pg, f"Vol_{tag}")

        gmsh.write(vol_msh)
        log(f"SUCCESS: {tag}")
        
    except Exception as e:
        log(f"FAIL: {tag} - {e}")
        sys.exit(1) 
        
    gmsh.finalize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=int, required=True)
    parser.add_argument("--box", action="store_true")
    args = parser.parse_args()
    
    if not os.path.exists(TEMP_DIR):
        try: os.makedirs(TEMP_DIR)
        except: pass
        
    process_single(args.tag, args.box)
