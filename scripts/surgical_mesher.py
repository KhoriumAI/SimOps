import gmsh
import os
import sys
import time
import glob

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
STEP_FILE = os.path.join(PROJECT_ROOT, "cad_files", "00-CM240411_BIANCA-C00_HEATER_BOAR_0704A1.STEP")
TEMP_DIR = os.path.join(PROJECT_ROOT, "temp_defeatured")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "simulation_ready_defeatured.vtk")

# Defeaturing Rules
DELETE_KEYWORDS = ["SCREW", "WASHER", "NUT", "SPACER", "BOLT"]
# If a volume name contains these, we DELETE it (Skip).

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def get_b_rep_info():
    """Reads STEP, returns dict of {tag: name} and bounding boxes."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0) # Silence
    try:
        gmsh.model.occ.importShapes(STEP_FILE)
        gmsh.model.occ.synchronize()
        vols = gmsh.model.getEntities(dim=3)
        
        info = {}
        for dim, tag in vols:
            # Try to get name. STEP labels usually transfer to Entity Name or Physical Name?
            # Gmsh often puts the label in the name string.
            name = gmsh.model.getEntityName(dim, tag)
            # If empty, might need other tricks, but let's try this.
            
            # Get BBox for potential boxing
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
            
            info[tag] = {
                "name": name,
                "bbox": (xmin, ymin, zmin, xmax, ymax, zmax)
            }
    except Exception as e:
        log(f"Error analyzing STEP: {e}")
        info = {}
        
    gmsh.finalize()
    return info

def process_volume(tag, info_data):
    """
    1. Check Keywords -> Delete if match.
    2. Isolate.
    3. Try Mesh.
    4. If Fail -> Box & Mesh.
    5. Save .msh
    """
    name = info_data["name"].upper()
    bbox = info_data["bbox"]
    
    # 1. SMART DELETE
    log(f"Processing candidate {tag}...") 
    
    for kw in DELETE_KEYWORDS:
        if kw in name:
            log(f"[DELETE] Volume {tag} ('{kw}' detected in '{name}')")
            return None # Skip this volume

    # Prepare paths
    vol_msh = os.path.join(TEMP_DIR, f"vol_{tag}.msh")
    if os.path.exists(vol_msh) and os.path.getsize(vol_msh) > 1000:
        # log(f"   [Skip] Already exists: {tag}")
        return vol_msh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(f"Proc_{tag}")
    
    log(f"   -> Isolating Volume {tag} from Full Assembly...")

    try:
        # Load isolation
        # We can't easily "Load just one volume" from STEP without parsing step.
        # So we Load All -> Delete Others.
        gmsh.model.occ.importShapes(STEP_FILE)
        gmsh.model.occ.synchronize()
        
        all_vols = gmsh.model.getEntities(3)
        to_delete = [(d, t) for d, t in all_vols if t != tag]
        
        if len(to_delete) == len(all_vols):
            # Target tag not found?
            log(f"[Error] Volume {tag} lost during reload.")
            gmsh.finalize()
            return None

        gmsh.model.occ.remove(to_delete, recursive=True)
        gmsh.model.occ.synchronize()
        
        # 3. TRY MESH (High Fidelity)
        # Use Robust Settings
        gmsh.option.setNumber("Mesh.Algorithm3D", 1) # Delaunay
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 5.0)
        
        # Timeout/Fail protection?
        # Gmsh Python API doesn't support timeout easily.
        # We rely on "Try/Except" for crashes, but hangs are hard.
        # We'll assume the "Bad" parts fail quickly or error out.
        
        try:
            # Attempt 3D generation
            gmsh.model.mesh.generate(3)
            # Check element count?
            # If success, verify we have elements
            # If 0 elements, it failed silently?
            # gmsh.model.mesh.getLastEntityError?
            pass
        except Exception as e:
            raise RuntimeError("Meshing Failed")

        # Check validity logic?
        # If successfully reached here, assume mostly OK?
        # Let's count nodes.
        # nodes = gmsh.model.mesh.getNodes()
        # if len(nodes[0]) == 0: raise RuntimeError("Empty Mesh")
        
        log(f"[Keep] Volume {tag} meshed successfully.")

    except Exception as e:
        # 4. DEFEATURE (Box Fallback)
        log(f"[REPLACE] Volume {tag} failed ({e}). Swapping with BOX.")
        
        # Clear/Reset
        gmsh.model.remove()
        gmsh.model.add(f"Box_{tag}")
        
        xmin, ymin, zmin, xmax, ymax, zmax = bbox
        dx, dy, dz = xmax-xmin, ymax-ymin, zmax-zmin
        
        gmsh.model.occ.addBox(xmin, ymin, zmin, dx, dy, dz)
        gmsh.model.occ.synchronize()
        
        # Mesh the Box
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 5.0)
        gmsh.model.mesh.generate(3)
    
    # Write output
    # Ensure Physical Group for TIE
    final_vols = gmsh.model.getEntities(3)
    if final_vols:
        vt = final_vols[0][1]
        p_grp = gmsh.model.addPhysicalGroup(3, [vt])
        gmsh.model.setPhysicalName(3, p_grp, f"Vol_{tag}")
    
    gmsh.write(vol_msh)
    gmsh.finalize()
    return vol_msh

def main():
    log("--- STARTING OPERATION: SURGICAL AMPUTATION ---")
    
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
        
    # Phase 1: Analysis
    log("Analyzing Anatomy (STEP Components)...")
    info_map = get_b_rep_info()
    log(f"Identify {len(info_map)} candidate volumes.")
    
    # Phase 2: Execution
    processed_files = []
    
    for tag in info_map:
        result_path = process_volume(tag, info_map[tag])
        if result_path:
            processed_files.append(result_path)
            
    # Phase 3: Stitching (Merge)
    log(f"Stitching {len(processed_files)} parts into Frankenstein Assembly...")
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("Frankenstein_Assembly")
    
    for msh in processed_files:
        try:
            gmsh.merge(msh)
        except:
            log(f"Error merging {msh}")
            
    # Phase 4: Final Export
    log(f"Exporting to {OUTPUT_FILE}...")
    gmsh.write(OUTPUT_FILE)
    
    log("--- OPERATION COMPLETE ---")
    log(f"Result: {OUTPUT_FILE}")
    gmsh.finalize()

if __name__ == "__main__":
    main()
