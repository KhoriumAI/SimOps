import gmsh
import sys
import os
import math

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
# Use the correct relative path to the CAD file
STEP_FILE = os.path.join(PROJECT_ROOT, "cad_files", "00-CM240411_BIANCA-C00_HEATER_BOAR_0704A1.STEP")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "core_sample.step") # Changed to .step for compatibility
BOX_SIZE = 20.0 # 20mm cube

def log(msg):
    print(f"[CoreSample] {msg}")

def main():
    log(f"Initializing Gmsh...")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    
    # 1. Import STEP
    log(f"Importing STEP file: {STEP_FILE}")
    try:
        gmsh.model.occ.importShapes(STEP_FILE)
        gmsh.model.occ.synchronize()
    except Exception as e:
        log(f"Error importing STEP: {e}")
        gmsh.finalize()
        sys.exit(1)

    # 2. Compute Bounding Box to find center
    # Note: We want a dense section. The center of the bounding box is a good heuristic.
    # Get all volumes
    vols = gmsh.model.getEntities(3)
    if not vols:
        log("No volumes found!")
        gmsh.finalize()
        sys.exit(1)

    # Calculate overall bounding box
    min_x, min_y, min_z = float('inf'), float('inf'), float('inf')
    max_x, max_y, max_z = float('-inf'), float('-inf'), float('-inf')

    for dim, tag in vols:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
        min_x = min(min_x, xmin)
        min_y = min(min_y, ymin)
        min_z = min(min_z, zmin)
        max_x = max(max_x, xmax)
        max_y = max(max_y, ymax)
        max_z = max(max_z, zmax)
    
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    center_z = (min_z + max_z) / 2.0

    log(f"Model Bounds: [{min_x:.2f}, {min_y:.2f}, {min_z:.2f}] to [{max_x:.2f}, {max_y:.2f}, {max_z:.2f}]")
    log(f"Targeting Center: [{center_x:.2f}, {center_y:.2f}, {center_z:.2f}]")

    # 3. Create Cutting Box
    # Centered at the model center
    box_x = center_x - (BOX_SIZE / 2.0)
    box_y = center_y - (BOX_SIZE / 2.0)
    box_z = center_z - (BOX_SIZE / 2.0)
    
    log(f"Creating Core Box at [{box_x:.2f}, {box_y:.2f}, {box_z:.2f}] with size {BOX_SIZE}")
    cut_tool = gmsh.model.occ.addBox(box_x, box_y, box_z, BOX_SIZE, BOX_SIZE, BOX_SIZE)
    gmsh.model.occ.synchronize()

    # 4. Intersect (The "Core Sample")
    # We want to keep the intersection of the Model AND the Box.
    # intersect(objectDimTags, toolDimTags)
    log("Running Intersection...")
    try:
        # We intersect ALL volumes with our tool box
        # Note: intersect returns the intersection, but might destroy the originals.
        # We pass -1 as tag to preserve nothing (default behavior is usually consume).
        # Actually in OCC API: intersect(objectDimTags, toolDimTags, tag=-1, removeObject=True, removeTool=True)
        # We want to intersect the whole assembly with the single box.
        
        tool_tag = (3, cut_tool)
        
        # Intersection
        # warning: intersect might return a list of bodies.
        out_dimtags, _ = gmsh.model.occ.intersect(vols, [tool_tag], removeObject=True, removeTool=True)
        gmsh.model.occ.synchronize()
        
        log(f"Intersection complete. Resulting fragments: {len(out_dimtags)}")
        
        if len(out_dimtags) == 0:
            log("Warning: Core sample is empty! Attempting to move box to origin (0,0,0) just in case...")
            # Fallback/Debug: Maybe the model isn't centered where we thought. 
            # But let's verify first.
            
    except Exception as e:
        log(f"Intersection failed: {e}")
        gmsh.finalize()
        sys.exit(1)

    # 5. Export
    log(f"Exporting to {OUTPUT_FILE}...")
    gmsh.write(OUTPUT_FILE)
    gmsh.finalize()
    log("Done.")

if __name__ == "__main__":
    main()
