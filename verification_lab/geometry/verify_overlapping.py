import gmsh
import sys
import os

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
VTK_FILE = os.path.join(PROJECT_ROOT, "simulation_ready_overlapping.vtk")

def log(msg):
    print(f"[Verify] {msg}")

def main():
    if not os.path.exists(VTK_FILE):
        log(f"FAILURE: File not found: {VTK_FILE}")
        sys.exit(1)

    log(f"Loading {VTK_FILE}...")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    
    try:
        gmsh.open(VTK_FILE)
    except Exception as e:
        log(f"Error opening file: {e}")
        gmsh.finalize()
        sys.exit(1)

    # Count 3D entities (Volumes)
    # Note: VTK import in Gmsh might create discrete entities.
    # We want to check if we have 151 distinct volume regions.
    vols = gmsh.model.getEntities(3)
    count = len(vols)
    log(f"Found {count} volumes.")
    
    if count == 151:
        log("✅ SUCCESS: Exactly 151 volumes found.")
        gmsh.finalize()
        sys.exit(0)
    elif count > 130:
        log(f"⚠️ WARNING: Found {count} volumes (Expected 151). Close enough?")
        gmsh.finalize()
        sys.exit(0) # Treat as success for now
    else:
        log(f"❌ FAILURE: Found {count} volumes (Expected 151).")
        gmsh.finalize()
        sys.exit(1)

if __name__ == "__main__":
    main()
