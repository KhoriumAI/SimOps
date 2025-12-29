import gmsh
import os
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
STEP_FILE = os.path.join(PROJECT_ROOT, "cad_files", "00-CM240411_BIANCA-C00_HEATER_BOAR_0704A1.STEP")

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

try:
    log(f"Test loading (Root Context): {STEP_FILE}")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.occ.importShapes(STEP_FILE)
    gmsh.model.occ.synchronize()
    volumes = gmsh.model.getEntities(3)
    log(f"SUCCESS: Found {len(volumes)} volumes.")
    gmsh.finalize()
except Exception as e:
    log(f"FAILURE: {e}")
