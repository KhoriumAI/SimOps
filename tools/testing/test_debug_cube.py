#!/usr/bin/env python3
"""Ultra-minimal test with logging after every single operation"""
import gmsh
import sys
import time

def log(msg):
    print(f"[{time.time():.3f}] {msg}", flush=True)

log("Starting...")
gmsh.initialize()
log("[OK] Initialized")

gmsh.option.setNumber("General.Terminal", 1)
log("[OK] Set terminal output")

log("Loading Cube.step...")
gmsh.open("CAD_files/Cube.step")
log("[OK] Loaded Cube.step")

# Set mesh sizes
log("Setting mesh sizes...")
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.1)  # 100mm
log(f"[OK] Set mesh sizes: min=0, max=100mm")

# Verify
cl_min = gmsh.option.getNumber("Mesh.CharacteristicLengthMin")
cl_max = gmsh.option.getNumber("Mesh.CharacteristicLengthMax")
log(f"[OK] Verified: min={cl_min*1000:.1f}mm, max={cl_max*1000:.1f}mm")

# Set algorithm
log("Setting algorithm...")
gmsh.option.setNumber("Mesh.Algorithm", 6)
log("[OK] Algorithm: Frontal-Delaunay")

# Generate 1D
log("Generating 1D mesh...")
start = time.time()
gmsh.model.mesh.generate(1)
elapsed = time.time() - start
log(f"[OK] 1D done in {elapsed:.3f}s")

# Count 1D
nodes = gmsh.model.mesh.getNodes()
log(f"[OK] 1D result: {len(nodes[0])} nodes")

# Generate 2D with manual flush
log("Generating 2D mesh...")
sys.stdout.flush()
start = time.time()

try:
    # Set a timeout manually
    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError("2D meshing timeout!")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(5)  # 5 second timeout

    gmsh.model.mesh.generate(2)
    signal.alarm(0)  # Cancel alarm

    elapsed = time.time() - start
    log(f"[OK] 2D done in {elapsed:.3f}s")

    # Count 2D
    elem_2d = gmsh.model.mesh.getElements(2)
    num_2d = sum(len(tags) for tags in elem_2d[1])
    log(f"[OK] 2D result: {num_2d} elements")

except TimeoutError as e:
    log(f"[X] TIMEOUT: {e}")
    log("2D meshing is hanging - gmsh is stuck in infinite loop")
    sys.exit(1)
except Exception as e:
    log(f"[X] ERROR: {e}")
    sys.exit(1)
finally:
    gmsh.finalize()

log("SUCCESS - mesh generated")
