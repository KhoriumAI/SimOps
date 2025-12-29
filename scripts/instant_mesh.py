#!/usr/bin/env python3
"""
INSTANT MESH - Memory-Only Pipeline
====================================
1. One Load, One Mesh (with Gmsh internal threading)
2. Virtual Separation (RAM extraction, no file I/O)
3. Boolean Fusion (Manifold3D)
4. TetWild Draft Mode

Expected time: ~2-3 minutes total
"""

import gmsh
import numpy as np
import subprocess
import os
import time
import sys
import signal
import psutil

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
STEP_FILE = os.path.join(PROJECT_ROOT, "cad_files", "00-CM240411_BIANCA-C00_HEATER_BOAR_0704A1.STEP")
FUSED_STL = os.path.join(PROJECT_ROOT, "fused_temp.stl")
FINAL_MESH = os.path.join(PROJECT_ROOT, "robust_mesh.msh")

# Gap size (coarse for speed)
GAP_SIZE = 0.5 

# --- CLEANUP TRACKING ---
child_processes = []
current_process = psutil.Process()

def cleanup_all():
    """Kill all child processes and cleanup temp files"""
    log("Cleaning up child processes...")
    
    # Kill all children of this process
    try:
        children = current_process.children(recursive=True)
        for child in children:
            try:
                log(f"  Killing PID {child.pid} ({child.name()})")
                child.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except Exception as e:
        log(f"  [!] Cleanup error: {e}")
    
    # Kill tracked processes
    for proc in child_processes:
        try:
            if proc.poll() is None:  # Still running
                proc.kill()
                log(f"  Killed subprocess PID {proc.pid}")
        except Exception:
            pass

def signal_handler(signum, frame):
    """Handle Ctrl+C and other signals"""
    log("")
    log("[!] Interrupted by user. Cleaning up...")
    cleanup_all()
    sys.exit(1)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def main():
    t0 = time.time()
    log("=" * 60)
    log("INSTANT MESH - Memory-Only Pipeline")
    log("=" * 60)
    log(f"Input: {STEP_FILE}")
    log(f"Gap Size: {GAP_SIZE}")
    log("")

    # =========================================================================
    # PHASE 1: ONE LOAD, ONE MESH
    # =========================================================================
    log("PHASE 1: Load and Mesh (Gmsh internal threading)")
    log("-" * 40)
    
    log("Loading STEP file (once)...")
    phase1_start = time.time()
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("General.NumThreads", 30)  # Use all cores for meshing
    
    # Aggressive healing
    gmsh.option.setNumber("Geometry.OCCFixSmallEdges", 1)
    gmsh.option.setNumber("Geometry.OCCFixSmallFaces", 1)
    
    try:
        gmsh.model.occ.importShapes(STEP_FILE)
        gmsh.model.occ.synchronize()
    except Exception as e:
        log(f"[X] Failed to load STEP: {e}")
        gmsh.finalize()
        sys.exit(1)
    
    volumes = gmsh.model.getEntities(3)
    log(f"Loaded {len(volumes)} volumes")
    
    # GLOVES OFF MESHING
    log(f"Meshing surfaces (parallel, Min={GAP_SIZE}, Max={GAP_SIZE*5})...")
    gmsh.option.setNumber("Mesh.MeshSizeMin", GAP_SIZE)
    gmsh.option.setNumber("Mesh.MeshSizeMax", GAP_SIZE * 5.0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal Delaunay (Fast)
    gmsh.option.setNumber("Mesh.Optimize", 0)
    gmsh.option.setNumber("Mesh.MaxRetries", 0)
    
    try:
        gmsh.model.mesh.generate(2)  # 2D ONLY
        log("[OK] 2D mesh generated")
    except Exception as e:
        log(f"[!] Mesh generation warning: {e}")
        log("[!] Continuing with partial mesh...")
    
    phase1_time = time.time() - phase1_start
    log(f"Phase 1 complete: {phase1_time:.1f}s")
    log("")

    # =========================================================================
    # PHASE 2: TRIANGLE SOUP EXTRACTION (No Boolean, No Validation)
    # =========================================================================
    log("PHASE 2: Triangle Soup Extraction (Raw Concatenation)")
    log("-" * 40)
    
    phase2_start = time.time()
    
    # Import trimesh
    try:
        import trimesh
    except ImportError as e:
        log(f"[X] Missing dependency: {e}")
        log("    Run: pip install trimesh")
        gmsh.finalize()
        sys.exit(1)
    
    # Get all nodes once (Vectorized = Fast)
    log("Extracting all triangles as soup...")
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    nodes = np.array(coords, dtype=np.float32).reshape(-1, 3)
    
    # Create tag->index mapping
    node_map = {int(tag): i for i, tag in enumerate(node_tags)}
    
    # Get ALL triangles from the entire model at once
    # getDimension=2 for surfaces, tag=-1 for all entities
    elem_types, elem_tags_list, elem_node_tags_list = gmsh.model.mesh.getElements(2, -1)
    
    gmsh.finalize()
    
    if len(elem_node_tags_list) == 0:
        log("[X] Error: Gmsh generated no triangles.")
        sys.exit(1)
    
    # Flatten all element node tags (handles multiple entity blocks)
    flat_node_tags = np.concatenate(elem_node_tags_list).astype(np.int64)
    
    # Convert tags to indices using vectorized approach
    # Get unique tags and create compact vertex array
    unique_tags, inverse_indices = np.unique(flat_node_tags, return_inverse=True)
    
    # Filter nodes to only those actually used
    used_nodes = np.array([nodes[node_map[int(t)]] for t in unique_tags])
    
    # Faces are the inverse indices reshaped into triangles
    faces = inverse_indices.reshape(-1, 3)
    
    log(f"[OK] Extracted {len(faces):,} triangles, {len(used_nodes):,} vertices")
    
    phase2_time = time.time() - phase2_start
    log(f"Phase 2 complete: {phase2_time:.1f}s")
    log("")

    # =========================================================================
    # PHASE 3: EXPORT SOUP (Simple Concatenation, No Boolean)
    # =========================================================================
    log("PHASE 3: Exporting Triangle Soup")
    log("-" * 40)
    
    phase3_start = time.time()
    log("Creating STL (no union, just raw triangles)...")
    
    # Create and export mesh
    mesh = trimesh.Trimesh(vertices=used_nodes, faces=faces, process=False)
    mesh.export(FUSED_STL)
    
    phase3_time = time.time() - phase3_start
    soup_size = os.path.getsize(FUSED_STL) / (1024 * 1024)
    log(f"[OK] Soup STL: {soup_size:.2f} MB in {phase3_time:.1f}s")
    log("")

    # =========================================================================
    # PHASE 4: TETWILD (Draft Mode)
    # =========================================================================
    log("PHASE 4: TetWild Volume Meshing (Docker)")
    log("-" * 40)
    
    phase4_start = time.time()
    
    stl_basename = os.path.basename(FUSED_STL)
    mesh_basename = os.path.basename(FINAL_MESH)
    
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{PROJECT_ROOT}:/data",
        "yixinhu/tetwild:latest",
        "--input", f"/data/{stl_basename}",
        "--output", f"/data/{mesh_basename}",
        "--level", "2",
    ]
    
    log(f"Command: {' '.join(cmd)}")
    log("")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Track this process for cleanup
        child_processes.append(process)
        
        for line in process.stdout:
            print(f"    {line.rstrip()}", flush=True)
        
        process.wait()
        
        phase4_time = time.time() - phase4_start
        
        if process.returncode == 0:
            log(f"[OK] TetWild completed in {phase4_time:.1f}s")
            if os.path.exists(FINAL_MESH):
                mesh_size = os.path.getsize(FINAL_MESH) / (1024 * 1024)
                log(f"[OK] Output: {FINAL_MESH} ({mesh_size:.2f} MB)")
        else:
            log(f"[X] TetWild failed with code {process.returncode}")
            if process.returncode == 137:
                log("[!] Exit 137 = OOM. Increase Docker memory.")
                
    except Exception as e:
        log(f"[X] Docker error: {e}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    total_time = time.time() - t0
    log("")
    log("=" * 60)
    log(f"TOTAL TIME: {total_time:.1f}s ({total_time/60:.1f} min)")
    log("=" * 60)
    log(f"  Phase 1 (Load+Mesh): {phase1_time:.1f}s")
    log(f"  Phase 2 (Soup Extract): {phase2_time:.1f}s")
    log(f"  Phase 3 (Export): {phase3_time:.1f}s")
    log(f"  Phase 4 (TetWild): {phase4_time:.1f}s")
    
    if os.path.exists(FINAL_MESH):
        log("")
        log("[OK] SUCCESS!")
        log(f"     Output: {FINAL_MESH}")
    else:
        log("")
        log("[X] FAILED - No output mesh")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    finally:
        # Always cleanup on exit
        cleanup_all()
