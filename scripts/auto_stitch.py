#!/usr/bin/env python3
"""
AUTO STITCH - Parallel "Stitch & Shrink" Pipeline
==================================================
1. Blast: Mesh all 151 volumes independently (parallel)
2. Fuse: Boolean union with Manifold3D (removes internal overlaps)
3. Volume Mesh: TetWild on the clean shell

This approach avoids the OOM issues from meshing a 734MB STL.
"""

import gmsh
import multiprocessing
import os
import time
import glob
import subprocess
import sys

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
STEP_FILE = os.path.join(PROJECT_ROOT, "cad_files", "00-CM240411_BIANCA-C00_HEATER_BOAR_0704A1.STEP")
TEMP_DIR = os.path.join(PROJECT_ROOT, "temp_parts")
OUTPUT_FUSED_STL = os.path.join(PROJECT_ROOT, "fused_assembly.stl")
FINAL_MESH = os.path.join(PROJECT_ROOT, "final_thermal_mesh.msh")
CORES = 30  # Use most of your power

def log(msg):
    """Print with timestamp."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# --- PART 1: PARALLEL SURFACE MESHING ---
def mesh_volume_worker(args):
    """Worker function to mesh a SINGLE volume ID."""
    vol_tag, step_path, out_dir = args
    
    try:
        # Initialize separate instance per core
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0) 
        gmsh.model.occ.importShapes(step_path)
        gmsh.model.occ.synchronize()
        
        # Isolate the specific volume, delete others
        all_vols = gmsh.model.getEntities(3)
        to_delete = [v for v in all_vols if v[1] != vol_tag]
        gmsh.model.occ.remove(to_delete, recursive=True)
        gmsh.model.occ.synchronize()
        
        # GLOVES OFF SETTINGS: Coarse but valid
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5) 
        gmsh.option.setNumber("Mesh.MeshSizeMax", 5.0) 
        gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal Delaunay
        gmsh.option.setNumber("Mesh.Optimize", 0)
        gmsh.option.setNumber("Mesh.MaxRetries", 0)
        
        gmsh.model.mesh.generate(2)
        out_file = os.path.join(out_dir, f"vol_{vol_tag}.stl")
        gmsh.write(out_file)
        gmsh.finalize()
        return (vol_tag, True, None)
    except Exception as e:
        try:
            gmsh.finalize()
        except:
            pass
        return (vol_tag, False, str(e))

def run_parallel_meshing():
    log("=" * 60)
    log(f"PHASE 1: Parallel Volume Meshing ({CORES} cores)")
    log("=" * 60)
    
    if not os.path.exists(TEMP_DIR): 
        os.makedirs(TEMP_DIR)
    else:
        # Clean old STLs
        for f in glob.glob(os.path.join(TEMP_DIR, "*.stl")):
            os.remove(f)
    
    # Get Volume Tags first (quick serial check)
    log("Loading STEP file to enumerate volumes...")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.occ.importShapes(STEP_FILE)
    gmsh.model.occ.synchronize()
    vols = [v[1] for v in gmsh.model.getEntities(3)]
    gmsh.finalize()
    
    log(f"Found {len(vols)} volumes. Starting parallel meshing...")
    
    # Prepare arguments
    tasks = [(v, STEP_FILE, TEMP_DIR) for v in vols]
    
    # Execute Parallel
    start = time.time()
    with multiprocessing.Pool(CORES) as pool:
        results = pool.map(mesh_volume_worker, tasks)
    
    elapsed = time.time() - start
    
    # Report results
    success = sum(1 for r in results if r[1])
    failed = sum(1 for r in results if not r[1])
    
    log(f"[OK] Meshed {success}/{len(vols)} volumes in {elapsed:.1f}s")
    if failed > 0:
        log(f"[!] {failed} volumes failed (will be skipped in fusion)")
        for r in results:
            if not r[1]:
                log(f"    - Volume {r[0]}: {r[2]}")
    
    return success > 0

# --- PART 2: THE "SEWING" (BOOLEAN UNION) ---
def fuse_shells():
    log("=" * 60)
    log("PHASE 2: Boolean Union (Manifold3D)")
    log("=" * 60)
    
    try:
        from manifold3d import Manifold, Mesh
        import trimesh
        import numpy as np
    except ImportError as e:
        log(f"[X] Missing dependency: {e}")
        log("    Run: pip install manifold3d trimesh")
        return False
    
    stl_files = glob.glob(os.path.join(TEMP_DIR, "*.stl"))
    log(f"Found {len(stl_files)} STL files to fuse...")
    
    if not stl_files:
        log("[X] No STLs found to fuse!")
        return False
    
    # Load all meshes into Manifold
    manifolds = []
    load_errors = 0
    
    for i, f in enumerate(stl_files):
        try:
            tm = trimesh.load(f)
            if isinstance(tm, trimesh.Scene):
                for g in tm.geometry.values():
                    if hasattr(g, 'vertices') and hasattr(g, 'faces'):
                        manifolds.append(Manifold(Mesh(
                            vert_properties=np.asarray(g.vertices, dtype=np.float32), 
                            tri_verts=np.asarray(g.faces, dtype=np.uint32)
                        )))
            else:
                if hasattr(tm, 'vertices') and hasattr(tm, 'faces'):
                    manifolds.append(Manifold(Mesh(
                        vert_properties=np.asarray(tm.vertices, dtype=np.float32), 
                        tri_verts=np.asarray(tm.faces, dtype=np.uint32)
                    )))
        except Exception as e:
            load_errors += 1
            if load_errors <= 5:
                log(f"[!] Failed to load {os.path.basename(f)}: {e}")
    
    if load_errors > 5:
        log(f"[!] ... and {load_errors - 5} more load errors")
    
    if not manifolds:
        log("[X] No valid manifolds created!")
        return False
    
    log(f"Loaded {len(manifolds)} manifolds. Starting tree union...")
    
    # THE MAGIC: Tree summation for Boolean Union
    start = time.time()
    level = 0
    while len(manifolds) > 1:
        level += 1
        next_level = []
        for i in range(0, len(manifolds), 2):
            if i + 1 < len(manifolds):
                try:
                    next_level.append(manifolds[i] + manifolds[i+1])
                except Exception as e:
                    log(f"[!] Union failed at level {level}: {e}")
                    next_level.append(manifolds[i])
            else:
                next_level.append(manifolds[i])
        manifolds = next_level
        log(f"   Level {level}: Reduced to {len(manifolds)} chunks")
    
    fused = manifolds[0]
    
    # Export
    mesh_out = fused.to_mesh()
    final_tm = trimesh.Trimesh(
        vertices=mesh_out.vert_properties, 
        faces=mesh_out.tri_verts
    )
    final_tm.export(OUTPUT_FUSED_STL)
    
    elapsed = time.time() - start
    size_mb = os.path.getsize(OUTPUT_FUSED_STL) / (1024 * 1024)
    log(f"[OK] Fused assembly saved: {size_mb:.2f} MB in {elapsed:.1f}s")
    
    return True

# --- PART 3: DOCKER TETWILD ---
def run_docker_safe():
    log("=" * 60)
    log("PHASE 3: TetWild Volume Meshing (Docker)")
    log("=" * 60)
    
    stl_basename = os.path.basename(OUTPUT_FUSED_STL)
    mesh_basename = os.path.basename(FINAL_MESH)
    
    # Docker command with aggressive coarsening to prevent OOM
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{PROJECT_ROOT}:/data",
        "yixinhu/tetwild:latest",
        "--input", f"/data/{stl_basename}",
        "--output", f"/data/{mesh_basename}",
        "--level", "2",  # Coarse quality
    ]
    
    log(f"Command: {' '.join(cmd)}")
    log("")
    log("-" * 40)
    log("TETWILD OUTPUT:")
    log("-" * 40)
    
    start = time.time()
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        for line in process.stdout:
            print(f"    {line.rstrip()}", flush=True)
        
        process.wait()
        log("-" * 40)
        
        elapsed = time.time() - start
        
        if process.returncode == 0:
            log(f"[OK] TetWild completed in {elapsed:.1f}s")
            if os.path.exists(FINAL_MESH):
                size_mb = os.path.getsize(FINAL_MESH) / (1024 * 1024)
                log(f"[OK] Output: {FINAL_MESH} ({size_mb:.2f} MB)")
            return True
        else:
            log(f"[X] TetWild failed with code {process.returncode}")
            if process.returncode == 137:
                log("[!] Exit code 137 = OOM. Increase Docker memory or use coarser settings.")
            return False
            
    except Exception as e:
        log(f"[X] Docker execution error: {e}")
        return False

# --- MAIN EXECUTION FLOW ---
if __name__ == "__main__":
    t0 = time.time()
    
    log("=" * 60)
    log("AUTO STITCH - Parallel Stitch & Shrink Pipeline")
    log("=" * 60)
    log(f"Input: {STEP_FILE}")
    log(f"Cores: {CORES}")
    log("")
    
    # 1. Parallel Mesh
    if not run_parallel_meshing():
        log("[X] Phase 1 failed. Exiting.")
        sys.exit(1)
    
    # 2. Fuse
    if not fuse_shells():
        log("[X] Phase 2 failed. Exiting.")
        sys.exit(1)
    
    # 3. Volume Mesh
    if not run_docker_safe():
        log("[X] Phase 3 failed.")
        sys.exit(1)
    
    total = time.time() - t0
    log("")
    log("=" * 60)
    log(f"PIPELINE COMPLETE! Total time: {total/60:.2f} minutes")
    log("=" * 60)
    
    if os.path.exists(FINAL_MESH):
        log(f"[OK] Final mesh: {FINAL_MESH}")
    
    # Optional: Beep when done (Windows)
    try:
        import winsound
        winsound.Beep(1000, 500)
    except:
        pass
