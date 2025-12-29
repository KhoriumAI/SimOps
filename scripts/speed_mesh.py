#!/usr/bin/env python3
"""
SPEED MESH - Optimized Draft Mode Meshing
=========================================
Uses TetWild via Docker with --max-passes 1 for fast (~60s) meshing.
Verbose output for debugging.
"""

import gmsh
import subprocess
import os
import sys
import time

# CONFIGURATION
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
INPUT_STEP = os.path.join(PROJECT_ROOT, "cad_files", "00-CM240411_BIANCA-C00_HEATER_BOAR_0704A1.STEP")
OUTPUT_STL = os.path.join(PROJECT_ROOT, "temp_fast_input.stl")
OUTPUT_MESH = os.path.join(PROJECT_ROOT, "final_fast_mesh.msh")

# Gap size from analysis (smallest gap = 0.0844, recommended = 0.0281)
# Using slightly larger for speed
GAP_SIZE = 0.1  # Coarser for speed

# Timeout in seconds (30 minutes)
DOCKER_TIMEOUT = 30 * 60

def log(msg):
    """Print with timestamp."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def phase_timer(phase_name):
    """Context manager for timing phases."""
    class Timer:
        def __init__(self, name):
            self.name = name
            self.start = None
        def __enter__(self):
            self.start = time.time()
            log(f">>> STARTING: {self.name}")
            return self
        def __exit__(self, *args):
            elapsed = time.time() - self.start
            log(f"<<< FINISHED: {self.name} in {elapsed:.1f}s")
    return Timer(phase_name)

def main():
    total_start = time.time()
    
    log("=" * 60)
    log("SPEED MESH - Draft Mode Pipeline")
    log("=" * 60)
    log(f"Input STEP: {INPUT_STEP}")
    log(f"Output STL: {OUTPUT_STL}")
    log(f"Output Mesh: {OUTPUT_MESH}")
    log(f"Target Edge Length: {GAP_SIZE}")
    log(f"Docker Timeout: {DOCKER_TIMEOUT}s ({DOCKER_TIMEOUT/60:.0f} min)")
    log("=" * 60)
    
    # Check if input exists
    if not os.path.exists(INPUT_STEP):
        log(f"[X] ERROR: Input file not found: {INPUT_STEP}")
        sys.exit(1)
    
    # =========================================================================
    # PHASE 1: Generate STL from STEP (if needed)
    # =========================================================================
    # Check if we can skip STL generation
    skip_stl = False
    if os.path.exists(OUTPUT_STL):
        stl_age = time.time() - os.path.getmtime(OUTPUT_STL)
        stl_size = os.path.getsize(OUTPUT_STL) / (1024 * 1024)
        log(f"[i] Existing STL found: {stl_size:.1f} MB, age: {stl_age/60:.1f} min")
        if stl_age < 3600 and stl_size > 1:  # Less than 1 hour old and > 1MB
            log("[i] Reusing existing STL (less than 1 hour old)")
            skip_stl = True
    
    if not skip_stl:
        with phase_timer("GMSH STL Generation"):
            log("Initializing Gmsh...")
            gmsh.initialize()
            
            # VERBOSE OUTPUT
            gmsh.option.setNumber("General.Terminal", 1)
            gmsh.option.setNumber("General.Verbosity", 5)  # Max verbosity
            
            # Aggressive healing
            gmsh.option.setNumber("Geometry.OCCFixSmallEdges", 1)
            gmsh.option.setNumber("Geometry.OCCFixSmallFaces", 1)
            gmsh.option.setNumber("Geometry.OCCSewFaces", 1)
            
            log(f"Loading STEP file: {os.path.basename(INPUT_STEP)}")
            try:
                gmsh.model.occ.importShapes(INPUT_STEP)
                gmsh.model.occ.synchronize()
            except Exception as e:
                log(f"[X] Error importing STEP: {e}")
                gmsh.finalize()
                sys.exit(1)
            
            # Count entities
            volumes = gmsh.model.getEntities(3)
            surfaces = gmsh.model.getEntities(2)
            log(f"[i] Loaded: {len(volumes)} volumes, {len(surfaces)} surfaces")
            
            # SKIP FRAGMENTATION - It causes infinite hangs on this assembly
            # TetWild will handle non-conformal boundaries
            log("[!] Skipping fragmentation (TetWild handles overlaps)")
            
            # COARSE MESH SETTINGS for speed
            # Larger min/max = fewer triangles = faster TetWild
            gmsh.option.setNumber("Mesh.MeshSizeMin", GAP_SIZE)
            gmsh.option.setNumber("Mesh.MeshSizeMax", GAP_SIZE * 50.0)  # Very coarse max
            gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay (fast)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)  # Disable curvature
            
            # SPEED OPTIMIZATIONS
            gmsh.option.setNumber("Mesh.Optimize", 0)  # No optimization
            gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)
            gmsh.option.setNumber("Mesh.HighOrderOptimize", 0)
            gmsh.option.setNumber("Mesh.MaxRetries", 0)  # Don't retry failed surfaces
            
            log(f"Generating 2D surface mesh (Min: {GAP_SIZE}, Max: {GAP_SIZE*50})...")
            try:
                gmsh.model.mesh.generate(2)
                log("[OK] 2D mesh generation completed")
            except Exception as e:
                log(f"[!] Warning during mesh generation: {e}")
                log("[!] Saving partial mesh anyway...")
            
            log(f"Writing STL: {OUTPUT_STL}")
            gmsh.write(OUTPUT_STL)
            gmsh.finalize()
            
            if os.path.exists(OUTPUT_STL):
                size_mb = os.path.getsize(OUTPUT_STL) / (1024 * 1024)
                log(f"[OK] STL generated: {size_mb:.2f} MB")
            else:
                log("[X] STL generation failed!")
                sys.exit(1)
    
    # =========================================================================
    # PHASE 2: Run TetWild via Docker (Draft Mode)
    # =========================================================================
    with phase_timer("TETWILD Docker Execution"):
        # Calculate epsilon
        eps = GAP_SIZE / 10.0
        
        # Build command
        # Using --input/--output format that works with yixinhu/tetwild
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{PROJECT_ROOT}:/data",
            "yixinhu/tetwild:latest",
            "--input", f"/data/{os.path.basename(OUTPUT_STL)}",
            "--output", f"/data/{os.path.basename(OUTPUT_MESH)}",
            "--level", "2",  # Coarse quality level
        ]
        
        log(f"Docker command: {' '.join(cmd)}")
        log(f"Timeout: {DOCKER_TIMEOUT}s")
        log("")
        log("-" * 40)
        log("TETWILD OUTPUT:")
        log("-" * 40)
        
        try:
            # Run with timeout and stream output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Stream output in real-time
            docker_start = time.time()
            while True:
                # Check timeout
                if time.time() - docker_start > DOCKER_TIMEOUT:
                    log(f"[X] TIMEOUT after {DOCKER_TIMEOUT}s - killing Docker")
                    process.kill()
                    break
                
                # Read line
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    print(f"    {line.rstrip()}", flush=True)
            
            process.wait()
            log("-" * 40)
            
            if process.returncode == 0:
                log("[OK] TetWild completed successfully!")
                if os.path.exists(OUTPUT_MESH):
                    size_mb = os.path.getsize(OUTPUT_MESH) / (1024 * 1024)
                    log(f"[OK] Output mesh: {OUTPUT_MESH} ({size_mb:.2f} MB)")
            else:
                log(f"[X] TetWild failed with code {process.returncode}")
                
        except Exception as e:
            log(f"[X] Docker execution error: {e}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    total_time = time.time() - total_start
    log("")
    log("=" * 60)
    log(f"TOTAL TIME: {total_time:.1f}s ({total_time/60:.1f} min)")
    log("=" * 60)
    
    if os.path.exists(OUTPUT_MESH):
        log("[OK] SUCCESS - Mesh generated!")
        log(f"    Output: {OUTPUT_MESH}")
    else:
        log("[X] FAILED - No output mesh produced")
        sys.exit(1)

if __name__ == "__main__":
    main()
