#!/usr/bin/env python3
"""
MASTER MESHING SCRIPT
=====================
One-click robust meshing for complex assemblies.

Pipeline:
1. Generate STL from STEP file (Gmsh)
2. Detect smallest gap via raycasting (trimesh)
3. Run TetWild with optimal -l parameter (Docker)

Usage:
    python mesh_master.py "path/to/your.step"
"""

import gmsh
import subprocess
import sys
import os
import numpy as np

# =========================================================================
# STEP 1: Generate STL from STEP
# =========================================================================
def generate_stl(step_file, stl_file="dirty_assembly.stl"):
    """Generate a surface mesh (STL) from a STEP file using Gmsh."""
    print(f"\n{'='*60}")
    print(f"STEP 1: Generating STL from {os.path.basename(step_file)}")
    print(f"{'='*60}")
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    
    # Enable aggressive healing
    gmsh.option.setNumber("Geometry.OCCFixSmallEdges", 1)
    gmsh.option.setNumber("Geometry.OCCFixSmallFaces", 1)
    gmsh.option.setNumber("Geometry.OCCSewFaces", 1)
    
    try:
        gmsh.model.occ.importShapes(step_file)
        gmsh.model.occ.synchronize()
    except Exception as e:
        print(f"Error loading file: {e}")
        gmsh.finalize()
        return None
    
    # NO fragmentation - let TetWild handle topology
    print("Skipping fragmentation (TetWild handles non-conformal boundaries)...")
    
    # Force Feed settings to avoid hanging
    gmsh.option.setNumber("Mesh.Optimize", 0)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)
    gmsh.option.setNumber("Mesh.HighOrderOptimize", 0)
    gmsh.option.setNumber("Mesh.MaxRetries", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 5.0)
    
    print("Generating 2D surface mesh...")
    try:
        gmsh.model.mesh.generate(2)
        print("2D mesh completed successfully.")
    except Exception as e:
        print(f"Warning: {e}")
        print("Saving partial mesh anyway...")
    
    print(f"Writing to {stl_file}...")
    gmsh.write(stl_file)
    gmsh.finalize()
    
    if os.path.exists(stl_file):
        size_mb = os.path.getsize(stl_file) / (1024 * 1024)
        print(f"[OK] STL generated: {stl_file} ({size_mb:.2f} MB)")
        return stl_file
    return None


# =========================================================================
# STEP 2: Detect Smallest Gap via Raycasting
# =========================================================================
def find_smallest_gap(stl_path, num_rays=5000):
    """
    Shoot rays through the mesh to detect the smallest meaningful gap.
    Returns the recommended mesh edge length (-l parameter for TetWild).
    """
    print(f"\n{'='*60}")
    print(f"STEP 2: Detecting Gaps in {os.path.basename(stl_path)}")
    print(f"{'='*60}")
    
    try:
        import trimesh
    except ImportError:
        print("[!] trimesh not installed. Using default gap size.")
        print("   Install with: pip install trimesh")
        return 0.3  # Safe default
    
    print(f"Loading mesh for gap analysis...")
    mesh = trimesh.load(stl_path)
    
    # Setup rays from triangle centers, pointing inward
    ray_origins = mesh.triangles_center
    ray_directions = -mesh.face_normals
    
    # Downsample for speed
    if len(ray_origins) > num_rays:
        indices = np.random.choice(len(ray_origins), num_rays, replace=False)
        ray_origins = ray_origins[indices]
        ray_directions = ray_directions[indices]
    
    print(f"Shooting {len(ray_origins)} rays to measure spacing...")
    
    # Raycast
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions
    )
    
    if len(locations) == 0:
        print("[!] No gaps detected. Using default.")
        return 0.3
    
    # Calculate gaps between consecutive hits
    unique_rays = np.unique(index_ray)
    gaps = []
    
    for r in unique_rays:
        mask = (index_ray == r)
        hits = locations[mask]
        dists = np.linalg.norm(hits - ray_origins[r], axis=1)
        dists.sort()
        diffs = np.diff(dists)
        
        # Filter: ignore slivers (<0.05), keep real gaps (0.05-50)
        valid_gaps = diffs[(diffs > 0.05) & (diffs < 50.0)]
        if len(valid_gaps) > 0:
            gaps.extend(valid_gaps)
    
    if not gaps:
        print("[!] Only slivers detected. Using safe coarse size.")
        return 1.0
    
    gaps = np.array(gaps)
    min_gap = np.percentile(gaps, 5)  # 5th percentile for robustness
    median_gap = np.median(gaps)
    
    print(f"\nGAP ANALYSIS RESULTS:")
    print(f"   Median Gap: {median_gap:.4f} units")
    print(f"   Smallest Consistent Gap: {min_gap:.4f} units")
    
    # Recommended: 1/3rd of smallest gap (2-3 elements across)
    rec_res = min_gap / 3.0
    print(f"[OK] Recommended -l: {rec_res:.4f}")
    
    return rec_res


# =========================================================================
# STEP 3: Run TetWild via Docker
# =========================================================================
def run_tetwild(stl_file, output_file="robust_mesh.msh", edge_length=0.3):
    """Run TetWild via Docker with optimized parameters."""
    print(f"\n{'='*60}")
    print(f"STEP 3: Running TetWild (Docker)")
    print(f"{'='*60}")
    
    # Calculate epsilon (1/10th of edge length for tight tolerance)
    eps = edge_length / 10.0
    
    # Get absolute paths for Docker volume mounting
    cwd = os.getcwd()
    stl_basename = os.path.basename(stl_file)
    out_basename = os.path.basename(output_file)
    
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{cwd}:/data",
        "yixinhu/tetwild:latest",
        "--input", f"/data/{stl_basename}",
        "--output", f"/data/{out_basename}",
        "--level", "2",  # Use level instead of -l for this tetwild version
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"\nParameters:")
    print(f"   -l {edge_length:.4f}  (target edge length)")
    print(f"   --eps {eps:.5f}  (envelope tolerance)")
    print(f"   --max-passes 4  (limit optimization)")
    print(f"\nStarting TetWild...")
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True, cwd=cwd)
        if result.returncode == 0:
            print(f"\n[OK] TetWild completed successfully!")
            if os.path.exists(output_file):
                size_mb = os.path.getsize(output_file) / (1024 * 1024)
                print(f"   Output: {output_file} ({size_mb:.2f} MB)")
            return True
        else:
            print(f"[X] TetWild failed with code {result.returncode}")
            return False
    except Exception as e:
        print(f"[X] Error running Docker: {e}")
        return False


# =========================================================================
# MAIN
# =========================================================================
def main():
    # Parse arguments
    if len(sys.argv) < 2:
        step_file = "cad_files/00-CM240411_BIANCA-C00_HEATER_BOAR_0704A1.STEP"
    else:
        step_file = sys.argv[1]
    
    if not os.path.exists(step_file):
        print(f"[X] Error: File not found: {step_file}")
        sys.exit(1)
    
    print(f"\n{'#'*60}")
    print(f"# MASTER MESHING PIPELINE")
    print(f"# Input: {step_file}")
    print(f"{'#'*60}")
    
    # Step 1: Generate STL
    stl_file = generate_stl(step_file)
    if not stl_file:
        print("[X] Failed to generate STL. Exiting.")
        sys.exit(1)
    
    # Step 2: Detect gap size
    edge_length = find_smallest_gap(stl_file)
    
    # Step 3: Run TetWild
    success = run_tetwild(stl_file, "robust_mesh.msh", edge_length)
    
    if success:
        print(f"\n{'#'*60}")
        print(f"# [OK] PIPELINE COMPLETE!")
        print(f"# Output: robust_mesh.msh")
        print(f"{'#'*60}")
    else:
        print(f"\n{'#'*60}")
        print(f"# [X] PIPELINE FAILED")
        print(f"{'#'*60}")
        sys.exit(1)


if __name__ == "__main__":
    main()
