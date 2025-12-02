"""
Winding Number Resurfacing Utility
===================================

Converts CoACD triangle soup output into single watertight surface
using fast winding number calculation + marching cubes.

This eliminates interface merging issues and sliver hex problems.
"""

import numpy as np
from core.fast_winding import compute_fast_winding_grid
from skimage import measure


def resurface_coacd_output(coacd_parts, resolution=64, verbose=True):
    """
    Convert CoACD output (list of overlapping meshes) into single watertight surface.
    
    Args:
        coacd_parts: List of trimesh objects from CoACD
        resolution: Grid resolution for winding number evaluation (default 64)
                   Higher = better detail capture, slower computation
        verbose: Print progress updates
    
    Returns:
        watertight_mesh: Single trimesh object with watertight surface
    """
    import trimesh
    
    if verbose:
        print(f"\n{'='*60}", flush=True)
        print("WINDING NUMBER RESURFACING", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"Input: {len(coacd_parts)} CoACD parts", flush=True)
        print(f"Resolution: {resolution}^3 grid", flush=True)
    
    # 1. FLATTEN: Combine all parts into one giant list of triangles
    # We don't care about connectivity or overlaps here
    all_vertices = []
    all_faces = []
    current_offset = 0
    
    for i, part in enumerate(coacd_parts):
        if verbose:
            print(f"  Part {i+1}: {len(part.vertices)} vertices, {len(part.faces)} faces", flush=True)
        all_vertices.append(part.vertices)
        all_faces.append(part.faces + current_offset)
        current_offset += len(part.vertices)
    
    soup_verts = np.vstack(all_vertices)
    soup_faces = np.vstack(all_faces)
    
    if verbose:
        print(f"Total triangle soup: {len(soup_verts)} vertices, {len(soup_faces)} faces", flush=True)
    
    # 2. DEFINE GRID (The "Resolution" of your resurfacing)
    # Add padding to ensure mesh is fully contained
    padding = 0.1 * (np.max(soup_verts, axis=0) - np.min(soup_verts, axis=0))
    min_box = np.min(soup_verts, axis=0) - padding
    max_box = np.max(soup_verts, axis=0) + padding
    
    if verbose:
        print(f"\nCreating {resolution}x{resolution}x{resolution} query grid...", flush=True)
    
    grid_x, grid_y, grid_z = np.meshgrid(
        np.linspace(min_box[0], max_box[0], resolution),
        np.linspace(min_box[1], max_box[1], resolution),
        np.linspace(min_box[2], max_box[2], resolution),
        indexing='ij'
    )
    query_points = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
    
    if verbose:
        print(f"  Grid contains {len(query_points)} query points", flush=True)
    
    # 3. RUN FAST WINDING NUMBER CALCULATION
    if verbose:
        print(f"\nComputing winding numbers (Barnes-Hut octree)...", flush=True)
    
    winding_values = compute_fast_winding_grid(soup_verts, soup_faces, query_points, verbose=verbose)
    
    # 4. MARCHING CUBES (Extract the single watertight skin)
    # Threshold 0.5 is the standard "solid" cutoff (inside vs outside)
    if verbose:
        print(f"\nRunning Marching Cubes (threshold=0.5)...", flush=True)
    
    verts, faces, normals, values = measure.marching_cubes(
        winding_values.reshape(resolution, resolution, resolution), 
        level=0.5,
        spacing=(max_box - min_box) / (resolution - 1)
    )
    
    # Offset vertices back to world space
    verts += min_box
    
    if verbose:
        print(f"  Extracted surface: {len(verts)} vertices, {len(faces)} faces", flush=True)
    
    # 5. CREATE THE CLEAN MESH
    watertight_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    
    if verbose:
        print(f"\n{'='*60}", flush=True)
        print(f"WATERTIGHT SURFACE CREATED", flush=True)
        print(f"  Is watertight: {watertight_mesh.is_watertight}", flush=True)
        print(f"  Volume: {watertight_mesh.volume:.6f}", flush=True)
        print(f"{'='*60}\n", flush=True)
    
    return watertight_mesh


def test_resurfacing():
    """
    Test the resurfacing on a simple overlapping sphere example.
    Run this with: python -c "from core.winding_resurface import test_resurfacing; test_resurfacing()"
    """
    import trimesh
    
    print("Creating test case: 2 overlapping spheres", flush=True)
    
    # Create two overlapping spheres
    sphere1 = trimesh.creation.icosphere(subdivisions=3)
    sphere1.vertices += [0.5, 0, 0]
    
    sphere2 = trimesh.creation.icosphere(subdivisions=3)
    sphere2.vertices += [-0.5, 0, 0]
    
    coacd_parts = [sphere1, sphere2]
    
    # Resurface
    watertight = resurface_coacd_output(coacd_parts, resolution=32, verbose=True)
    
    # Save result
    watertight.export("test_resurfaced.stl")
    print(f"Saved to test_resurfaced.stl", flush=True)
    
    return watertight


if __name__ == "__main__":
    test_resurfacing()
