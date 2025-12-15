"""
Boundary Layer Health Verification Script
==========================================

Tests the GPU mesher's boundary layer quality without GUI.
Generates a simple test shape and validates the mesh transition.

Usage:
    python verify_boundary_layer.py

Output:
    Console report with PASS/FAIL status.
"""

import numpy as np
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def create_test_sphere(center=(0, 0, 0), radius=1.0, subdivisions=3):
    """
    Creates a sphere surface mesh using icosphere subdivision.
    Returns vertices and faces.
    """
    # Start with icosahedron
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    
    verts = np.array([
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1]
    ], dtype=np.float64)
    
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ], dtype=np.int32)
    
    # Normalize vertices to unit sphere
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Subdivide
    for _ in range(subdivisions):
        new_faces = []
        edge_midpoints = {}
        
        for face in faces:
            v0, v1, v2 = face
            
            # Get or create midpoints
            def get_midpoint(i0, i1):
                key = (min(i0, i1), max(i0, i1))
                if key not in edge_midpoints:
                    mid = (verts[i0] + verts[i1]) / 2.0
                    mid = mid / np.linalg.norm(mid)  # Project to sphere
                    edge_midpoints[key] = len(verts)
                    verts_list.append(mid)
                return edge_midpoints[key]
            
            # Need to convert to list for appending
            if not isinstance(verts, list):
                verts_list = list(verts)
            else:
                verts_list = verts
            
            m01 = get_midpoint(v0, v1)
            m12 = get_midpoint(v1, v2)
            m20 = get_midpoint(v2, v0)
            
            new_faces.append([v0, m01, m20])
            new_faces.append([v1, m12, m01])
            new_faces.append([v2, m20, m12])
            new_faces.append([m01, m12, m20])
            
            verts = np.array(verts_list)
        
        faces = np.array(new_faces, dtype=np.int32)
    
    # Scale and translate
    verts = verts * radius + np.array(center)
    
    return verts.astype(np.float64), faces.astype(np.int32)


def create_test_cube(center=(0, 0, 0), size=2.0, triangulate=True):
    """
    Creates a cube surface mesh with triangulated faces.
    Face winding order is CCW when viewed from outside (outward normals).
    """
    half = size / 2.0
    cx, cy, cz = center
    
    verts = np.array([
        [cx - half, cy - half, cz - half],  # 0: ---
        [cx + half, cy - half, cz - half],  # 1: +--
        [cx + half, cy + half, cz - half],  # 2: ++-
        [cx - half, cy + half, cz - half],  # 3: -+-
        [cx - half, cy - half, cz + half],  # 4: --+
        [cx + half, cy - half, cz + half],  # 5: +-+
        [cx + half, cy + half, cz + half],  # 6: +++
        [cx - half, cy + half, cz + half],  # 7: -++
    ], dtype=np.float64)
    
    # 12 triangles (2 per face) - CCW winding when viewed from outside
    faces = np.array([
        # Front face (z = +half) - normal points +z
        [4, 5, 6], [4, 6, 7],
        # Back face (z = -half) - normal points -z
        [0, 3, 2], [0, 2, 1],
        # Right face (x = +half) - normal points +x
        [1, 2, 6], [1, 6, 5],
        # Left face (x = -half) - normal points -x
        [0, 4, 7], [0, 7, 3],
        # Top face (y = +half) - normal points +y
        [3, 7, 6], [3, 6, 2],
        # Bottom face (y = -half) - normal points -y
        [0, 1, 5], [0, 5, 4],
    ], dtype=np.int32)
    
    return verts, faces


def run_verification():
    """Main verification routine."""
    print("=" * 60)
    print("GPU Mesher Boundary Layer Health Verification")
    print("=" * 60)
    
    # Try to import GPU mesher
    try:
        from core.gpu_mesher import (
            gpu_delaunay_fill_and_filter,
            compute_vertex_lfs,
            generate_boundary_layer,
            validate_boundary_layer_health,
            GPU_AVAILABLE
        )
    except ImportError as e:
        print("[ERROR] Failed to import gpu_mesher: {}".format(e))
        return False
    
    if not GPU_AVAILABLE:
        print("[ERROR] GPU mesher not available. Check _gpumesher.pyd installation.")
        return False
    
    print("[OK] GPU mesher imported successfully")
    
    # Create test shape - use cube for simplicity
    print("\n[TEST 1] Creating test cube...")
    surface_verts, surface_faces = create_test_cube(center=(0, 0, 0), size=10.0)
    print("  Surface: {} vertices, {} triangles".format(
        len(surface_verts), len(surface_faces)))
    
    # Compute bounding box
    bbox_min = surface_verts.min(axis=0)
    bbox_max = surface_verts.max(axis=0)
    print("  BBox: {} to {}".format(bbox_min, bbox_max))
    
    # Test LFS computation
    print("\n[TEST 2] Computing Local Feature Size...")
    lfs_field = compute_vertex_lfs(surface_verts, surface_faces)
    print("  LFS Stats: min={:.4f}, avg={:.4f}, max={:.4f}".format(
        np.min(lfs_field), np.mean(lfs_field), np.max(lfs_field)))
    
    # Test boundary layer generation
    print("\n[TEST 3] Generating Boundary Layer...")
    layer_points = generate_boundary_layer(surface_verts, surface_faces, lfs_field)
    print("  Generated {} boundary layer points".format(len(layer_points)))
    
    # Run full pipeline
    print("\n[TEST 4] Running Full GPU Pipeline...")
    try:
        vertices, tetrahedra, surf_faces = gpu_delaunay_fill_and_filter(
            surface_verts, surface_faces, 
            bbox_min, bbox_max,
            resolution=30,
            target_sicn=0.1
        )
        print("  Result: {} vertices, {} tetrahedra, {} surface tris".format(
            len(vertices), len(tetrahedra), len(surf_faces)))
    except Exception as e:
        print("[ERROR] Pipeline failed: {}".format(e))
        import traceback
        traceback.print_exc()
        return False
    
    # Run validation
    print("\n[TEST 5] Running Boundary Layer Validation...")
    validation = validate_boundary_layer_health(vertices, tetrahedra, surf_faces, surface_verts)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Total Boundary Tets: {}".format(validation.get('total_boundary_tets', 0)))
    print("Critical Failures:   {} ({:.1f}%)".format(
        validation.get('critical_fails', 0),
        validation.get('critical_fail_pct', 0)))
    print("Ratio Range:         {:.3f} to {:.3f}".format(
        validation.get('ratio_min', 0),
        validation.get('ratio_max', 0)))
    print("Average Ratio:       {:.3f}".format(validation.get('ratio_avg', 0)))
    
    if validation.get('pass', False):
        print("\n>>> RESULT: PASS <<<")
        print("Boundary layer health is acceptable.")
        return True
    else:
        print("\n>>> RESULT: FAIL <<<")
        print("Boundary layer has too many pancake tetrahedra.")
        print("Review the LFS-aware sampling and boundary layer generation.")
        return False


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
