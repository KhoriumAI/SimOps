"""
Test GPU Mesher with Real STEP Files
====================================

Tests the boundary layer quality fix with actual CAD models.

Usage:
    python step_boundary_test.py

Models tested:
    - Cube.step (simple geometry)
    - Cylinder.step (curved surfaces)
    - Loft.step (complex geometry)
"""

import numpy as np
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def load_step_and_mesh(step_file, resolution=30):
    """
    Load a STEP file, generate surface mesh with Gmsh,
    then run GPU mesher with boundary layer quality enforcement.
    """
    import gmsh
    
    print("\n" + "=" * 60)
    print("Testing: {}".format(Path(step_file).name))
    print("=" * 60)
    
    # Step 1: Generate surface mesh with Gmsh
    print("[1] Generating surface mesh with Gmsh...")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("test_model")
    
    try:
        gmsh.model.occ.importShapes(step_file)
        gmsh.model.occ.synchronize()
    except Exception as e:
        gmsh.finalize()
        print("[ERROR] Failed to load STEP file: {}".format(e))
        return None
    
    # Generate 2D surface mesh
    gmsh.model.mesh.generate(2)
    
    # Extract surface mesh
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    surface_verts = np.array(node_coords).reshape(-1, 3)
    
    # Get triangular elements
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2)
    surface_faces = []
    for etype, tags, nodes in zip(elem_types, elem_tags, elem_node_tags):
        if etype == 2:  # Triangle
            faces = np.array(nodes).reshape(-1, 3) - 1  # Convert to 0-indexed
            surface_faces.append(faces)
    
    if not surface_faces:
        gmsh.finalize()
        print("[ERROR] No surface triangles generated")
        return None
    
    surface_faces = np.vstack(surface_faces)
    gmsh.finalize()
    
    print("    Surface: {} vertices, {} triangles".format(
        len(surface_verts), len(surface_faces)))
    
    # Step 2: Compute bounding box
    bbox_min = surface_verts.min(axis=0)
    bbox_max = surface_verts.max(axis=0)
    dims = bbox_max - bbox_min
    print("    BBox: {} to {}".format(bbox_min, bbox_max))
    print("    Dimensions: {}".format(dims))
    
    # Step 3: Run GPU mesher
    print("[2] Running GPU Mesher with boundary layer enforcement...")
    
    try:
        from core.gpu_mesher import (
            gpu_delaunay_fill_and_filter,
            compute_vertex_lfs,
            validate_boundary_layer_health,
            GPU_AVAILABLE
        )
        
        if not GPU_AVAILABLE:
            print("[ERROR] GPU mesher not available")
            return None
        
        # Compute LFS first for logging
        lfs_field = compute_vertex_lfs(surface_verts, surface_faces)
        print("    LFS: min={:.4f}, avg={:.4f}, max={:.4f}".format(
            np.min(lfs_field), np.mean(lfs_field), np.max(lfs_field)))
        
        # Run pipeline
        vertices, tetrahedra, surf_faces = gpu_delaunay_fill_and_filter(
            surface_verts, surface_faces,
            bbox_min, bbox_max,
            resolution=resolution,
            target_sicn=0.1
        )
        
        print("[3] Results:")
        print("    Vertices: {}".format(len(vertices)))
        print("    Tetrahedra: {}".format(len(tetrahedra)))
        print("    Surface Triangles: {}".format(len(surf_faces)))
        
        # Validate
        print("[4] Boundary Layer Validation:")
        validation = validate_boundary_layer_health(
            vertices, tetrahedra, surf_faces, surface_verts
        )
        
        return {
            'file': step_file,
            'vertices': len(vertices),
            'tetrahedra': len(tetrahedra),
            'surface_tris': len(surf_faces),
            'validation': validation
        }
        
    except Exception as e:
        import traceback
        print("[ERROR] GPU mesher failed: {}".format(e))
        traceback.print_exc()
        return None


def run_tests():
    """Run tests on all available STEP files."""
    print("=" * 60)
    print("GPU Mesher STEP File Boundary Layer Test Suite")
    print("=" * 60)
    
    # Find STEP files
    step_files = [
        Path(project_root) / "cad_files" / "Cube.step",
        Path(project_root) / "cad_files" / "Cylinder.step",
        Path(project_root) / "cad_files" / "Loft.step",
    ]
    
    results = []
    
    for step_file in step_files:
        if step_file.exists():
            result = load_step_and_mesh(str(step_file), resolution=30)
            if result:
                results.append(result)
        else:
            print("[SKIP] File not found: {}".format(step_file))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for r in results:
        name = Path(r['file']).name
        v = r['validation']
        passed = v.get('pass', False)
        fails = v.get('critical_fails', 0)
        total = v.get('total_boundary_tets', 0)
        
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        
        print("{}: {} ({} critical fails / {} boundary tets)".format(
            name.ljust(20), status, fails, total
        ))
        print("    Ratio: min={:.3f}, avg={:.3f}, max={:.3f}".format(
            v.get('ratio_min', 0),
            v.get('ratio_avg', 0),
            v.get('ratio_max', 0)
        ))
    
    print("\n" + "=" * 60)
    if all_passed:
        print(">>> ALL TESTS PASSED <<<")
    else:
        print(">>> SOME TESTS FAILED <<<")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
