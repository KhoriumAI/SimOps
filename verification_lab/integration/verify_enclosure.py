"""
Enclosure Generation Verification Script
=========================================

Tests the enclosure geometry generator for external flow CFD.
Validates box, cylinder, and sphere enclosure creation with proper sizing.

Usage:
    python verification_lab/integration/verify_enclosure.py

Output:
    Console report with PASS/FAIL status.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def create_test_stl(output_path: str, size: float = 10.0):
    """Create a simple cube STL for testing."""
    half = size / 2.0
    
    vertices = np.array([
        [-half, -half, -half], [half, -half, -half],
        [half, half, -half], [-half, half, -half],
        [-half, -half, half], [half, -half, half],
        [half, half, half], [-half, half, half],
    ], dtype=np.float32)
    
    # 12 triangles for cube
    faces = np.array([
        [0, 2, 1], [0, 3, 2],  # Back
        [4, 5, 6], [4, 6, 7],  # Front
        [0, 1, 5], [0, 5, 4],  # Bottom
        [3, 6, 2], [3, 7, 6],  # Top
        [0, 4, 7], [0, 7, 3],  # Left
        [1, 2, 6], [1, 6, 5],  # Right
    ], dtype=np.int32)
    
    # Write binary STL
    with open(output_path, 'wb') as f:
        f.write(b'\x00' * 80)  # Header
        f.write(np.array([len(faces)], dtype=np.uint32).tobytes())
        
        for face in faces:
            v0, v1, v2 = vertices[face]
            normal = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal /= norm
            f.write(normal.astype(np.float32).tobytes())
            f.write(v0.tobytes())
            f.write(v1.tobytes())
            f.write(v2.tobytes())
            f.write(np.array([0], dtype=np.uint16).tobytes())
    
    return output_path


def run_verification():
    """Main verification routine."""
    print("=" * 60)
    print("Enclosure Geometry Generator Verification")
    print("=" * 60)
    
    try:
        from core.enclosure_geometry import EnclosureGenerator
    except ImportError as e:
        print(f"[ERROR] Failed to import enclosure_geometry: {e}")
        return False
    
    print("[OK] EnclosureGenerator imported successfully")
    
    # Create test STL
    test_dir = project_root / "temp_geometry"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_stl = str(test_dir / "test_cube.stl")
    
    print("\n[TEST 1] Creating test cube STL...")
    create_test_stl(test_stl, size=10.0)
    print(f"  Created: {test_stl}")
    
    gen = EnclosureGenerator(output_dir=str(test_dir))
    all_passed = True
    
    # === Test Box Enclosure ===
    print("\n[TEST 2] Box Enclosure Generation...")
    try:
        result = gen.create_box_enclosure(test_stl, multiplier=5.0, flow_direction='+X')
        
        # Validate result
        assert result.enclosure_type == 'box', "Wrong enclosure type"
        assert os.path.exists(result.enclosure_stl_path), "Enclosure STL not created"
        
        # Check bounds (10mm cube * 5x multiplier = 50mm box)
        expected_size = 10.0 * 5.0  # 50mm
        actual_size = result.bounds_max - result.bounds_min
        
        print(f"  Enclosure bounds: {result.bounds_min} to {result.bounds_max}")
        print(f"  Actual size: {actual_size}")
        print(f"  Expected size: ~{expected_size} per axis")
        
        # Check boundary patches
        assert 'inlet' in result.boundary_patches, "Missing inlet patch"
        assert 'outlet' in result.boundary_patches, "Missing outlet patch"
        assert 'walls' in result.boundary_patches, "Missing walls patch"
        
        print(f"  Boundary patches: {list(result.boundary_patches.keys())}")
        print("  [PASS] Box enclosure test passed")
        
    except Exception as e:
        print(f"  [FAIL] Box enclosure test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # === Test Cylinder Enclosure ===
    print("\n[TEST 3] Cylinder Enclosure Generation...")
    try:
        result = gen.create_cylindrical_enclosure(
            test_stl, 
            length_multiplier=5.0,
            radius_multiplier=3.0,
            axis='X'
        )
        
        assert result.enclosure_type == 'cylinder', "Wrong enclosure type"
        assert os.path.exists(result.enclosure_stl_path), "Enclosure STL not created"
        
        print(f"  Enclosure bounds: {result.bounds_min} to {result.bounds_max}")
        print(f"  Boundary patches: {list(result.boundary_patches.keys())}")
        print("  [PASS] Cylinder enclosure test passed")
        
    except Exception as e:
        print(f"  [FAIL] Cylinder enclosure test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # === Test Sphere Enclosure ===
    print("\n[TEST 4] Sphere Enclosure Generation...")
    try:
        result = gen.create_spherical_enclosure(test_stl, radius_multiplier=5.0)
        
        assert result.enclosure_type == 'sphere', "Wrong enclosure type"
        assert os.path.exists(result.enclosure_stl_path), "Enclosure STL not created"
        
        # Check it's roughly spherical (bounds should be similar in all directions)
        bounds_size = result.bounds_max - result.bounds_min
        size_ratio = max(bounds_size) / min(bounds_size)
        
        print(f"  Enclosure bounds: {result.bounds_min} to {result.bounds_max}")
        print(f"  Size ratio (should be ~1.0): {size_ratio:.3f}")
        print(f"  Boundary patches: {list(result.boundary_patches.keys())}")
        
        assert size_ratio < 1.1, f"Sphere not spherical enough: ratio={size_ratio}"
        print("  [PASS] Sphere enclosure test passed")
        
    except Exception as e:
        print(f"  [FAIL] Sphere enclosure test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # === Summary ===
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if all_passed:
        print(">>> RESULT: ALL TESTS PASSED <<<")
        return True
    else:
        print(">>> RESULT: SOME TESTS FAILED <<<")
        return False


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
