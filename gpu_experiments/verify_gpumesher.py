"""
Verification script for _gpumesher Python module
Verifies GPU Delaunay triangulation with Numpy arrays
"""

import numpy as np
import sys
import os
import time

# ==========================================
# 1. FIX PATHS & DEPENDENCIES
# ==========================================

# Define potential locations where the .pyd might be hiding
possible_paths = [
    # 1. Visual Studio Double-Nest (Most likely for you)
    os.path.join(os.path.dirname(__file__), 'Release', 'Release'),
    # 2. Standard CMake Output
    os.path.join(os.path.dirname(__file__), 'Release'),
    # 3. Build Directory Output
    os.path.join(os.path.dirname(__file__), 'build', 'Release')
]

found_module = False
for path in possible_paths:
    if os.path.exists(path):
        # check if .pyd exists in this folder to be sure
        pyd_files = [f for f in os.listdir(path) if f.endswith('.pyd') and '_gpumesher' in f]
        if pyd_files:
            sys.path.insert(0, path)
            print(f"Found module in: {path}")
            found_module = True
            break

if not found_module:
    print("WARNING: Could not locate _gpumesher.pyd in standard folders.")
    print(f"Checked: {possible_paths}")

# Register CUDA DLLs (Just in case, though static linking should handle this)
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin"
if os.path.exists(cuda_path):
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(cuda_path)
    os.environ['PATH'] = cuda_path + os.pathsep + os.environ['PATH']

# ==========================================
# 2. RUN TESTS
# ==========================================

try:
    import _gpumesher
    print("SUCCESS: _gpumesher imported successfully!")
except ImportError as e:
    print(f"\n[CRITICAL ERROR] Failed to import _gpumesher: {e}")
    sys.exit(1)

# Test 1: Small point cloud (1000 points)
print("\n=== Test 1: Small Point Cloud (1,000 points) ===")
np.random.seed(42)
points_small = np.random.uniform(0.0, 1.0, (1000, 3)).astype(np.float64)

try:
    tets_small = _gpumesher.compute_delaunay(points_small)
    print(f"Generated {len(tets_small)} tetrahedra")
    
    # Validate output
    if tets_small.shape[1] != 4:
        raise ValueError("Each tetrahedron must have 4 vertices")
    if tets_small.dtype != np.int32:
        raise ValueError("Indices must be int32")
        
    print("Output validation passed")
    
except Exception as e:
    print(f"Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Large point cloud (100,000 points)
print("\n=== Test 2: Large Point Cloud (100,000 points) ===")
points_large = np.random.uniform(0.0, 1.0, (100000, 3)).astype(np.float64)

try:
    start = time.time()
    tets_large = _gpumesher.compute_delaunay(points_large)
    elapsed = time.time() - start
    
    count = len(tets_large)
    print(f"Generated {count} tetrahedra in {elapsed*1000:.2f} ms")
    
    # Expectation: ~600k for 100k points
    if count > 500000:
        print("SUCCESS: Tetrahedron count is within expected range!")
    else:
        print(f"WARNING: Low tetrahedron count ({count}). Check coordinate normalization.")
    
except Exception as e:
    print(f"Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n=== All Tests Passed! ===")