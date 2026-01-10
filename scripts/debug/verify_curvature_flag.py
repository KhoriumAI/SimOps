
import sys
import os
from pathlib import Path
import gmsh
from unittest.mock import MagicMock

# Mock generic libraries
sys.modules['numpy'] = MagicMock()
sys.modules['requests'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['scipy'] = MagicMock()
sys.modules['scipy.spatial'] = MagicMock()
sys.modules['skimage'] = MagicMock()
sys.modules['skimage.measure'] = MagicMock()
sys.modules['trimesh'] = MagicMock()

# Mock internal heavy modules to prevent them from importing their dependencies
sys.modules['core.winding_resurface'] = MagicMock()
sys.modules['strategies.tetgen_strategy'] = MagicMock()
sys.modules['strategies.pymesh_strategy'] = MagicMock()

# Adjust path to import core modules

# Adjust path to import core modules
sys.path.insert(0, str(Path(__file__).parent))

from core.config import Config
from strategies.exhaustive_strategy import ExhaustiveMeshGenerator

def verify_curvature_flag():
    print("Verifying Curvature Adaptive Flag...")
    
    # 1. Test ENABLED
    print("\n[TEST 1] Enabling curvature_adaptive...")
    config = Config()
    config.mesh_params.curvature_adaptive = True
    
    generator = ExhaustiveMeshGenerator(config)
    generator.initialize_gmsh()
    
    # Mock some geometry info so it doesn't crash on 'diagonal'
    generator.geometry_info = {'diagonal': 100.0}
    
    # Run the strategy logic (we can't run full meshing without a file, 
    # but we can check the option after setting it)
    
    # We'll inject a mock method to avoid running actual mesh generation
    def mock_generate_and_analyze():
        val = gmsh.option.getNumber("Mesh.MeshSizeFromCurvature")
        print(f"  -> Mesh.MeshSizeFromCurvature = {val}")
        return True, {}
        
    generator._generate_and_analyze = mock_generate_and_analyze
    
    print("  Running _try_tet_delaunay_optimized...")
    generator._try_tet_delaunay_optimized()
    if gmsh.option.getNumber("Mesh.MeshSizeFromCurvature") == 1:
        print("  [PASS] Option correctly set to 1")
    else:
        print("  [FAIL] Option is NOT 1")

    # 2. Test DISABLED
    print("\n[TEST 2] Disabling curvature_adaptive...")
    config.mesh_params.curvature_adaptive = False
    
    print("  Running _try_tet_delaunay_optimized...")
    generator._try_tet_delaunay_optimized()
    if gmsh.option.getNumber("Mesh.MeshSizeFromCurvature") == 0:
        print("  [PASS] Option correctly set to 0")
    else:
        print("  [FAIL] Option is NOT 0")
        
    generator.finalize_gmsh()

if __name__ == "__main__":
    verify_curvature_flag()
