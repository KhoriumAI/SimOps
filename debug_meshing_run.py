
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from apps.cli.mesh_worker_subprocess import SimpleTetGenerator
from core.config import Config

def test_meshing():
    # Use a known existing file
    input_file = str(project_root / "cad_files" / "Cylinder.step")
    output_file = str(project_root / "test_cylinder_mesh.msh")
    
    print(f"Testing meshing on: {input_file}")
    
    config = Config()
    generator = SimpleTetGenerator(algorithm_name="Delaunay", config=config)
    
    try:
        success = generator.run_meshing_strategy(input_file, output_file)
        print(f"Meshing success: {success}")
    except Exception as e:
        print(f"Meshing failed with exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_meshing()
