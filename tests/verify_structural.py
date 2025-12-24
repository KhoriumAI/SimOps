
import os
import sys
import shutil
import logging
from pathlib import Path
import unittest
import gmsh

# Add root to path so we can import simops_worker
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simops_worker import run_simulation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StructuralVerifier")

def generate_cylinder_step(output_path: str, radius=0.05, height=0.2):
    try:
        gmsh.initialize()
        gmsh.model.add("Cylinder")
        # addCylinder(x, y, z, dx, dy, dz, r)
        gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, height, radius)
        gmsh.model.occ.synchronize()
        gmsh.write(output_path)
    finally:
        gmsh.finalize()

class TestStructural(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("./structural_test_env")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir()
        
        self.input_dir = self.test_dir / "input"
        self.output_dir = self.test_dir / "output"
        self.input_dir.mkdir()
        self.output_dir.mkdir()
        
        # Generate dummy CAD
        self.cad_path = self.input_dir / "Cylinder.step"
        generate_cylinder_step(str(self.cad_path), radius=0.05, height=0.2)
        
    def test_structural_gravity(self):
        print(f"\n==================================================")
        print(f" TESTING STRATEGY: structural_gravity")
        print(f"==================================================")
        
        config_content = """
        {
          "version": "1.0",
          "job_name": "Test_Structural",
          "physics": {
            "simulation_type": "structural",
            "gravity_load_g": 9.81,
            "material": "Steel",
            "youngs_modulus": 200e9,
            "poissons_ratio": 0.3,
            "density": 7850
          },
          "meshing": { "mesh_size_multiplier": 2.0 }
        }
        """
        
        config_path = self.input_dir / "structural.json"
        config_path.write_text(config_content)
        
        # Run worker directly
        try:
            result = run_simulation(str(self.cad_path), str(self.output_dir), config_path=str(config_path))
            print(f"RESULT: Success={result.success}, Strategy={result.strategy_name}")
            
            if result.success:
                 print(f"VTK: {result.vtk_file}")
                 print(f"Report: {result.report_file}")
            else:
                 print(f"ERROR: {result.error}")
                 
            self.assertTrue(result and result.success, "Structural Simulation Failed")
            
        except Exception as e:
            print(f"CRITICAL FAILURE: {e}")
            import traceback
            traceback.print_exc()
            self.fail(f"Exception: {e}")

if __name__ == '__main__':
    unittest.main()
