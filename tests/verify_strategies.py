
import os
import sys
import shutil
import logging
from pathlib import Path
import unittest

# Add root to path so we can import simops_worker
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simops_worker import run_simulation
import gmsh

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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StrategyVerifier")

class TestStrategies(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("./strategy_test_env")
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
        
    def run_strategy(self, name, config_content):
        print(f"\n==================================================")
        print(f" TESTING STRATEGY: {name}")
        print(f"==================================================")
        
        config_path = self.input_dir / f"{name}.json"
        config_path.write_text(config_content)
        
        # Run worker directly
        try:
            result = run_simulation(str(self.cad_path), str(self.output_dir), config_path=str(config_path))
            print(f"RESULT: Success={result.success}, Strategy={result.strategy_name}")
            if not result.success:
                print(f"ERROR: {result.error}")
            return result
        except Exception as e:
            print(f"CRITICAL FAILURE: {e}")
            import traceback
            traceback.print_exc()
            return None

    def test_01_cfd_airflow(self):
        config = """
        {
          "version": "1.0",
          "job_name": "Test_CFD",
          "physics": {
            "simulation_type": "cfd",
            "inlet_velocity": 1.0,
            "material": "Air"
          },
          "meshing": { "mesh_size_multiplier": 2.0 }
        }
        """
        result = self.run_strategy("cfd", config)
        self.assertTrue(result and result.success, "CFD Simulation Failed")
        self.assertTrue((self.output_dir / "Test_CFD" / "results.txt").exists())

    def test_02_thermal_flux(self):
        config = """
        {
          "version": "1.0",
          "job_name": "Test_Thermal",
          "physics": {
            "simulation_type": "thermal",
            "surface_flux_wm2": 1000.0,
            "material": "Aluminum"
          }
        }
        """
        result = self.run_strategy("thermal", config)
        self.assertTrue(result and result.success, "Thermal Simulation Failed")

    def test_03_structural_gravity(self):
        config = """
        {
          "version": "1.0",
          "job_name": "Test_Structural",
          "physics": {
            "simulation_type": "structural",
            "gravity_load_g": 1.0,
            "material": "Steel"
          }
        }
        """
        result = self.run_strategy("structural", config)
        self.assertTrue(result and result.success, "Structural Simulation Failed")

if __name__ == '__main__':
    unittest.main()
