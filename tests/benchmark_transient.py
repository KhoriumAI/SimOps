
import gmsh
import sys
import os
import logging
import numpy as np
import unittest
from pathlib import Path

# Add core path
sys.path.insert(0, "/app")
# Import path workaround for local execution
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from core.solvers.calculix_adapter import CalculiXAdapter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BenchmarkTransient")

class TestTransientHeating(unittest.TestCase):
    
    def setUp(self):
        self.output_dir = Path("output/benchmark_transient")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mesh_file = self.output_dir / "fin.msh"
        self.create_fin_mesh()
        
    def create_fin_mesh(self):
        gmsh.initialize()
        gmsh.model.add("fin")
        
        # Dimensions (Meters): Z-aligned Fin
        self.L = 0.1  # Length (Z)
        self.W = 0.01 # Width (X)
        self.D = 0.01 # Depth (Y)
        
        # Box
        gmsh.model.occ.addBox(0,0,0, self.W, self.D, self.L)
        gmsh.model.occ.synchronize()
        
        # Mesh
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.005) 
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.005)
        
        # Use Tet4 for speed in transient check
        gmsh.model.mesh.generate(3)
        gmsh.write(str(self.mesh_file))
        gmsh.finalize()
        
    def test_heating_curve(self):
        adapter = CalculiXAdapter()
        
        # Physics Parameters
        k = 167.0 # Aluminum 6061 approx
        rho = 2700.0
        cp = 896.0
        
        h = 50.0 # Natural/Forced Convection
        T_base = 350.0  # Heat Source (C) -> 350K? Let's stick to K. 800K is very hot.
        # Let's say Chip is 85C = 358K.
        # Amb = 25C = 298K.
        
        T_base = 358.0
        T_inf = 298.0
        
        config = {
            "thermal_conductivity": k,
            "density": rho,
            "specific_heat": cp,
            
            "heat_source_temperature": T_base,
            "ambient_temperature": T_inf,
            "initial_temperature": T_inf,
            
            "convection_coeff": h,
            
            "fix_hot_boundary": True,
            "fix_cold_boundary": False, # Important: Let the tip float!
            
            "transient": True,
            "time_step": 5.0,
            "duration": 300.0 # 5 mins
        }
        
        # Run
        logger.info("Running Transient Heating Simulation...")
        result = adapter.run(self.mesh_file, self.output_dir, config)
        
        # Analysis
        if 'time_series_stats' in result:
            logger.info("Time Series Data:")
            print(f"{'Time(s)':<10} | {'Min(K)':<10} | {'Max(K)':<10} | {'Mean(K)':<10}")
            print("-" * 46)
            for step in result['time_series_stats']:
                print(f"{step['time']:<10.2f} | {step['min']:<10.2f} | {step['max']:<10.2f} | {step['mean']:<10.2f}")
                
            # Check heating
            final_step = result['time_series_stats'][-1]
            first_step = result['time_series_stats'][0]
            
            # Max temp should stay near T_base (358)
            self.assertAlmostEqual(final_step['max'], T_base, delta=5.0)
            
            # Mean temp should rise
            self.assertGreater(final_step['mean'], first_step['mean'])
            
            logger.info(f"[PASS] Heating verified. Final Mean Temp: {final_step['mean']:.2f}K (started at {first_step['mean']:.2f}K)")
        else:
            self.fail("No time series data returned!")

if __name__ == '__main__':
    unittest.main()
