
import gmsh
import sys
import os
import json
import logging
import numpy as np
import unittest
from pathlib import Path

# Add core path
sys.path.insert(0, "/app")
from core.solvers.calculix_adapter import CalculiXAdapter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BenchmarkFin")

class TestFinConvection(unittest.TestCase):
    
    def setUp(self):
        self.output_dir = Path("/output/benchmark_fin")
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
        # Use Tet10 for accuracy (optional, but good to verify integration)
        gmsh.option.setNumber("Mesh.ElementOrder", 2) 
        
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(2)
        gmsh.write(str(self.mesh_file))
        gmsh.finalize()
        
    def test_fin_cooling(self):
        adapter = CalculiXAdapter()
        
        # Parameters
        k = 150.0 
        h = 50.0 
        T_base = 800.0 # Top (Z=L)
        T_tip = 300.0  # Bottom (Z=0)
        T_inf = 293.0
        scale = 1.0 
        
        # NOTE: Adapter Heuristic:
        # Z > 99% -> Hot (800)
        # Z < 1% -> Cold (300)
        # So Z=L is Hot, Z=0 is Cold.
        # This matches T_base and T_tip nicely.
        
        config = {
            "thermal_conductivity": k,
            "heat_source_temperature": T_base,
            "ambient_temperature": T_tip, # Defines T_cold bc
            "convection_coeff": h,
            "ambient_temperature": T_inf, # Wait, logic reuse key?
            # Adapter uses "ambient_temperature" for BOTH T_cold (BC) and T_inf (FILM)?
            # Let's check code from Step 992.
            # t_cold = config.get("ambient_temperature", 300.0) -> For BC
            # T_inf = config.get("ambient_temperature", 293.0) -> For FILM
            # They use the SAME KEY!
            # If I set "ambient_temperature": 293 (T_inf), then T_tip becomes 293.
            # This is acceptable. Fixed Tip = T_inf.
        }
        
        # Explicitly set ambient_temperature to T_inf
        config["ambient_temperature"] = T_inf
        # So T_tip (Fixed BC) = T_inf.
        T_tip = T_inf
        
        # Run
        logger.info("Running Convection Benchmark...")
        result = adapter.run(self.mesh_file, self.output_dir, config)
        
        # Verify
        temps = result['temperature']
        coords = result['node_coords']
        z = coords[:, 2]
        
        # Analytical Solution
        # Theta = T - Tinf
        # Theta_b = T_base - T_inf
        # Theta_tip = T_tip - T_inf = 0 (Since T_tip = T_inf)
        
        # Fin Equation for Prescribed Tip Temp:
        # P = 2*(W+D)
        # Ac = W*D
        # m = sqrt(h*P / k*Ac)
        
        P = 2*(self.W + self.D)
        Ac = self.W * self.D
        m = np.sqrt(h*P / (k*Ac))
        
        # Coordinate x (distance from base).
        # Base is at Z=L. Tip is at Z=0.
        # Let x be distance from Base (Z=L). x = L - z.
        # x goes 0 to L.
        
        # Solution for Theta(x):
        # theta(x) = (theta_L * sinh(mx) + theta_b * sinh(m(L-x))) / sinh(mL) ??
        # Wait, standard variable defs: 
        # Usually x is from base.
        # theta(x) = C1 e^mx + C2 e^-mx.
        # x=0 (Base): theta = theta_b
        # x=L (Tip): theta = theta_tip
        
        # Solving:
        # If theta_tip = 0:
        # theta(x) / theta_b = sinh(m(L-x)) / sinh(mL).
        
        theta_b = T_base - T_inf
        
        # Calculate Error
        max_error = 0
        sum_error = 0
        for i, t_sim in enumerate(temps):
            z_i = z[i]
            x_i = self.L - z_i # Distance from base
            
            # Analytical
            theta_analytical = theta_b * np.sinh(m*(self.L - x_i)) / np.sinh(m*self.L)
            t_analytical = theta_analytical + T_inf
            
            err = abs(t_sim - t_analytical)
            max_error = max(max_error, err)
            sum_error += err
            
        avg_error = sum_error / len(temps)
        logger.info(f"Analysis: m={m:.2f}, L={self.L}")
        logger.info(f"Errors: Max={max_error:.4f}K, Avg={avg_error:.4f}K")
        
        # Assertions
        # Convection numerical error ~1-2K is expected for coarse mesh?
        # Or better?
        # If Film application is wrong, error will be huge (e.g. Linear profile).
        # Linear profile (Conduction only) would be: T = 300 + (500/L)*z.
        # T(mid) = 550.
        # With Convection (h=50):
        # m = sqrt(50 * 0.04 / (150 * 0.0001)) = sqrt(2 / 0.015) = sqrt(133) = 11.5
        # sinh(mL) = sinh(1.15).
        # At mid (x=L/2=0.05): sinh(1.15/2) / sinh(1.15) = 0.61 / 1.45 = 0.42.
        # theta(mid) = 500 * 0.42 = 210.
        # T(mid) = 293 + 210 = 503K.
        # Conduction only T(mid) = 546K (approx).
        # Difference 43K.
        # So we can easily distinguish Convection ON vs OFF.
        
        self.assertLess(max_error, 5.0, f"Max error {max_error}K too high for Convection validation.")
        logger.info("[PASS] Convection Validated")

if __name__ == '__main__':
    unittest.main()
