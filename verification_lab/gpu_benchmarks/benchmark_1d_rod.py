
import sys
import unittest
import numpy as np
from pathlib import Path
import gmsh
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from core.solvers.calculix_adapter import CalculiXAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Benchmark")

class Test1DRodConduction(unittest.TestCase):
    """
    Validation Case: 1D Steady State Conduction
    Solid: Aluminum Rod 10mm x 1mm x 1mm
    BCs: T(z=0) = 300K, T(z=10) = 800K
    Material: k = 150 W/mK (Constant)
    
    Analytical Solution:
    T(z) = 300 + (500 / 10) * z
    Linear gradient from 300 to 800.
    """
    
    def setUp(self):
        self.output_dir = Path("output/benchmark_rod")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mesh_file = self.output_dir / "rod.msh"
        
    def generate_mesh(self):
        """Generate a simple hex mesh for the rod"""
        gmsh.initialize()
        gmsh.model.add("rod")
        
        # Create Box: 1x1x10
        # Align with Z axis for easy analytical check
        gmsh.model.occ.addBox(0, 0, 0, 1.0, 1.0, 10.0)
        gmsh.model.occ.synchronize()
        
        # Mesh settings
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.5)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.5)
        # gmsh.option.setNumber("Mesh.ElementOrder", 2) 
        
        # Generate 3D mesh (Linear first)
        gmsh.model.mesh.generate(3)
        # Convert to Second Order
        gmsh.model.mesh.setOrder(2)
        
        gmsh.write(str(self.mesh_file))
        gmsh.finalize()
        
    def test_linear_gradient(self):
        logger.info("Generating mesh...")
        self.generate_mesh()
        
        logger.info("Running CalculiX...")
        adapter = CalculiXAdapter()
        
        # Run solver
        result = adapter.run(self.mesh_file, self.output_dir, {})
        
        # Verify Results
        node_coords = result['node_coords']
        temperatures = result['temperature']
        
        z = node_coords[:, 2]
        
        # Analytical Solution
        # T(z) = 300 + 50 * z
        T_exact = 300.0 + 50.0 * z
        
        # Calculate Error
        error = np.abs(temperatures - T_exact)
        max_error = np.max(error)
        avg_error = np.mean(error)
        
        logger.info(f"Checking results against Analytical Solution:")
        logger.info(f"  Max Error: {max_error:.4f} K")
        logger.info(f"  Avg Error: {avg_error:.4f} K")
        
        # Pass criteria: < 1% error (5K on 500K range)
        # Numerical error should be very small for linear tet4 on linear problem
        self.assertTrue(max_error < 5.0, f"Max error {max_error} too high!")
        self.assertTrue(avg_error < 1.0, f"Avg error {avg_error} too high!")
        
        logger.info("[PASS] 1D Rod Benchmark Validated")

if __name__ == "__main__":
    unittest.main()
