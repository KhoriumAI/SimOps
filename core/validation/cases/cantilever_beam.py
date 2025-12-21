import gmsh
import math
import numpy as np
from pathlib import Path
from typing import Dict
from ..base import BenchmarkCase
import logging

logger = logging.getLogger(__name__)

class CantileverBeamCase(BenchmarkCase):
    """
    Structural Validation: Cantilever Beam under Tip Load.
    Geometry: 10x10x100mm Steel Bar.
    BCs: Fixed at Z=0.
    Load: 100N Tip Load in -Y direction.
    """
    
    def __init__(self, output_dir: Path):
        super().__init__("CantileverBeam", output_dir)
        # Dimensions (mm)
        self.L = 100.0
        self.W = 10.0
        self.H = 10.0
        
        # Material (Steel)
        self.E = 210000.0 # MPa
        self.nu = 0.3
        self.rho = 7850.0 # kg/m^3
        
        # Load
        # self.g_load = 1000.0 
        self.tip_load_N = 100.0
        
        # Derived
        self.I = (self.W * self.H**3) / 12.0 # Second moment of area
        # For Y-load, bending about X-axis. Ixx = W H^3 / 12. Correct.
        self.A = self.W * self.H
        
    def generate_geometry(self) -> Path:
        step_file = self.output_dir / "beam.step"
        
        gmsh.initialize()
        gmsh.model.add("Beam")
        
        # Beam aligned with Z-axis (0 to L)
        # Fixed at Z=0
        gmsh.model.occ.addBox(-self.W/2, -self.H/2, 0, self.W, self.H, self.L)
        
        gmsh.model.occ.synchronize()
        gmsh.write(str(step_file))
        gmsh.finalize()
        
        return step_file
        
    def get_config_overrides(self) -> Dict:
        return {
            "simulation_type": "structural",
            "gravity_load_g": 0.0,
            "tip_load": [0.0, self.tip_load_N, 0.0], # Load in Y
            "youngs_modulus": self.E,
            "poissons_ratio": self.nu,
            "density": self.rho,
            "unit_scaling": 1.0
        }
        
    def get_analytical_solution(self, x, y, z):
        return 0.0

    def verify(self, result: Dict, tolerance: float = 0.20) -> Dict: # 20% tolerance (mesh sensitive)
        """
        Verify Max Stress and Max Deflection.
        """
        
        # 1. Analytical Values
        # Max Stress at Z=0 (Fixed End)
        M_max = self.tip_load_N * self.L
        sigma_max_theory = (M_max * (self.H/2.0)) / self.I
        
        # Max Deflection at Z=L (Free End)
        delta_max_theory = (self.tip_load_N * self.L**3) / (3 * self.E * self.I)
        
        logger.info(f"Analytical Theory (Tip Load {self.tip_load_N} N):")
        logger.info(f"  Max Stress:   {sigma_max_theory:.4f} MPa")
        logger.info(f"  Max Deflection:{delta_max_theory:.6f} mm")
        
        # 2. Simulation Results
        vm = result['von_mises']
        max_vm_sim = np.max(vm)
        
        disp_mag = result['displacement_magnitude']
        max_disp_sim = np.max(disp_mag)
        
        # 3. Comparison
        err_stress = abs(max_vm_sim - sigma_max_theory) / sigma_max_theory
        err_disp = abs(max_disp_sim - delta_max_theory) / delta_max_theory
        
        logger.info(f"Simulation Results:")
        logger.info(f"  Max Stress:   {max_vm_sim:.4f} MPa (Error: {err_stress*100:.2f}%)")
        logger.info(f"  Max Deflection:{max_disp_sim:.6f} mm (Error: {err_disp*100:.2f}%)")
        
        passed = (err_stress <= tolerance) and (err_disp <= tolerance)
        
        return {
            'case': self.name,
            'passed': passed,
            'max_relative_error': max(err_stress, err_disp),
            'metrics': {
                'stress_theory': sigma_max_theory,
                'stress_sim': max_vm_sim,
                'stress_error': err_stress,
                'disp_theory': delta_max_theory,
                'disp_sim': max_disp_sim,
                'disp_error': err_disp
            }
        }
