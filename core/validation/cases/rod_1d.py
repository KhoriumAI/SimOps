import gmsh
import math
from pathlib import Path
from typing import Dict
from ..base import BenchmarkCase

class Rod1DCase(BenchmarkCase):
    """
    1D Conduction Benchmark.
    Geometry: 10x10x100mm bar aligned with Z-axis.
    BCs: Top (Z=0.1) = 800K, Bottom (Z=0) = 300K.
    Analytical Solution: Linear gradient T(z) = 300 + 5000 * z
    """
    
    def __init__(self, output_dir: Path):
        super().__init__("Rod1D", output_dir)
        self.L = 0.1 # 100mm
        self.W = 0.01 # 10mm
        self.T_hot = 800.0
        self.T_cold = 300.0
        
    def generate_geometry(self) -> Path:
        step_file = self.output_dir / "rod_1d.step"
        
        gmsh.initialize()
        gmsh.model.add("Rod1D")
        
        # Create Box (x, y, z, dx, dy, dz)
        # Aligned with Z-axis for compatibility with CalculiXAdapter heuristics
        gmsh.model.occ.addBox(0, 0, 0, self.W, self.W, self.L)
        
        gmsh.model.occ.synchronize()
        gmsh.write(str(step_file))
        gmsh.finalize()
        
        return step_file
        
    def get_config_overrides(self) -> Dict:
        return {
            "heat_source_temperature": self.T_hot,
            "ambient_temperature": self.T_cold,
            "thermal_conductivity": 150.0
        }
        
    def get_analytical_solution(self, x: float, y: float, z: float) -> float:
        # Linear interpolation based on Z
        # T(z) = T_cold + (T_hot - T_cold) * (z / L)
        
        # Clamp z to [0, L] to avoid extrapolation errors at boundaries
        z_clamped = max(0.0, min(z, self.L))
        
        return self.T_cold + (self.T_hot - self.T_cold) * (z_clamped / self.L)
