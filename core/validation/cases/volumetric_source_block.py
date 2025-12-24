
import gmsh
import math
from pathlib import Path
from typing import Dict
from ..base import BenchmarkCase

class VolumetricSourceBlockCase(BenchmarkCase):
    """
    1D Conduction with Volumetric Heat Generation.
    Geometry: 10x10x100mm bar aligned with Z-axis.
    BCs: Top (Z=L) = 300K, Bottom (Z=0) = 300K.
    Source: uniform q''' = 1 MW/m^3.
    
    Analytical Solution: Parabolic
    T(z) = T_surface + (q''' / (2k)) * z * (L - z)
    """
    
    def __init__(self, output_dir: Path):
        super().__init__("VolumetricSourceBlock", output_dir)
        self.L = 0.1 # 100mm
        self.W = 0.01 # 10mm
        self.T_surf = 300.0
        self.q_vol = 1000000.0 # 1 MW/m^3
        self.k = 150.0 # W/mK
        
    def generate_geometry(self) -> Path:
        step_file = self.output_dir / "block_vol.step"
        
        gmsh.initialize()
        gmsh.model.add("BlockVol")
        
        # Create Box (x, y, z, dx, dy, dz)
        # 10x10x100mm
        gmsh.model.occ.addBox(0, 0, 0, self.W, self.W, self.L)
        
        gmsh.model.occ.synchronize()
        gmsh.write(str(step_file))
        gmsh.finalize()
        
        return step_file
        
    def get_config_overrides(self) -> Dict:
        return {
            "ambient_temperature": self.T_surf, # Used for T_cold
            "heat_source_temperature": self.T_surf, # Used for T_hot
            "thermal_conductivity": self.k,
            
            # Enable Volumetric Source
            "volumetric_heat_wm3": self.q_vol,
            "transient": False, # Steady State
            "convection_coeff": 0.0, # Adiabatic sides
            
            # Boundary Controls
            # We want BOTH ends fixed at T=300K.
            "heat_source_at_z_min": True, # Hot group at Z_min
            
            "fix_hot_boundary": True,     # Fix Z_min to T_hot (300K)
            "fix_cold_boundary": True,    # Fix Z_max to T_cold (300K)
        }
        
    def get_analytical_solution(self, x: float, y: float, z: float) -> float:
        # T(z) = T_surf + (q''' / (2k)) * z * (L - z)
        
        z = max(0.0, min(z, self.L))
        
        return self.T_surf + (self.q_vol / (2 * self.k)) * z * (self.L - z)
