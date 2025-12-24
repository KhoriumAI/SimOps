
import gmsh
import math
from pathlib import Path
from typing import Dict
from ..base import BenchmarkCase

class SurfaceFluxSlabCase(BenchmarkCase):
    """
    1D Conduction with Surface Flux.
    Geometry: 10x10x50mm bar aligned with Z-axis.
    BCs: Top (Z=0.05) = 300K, Bottom (Z=0) = Flux (10 kW/m^2).
    Analytical Solution: Linear gradient
    T(z) = T_cold + (q'' / k) * (L - z)
    """
    
    def __init__(self, output_dir: Path):
        super().__init__("SurfaceFluxSlab", output_dir)
        self.L = 0.05 # 50mm
        self.W = 0.01 # 10mm
        self.T_cold = 300.0
        self.q_flux = 50000.0 # 50 kW/m^2 (Expect ~15K rise)
        self.k = 150.0 # W/mK
        
    def generate_geometry(self) -> Path:
        step_file = self.output_dir / "slab_flux.step"
        
        gmsh.initialize()
        gmsh.model.add("SlabFlux")
        
        # Create Box (x, y, z, dx, dy, dz)
        # 10x10x50mm
        gmsh.model.occ.addBox(0, 0, 0, self.W, self.W, self.L)
        
        gmsh.model.occ.synchronize()
        gmsh.write(str(step_file))
        gmsh.finalize()
        
        return step_file
        
    def get_config_overrides(self) -> Dict:
        return {
            "ambient_temperature": self.T_cold, # T_cold applied to "cold" boundary (Z=L)
            "thermal_conductivity": self.k,
            
            # Enable Flux
            "surface_flux_wm2": self.q_flux,
            "transient": False, # Steady State
            "convection_coeff": 0.0, # Adiabatic sides
            
            # Boundary Controls
            "heat_source_at_z_min": True, # Flux at Z_min (Bottom)
            "fix_hot_boundary": False,    # IMPORTANT: Do NOT fix T at bottom
            "fix_cold_boundary": True,    # Fix T at top (Z=L) -> Logic flips if heat_source_at_z_min?
            
            # Clarifying "cold" boundary:
            # If heat_source_at_z_min=True: "Hot" zone is Z_min, "Cold" zone is Z_max.
            # We want Flux at "Hot" (Z=0) and Fixed T at "Cold" (Z=L).
            # So fix_hot=False, fix_cold=True.
        }
        
    def get_analytical_solution(self, x: float, y: float, z: float) -> float:
        # T(z) = T_cold + (q'' / k) * (L - z)
        
        z = max(0.0, min(z, self.L))
        
        return self.T_cold + (self.q_flux / self.k) * (self.L - z)
