"""
Structural Meshing Strategy
===========================

Provides StructuralMeshConfig and StructuralMeshStrategy for use in the worker pipeline.
Aligns with the 'Checks & Balances' fix required.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import gmsh
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class StructuralMeshConfig:
    """Configuration for Structural Mesh Generation"""
    mesh_size_factor: float = 1.0
    second_order: bool = False
    optimize: bool = True
    element_order: int = 1 # Derived from second_order usually, or explicit

    def __post_init__(self):
        if self.second_order:
            self.element_order = 2
        else:
            self.element_order = 1

class StructuralMeshStrategy:
    """
    Generates meshes suitable for Structural Analysis (Tet4/Tet10).
    """
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            logger.info(msg)
            
    def generate_mesh(self, cad_file: str, output_file: str, config: StructuralMeshConfig) -> Tuple[bool, Dict]:
        self._log(f"Structural Meshing: {cad_file} -> {output_file}")
        self._log(f"Config: Size={config.mesh_size_factor}, Order={config.element_order}")
        
        try:
            if not gmsh.isInitialized():
                gmsh.initialize()
            gmsh.clear()
            gmsh.option.setNumber("General.Terminal", 1 if self.verbose else 0)
            
            # Load
            gmsh.model.occ.importShapes(cad_file)
            gmsh.model.occ.synchronize()
            
            # Config
            # Basic sizing
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
            diag = ((xmax-xmin)**2 + (ymax-ymin)**2 + (zmax-zmin)**2)**0.5
            base_size = diag / 20.0 * config.mesh_size_factor
            
            gmsh.option.setNumber("Mesh.MeshSizeMin", base_size * 0.5)
            gmsh.option.setNumber("Mesh.MeshSizeMax", base_size * 1.5)
            
            # Order
            gmsh.option.setNumber("Mesh.ElementOrder", config.element_order)
            if config.element_order == 2:
                gmsh.option.setNumber("Mesh.SecondOrderLinear", 0) # Curved
            
            # Algorithms (HXT for 3D is good)
            # Enable parallel meshing with HXT for speed
            import os
            # Cap at 8 threads to prevent system instability
            num_threads = min(8, max(1, os.cpu_count() or 4))
            gmsh.option.setNumber("General.NumThreads", num_threads)
            gmsh.option.setNumber("Mesh.MaxNumThreads1D", num_threads)
            gmsh.option.setNumber("Mesh.MaxNumThreads2D", num_threads)
            gmsh.option.setNumber("Mesh.MaxNumThreads3D", num_threads)
            gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT (Parallel, Robust) 
            
            if config.optimize:
                gmsh.option.setNumber("Mesh.Optimize", 1)
                gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
                
            # Generate
            gmsh.model.mesh.generate(3)
            
            # Write MSH
            gmsh.write(output_file)
            
            # Write VTK using PyVista (more robust node ordering for VTK)
            vtk_file = str(Path(output_file).with_suffix('.vtk'))
            try:
                import pyvista as pv
                # Reading MSH via PyVista/meshio often fixes orientation/convention issues
                mesh = pv.read(str(output_file))
                mesh.save(vtk_file)
                self._log(f"VTK written via PyVista: {vtk_file}")
            except Exception as ev:
                self._log(f"PyVista VTK export failed: {ev}. Falling back to Gmsh.")
                gmsh.write(vtk_file)
            
            return True, {'vtk_file': vtk_file}
            
        except Exception as e:
            self._log(f"Structural Meshing Failed: {e}")
            return False, {'error': str(e)}
        finally:
            if gmsh.isInitialized():
                gmsh.finalize()
