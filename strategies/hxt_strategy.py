
import gmsh
from core.config import Config
from core.mesh_generator import BaseMeshGenerator
from typing import Dict, Any

class HXTHexDominantGenerator(BaseMeshGenerator):
    """
    High-performance Hex-Dominant meshing using the HXT algorithm (Algorithm 3D = 10).
    This replaces the CoACD + Subdivision pipeline with a direct, parallel approach.
    """
    
    def __init__(self, config: Config = None):
        super().__init__(config)
        
    def run_meshing_strategy(self, input_file: str, output_file: str) -> bool:
        """
        Run HXT hex-dominant meshing.
        """
        try:
            self.log_message("Starting HXT Hex-Dominant Meshing...")
            
            # 1. Load Geometry
            if not gmsh.isInitialized():
                gmsh.initialize()
                
            gmsh.clear()
            gmsh.merge(input_file)
            
            # 2. Set HXT Options
            # Use all available threads
            import multiprocessing
            num_threads = multiprocessing.cpu_count()
            gmsh.option.setNumber("General.NumThreads", num_threads)
            self.log_message(f"Using {num_threads} threads for HXT")
            
            # HXT Algorithm (10)
            gmsh.option.setNumber("Mesh.Algorithm3D", 10)
            
            # Recombination (Tet -> Hex)
            gmsh.option.setNumber("Mesh.Recombine3DAll", 1)
            gmsh.option.setNumber("Mesh.Recombine3DLevel", 1) # Standard recombination
            gmsh.option.setNumber("Mesh.Recombine3DConformity", 1) # Ensure conformity
            
            # 2D Algorithm: Frontal-Delaunay for better surface quality
            gmsh.option.setNumber("Mesh.Algorithm", 6)
            
            # 3. Generate Mesh
            self.log_message("Generating 3D mesh (HXT)...")
            if not self.generate_mesh_internal(3):
                return False
                
            # 4. Optimize
            self.log_message("Optimizing mesh (Netgen)...")
            gmsh.model.mesh.optimize("Netgen")
            
            # 5. Save
            if self.save_mesh(output_file):
                self._finalize_success(output_file)
                return True
                
            return False
            
        except Exception as e:
            self.log_message(f"HXT meshing failed: {e}")
            return False

    def _finalize_success(self, output_file: str):
        """Save and log success"""
        # Analyze quality
        metrics = self.analyze_current_mesh()
        if metrics:
            self.quality_history.append({
                'iteration': self.current_iteration,
                'algorithm': 'HXT',
                'metrics': metrics
            })
            
        self.log_message(f"[OK] HXT Mesh generated successfully: {output_file}")
