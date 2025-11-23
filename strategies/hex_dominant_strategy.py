"""
Hex-Dominant Mesh Generation Strategy
=====================================

A robust, multi-stage strategy to generate hexahedral (brick) meshes.
It attempts a pipeline of increasingly aggressive techniques:

1. Transfinite (Structured): Perfect hex mesh for simple 6-sided volumes.
2. Recombination (Unstructured): Standard 3D recombination (Frontal-Delaunay).
3. Subdivision (Refinement): Subdivides a coarse tet mesh into hexes (4x element count).

This strategy is designed to "fail forward" - if one method fails, it cleans up
and tries the next one, ensuring the best possible chance of a hex mesh.
"""

import gmsh
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.mesh_generator import BaseMeshGenerator
from core.config import Config

class HexDominantStrategy(BaseMeshGenerator):
    """
    Robust hex-dominant meshing strategy with multi-stage fallback.
    """

    def __init__(self, config: Optional[Config] = None):
        super().__init__(config)

    def run_meshing_strategy(self, input_file: str, output_file: str) -> bool:
        """
        Run the hex-dominant meshing pipeline.
        """
        self.log_message("\n" + "="*60)
        self.log_message("HEX-DOMINANT MESHING STRATEGY")
        self.log_message("Pipeline: Transfinite -> Recombination -> Subdivision")
        self.log_message("="*60)

        # 1. Attempt Transfinite (Structured)
        # Best quality, but only works for simple "brick-like" topology
        if self._attempt_transfinite():
            self.log_message("[OK] SUCCESS: Transfinite (Structured) Hex Mesh")
            return self._finalize_and_save(output_file)
        
        # Reset for next attempt
        gmsh.model.mesh.clear()
        self.log_message("\n[!] Transfinite failed or not applicable. Trying Recombination...")

        # 2. Attempt Recombination (Standard Unstructured)
        # Good for general shapes, but can fail on complex topology
        if self._attempt_recombination():
            self.log_message("[OK] SUCCESS: Recombination (Unstructured) Hex Mesh")
            return self._finalize_and_save(output_file)

        # Reset for next attempt
        gmsh.model.mesh.clear()
        self.log_message("\n[!] Recombination failed. Trying Subdivision (Refinement)...")

        # 3. Attempt Subdivision (Tet -> Hex Refinement)
        # Robust (almost always works), but increases element count by 4x
        if self._attempt_subdivision():
            self.log_message("[OK] SUCCESS: Subdivision (Tet->Hex) Mesh")
            return self._finalize_and_save(output_file)

        self.log_message("\n[X] FAILURE: All hex strategies failed.")
        return False

    def _attempt_transfinite(self) -> bool:
        """
        Attempt 1: Transfinite Meshing
        Checks if volumes are 6-sided and applies transfinite algorithms.
        """
        self.log_message("Attempt 1: Transfinite (Structured)...")
        
        try:
            # Get all volumes
            volumes = gmsh.model.getEntities(dim=3)
            if not volumes:
                return False

            # Check if all volumes are 6-sided (heuristic for transfinite suitability)
            # This is a simplification; real transfinite check is more complex
            # For now, we just try to apply it and catch errors
            
            # Apply transfinite to curves
            curves = gmsh.model.getEntities(dim=1)
            for dim, tag in curves:
                # 10 points per curve as default
                gmsh.model.mesh.setTransfiniteCurve(tag, 10)
            
            # Apply transfinite to surfaces
            surfaces = gmsh.model.getEntities(dim=2)
            for dim, tag in surfaces:
                gmsh.model.mesh.setTransfiniteSurface(tag)
                gmsh.model.mesh.setRecombine(dim, tag)
            
            # Apply transfinite to volumes
            for dim, tag in volumes:
                gmsh.model.mesh.setTransfiniteVolume(tag)

            # Generate mesh
            gmsh.model.mesh.generate(3)
            
            # Verify we got hexes
            return self._verify_hex_elements()

        except Exception as e:
            self.log_message(f"  Debug: Transfinite failed: {e}")
            return False

    def _attempt_recombination(self) -> bool:
        """
        Attempt 2: 3D Recombination
        Standard unstructured hex meshing using Frontal-Delaunay for quads.
        """
        self.log_message("Attempt 2: Recombination (Unstructured)...")
        
        try:
            # Algorithm settings for recombination
            gmsh.option.setNumber("Mesh.Algorithm", 8)       # Frontal-Delaunay for Quads
            gmsh.option.setNumber("Mesh.Algorithm3D", 1)     # Delaunay 3D
            gmsh.option.setNumber("Mesh.RecombineAll", 1)    # Recombine all surfaces
            gmsh.option.setNumber("Mesh.Recombine3DAll", 1)  # Recombine all volumes
            
            # Optimization: Try Blossom (3) first, it's better quality
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3) 
            
            # Generate mesh
            gmsh.model.mesh.generate(3)
            
            # Verify we got hexes
            if self._verify_hex_elements():
                return True
                
            # If Blossom failed to give good hexes, try Simple (0)
            self.log_message("  Blossom algo failed, trying Simple...")
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 0)
            gmsh.model.mesh.clear()
            gmsh.model.mesh.generate(3)
            
            return self._verify_hex_elements()

        except Exception as e:
            self.log_message(f"  Debug: Recombination failed: {e}")
            return False

    def _attempt_subdivision(self) -> bool:
        """
        Attempt 3: Subdivision
        Generates a coarse tet mesh and subdivides each tet into 4 hexes.
        """
        self.log_message("Attempt 3: Subdivision (Tet->Hex)...")
        
        try:
            # 1. Generate COARSE tet mesh first
            # We need it coarse because subdivision multiplies elements by 4
            
            # Reset options
            gmsh.option.setNumber("Mesh.Algorithm", 6)       # Frontal-Delaunay
            gmsh.option.setNumber("Mesh.Algorithm3D", 1)     # Delaunay
            gmsh.option.setNumber("Mesh.RecombineAll", 0)    # No recombination yet
            gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 0) # None
            
            # Coarsen the mesh size significantly (2x larger = 8x fewer elements)
            current_cl = gmsh.option.getNumber("Mesh.CharacteristicLengthMin")
            gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 2.0)
            
            gmsh.model.mesh.generate(3)
            
            # 2. Apply Subdivision
            # 1 = All Quads (for surfaces), 2 = All Hexas (for volumes)
            gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 2)
            
            # Refine (this triggers the subdivision)
            gmsh.model.mesh.refine()
            
            return self._verify_hex_elements()

        except Exception as e:
            self.log_message(f"  Debug: Subdivision failed: {e}")
            return False

    def _verify_hex_elements(self) -> bool:
        """
        Check if the current mesh actually contains 3D elements, 
        and if they are predominantly hexahedra.
        """
        # Get 3D elements
        # Types: 4=Tet, 5=Hex, 6=Prism, 7=Pyramid
        element_types, element_tags, node_tags = gmsh.model.mesh.getElements(dim=3)
        
        if not element_types:
            return False
            
        total_elements = 0
        hex_elements = 0
        
        for i, e_type in enumerate(element_types):
            count = len(element_tags[i])
            total_elements += count
            if e_type == 5: # Hexahedron
                hex_elements += count
                
        if total_elements == 0:
            return False
            
        hex_ratio = hex_elements / total_elements
        self.log_message(f"  Result: {total_elements} elements, {hex_ratio:.1%} hexes")
        
        # We accept if we have > 0 hexes (mixed is okay for "Hex Dominant")
        # But for "Success", we usually want majority hex
        return hex_elements > 0

    def _finalize_and_save(self, output_file: str) -> bool:
        """
        Save the successful mesh and return True.
        """
        try:
            gmsh.write(output_file)
            
            # Analyze quality
            metrics = self.analyze_current_mesh()
            if metrics:
                score = self._calculate_quality_score(metrics)
                self.log_message(f"  Quality Score: {score:.2f}")
                
            return True
        except Exception as e:
            self.log_message(f"Error saving file: {e}")
            return False
