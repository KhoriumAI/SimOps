"""
Geometry Analyzer for SimOps
============================

Performs pre-meshing analysis of CAD geometry to determine:
1. Overall dimensions and scale
2. Thin wall detection (heuristic)
3. Suitability for High-Fi CFD meshing

Usage:
    analysis = analyze_cad_geometry("model.step")
    if analysis.is_too_small_for_highfi:
        # Switch to robust strategy
"""

import gmsh
import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from pathlib import Path

@dataclass
class GeometryAnalysis:
    """Results of geometry analysis"""
    
    # Dimensions
    bbox: Tuple[float, float, float, float, float, float]
    diagonal: float
    min_dimension: float      # Smallest extent (dx, dy, or dz)
    max_dimension: float      # Largest extent
    aspect_ratio: float       # Max/min dimension
    volume: float             # Total volume (approx)
    surface_area: float       # Total surface area
    
    # Geometric features
    estimated_wall_thickness: float   # Heuristic estimation
    min_radius_of_curvature: float    # Smallest radius found
    
    # Strategy recommendations
    is_too_small_for_highfi: bool
    recommended_mesh_size: float
    warnings: List[str] = field(default_factory=list)
    
    def __str__(self):
        return (
            f"Geometry Analysis:\n"
            f"  Diagonal: {self.diagonal:.2f} mm\n"
            f"  Min Dim:  {self.min_dimension:.2f} mm\n"
            f"  Est. Wall:{self.estimated_wall_thickness:.2f} mm\n"
            f"  Too Small:{self.is_too_small_for_highfi}\n"
            f"  Warnings: {len(self.warnings)}"
        )


def analyze_cad_geometry(cad_file: str, verbose: bool = False) -> GeometryAnalysis:
    """
    Analyze CAD file to determine geometric properties and strategy suitability.
    
    Args:
        cad_file: Path to CAD file
        verbose: Print debug info
        
    Returns:
        GeometryAnalysis object
    """
    file_path = Path(cad_file)
    if not file_path.exists():
        raise FileNotFoundError(f"CAD file not found: {cad_file}")
        
    # Initialize Gmsh
    if not gmsh.isInitialized():
        gmsh.initialize()
    
    # We use a separate model to avoid interfering with main meshing
    gmsh.model.add("analysis_model")
    
    try:
        # Load CAD
        if verbose:
            print(f"Loading {file_path.name}...")
            
        ext = file_path.suffix.lower()
        if ext in ['.step', '.stp', '.iges', '.igs', '.brep']:
            gmsh.model.occ.importShapes(str(file_path))
            gmsh.model.occ.synchronize()
        else:
            gmsh.merge(str(file_path))
            
        # Get bounding box
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
        dx, dy, dz = xmax-xmin, ymax-ymin, zmax-zmin
        
        diagonal = math.sqrt(dx**2 + dy**2 + dz**2)
        dims = [dx, dy, dz]
        min_dim = min(dims) if any(d > 0 for d in dims) else 0.0
        max_dim = max(dims)
        aspect = max_dim / min_dim if min_dim > 1e-9 else 0.0
        
        # Get entities
        volumes = gmsh.model.getEntities(dim=3)
        surfaces = gmsh.model.getEntities(dim=2)
        curves = gmsh.model.getEntities(dim=1)
        
        # Calculate approximate volume and area
        # Note: Mass properties can be slow on complex models, 
        # but are usually fast enough for single parts
        total_vol = 0.0
        total_area = 0.0
        
        try:
            # We need to mesh superficially or use OCC mass props
            # OCC mass props are instant for STEP files
            
            # Sum up volume of all 3D entities
            for dim, tag in volumes:
                total_vol += gmsh.model.occ.getMass(dim, tag)
            
            # Sum up area of all 2D entities
            for dim, tag in surfaces:
                total_area += gmsh.model.occ.getMass(dim, tag)
                
        except Exception as e:
            # Fallback if OCC mass props fail (e.g. STL)
            # print(f"Mass props failed: {e}")
            pass
            
        # Heuristic for wall thickness: Volume / (0.5 * Surface Area)
        # For a thin plate, V = A_face * t, Total Area ≈ 2 * A_face
        # So t ≈ V / (0.5 * Total Area) = 2V / A
        # This is very rough but useful for "is this a thin sheet?"
        est_thickness = 0.0
        if total_area > 1e-9:
            est_thickness = 2.0 * total_vol / total_area
        else:
            est_thickness = min_dim / 5.0 # Fallback
            
        # --- Analysis Logic ---
        
        warnings = []
        is_too_small = False
        
        # 1. Check for micro-scale geometry (< 1mm)
        if diagonal < 1.0:
            warnings.append(f"Model is extremely small (diagonal = {diagonal:.3f} mm). Check units.")
            is_too_small = True
            
        # 2. Check for thin walls vs HighFi mesh size
        # HighFi typically wants ~0.5mm base size. If wall is < 1.0mm, we can't fit enough layers.
        if est_thickness < 1.0 and est_thickness > 0:
            warnings.append(f"Thin features detected (est. {est_thickness:.2f} mm). HighFi boundary layers may collapse.")
            # If extremely thin, might flag for robust strategy
            if est_thickness < 0.2:
                is_too_small = True
        
        # 3. Check for extreme aspect ratio (long needles or flat sheets)
        if aspect > 50:
            warnings.append(f"High aspect ratio ({aspect:.1f}). Mesh generation may struggle.")
            
        # Recommendation
        rec_size = diagonal / 50.0  # Conservative default
        
        return GeometryAnalysis(
            bbox=(xmin, ymin, zmin, xmax, ymax, zmax),
            diagonal=diagonal,
            min_dimension=min_dim,
            max_dimension=max_dim,
            aspect_ratio=aspect,
            volume=total_vol,
            surface_area=total_area,
            estimated_wall_thickness=est_thickness,
            min_radius_of_curvature=0.0, # TODO: curve analysis
            is_too_small_for_highfi=is_too_small,
            recommended_mesh_size=rec_size,
            warnings=warnings
        )
        
    except Exception as e:
        # Fallback if analysis fails - don't crash the pipeline, just return conservative defaults
        print(f"Geometry analysis failed: {e}")
        return GeometryAnalysis(
            bbox=(0,0,0,0,0,0), diagonal=0, min_dimension=0, max_dimension=0, aspect_ratio=0,
            volume=0, surface_area=0, estimated_wall_thickness=0, min_radius_of_curvature=0,
            is_too_small_for_highfi=False, recommended_mesh_size=1.0, warnings=[f"Analysis failed: {e}"]
        )
        
    finally:
        # Switch back to default model to avoid side effects
        try:
            gmsh.model.setCurrent("default")
            gmsh.model.remove()
        except:
            pass
