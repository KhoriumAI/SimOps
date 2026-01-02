"""
CAD Operations Module for Mesh Preparation
==========================================

OCC-based CAD operations for external flow meshing:
- Hollow shell creation (boundary layer regions)
- Boolean operations for enclosure subtraction
- Surface offsetting for thickness-based meshing

Uses gmsh's OCC kernel for robust CAD operations.
"""

import gmsh
import os
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class HollowShellResult:
    """Result of hollow shell operation."""
    output_step_path: str
    outer_surface_tags: List[int]  # Surface tags for outer boundary
    inner_surface_tags: List[int]  # Surface tags for inner boundary
    wall_thickness: float
    success: bool
    message: str


class CADOperations:
    """
    OCC-based CAD operations for CFD mesh preparation.
    
    Provides high-level operations for creating external flow domains
    and boundary layer regions from solid CAD models.
    """
    
    def __init__(self, output_dir: str = None, verbose: bool = True):
        """
        Initialize CAD operations.
        
        Args:
            output_dir: Directory for output files
            verbose: Print progress messages
        """
        if output_dir is None:
            project_root = Path(__file__).parent.parent
            self.output_dir = project_root / "temp_geometry"
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
    
    def _log(self, message: str):
        """Print message if verbose mode is on."""
        if self.verbose:
            print(f"[CADOperations] {message}")
    
    def _init_gmsh(self):
        """Initialize gmsh if not already running."""
        if not gmsh.isInitialized():
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 1 if self.verbose else 0)
    
    def _finalize_gmsh(self):
        """Clean up gmsh."""
        if gmsh.isInitialized():
            gmsh.finalize()
    
    def create_hollow_shell(self, step_path: str, 
                            thickness: float,
                            preserve_outer: bool = True,
                            exclude_faces: List[int] = None) -> HollowShellResult:
        """
        Creates hollow shell from solid by offsetting surfaces inward.
        
        This operation creates a "boundary layer region" - a shell of specified
        thickness that can be meshed with prism layers for CFD.
        
        Technical approach:
        1. Load STEP file
        2. Use OCC's addThickSolid to create hollow shell
        3. Export modified geometry
        
        Args:
            step_path: Input STEP file path
            thickness: Wall thickness in mm (positive = inward offset)
            preserve_outer: If True, the outer surface remains unchanged
            exclude_faces: Optional list of face tags to NOT offset (e.g., inlets)
            
        Returns:
            HollowShellResult with output path and surface tags
        """
        self._log(f"Creating hollow shell (thickness={thickness}mm)")
        
        if thickness <= 0:
            return HollowShellResult(
                output_step_path=step_path,
                outer_surface_tags=[],
                inner_surface_tags=[],
                wall_thickness=0,
                success=False,
                message="Thickness must be positive"
            )
        
        try:
            self._init_gmsh()
            gmsh.model.add("hollow_shell")
            
            # Load STEP file
            self._log(f"Loading: {step_path}")
            try:
                dimTags = gmsh.model.occ.importShapes(step_path)
            except Exception as e:
                return HollowShellResult(
                    output_step_path="",
                    outer_surface_tags=[],
                    inner_surface_tags=[],
                    wall_thickness=thickness,
                    success=False,
                    message=f"Failed to load STEP: {e}"
                )
            
            gmsh.model.occ.synchronize()
            
            # Get all volumes
            volumes = gmsh.model.getEntities(dim=3)
            if not volumes:
                return HollowShellResult(
                    output_step_path="",
                    outer_surface_tags=[],
                    inner_surface_tags=[],
                    wall_thickness=thickness,
                    success=False,
                    message="No solid volumes found in STEP file"
                )
            
            self._log(f"Found {len(volumes)} volume(s)")
            
            # Get all surfaces before operation (these become "outer" surfaces)
            original_surfaces = [tag for dim, tag in gmsh.model.getEntities(dim=2)]
            self._log(f"Original surfaces: {len(original_surfaces)}")
            
            # Determine which faces to offset
            if exclude_faces is None:
                exclude_faces = []
            
            offset_faces = [(2, tag) for tag in original_surfaces 
                           if tag not in exclude_faces]
            
            # Create thick solid (hollow shell)
            # addThickSolid(volumeTag, excludeFaceTags, offset, ...)
            # Negative offset = inward (creates hollow interior)
            new_volumes = []
            
            for dim, vol_tag in volumes:
                try:
                    # Get faces to exclude from thickening
                    vol_surfaces = gmsh.model.getBoundary([(dim, vol_tag)], oriented=False)
                    faces_to_exclude = [(2, tag) for _, tag in vol_surfaces 
                                       if tag in exclude_faces]
                    
                    # Create thick solid
                    # Note: offset is positive for thickening inward
                    result = gmsh.model.occ.addThickSolid(
                        vol_tag, 
                        [tag for _, tag in faces_to_exclude] if faces_to_exclude else [],
                        -thickness  # Negative = hollow inward
                    )
                    new_volumes.append(result)
                    self._log(f"Volume {vol_tag} -> Hollow shell created")
                    
                except Exception as e:
                    self._log(f"WARNING: addThickSolid failed for volume {vol_tag}: {e}")
                    self._log("Falling back to boolean subtraction method...")
                    
                    # Fallback: create offset surface and boolean subtract
                    try:
                        new_vol = self._hollow_via_offset(vol_tag, thickness)
                        if new_vol:
                            new_volumes.append(new_vol)
                    except Exception as e2:
                        self._log(f"Fallback also failed: {e2}")
                        continue
            
            if not new_volumes:
                return HollowShellResult(
                    output_step_path="",
                    outer_surface_tags=[],
                    inner_surface_tags=[],
                    wall_thickness=thickness,
                    success=False,
                    message="Failed to create hollow shell for any volume"
                )
            
            gmsh.model.occ.synchronize()
            
            # Get new surfaces (after operation)
            all_surfaces_after = [tag for dim, tag in gmsh.model.getEntities(dim=2)]
            
            # Surfaces that existed before are "outer", new ones are "inner"
            outer_surface_tags = [t for t in original_surfaces if t in all_surfaces_after]
            inner_surface_tags = [t for t in all_surfaces_after if t not in original_surfaces]
            
            self._log(f"Outer surfaces: {len(outer_surface_tags)}")
            self._log(f"Inner surfaces: {len(inner_surface_tags)}")
            
            # Export
            base_name = Path(step_path).stem
            output_path = str(self.output_dir / f"{base_name}_hollow_{thickness}mm.step")
            
            gmsh.write(output_path)
            self._log(f"Saved to: {output_path}")
            
            return HollowShellResult(
                output_step_path=output_path,
                outer_surface_tags=outer_surface_tags,
                inner_surface_tags=inner_surface_tags,
                wall_thickness=thickness,
                success=True,
                message="Hollow shell created successfully"
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return HollowShellResult(
                output_step_path="",
                outer_surface_tags=[],
                inner_surface_tags=[],
                wall_thickness=thickness,
                success=False,
                message=f"Exception: {e}"
            )
        finally:
            self._finalize_gmsh()
    
    def _hollow_via_offset(self, vol_tag: int, thickness: float) -> Optional[int]:
        """
        Fallback method: create hollow shell via offset surface + boolean subtract.
        
        Less robust than addThickSolid but works on some geometries where it fails.
        """
        try:
            # Copy the volume
            copied = gmsh.model.occ.copy([(3, vol_tag)])
            copied_tag = copied[0][1]
            
            # Get the bounding box and shrink
            bbox = gmsh.model.occ.getBoundingBox(3, vol_tag)
            xmin, ymin, zmin, xmax, ymax, zmax = bbox
            
            # Scale down uniformly from center
            center = [(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2]
            size = max(xmax - xmin, ymax - ymin, zmax - zmin)
            
            if size <= 0:
                return None
            
            scale_factor = (size - 2 * thickness) / size
            if scale_factor <= 0:
                self._log(f"WARNING: Thickness {thickness} too large for volume size {size}")
                return None
            
            # Scale the copied volume
            gmsh.model.occ.dilate(
                [(3, copied_tag)],
                center[0], center[1], center[2],
                scale_factor, scale_factor, scale_factor
            )
            
            # Boolean subtract: original - scaled copy
            result = gmsh.model.occ.cut(
                [(3, vol_tag)], 
                [(3, copied_tag)],
                removeObject=True,
                removeTool=True
            )
            
            if result[0]:
                return result[0][0][1]
            return None
            
        except Exception as e:
            self._log(f"Offset fallback failed: {e}")
            return None
    
    def subtract_part_from_enclosure(self, enclosure_stl: str,
                                     part_stl: str,
                                     output_name: str = None) -> Tuple[str, bool]:
        """
        Boolean subtract part from enclosure for external flow domain.
        
        This creates the "air volume" around a part - the region to be meshed
        for external aerodynamic analysis.
        
        Args:
            enclosure_stl: Path to enclosure STL (the bounding domain)
            part_stl: Path to part STL (the object to subtract)
            output_name: Optional name for output file
            
        Returns:
            Tuple of (output_path, success)
        """
        self._log("Subtracting part from enclosure...")
        
        try:
            self._init_gmsh()
            gmsh.model.add("external_flow_domain")
            
            # Import enclosure
            self._log(f"Loading enclosure: {enclosure_stl}")
            enc_entities = gmsh.model.occ.importShapes(enclosure_stl)
            
            # Import part
            self._log(f"Loading part: {part_stl}")
            part_entities = gmsh.model.occ.importShapes(part_stl)
            
            gmsh.model.occ.synchronize()
            
            # Get volumes (STL import creates surfaces, need to make volumes)
            enc_volumes = [e for e in enc_entities if e[0] == 3]
            part_volumes = [e for e in part_entities if e[0] == 3]
            
            # If no volumes, try to create from surfaces
            if not enc_volumes:
                enc_surfaces = [e for e in enc_entities if e[0] == 2]
                if enc_surfaces:
                    try:
                        loops = gmsh.model.occ.addSurfaceLoop([e[1] for e in enc_surfaces])
                        enc_vol = gmsh.model.occ.addVolume([loops])
                        enc_volumes = [(3, enc_vol)]
                    except:
                        pass
            
            if not part_volumes:
                part_surfaces = [e for e in part_entities if e[0] == 2]
                if part_surfaces:
                    try:
                        loops = gmsh.model.occ.addSurfaceLoop([e[1] for e in part_surfaces])
                        part_vol = gmsh.model.occ.addVolume([loops])
                        part_volumes = [(3, part_vol)]
                    except:
                        pass
            
            if not enc_volumes or not part_volumes:
                self._log("ERROR: Could not create volumes from STL files")
                return "", False
            
            gmsh.model.occ.synchronize()
            
            # Boolean subtraction
            self._log("Performing boolean subtraction...")
            result = gmsh.model.occ.cut(
                enc_volumes,
                part_volumes,
                removeObject=True,
                removeTool=True
            )
            
            if not result[0]:
                self._log("ERROR: Boolean subtraction produced no result")
                return "", False
            
            gmsh.model.occ.synchronize()
            
            # Export
            if output_name is None:
                base = Path(part_stl).stem
                output_name = f"{base}_external_domain"
            
            output_path = str(self.output_dir / f"{output_name}.step")
            gmsh.write(output_path)
            
            self._log(f"External flow domain saved to: {output_path}")
            return output_path, True
            
        except Exception as e:
            self._log(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            return "", False
        finally:
            self._finalize_gmsh()
    
    def get_volume_properties(self, step_path: str) -> dict:
        """
        Get geometric properties of a STEP file.
        
        Returns:
            Dict with volume, surface_area, bounding_box, num_volumes, num_surfaces
        """
        try:
            self._init_gmsh()
            gmsh.model.add("properties")
            
            gmsh.model.occ.importShapes(step_path)
            gmsh.model.occ.synchronize()
            
            volumes = gmsh.model.getEntities(dim=3)
            surfaces = gmsh.model.getEntities(dim=2)
            
            total_volume = 0
            total_area = 0
            
            for dim, tag in volumes:
                mass = gmsh.model.occ.getMass(dim, tag)
                total_volume += mass
            
            for dim, tag in surfaces:
                mass = gmsh.model.occ.getMass(dim, tag)
                total_area += mass
            
            # Get overall bounding box
            if volumes:
                xmin, ymin, zmin = float('inf'), float('inf'), float('inf')
                xmax, ymax, zmax = float('-inf'), float('-inf'), float('-inf')
                
                for dim, tag in volumes:
                    bbox = gmsh.model.occ.getBoundingBox(dim, tag)
                    xmin = min(xmin, bbox[0])
                    ymin = min(ymin, bbox[1])
                    zmin = min(zmin, bbox[2])
                    xmax = max(xmax, bbox[3])
                    ymax = max(ymax, bbox[4])
                    zmax = max(zmax, bbox[5])
                
                bbox = [xmin, ymin, zmin, xmax, ymax, zmax]
            else:
                bbox = [0, 0, 0, 0, 0, 0]
            
            return {
                'volume': total_volume,
                'surface_area': total_area,
                'bounding_box': bbox,
                'num_volumes': len(volumes),
                'num_surfaces': len(surfaces)
            }
            
        except Exception as e:
            return {'error': str(e)}
        finally:
            self._finalize_gmsh()


# === CLI ===
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python cad_operations.py <step_path> <thickness_mm>")
        print("  Creates hollow shell with specified wall thickness")
        sys.exit(1)
    
    step_path = sys.argv[1]
    thickness = float(sys.argv[2])
    
    ops = CADOperations()
    result = ops.create_hollow_shell(step_path, thickness)
    
    print(f"\nResult:")
    print(f"  Success: {result.success}")
    print(f"  Message: {result.message}")
    if result.success:
        print(f"  Output: {result.output_step_path}")
        print(f"  Outer surfaces: {len(result.outer_surface_tags)}")
        print(f"  Inner surfaces: {len(result.inner_surface_tags)}")
