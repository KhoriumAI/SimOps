"""
Virtual Wind Tunnel Generator
==============================

Creates a fluid domain enclosure around an object for external aerodynamics CFD.
Automatically tags boundaries as:
- BC_Inlet: Upstream inlet face
- BC_Outlet: Downstream outlet face  
- BC_FarField: Side walls (slip condition)
- BC_Wall_Object: Object surface (no-slip)
"""

import gmsh
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class WindTunnelConfig:
    """Configuration for virtual wind tunnel dimensions."""
    
    # Domain sizing relative to object diagonal
    upstream_length: float = 5.0    # L_upstream = diagonal * this
    downstream_length: float = 10.0  # L_downstream = diagonal * this  
    side_clearance: float = 3.0      # Clearance = diagonal * this (each side)
    
    # Flow direction (X-positive by default)
    flow_direction: Tuple[float, float, float] = (1.0, 0.0, 0.0)


class VirtualWindTunnel:
    """
    Creates virtual wind tunnel domain around an object for external CFD.
    
    Usage:
        wt = VirtualWindTunnel()
        wt.create_domain(cad_file, config)
        # Boundaries are now tagged as BC_Inlet, BC_Outlet, BC_FarField, BC_Wall_Object
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.object_bounds = None
        self.domain_bounds = None
        self.L_char = 1.0  # Characteristic length for Reynolds number
        
    def _log(self, msg: str):
        if self.verbose:
            print(msg, flush=True)
            logger.info(msg)
    
    def create_domain(
        self,
        object_volumes: List[int],
        config: Optional[WindTunnelConfig] = None
    ) -> Dict:
        """
        Create a virtual wind tunnel around the existing object geometry.
        
        This expects gmsh to already be initialized with object geometry loaded.
        Returns metadata about the domain.
        
        Args:
            object_volumes: List of volume tags for the object (will be subtracted)
            config: Wind tunnel configuration
            
        Returns:
            Dict with domain info including L_char for Reynolds calculation
        """
        config = config or WindTunnelConfig()
        
        # 1. Get object bounding box
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
        
        dx = xmax - xmin
        dy = ymax - ymin  
        dz = zmax - zmin
        diagonal = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # Characteristic length = max projected dimension in flow direction
        # For X-flow, this is typically max(dy, dz) or dx
        flow_dir = np.array(config.flow_direction)
        flow_dir = flow_dir / np.linalg.norm(flow_dir)
        
        # Project dimensions onto flow direction for L_char
        self.L_char = max(dx, dy, dz)  # Simplified: use max dimension
        
        self.object_bounds = (xmin, ymin, zmin, xmax, ymax, zmax)
        
        self._log(f"  Object Dimensions: L={dx:.2f}, W={dy:.2f}, H={dz:.2f}")
        
        # 2. Calculate domain extent (assuming X-direction flow)
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        center_z = (zmin + zmax) / 2
        
        domain_xmin = xmin - diagonal * config.upstream_length
        domain_xmax = xmax + diagonal * config.downstream_length
        domain_ymin = center_y - diagonal * config.side_clearance
        domain_ymax = center_y + diagonal * config.side_clearance
        domain_zmin = center_z - diagonal * config.side_clearance
        domain_zmax = center_z + diagonal * config.side_clearance
        
        domain_dx = domain_xmax - domain_xmin
        domain_dy = domain_ymax - domain_ymin
        domain_dz = domain_zmax - domain_zmin
        
        self.domain_bounds = (domain_xmin, domain_ymin, domain_zmin, 
                              domain_xmax, domain_ymax, domain_zmax)
        
        self._log(f"  Enclosure Dimensions: {domain_dx:.2f} x {domain_dy:.2f} x {domain_zmax:.2f}")
        
        # 3. Create enclosure box
        box = gmsh.model.occ.addBox(
            domain_xmin, domain_ymin, domain_zmin,
            domain_dx, domain_dy, domain_dz
        )
        
        # 4. Subtract object from domain (boolean cut)
        try:
            # Get all object volumes
            if object_volumes:
                object_dimtags = [(3, v) for v in object_volumes]
                
                # cut(object_to_cut_from, tools_to_cut)
                result = gmsh.model.occ.cut(
                    [(3, box)],
                    object_dimtags,
                    removeObject=True,
                    removeTool=False  # Keep object for wall BC
                )
                
                fluid_volumes = [tag for dim, tag in result[0] if dim == 3]
                self._log(f"  [OK] Fluid domain created with {len(fluid_volumes)} volumes")
            else:
                fluid_volumes = [box]
                
        except Exception as e:
            self._log(f"  [Warning] Boolean cut failed: {e}")
            self._log(f"  [Warning] Using simple enclosure without subtraction")
            fluid_volumes = [box]
        
        gmsh.model.occ.synchronize()
        
        # 5. Tag boundaries
        self._tag_wind_tunnel_boundaries(object_volumes)
        
        return {
            'fluid_volumes': fluid_volumes,
            'object_bounds': self.object_bounds,
            'domain_bounds': self.domain_bounds,
            'L_char': self.L_char,
            'diagonal': diagonal
        }
    
    def _tag_wind_tunnel_boundaries(self, object_volumes: List[int]):
        """
        Tag wind tunnel surfaces as BC_Inlet, BC_Outlet, BC_FarField, BC_Wall_Object.
        """
        # Remove any existing physical groups (start fresh)
        gmsh.model.removePhysicalGroups()
        
        surfaces = gmsh.model.getEntities(dim=2)
        
        inlet_surfaces = []
        outlet_surfaces = []
        farfield_surfaces = []
        wall_surfaces = []
        
        dx, dy, dz = self.domain_bounds[3] - self.domain_bounds[0], \
                     self.domain_bounds[4] - self.domain_bounds[1], \
                     self.domain_bounds[5] - self.domain_bounds[2]
        
        tol = min(dx, dy, dz) * 0.01  # 1% tolerance
        
        for dim, tag in surfaces:
            sxmin, symin, szmin, sxmax, symax, szmax = gmsh.model.getBoundingBox(dim, tag)
            
            # Surface center
            cx = (sxmin + sxmax) / 2
            cy = (symin + symax) / 2
            cz = (szmin + szmax) / 2
            
            # Check if it's on the domain boundary
            on_xmin = abs(sxmin - self.domain_bounds[0]) < tol and abs(sxmax - self.domain_bounds[0]) < tol
            on_xmax = abs(sxmin - self.domain_bounds[3]) < tol and abs(sxmax - self.domain_bounds[3]) < tol
            on_ymin = abs(symin - self.domain_bounds[1]) < tol and abs(symax - self.domain_bounds[1]) < tol
            on_ymax = abs(symin - self.domain_bounds[4]) < tol and abs(symax - self.domain_bounds[4]) < tol
            on_zmin = abs(szmin - self.domain_bounds[2]) < tol and abs(szmax - self.domain_bounds[2]) < tol
            on_zmax = abs(szmin - self.domain_bounds[5]) < tol and abs(szmax - self.domain_bounds[5]) < tol
            
            if on_xmin:
                inlet_surfaces.append(tag)
            elif on_xmax:
                outlet_surfaces.append(tag)
            elif on_ymin or on_ymax or on_zmin or on_zmax:
                farfield_surfaces.append(tag)
            else:
                # Interior surface = object wall  
                wall_surfaces.append(tag)
        
        # Create physical groups
        if inlet_surfaces:
            p = gmsh.model.addPhysicalGroup(2, inlet_surfaces)
            gmsh.model.setPhysicalName(2, p, "BC_Inlet")
            self._log(f"    - Inlet: {len(inlet_surfaces)} face(s)")
            
        if outlet_surfaces:
            p = gmsh.model.addPhysicalGroup(2, outlet_surfaces)
            gmsh.model.setPhysicalName(2, p, "BC_Outlet")
            self._log(f"    - Outlet: {len(outlet_surfaces)} face(s)")
            
        if farfield_surfaces:
            p = gmsh.model.addPhysicalGroup(2, farfield_surfaces)
            gmsh.model.setPhysicalName(2, p, "BC_FarField")
            self._log(f"    - Far Field: {len(farfield_surfaces)} face(s)")
            
        if wall_surfaces:
            p = gmsh.model.addPhysicalGroup(2, wall_surfaces)
            gmsh.model.setPhysicalName(2, p, "BC_Wall_Object")
            self._log(f"    - Object (No-slip): {len(wall_surfaces)} face(s)")
        
        # Tag fluid volume
        volumes = gmsh.model.getEntities(dim=3)
        all_vol_tags = [tag for dim, tag in volumes]
        if all_vol_tags:
            p = gmsh.model.addPhysicalGroup(3, all_vol_tags)
            gmsh.model.setPhysicalName(3, p, "Fluid_Domain")
            self._log(f"    - Fluid Domain: {len(all_vol_tags)} volume(s)")
        
        self._log(f"  Identified {len(surfaces)} boundary faces:")


def create_wind_tunnel_mesh(
    cad_file: str,
    output_file: str,
    config: Optional[WindTunnelConfig] = None,
    verbose: bool = True
) -> Tuple[bool, Dict]:
    """
    Convenience function to create a wind tunnel mesh from a CAD file.
    
    Args:
        cad_file: Path to CAD file (object to analyze)
        output_file: Output mesh path
        config: Wind tunnel configuration
        verbose: Enable logging
        
    Returns:
        (success, metadata_dict)
    """
    config = config or WindTunnelConfig()
    wt = VirtualWindTunnel(verbose=verbose)
    
    try:
        if not gmsh.isInitialized():
            gmsh.initialize()
        gmsh.clear()
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        
        # Load CAD
        path = Path(cad_file)
        ext = path.suffix.lower()
        
        if ext in ['.step', '.stp', '.iges', '.igs', '.brep']:
            gmsh.model.occ.importShapes(str(path))
        else:
            gmsh.merge(str(path))
            
        gmsh.model.occ.synchronize()
        
        # Get object volumes
        volumes = gmsh.model.getEntities(dim=3)
        object_volumes = [tag for dim, tag in volumes]
        
        # Create wind tunnel
        result = wt.create_domain(object_volumes, config)
        
        # Mesh
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.optimize("Netgen")
        
        # Write
        gmsh.write(output_file)
        
        # VTK too
        vtk_file = str(Path(output_file).with_suffix('.vtk'))
        gmsh.write(vtk_file)
        result['vtk_file'] = vtk_file
        
        return True, result
        
    except Exception as e:
        logger.error(f"Wind tunnel mesh generation failed: {e}")
        return False, {'error': str(e)}
        
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()
