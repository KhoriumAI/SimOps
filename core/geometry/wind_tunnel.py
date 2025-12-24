"""
Virtual Wind Tunnel Generator
=============================

Automates the creation of a fluid domain for external aerodynamics.
Implements the "Goldilocks" enclosure sizing rules:
- Inlet: 3x Length upstream
- Outlet: 15x Length downstream (for wake capture)
- Far Field: 5x max(Width, Height) (blockage < 5%)
- BOI: 1.5x refinement zone

Usage:
    tunnel = VirtualWindTunnel(gmsh_model)
    tunnel.generate_domain(input_volumes)
"""

import math
import gmsh
from typing import List, Tuple, Dict, Any

class VirtualWindTunnel:
    def __init__(self, log_func=print):
        self.log = log_func
        self.enclosure_tag = None
        self.boi_tag = None
        self.fluid_tag = None
        
    def generate_domain(self, object_volumes: List[Tuple[int, int]], flow_axis: str = 'x') -> Dict[str, Any]:
        """
        Wrap the object volumes in a wind tunnel enclosure.
        
        Args:
            object_volumes: List of (dim, tag) tuples for the object
            flow_axis: Direction of flow ('x', 'y', 'z')
            
        Returns:
            Dict containing tags for fluid, wall, inlet, outlet, etc.
        """
        self.log("Generating Virtual Wind Tunnel Domain...")
        
        # 1. Calculate Oriented Bounding Box (OBB)
        # For now, we assume Axis-Aligned BB (AABB) is sufficient if the user
        # oriented the CAD correctly. 
        # TODO: Implement true OBB if needed later.
        
        bbox = self._get_bounding_box(object_volumes)
        L, W, H = bbox['dims']
        center = bbox['center']
        
        self.log(f"  Object Dimensions: L={L:.3f}, W={W:.3f}, H={H:.3f}")
        
        # Determine characteristic length based on flow direction
        # Assuming X-flow for now as per industry standard default
        char_length = L
        max_transverse = max(W, H)
        
        # 2. Calculate Enclosure Dimensions
        # Inlet: 3L upstream
        # Outlet: 15L downstream
        # Walls: 5 * max(W, H) clearance
        
        upstream_dist = 3.0 * char_length
        downstream_dist = 15.0 * char_length
        wall_clearance = 5.0 * max_transverse
        
        # Calculate domain bounds
        # X-min (Inlet), X-max (Outlet)
        # Y-min/max, Z-min/max (Walls)
        
        # Enclosure Box
        # Using AABB center as reference
        x_min_fluid = bbox['min'][0] - upstream_dist
        x_max_fluid = bbox['max'][0] + downstream_dist
        
        y_min_fluid = bbox['min'][1] - wall_clearance
        y_max_fluid = bbox['max'][1] + wall_clearance
        
        z_min_fluid = bbox['min'][2] - wall_clearance
        z_max_fluid = bbox['max'][2] + wall_clearance
        
        domain_L = x_max_fluid - x_min_fluid
        domain_W = y_max_fluid - y_min_fluid
        domain_H = z_max_fluid - z_min_fluid
        
        self.log(f"  Enclosure Dimensions: {domain_L:.2f} x {domain_W:.2f} x {domain_H:.2f}")
        self.log(f"  Blockage Ratio: {(W*H)/(domain_W*domain_H):.2%}")
        
        # 3. Create Enclosure Volume
        # gmsh.model.occ.addBox(x, y, z, dx, dy, dz)
        self.log("  Creating enclosure volume...")
        enclosure = gmsh.model.occ.addBox(
            x_min_fluid, y_min_fluid, z_min_fluid,
            domain_L, domain_W, domain_H
        )
        
        # 4. Create Body of Influence (BOI)
        # 1.5x the object size
        boi_scale = 1.5
        boi_margin_x = (L * boi_scale - L) / 2.0
        boi_margin_y = (W * boi_scale - W) / 2.0
        boi_margin_z = (H * boi_scale - H) / 2.0
        
        # Make BOI extend slightly more downstream for wake
        boi_wake_factor = 2.0 
        
        boi_x_min = bbox['min'][0] - boi_margin_x
        boi_x_max = bbox['max'][0] + boi_margin_x + (L * (boi_wake_factor - 1.0))
        
        boi_y_min = bbox['min'][1] - boi_margin_y
        boi_y_max = bbox['max'][1] + boi_margin_y
        
        boi_z_min = bbox['min'][2] - boi_margin_z
        boi_z_max = bbox['max'][2] + boi_margin_z
        
        self.log(f"  Creating BOI (Body of Influence)...")
        # We don't cut the BOI, we just use it for sizing.
        # But wait, to be a BOI in Gmsh it helps if it's a volume? 
        # Actually, best practice is to define a field inside this box.
        # But creating a volume helps visualize/debug.
        # We will create it but NOT subtract it.
        # Important: Gmsh might mesh inside it if we don't be careful.
        # Usually BOI is a "phantom" structure. 
        # For now, let's just calculate the coords and return them for the mesher to setup Fields.
        
        boi_coords = {
            'x_min': boi_x_min, 'x_max': boi_x_max,
            'y_min': boi_y_min, 'y_max': boi_y_max,
            'z_min': boi_z_min, 'z_max': boi_z_max
        }
        
        # 5. Boolean Subtraction
        # Fluid = Enclosure - Object
        self.log("  Performing Boolean Subtraction (Fluid = Enclosure - Object)...")
        # cut(object, tool) -> returns new object
        # object: enclosure
        # tool: object_volumes
        
        # Ensure we have the list in right format for gmsh [(dim, tag)]
        tool_shapes = object_volumes
        object_shapes = [(3, enclosure)]
        
        # CRITICAL: synchronize before boolean
        gmsh.model.occ.synchronize()
        
        # Result tags: List of (dim, tag)
        # map: mapping between input and output
        res, map = gmsh.model.occ.cut(object_shapes, tool_shapes)
        
        if not res:
            raise RuntimeError("Boolean cut failed! Check for self-intersections or validity.")
            
        self.fluid_tag = res[0][1] # tag of the fluid volume
        self.log(f"  [OK] Fluid domain created (Tag: {self.fluid_tag})")
        
        gmsh.model.occ.synchronize()
        
        # 6. Boundary Identification
        # We need to find the faces of the new fluid volume and tag them.
        # Faces at x_min_fluid -> Inlet
        # Faces at x_max_fluid -> Outlet
        # Faces at y_min/y_max/z_min/z_max -> Far Field (Walls)
        # Faces matching the object -> "object_wall" (No-slip)
        
        boundaries = self._identify_boundaries(
            self.fluid_tag, 
            x_min_fluid, x_max_fluid,
            y_min_fluid, y_max_fluid,
            z_min_fluid, z_max_fluid
        )
        
        return {
            'fluid_volume': self.fluid_tag,
            'boundaries': boundaries,
            'boi_coords': boi_coords,
            'dimensions': {
                'upstream': upstream_dist,
                'downstream': downstream_dist,
                'L_char': char_length
            }
        }

    def _get_bounding_box(self, volumes) -> Dict[str, Any]:
        """Calculate AABB of the input volumes"""
        b_min = [float('inf')] * 3
        b_max = [float('-inf')] * 3
        
        for v_dim, v_tag in volumes:
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(v_dim, v_tag)
            
            b_min[0] = min(b_min[0], xmin)
            b_min[1] = min(b_min[1], ymin)
            b_min[2] = min(b_min[2], zmin)
            
            b_max[0] = max(b_max[0], xmax)
            b_max[1] = max(b_max[1], ymax)
            b_max[2] = max(b_max[2], zmax)
            
        return {
            'min': b_min,
            'max': b_max,
            'dims': [b_max[i] - b_min[i] for i in range(3)],
            'center': [(b_min[i] + b_max[i])/2.0 for i in range(3)]
        }

    def _identify_boundaries(self, volume_tag, xmin, xmax, ymin, ymax, zmin, zmax, tol=1e-5):
        """Identify faces by coordinate matching"""
        # Get all faces of the fluid volume
        # We use gmsh.model.getBoundary logic or getEntitiesInBoundingBox?
        # getBoundary is better for topological correctness.
        
        # Get boundary surfaces (dim=2) of the volume (dim=3)
        # combined=False (don't sum up), oriented=False, recursive=False
        surfaces = gmsh.model.getBoundary([(3, volume_tag)], combined=False, oriented=False, recursive=False)
        
        boundaries = {
            'inlet': [],
            'outlet': [],
            'walls': [],
            'object': []
        }
        
        found_faces = 0
        
        for dim, tag in surfaces:
            # Check bounding box of the surface
            s_xmin, s_ymin, s_zmin, s_xmax, s_ymax, s_zmax = gmsh.model.getBoundingBox(dim, tag)
            
            # Check for planar matches with domain bounds
            
            # Inlet (At X_MIN)
            if abs(s_xmin - xmin) < tol and abs(s_xmax - xmin) < tol:
                boundaries['inlet'].append(tag)
            
            # Outlet (At X_MAX)
            elif abs(s_xmin - xmax) < tol and abs(s_xmax - xmax) < tol:
                boundaries['outlet'].append(tag)
                
            # Far Field Walls (Y_MIN, Y_MAX, Z_MIN, Z_MAX)
            elif (abs(s_ymin - ymin) < tol and abs(s_ymax - ymin) < tol) or \
                 (abs(s_ymin - ymax) < tol and abs(s_ymax - ymax) < tol) or \
                 (abs(s_zmin - zmin) < tol and abs(s_zmax - zmin) < tol) or \
                 (abs(s_zmin - zmax) < tol and abs(s_zmax - zmax) < tol):
                boundaries['walls'].append(tag)
                
            # If not matching any outer box, it must be the object!
            else:
                boundaries['object'].append(tag)
                
            found_faces += 1
            
        self.log(f"  Identified {found_faces} boundary faces:")
        self.log(f"    - Inlet: {len(boundaries['inlet'])}")
        self.log(f"    - Outlet: {len(boundaries['outlet'])}")
        self.log(f"    - Far Field: {len(boundaries['walls'])}")
        self.log(f"    - Object (No-slip): {len(boundaries['object'])}")
        
        return boundaries
