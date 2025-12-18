"""
Sharp Edge Classifier
=====================

Detects edges with high dihedral angles (sharp features) and applies
mesh size fields to prevent false curvature-based refinement.

The key insight: curvature-based sizing treats sharp edges as infinite
curvature, causing excessive refinement. By detecting these edges via
dihedral angle and overriding their size, we get proper element sizes.
"""

import gmsh
import math
import numpy as np
from typing import List, Tuple, Optional, Callable


def classify_sharp_edges(
    threshold_deg: float = 30.0,
    log_fn: Optional[Callable[[str], None]] = None,
    verbose: bool = False
) -> List[int]:
    """
    Classify curves as sharp based on dihedral angle between adjacent surfaces.
    
    Args:
        threshold_deg: Minimum angle (degrees) to classify as sharp.
                       Default 30° catches most trailing edges while ignoring
                       gentle curves.
        log_fn: Optional logging function
        verbose: If True, log diagnostic info for all edges
    
    Returns:
        List of curve tags classified as sharp edges.
    """
    def log(msg):
        if log_fn:
            log_fn(msg)
    
    sharp_curves = []
    curves = gmsh.model.getEntities(dim=1)
    
    analyzed = 0
    boundary_edges = 0
    multi_surface_edges = 0
    normal_failures = 0
    
    for dim, curve_tag in curves:
        try:
            # Get adjacent surfaces (upward adjacency: dim 1 → dim 2)
            upward_tags, _ = gmsh.model.getAdjacencies(dim, curve_tag)
            
            # Track edge types
            if len(upward_tags) == 1:
                boundary_edges += 1
                if verbose:
                    log(f"  [Boundary] Curve {curve_tag}: only 1 adjacent surface")
                continue
            elif len(upward_tags) > 2:
                multi_surface_edges += 1
                if verbose:
                    log(f"  [Multi] Curve {curve_tag}: {len(upward_tags)} adjacent surfaces")
                continue
            elif len(upward_tags) == 0:
                continue
            
            analyzed += 1
            surf1_tag, surf2_tag = upward_tags[0], upward_tags[1]
            
            # Get curve midpoint parameter
            param_bounds = gmsh.model.getParametrizationBounds(dim, curve_tag)
            t_mid = (param_bounds[0][0] + param_bounds[0][1]) / 2.0
            
            # Get point on curve at midpoint
            point = gmsh.model.getValue(dim, curve_tag, [t_mid])
            
            # Get normals of both surfaces at this point
            n1 = _get_surface_normal_at_point(surf1_tag, point)
            n2 = _get_surface_normal_at_point(surf2_tag, point)
            
            if n1 is None or n2 is None:
                normal_failures += 1
                if verbose:
                    log(f"  [NormalFail] Curve {curve_tag}: n1={n1 is not None}, n2={n2 is not None}")
                continue
            
            # Calculate dihedral angle
            angle_deg = _angle_between_normals(n1, n2)
            
            if verbose:
                log(f"  [Edge] Curve {curve_tag}: {angle_deg:.1f}° between surfaces {surf1_tag} and {surf2_tag}")
            
            if angle_deg > threshold_deg:
                sharp_curves.append(curve_tag)
                # Classify the edge type for debugging
                if angle_deg > 150:
                    edge_type = "knife-edge"
                elif angle_deg > 90:
                    edge_type = "acute"
                else:
                    edge_type = "sharp"
                log(f"  [Sharp Edge] Curve {curve_tag}: {angle_deg:.1f}° ({edge_type})")
        
        except Exception as e:
            # Some curves may not support parametrization (e.g., points)
            if verbose:
                log(f"  [Error] Curve {curve_tag}: {e}")
            continue
    
    log(f"[CAD Cleaning] Edge analysis: {analyzed} shared, {boundary_edges} boundary, {multi_surface_edges} multi-surface, {normal_failures} normal failures")
    log(f"[CAD Cleaning] Found {len(sharp_curves)} sharp edges (threshold: {threshold_deg}°)")
    
    return sharp_curves


def _get_surface_normal_at_point(
    surface_tag: int,
    point: List[float]
) -> Optional[np.ndarray]:
    """
    Get the surface normal at a given point.
    
    Uses Gmsh's parametric surface evaluation to find the UV coordinates
    of the closest point, then computes the normal from partial derivatives.
    
    Args:
        surface_tag: Surface entity tag
        point: 3D point coordinates [x, y, z]
    
    Returns:
        Unit normal vector as numpy array, or None if evaluation fails.
    """
    try:
        # Get parametric bounds of surface
        bounds = gmsh.model.getParametrizationBounds(2, surface_tag)
        u_min, v_min = bounds[0][0], bounds[0][1]
        u_max, v_max = bounds[1][0], bounds[1][1]
        
        # Sample the surface to find closest UV to our point
        # Use a coarse grid search (5x5) then refine
        best_uv = None
        best_dist = float('inf')
        
        for i in range(5):
            for j in range(5):
                u = u_min + (u_max - u_min) * i / 4.0
                v = v_min + (v_max - v_min) * j / 4.0
                
                try:
                    surf_point = gmsh.model.getValue(2, surface_tag, [u, v])
                    dist = sum((surf_point[k] - point[k])**2 for k in range(3))
                    if dist < best_dist:
                        best_dist = dist
                        best_uv = (u, v)
                except:
                    continue
        
        if best_uv is None:
            return None
        
        # Compute normal using finite differences on partial derivatives
        u, v = best_uv
        eps = 1e-6 * max(u_max - u_min, v_max - v_min, 1e-3)
        
        # dS/du
        p_u0 = np.array(gmsh.model.getValue(2, surface_tag, [u - eps, v]))
        p_u1 = np.array(gmsh.model.getValue(2, surface_tag, [u + eps, v]))
        dS_du = (p_u1 - p_u0) / (2 * eps)
        
        # dS/dv
        p_v0 = np.array(gmsh.model.getValue(2, surface_tag, [u, v - eps]))
        p_v1 = np.array(gmsh.model.getValue(2, surface_tag, [u, v + eps]))
        dS_dv = (p_v1 - p_v0) / (2 * eps)
        
        # Normal = cross product of partials
        normal = np.cross(dS_du, dS_dv)
        norm = np.linalg.norm(normal)
        
        if norm < 1e-10:
            return None
        
        return normal / norm
    
    except Exception as e:
        return None


def _angle_between_normals(n1: np.ndarray, n2: np.ndarray) -> float:
    """
    Calculate dihedral angle (in degrees) for sharp edge detection.
    
    Returns the angle that represents how "sharp" the edge is:
    - 0° = flat (normals parallel, pointing same direction)
    - 90° = right angle edge
    - 180° = knife edge (normals opposite, zero-thickness tip)
    
    For a trailing edge approaching zero thickness, the normals point
    in opposite directions, giving a dihedral angle near 180°.
    """
    cos_angle = np.dot(n1, n2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # Full angle calculation (0° to 180°)
    # cos = 1.0 → 0° (flat, normals same direction)
    # cos = 0.0 → 90° (perpendicular)
    # cos = -1.0 → 180° (knife edge, normals opposite)
    angle_rad = math.acos(cos_angle)
    return math.degrees(angle_rad)


def apply_sharp_edge_fields(
    sharp_curves: List[int],
    max_size: float,
    influence_distance: float = None
) -> int:
    """
    Apply Gmsh mesh size fields to override curvature-based sizing on sharp edges.
    
    Creates a Distance field from the sharp curves, then a Threshold field
    that enforces max_size near those curves.
    
    Args:
        sharp_curves: List of curve tags to apply size override
        max_size: Maximum element size to enforce on sharp edges
        influence_distance: Distance from edge where override applies (default: 5*max_size)
    
    Returns:
        Field ID of the combined field (for further composition)
    """
    if not sharp_curves:
        return -1
    
    if influence_distance is None:
        influence_distance = max_size * 5.0
    
    try:
        # Create Distance field from sharp curves
        dist_field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist_field, "CurvesList", sharp_curves)
        gmsh.model.mesh.field.setNumber(dist_field, "Sampling", 100)
        
        # Create Threshold field: use max_size near sharp edges
        threshold_field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold_field, "InField", dist_field)
        gmsh.model.mesh.field.setNumber(threshold_field, "SizeMin", max_size * 0.8)
        gmsh.model.mesh.field.setNumber(threshold_field, "SizeMax", max_size * 2.0)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", 0)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", influence_distance)
        gmsh.model.mesh.field.setNumber(threshold_field, "StopAtDistMax", 1)
        
        # Get existing background field (if any)
        try:
            existing_bg = gmsh.model.mesh.field.list()
            
            # If there are other fields, combine with Max to avoid over-refinement
            if len(existing_bg) > 1:
                # Find if there's already a background mesh set
                # We'll use Max to take the larger of (existing, our threshold)
                max_field = gmsh.model.mesh.field.add("Max")
                gmsh.model.mesh.field.setNumbers(max_field, "FieldsList", 
                                                  [threshold_field] + [f for f in existing_bg if f != threshold_field and f != max_field])
                gmsh.model.mesh.field.setAsBackgroundMesh(max_field)
                return max_field
            else:
                gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)
                return threshold_field
                
        except:
            gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)
            return threshold_field
    
    except Exception as e:
        print(f"[CAD Cleaning] Warning: Failed to apply sharp edge fields: {e}")
        return -1
