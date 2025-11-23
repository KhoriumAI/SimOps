"""
Curvature-Adaptive Meshing Module
==================================

Implements intelligent mesh refinement based on surface curvature.
This is a key feature in ANSYS that ensures curved surfaces get fine meshes
while flat surfaces remain coarse, optimizing element count and accuracy.

Key concepts:
- High curvature (tight curves) -> small elements
- Low curvature (flat areas) -> large elements
- Gradual transitions between refinement zones
"""

import gmsh
import math
import numpy as np
from typing import Dict, List, Tuple, Optional


class CurvatureAdaptiveMesher:
    """
    Analyzes geometry curvature and creates adaptive mesh size fields.

    ANSYS-style approach:
    1. Calculate curvature (radius of curvature) for all surfaces
    2. Define size = f(curvature) with user-controllable parameters
    3. Create smooth transitions between zones
    """

    def __init__(self, min_size: float, max_size: float):
        """
        Initialize curvature-adaptive mesher

        Args:
            min_size: Minimum element size (mm) for high-curvature regions
            max_size: Maximum element size (mm) for flat regions
        """
        self.min_size_mm = min_size
        self.max_size_mm = max_size
        self.curvature_data = {}

    def analyze_geometry_curvature(self) -> Dict:
        """
        Analyze all surfaces and calculate curvature metrics

        Returns:
            Dictionary with curvature analysis results
        """
        surfaces = gmsh.model.getEntities(dim=2)

        analysis = {
            'num_surfaces': len(surfaces),
            'surface_curvatures': {},
            'min_radius': float('inf'),
            'max_radius': 0.0,
            'curved_surfaces': [],
            'flat_surfaces': []
        }

        for dim, tag in surfaces:
            # Get surface type and curvature
            surf_type = gmsh.model.getType(dim, tag)

            # Calculate representative curvature
            curvature_info = self._calculate_surface_curvature(dim, tag, surf_type)

            analysis['surface_curvatures'][tag] = curvature_info

            if curvature_info['is_curved']:
                analysis['curved_surfaces'].append(tag)
                if curvature_info['radius'] < analysis['min_radius']:
                    analysis['min_radius'] = curvature_info['radius']
                if curvature_info['radius'] > analysis['max_radius']:
                    analysis['max_radius'] = curvature_info['radius']
            else:
                analysis['flat_surfaces'].append(tag)

        self.curvature_data = analysis
        return analysis

    def _calculate_surface_curvature(self, dim: int, tag: int, surf_type: str) -> Dict:
        """
        Calculate curvature for a single surface

        Args:
            dim: Entity dimension (2 for surface)
            tag: Entity tag
            surf_type: Surface type string from gmsh

        Returns:
            Dictionary with curvature info
        """
        info = {
            'type': surf_type,
            'is_curved': False,
            'radius': float('inf'),
            'curvature': 0.0
        }

        # Handle "Unknown" surfaces by analyzing geometry
        if surf_type == 'Unknown':
            # Check if surface is actually curved by looking at bbox aspect ratio
            bounds = gmsh.model.getBoundingBox(dim, tag)
            dx = bounds[3] - bounds[0]
            dy = bounds[4] - bounds[1]
            dz = bounds[5] - bounds[2]

            # Get surface area
            try:
                mass_props = gmsh.model.occ.getMass(dim, tag)
                area = mass_props  # For surfaces, mass = area
            except:
                area = dx * dy + dy * dz + dx * dz  # Rough estimate

            # Estimate curvature: if surface wraps around, it's curved
            # For a cylinder: one dimension is much larger (height)
            dims = sorted([dx, dy, dz], reverse=True)

            if dims[1] > 1e-6:  # Not a line
                aspect = dims[0] / dims[1]
                perimeter_est = 2 * (dims[0] + dims[1])

                # If aspect ratio suggests cylindrical (one dim much larger)
                if aspect > 3.0:
                    # Likely a cylindrical surface
                    radius_est = min(dims[1], dims[2]) / 2  # Smaller dimensions
                    if radius_est > 1e-6:
                        info['is_curved'] = True
                        info['radius'] = radius_est
                        info['curvature'] = 1.0 / radius_est
                elif aspect < 1.5:
                    # Could be spherical or complex curved surface
                    # Estimate radius from dimensions
                    radius_est = (dims[0] + dims[1]) / 4
                    if radius_est > 1e-6:
                        info['is_curved'] = True
                        info['radius'] = radius_est
                        info['curvature'] = 1.0 / radius_est

        # Simple heuristic based on surface type
        elif surf_type in ['Plane', 'Ruled Surface']:
            # Flat surface
            info['is_curved'] = False
            info['radius'] = float('inf')
            info['curvature'] = 0.0

        elif surf_type in ['Cylinder', 'Cone']:
            # Cylindrical/conical - has curvature
            # Sample the surface to estimate radius
            try:
                # Get bounds
                bounds = gmsh.model.getBoundingBox(dim, tag)
                u_min, v_min = bounds[0], bounds[1]
                u_max, v_max = bounds[3], bounds[4]

                # Sample at parametric center
                u_mid = (u_min + u_max) / 2
                v_mid = (v_min + v_max) / 2

                # Get curvature at this point
                curvatures = gmsh.model.getCurvature(dim, tag, [u_mid, v_mid])

                if curvatures:
                    # Principal curvatures are returned
                    k1, k2 = abs(curvatures[0]), abs(curvatures[1])
                    max_curvature = max(k1, k2)

                    if max_curvature > 1e-6:
                        info['is_curved'] = True
                        info['curvature'] = max_curvature
                        info['radius'] = 1.0 / max_curvature  # meters

            except Exception as e:
                # Fallback: assume moderate curvature
                info['is_curved'] = True
                info['radius'] = 0.05  # 50mm default
                info['curvature'] = 20.0

        elif surf_type in ['Sphere', 'Torus', 'BSpline Surface', 'Bezier Surface']:
            # Highly curved surfaces - sample to find curvature
            try:
                bounds = gmsh.model.getBoundingBox(dim, tag)
                u_min, v_min = bounds[0], bounds[1]
                u_max, v_max = bounds[3], bounds[4]

                # Sample multiple points and take max curvature
                sample_points = [
                    [(u_min + u_max) / 2, (v_min + v_max) / 2],
                    [u_min + 0.25 * (u_max - u_min), v_min + 0.25 * (v_max - v_min)],
                    [u_min + 0.75 * (u_max - u_min), v_min + 0.75 * (v_max - v_min)]
                ]

                max_curvature = 0.0
                for u, v in sample_points:
                    try:
                        curvatures = gmsh.model.getCurvature(dim, tag, [u, v])
                        if curvatures:
                            k1, k2 = abs(curvatures[0]), abs(curvatures[1])
                            max_curvature = max(max_curvature, k1, k2)
                    except:
                        pass

                if max_curvature > 1e-6:
                    info['is_curved'] = True
                    info['curvature'] = max_curvature
                    info['radius'] = 1.0 / max_curvature

            except Exception as e:
                # Fallback
                info['is_curved'] = True
                info['radius'] = 0.02  # 20mm default for complex surfaces
                info['curvature'] = 50.0

        return info

    def create_curvature_adaptive_field(self, elements_per_curve: int = 12,
                                       transition_rate: float = 1.5) -> int:
        """
        Create a gmsh background mesh field based on curvature

        Args:
            elements_per_curve: Target number of elements along a curved edge (like ANSYS)
            transition_rate: How quickly size transitions (1.0=gradual, 2.0=fast)

        Returns:
            Field tag
        """
        if not self.curvature_data:
            self.analyze_geometry_curvature()

        # Strategy: Create a MathEval field that computes size based on distance
        # from curved features

        # First, create distance field to curved surfaces
        curved_surfaces = self.curvature_data['curved_surfaces']

        if not curved_surfaces:
            # No curved surfaces - return uniform field
            field_tag = gmsh.model.mesh.field.add("MathEval")
            gmsh.model.mesh.field.setString(field_tag, "F", str(self.max_size_mm))
            return field_tag

        # Create distance field to curved surfaces
        dist_field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist_field, "SurfacesList", curved_surfaces)

        # Create threshold field: size varies with distance from curves
        # Near curves: min_size
        # Far from curves: max_size
        threshold_field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold_field, "InField", dist_field)

        # Distance thresholds (in meters, will be converted)
        dist_min = self.min_size_mm / 1000.0 * 2  # Start transition at 2x min size
        dist_max = self.max_size_mm / 1000.0 * 3  # Complete transition at 3x max size

        gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", dist_min)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", dist_max)
        gmsh.model.mesh.field.setNumber(threshold_field, "SizeMin", self.min_size_mm)
        gmsh.model.mesh.field.setNumber(threshold_field, "SizeMax", self.max_size_mm)
        gmsh.model.mesh.field.setNumber(threshold_field, "StopAtDistMax", 1)

        return threshold_field

    def create_curvature_sensitive_field_advanced(self,
                                                  curvature_resolution: float = 0.02) -> int:
        """
        Advanced field that directly uses surface curvature values

        Args:
            curvature_resolution: Target resolution as fraction of radius of curvature
                                 (e.g., 0.02 = 50 elements per circle)

        Returns:
            Field tag
        """
        if not self.curvature_data:
            self.analyze_geometry_curvature()

        # Create a curvature-based field
        # For each surface, set size = radius * curvature_resolution

        # Use MathEval with curvature information
        # This is more complex - need to create per-surface fields and combine

        surface_fields = []

        for surf_tag, curv_info in self.curvature_data['surface_curvatures'].items():
            if curv_info['is_curved']:
                # Calculate target size for this surface based on its curvature
                radius_m = curv_info['radius']
                target_size_m = radius_m * curvature_resolution
                target_size_mm = target_size_m * 1000

                # Clamp to user's min/max
                target_size_mm = max(self.min_size_mm, min(target_size_mm, self.max_size_mm))

                # Create restrict field for this surface
                restrict_field = gmsh.model.mesh.field.add("Restrict")

                # Create constant field for this surface
                const_field = gmsh.model.mesh.field.add("MathEval")
                gmsh.model.mesh.field.setString(const_field, "F", str(target_size_mm))

                gmsh.model.mesh.field.setNumber(restrict_field, "InField", const_field)
                gmsh.model.mesh.field.setNumbers(restrict_field, "SurfacesList", [surf_tag])

                surface_fields.append(restrict_field)

        # Combine all fields using Min
        if surface_fields:
            min_field = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", surface_fields)
            return min_field
        else:
            # No curved surfaces, return uniform
            field_tag = gmsh.model.mesh.field.add("MathEval")
            gmsh.model.mesh.field.setString(field_tag, "F", str(self.max_size_mm))
            return field_tag

    def get_curvature_report(self) -> str:
        """Generate a human-readable report of curvature analysis"""
        if not self.curvature_data:
            return "No curvature analysis performed yet"

        report = []
        report.append("=" * 60)
        report.append("CURVATURE ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Total surfaces: {self.curvature_data['num_surfaces']}")
        report.append(f"Curved surfaces: {len(self.curvature_data['curved_surfaces'])}")
        report.append(f"Flat surfaces: {len(self.curvature_data['flat_surfaces'])}")

        if self.curvature_data['min_radius'] < float('inf'):
            report.append(f"\nCurvature range:")
            report.append(f"  Min radius: {self.curvature_data['min_radius']*1000:.2f} mm")
            report.append(f"  Max radius: {self.curvature_data['max_radius']*1000:.2f} mm")

        report.append(f"\nMesh sizing strategy:")
        report.append(f"  Curved regions: {self.min_size_mm:.1f} mm")
        report.append(f"  Flat regions: {self.max_size_mm:.1f} mm")
        report.append(f"  Refinement ratio: {self.max_size_mm/self.min_size_mm:.1f}x")

        report.append("\nPer-surface analysis:")
        for surf_tag, curv_info in sorted(self.curvature_data['surface_curvatures'].items()):
            if curv_info['is_curved']:
                report.append(f"  Surface {surf_tag}: {curv_info['type']}, "
                            f"R={curv_info['radius']*1000:.2f}mm")

        report.append("=" * 60)
        return "\n".join(report)
