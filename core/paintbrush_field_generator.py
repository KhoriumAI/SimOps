"""
Paintbrush Field Generator
===========================

Creates gmsh size fields for paintbrush-based mesh refinement.

This module generates Distance and Threshold fields around painted surfaces
and combines them with existing fields (curvature, intelligent sizing) using
Min operation to ensure the finest size wins everywhere.

Features:
- Per-region size field generation
- Smooth size transitions (gradient control)
- Field combination with existing fields
- Safety limits to prevent degenerate meshes

Usage:
    from core.paintbrush_geometry import PaintedRegion, PaintbrushSelector

    selector = PaintbrushSelector()
    # ... paint regions ...

    generator = PaintbrushFieldGenerator()
    field_tag = generator.create_paintbrush_fields(
        selector.get_painted_regions(),
        base_size=10.0
    )

    gmsh.model.mesh.field.setAsBackgroundMesh(field_tag)
"""

import gmsh
from typing import List, Optional, Tuple
import numpy as np


class PaintbrushFieldGenerator:
    """
    Generates gmsh mesh size fields for paintbrush refinement.

    Creates Distance and Threshold fields around painted surfaces with
    smooth size transitions.
    """

    def __init__(self,
                 gradient_factor: float = 10.0,
                 absolute_min_size: float = 0.001,
                 absolute_max_size: float = 1000.0):
        """
        Initialize field generator.

        Args:
            gradient_factor: Controls size transition distance (higher = smoother)
            absolute_min_size: Minimum allowed element size (safety limit)
            absolute_max_size: Maximum allowed element size (safety limit)
        """
        self.gradient_factor = gradient_factor
        self.absolute_min_size = absolute_min_size
        self.absolute_max_size = absolute_max_size
        self.created_fields: List[int] = []

    def create_paintbrush_fields(self,
                                 painted_regions: List,  # List[PaintedRegion]
                                 base_size: float,
                                 existing_fields: Optional[List[int]] = None) -> int:
        """
        Create mesh size fields for all painted regions.

        Args:
            painted_regions: List of PaintedRegion objects
            base_size: Base mesh size (before refinement)
            existing_fields: Optional list of existing field tags to combine with

        Returns:
            Field tag of combined field (ready to set as background mesh)
        """
        if not painted_regions:
            # No painted regions - return existing field or create default
            if existing_fields and len(existing_fields) > 0:
                if len(existing_fields) == 1:
                    return existing_fields[0]
                else:
                    # Combine existing fields
                    return self._create_min_field(existing_fields)
            else:
                # Create constant field with base size
                return self._create_constant_field(base_size)

        # Create field for each painted region
        region_fields = []

        for i, region in enumerate(painted_regions):
            print(f"Creating field for region {i+1}/{len(painted_regions)}: "
                  f"{len(region.surface_tags)} surfaces, "
                  f"{region.refinement_level}x refinement")

            field_tag = self._create_region_field(region, base_size)

            if field_tag is not None:
                region_fields.append(field_tag)
                self.created_fields.append(field_tag)

        # Combine all region fields
        if not region_fields:
            print("Warning: No fields created for painted regions")
            return self._create_constant_field(base_size)

        # Combine region fields with Min (finest wins)
        combined_region_field = self._create_min_field(region_fields)

        # If there are existing fields (curvature, intelligent sizing),
        # combine with those too
        all_fields = [combined_region_field]
        if existing_fields:
            all_fields.extend(existing_fields)

        if len(all_fields) == 1:
            return all_fields[0]
        else:
            final_field = self._create_min_field(all_fields)
            return final_field

    def _create_region_field(self, region, base_size: float) -> Optional[int]:
        """
        Create Distance + Threshold field for a single painted region.

        Args:
            region: PaintedRegion object
            base_size: Base mesh size

        Returns:
            Field tag, or None if creation failed
        """
        try:
            surface_tags = region.surface_tags
            brush_radius = region.brush_radius
            refinement_level = region.refinement_level

            if not surface_tags:
                print("Warning: Region has no surfaces")
                return None

            # Calculate target sizes
            size_min = base_size / refinement_level
            size_max = base_size

            # Apply safety limits
            size_min = max(size_min, self.absolute_min_size)
            size_max = min(size_max, self.absolute_max_size)

            # Ensure min < max
            if size_min >= size_max:
                size_min = size_max * 0.5

            # Create Distance field to surfaces
            dist_field = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(dist_field, "SurfacesList", surface_tags)
            gmsh.model.mesh.field.setNumber(dist_field, "Sampling", 100)  # Accuracy

            # Distance parameters for smooth transition
            # Fine mesh near surface, coarse mesh far away
            dist_min = 0.0  # Start refinement immediately at surface
            dist_max = brush_radius * self.gradient_factor  # End refinement here

            # Create Threshold field for size gradient
            threshold_field = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(threshold_field, "InField", dist_field)
            gmsh.model.mesh.field.setNumber(threshold_field, "SizeMin", size_min)
            gmsh.model.mesh.field.setNumber(threshold_field, "SizeMax", size_max)
            gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", dist_min)
            gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", dist_max)

            # Use StopAtDistMax to limit field influence
            gmsh.model.mesh.field.setNumber(threshold_field, "StopAtDistMax", 1)

            self.created_fields.append(dist_field)
            self.created_fields.append(threshold_field)

            print(f"  Field {threshold_field}: size {size_min:.3f} - {size_max:.3f} mm, "
                  f"distance 0 - {dist_max:.1f} mm")

            return threshold_field

        except Exception as e:
            print(f"Error creating field for region: {e}")
            return None

    def _create_min_field(self, field_list: List[int]) -> int:
        """
        Create Min field that combines multiple fields.
        The minimum (finest) size is taken at each point.

        Args:
            field_list: List of field tags to combine

        Returns:
            Tag of Min field
        """
        min_field = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", field_list)
        self.created_fields.append(min_field)
        return min_field

    def _create_constant_field(self, size: float) -> int:
        """
        Create constant size field.

        Args:
            size: Constant element size

        Returns:
            Field tag
        """
        size = np.clip(size, self.absolute_min_size, self.absolute_max_size)
        const_field = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(const_field, "F", str(size))
        self.created_fields.append(const_field)
        return const_field

    def clear_created_fields(self):
        """
        Remove all fields created by this generator.
        Call this before creating new fields to avoid conflicts.
        """
        for field_tag in self.created_fields:
            try:
                gmsh.model.mesh.field.remove(field_tag)
            except:
                pass  # Field may already be removed
        self.created_fields = []

    def estimate_element_sizes(self,
                               painted_regions: List,
                               base_size: float,
                               sample_points: List[Tuple[float, float, float]]) -> List[float]:
        """
        Estimate element sizes at given sample points.

        Useful for visualizing refinement before meshing.

        Args:
            painted_regions: List of PaintedRegion objects
            base_size: Base mesh size
            sample_points: List of [x, y, z] points to sample

        Returns:
            List of element sizes at sample points
        """
        sizes = []

        for point in sample_points:
            x, y, z = point
            min_size = base_size

            # Check each painted region
            for region in painted_regions:
                # Find minimum distance to any surface in region
                for surface_tag in region.surface_tags:
                    try:
                        # Get surface bounding box
                        bbox = gmsh.model.getBoundingBox(2, surface_tag)
                        xmin, ymin, zmin, xmax, ymax, zmax = bbox

                        # Distance to bbox (approximation)
                        closest_x = np.clip(x, xmin, xmax)
                        closest_y = np.clip(y, ymin, ymax)
                        closest_z = np.clip(z, zmin, zmax)

                        dist = np.sqrt((x - closest_x)**2 +
                                      (y - closest_y)**2 +
                                      (z - closest_z)**2)

                        # Calculate size based on distance
                        dist_max = region.brush_radius * self.gradient_factor

                        if dist <= dist_max:
                            size_min = base_size / region.refinement_level
                            size_max = base_size

                            # Linear interpolation
                            if dist_max > 0:
                                t = dist / dist_max
                                size = size_min + t * (size_max - size_min)
                            else:
                                size = size_min

                            min_size = min(min_size, size)

                    except:
                        pass

            sizes.append(min_size)

        return sizes


def create_paintbrush_field_simple(surface_tags: List[int],
                                   refinement_level: float,
                                   base_size: float,
                                   brush_radius: float = 5.0) -> int:
    """
    Simple helper function to create a paintbrush field for given surfaces.

    Args:
        surface_tags: List of surface tags to refine
        refinement_level: Refinement multiplier (e.g., 5.0 for 5x finer)
        base_size: Base mesh size
        brush_radius: Brush radius in mm

    Returns:
        Field tag ready to set as background mesh
    """
    # Create distance field
    dist_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(dist_field, "SurfacesList", surface_tags)
    gmsh.model.mesh.field.setNumber(dist_field, "Sampling", 100)

    # Create threshold field
    size_min = base_size / refinement_level
    size_max = base_size
    dist_max = brush_radius * 10.0  # Gradient factor

    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_field, "InField", dist_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "SizeMin", size_min)
    gmsh.model.mesh.field.setNumber(threshold_field, "SizeMax", size_max)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", dist_max)
    gmsh.model.mesh.field.setNumber(threshold_field, "StopAtDistMax", 1)

    return threshold_field


if __name__ == "__main__":
    print("Paintbrush Field Generator - Test Mode")
    print("This module requires gmsh.initialize() to be called")
    print("\nExample usage:")
    print("  generator = PaintbrushFieldGenerator()")
    print("  field = generator.create_paintbrush_fields(painted_regions, base_size=10.0)")
    print("  gmsh.model.mesh.field.setAsBackgroundMesh(field)")
