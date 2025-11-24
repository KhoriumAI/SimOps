"""
Paintbrush Geometry Selector
=============================

Handles surface selection and painted region management for the paintbrush
mesh refinement feature.

Features:
- Ray casting from mouse to 3D CAD surfaces
- Surface proximity detection with brush radius
- Painted region storage and serialization
- Pre-meshing mode: paint CAD surfaces
- Post-meshing mode: paint mesh faces

Usage:
    selector = PaintbrushSelector()
    selector.load_cad_geometry()

    # Paint a region
    surface_tags = selector.get_surfaces_at_point(x, y, z, brush_radius)
    region = selector.add_painted_region(surface_tags, brush_radius, refinement_level)

    # Get all painted regions for meshing
    regions = selector.get_painted_regions()
"""

import gmsh
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import json
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class PaintedRegion:
    """
    Represents a region painted by the user for mesh refinement.

    Attributes:
        surface_tags: List of gmsh surface tags (dim=2) in this region
        brush_radius: Brush radius in mm used to paint this region
        refinement_level: Multiplier for refinement (1.0 = normal, 5.0 = 5x finer)
        center_point: Approximate center of painted region [x, y, z]
        timestamp: When this region was painted
    """
    surface_tags: List[int]
    brush_radius: float
    refinement_level: float
    center_point: Optional[Tuple[float, float, float]] = None
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict:
        """
        Convert to dictionary for JSON serialization.
        
        IMPORTANT: The backend expects volumetric refinement zones with 'center' and 'radius' keys.
        This method converts surface-based painting to volumetric format by computing a bounding sphere.
        """
        data = asdict(self)
        if data['center_point'] is not None:
            data['center_point'] = list(data['center_point'])
        
        # CRITICAL FIX: Convert surface-based data to volumetric format
        # Backend needs {'center': [x,y,z], 'radius': r}
        # We compute a bounding sphere from the painted surfaces
        
        if self.center_point is not None:
            # If we have a center point, use it directly
            data['center'] = list(self.center_point)
            # Use brush radius as the volumetric refinement radius
            data['radius'] = self.brush_radius
        else:
            # Fallback: If no center point stored, try to compute from surfaces
            # This requires gmsh to be initialized with the geometry
            try:
                import gmsh
                if self.surface_tags and gmsh.is_initialized():
                    # Compute bounding box of all painted surfaces
                    xmin, ymin, zmin = float('inf'), float('inf'), float('inf')
                    xmax, ymax, zmax = float('-inf'), float('-inf'), float('-inf')
                    
                    for surf_tag in self.surface_tags:
                        try:
                            bbox = gmsh.model.getBoundingBox(2, surf_tag)
                            xmin = min(xmin, bbox[0])
                            ymin = min(ymin, bbox[1])
                            zmin = min(zmin, bbox[2])
                            xmax = max(xmax, bbox[3])
                            ymax = max(ymax, bbox[4])
                            zmax = max(zmax, bbox[5])
                        except:
                            pass
                    
                    # Compute center and radius of bounding sphere
                    if xmin != float('inf'):
                        cx = (xmin + xmax) / 2
                        cy = (ymin + ymax) / 2
                        cz = (zmin + zmax) / 2
                        data['center'] = [cx, cy, cz]
                        
                        # Radius is half the diagonal of the bounding box
                        dx = xmax - xmin
                        dy = ymax - ymin
                        dz = zmax - zmin
                        radius = np.sqrt(dx**2 + dy**2 + dz**2) / 2
                        data['radius'] = max(radius, self.brush_radius)
                    else:
                        # No valid surfaces - use defaults
                        data['center'] = [0, 0, 0]
                        data['radius'] = self.brush_radius
                else:
                    # Gmsh not available - use defaults
                    data['center'] = [0, 0, 0]
                    data['radius'] = self.brush_radius
            except Exception as e:
                print(f"[!] Warning: Could not compute volumetric bounds from surfaces: {e}")
                data['center'] = [0, 0, 0]
                data['radius'] = self.brush_radius
        
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'PaintedRegion':
        """Create from dictionary"""
        if data.get('center_point') is not None:
            data['center_point'] = tuple(data['center_point'])
        return cls(**data)


class PaintbrushSelector:
    """
    Manages surface selection and painted regions for paintbrush refinement.

    This class interfaces with gmsh to identify CAD surfaces under the brush
    cursor and maintains a list of painted regions for mesh refinement.
    """

    def __init__(self):
        """Initialize paintbrush selector"""
        self.painted_regions: List[PaintedRegion] = []
        self.available_surfaces: List[Tuple[int, int]] = []  # [(dim, tag), ...]
        self.surface_bboxes: Dict[int, Tuple[float, ...]] = {}  # tag -> (xmin, ymin, zmin, xmax, ymax, zmax)
        self.geometry_loaded = False

    def load_cad_geometry(self):
        """
        Load geometry information from current gmsh model.
        Extracts all surfaces and their bounding boxes for picking.

        Must be called after CAD file is loaded into gmsh.
        """
        try:
            # Get all 2D surfaces from gmsh model
            self.available_surfaces = gmsh.model.getEntities(dim=2)

            if not self.available_surfaces:
                print("Warning: No surfaces found in gmsh model")
                return False

            # Cache bounding boxes for fast proximity queries
            self.surface_bboxes = {}
            for dim, tag in self.available_surfaces:
                try:
                    bbox = gmsh.model.getBoundingBox(dim, tag)
                    self.surface_bboxes[tag] = bbox
                except Exception as e:
                    print(f"Warning: Could not get bbox for surface {tag}: {e}")

            self.geometry_loaded = True
            print(f"Paintbrush: Loaded {len(self.available_surfaces)} surfaces")
            return True

        except Exception as e:
            print(f"Error loading CAD geometry for paintbrush: {e}")
            return False

    def get_surfaces_near_point(self,
                                point: Tuple[float, float, float],
                                radius: float) -> List[int]:
        """
        Find all surfaces within brush radius of a 3D point.

        Args:
            point: 3D point [x, y, z] in model coordinates
            radius: Brush radius in same units as model

        Returns:
            List of surface tags within radius
        """
        if not self.geometry_loaded:
            return []

        x, y, z = point
        nearby_surfaces = []

        for tag, bbox in self.surface_bboxes.items():
            xmin, ymin, zmin, xmax, ymax, zmax = bbox

            # Check if brush sphere intersects bounding box
            # Find closest point on bbox to sphere center
            closest_x = np.clip(x, xmin, xmax)
            closest_y = np.clip(y, ymin, ymax)
            closest_z = np.clip(z, zmin, zmax)

            # Distance from sphere center to closest point on bbox
            dist = np.sqrt((x - closest_x)**2 +
                          (y - closest_y)**2 +
                          (z - closest_z)**2)

            if dist <= radius:
                nearby_surfaces.append(tag)

        return nearby_surfaces

    def get_surface_at_point(self, point: Tuple[float, float, float]) -> Optional[int]:
        """
        Find the single closest surface to a point.

        Args:
            point: 3D point [x, y, z]

        Returns:
            Surface tag of closest surface, or None if none found
        """
        if not self.geometry_loaded:
            return None

        x, y, z = point
        min_dist = float('inf')
        closest_surface = None

        for tag, bbox in self.surface_bboxes.items():
            xmin, ymin, zmin, xmax, ymax, zmax = bbox

            # Find closest point on bbox
            closest_x = np.clip(x, xmin, xmax)
            closest_y = np.clip(y, ymin, ymax)
            closest_z = np.clip(z, zmin, zmax)

            dist = np.sqrt((x - closest_x)**2 +
                          (y - closest_y)**2 +
                          (z - closest_z)**2)

            if dist < min_dist:
                min_dist = dist
                closest_surface = tag

        return closest_surface

    def add_painted_region(self,
                          surface_tags: List[int],
                          brush_radius: float,
                          refinement_level: float,
                          center_point: Optional[Tuple[float, float, float]] = None) -> PaintedRegion:
        """
        Add a new painted region.

        Args:
            surface_tags: List of surface tags to refine
            brush_radius: Brush radius used
            refinement_level: Refinement multiplier (1.0 - 10.0)
            center_point: Optional center point of region

        Returns:
            Created PaintedRegion object
        """
        from datetime import datetime

        # Remove duplicates from surface tags
        surface_tags = list(set(surface_tags))

        region = PaintedRegion(
            surface_tags=surface_tags,
            brush_radius=brush_radius,
            refinement_level=refinement_level,
            center_point=center_point,
            timestamp=datetime.now().isoformat()
        )

        self.painted_regions.append(region)
        return region

    def merge_overlapping_regions(self):
        """
        Merge painted regions that share surfaces.
        This reduces the number of fields and improves performance.
        """
        if len(self.painted_regions) < 2:
            return

        merged = []
        used = set()

        for i, region1 in enumerate(self.painted_regions):
            if i in used:
                continue

            # Find all regions that overlap with this one
            merged_tags = set(region1.surface_tags)
            merged_radius = region1.brush_radius
            merged_level = region1.refinement_level

            for j, region2 in enumerate(self.painted_regions[i+1:], start=i+1):
                if j in used:
                    continue

                # Check for overlap
                if set(region2.surface_tags) & merged_tags:
                    merged_tags.update(region2.surface_tags)
                    merged_radius = max(merged_radius, region2.brush_radius)
                    merged_level = max(merged_level, region2.refinement_level)
                    used.add(j)

            # Create merged region
            merged_region = PaintedRegion(
                surface_tags=list(merged_tags),
                brush_radius=merged_radius,
                refinement_level=merged_level,
                center_point=region1.center_point
            )
            merged.append(merged_region)
            used.add(i)

        self.painted_regions = merged
        print(f"Merged to {len(self.painted_regions)} regions")

    def remove_region(self, index: int):
        """Remove painted region by index"""
        if 0 <= index < len(self.painted_regions):
            self.painted_regions.pop(index)

    def clear_all_regions(self):
        """Clear all painted regions"""
        self.painted_regions = []

    def get_painted_regions(self) -> List[PaintedRegion]:
        """Get all painted regions"""
        return self.painted_regions.copy()

    def get_all_painted_surfaces(self) -> List[int]:
        """Get unique list of all painted surface tags"""
        all_surfaces = set()
        for region in self.painted_regions:
            all_surfaces.update(region.surface_tags)
        return list(all_surfaces)

    def is_surface_painted(self, surface_tag: int) -> bool:
        """Check if a surface is in any painted region"""
        return surface_tag in self.get_all_painted_surfaces()

    def save_to_file(self, filepath: Path):
        """
        Save painted regions to JSON file.

        Args:
            filepath: Path to save JSON file
        """
        data = {
            'version': '1.0',
            'painted_regions': [region.to_dict() for region in self.painted_regions]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load_from_file(self, filepath: Path) -> bool:
        """
        Load painted regions from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            self.painted_regions = [
                PaintedRegion.from_dict(region_data)
                for region_data in data['painted_regions']
            ]

            return True

        except Exception as e:
            print(f"Error loading painted regions: {e}")
            return False

    def get_statistics(self) -> Dict:
        """
        Get statistics about painted regions.

        Returns:
            Dictionary with statistics
        """
        if not self.painted_regions:
            return {
                'num_regions': 0,
                'num_surfaces': 0,
                'avg_refinement': 0.0,
                'max_refinement': 0.0
            }

        all_surfaces = self.get_all_painted_surfaces()
        refinement_levels = [r.refinement_level for r in self.painted_regions]

        return {
            'num_regions': len(self.painted_regions),
            'num_surfaces': len(all_surfaces),
            'avg_refinement': np.mean(refinement_levels),
            'max_refinement': np.max(refinement_levels),
            'min_refinement': np.min(refinement_levels)
        }


def estimate_element_count_increase(painted_regions: List[PaintedRegion],
                                    base_element_count: int,
                                    geometry_info: Dict) -> int:
    """
    Estimate how many additional elements will be created by paintbrush refinement.

    Args:
        painted_regions: List of painted regions
        base_element_count: Expected element count without refinement
        geometry_info: Geometry information dict with surface areas

    Returns:
        Estimated total element count with refinement
    """
    if not painted_regions:
        return base_element_count

    # Rough estimate: refinement_level^3 for 3D elements
    # (each dimension gets finer by refinement_level factor)
    total_multiplier = 1.0

    for region in painted_regions:
        # Estimate fraction of model affected by this region
        # (simplified: assumes uniform surface area distribution)
        num_surfaces = len(region.surface_tags)
        total_surfaces = len(geometry_info.get('surfaces', []))

        if total_surfaces > 0:
            fraction = num_surfaces / total_surfaces
            refinement_cube = region.refinement_level ** 3
            total_multiplier += fraction * (refinement_cube - 1)

    estimated_count = int(base_element_count * total_multiplier)
    return estimated_count


if __name__ == "__main__":
    # Test mode - requires gmsh to be initialized
    print("Paintbrush Geometry Selector - Test Mode")
    print("This module requires gmsh.initialize() to be called with a loaded model")
