"""
Geometry Cleanup and Defeaturing Module
========================================

Handles problematic CAD geometry that causes meshing failures:
- Removes or merges small edges and curves
- Identifies and handles sharp features
- Detects infinitely thin surfaces
- Provides geometry repair utilities

Uses Gmsh's OpenCASCADE kernel capabilities.
"""

import gmsh
import math
from typing import Dict, List, Tuple, Optional
import numpy as np


class GeometryCleanup:
    """
    Geometry cleanup and defeaturing utilities

    Helps prepare CAD geometry for robust meshing by:
    - Identifying problematic features
    - Removing/merging small curves
    - Handling sharp edges
    - Detecting thin surfaces
    """

    def __init__(self, min_feature_size: Optional[float] = None):
        """
        Initialize geometry cleanup

        Args:
            min_feature_size: Minimum feature size to preserve (auto-detect if None)
        """
        self.min_feature_size = min_feature_size
        self.geometry_stats = {}

    def analyze_geometry(self) -> Dict:
        """
        Analyze loaded geometry for problematic features

        Returns:
            Dictionary with geometry statistics and issues
        """
        stats = {
            'volumes': [],
            'surfaces': [],
            'curves': [],
            'small_curves': [],
            'sharp_edges': [],
            'thin_surfaces': [],
            'bounding_box': None,
            'characteristic_length': None
        }

        try:
            # Get all entities
            volumes = gmsh.model.getEntities(dim=3)
            surfaces = gmsh.model.getEntities(dim=2)
            curves = gmsh.model.getEntities(dim=1)

            stats['volumes'] = volumes
            stats['surfaces'] = surfaces
            stats['curves'] = curves

            # Calculate bounding box and characteristic length
            if volumes:
                bb = gmsh.model.getBoundingBox(-1, -1)
                stats['bounding_box'] = {
                    'min': [bb[0], bb[1], bb[2]],
                    'max': [bb[3], bb[4], bb[5]]
                }
                diagonal = math.sqrt(
                    (bb[3] - bb[0])**2 + (bb[4] - bb[1])**2 + (bb[5] - bb[2])**2
                )
                stats['characteristic_length'] = diagonal

                # Auto-set minimum feature size if not provided
                if self.min_feature_size is None:
                    self.min_feature_size = diagonal / 1000.0  # 0.1% of diagonal

            # Analyze curves for small features
            curve_lengths = []
            for dim, tag in curves:
                bb = gmsh.model.getBoundingBox(dim, tag)
                length = math.sqrt(
                    (bb[3] - bb[0])**2 + (bb[4] - bb[1])**2 + (bb[5] - bb[2])**2
                )
                curve_lengths.append((tag, length))

                # Flag small curves
                if length < self.min_feature_size:
                    stats['small_curves'].append({
                        'tag': tag,
                        'length': length
                    })

            # Detect sharp edges (high curvature)
            stats['sharp_edges'] = self._detect_sharp_edges(curves)

            # Detect thin surfaces
            stats['thin_surfaces'] = self._detect_thin_surfaces(surfaces)

            self.geometry_stats = stats
            return stats

        except Exception as e:
            print(f"Geometry analysis failed: {e}")
            return stats

    def _detect_sharp_edges(self, curves: List[Tuple]) -> List[Dict]:
        """Detect curves with sharp angles (high curvature)"""
        sharp_edges = []

        for dim, tag in curves:
            try:
                # Get curve type - circles/arcs have high curvature
                curve_type = gmsh.model.getType(dim, tag)

                # Small circles are problematic
                if curve_type == "Circle":
                    bb = gmsh.model.getBoundingBox(dim, tag)
                    radius = math.sqrt(
                        (bb[3] - bb[0])**2 + (bb[4] - bb[1])**2 + (bb[5] - bb[2])**2
                    ) / 2.0

                    if radius < self.min_feature_size:
                        sharp_edges.append({
                            'tag': tag,
                            'type': 'circle',
                            'radius': radius
                        })
            except:
                continue

        return sharp_edges

    def _detect_thin_surfaces(self, surfaces: List[Tuple]) -> List[Dict]:
        """Detect surfaces that are very thin (near-zero thickness)"""
        thin_surfaces = []

        # This is hard to detect without proper CAD analysis
        # For now, flag surfaces with very small bounding boxes
        for dim, tag in surfaces:
            try:
                bb = gmsh.model.getBoundingBox(dim, tag)
                dx = bb[3] - bb[0]
                dy = bb[4] - bb[1]
                dz = bb[5] - bb[2]

                min_dimension = min(dx, dy, dz)

                # If one dimension is very small compared to others
                if min_dimension < self.min_feature_size and max(dx, dy, dz) > 10 * min_dimension:
                    thin_surfaces.append({
                        'tag': tag,
                        'min_dimension': min_dimension,
                        'aspect_ratio': max(dx, dy, dz) / max(min_dimension, 1e-10)
                    })
            except:
                continue

        return thin_surfaces

    def remove_small_curves(self, threshold: Optional[float] = None) -> int:
        """
        Attempt to remove or suppress small curves

        Args:
            threshold: Size threshold (uses auto-detected if None)

        Returns:
            Number of curves removed/suppressed
        """
        if threshold is None:
            threshold = self.min_feature_size

        curves = gmsh.model.getEntities(dim=1)
        removed_count = 0

        for dim, tag in curves:
            try:
                bb = gmsh.model.getBoundingBox(dim, tag)
                length = math.sqrt(
                    (bb[3] - bb[0])**2 + (bb[4] - bb[1])**2 + (bb[5] - bb[2])**2
                )

                if length < threshold:
                    # Try to remove the curve
                    # Note: This may break geometry - use cautiously
                    try:
                        gmsh.model.occ.remove([(dim, tag)], recursive=False)
                        removed_count += 1
                    except:
                        # Can't remove - it's part of surface definition
                        pass
            except:
                continue

        if removed_count > 0:
            gmsh.model.occ.synchronize()

        return removed_count

    def smooth_sharp_features(self, fillet_radius: float = None) -> int:
        """
        Apply fillets to sharp edges to prevent degenerate elements

        This is critical for geometries like airfoils with sharp trailing edges.
        Sharp features often cause extremely poor quality elements (SICN < 0.01).

        Args:
            fillet_radius: Radius for fillets (auto if None)

        Returns:
            Number of edges filleted
        """
        if fillet_radius is None:
            if self.min_feature_size:
                fillet_radius = self.min_feature_size * 3.0  # 3x min feature size
            else:
                # Default fallback
                fillet_radius = 0.5

        filleted_count = 0

        try:
            print(f"  Analyzing sharp features for smoothing (fillet radius: {fillet_radius:.4f})...")

            # Get all curves (edges)
            curves = gmsh.model.getEntities(dim=1)

            curves_to_fillet = []

            for dim, tag in curves:
                try:
                    # Get surfaces adjacent to this curve
                    upward, _ = gmsh.model.getAdjacencies(dim, tag)

                    # Only consider edges between 2 surfaces (not boundaries)
                    if len(upward) >= 2:
                        # Get curve length
                        bb = gmsh.model.getBoundingBox(dim, tag)
                        length = math.sqrt(
                            (bb[3] - bb[0])**2 + (bb[4] - bb[1])**2 + (bb[5] - bb[2])**2
                        )

                        # Only fillet small sharp edges (likely to cause problems)
                        if length < self.min_feature_size * 10:
                            curves_to_fillet.append((tag, length))
                except:
                    continue

            if curves_to_fillet:
                print(f"  Found {len(curves_to_fillet)} potentially sharp edges")
                print(f"  Applying fillets to smooth sharp corners...")

                for curve_tag, length in curves_to_fillet:
                    try:
                        # Try to apply fillet
                        # Note: This may fail for some geometries - that's ok
                        gmsh.model.occ.fillet([curve_tag], [fillet_radius])
                        filleted_count += 1
                    except:
                        # Some edges can't be filleted (boundary edges, etc.)
                        pass

                if filleted_count > 0:
                    gmsh.model.occ.synchronize()
                    print(f"  [OK] Successfully filleted {filleted_count} sharp edges")
                else:
                    print(f"  [!] Could not fillet any edges (geometry may not support it)")
            else:
                print(f"  ℹ No sharp edges detected that need smoothing")

        except Exception as e:
            print(f"  Warning: Sharp feature smoothing failed: {e}")

        return filleted_count

    def apply_geometry_tolerance(self, tolerance: float = 1e-6) -> None:
        """
        Apply geometry tolerance to merge nearby points

        This can help close small gaps and merge duplicate vertices.

        Args:
            tolerance: Merging tolerance
        """
        try:
            # Set geometry tolerance
            gmsh.option.setNumber("Geometry.Tolerance", tolerance)
            gmsh.option.setNumber("Geometry.ToleranceBoolean", tolerance)

            # Try to heal the geometry
            gmsh.model.occ.synchronize()
            gmsh.model.occ.removeAllDuplicates()
            gmsh.model.occ.synchronize()

        except Exception as e:
            print(f"Tolerance application failed: {e}")

    def create_defeatured_mesh_size_field(self) -> None:
        """
        Create a mesh size field that handles small features gracefully

        Sets larger element sizes near small features to avoid meshing issues.
        """
        if not self.geometry_stats:
            self.analyze_geometry()

        stats = self.geometry_stats

        # If we have small curves, create a distance field with larger sizes
        if stats['small_curves']:
            try:
                # Create distance field from small curves
                small_curve_tags = [c['tag'] for c in stats['small_curves']]

                # Distance field
                dist_field = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(dist_field, "CurvesList", small_curve_tags)

                # Threshold field - larger mesh near small features
                threshold_field = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(threshold_field, "InField", dist_field)
                gmsh.model.mesh.field.setNumber(threshold_field, "SizeMin", self.min_feature_size * 5)
                gmsh.model.mesh.field.setNumber(threshold_field, "SizeMax", self.min_feature_size * 10)
                gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", self.min_feature_size * 2)
                gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", self.min_feature_size * 10)

                gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)

            except Exception as e:
                print(f"Size field creation failed: {e}")

    def get_cleanup_report(self) -> str:
        """Generate human-readable cleanup report"""
        if not self.geometry_stats:
            self.analyze_geometry()

        stats = self.geometry_stats

        report = []
        report.append("=" * 60)
        report.append("GEOMETRY CLEANUP ANALYSIS")
        report.append("=" * 60)

        report.append(f"\nGeometry Overview:")
        report.append(f"  Volumes: {len(stats['volumes'])}")
        report.append(f"  Surfaces: {len(stats['surfaces'])}")
        report.append(f"  Curves: {len(stats['curves'])}")

        if stats['characteristic_length']:
            report.append(f"  Characteristic Length: {stats['characteristic_length']:.4f}")
            report.append(f"  Min Feature Size Threshold: {self.min_feature_size:.6f}")

        # Issues found
        report.append(f"\nProblematic Features Detected:")

        if stats['small_curves']:
            report.append(f"  [!] {len(stats['small_curves'])} small curves (< {self.min_feature_size:.6f})")
            if len(stats['small_curves']) <= 5:
                for c in stats['small_curves']:
                    report.append(f"    - Curve {c['tag']}: length={c['length']:.6f}")

        if stats['sharp_edges']:
            report.append(f"  [!] {len(stats['sharp_edges'])} sharp edges detected")

        if stats['thin_surfaces']:
            report.append(f"  [!] {len(stats['thin_surfaces'])} thin surfaces detected")
            for s in stats['thin_surfaces'][:3]:  # Show first 3
                report.append(f"    - Surface {s['tag']}: min_dim={s['min_dimension']:.6f}, aspect={s['aspect_ratio']:.1f}")

        if not (stats['small_curves'] or stats['sharp_edges'] or stats['thin_surfaces']):
            report.append("  [OK] No obvious problematic features detected")

        report.append("\nRecommendations:")
        if stats['small_curves'] or stats['sharp_edges']:
            report.append("  1. Use defeatured mesh size field (coarser mesh near small features)")
            report.append("  2. Consider geometry simplification in CAD software")
            report.append("  3. Increase Geometry.Tolerance if appropriate")

        if stats['thin_surfaces']:
            report.append("  4. Thin surfaces may cause inverted elements")
            report.append("  5. Consider thickening or removing thin features in CAD")

        report.append("=" * 60)

        return "\n".join(report)
