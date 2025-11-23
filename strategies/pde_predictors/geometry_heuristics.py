"""
Geometry-Based Heuristics for Mesh Refinement
==============================================

Predicts mesh refinement zones from geometry alone (no PDE solve).
Instant results (<10ms) based on curvature, sharp features, thin walls, and holes.

Inspired by ANSYS automatic refinement and COMSOL physics-controlled mesh.
"""

import gmsh
import numpy as np
from typing import Dict, List, Tuple, Optional
import time


class GeometryHeuristics:
    """
    Fast geometry analysis for physics-informed mesh refinement

    Predicts critical regions that typically require finer mesh:
    - High curvature areas (stress concentration, flow separation)
    - Sharp corners (stress risers, singularities)
    - Thin sections (buckling risk, thermal gradients)
    - Holes/cutouts (stress concentration)
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize geometry heuristics analyzer

        Args:
            verbose: Print analysis details
        """
        self.verbose = verbose
        self.refinement_zones = []
        self.mesh_size_field_tags = []

    def analyze_geometry(self, input_file: str = None) -> Dict:
        """
        Analyze geometry and generate refinement zones

        Args:
            input_file: Path to CAD file (optional if gmsh already loaded)

        Returns:
            Dictionary with:
            - refinement_zones: List of (region_type, entities, priority)
            - curvature_stats: Min/max/avg curvature
            - sharp_features: List of sharp edge pairs
            - thin_sections: List of thin surfaces
            - holes_detected: Number of holes found
            - execution_time: Analysis time in ms
        """
        start_time = time.time()

        if input_file and not gmsh.isInitialized():
            gmsh.initialize()
            gmsh.open(input_file)

        self._log("🔍 Analyzing geometry for critical features...")

        # Get geometry entities
        surfaces = gmsh.model.getEntities(dim=2)
        curves = gmsh.model.getEntities(dim=1)
        volumes = gmsh.model.getEntities(dim=3)

        # Get bounding box
        bbox = gmsh.model.getBoundingBox(-1, -1)
        bbox_size = np.array([bbox[3]-bbox[0], bbox[4]-bbox[1], bbox[5]-bbox[2]])
        char_length = np.linalg.norm(bbox_size)

        self._log(f"  Geometry: {len(volumes)} volumes, {len(surfaces)} surfaces, {len(curves)} curves")
        self._log(f"  Bounding box: {bbox_size[0]:.2f} x {bbox_size[1]:.2f} x {bbox_size[2]:.2f}")

        results = {
            'refinement_zones': [],
            'curvature_stats': {},
            'sharp_features': [],
            'thin_sections': [],
            'holes_detected': 0,
            'execution_time_ms': 0
        }

        # 1. Curvature analysis
        self._log("\n  [1/4] Curvature analysis...")
        curvature_zones = self._analyze_curvature(surfaces, char_length)
        results['curvature_stats'] = curvature_zones

        # 2. Sharp feature detection
        self._log("  [2/4] Sharp feature detection...")
        sharp_zones = self._detect_sharp_features(curves, char_length)
        results['sharp_features'] = sharp_zones

        # 3. Thin wall detection
        self._log("  [3/4] Thin wall detection...")
        thin_zones = self._detect_thin_walls(surfaces, char_length)
        results['thin_sections'] = thin_zones

        # 4. Hole/cutout detection
        self._log("  [4/4] Hole/cutout detection...")
        hole_zones = self._detect_holes(surfaces, char_length)
        results['holes_detected'] = len(hole_zones)

        # Combine all zones
        results['refinement_zones'] = (
            curvature_zones.get('zones', []) +
            sharp_zones +
            thin_zones +
            hole_zones
        )

        execution_time = (time.time() - start_time) * 1000
        results['execution_time_ms'] = execution_time

        self._log(f"\n[OK] Analysis complete in {execution_time:.1f}ms")
        self._log(f"  Found {len(results['refinement_zones'])} refinement zones:")
        self._log(f"    - {len(curvature_zones.get('zones', []))} high curvature regions")
        self._log(f"    - {len(sharp_zones)} sharp features")
        self._log(f"    - {len(thin_zones)} thin walls")
        self._log(f"    - {len(hole_zones)} holes/cutouts")

        return results

    def _analyze_curvature(self, surfaces: List, char_length: float) -> Dict:
        """
        Analyze surface curvature to identify stress/flow concentration zones

        High curvature -> stress concentration, flow separation, thermal gradients
        """
        zones = []
        curvatures = []

        for dim, tag in surfaces:
            try:
                # Get surface curvature at parameter center
                bounds = gmsh.model.getParametrizationBounds(dim, tag)
                if bounds:
                    u_mid = (bounds[0][0] + bounds[0][1]) / 2
                    v_mid = (bounds[1][0] + bounds[1][1]) / 2

                    # Get curvature at this point
                    curvature = gmsh.model.getCurvature(dim, tag, [u_mid, v_mid])

                    if curvature:
                        max_curv = max(abs(curvature[0]), abs(curvature[1]))
                        curvatures.append(max_curv)

                        # High curvature threshold: top 20%
                        if len(curvatures) > 5:
                            threshold = np.percentile(curvatures, 80)
                            if max_curv > threshold:
                                zones.append({
                                    'type': 'high_curvature',
                                    'entity': (dim, tag),
                                    'curvature': max_curv,
                                    'priority': 0.8,  # High priority
                                    'refinement_factor': 2.0  # 2x finer
                                })
            except Exception as e:
                # Surface may not support parameterization
                pass

        return {
            'zones': zones,
            'min': min(curvatures) if curvatures else 0,
            'max': max(curvatures) if curvatures else 0,
            'avg': np.mean(curvatures) if curvatures else 0
        }

    def _detect_sharp_features(self, curves: List, char_length: float) -> List[Dict]:
        """
        Detect sharp edges and corners (dihedral angle < 30°)

        Sharp features -> stress singularities, flow separation
        """
        zones = []

        # Get all curve adjacencies (curves shared by 2 surfaces)
        for dim, tag in curves:
            try:
                # Get surfaces adjacent to this curve
                upward, _ = gmsh.model.getAdjacencies(dim, tag)

                if len(upward) >= 2:
                    # Check dihedral angle between adjacent surfaces
                    # For now, mark all shared edges as potentially sharp
                    # (More sophisticated: compute actual angle from normals)

                    zones.append({
                        'type': 'sharp_edge',
                        'entity': (dim, tag),
                        'priority': 0.9,  # Very high priority
                        'refinement_factor': 3.0  # 3x finer near edges
                    })
            except Exception as e:
                pass

        return zones

    def _detect_thin_walls(self, surfaces: List, char_length: float) -> List[Dict]:
        """
        Detect thin sections using aspect ratio analysis

        Thin walls -> buckling risk, thermal gradients
        """
        zones = []

        for dim, tag in surfaces:
            try:
                # Get surface bounds
                bounds = gmsh.model.getParametrizationBounds(dim, tag)
                if bounds:
                    u_range = abs(bounds[0][1] - bounds[0][0])
                    v_range = abs(bounds[1][1] - bounds[1][0])

                    # Estimate aspect ratio (crude)
                    if u_range > 0 and v_range > 0:
                        aspect_ratio = max(u_range, v_range) / min(u_range, v_range)

                        # Thin wall threshold: aspect ratio > 10
                        if aspect_ratio > 10:
                            zones.append({
                                'type': 'thin_wall',
                                'entity': (dim, tag),
                                'aspect_ratio': aspect_ratio,
                                'priority': 0.6,  # Medium priority
                                'refinement_factor': 1.5  # 1.5x finer
                            })
            except Exception as e:
                pass

        return zones

    def _detect_holes(self, surfaces: List, char_length: float) -> List[Dict]:
        """
        Detect holes and cutouts using topology analysis

        Holes -> stress concentration (3x radius refinement)
        """
        zones = []

        # Detect cylindrical surfaces (common for holes)
        for dim, tag in surfaces:
            try:
                # Get surface type
                surf_type = gmsh.model.getType(dim, tag)

                # Cylindrical surfaces often indicate holes
                if 'Cylinder' in surf_type or 'Cylindrical' in surf_type:
                    zones.append({
                        'type': 'hole',
                        'entity': (dim, tag),
                        'priority': 0.85,  # High priority
                        'refinement_factor': 3.0  # 3x finer around holes
                    })
            except Exception as e:
                pass

        return zones

    def apply_refinement_field(self, refinement_zones: List[Dict],
                                base_size: float,
                                aggressive: bool = False) -> int:
        """
        Apply mesh size field based on refinement zones

        Args:
            refinement_zones: List of refinement zone dictionaries
            base_size: Base mesh size (will be refined locally)
            aggressive: If True, use more aggressive refinement (smaller elements)

        Returns:
            Field tag that was created
        """
        if not refinement_zones:
            self._log("[!] No refinement zones to apply")
            return None

        self._log(f"\n📐 Applying mesh size field with {len(refinement_zones)} zones...")

        # Create distance-based fields for each zone
        field_tags = []

        for i, zone in enumerate(refinement_zones):
            try:
                entity_dim, entity_tag = zone['entity']
                refinement_factor = zone['refinement_factor']

                if aggressive:
                    refinement_factor *= 1.5  # Even finer

                # Create distance field
                dist_field = gmsh.model.mesh.field.add("Distance")

                if entity_dim == 1:  # Curve
                    gmsh.model.mesh.field.setNumbers(dist_field, "CurvesList", [entity_tag])
                elif entity_dim == 2:  # Surface
                    gmsh.model.mesh.field.setNumbers(dist_field, "SurfacesList", [entity_tag])

                # Create threshold field (refine near entity)
                thresh_field = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(thresh_field, "InField", dist_field)
                gmsh.model.mesh.field.setNumber(thresh_field, "SizeMin", base_size / refinement_factor)
                gmsh.model.mesh.field.setNumber(thresh_field, "SizeMax", base_size)
                gmsh.model.mesh.field.setNumber(thresh_field, "DistMin", base_size * 0.5)
                gmsh.model.mesh.field.setNumber(thresh_field, "DistMax", base_size * 3.0)

                field_tags.append(thresh_field)

            except Exception as e:
                self._log(f"  [!] Failed to create field for zone {i}: {e}")

        # Combine all fields with Min (take smallest size)
        if field_tags:
            min_field = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", field_tags)
            gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

            self._log(f"[OK] Applied {len(field_tags)} refinement fields")
            self.mesh_size_field_tags.append(min_field)

            return min_field

        return None

    def _log(self, message: str):
        """Log message if verbose"""
        if self.verbose:
            print(message)


def test_geometry_heuristics():
    """Test geometry heuristics on sample CAD files"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python geometry_heuristics.py <cad_file.step>")
        sys.exit(1)

    cad_file = sys.argv[1]

    print("=" * 70)
    print("GEOMETRY HEURISTICS TEST")
    print("=" * 70)
    print(f"Input: {cad_file}\n")

    # Initialize gmsh
    gmsh.initialize()
    gmsh.open(cad_file)

    # Run analysis
    analyzer = GeometryHeuristics(verbose=True)
    results = analyzer.analyze_geometry()

    # Print detailed results
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Execution time: {results['execution_time_ms']:.1f}ms")
    print(f"\nCurvature stats:")
    print(f"  Min: {results['curvature_stats'].get('min', 0):.4f}")
    print(f"  Max: {results['curvature_stats'].get('max', 0):.4f}")
    print(f"  Avg: {results['curvature_stats'].get('avg', 0):.4f}")

    print(f"\nRefinement zones: {len(results['refinement_zones'])}")
    for i, zone in enumerate(results['refinement_zones'][:10]):  # Show first 10
        print(f"  {i+1}. {zone['type']}: priority={zone['priority']:.2f}, "
              f"refinement={zone['refinement_factor']:.1f}x")

    if len(results['refinement_zones']) > 10:
        print(f"  ... and {len(results['refinement_zones']) - 10} more")

    # Test field application
    print("\n" + "=" * 70)
    print("TESTING FIELD APPLICATION")
    print("=" * 70)

    base_size = 10.0  # mm
    field_tag = analyzer.apply_refinement_field(
        results['refinement_zones'],
        base_size=base_size,
        aggressive=False
    )

    if field_tag:
        print(f"[OK] Mesh size field applied (tag={field_tag})")
        print(f"  Base size: {base_size}mm")
        print(f"  Local refinement: up to {max(z['refinement_factor'] for z in results['refinement_zones']):.1f}x finer")

    gmsh.finalize()
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    test_geometry_heuristics()
