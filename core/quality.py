"""
Mesh Quality Analysis Module
============================

Provides comprehensive quality metrics calculation for FEA meshes.
This module contains all quality analysis functions extracted from individual
mesh generators to eliminate code duplication.
"""

import gmsh
import numpy as np
from typing import Dict, List, Optional, Tuple


class MeshQualityAnalyzer:
    """Centralized mesh quality analysis for tetrahedral meshes"""

    # Element type constants
    ELEMENT_NAMES = {
        1: "Line (2-node)",
        2: "Triangle (3-node)",
        3: "Quadrangle (4-node)",
        4: "Tetrahedron (4-node)",
        5: "Hexahedron (8-node)",
        6: "Prism (6-node)",
        7: "Pyramid (5-node)",
        8: "Line (3-node)",
        9: "Triangle (6-node)",
        10: "Quadrangle (9-node)",
        11: "Tetrahedron (10-node)",
        12: "Hexahedron (27-node)",
        13: "Prism (18-node)",
        14: "Pyramid (14-node)",
        15: "Point (1-node)"
    }

    # Tetrahedral element types (linear and quadratic)
    TET_ELEMENT_TYPES = [4, 11]

    def __init__(self):
        """Initialize the quality analyzer"""
        self.cache_enabled = True
        self._node_coords_cache = {}

    def analyze_mesh(self, include_advanced_metrics: bool = False, include_cfd_metrics: bool = False) -> Optional[Dict]:
        """
        Comprehensive mesh quality analysis using GMSH's built-in quality functions

        Now uses Gmsh's native getElementQualities() for accurate metrics.
        Our custom calculations are included as 'custom_*' for comparison/debugging.

        Args:
            include_advanced_metrics: Calculate additional metrics (Jacobian, etc.)
            include_cfd_metrics: Calculate CFD-specific metrics (non-orthogonality, face pyramids, etc.)

        Returns:
            Dictionary containing all quality metrics, or None if analysis fails
        """
        try:
            # Get mesh statistics
            element_types, element_tags, node_tags = gmsh.model.mesh.getElements()


            if not element_tags:
                return None

            total_elements = sum(len(tags) for tags in element_tags)
            
            # Fast node counting using NumPy
            all_node_tags = np.concatenate(node_tags) if node_tags else np.array([], dtype=int)
            total_nodes = len(np.unique(all_node_tags))

            quality_metrics = {
                'total_elements': total_elements,
                'total_nodes': total_nodes,
                'element_types': {},
                # PRIMARY: Gmsh's built-in quality metrics (ACCURATE)
                'gmsh_sicn': None,      # Signed Inverse Condition Number
                'gmsh_gamma': None,     # inscribed/circumscribed radius ratio
                'gmsh_min_angle': None, # Minimum dihedral angle
                # SECONDARY: Our custom calculations (for comparison/debug)
                'custom_skewness': None,
                'custom_aspect_ratio': None,
                'custom_min_angle': None,
                # Legacy names point to Gmsh metrics for compatibility
                'skewness': None,
                'aspect_ratio': None,
                'min_angle': None,
                'edge_length_ratio': None
            }

            # Analyze element types
            for elem_type, tags in zip(element_types, element_tags):
                elem_name = self.get_element_name(elem_type)
                quality_metrics['element_types'][elem_name] = len(tags)

            # PRIMARY: Get Gmsh's accurate built-in quality metrics
            quality_metrics['gmsh_sicn'] = self.calculate_gmsh_sicn()
            quality_metrics['gmsh_gamma'] = self.calculate_gmsh_gamma()
            quality_metrics['gmsh_min_angle'] = self.calculate_gmsh_min_angle()

            # Map Gmsh metrics to legacy names for compatibility
            # SICN is inversely related to skewness (higher SICN = better = lower skewness)
            if quality_metrics['gmsh_sicn']:
                # Convert SICN to skewness-like metric
                # SICN range: [-inf, 1], where 1=perfect, 0=degenerate, <0=inverted
                # Skewness should be [0, 1], where 0=perfect, 1=degenerate
                # Clamp SICN to [0, 1] first (negative = inverted elements, treat as 0)
                sicn_min_clamped = max(0.0, quality_metrics['gmsh_sicn']['min'])
                sicn_max_clamped = max(0.0, quality_metrics['gmsh_sicn']['max'])
                sicn_avg_clamped = max(0.0, quality_metrics['gmsh_sicn']['avg'])

                quality_metrics['skewness'] = {
                    'min': 1.0 - sicn_max_clamped,  # Best element
                    'max': 1.0 - sicn_min_clamped,  # Worst element (clamped to 1.0 max)
                    'avg': 1.0 - sicn_avg_clamped
                }

            if quality_metrics['gmsh_gamma']:
                # Gamma is quality (0-1, higher better), convert to aspect-ratio-like (lower better)
                # Approximate: aspect_ratio ~= 1/gamma
                # Clamp to avoid division by zero and unrealistic values
                gamma_min_safe = max(quality_metrics['gmsh_gamma']['min'], 0.001)
                gamma_max_safe = max(quality_metrics['gmsh_gamma']['max'], 0.001)
                gamma_avg_safe = max(quality_metrics['gmsh_gamma']['avg'], 0.001)

                quality_metrics['aspect_ratio'] = {
                    'min': 1.0 / gamma_max_safe,
                    'max': min(1.0 / gamma_min_safe, 100.0),  # Cap at 100
                    'avg': 1.0 / gamma_avg_safe
                }

            quality_metrics['min_angle'] = quality_metrics['gmsh_min_angle']

            # SECONDARY: Calculate our custom metrics for comparison/debugging
            quality_metrics['custom_skewness'] = self.calculate_skewness_metrics()
            quality_metrics['custom_aspect_ratio'] = self.calculate_aspect_ratio_metrics()
            quality_metrics['custom_min_angle'] = self.calculate_min_angle_metrics()
            quality_metrics['edge_length_ratio'] = self.calculate_edge_length_ratio_metrics()

            if include_advanced_metrics:
                quality_metrics['jacobian'] = self.calculate_jacobian_metrics()
                quality_metrics['volume_ratio'] = self.calculate_volume_ratio_metrics()

            # CFD-SPECIFIC: OpenFOAM checkMesh-equivalent metrics
            if include_cfd_metrics:
                try:
                    from .cfd_quality import CFDQualityAnalyzer
                    cfd_analyzer = CFDQualityAnalyzer(verbose=False)
                    cfd_report = cfd_analyzer.analyze_current_mesh()
                    quality_metrics['cfd'] = cfd_report.to_dict()
                except Exception as cfd_err:
                    print(f"CFD metrics calculation failed: {cfd_err}")
                    quality_metrics['cfd'] = None

            return quality_metrics


        except Exception as e:
            print(f"Quality analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def calculate_gmsh_sicn(self) -> Optional[Dict[str, float]]:
        """
        Calculate Gmsh's SICN (Signed Inverse Condition Number) quality metric

        SICN is in range [0, 1], where 1 = perfect quality, 0 = degenerate
        This is Gmsh's primary quality measure.

        Target: > 0.3 for acceptable quality, > 0.5 for good quality

        Returns:
            Dictionary with min, max, avg SICN values
        """
        try:
            element_types, element_tags, node_tags = gmsh.model.mesh.getElements(dim=3)

            all_sicn = []
            for elem_type, tags in zip(element_types, element_tags):
                if elem_type in self.TET_ELEMENT_TYPES:
                    # Get SICN for these elements
                    qualities = gmsh.model.mesh.getElementQualities(tags, "minSICN")
                    all_sicn.extend(qualities)

            if all_sicn:
                return {
                    'min': float(np.min(all_sicn)),
                    'max': float(np.max(all_sicn)),
                    'avg': float(np.mean(all_sicn)),
                    'std': float(np.std(all_sicn)),
                    'count': len(all_sicn)
                }
            return None

        except Exception as e:
            print(f"Gmsh SICN calculation failed: {e}")
            return None

    def calculate_gmsh_gamma(self) -> Optional[Dict[str, float]]:
        """
        Calculate Gmsh's Gamma quality metric

        Gamma = inscribed_radius / circumscribed_radius
        Range [0, 1], where 1 = perfect (equilateral), 0 = degenerate

        Target: > 0.2 for acceptable quality, > 0.4 for good quality

        Returns:
            Dictionary with min, max, avg Gamma values
        """
        try:
            element_types, element_tags, node_tags = gmsh.model.mesh.getElements(dim=3)

            all_gamma = []
            for elem_type, tags in zip(element_types, element_tags):
                if elem_type in self.TET_ELEMENT_TYPES:
                    qualities = gmsh.model.mesh.getElementQualities(tags, "gamma")
                    all_gamma.extend(qualities)

            if all_gamma:
                return {
                    'min': float(np.min(all_gamma)),
                    'max': float(np.max(all_gamma)),
                    'avg': float(np.mean(all_gamma)),
                    'std': float(np.std(all_gamma)),
                    'count': len(all_gamma)
                }
            return None

        except Exception as e:
            print(f"Gmsh Gamma calculation failed: {e}")
            return None

    def calculate_gmsh_min_angle(self) -> Optional[Dict[str, float]]:
        """
        Calculate minimum dihedral angles using Gmsh

        Returns minimum angle for each tet in degrees.
        Target: > 10 degrees for acceptable quality

        Returns:
            Dictionary with min, max, avg angles
        """
        try:
            element_types, element_tags, node_tags = gmsh.model.mesh.getElements(dim=3)

            all_angles = []
            for elem_type, tags in zip(element_types, element_tags):
                if elem_type in self.TET_ELEMENT_TYPES:
                    # Gmsh's angleShape returns minimum dihedral angle
                    qualities = gmsh.model.mesh.getElementQualities(tags, "angleShape")
                    all_angles.extend(qualities)

            if all_angles:
                return {
                    'min': float(np.min(all_angles)),
                    'max': float(np.max(all_angles)),
                    'avg': float(np.mean(all_angles)),
                    'std': float(np.std(all_angles)),
                    'count': len(all_angles)
                }
            return None

        except Exception as e:
            print(f"Gmsh angle calculation failed: {e}")
            return None

    def calculate_skewness_metrics(self) -> Optional[Dict[str, float]]:
        """
        Calculate skewness metrics for tetrahedral elements

        NOTE: This is our CUSTOM calculation (potentially buggy).
        For accurate metrics, use gmsh_sicn instead.

        Skewness measures deviation from ideal tetrahedral angles.
        Lower values (closer to 0) indicate better quality.
        Target: < 0.7 for good quality

        Returns:
            Dictionary with min, max, avg skewness values
        """
        try:
            element_types, element_tags, node_tags = gmsh.model.mesh.getElements(dim=3)

            skewness_values = []
            for i, (elem_type, tags) in enumerate(zip(element_types, element_tags)):
                if elem_type in self.TET_ELEMENT_TYPES:
                    values = self._calculate_tetrahedron_skewness(tags, node_tags[i])
                    skewness_values.extend(values)

            if skewness_values:
                return {
                    'min': float(np.min(skewness_values)),
                    'max': float(np.max(skewness_values)),
                    'avg': float(np.mean(skewness_values)),
                    'std': float(np.std(skewness_values)),
                    'count': len(skewness_values)
                }
            return None

        except Exception as e:
            print(f"Skewness calculation failed: {e}")
            return None

    def calculate_aspect_ratio_metrics(self) -> Optional[Dict[str, float]]:
        """
        Calculate aspect ratio metrics for tetrahedral elements

        Aspect ratio is the ratio of longest edge to shortest edge.
        Lower values (closer to 1) indicate better quality.
        Target: < 5.0 for good quality

        Returns:
            Dictionary with min, max, avg aspect ratio values
        """
        try:
            element_types, element_tags, node_tags = gmsh.model.mesh.getElements(dim=3)

            aspect_ratios = []
            for i, (elem_type, tags) in enumerate(zip(element_types, element_tags)):
                if elem_type in self.TET_ELEMENT_TYPES:
                    values = self._calculate_tetrahedron_aspect_ratio(tags, node_tags[i])
                    aspect_ratios.extend(values)

            if aspect_ratios:
                return {
                    'min': float(np.min(aspect_ratios)),
                    'max': float(np.max(aspect_ratios)),
                    'avg': float(np.mean(aspect_ratios)),
                    'std': float(np.std(aspect_ratios)),
                    'count': len(aspect_ratios)
                }
            return None

        except Exception as e:
            print(f"Aspect ratio calculation failed: {e}")
            return None

    def calculate_min_angle_metrics(self) -> Optional[Dict[str, float]]:
        """
        Calculate minimum angle metrics for tetrahedral elements

        Minimum dihedral angle in degrees.
        Higher values indicate better quality.
        Target: > 10 degrees for acceptable quality

        Returns:
            Dictionary with min, max, avg minimum angle values
        """
        try:
            element_types, element_tags, node_tags = gmsh.model.mesh.getElements(dim=3)

            min_angles = []
            for i, (elem_type, tags) in enumerate(zip(element_types, element_tags)):
                if elem_type in self.TET_ELEMENT_TYPES:
                    values = self._calculate_tetrahedron_min_angle(tags, node_tags[i])
                    min_angles.extend(values)

            if min_angles:
                return {
                    'min': float(np.min(min_angles)),
                    'max': float(np.max(min_angles)),
                    'avg': float(np.mean(min_angles)),
                    'std': float(np.std(min_angles)),
                    'count': len(min_angles)
                }
            return None

        except Exception as e:
            print(f"Minimum angle calculation failed: {e}")
            return None

    def calculate_edge_length_ratio_metrics(self) -> Optional[Dict[str, float]]:
        """
        Calculate edge length ratio metrics for tetrahedral elements

        Ratio of maximum to minimum edge length in neighborhood.
        Lower values indicate more uniform mesh.

        Returns:
            Dictionary with min, max, avg edge length ratio values
        """
        try:
            element_types, element_tags, node_tags = gmsh.model.mesh.getElements(dim=3)

            edge_ratios = []
            for i, (elem_type, tags) in enumerate(zip(element_types, element_tags)):
                if elem_type in self.TET_ELEMENT_TYPES:
                    values = self._calculate_tetrahedron_edge_ratio(tags, node_tags[i])
                    edge_ratios.extend(values)

            if edge_ratios:
                return {
                    'min': float(np.min(edge_ratios)),
                    'max': float(np.max(edge_ratios)),
                    'avg': float(np.mean(edge_ratios)),
                    'std': float(np.std(edge_ratios)),
                    'count': len(edge_ratios)
                }
            return None

        except Exception as e:
            print(f"Edge length ratio calculation failed: {e}")
            return None

    def calculate_jacobian_metrics(self) -> Optional[Dict[str, float]]:
        """
        Calculate Jacobian determinant metrics for tetrahedral elements

        Positive Jacobian indicates valid element orientation.
        Target: All elements with positive Jacobian > 0

        Returns:
            Dictionary with min, max, avg Jacobian values
        """
        try:
            element_types, element_tags, node_tags = gmsh.model.mesh.getElements(dim=3)

            jacobians = []
            for i, (elem_type, tags) in enumerate(zip(element_types, element_tags)):
                if elem_type in self.TET_ELEMENT_TYPES:
                    values = self._calculate_tetrahedron_jacobian(tags, node_tags[i])
                    jacobians.extend(values)

            if jacobians:
                return {
                    'min': float(np.min(jacobians)),
                    'max': float(np.max(jacobians)),
                    'avg': float(np.mean(jacobians)),
                    'negative_count': sum(1 for j in jacobians if j < 0),
                    'count': len(jacobians)
                }
            return None

        except Exception as e:
            print(f"Jacobian calculation failed: {e}")
            return None

    def calculate_volume_ratio_metrics(self) -> Optional[Dict[str, float]]:
        """
        Calculate volume ratio metrics between adjacent elements

        Measures how uniformly sized elements are.
        Values closer to 1.0 indicate more uniform mesh.

        Returns:
            Dictionary with volume ratio statistics
        """
        try:
            element_types, element_tags, node_tags = gmsh.model.mesh.getElements(dim=3)

            volumes = []
            for i, (elem_type, tags) in enumerate(zip(element_types, element_tags)):
                if elem_type in self.TET_ELEMENT_TYPES:
                    values = self._calculate_tetrahedron_volumes(tags, node_tags[i])
                    volumes.extend(values)

            if volumes and len(volumes) > 1:
                volumes = np.array(volumes)
                max_vol = np.max(volumes)
                min_vol = np.min(volumes)
                return {
                    'min_volume': float(min_vol),
                    'max_volume': float(max_vol),
                    'avg_volume': float(np.mean(volumes)),
                    'volume_ratio': float(max_vol / min_vol if min_vol > 0 else float('inf')),
                    'count': len(volumes)
                }
            return None

        except Exception as e:
            print(f"Volume ratio calculation failed: {e}")
            return None

    # Private helper methods for individual element calculations

    def _get_node_coords(self, node_tags: List[int]) -> Dict[int, np.ndarray]:
        """Get node coordinates with batch fetching for performance
        
        CRITICAL FIX: Uses gmsh.model.mesh.getNodes() to fetch ALL nodes in one call
        instead of calling getNode() 38K+ times individually.
        This reduces quality calculation from ~55s to ~1s for large meshes.
        """
        node_coords = {}
        
        # Check cache first - return early if all nodes are cached
        if self.cache_enabled and self._node_coords_cache:
            uncached_tags = [t for t in node_tags if t not in self._node_coords_cache]
            if not uncached_tags:
                # All nodes already cached
                return {tag: self._node_coords_cache[tag] for tag in node_tags if tag in self._node_coords_cache}
        
        try:
            # BATCH FETCH: Get all nodes in one API call
            all_node_tags, all_coords, _ = gmsh.model.mesh.getNodes()
            
            # Reshape coordinates: flat array [x1,y1,z1,x2,y2,z2,...] -> (N, 3)
            coords_reshaped = np.array(all_coords).reshape(-1, 3)
            
            # Build lookup dictionary: tag -> coordinates
            all_node_dict = {int(tag): coords_reshaped[i] for i, tag in enumerate(all_node_tags)}
            
            # Cache all fetched nodes
            if self.cache_enabled:
                self._node_coords_cache.update(all_node_dict)
            
            # Return only requested nodes
            for tag in node_tags:
                if tag in all_node_dict:
                    node_coords[tag] = all_node_dict[tag]
                    
        except Exception as e:
            # Fallback to individual fetch if batch fails (shouldn't happen)
            print(f"[QUALITY] Batch node fetch failed, using fallback: {e}")
            for tag in node_tags:
                if self.cache_enabled and tag in self._node_coords_cache:
                    node_coords[tag] = self._node_coords_cache[tag]
                else:
                    try:
                        coords = gmsh.model.mesh.getNode(tag)
                        if coords:
                            coord_array = np.array(coords[0])
                            node_coords[tag] = coord_array
                            if self.cache_enabled:
                                self._node_coords_cache[tag] = coord_array
                    except:
                        continue
        
        return node_coords

    def _calculate_tetrahedron_skewness(self, element_tags, node_tags) -> List[float]:
        """
        Calculate skewness for tetrahedral elements

        IMPORTANT: Handles both linear (4-node) and quadratic (10-node) tets.
        For quality metrics, we only use the 4 corner nodes (first 4 nodes).
        """
        skewness_values = []
        node_coords = self._get_node_coords(node_tags)

        # Determine nodes per element based on total nodes
        # Linear tet: 4 nodes, Quadratic tet: 10 nodes
        num_elements = len(element_tags)
        nodes_per_element = len(node_tags) // num_elements if num_elements > 0 else 4

        for i in range(num_elements):
            try:
                # Get the correct slice for this element
                start_idx = i * nodes_per_element
                end_idx = start_idx + nodes_per_element
                elem_nodes = node_tags[start_idx:end_idx]

                # Use only the first 4 corner nodes (works for both linear and quadratic)
                if len(elem_nodes) >= 4:
                    corner_nodes = elem_nodes[:4]
                    coords = [node_coords[node] for node in corner_nodes if node in node_coords]

                    if len(coords) == 4:
                        skewness = self._tetrahedron_skewness(coords)
                        if skewness is not None:
                            skewness_values.append(skewness)
            except:
                continue

        return skewness_values

    def _tetrahedron_skewness(self, coords: List[np.ndarray]) -> Optional[float]:
        """Calculate skewness for a single tetrahedron"""
        try:
            p0, p1, p2, p3 = coords

            # Calculate edge vectors
            v01 = p1 - p0
            v02 = p2 - p0
            v03 = p3 - p0

            # Calculate face normals
            n0 = np.cross(v01, v02)
            n1 = np.cross(v02, v03)
            n2 = np.cross(v03, v01)
            n3 = np.cross(v02 - v01, v03 - v01)

            # Normalize normals
            normals = []
            for n in [n0, n1, n2, n3]:
                norm = np.linalg.norm(n)
                if norm == 0:
                    return None
                normals.append(n / norm)

            # Calculate dihedral angles
            angles = []
            for i in range(len(normals)):
                for j in range(i+1, len(normals)):
                    cos_angle = np.dot(normals[i], normals[j])
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle = np.arccos(abs(cos_angle))
                    angles.append(angle)

            # Calculate skewness as deviation from ideal angle
            ideal_angle = np.arccos(1/3)
            max_deviation = max(abs(angle - ideal_angle) for angle in angles)
            skewness = max_deviation / ideal_angle

            return min(skewness, 1.0)

        except Exception:
            return None

    def _calculate_tetrahedron_aspect_ratio(self, element_tags, node_tags) -> List[float]:
        """
        Calculate aspect ratio for tetrahedral elements

        IMPORTANT: Handles both linear (4-node) and quadratic (10-node) tets.
        For quality metrics, we only use the 4 corner nodes (first 4 nodes).
        """
        aspect_ratios = []
        node_coords = self._get_node_coords(node_tags)

        # Determine nodes per element based on total nodes
        # Linear tet: 4 nodes, Quadratic tet: 10 nodes
        num_elements = len(element_tags)
        nodes_per_element = len(node_tags) // num_elements if num_elements > 0 else 4

        for i in range(num_elements):
            try:
                # Get the correct slice for this element
                start_idx = i * nodes_per_element
                end_idx = start_idx + nodes_per_element
                elem_nodes = node_tags[start_idx:end_idx]

                # Use only the first 4 corner nodes (works for both linear and quadratic)
                if len(elem_nodes) >= 4:
                    corner_nodes = elem_nodes[:4]
                    coords = [node_coords[node] for node in corner_nodes if node in node_coords]

                    if len(coords) == 4:
                        aspect_ratio = self._tetrahedron_aspect_ratio(coords)
                        if aspect_ratio is not None:
                            aspect_ratios.append(aspect_ratio)
            except:
                continue

        return aspect_ratios

    def _tetrahedron_aspect_ratio(self, coords: List[np.ndarray]) -> Optional[float]:
        """Calculate aspect ratio for a single tetrahedron"""
        try:
            p0, p1, p2, p3 = coords

            # Calculate edge lengths
            edges = [
                np.linalg.norm(p1 - p0),
                np.linalg.norm(p2 - p0),
                np.linalg.norm(p3 - p0),
                np.linalg.norm(p2 - p1),
                np.linalg.norm(p3 - p1),
                np.linalg.norm(p3 - p2)
            ]

            max_edge = max(edges)
            min_edge = min(edges)

            if min_edge <= 0:
                return None

            return max_edge / min_edge

        except Exception:
            return None

    def _calculate_tetrahedron_min_angle(self, element_tags, node_tags) -> List[float]:
        """
        Calculate minimum dihedral angle for tetrahedral elements

        IMPORTANT: Handles both linear (4-node) and quadratic (10-node) tets.
        For quality metrics, we only use the 4 corner nodes (first 4 nodes).
        """
        min_angles = []
        node_coords = self._get_node_coords(node_tags)

        # Determine nodes per element based on total nodes
        # Linear tet: 4 nodes, Quadratic tet: 10 nodes
        num_elements = len(element_tags)
        nodes_per_element = len(node_tags) // num_elements if num_elements > 0 else 4

        for i in range(num_elements):
            try:
                # Get the correct slice for this element
                start_idx = i * nodes_per_element
                end_idx = start_idx + nodes_per_element
                elem_nodes = node_tags[start_idx:end_idx]

                # Use only the first 4 corner nodes (works for both linear and quadratic)
                if len(elem_nodes) >= 4:
                    corner_nodes = elem_nodes[:4]
                    coords = [node_coords[node] for node in corner_nodes if node in node_coords]

                    if len(coords) == 4:
                        min_angle = self._tetrahedron_min_angle_single(coords)
                        if min_angle is not None:
                            min_angles.append(min_angle)
            except:
                continue

        return min_angles

    def _tetrahedron_min_angle_single(self, coords: List[np.ndarray]) -> Optional[float]:
        """Calculate minimum dihedral angle for a single tetrahedron (in degrees)"""
        try:
            p0, p1, p2, p3 = coords

            # Calculate edge vectors
            edges = [
                (p1 - p0, p2 - p0),
                (p1 - p0, p3 - p0),
                (p2 - p0, p3 - p0),
                (p2 - p1, p3 - p1)
            ]

            angles = []
            for e1, e2 in edges:
                norm1 = np.linalg.norm(e1)
                norm2 = np.linalg.norm(e2)
                if norm1 > 0 and norm2 > 0:
                    cos_angle = np.dot(e1, e2) / (norm1 * norm2)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle = np.arccos(cos_angle)
                    angles.append(np.degrees(angle))

            return min(angles) if angles else None

        except Exception:
            return None

    def _calculate_tetrahedron_edge_ratio(self, element_tags, node_tags) -> List[float]:
        """Calculate edge length ratios for tetrahedral elements"""
        # For now, this is the same as aspect ratio
        # Could be enhanced to compare with neighboring elements
        return self._calculate_tetrahedron_aspect_ratio(element_tags, node_tags)

    def _calculate_tetrahedron_jacobian(self, element_tags, node_tags) -> List[float]:
        """
        Calculate Jacobian determinants for tetrahedral elements

        IMPORTANT: Handles both linear (4-node) and quadratic (10-node) tets.
        For quality metrics, we only use the 4 corner nodes (first 4 nodes).
        """
        jacobians = []
        node_coords = self._get_node_coords(node_tags)

        # Determine nodes per element based on total nodes
        # Linear tet: 4 nodes, Quadratic tet: 10 nodes
        num_elements = len(element_tags)
        nodes_per_element = len(node_tags) // num_elements if num_elements > 0 else 4

        for i in range(num_elements):
            try:
                # Get the correct slice for this element
                start_idx = i * nodes_per_element
                end_idx = start_idx + nodes_per_element
                elem_nodes = node_tags[start_idx:end_idx]

                # Use only the first 4 corner nodes (works for both linear and quadratic)
                if len(elem_nodes) >= 4:
                    corner_nodes = elem_nodes[:4]
                    coords = [node_coords[node] for node in corner_nodes if node in node_coords]

                    if len(coords) == 4:
                        jacobian = self._tetrahedron_jacobian_single(coords)
                        if jacobian is not None:
                            jacobians.append(jacobian)
            except:
                continue

        return jacobians

    def _tetrahedron_jacobian_single(self, coords: List[np.ndarray]) -> Optional[float]:
        """Calculate Jacobian determinant for a single tetrahedron"""
        try:
            p0, p1, p2, p3 = coords

            # Calculate edge vectors from p0
            v1 = p1 - p0
            v2 = p2 - p0
            v3 = p3 - p0

            # Jacobian matrix
            J = np.column_stack([v1, v2, v3])

            # Determinant
            det = np.linalg.det(J)

            return det

        except Exception:
            return None

    def _calculate_tetrahedron_volumes(self, element_tags, node_tags) -> List[float]:
        """
        Calculate volumes for tetrahedral elements

        IMPORTANT: Handles both linear (4-node) and quadratic (10-node) tets.
        For quality metrics, we only use the 4 corner nodes (first 4 nodes).
        """
        volumes = []
        node_coords = self._get_node_coords(node_tags)

        # Determine nodes per element based on total nodes
        # Linear tet: 4 nodes, Quadratic tet: 10 nodes
        num_elements = len(element_tags)
        nodes_per_element = len(node_tags) // num_elements if num_elements > 0 else 4

        for i in range(num_elements):
            try:
                # Get the correct slice for this element
                start_idx = i * nodes_per_element
                end_idx = start_idx + nodes_per_element
                elem_nodes = node_tags[start_idx:end_idx]

                # Use only the first 4 corner nodes (works for both linear and quadratic)
                if len(elem_nodes) >= 4:
                    corner_nodes = elem_nodes[:4]
                    coords = [node_coords[node] for node in corner_nodes if node in node_coords]

                    if len(coords) == 4:
                        volume = self._tetrahedron_volume(coords)
                        if volume is not None and volume > 0:
                            volumes.append(volume)
            except:
                continue

        return volumes

    def _tetrahedron_volume(self, coords: List[np.ndarray]) -> Optional[float]:
        """Calculate volume for a single tetrahedron"""
        try:
            p0, p1, p2, p3 = coords

            # Volume = |det(v1, v2, v3)| / 6
            v1 = p1 - p0
            v2 = p2 - p0
            v3 = p3 - p0

            J = np.column_stack([v1, v2, v3])
            volume = abs(np.linalg.det(J)) / 6.0

            return volume

        except Exception:
            return None

    @staticmethod
    def get_element_name(elem_type: int) -> str:
        """Get human-readable element type name"""
        return MeshQualityAnalyzer.ELEMENT_NAMES.get(elem_type, f"Unknown ({elem_type})")

    def clear_cache(self):
        """Clear the node coordinates cache"""
        self._node_coords_cache.clear()

    def format_quality_report(self, metrics: Dict, detailed: bool = True) -> str:
        """
        Format quality metrics into a human-readable report

        Args:
            metrics: Quality metrics dictionary from analyze_mesh()
            detailed: Include detailed statistics

        Returns:
            Formatted report string
        """
        if not metrics:
            return "No quality metrics available"

        report = []
        report.append("=" * 60)
        report.append("MESH QUALITY REPORT")
        report.append("=" * 60)

        # Mesh statistics
        report.append(f"\nMesh Statistics:")
        report.append(f"  Total Elements: {metrics['total_elements']:,}")
        report.append(f"  Total Nodes: {metrics['total_nodes']:,}")

        if detailed and metrics['element_types']:
            report.append(f"\nElement Types:")
            for elem_name, count in metrics['element_types'].items():
                report.append(f"  {elem_name}: {count:,}")

        # Quality metrics
        report.append(f"\nQuality Metrics:")

        if metrics['skewness']:
            s = metrics['skewness']
            status = "[OK]" if s['max'] <= 0.7 else "[!]" if s['max'] <= 0.85 else "[X]"
            report.append(f"  {status} Skewness:")
            report.append(f"      Min: {s['min']:.4f}, Max: {s['max']:.4f}, Avg: {s['avg']:.4f}")
            if detailed:
                report.append(f"      Std: {s['std']:.4f}, Count: {s['count']:,}")

        if metrics['aspect_ratio']:
            a = metrics['aspect_ratio']
            status = "[OK]" if a['max'] <= 5.0 else "[!]" if a['max'] <= 10.0 else "[X]"
            report.append(f"  {status} Aspect Ratio:")
            report.append(f"      Min: {a['min']:.4f}, Max: {a['max']:.4f}, Avg: {a['avg']:.4f}")
            if detailed:
                report.append(f"      Std: {a['std']:.4f}, Count: {a['count']:,}")

        if metrics.get('min_angle'):
            m = metrics['min_angle']
            status = "[OK]" if m['min'] >= 10.0 else "[!]" if m['min'] >= 5.0 else "[X]"
            report.append(f"  {status} Minimum Angle (degrees):")
            report.append(f"      Min: {m['min']:.2f}, Max: {m['max']:.2f}, Avg: {m['avg']:.2f}")

        if metrics.get('jacobian'):
            j = metrics['jacobian']
            status = "[OK]" if j['negative_count'] == 0 else "[X]"
            report.append(f"  {status} Jacobian:")
            report.append(f"      Min: {j['min']:.6f}, Max: {j['max']:.6f}, Avg: {j['avg']:.6f}")
            if j['negative_count'] > 0:
                report.append(f"      WARNING: {j['negative_count']} elements with negative Jacobian!")

        if metrics.get('volume_ratio'):
            v = metrics['volume_ratio']
            report.append(f"  Volume Statistics:")
            report.append(f"      Min: {v['min_volume']:.6f}, Max: {v['max_volume']:.6f}")
            report.append(f"      Ratio: {v['volume_ratio']:.2f}")

        report.append("=" * 60)

        return "\n".join(report)
