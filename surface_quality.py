#!/usr/bin/env python3
"""
Surface Triangle Quality Analysis
==================================

Calculate quality metrics for 2D surface triangles:
- Skewness (angle deviation from equilateral)
- Aspect ratio
- Identify worst 10% for visualization
"""

import gmsh
import numpy as np
from typing import Dict, Tuple


def calculate_triangle_quality(node_coords: np.ndarray) -> Dict[str, float]:
    """
    Calculate quality metrics for a single triangle

    Args:
        node_coords: 3x3 array of node coordinates [[x1,y1,z1], [x2,y2,z2], [x3,y3,z3]]

    Returns:
        Dictionary with quality metrics
    """
    # Extract coordinates
    p1, p2, p3 = node_coords[0], node_coords[1], node_coords[2]

    # Calculate edge vectors
    v1 = p2 - p1
    v2 = p3 - p2
    v3 = p1 - p3

    # Edge lengths
    L1 = np.linalg.norm(v1)
    L2 = np.linalg.norm(v2)
    L3 = np.linalg.norm(v3)

    # Prevent division by zero
    if L1 < 1e-12 or L2 < 1e-12 or L3 < 1e-12:
        return {
            'skewness': 1.0,  # Worst quality
            'aspect_ratio': 1000.0,
            'area': 0.0,
            'quality_score': 0.0
        }

    # Calculate angles using dot product
    angle1 = np.arccos(np.clip(np.dot(v1, -v3) / (L1 * L3), -1, 1))
    angle2 = np.arccos(np.clip(np.dot(v2, -v1) / (L2 * L1), -1, 1))
    angle3 = np.arccos(np.clip(np.dot(v3, -v2) / (L3 * L2), -1, 1))

    # Convert to degrees
    angles_deg = np.degrees([angle1, angle2, angle3])

    # Skewness: deviation from ideal 60° angles
    # 0 = perfect equilateral, 1 = worst (degenerate)
    angle_deviations = np.abs(angles_deg - 60.0)
    max_deviation = np.max(angle_deviations)
    skewness = max_deviation / 60.0  # Normalize to 0-1

    # Aspect ratio: ratio of longest to shortest edge
    # 1.0 = equilateral, higher = worse
    max_edge = max(L1, L2, L3)
    min_edge = min(L1, L2, L3)
    aspect_ratio = max_edge / min_edge

    # Triangle area using cross product
    cross = np.cross(v1, -v3)
    area = 0.5 * np.linalg.norm(cross)

    # Combined quality score (0-1, higher is better)
    # Similar to SICN for tets
    # Good triangles have:
    # - Low skewness (< 0.3)
    # - Low aspect ratio (< 2.0)
    # - Reasonable area

    # Normalize metrics to 0-1 (lower is better)
    skewness_norm = min(skewness, 1.0)
    aspect_norm = min((aspect_ratio - 1.0) / 4.0, 1.0)  # AR of 5 = normalized 1.0

    # Combined quality (higher is better, range 0-1)
    quality_score = 1.0 - (0.6 * skewness_norm + 0.4 * aspect_norm)

    return {
        'skewness': skewness,
        'aspect_ratio': aspect_ratio,
        'area': area,
        'quality_score': quality_score,
        'angles': angles_deg.tolist()
    }


def extract_surface_quality(mesh_file: str) -> Dict:
    """
    Extract quality metrics for all surface triangles in a mesh

    Args:
        mesh_file: Path to .msh file with surface mesh

    Returns:
        Dictionary containing:
        - per_element_quality: dict mapping triangle_id -> quality_score
        - quality_threshold_10: threshold for worst 10%
        - statistics: min/max/avg metrics
    """
    try:
        gmsh.initialize()
        gmsh.open(mesh_file)

        # Get surface triangles
        surf_types, surf_tags, surf_nodes = gmsh.model.mesh.getElements(dim=2)

        # Get all node coordinates
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()

        # Build node map: node_id -> [x, y, z]
        node_map = {}
        for i, tag in enumerate(node_tags):
            idx = i * 3
            node_map[int(tag)] = np.array([
                node_coords[idx],
                node_coords[idx + 1],
                node_coords[idx + 2]
            ])

        # Calculate quality for each triangle
        triangle_qualities = {}
        all_quality_scores = []
        all_skewness = []
        all_aspect_ratios = []

        for surf_type, tri_tags, tri_nodes in zip(surf_types, surf_tags, surf_nodes):
            if surf_type in [2, 9]:  # Triangle types (linear and quadratic)
                nodes_per_tri = 3 if surf_type == 2 else 6

                for i, tri_id in enumerate(tri_tags):
                    # Get first 3 nodes (corner nodes)
                    start_idx = i * nodes_per_tri
                    node_ids = tri_nodes[start_idx:start_idx+3]

                    # Get node coordinates
                    coords = np.array([node_map[int(nid)] for nid in node_ids])

                    # Calculate quality
                    quality_metrics = calculate_triangle_quality(coords)

                    # Store quality score for this triangle
                    quality_score = quality_metrics['quality_score']
                    triangle_qualities[int(tri_id)] = quality_score

                    all_quality_scores.append(quality_score)
                    all_skewness.append(quality_metrics['skewness'])
                    all_aspect_ratios.append(quality_metrics['aspect_ratio'])

        gmsh.finalize()

        # Calculate 10th percentile threshold (worst 10%)
        if all_quality_scores:
            sorted_scores = sorted(all_quality_scores)
            threshold_idx = max(0, int(len(sorted_scores) * 0.1))
            quality_threshold = sorted_scores[threshold_idx]
        else:
            quality_threshold = 0.3

        # Count elements in each quality bin
        worst_10_count = sum(1 for q in all_quality_scores if q <= quality_threshold)

        return {
            'per_element_quality': triangle_qualities,
            'quality_threshold_10': quality_threshold,
            'statistics': {
                'min_quality': min(all_quality_scores) if all_quality_scores else 0.0,
                'max_quality': max(all_quality_scores) if all_quality_scores else 1.0,
                'avg_quality': np.mean(all_quality_scores) if all_quality_scores else 0.5,
                'min_skewness': min(all_skewness) if all_skewness else 0.0,
                'max_skewness': max(all_skewness) if all_skewness else 1.0,
                'avg_skewness': np.mean(all_skewness) if all_skewness else 0.5,
                'min_aspect_ratio': min(all_aspect_ratios) if all_aspect_ratios else 1.0,
                'max_aspect_ratio': max(all_aspect_ratios) if all_aspect_ratios else 10.0,
                'avg_aspect_ratio': np.mean(all_aspect_ratios) if all_aspect_ratios else 2.0,
                'worst_10_count': worst_10_count,
                'total_triangles': len(all_quality_scores)
            }
        }

    except Exception as e:
        print(f"Error extracting surface quality: {e}")
        import traceback
        traceback.print_exc()
        return {}


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python surface_quality.py <mesh_file.msh>")
        sys.exit(1)

    mesh_file = sys.argv[1]

    print(f"\n{'='*70}")
    print("SURFACE MESH QUALITY ANALYSIS")
    print(f"{'='*70}")
    print(f"File: {mesh_file}\n")

    result = extract_surface_quality(mesh_file)

    if result:
        stats = result['statistics']
        print(f"Total triangles: {stats['total_triangles']:,}")
        print(f"Worst 10%: {stats['worst_10_count']} triangles")
        print(f"\nQuality Score (0-1, higher=better):")
        print(f"  Min: {stats['min_quality']:.3f}")
        print(f"  Avg: {stats['avg_quality']:.3f}")
        print(f"  Max: {stats['max_quality']:.3f}")
        print(f"  10th percentile threshold: {result['quality_threshold_10']:.3f}")
        print(f"\nSkewness (0-1, lower=better):")
        print(f"  Min: {stats['min_skewness']:.3f}")
        print(f"  Avg: {stats['avg_skewness']:.3f}")
        print(f"  Max: {stats['max_skewness']:.3f}")
        print(f"\nAspect Ratio (1+=, lower=better):")
        print(f"  Min: {stats['min_aspect_ratio']:.2f}")
        print(f"  Avg: {stats['avg_aspect_ratio']:.2f}")
        print(f"  Max: {stats['max_aspect_ratio']:.2f}")
        print(f"\n{'='*70}")
    else:
        print("[X] Failed to analyze mesh quality")
        sys.exit(1)
