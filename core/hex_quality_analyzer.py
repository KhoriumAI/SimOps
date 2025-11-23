"""
Hexahedral Mesh Quality Analyzer
=================================

Provides advanced quality metrics for hexahedral meshes:
- Scaled Jacobian (volume integrity, detects inverted elements)
- Skewness (angular deviation from 90 degrees)
- Edge ratio (aspect ratio)

Based on VTK quality metrics and ANSYS mesh quality standards.
"""

import numpy as np
import gmsh


def calculate_hex_metrics(nodes, elements):
    """
    Calculate quality metrics for hexahedral elements

    Args:
        nodes: (N, 3) array of vertex coordinates
        elements: (M, 8) array of node indices for each hex

    Returns:
        dict with metrics:
            - scaled_jacobian: (M,) array, range [-1, 1], >0.3 is good, <0 is inverted
            - max_skewness: (M,) array, range [0, 1], <0.5 is good
            - edge_ratio: (M,) array, aspect ratio of edges
    """
    # Get coordinates for all 8 nodes of all elements
    # Shape: (M, 8, 3)
    hex_coords = nodes[elements]

    # --- Metric 1: Scaled Jacobian (Volume Integrity) ---
    # Jacobian at hex center or corners. Simplified to center for speed.
    # J = det(Jacobian_matrix) / (edge_length_product)
    # This detects inverted elements (negative volume).

    # VTK hexahedron node ordering (right-hand rule):
    #        7------6
    #       /|     /|
    #      4------5 |
    #      | 3----|-2
    #      |/     |/
    #      0------1
    #
    # Bottom face (z=0): 0,1,2,3 (counterclockwise from above)
    # Top face (z=1):    4,5,6,7 (counterclockwise from above)

    # Vectors for local axes (approximation at center)
    v0 = hex_coords[:, 1] - hex_coords[:, 0]  # Edge x (0->1)
    v1 = hex_coords[:, 3] - hex_coords[:, 0]  # Edge y (0->3)
    v2 = hex_coords[:, 4] - hex_coords[:, 0]  # Edge z (0->4)

    # Cross product for volume calculation
    cross_prod = np.cross(v0, v1)
    # Dot product for determinant (volume)
    volumes = np.sum(cross_prod * v2, axis=1)

    # Normalize by edge lengths
    l0 = np.linalg.norm(v0, axis=1)
    l1 = np.linalg.norm(v1, axis=1)
    l2 = np.linalg.norm(v2, axis=1)

    # Range: [-1, 1]. 1 is perfect. < 0 is inverted.
    # ANSYS/VTK quality threshold: >0.3 is acceptable, >0.5 is good
    scaled_jacobian = volumes / (l0 * l1 * l2 + 1e-12)

    # --- Metric 2: Skewness (Shape Distortion) ---
    # Measures deviation from 90 degrees between edges.
    # We calculate cosines of angles between adjacent edges.

    # Normalize vectors
    v0_n = v0 / (l0[:, None] + 1e-12)
    v1_n = v1 / (l1[:, None] + 1e-12)
    v2_n = v2 / (l2[:, None] + 1e-12)

    # Dot products give cos(theta). For 90 deg, cos=0.
    # We want deviation from 0 (perfect orthogonality).
    skew1 = np.abs(np.sum(v0_n * v1_n, axis=1))  # Angle between x and y
    skew2 = np.abs(np.sum(v1_n * v2_n, axis=1))  # Angle between y and z
    skew3 = np.abs(np.sum(v2_n * v0_n, axis=1))  # Angle between z and x

    # Worst skew for each element (0 = perfect 90°, 1 = parallel/perpendicular)
    # ANSYS quality threshold: <0.5 is good, <0.8 is acceptable
    max_skew = np.maximum.reduce([skew1, skew2, skew3])

    # --- Metric 3: Edge Ratio (Aspect Ratio) ---
    # Ratio of longest edge to shortest edge
    # Also check other edges for completeness
    e01 = np.linalg.norm(hex_coords[:, 1] - hex_coords[:, 0], axis=1)  # Bottom front
    e03 = np.linalg.norm(hex_coords[:, 3] - hex_coords[:, 0], axis=1)  # Bottom left
    e04 = np.linalg.norm(hex_coords[:, 4] - hex_coords[:, 0], axis=1)  # Vertical
    e12 = np.linalg.norm(hex_coords[:, 2] - hex_coords[:, 1], axis=1)  # Bottom right
    e56 = np.linalg.norm(hex_coords[:, 6] - hex_coords[:, 5], axis=1)  # Top right
    e67 = np.linalg.norm(hex_coords[:, 7] - hex_coords[:, 6], axis=1)  # Top back

    # Find min and max edge lengths
    all_edges = np.stack([e01, e03, e04, e12, e56, e67], axis=1)
    max_edge = np.max(all_edges, axis=1)
    min_edge = np.min(all_edges, axis=1)

    # Edge ratio (aspect ratio)
    # ANSYS quality threshold: <5 is good, <10 is acceptable
    edge_ratio = max_edge / (min_edge + 1e-12)

    return {
        'scaled_jacobian': scaled_jacobian,
        'max_skewness': max_skew,
        'edge_ratio': edge_ratio
    }


def analyze_hex_mesh_quality(mesh_file: str) -> dict:
    """
    Analyze quality of hexahedral mesh from Gmsh file

    Args:
        mesh_file: Path to .msh file

    Returns:
        dict with quality statistics
    """
    try:
        # Initialize Gmsh and read mesh
        gmsh.initialize()
        gmsh.open(mesh_file)

        # Get all nodes
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        nodes = np.array(node_coords).reshape(-1, 3)

        # Create node tag to index mapping
        node_tag_to_idx = {tag: idx for idx, tag in enumerate(node_tags)}

        # Get hexahedral elements (type 5 in Gmsh)
        elem_types = gmsh.model.mesh.getElementTypes()

        hex_elements = []
        if 5 in elem_types:  # Type 5 = 8-node hexahedron
            elem_tags, elem_node_tags = gmsh.model.mesh.getElementsByType(5)
            # Reshape to (num_elements, 8)
            elem_node_tags = np.array(elem_node_tags).reshape(-1, 8)

            # Convert node tags to indices
            for elem_nodes in elem_node_tags:
                node_indices = [node_tag_to_idx[tag] for tag in elem_nodes]
                hex_elements.append(node_indices)

        gmsh.finalize()

        if not hex_elements:
            return {
                'success': False,
                'error': 'No hexahedral elements found in mesh'
            }

        hex_elements = np.array(hex_elements)

        # Calculate metrics
        metrics = calculate_hex_metrics(nodes, hex_elements)

        # Compute statistics
        scaled_jac = metrics['scaled_jacobian']
        skewness = metrics['max_skewness']
        edge_ratio = metrics['edge_ratio']

        # Count quality categories
        num_elements = len(scaled_jac)
        inverted = np.sum(scaled_jac < 0)
        poor_jac = np.sum((scaled_jac >= 0) & (scaled_jac < 0.3))
        acceptable_jac = np.sum((scaled_jac >= 0.3) & (scaled_jac < 0.5))
        good_jac = np.sum(scaled_jac >= 0.5)

        poor_skew = np.sum(skewness > 0.8)
        acceptable_skew = np.sum((skewness > 0.5) & (skewness <= 0.8))
        good_skew = np.sum(skewness <= 0.5)

        poor_aspect = np.sum(edge_ratio > 10)
        acceptable_aspect = np.sum((edge_ratio > 5) & (edge_ratio <= 10))
        good_aspect = np.sum(edge_ratio <= 5)

        return {
            'success': True,
            'num_elements': num_elements,
            'num_nodes': len(nodes),

            # Per-element metrics
            'per_element': {
                'scaled_jacobian': scaled_jac.tolist(),
                'skewness': skewness.tolist(),
                'edge_ratio': edge_ratio.tolist()
            },

            # Summary statistics
            'scaled_jacobian': {
                'min': float(np.min(scaled_jac)),
                'max': float(np.max(scaled_jac)),
                'mean': float(np.mean(scaled_jac)),
                'inverted': int(inverted),
                'poor': int(poor_jac),
                'acceptable': int(acceptable_jac),
                'good': int(good_jac),
                'inverted_pct': float(inverted / num_elements * 100),
                'poor_pct': float(poor_jac / num_elements * 100),
                'acceptable_pct': float(acceptable_jac / num_elements * 100),
                'good_pct': float(good_jac / num_elements * 100),
            },

            'skewness': {
                'min': float(np.min(skewness)),
                'max': float(np.max(skewness)),
                'mean': float(np.mean(skewness)),
                'poor': int(poor_skew),
                'acceptable': int(acceptable_skew),
                'good': int(good_skew),
                'poor_pct': float(poor_skew / num_elements * 100),
                'acceptable_pct': float(acceptable_skew / num_elements * 100),
                'good_pct': float(good_skew / num_elements * 100),
            },

            'edge_ratio': {
                'min': float(np.min(edge_ratio)),
                'max': float(np.max(edge_ratio)),
                'mean': float(np.mean(edge_ratio)),
                'poor': int(poor_aspect),
                'acceptable': int(acceptable_aspect),
                'good': int(good_aspect),
                'poor_pct': float(poor_aspect / num_elements * 100),
                'acceptable_pct': float(acceptable_aspect / num_elements * 100),
                'good_pct': float(good_aspect / num_elements * 100),
            },

            # Overall quality score (0-100)
            'overall_quality': float(
                (good_jac / num_elements * 100 * 0.5) +  # Jacobian is most important
                (good_skew / num_elements * 100 * 0.3) +  # Skewness is second
                (good_aspect / num_elements * 100 * 0.2)  # Aspect ratio is third
            )
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python hex_quality_analyzer.py <mesh_file.msh>")
        sys.exit(1)

    mesh_file = sys.argv[1]
    result = analyze_hex_mesh_quality(mesh_file)

    if result['success']:
        print("\n" + "=" * 70)
        print("HEXAHEDRAL MESH QUALITY ANALYSIS")
        print("=" * 70)
        print(f"\nMesh: {mesh_file}")
        print(f"Elements: {result['num_elements']:,}")
        print(f"Nodes: {result['num_nodes']:,}")

        print("\n--- Scaled Jacobian (Volume Integrity) ---")
        print(f"Range: [{result['scaled_jacobian']['min']:.3f}, {result['scaled_jacobian']['max']:.3f}]")
        print(f"Mean: {result['scaled_jacobian']['mean']:.3f}")
        print(f"Inverted elements (<0): {result['scaled_jacobian']['inverted']} ({result['scaled_jacobian']['inverted_pct']:.1f}%)")
        print(f"Poor (0-0.3): {result['scaled_jacobian']['poor']} ({result['scaled_jacobian']['poor_pct']:.1f}%)")
        print(f"Acceptable (0.3-0.5): {result['scaled_jacobian']['acceptable']} ({result['scaled_jacobian']['acceptable_pct']:.1f}%)")
        print(f"Good (>0.5): {result['scaled_jacobian']['good']} ({result['scaled_jacobian']['good_pct']:.1f}%)")

        print("\n--- Skewness (Angular Deviation) ---")
        print(f"Range: [{result['skewness']['min']:.3f}, {result['skewness']['max']:.3f}]")
        print(f"Mean: {result['skewness']['mean']:.3f}")
        print(f"Poor (>0.8): {result['skewness']['poor']} ({result['skewness']['poor_pct']:.1f}%)")
        print(f"Acceptable (0.5-0.8): {result['skewness']['acceptable']} ({result['skewness']['acceptable_pct']:.1f}%)")
        print(f"Good (<0.5): {result['skewness']['good']} ({result['skewness']['good_pct']:.1f}%)")

        print("\n--- Edge Ratio (Aspect Ratio) ---")
        print(f"Range: [{result['edge_ratio']['min']:.2f}, {result['edge_ratio']['max']:.2f}]")
        print(f"Mean: {result['edge_ratio']['mean']:.2f}")
        print(f"Poor (>10): {result['edge_ratio']['poor']} ({result['edge_ratio']['poor_pct']:.1f}%)")
        print(f"Acceptable (5-10): {result['edge_ratio']['acceptable']} ({result['edge_ratio']['acceptable_pct']:.1f}%)")
        print(f"Good (<5): {result['edge_ratio']['good']} ({result['edge_ratio']['good_pct']:.1f}%)")

        print("\n--- Overall Quality Score ---")
        print(f"Score: {result['overall_quality']:.1f}/100")

        # Save detailed results to JSON
        json_file = mesh_file.replace('.msh', '_hex_quality.json')
        with open(json_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nDetailed results saved to: {json_file}")

    else:
        print(f"ERROR: {result['error']}")
        sys.exit(1)
