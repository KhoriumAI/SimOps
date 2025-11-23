#!/usr/bin/env python3
"""
Compute geometric deviation between mesh and original CAD surface.

This measures how well the mesh approximates the CAD geometry,
which is critical for evaluating mesh quality beyond element quality.
"""

import gmsh
import numpy as np
from typing import Dict


def compute_surface_deviation_sampled(cad_file: str, mesh_file: str, sample_pct: float = 0.1) -> Dict:
    """
    Compute geometric deviation by sampling mesh surface nodes.

    Args:
        cad_file: Path to CAD file
        mesh_file: Path to mesh file
        sample_pct: Percentage of nodes to sample (0.0-1.0). Default 0.1 (10%)

    Returns:
        Dict with deviation metrics including mean and max surface deviation
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    try:
        # Load CAD geometry
        gmsh.open(cad_file)
        surfaces = gmsh.model.getEntities(dim=2)

        # Load mesh and get surface nodes only
        gmsh.merge(mesh_file)
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes(dim=2)

        if len(node_tags) == 0:
            # Fallback: get all nodes
            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()

        coords = np.array(node_coords).reshape(-1, 3)

        # Sample a percentage of nodes for efficiency
        num_samples = max(100, int(len(coords) * sample_pct))
        num_samples = min(num_samples, len(coords))

        if num_samples < len(coords):
            sample_indices = np.random.choice(len(coords), num_samples, replace=False)
            sampled_coords = coords[sample_indices]
        else:
            sampled_coords = coords

        # Compute distances for sampled nodes
        deviations = []
        for coord in sampled_coords:
            min_dist = float('inf')
            for dim, tag in surfaces:
                try:
                    # Get closest point on CAD surface
                    closest = gmsh.model.getClosestPoint(dim, tag, coord.tolist())
                    if len(closest) >= 3:
                        dist = np.linalg.norm(coord - np.array(closest[:3]))
                        min_dist = min(min_dist, dist)
                except:
                    pass

            if min_dist != float('inf'):
                deviations.append(min_dist)

        gmsh.finalize()

        if len(deviations) == 0:
            return {'error': 'Could not compute surface deviations'}

        deviations = np.array(deviations)
        max_dev = float(np.max(deviations)) * 1000  # Convert to mm
        mean_dev = float(np.mean(deviations)) * 1000
        rms_dev = float(np.sqrt(np.mean(deviations**2))) * 1000

        # Geometric accuracy score: lower deviation = higher score
        # 0.01mm = 0.99, 0.1mm = 0.90, 1mm = 0.50, 10mm = 0.09
        accuracy_score = 1.0 / (1.0 + mean_dev / 10.0)

        return {
            'max_deviation_mm': max_dev,
            'mean_deviation_mm': mean_dev,
            'rms_deviation_mm': rms_dev,
            'geometric_accuracy': accuracy_score,  # 0-1, higher is better
            'nodes_sampled': len(deviations),
            'sample_pct': sample_pct * 100
        }

    except Exception as e:
        gmsh.finalize()
        return {'error': f'Failed to compute sampled deviation: {str(e)}'}


def compute_surface_deviation_fast(cad_file: str, mesh_file: str) -> Dict:
    """
    Fast approximation using volume comparison.

    This is faster but less accurate than full geometric deviation.
    Good for quick feedback during iteration.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    try:
        # Get CAD volume
        gmsh.open(cad_file)
        volumes_3d = gmsh.model.getEntities(dim=3)
        cad_volume = 0.0
        for vol_dim, vol_tag in volumes_3d:
            cad_volume += gmsh.model.occ.getMass(vol_dim, vol_tag)

        cad_bbox = gmsh.model.getBoundingBox(-1, -1)
        gmsh.clear()

        # Get mesh volume
        gmsh.merge(mesh_file)
        mesh_bbox = gmsh.model.getBoundingBox(-1, -1)
        bbox_volume = (mesh_bbox[3] - mesh_bbox[0]) * (mesh_bbox[4] - mesh_bbox[1]) * (mesh_bbox[5] - mesh_bbox[2])

        gmsh.finalize()

        # Compute volume error as proxy for accuracy
        volume_error = abs(cad_volume - bbox_volume) / cad_volume if cad_volume > 0 else 0
        accuracy_score = max(0.0, 1.0 - volume_error)  # Higher is better

        # Compute bbox diagonal for scale reference
        bbox_diag = np.sqrt(
            (cad_bbox[3] - cad_bbox[0])**2 +
            (cad_bbox[4] - cad_bbox[1])**2 +
            (cad_bbox[5] - cad_bbox[2])**2
        )

        return {
            'volume_error_pct': volume_error * 100,
            'geometric_accuracy': accuracy_score,  # Key metric for AI (0-1, higher is better)
            'cad_volume': cad_volume,
            'mesh_bbox_volume': bbox_volume,
            'bbox_diagonal': bbox_diag
        }

    except Exception as e:
        gmsh.finalize()
        return {
            'error': f'Failed to compute fast deviation: {str(e)}'
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python compute_geometric_deviation.py <cad_file> <mesh_file>")
        sys.exit(1)

    cad_file = sys.argv[1]
    mesh_file = sys.argv[2]

    print("Computing geometric deviation (fast method)...")
    fast_result = compute_surface_deviation_fast(cad_file, mesh_file)
    print(f"Volume error: {fast_result.get('volume_error_pct', 'N/A'):.2f}%")
    print(f"Geometric accuracy: {fast_result.get('geometric_accuracy', 'N/A'):.3f}")
