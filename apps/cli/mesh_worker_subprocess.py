#!/usr/bin/env python3
"""
Subprocess-based mesh generation worker

Runs mesh generation in a separate process to avoid gmsh threading issues.
Gmsh uses signals internally which only work in the main thread.
"""

import sys
import json
from pathlib import Path
from typing import Dict

# Add project root to path
# From apps/cli/, go up 2 levels to MeshPackageLean/
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from strategies.exhaustive_strategy import ExhaustiveMeshGenerator
from core.config import Config


def generate_mesh(cad_file: str, output_dir: str = None) -> Dict:
    """
    Generate mesh in subprocess

    Args:
        cad_file: Path to CAD file
        output_dir: Optional output directory

    Returns:
        Dict with success status and results
    """
    try:
        # Initialize generator
        config = Config()
        generator = ExhaustiveMeshGenerator(config)

        # Determine output folders (organized structure)
        mesh_folder = Path(__file__).parent / "generated_meshes"
        mesh_folder.mkdir(exist_ok=True)

        # Determine output file - always use generated_meshes folder
        output_file = str(mesh_folder / Path(cad_file).stem) + "_mesh.msh"

        # Generate mesh
        result = generator.generate_mesh(cad_file, output_file)

        if result.success:
            # Get best attempt metrics
            best_attempt = min(
                generator.all_attempts,
                key=lambda x: x.get('score', float('inf'))
            ) if generator.all_attempts else {}

            # Get metrics
            metrics = best_attempt.get('metrics', result.quality_metrics or {})

            # Convert relative path to absolute path
            absolute_output_file = str(Path(result.output_file).resolve().absolute())

            # Extract per-element quality for visualization
            per_element_quality = {}
            quality_metrics = {}
            
            # Try to get quality metrics from best attempt
            if 'gmsh_sicn' in metrics:
                quality_metrics['sicn_min'] = metrics['gmsh_sicn'].get('min', 0)
                quality_metrics['sicn_avg'] = metrics['gmsh_sicn'].get('avg', 0)
                quality_metrics['sicn_max'] = metrics['gmsh_sicn'].get('max', 1)
                quality_metrics['sicn_10_percentile'] = metrics.get('quality_10_percentile', 0.3)
            
            if 'gmsh_gamma' in metrics:
                quality_metrics['gamma_min'] = metrics['gmsh_gamma'].get('min', 0)
                quality_metrics['gamma_avg'] = metrics['gmsh_gamma'].get('avg', 0)
                quality_metrics['gamma_max'] = metrics['gmsh_gamma'].get('max', 1)
            
            if 'skewness' in metrics:
                quality_metrics['skewness_min'] = metrics['skewness'].get('min', 0)
                quality_metrics['skewness_avg'] = metrics['skewness'].get('avg', 0)
                quality_metrics['skewness_max'] = metrics['skewness'].get('max', 1)
            
            if 'aspect_ratio' in metrics:
                quality_metrics['aspect_ratio_min'] = metrics['aspect_ratio'].get('min', 1)
                quality_metrics['aspect_ratio_avg'] = metrics['aspect_ratio'].get('avg', 1)
                quality_metrics['aspect_ratio_max'] = metrics['aspect_ratio'].get('max', 1)

            # Try to extract per-element quality from the mesh file
            try:
                import gmsh
                gmsh.initialize()
                gmsh.open(absolute_output_file)
                
                # Get 2D elements (triangles) which are displayed
                element_types, element_tags, node_tags = gmsh.model.mesh.getElements(2)
                
                # Get quality for each triangle
                all_qualities = []
                for elem_type, tags, nodes in zip(element_types, element_tags, node_tags):
                    if elem_type == 2:  # Triangle
                        qualities = gmsh.model.mesh.getElementQualities(tags.tolist(), "minSICN")
                        for tag, q in zip(tags, qualities):
                            per_element_quality[int(tag)] = float(q)
                            all_qualities.append(q)
                
                # Calculate 10th percentile as threshold
                if all_qualities:
                    all_qualities.sort()
                    idx_10 = int(len(all_qualities) * 0.10)
                    quality_metrics['sicn_10_percentile'] = all_qualities[idx_10]
                
                gmsh.finalize()
            except Exception as e:
                print(f"Could not extract per-element quality: {e}")

            return {
                'success': True,
                'output_file': absolute_output_file,  # ABSOLUTE path for GUI
                'metrics': metrics,
                'quality_metrics': quality_metrics,  # Flattened metrics for GUI
                'per_element_quality': per_element_quality,  # Per-triangle quality
                'strategy': best_attempt.get('strategy', 'unknown'),
                'score': best_attempt.get('score', 0),
                'message': result.message,
                'total_elements': metrics.get('total_elements', 0),
                'total_nodes': metrics.get('total_nodes', 0)
            }
        else:
            return {
                'success': False,
                'error': result.message or 'Mesh generation failed'
            }

    except Exception as e:
        import traceback
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


if __name__ == "__main__":
    # Read arguments from command line
    if len(sys.argv) < 2:
        print(json.dumps({'success': False, 'error': 'No CAD file specified'}))
        sys.exit(1)

    cad_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    # Generate mesh
    result = generate_mesh(cad_file, output_dir)

    # Output result as JSON
    print(json.dumps(result))
