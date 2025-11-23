#!/usr/bin/env python3
"""
Subprocess-based mesh generation worker

Runs mesh generation in a separate process to avoid gmsh threading issues.
Gmsh uses signals internally which only work in the main thread.
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict

# Add project root to path
# From apps/cli/, go up 2 levels to MeshPackageLean/
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from strategies.exhaustive_strategy import ExhaustiveMeshGenerator
from core.config import Config


def generate_mesh(cad_file: str, output_dir: str = None, quality_params: Dict = None) -> Dict:
    """
    Generate mesh in subprocess

    Args:
        cad_file: Path to CAD file
        output_dir: Optional output directory
        quality_params: Optional dictionary of quality parameters (including painted regions)

    Returns:
        Dict with success status and results
    """
    try:
        # Initialize generator
        config = Config()
        
        # Apply quality params to config
        if quality_params:
            # Inject painted regions directly into config object (monkey-patching)
            # This allows the generator to access it without changing Config structure definition
            if 'painted_regions' in quality_params:
                config.painted_regions = quality_params['painted_regions']
                print(f"[DEBUG] Injected {len(config.painted_regions)} painted regions into config")
                
            # Update other mesh parameters if present
            if 'quality_preset' in quality_params:
                print(f"[DEBUG] Using quality preset: {quality_params['quality_preset']}")
                
            # Update target elements if present (could be used by generator)
            if 'target_elements' in quality_params:
                config.target_elements = quality_params['target_elements']

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
            # For now, create a simple quality mapping based on aggregate metrics
            # In future: extract this directly from gmsh before finalization
            per_element_quality = {}
            quality_metrics = {}
            
            #Try to get quality metrics from best attempt
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

            # Extract per-element quality by re-opening the saved mesh
            # This needs to be done AFTER gmsh.finalize() in the mesh generator
            try:
                import gmsh as gmsh_reload
                import numpy as np
                
                print("[DEBUG] Attempting to extract per-element quality...")
                gmsh_reload.initialize()
                gmsh_reload.option.setNumber("General.Terminal", 0)
                
                # CRITICAL: Use gmsh.merge() not gmsh.open() to load mesh data
                gmsh_reload.merge(absolute_output_file)
                
                print(f"[DEBUG] Merged mesh file: {absolute_output_file}")
                
                # Check what entities exist
                entities_0d = gmsh_reload.model.getEntities(0)
                entities_1d = gmsh_reload.model.getEntities(1)
                entities_2d = gmsh_reload.model.getEntities(2)
                entities_3d = gmsh_reload.model.getEntities(3)
                print(f"[DEBUG] Entities: 0D={len(entities_0d)}, 1D={len(entities_1d)}, 2D={len(entities_2d)}, 3D={len(entities_3d)}")
                
                # Initialize quality maps
                per_element_quality = {} # Default (SICN)
                per_element_gamma = {}
                per_element_skewness = {}
                per_element_aspect_ratio = {}
                
                # Get 2D elements (triangles)
                tri_types, tri_tags, tri_nodes = gmsh_reload.model.mesh.getElements(2)
                triangle_count = 0
                for elem_type, tags in zip(tri_types, tri_tags):
                    if elem_type in [2, 9]: # Linear & Quadratic Triangles
                        try:
                            # 1. SICN (Default)
                            sicn_vals = gmsh_reload.model.mesh.getElementQualities(tags.tolist(), "sicn")
                            
                            # 2. Gamma
                            gamma_vals = gmsh_reload.model.mesh.getElementQualities(tags.tolist(), "gamma")
                            
                            for tag, sicn, gamma in zip(tags, sicn_vals, gamma_vals):
                                tag_int = int(tag)
                                per_element_quality[tag_int] = float(sicn)
                                per_element_gamma[tag_int] = float(gamma)
                                # Derived metrics matching ExhaustiveStrategy logic
                                per_element_skewness[tag_int] = 1.0 - float(sicn)
                                per_element_aspect_ratio[tag_int] = 1.0 / float(sicn) if sicn > 0 else 100.0
                                
                            triangle_count += len(tags)
                            print(f"[DEBUG] Extracted qualities for {len(tags)} triangles (type {elem_type})")
                        except Exception as e:
                            print(f"[DEBUG] Error getting triangle qualities: {e}")
                
                # Get 3D elements (tets)
                tet_types, tet_tags, tet_nodes = gmsh_reload.model.mesh.getElements(3)
                tet_count = 0
                all_qualities = [] # For statistics (SICN)
                
                for elem_type, tags in zip(tet_types, tet_tags):
                    if elem_type in [4, 11]: # Linear & Quadratic Tets
                        try:
                            # 1. SICN
                            sicn_vals = gmsh_reload.model.mesh.getElementQualities(tags.tolist(), "sicn")
                            # 2. Gamma
                            gamma_vals = gmsh_reload.model.mesh.getElementQualities(tags.tolist(), "gamma")
                            
                            for tag, sicn, gamma in zip(tags, sicn_vals, gamma_vals):
                                tag_int = int(tag)
                                per_element_quality[tag_int] = float(sicn)
                                per_element_gamma[tag_int] = float(gamma)
                                per_element_skewness[tag_int] = 1.0 - float(sicn)
                                per_element_aspect_ratio[tag_int] = 1.0 / float(sicn) if sicn > 0 else 100.0
                                all_qualities.append(sicn)
                                
                            tet_count += len(tags)
                            print(f"[DEBUG] Extracted qualities for {len(tags)} tets (type {elem_type})")
                        except Exception as e:
                            print(f"[DEBUG] Error getting tet qualities: {e}")
                
                # Calculate statistics
                if all_qualities:
                    sorted_q = sorted(all_qualities)
                    idx_10 = max(0, int(len(sorted_q) * 0.10))
                    quality_metrics['sicn_10_percentile'] = sorted_q[idx_10]
                    print(f"[DEBUG] Extracted quality for {triangle_count} triangles and {tet_count} tets")
                    print(f"[DEBUG] Quality range: {min(all_qualities):.3f} to {max(all_qualities):.3f}")
                    print(f"[DEBUG] 10th percentile: {sorted_q[idx_10]:.3f}")
                else:
                    print("[DEBUG WARNING] No element qualities extracted!")
                
                gmsh_reload.finalize()
            except Exception as e:
                import traceback
                print(f"[ERROR] Failed to extract per-element quality: {e}")
                traceback.print_exc()

            return {
                'success': True,
                'output_file': absolute_output_file,  # ABSOLUTE path for GUI
                'metrics': metrics,
                'quality_metrics': quality_metrics,  # Flattened metrics for GUI
                'per_element_quality': per_element_quality,  # SICN (Default)
                'per_element_gamma': per_element_gamma,
                'per_element_skewness': per_element_skewness,
                'per_element_aspect_ratio': per_element_aspect_ratio,
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Mesh Generation Worker')
    parser.add_argument('cad_file', help='Path to CAD file')
    parser.add_argument('output_dir', nargs='?', help='Output directory')
    parser.add_argument('--config-file', help='Path to configuration JSON file')
    parser.add_argument('--quality-params', help='JSON string of quality parameters (legacy)')
    
    args = parser.parse_args()
    
    cad_file = args.cad_file
    output_dir = args.output_dir
    
    # Load quality params
    quality_params = {}
    if args.config_file and os.path.exists(args.config_file):
        try:
            with open(args.config_file, 'r') as f:
                quality_params = json.load(f)
        except Exception as e:
            print(json.dumps({'success': False, 'error': f'Failed to load config file: {e}'}))
            sys.exit(1)
    elif args.quality_params:
        try:
            quality_params = json.loads(args.quality_params)
        except:
            pass
            
    # Generate mesh
    result = generate_mesh(cad_file, output_dir, quality_params)

    # Output result as JSON
    print(json.dumps(result))
