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
import time
import numpy as np

# Add project root to path
# From apps/cli/, go up 2 levels to MeshPackageLean/
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from strategies.exhaustive_strategy import ExhaustiveMeshGenerator
from strategies.hex_dominant_strategy import (
    HighFidelityDiscretization,
    ConvexDecomposition
)
from core.config import Config
import tempfile
import gmsh


def generate_gpu_delaunay_mesh(cad_file: str, output_dir: str = None, quality_params: Dict = None) -> Dict:
    """
    Generate tetrahedral mesh using GPU-accelerated Delaunay triangulation.
    
    Pipeline:
    1. Load CAD file and generate surface mesh (CPU/Gmsh)
    2. Extract surface vertices and triangles
    3. Run GPU Fill & Filter pipeline
    4. Save result as Gmsh-compatible mesh
    """
    try:
        from core.gpu_mesher import gpu_delaunay_fill_and_filter, GPU_AVAILABLE
        
        if not GPU_AVAILABLE:
            print("[GPU Mesher] GPU not available, falling back to CPU", flush=True)
            return {'success': False, 'message': 'GPU Mesher not available. Falling back to CPU meshing.'}
        
        print("[GPU Mesher] Starting GPU Delaunay pipeline...", flush=True)
        start_total = time.time()
        
        # Determine output path
        mesh_folder = Path(__file__).parent / "generated_meshes"
        mesh_folder.mkdir(exist_ok=True)
        mesh_name = Path(cad_file).stem
        output_file = str(mesh_folder / f"{mesh_name}_gpu_mesh.msh")
        
        # Step 1: Load CAD and generate surface mesh using Gmsh
        print("[GPU Mesher] Step 1: Loading CAD and generating surface mesh...", flush=True)
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.add("gpu_surface")
        
        # Import the CAD file
        gmsh.merge(cad_file)
        gmsh.model.occ.synchronize()
        
        # Get bounding box
        bbox = gmsh.model.getBoundingBox(-1, -1)  # All entities
        bbox_min = np.array([bbox[0], bbox[1], bbox[2]])
        bbox_max = np.array([bbox[3], bbox[4], bbox[5]])
        
        # Calculate appropriate mesh size based on target elements
        target_elements = quality_params.get('target_elements', 10000) if quality_params else 10000
        volume = np.prod(bbox_max - bbox_min)
        # Rough estimate: target_elements ~= volume / (element_size^3 / 6)
        mesh_size = (volume / (target_elements / 6)) ** (1/3)
        mesh_size = max(mesh_size, (bbox_max - bbox_min).min() / 100)  # Don't go too small
        
        print(f"[GPU Mesher] Bounding box: {bbox_min} to {bbox_max}", flush=True)
        print(f"[GPU Mesher] Target elements: {target_elements}, Mesh size: {mesh_size:.3f}", flush=True)
        
        # Set mesh size
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.5)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
        
        # Generate 2D surface mesh only
        gmsh.model.mesh.generate(2)
        
        # Extract surface mesh data
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_coords = np.array(node_coords).reshape(-1, 3)
        
        # Build node index mapping (gmsh tags start at 1)
        tag_to_idx = {tag: idx for idx, tag in enumerate(node_tags)}
        
        # Get triangle elements (type 2 = 3-node triangle)
        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2)  # dim=2 for surfaces
        
        surface_faces = []
        for etype, etags, enodes in zip(elem_types, elem_tags, elem_node_tags):
            if etype == 2:  # 3-node triangle
                enodes = np.array(enodes).reshape(-1, 3)
                for tri_nodes in enodes:
                    face = [tag_to_idx[n] for n in tri_nodes]
                    surface_faces.append(face)
        
        surface_faces = np.array(surface_faces)
        surface_verts = node_coords
        
        print(f"[GPU Mesher] Surface mesh: {len(surface_verts)} vertices, {len(surface_faces)} triangles", flush=True)
        
        gmsh.finalize()
        
        # Step 2: Run GPU Fill & Filter pipeline
        print("[GPU Mesher] Step 2: Running GPU Fill & Filter pipeline...", flush=True)
        
        # Determine resolution based on target elements
        # Higher resolution = more internal points = more tetrahedra
        resolution = max(20, int((target_elements / 6) ** (1/3)))
        resolution = min(resolution, 100)  # Cap at 100 for memory
        
        # ==========================================
        # 1. AUTO-CALCULATE SIZING
        # ==========================================
        # Get volume of the bounding box (Approximation of shape volume)
        bbox_size = bbox_max - bbox_min
        volume_approx = np.prod(bbox_size)
        
        # Theoretical relationship: Volume ≈ N_tets * (L^3 / (6*sqrt(2)))
        # Inverted to find L (Target Edge Length):
        target_L = (volume_approx / target_elements * 6 * np.sqrt(2)) ** (1/3)
        
        print(f"[Auto-Sizing] Target: {target_elements} elements for Volume {volume_approx:.2f}")
        print(f"[Auto-Sizing] Calculated ideal edge length: {target_L:.4f}")
        
        # Set Sizing Parameters
        min_spacing = target_L 
        max_spacing = target_L * 3.0
        grading = 1.8 
        target_sicn = 0.15
        
        print(f"[GPU Mesher] Using resolution: {resolution} (fallback)", flush=True)
        print(f"[GPU Mesher] Adaptive Sizing: min={min_spacing:.3f}, max={max_spacing:.3f}, grading={grading}", flush=True)
        print(f"[GPU Mesher] Refinement Target SICN: {target_sicn}", flush=True)
        
        def progress_callback(msg, pct):
            print(f"[GPU Mesher] {msg} ({pct}%)", flush=True)
        
        vertices, tetrahedra, surface_faces = gpu_delaunay_fill_and_filter(
            surface_verts, surface_faces, 
            bbox_min, bbox_max,
            min_spacing=min_spacing,
            max_spacing=max_spacing,
            grading=grading,
            resolution=resolution,
            target_sicn=target_sicn,
            progress_callback=progress_callback
        )
        
        elapsed = time.time() - start_total
        print(f"[GPU Mesher] Generated {len(tetrahedra)} tetrahedra in {elapsed:.2f}s")
        
        # Step 3: Save mesh in Gmsh format
        print("[GPU Mesher] Step 3: Saving mesh to Gmsh format...", flush=True)
        
        gmsh.initialize()
        gmsh.model.add("gpu_result")
        
        # Create discrete entities
        gmsh.model.addDiscreteEntity(3, 1) # Volume
        gmsh.model.addDiscreteEntity(2, 2) # Surface
        
        # Add nodes to the volume entity (simplification: all nodes in volume)
        node_tags = list(range(1, len(vertices) + 1))
        node_coords_flat = vertices.flatten().tolist()
        gmsh.model.mesh.addNodes(3, 1, node_tags, node_coords_flat)
        
        # Add tetrahedra (type 4 = 4-node tet) to Volume (1)
        tet_tags = list(range(1, len(tetrahedra) + 1))
        tet_nodes_flat = (tetrahedra + 1).flatten().tolist()
        gmsh.model.mesh.addElementsByType(1, 4, tet_tags, tet_nodes_flat)
        
        # Add surface triangles (type 2 = 3-node triangle) to Surface (2)
        tri_tags = list(range(1, len(surface_faces) + 1))
        tri_nodes_flat = (surface_faces + 1).flatten().tolist()
        gmsh.model.mesh.addElementsByType(2, 2, tri_tags, tri_nodes_flat)
        
        # Add physical groups
        gmsh.model.addPhysicalGroup(3, [1], tag=1, name="Volume")
        gmsh.model.addPhysicalGroup(2, [2], tag=2, name="Surface")
        
        # Write mesh file
        gmsh.write(output_file)
        
        # Compute quality metrics
        print("[GPU Mesher] Computing quality metrics...")
        try:
            all_tags = list(range(1, len(tetrahedra) + 1))
            sicn_vals = gmsh.model.mesh.getElementQualities(all_tags, "minSICN")
            gamma_vals = gmsh.model.mesh.getElementQualities(all_tags, "gamma")
            
            per_element_quality = {i+1: float(sicn_vals[i]) for i in range(len(sicn_vals))}
            per_element_gamma = {i+1: float(gamma_vals[i]) for i in range(len(gamma_vals))}
            per_element_skewness = {i+1: 1.0 - float(sicn_vals[i]) for i in range(len(sicn_vals))}
            per_element_aspect_ratio = {i+1: 1.0/float(sicn_vals[i]) if sicn_vals[i] > 0 else 100.0 for i in range(len(sicn_vals))}
            
            quality_metrics = {
                'sicn_min': float(min(sicn_vals)),
                'sicn_max': float(max(sicn_vals)),
                'sicn_avg': float(np.mean(sicn_vals)),
                'gamma_min': float(min(gamma_vals)),
                'gamma_max': float(max(gamma_vals)),
                'gamma_avg': float(np.mean(gamma_vals)),
            }
            
            print(f"[GPU Mesher] Quality - SICN: min={quality_metrics['sicn_min']:.3f}, avg={quality_metrics['sicn_avg']:.3f}, max={quality_metrics['sicn_max']:.3f}")
        except Exception as e:
            print(f"[GPU Mesher] Warning: Could not compute quality: {e}")
            per_element_quality = {}
            per_element_gamma = {}
            per_element_skewness = {}
            per_element_aspect_ratio = {}
            quality_metrics = {}
        
        gmsh.finalize()
        
        print(f"[GPU Mesher] SUCCESS! Mesh saved to: {output_file}")
        
        return {
            'success': True,
            'output_file': str(Path(output_file).absolute()),
            'strategy': 'gpu_delaunay_fill_filter',
            'message': f'GPU Delaunay: {len(tetrahedra)} tetrahedra in {elapsed:.2f}s',
            'total_elements': len(tetrahedra),
            'total_nodes': len(vertices),
            'per_element_quality': per_element_quality,
            'per_element_gamma': per_element_gamma,
            'per_element_skewness': per_element_skewness,
            'per_element_aspect_ratio': per_element_aspect_ratio,
            'quality_metrics': quality_metrics,
            'metrics': {
                'total_elements': len(tetrahedra),
                'total_nodes': len(vertices),
                'gpu_time_ms': elapsed * 1000,
                'surface_triangles': len(surface_faces),
                'resolution': resolution
            }
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'message': f'GPU Delaunay meshing failed: {str(e)}',
            'traceback': traceback.format_exc()
        }


def generate_hex_dominant_mesh(cad_file: str, output_dir: str = None, quality_params: Dict = None) -> Dict:
    """
    Generate hex-dominant mesh using CoACD + subdivision pipeline
    
    Steps:
    1. High-fidelity STEP → STL
    2. CoACD convex decomposition
    3. Discrete mesh approach (merge cleaned STLs)
    4. Surface classification → volume
    5. Subdivision algorithm (tets → 4 hexes each)
    """
    try:
        import trimesh
        save_stl = quality_params.get('save_stl', False) if quality_params else False
        
        print("[HEX-DOM] Starting hex-dominant meshing pipeline...")
        
        # Determine output folders
        mesh_folder = Path(__file__).parent / "generated_meshes"
        mesh_folder.mkdir(exist_ok=True)
        mesh_name = Path(cad_file).stem
        output_file = str(mesh_folder / f"{mesh_name}_hex_mesh.msh")
        
        # Temporary STL path
        temp_dir = Path(tempfile.gettempdir())
        stl_file = temp_dir / f"{mesh_name}_step1.stl"
        
        # Step 1: STEP → STL
        print("[HEX-DOM] Step 1: Converting STEP to STL...")
        step1 = HighFidelityDiscretization()
        success = step1.convert_step_to_stl(cad_file, str(stl_file))
        if not success:
            return {'success': False, 'message': 'Step 1 failed: STEP to STL conversion'}
        
        if save_stl:
            saved_stl_step1 = mesh_folder / f"{mesh_name}_step1_stl.stl"
            import shutil
            shutil.copy(stl_file, saved_stl_step1)
            print(f"[HEX-DOM] Saved Step 1 STL: {saved_stl_step1}")
        
        # Step 2: CoACD Decomposition
        print("[HEX-DOM] Step 2: CoACD convex decomposition...")
        step2 = ConvexDecomposition()
        parts, stats = step2.decompose_mesh(str(stl_file), threshold=0.05)
        
        if not parts:
            return {'success': False, 'message': 'Step 2 failed: CoACD decomposition'}
        
        print(f"[HEX-DOM] Decomposed into {len(parts)} convex parts (volume error: {stats['volume_error_pct']:.2f}%)")
        
        # Step 3-5: Hex Meshing
        print("[HEX-DOM] Step 3-5: Generating hex mesh via subdivision...")
        
        gmsh.initialize()
        gmsh.model.add("hex_dom_final")
        
        # Set tolerances
        gmsh.option.setNumber("Geometry.Tolerance", 1e-4)
        gmsh.option.setNumber("Mesh.ToleranceInitialDelaunay", 1e-4)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay
        
        # Clean and merge each part
        for i, (verts, faces) in enumerate(parts):
            chunk_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            chunk_mesh.merge_vertices()
            chunk_mesh.remove_degenerate_faces()
            chunk_mesh.remove_duplicate_faces()
            
            chunk_file = temp_dir / f"temp_chunk_{i}.stl"
            chunk_mesh.export(str(chunk_file))
            gmsh.merge(str(chunk_file))
            chunk_file.unlink()  # Delete temp file
        
        # Classify surfaces to create volumes
        try:
            angle = 40
            gmsh.model.mesh.classifySurfaces(angle * 3.14159 / 180, True, False, 180 * 3.14159 / 180)
            gmsh.model.mesh.createGeometry()
            
            s = gmsh.model.getEntities(2)
            l = gmsh.model.geo.addSurfaceLoop([s[i][1] for i in range(len(s))])
            gmsh.model.geo.addVolume([l])
            gmsh.model.geo.synchronize()
            
            print(f"[HEX-DOM] Created volume from {len(s)} classified surfaces")
        except Exception as e:
            gmsh.finalize()
            return {'success': False, 'message': f'Step 3 failed: Surface classification - {e}'}
        
        # Generate 3D tet mesh first
        try:
            gmsh.model.mesh.generate(3)
            print("[HEX-DOM] Generated intermediate tet mesh")
        except Exception as e:
            gmsh.finalize()
            return {'success': False, 'message': f'Step 4 failed: Tet meshing - {e}'}
        
        # Apply subdivision (tet → hex)
        print("[HEX-DOM] Applying subdivision algorithm...")
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 2)  # All hexes
        gmsh.model.mesh.refine()
        
        # Write output
        gmsh.write(output_file)
        
        # Count elements
        element_types = gmsh.model.mesh.getElementTypes()
        element_counts = {}
        for etype in element_types:
            elem_name = gmsh.model.mesh.getElementProperties(etype)[0]
            elem_tags, _ = gmsh.model.mesh.getElementsByType(etype)
            element_counts[elem_name] = len(elem_tags)
        
        num_hexes = element_counts.get("8-node hexahedron", 0) + element_counts.get("Hexahedron 8", 0)
        num_tets = element_counts.get("4-node tetrahedron", 0) + element_counts.get("Tetrahedron 4", 0)
        total_3d = num_hexes + num_tets
        
        # Extract per-element quality for visualization
        try:
            from core.quality import compute_element_quality_gmsh
            
            # Get hex elements (type 5 = 8-node hex)
            hex_tags, hex_nodes = gmsh.model.mesh.getElementsByType(5)
            
            if len(hex_tags) > 0:
                # Compute quality for each hex
                per_element_quality = []
                for tag in hex_tags:
                    quality = gmsh.model.mesh.getElementQualities([tag], "minSICN")
                    per_element_quality.append(quality[0] if quality else 0.5)
                
                print(f"[HEX-DOM] Computed quality for {len(per_element_quality)} hex elements")
            else:
                per_element_quality = []
        except Exception as e:
            print(f"[HEX-DOM] Warning: Could not compute quality: {e}")
            per_element_quality = []
        
        gmsh.finalize()
        
        print(f"[HEX-DOM] Success! Generated {num_hexes} hexahedra ({total_3d} total 3D elements)")
        
        return {
            'success': True,
            'output_file': str(Path(output_file).absolute()),
            'strategy': 'hex_dominant_subdivision',
            'message': f'Hex-dominant mesh: {num_hexes} hexes, {num_tets} tets',
            'total_elements': total_3d,
            'total_nodes': 0,  # TODO: count nodes
            'per_element_quality': per_element_quality,  # For VTK visualization
            'metrics': {
                'num_hexes': num_hexes,
                'num_tets': num_tets,
                'hex_ratio': (num_hexes / total_3d * 100) if total_3d > 0 else 0,
                'volume_error_pct': stats['volume_error_pct'],
                'num_parts': len(parts)
            },
            'quality_metrics': {
                'min_quality': min(per_element_quality) if per_element_quality else 0,
                'max_quality': max(per_element_quality) if per_element_quality else 1,
                'avg_quality': sum(per_element_quality) / len(per_element_quality) if per_element_quality else 0
            }
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'message': f'Hex dominant meshing failed: {str(e)}'
        }


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
        # Check if hex dominant strategy is requested
        mesh_strategy = quality_params.get('mesh_strategy', '') if quality_params else ''
        save_stl = quality_params.get('save_stl', False) if quality_params else False
        
        if 'Hex Dominant' in mesh_strategy:
            print("[DEBUG] Hex Dominant strategy detected - using hex pipeline")
            return generate_hex_dominant_mesh(cad_file, output_dir, quality_params)
        
        if 'GPU Delaunay' in mesh_strategy:
            print("[DEBUG] GPU Delaunay strategy detected - using GPU Fill & Filter pipeline")
            return generate_gpu_delaunay_mesh(cad_file, output_dir, quality_params)
        
        # Default: use exhaustive tet strategy
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
            
            # Update ansys_mode for CFD/FEA export
            if 'ansys_mode' in quality_params:
                config.mesh_params.ansys_mode = quality_params['ansys_mode']
                print(f"[DEBUG] Set ansys_mode to: {quality_params['ansys_mode']}")

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
                            # 1. SICN (Default) - use "minSICN" for post-processing
                            sicn_vals = gmsh_reload.model.mesh.getElementQualities(tags.tolist(), "minSICN")
                            
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
                            # 1. SICN - use "minSICN" for post-processing
                            sicn_vals = gmsh_reload.model.mesh.getElementQualities(tags.tolist(), "minSICN")
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
