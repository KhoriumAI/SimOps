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
from strategies.hex_dominant_strategy import (
    HighFidelityDiscretization,
    ConvexDecomposition
)
from strategies.polyhedral_strategy import PolyhedralMeshGenerator
from core.config import Config
import tempfile
import gmsh
import numpy as np


def generate_hex_testing_visualization(cad_file: str, output_dir: str = None, quality_params: Dict = None) -> Dict:
    """
    Generate CoACD component visualization for debugging
    
    Steps:
    1. STEP → STL
    2. CoACD decomposition
    3. Export each component as separate mesh with Component_ID
    
    Returns dict with component_files list for PyVista loading
    """
    try:
        import trimesh
        import numpy as np
        import multiprocessing
        import time
        
        print("[HEX-TEST] Starting CoACD component visualization...")
        
        # Determine output folders
        mesh_folder = Path(__file__).parent / "generated_meshes"
        mesh_folder.mkdir(exist_ok=True)
        mesh_name = Path(cad_file).stem
        
        # Temporary STL path
        temp_dir = Path(tempfile.gettempdir())
        stl_file = temp_dir / f"{mesh_name}_step1.stl"
        
        # Step 1: STEP → STL
        print("[HEX-TEST] Step 1: Converting STEP to STL...")
        step1 = HighFidelityDiscretization()
        success = step1.convert_step_to_stl(cad_file, str(stl_file))
        if not success:
            return {'success': False, 'message': 'Step 1 failed: STEP to STL conversion'}
        
        # Step 2: CoACD Decomposition
        print("[HEX-TEST] Step 2: CoACD convex decomposition...")
        step2 = ConvexDecomposition()
        parts, stats = step2.decompose_mesh(str(stl_file), threshold=0.03)
        
        if not parts:
            return {'success': False, 'message': 'Step 2 failed: CoACD decomposition'}
        
        print(f"[HEX-TEST] Decomposed into {len(parts)} convex parts")
        
        # Step 3: Serialized Chunk Meshing (Fail-Fast)
        print("[HEX-TEST] Step 3: Serialized Chunk Meshing (Fail-Fast)...")
        
        component_files = []
        timeout_seconds = 30
        
        for i, (verts, faces) in enumerate(parts):
            print(f"  > Processing Chunk {i+1}/{len(parts)}...")
            
            # Export chunk to temp STL
            chunk_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            chunk_mesh.merge_vertices()
            chunk_mesh.remove_degenerate_faces()
            chunk_mesh.remove_duplicate_faces()
            
            chunk_stl = temp_dir / f"temp_chunk_{i}.stl"
            chunk_vtk = temp_dir / f"temp_chunk_{i}.vtk"
            chunk_mesh.export(str(chunk_stl))
            
            # Spawn worker process to mesh this chunk
            p = multiprocessing.Process(
                target=_mesh_chunk_process,
                args=(str(chunk_stl), str(chunk_vtk), quality_params)
            )
            p.start()
            
            # Wait with timeout
            p.join(timeout=timeout_seconds)
            
            if p.is_alive():
                print(f"    [TIMEOUT] Chunk {i} hung after {timeout_seconds}s - Terminating...")
                p.terminate()
                p.join()
                # Fallback: just show the STL if meshing failed
                component_file = mesh_folder / f"{mesh_name}_component_{i}.stl"
                chunk_mesh.export(str(component_file))
                component_files.append(str(component_file))
                print(f"    [FALLBACK] Saved unmeshed STL: {component_file.name}")
                
            elif p.exitcode != 0:
                print(f"    [FAILED] Chunk {i} failed with exit code {p.exitcode}")
                # Fallback: just show the STL if meshing failed
                component_file = mesh_folder / f"{mesh_name}_component_{i}.stl"
                chunk_mesh.export(str(component_file))
                component_files.append(str(component_file))
                print(f"    [FALLBACK] Saved unmeshed STL: {component_file.name}")
                
            else:
                print(f"    [SUCCESS] Chunk {i} meshed successfully")
                # Load the generated VTK and add component ID
                try:
                    import pyvista as pv
                    # Load the VTK file directly
                    chunk_vtk_pv = pv.read(str(chunk_vtk))
                    
                    # Add component ID
                    if "Component_ID" not in chunk_vtk_pv.array_names:
                        chunk_vtk_pv["Component_ID"] = np.full(chunk_vtk_pv.n_cells, i, dtype=int)
                    
                    # Save final VTK
                    component_file = mesh_folder / f"{mesh_name}_component_{i}.vtk"
                    chunk_vtk_pv.save(str(component_file))
                    component_files.append(str(component_file))
                    print(f"    [EXPORT] Saved meshed component to {component_file.name}")
                    
                except Exception as e:
                    print(f"    [ERROR] Failed to process VTK: {e}")
                    # Fallback to STL
                    component_file = mesh_folder / f"{mesh_name}_component_{i}.stl"
                    chunk_mesh.export(str(component_file))
                    component_files.append(str(component_file))
            
            # Cleanup temp files
            if chunk_stl.exists():
                chunk_stl.unlink()
            if chunk_vtk.exists():
                chunk_vtk.unlink()
        
        print(f"[HEX-TEST] Success! Exported {len(component_files)} components")
        
        return {
            'success': True,
            'strategy': 'hex_testing',
            'message': f'CoACD decomposition: {len(parts)} components',
            'component_files': component_files,
            'num_components': len(parts),
            'volume_error_pct': stats['volume_error_pct'],
            'visualization_mode': 'components'
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'message': f'Hex testing visualization failed: {str(e)}'
        }


def _mesh_chunk_process(chunk_stl_path: str, output_msh_path: str, quality_params: Dict):
    """
    Worker process to mesh a single convex chunk.
    Isolated process prevents GMSH hangs from freezing the main application.
    """
    import gmsh
    import sys
    
    try:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("General.Verbosity", 2)
        
        gmsh.model.add("chunk_mesh")
        
        # Set tolerances
        gmsh.option.setNumber("Geometry.Tolerance", 1e-4)
        gmsh.option.setNumber("Mesh.ToleranceInitialDelaunay", 1e-4)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay
        
        # Merge the chunk STL
        gmsh.merge(chunk_stl_path)
        
        # Classify surfaces
        angle = 40
        gmsh.model.mesh.classifySurfaces(angle * 3.14159 / 180, True, False, 180 * 3.14159 / 180)
        gmsh.model.mesh.createGeometry()
        
        # Create volume
        s = gmsh.model.getEntities(2)
        l = gmsh.model.geo.addSurfaceLoop([s[i][1] for i in range(len(s))])
        gmsh.model.geo.addVolume([l])
        gmsh.model.geo.synchronize()
        
        # Generate 3D tet mesh
        gmsh.model.mesh.generate(3)
        
        # Apply subdivision (tet -> hex)
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 2)  # All hexes
        gmsh.model.mesh.refine()
        
        # Write output (VTK format for visualization)
        gmsh.write(output_msh_path)
        gmsh.finalize()
        sys.exit(0)
        
    except Exception as e:
        print(f"[CHUNK-WORKER] Failed: {e}")
        if gmsh.isInitialized():
            gmsh.finalize()
        sys.exit(1)

def generate_hex_dominant_mesh(cad_file: str, output_dir: str = None, quality_params: Dict = None) -> Dict:
    """
    Generate hex-dominant mesh using CoACD + subdivision pipeline
    
    Steps:
    1. High-fidelity STEP → STL
    2. CoACD convex decomposition
    3. Serialized "Fail-Fast" Meshing of Chunks
    4. Merge successful chunks
    """
    try:
        import trimesh
        import numpy as np
        import multiprocessing
        import time
        
        save_stl = quality_params.get('save_stl', False) if quality_params else False
        timeout_seconds = 30  # Timeout per chunk
        
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
            shutil.copy(stl_file, saved_stl_step1)
            print(f"[HEX-DOM] Saved Step 1 STL: {saved_stl_step1}")
        
        # Step 2: CoACD Decomposition
        print("[HEX-DOM] Step 2: CoACD convex decomposition...")
        step2 = ConvexDecomposition()
        parts, stats = step2.decompose_mesh(str(stl_file), threshold=0.03)
        
        if not parts:
            return {'success': False, 'message': 'Step 2 failed: CoACD decomposition'}
        
        print(f"[HEX-DOM] Decomposed into {len(parts)} convex parts (volume error: {stats['volume_error_pct']:.2f}%)")
        
        # Export component surfaces for preview
        print("[HEX-DOM] Exporting CoACD components for preview...")
        preview_files = []
        for i, (verts, faces) in enumerate(parts):
            chunk_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            chunk_mesh.merge_vertices()
            chunk_mesh.remove_degenerate_faces()
            chunk_mesh.remove_duplicate_faces()
            
            # Save component surface for preview
            preview_file = mesh_folder / f"{mesh_name}_preview_component_{i}.vtk"
            try:
                import pyvista as pv
                pv_mesh = pv.PolyData(chunk_mesh.vertices, np.hstack([np.full((len(chunk_mesh.faces), 1), 3), chunk_mesh.faces]))
                pv_mesh["Component_ID"] = np.full(len(chunk_mesh.faces), i, dtype=int)
                pv_mesh.save(str(preview_file))
                preview_files.append(str(preview_file))
            except Exception as e:
                print(f"[HEX-DOM] Warning: Could not export preview for component {i}: {e}")
        
        # Emit progress update with component preview
        if preview_files:
            print(json.dumps({
                'type': 'progress',
                'phase': 'coacd_preview',
                'percentage': 100,
                'component_files': preview_files,
                'num_components': len(parts),
                'volume_error_pct': stats['volume_error_pct']
            }))
            sys.stdout.flush()
        
        # Step 3: Serialized Chunk Meshing
        print("[HEX-DOM] Step 3: Serialized Chunk Meshing (Fail-Fast)...")
        
        successful_chunks = []
        failed_chunks = []
        
        for i, (verts, faces) in enumerate(parts):
            print(f"  > Processing Chunk {i+1}/{len(parts)}...")
            
            # Export chunk to temp STL
            chunk_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            chunk_mesh.merge_vertices()
            chunk_mesh.remove_degenerate_faces()
            chunk_mesh.remove_duplicate_faces()
            
            chunk_stl = temp_dir / f"temp_chunk_{i}.stl"
            chunk_msh = temp_dir / f"temp_chunk_{i}.msh"
            chunk_mesh.export(str(chunk_stl))
            
            # Spawn worker process
            p = multiprocessing.Process(
                target=_mesh_chunk_process,
                args=(str(chunk_stl), str(chunk_msh), quality_params)
            )
            p.start()
            
            # Wait with timeout
            p.join(timeout=timeout_seconds)
            
            if p.is_alive():
                print(f"    [TIMEOUT] Chunk {i} hung after {timeout_seconds}s - Terminating...")
                p.terminate()
                p.join()
                
                # Save failed chunk
                fail_file = mesh_folder / f"FAIL_chunk_{i}.stl"
                shutil.copy(chunk_stl, fail_file)
                print(f"    [EXPORT] Saved failed chunk to {fail_file}")
                failed_chunks.append(i)
                
            elif p.exitcode != 0:
                print(f"    [FAILED] Chunk {i} failed with exit code {p.exitcode}")
                
                # Save failed chunk
                fail_file = mesh_folder / f"FAIL_chunk_{i}.stl"
                shutil.copy(chunk_stl, fail_file)
                print(f"    [EXPORT] Saved failed chunk to {fail_file}")
                failed_chunks.append(i)
                
            else:
                print(f"    [SUCCESS] Chunk {i} meshed successfully")
                successful_chunks.append(str(chunk_msh))
            
            # Cleanup temp STL
            if chunk_stl.exists():
                chunk_stl.unlink()

        # Step 4: Merge Successful Chunks
        if not successful_chunks:
            return {'success': False, 'message': 'All chunks failed to mesh'}
            
        print(f"[HEX-DOM] Step 4: Merging {len(successful_chunks)} successful chunks...")
        
        gmsh.initialize()
        gmsh.model.add("merged_hex_mesh")
        
        for msh_file in successful_chunks:
            gmsh.merge(msh_file)
            # Cleanup temp msh
            Path(msh_file).unlink()
            
        # Write final output
        gmsh.write(output_file)
        
        # Count elements (same as before)
        element_types = gmsh.model.mesh.getElementTypes()
        element_counts = {}
        for etype in element_types:
            elem_name = gmsh.model.mesh.getElementProperties(etype)[0]
            elem_tags, _ = gmsh.model.mesh.getElementsByType(etype)
            element_counts[elem_name] = len(elem_tags)
        
        num_hexes = element_counts.get("8-node hexahedron", 0) + element_counts.get("Hexahedron 8", 0)
        num_tets = element_counts.get("4-node tetrahedron", 0) + element_counts.get("Tetrahedron 4", 0)
        total_3d = num_hexes + num_tets
        
        # Count nodes
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        total_nodes = len(node_tags)
        
        # Extract per-element quality (simplified for merged mesh)
        per_element_quality = {}
        # ... (rest of quality extraction logic can remain similar, 
        # but we need to re-implement it since we replaced the whole block)
        
        # Re-implement quality extraction
        per_element_gamma = {}
        per_element_skewness = {}
        per_element_aspect_ratio = {}
        volume_qualities = []
        
        try:
            def _as_list(seq):
                if hasattr(seq, "tolist"):
                    return seq.tolist()
                return list(seq)
            
            # Surface elements
            surf_types, surf_tags, _ = gmsh.model.mesh.getElements(2)
            for elem_type, tags in zip(surf_types, surf_tags):
                if elem_type in (2, 3, 9, 10):
                    tag_list = _as_list(tags)
                    sicn_vals = gmsh.model.mesh.getElementQualities(tag_list, "minSICN")
                    for tag, sicn in zip(tag_list, sicn_vals):
                        per_element_quality[int(tag)] = float(sicn)
            
            # Volume elements
            vol_types, vol_tags, _ = gmsh.model.mesh.getElements(3)
            for elem_type, tags in zip(vol_types, vol_tags):
                if elem_type in (4, 5, 11, 12):
                    tag_list = _as_list(tags)
                    sicn_vals = gmsh.model.mesh.getElementQualities(tag_list, "minSICN")
                    for tag, sicn in zip(tag_list, sicn_vals):
                        val = float(sicn)
                        per_element_quality[int(tag)] = val
                        volume_qualities.append(val)
                        
        except Exception as e:
            print(f"[HEX-DOM] Warning: Could not compute quality: {e}")
            
        # Summary metrics
        sicn_avg = sum(volume_qualities) / len(volume_qualities) if volume_qualities else 0.0
        
        gmsh.finalize()
        
        msg = f'Hex-dominant mesh: {num_hexes} hexes'
        if failed_chunks:
            msg += f' ({len(failed_chunks)} chunks failed/skipped)'
            
        return {
            'success': True,
            'output_file': str(Path(output_file).absolute()),
            'strategy': 'hex_dominant_subdivision',
            'message': msg,
            'total_elements': total_3d,
            'total_nodes': total_nodes,
            'per_element_quality': per_element_quality,
            'metrics': {
                'num_hexes': num_hexes,
                'num_parts': len(parts),
                'failed_parts': len(failed_chunks)
            },
            'quality_metrics': {
                'sicn_avg': sicn_avg
            }
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'message': f'Hex dominant meshing failed: {str(e)}'
        }


def generate_polyhedral_mesh(cad_file: str, output_dir: str = None, quality_params: Dict = None) -> Dict:
    """
    Generate Polyhedral (Dual) mesh
    """
    try:
        print("[POLY] Starting Polyhedral (Dual) meshing...")
        
        # Determine output folders
        mesh_folder = Path(__file__).parent / "generated_meshes"
        mesh_folder.mkdir(exist_ok=True)
        mesh_name = Path(cad_file).stem
        output_file = str(mesh_folder / f"{mesh_name}_poly_dual.msh")
        
        # Initialize generator
        config = Config()
        if quality_params and 'painted_regions' in quality_params:
            config.painted_regions = quality_params['painted_regions']
            
        generator = PolyhedralMeshGenerator(config)
        
        # Run generation
        success = generator.run_meshing_strategy(cad_file, output_file)
        
        if success:
            # Check for JSON file with full polyhedral data
            json_file = Path(output_file).with_suffix('.json')
            poly_data_file = str(json_file.absolute()) if json_file.exists() else None
            
            return {
                'success': True,
                'output_file': str(Path(output_file).absolute()),
                'polyhedral_data_file': poly_data_file,
                'strategy': 'polyhedral_dual',
                'message': 'Polyhedral dual mesh generated',
                'visualization_mode': 'polyhedral' if poly_data_file else 'surface'
            }
        else:
            return {
                'success': False,
                'message': 'Polyhedral meshing failed'
            }
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'message': f'Polyhedral meshing failed: {str(e)}'
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
        
        if 'Hex Dominant Testing' in mesh_strategy:
            print("[DEBUG] Hex Dominant Testing detected - visualizing CoACD components")
            return generate_hex_testing_visualization(cad_file, output_dir, quality_params)
        elif 'Hex Dominant' in mesh_strategy:
            print("[DEBUG] Hex Dominant strategy detected - using hex pipeline")
            return generate_hex_dominant_mesh(cad_file, output_dir, quality_params)
        elif 'Polyhedral' in mesh_strategy:
            print("[DEBUG] Polyhedral strategy detected - using dual graph pipeline")
            return generate_polyhedral_mesh(cad_file, output_dir, quality_params)
        
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
