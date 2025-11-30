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
from core.mesh_generator import BaseMeshGenerator
import tempfile
import gmsh
import numpy as np


class SimpleTetGenerator(BaseMeshGenerator):
    """
    Simple, fast tetrahedral mesh generator
    
    Uses a single algorithm (Delaunay or Frontal) for fast meshing
    without the overhead of exhaustive trying.
    """
    
    def __init__(self, algorithm_name: str = "Delaunay", config: Config = None):
        super().__init__(config)
        self.algorithm_name = algorithm_name
        
        # Map algorithm names to Gmsh algorithm codes
        self.algorithm_map = {
            "Delaunay": 1,  # Delaunay
            "Frontal": 4,   # Frontal
            "HXT": 10,      # HXT (requires HXT compiled)
        }
        
    def run_meshing_strategy(self, input_file: str, output_file: str) -> bool:
        """Meshing with linear-first strategy and repair retries"""
        try:
            self.log_message(f"Using {self.algorithm_name} algorithm with Linear-First Strategy")
            
            # 0. Pre-processing: Boolean Fragments (Fix for Assembly Overlaps)
            self._apply_boolean_fragments()

            # Calculate initial mesh parameters
            self.current_mesh_params = self.calculate_initial_mesh_parameters()
            self.apply_mesh_parameters(self.current_mesh_params)
            
            # PRODUCTION SETTINGS: Prevent 4-million element explosion
            # Set reasonable mesh sizes (assuming typical mechanical part in mm)
            diagonal = self.geometry_info.get('diagonal', 100.0)
            gmsh.option.setNumber("Mesh.MeshSizeMin", diagonal / 200.0)  # Prevent microscopic tets
            gmsh.option.setNumber("Mesh.MeshSizeMax", diagonal / 10.0)   # Allow coarser tets
            
            # OPTIMIZATION: Disable slow Netgen, use fast standard optimizer
            gmsh.option.setNumber("Mesh.Optimize", 1)         # Enable standard optimization
            gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)   # Disable slow Netgen (single-threaded)
            
            # Set mesh algorithms
            gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay for 2D
            
            # CRITICAL: Use HXT for assemblies (multiple volumes) or complex geometry
            # HXT (Algorithm 10) is parallel and handles dirty topology much better than Delaunay (1)
            num_volumes = len(gmsh.model.getEntities(dim=3))
            algo_3d = self.algorithm_map.get(self.algorithm_name, 1)
            
            if num_volumes > 1:
                # Assembly detected - use HXT for robustness and parallelism
                gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT (parallel, robust)
                self.log_message(f"Using HXT algorithm (parallel) for {num_volumes}-volume assembly")
            else:
                # Single volume - use HXT anyway for speed (fully parallelized)
                gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT for speed
            
            self.log_message(f"Mesh algorithms: 2D=6, 3D={int(gmsh.option.getNumber('Mesh.Algorithm3D'))}")
            self.log_message(f"Mesh size range: {gmsh.option.getNumber('Mesh.MeshSizeMin'):.2f} - {gmsh.option.getNumber('Mesh.MeshSizeMax'):.2f}")
            
            # Attempt 1: Standard Linear-First
            self.log_message("[ATTEMPT 1] Standard Linear-First Meshing...")
            if self._attempt_meshing(linear_first=True, optimize=True):
                self._finalize_success(output_file)
                return True
                
            # Attempt 2: Discrete Remeshing (Fix for PLC Error)
            self.log_message("[ATTEMPT 2] Standard failed. Trying Discrete Remeshing...")
            if self._attempt_discrete_remeshing(input_file):
                self._finalize_success(output_file)
                return True

            # Attempt 3: Aggressive Tolerance + Frontal Algo (Last Resort)
            self.log_message("[ATTEMPT 3] Discrete remeshing failed, retrying with aggressive tolerance & Frontal algo...")
            gmsh.option.setNumber("Geometry.Tolerance", 1e-4)
            gmsh.option.setNumber("Mesh.StlRemoveDuplicateTriangles", 1)
            gmsh.option.setNumber("Mesh.Algorithm3D", 4) # Frontal
            gmsh.option.setNumber("Mesh.ToleranceInitialDelaunay", 1e-2)
            
            if self._attempt_meshing(linear_first=True, optimize=True):
                self._finalize_success(output_file)
                return True
                
            self.log_message(f"[!] All meshing attempts failed")
            return False
            
        except Exception as e:
            self.log_message(f"[!] Meshing failed: {e}", level="ERROR")
            import traceback
            traceback.print_exc()
            return False

    def _attempt_meshing(self, linear_first=True, optimize=True) -> bool:
        """Helper to run a single meshing attempt"""
        import time
        self.timings = {}
        
        try:
            # 1. Generate 2D mesh (Linear first for stability)
            t0 = time.time()
            self.log_message("Generating 2D surface mesh (Linear)...")
            self.set_element_order(1) 
            if not self.generate_mesh_internal(2):
                return False
            self.timings['2d_mesh'] = time.time() - t0
            
            # Analyze surface quality
            self.analyze_surface_mesh_quality()
                
            # 2. Optimize Surface
            if optimize:
                t0 = time.time()
                self.log_message("Optimizing surface mesh (Netgen)...")
                gmsh.model.mesh.optimize("Netgen")
                self.timings['surface_opt'] = time.time() - t0
                
            # 3. Generate 3D mesh
            t0 = time.time()
            self.log_message("Generating 3D volume mesh...")
            if not self.generate_mesh_internal(3):
                return False
            self.timings['3d_mesh'] = time.time() - t0
                
            # 4. Optimize Volume
            if optimize:
                t0 = time.time()
                self.log_message("Optimizing volume mesh (Netgen)...")
                gmsh.model.mesh.optimize("Netgen")
                self.timings['volume_opt'] = time.time() - t0
                
            # 5. Compatibility Settings
            # CRITICAL: Force saving of ALL elements (including surface triangles)
            # This ensures the viewer sees the skin of the mesh even if it doesn't support volume rendering
            gmsh.option.setNumber("Mesh.SaveAll", 1)
            self.log_message("Enabled Mesh.SaveAll=1 for viewer compatibility")

            # 6. Convert to Quadratic (Order 2) - Requested by User
            # We keep SaveAll=1 so that if the viewer struggles with 2nd order volumes,
            # it might still show the 2nd order surfaces.
            t0 = time.time()
            self.log_message("Converting to quadratic (Order 2)...")
            gmsh.model.mesh.setOrder(2)
            self.timings['quadratic_conversion'] = time.time() - t0
                
            return True
        except Exception as e:
            self.log_message(f"Meshing attempt failed: {e}")
            return False

    def _apply_boolean_fragments(self):
        """
        DISABLED: Fragment operations cause BOPAlgo_AlertSelfInterferingShape errors.
        
        The GUI doesn't use fragments - it relies on:
        - Geometry.OCCAutoFix = 1 (fixes micro-gaps during import)
        - Geometry.AutoCoherence = 1 (merges vertices during import)
        
        These settings are now applied in BaseMeshGenerator.load_cad_file()
        BEFORE the file is loaded, so no post-processing is needed.
        """
        # Do nothing - the GUI settings handle everything during import
        self.log_message("Skipping fragment operations (using GUI import settings instead)")
        return

    def _attempt_discrete_remeshing(self, input_file: str) -> bool:
        """
        Attempt to mesh by creating a discrete topology from the surface mesh.
        This handles 'dirty' geometry by shrink-wrapping a new mesh over it.
        """
        try:
            import time
            import os
            self.timings = {}
            t0 = time.time()
            self.log_message("Starting Discrete Remeshing Strategy...")
            
            gmsh.clear()
            gmsh.merge(input_file)
            
            self.log_message("Generating initial surface mesh...")
            gmsh.option.setNumber("Mesh.Algorithm", 6) 
            gmsh.model.mesh.generate(2)
            
            self.log_message("Disconnecting mesh from CAD (Save -> Clear -> Load)...")
            temp_msh = input_file + ".temp_discrete.msh"
            gmsh.write(temp_msh)
            gmsh.clear()
            gmsh.merge(temp_msh)
            
            self.log_message("Creating discrete topology from mesh...")
            gmsh.model.mesh.createTopology()
            gmsh.model.mesh.createGeometry()
            
            self.log_message("Classifying discrete surfaces (40 deg)...")
            gmsh.model.mesh.classifySurfaces(40 * 3.14159 / 180, True, True)
            gmsh.model.mesh.createGeometry()
            
            self.log_message("Re-creating volume from discrete shell...")
            gmsh.option.setNumber("Mesh.ToleranceInitialDelaunay", 1e-3)
            
            surfaces = gmsh.model.getEntities(2)
            if surfaces:
                s_tags = [s[1] for s in surfaces]
                vol_tag = gmsh.model.addDiscreteEntity(3, -1, s_tags)
                self.log_message(f"Created discrete volume {vol_tag} with {len(s_tags)} surfaces")
            
            self.log_message("Generating 3D volume mesh from discrete geometry...")
            gmsh.model.mesh.generate(3)
            
            self.log_message("Optimizing discrete volume mesh...")
            gmsh.model.mesh.optimize("Netgen")
            
            try:
                if os.path.exists(temp_msh):
                    os.remove(temp_msh)
            except:
                pass

            self.timings['discrete_remeshing'] = time.time() - t0
            return True
            
        except Exception as e:
            self.log_message(f"Discrete remeshing failed: {e}")
            return False

    def _finalize_success(self, output_file: str):
        """Save and log success"""
        # CRITICAL: Create Physical Groups before saving!
        # Many viewers (including ours) require Physical Groups to visualize the mesh
        # If they are missing, the mesh file might appear empty or invisible
        
        # 1. Create Physical Volume (if 3D)
        volumes = gmsh.model.getEntities(3)
        if volumes:
            # Check if physical groups already exist
            phys_vols = gmsh.model.getPhysicalGroups(3)
            if not phys_vols:
                p_tag = gmsh.model.addPhysicalGroup(3, [v[1] for v in volumes])
                gmsh.model.setPhysicalName(3, p_tag, "Volume")
                self.log_message(f"Created Physical Volume {p_tag} for {len(volumes)} volumes")
        
        # 2. Create Physical Surface (if 2D or 3D)
        surfaces = gmsh.model.getEntities(2)
        if surfaces:
            phys_surfs = gmsh.model.getPhysicalGroups(2)
            if not phys_surfs:
                p_tag = gmsh.model.addPhysicalGroup(2, [s[1] for s in surfaces])
                gmsh.model.setPhysicalName(2, p_tag, "Surface")
                self.log_message(f"Created Physical Surface {p_tag} for {len(surfaces)} surfaces")

        metrics = self.analyze_current_mesh()
        if metrics:
            self.quality_history.append({
                'iteration': self.current_iteration,
                'algorithm': self.algorithm_name,
                'metrics': metrics
            })
        
        if self.save_mesh(output_file):
            self.log_message(f"[OK] Mesh generated successfully")



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
    Generate hex-dominant mesh using the HXT algorithm (Algorithm 3D = 10).
    This replaces the legacy CoACD + Subdivision pipeline with a direct, robust approach.
    """
    try:
        from strategies.hxt_strategy import HXTHexDominantGenerator
        
        print("[HEX-DOM] Starting HXT Hex-Dominant meshing strategy...")
        
        # Determine output folders
        mesh_folder = Path(__file__).parent / "generated_meshes"
        mesh_folder.mkdir(exist_ok=True)
        mesh_name = Path(cad_file).stem
        output_file = str(mesh_folder / f"{mesh_name}_hex_mesh.msh")
        
        # Initialize config
        config = Config()
        if quality_params:
            if 'painted_regions' in quality_params:
                config.painted_regions = quality_params['painted_regions']
            if 'target_elements' in quality_params:
                config.target_elements = quality_params['target_elements']
                
        # Create generator
        generator = HXTHexDominantGenerator(config=config)
        
        # Run generation
        success = generator.run_meshing_strategy(cad_file, output_file)
        
        if success:
            # Extract quality metrics for return
            # We can reuse the logic from generate_tetrahedral_mesh or rely on what's in the file
            # For now, we'll reload and extract basic stats
            
            import gmsh as gmsh_reload
            gmsh_reload.initialize()
            gmsh_reload.option.setNumber("General.Terminal", 0)
            gmsh_reload.merge(output_file)
            
            # Count elements
            element_counts = {}
            element_types = gmsh_reload.model.mesh.getElementTypes()
            for etype in element_types:
                elem_name = gmsh_reload.model.mesh.getElementProperties(etype)[0]
                elem_tags, _ = gmsh_reload.model.mesh.getElementsByType(etype)
                element_counts[elem_name] = len(elem_tags)
            
            num_hexes = element_counts.get("8-node hexahedron", 0) + element_counts.get("Hexahedron 8", 0)
            num_tets = element_counts.get("4-node tetrahedron", 0) + element_counts.get("Tetrahedron 4", 0)
            total_3d = num_hexes + num_tets
            
            # Count nodes
            node_tags, _, _ = gmsh_reload.model.mesh.getNodes()
            total_nodes = len(node_tags)
            
            gmsh_reload.finalize()
            
            return {
                'success': True,
                'output_file': str(Path(output_file).absolute()),
                'strategy': 'hex_dominant_hxt',
                'message': f'HXT Hex-Dominant mesh generated: {num_hexes} hexes, {num_tets} tets',
                'total_elements': total_3d,
                'total_nodes': total_nodes,
                'metrics': {
                    'num_hexes': num_hexes,
                    'num_tets': num_tets
                }
            }
        else:
            return {
                'success': False,
                'message': 'HXT meshing failed'
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
