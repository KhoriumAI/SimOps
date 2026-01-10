#!/usr/bin/env python3
"""
Subprocess-based mesh generation worker

Runs mesh generation in a separate process to avoid gmsh threading issues.
Gmsh uses signals internally which only work in the main thread.

OPTIMIZATION: Heavy imports are done at module load time so subsequent
calls are faster. The GUI subprocess inherits cached imports.
"""

import sys
import shutil
import os
import json
from pathlib import Path
from typing import Dict
import time
import multiprocessing
import gmsh

# Add project root to path FIRST
# From apps/cli/, go up 2 levels to MeshPackageLean/
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Check verbosity early
IS_VERBOSE = os.environ.get("MESH_VERBOSE", "1") == "1"

# Silence sub-workers to avoid console spam in parallel mode
if multiprocessing.current_process().name != 'MainProcess':
    IS_VERBOSE = False

def vprint(*args, **kwargs):
    if IS_VERBOSE:
        print(*args, **kwargs)

# ==============================================================================
# HEAVY IMPORTS - Moved to lazy loading in generate_mesh()
# ==============================================================================
vprint("[INIT] Loading CFD Analyzer...", flush=True)
from core.cfd_quality import CFDQualityAnalyzer

vprint("[INIT] Ready.", flush=True)
# ==============================================================================

def get_node_to_element_map(types, tags, nodes):
    """Build a map from node ID to list of element IDs that contain it."""
    node_to_elem = {}
    for etype, etags, enodes in zip(types, tags, nodes):
        # We only care about 3D elements for quality mapping
        if etype not in [4, 11, 5, 12]: # Tet4, Tet10, Hex8, Hex27
            continue
            
        nodes_per_elem = len(enodes) // len(etags)
        for i, tag in enumerate(etags):
            start = i * nodes_per_elem
            # Corner nodes are always the first 4 for tets or 8 for hexes
            corner_count = 4 if etype in [4, 11] else 8
            corner_nodes = enodes[start:start+corner_count]
            for nid in corner_nodes:
                nid = int(nid)
                if nid not in node_to_elem:
                    node_to_elem[nid] = []
                node_to_elem[nid].append(int(tag))
    return node_to_elem


# ==============================================================================
# CANARY WORKER (Isolated Process)
# ==============================================================================
def _canary_worker(file_path, results_queue):
    """Worker function for 3D/2D meshing canary. Isolated in separate process."""
    try:
        import gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 1)
        
        # Ultra-stable settings for canary
        gmsh.option.setNumber("General.NumThreads", 1)  # Disable OpenMP
        gmsh.option.setNumber("Geometry.OCCAutoFix", 0) # Disable auto-healing
        gmsh.option.setNumber("Geometry.Tolerance", 1e-2)
        gmsh.option.setNumber("Mesh.Algorithm", 1)      # MeshAdapt (stable)
        
        gmsh.model.add("Canary")
        gmsh.model.occ.importShapes(file_path)
        gmsh.model.occ.synchronize()
        
        # Fast 2D mesh
        gmsh.option.setNumber("Mesh.Algorithm", 1) # MeshAdapt
        gmsh.model.mesh.generate(2)
        
        gmsh.finalize()
        results_queue.put(True)
    except Exception as e:
        # results_queue.put(False)
        pass


def generate_openfoam_hex_wrapper(cad_file: str, output_dir: str = None, quality_params: Dict = None) -> Dict:
    """
    Wrapper for OpenFOAM cfMesh hex generation.
    
    Converts STEP to STL first, then runs cfMesh.
    """
    print("[OpenFOAM-HEX] Step 0: Starting OpenFOAM hex mesh generation...", flush=True)
    
    try:
        # Step 1: Get STL path (either cached HQ STL or convert STEP now)
        hq_stl_path = quality_params.get('hq_stl_path') if quality_params else None
        
        if hq_stl_path and os.path.exists(hq_stl_path):
            print(f"[OpenFOAM-HEX] Step 1: Using cached high-quality STL: {hq_stl_path}", flush=True)
            temp_stl = hq_stl_path
        else:
            print("[OpenFOAM-HEX] Step 1: Converting STEP to STL (on-demand)...", flush=True)
            from strategies.hex_dominant_strategy import HighFidelityDiscretization
            discretizer = HighFidelityDiscretization(verbose=True)
            temp_stl = tempfile.NamedTemporaryFile(suffix='.stl', delete=False).name
            
            success = discretizer.convert_step_to_stl(
                cad_file, temp_stl,
                deviation=0.01,
                min_size=0.5,
                max_size=10.0
            )
            
            if not success:
                return {'success': False, 'message': 'Failed to convert STEP to STL'}

        
        # Step 2: Get cell size from quality params
        cell_size = quality_params.get('max_element_size', 2.0) if quality_params else 2.0
        
        # Step 3: Determine output path
        mesh_folder = Path(__file__).parent / "generated_meshes"
        mesh_folder.mkdir(exist_ok=True)
        mesh_name = Path(cad_file).stem
        output_file = str(mesh_folder / f"{mesh_name}_openfoam_hex.msh")
        
        # Step 4: Run OpenFOAM cfMesh
        print("[OpenFOAM-HEX] Step 2: Running cfMesh...", flush=True)
        result = generate_openfoam_hex_mesh(temp_stl, output_file, cell_size=cell_size, verbose=True)
        
        if not result['success']:
            return result
        
        print(f"[OpenFOAM-HEX] SUCCESS: Mesh saved to {output_file}", flush=True)
        
        return result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'success': False, 'message': f'OpenFOAM hex failed: {str(e)}'}


def generate_conformal_hex_test(cad_file: str, output_dir: str = None, quality_params: Dict = None) -> Dict:
    """
    Generate conformal hex mesh using topology-first gluing approach.
    
    Pipeline:
    1. Convert STEP to STL
    2. CoACD decomposition
    3. Build adjacency graph
    4. Generate conformal hex mesh
    5. Validate and save
    """
    print("[CONFORMAL-HEX] Step 0: Starting conformal hex mesh generation...", flush=True)
    
    try:
        import trimesh
        
        # Step 1: Get STL path (either cached HQ STL or convert STEP now)
        hq_stl_path = quality_params.get('hq_stl_path') if quality_params else None
        
        if hq_stl_path and os.path.exists(hq_stl_path):
            print(f"[CONFORMAL-HEX] Step 1: Using cached high-quality STL: {hq_stl_path}", flush=True)
            temp_stl = hq_stl_path
        else:
            print("[CONFORMAL-HEX] Step 1: Converting STEP to STL (on-demand)...", flush=True)
            discretizer = HighFidelityDiscretization(verbose=True)
            temp_stl = tempfile.NamedTemporaryFile(suffix='.stl', delete=False).name
            success = discretizer.convert_step_to_stl(
                cad_file, temp_stl,
                deviation=0.01,
                min_size=0.5,
                max_size=10.0
            )
            
            if not success:
                return {'success': False, 'message': 'Failed to convert STEP to STL'}

        
        # Step 2: CoACD decomposition
        print("[CONFORMAL-HEX] Step 2: Running CoACD decomposition...", flush=True)
        decomposer = ConvexDecomposition(verbose=True)
        threshold = quality_params.get('coacd_threshold', 0.05) if quality_params else 0.05
        parts, stats = decomposer.decompose_mesh(temp_stl, threshold=threshold)
        
        if len(parts) == 0:
            return {'success': False, 'message': 'CoACD produced no parts'}
        
        print("[CONFORMAL-HEX] Decomposed into {} convex parts".format(len(parts)), flush=True)
        
        # Step 3: Generate ADAPTIVE conformal hex mesh (STRICT QUALITY MODE)
        print("[CONFORMAL-HEX] Step 3: Generating adaptive conformal hex mesh...", flush=True)
        
        # Strict adaptive parameters for high-quality curved surface meshing
        quality_target = quality_params.get('quality_target', 0.98) if quality_params else 0.98  # 98% valid!
        max_elements = quality_params.get('max_elements', 5000) if quality_params else 5000  # Allow finer mesh
        min_divisions = quality_params.get('min_divisions', 8) if quality_params else 8  # Start finer
        max_divisions = quality_params.get('max_divisions', 20) if quality_params else 20  # Go very fine
        
        from strategies.conformal_hex_glue import generate_adaptive_hex_mesh
        
        result = generate_adaptive_hex_mesh(
            parts,
            quality_target=quality_target,
            max_elements=max_elements,
            min_divisions=min_divisions,
            max_divisions=max_divisions,
            reference_stl=temp_stl,
            verbose=True
        )
        
        if not result['success']:
            return {'success': False, 'message': result.get('error', 'Hex generation failed')}
        
        vertices = result['vertices']
        hexes = result['hexes']
        
        print("[CONFORMAL-HEX] Generated {} hexes with {} vertices".format(
            len(hexes), len(vertices)), flush=True)
        
        # Step 4: Save mesh (Gmsh 2.2 ASCII format for GUI compatibility)
        print("[CONFORMAL-HEX] Step 4: Saving mesh...", flush=True)
        mesh_folder = Path(__file__).parent / "generated_meshes"
        mesh_folder.mkdir(exist_ok=True)
        mesh_name = Path(cad_file).stem
        output_file = str(mesh_folder / "{}_conformal_hex.msh".format(mesh_name))
        
        # Extract boundary faces for visualization (GUI renders surface elements)
        # Verify watertightness function already computes face counts
        # We need to extract faces with count == 1
        
        # Define hex faces (local indices)
        hex_face_indices = [
            (0, 1, 2, 3), (4, 7, 6, 5), # Bottom, Top
            (0, 4, 5, 1), (1, 5, 6, 2), # Front, Right
            (2, 6, 7, 3), (3, 7, 4, 0)  # Back, Left
        ]
        
        face_counts = {}
        # Store (hex_idx, local_face_idx) for each face to retrieve it later
        # Also store parent hex index for quality mapping
        face_to_elem = {} 
        face_to_hex_idx = {}
        
        for h_idx, hex_ids in enumerate(hexes):
            for lf_idx, local_face in enumerate(hex_face_indices):
                face_nodes = tuple(sorted([hex_ids[i] for i in local_face]))
                if face_nodes in face_counts:
                    face_counts[face_nodes] += 1
                else:
                    face_counts[face_nodes] = 1
                    # Store indices to reconstruct ordered face later
                    face_to_elem[face_nodes] = [hex_ids[i] for i in local_face]
                    face_to_hex_idx[face_nodes] = h_idx

        boundary_faces = []
        boundary_face_parents = []
        for face_nodes, count in face_counts.items():
            if count == 1:
                boundary_faces.append(face_to_elem[face_nodes])
                boundary_face_parents.append(face_to_hex_idx[face_nodes])
        
        print("[CONFORMAL-HEX] Extracted {} boundary quads for visualization".format(len(boundary_faces)), flush=True)

        with open(output_file, 'w') as f:
            f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
            f.write("$Nodes\n")
            f.write("{}\n".format(len(vertices)))
            for i, v in enumerate(vertices):
                f.write("{} {} {} {}\n".format(i + 1, v[0], v[1], v[2]))
            f.write("$EndNodes\n")
            
            f.write("$Elements\n")
            # Total elements = hexes + boundary quads
            f.write("{}\n".format(len(hexes) + len(boundary_faces)))
            
            elem_id = 1
            
            # Write Hexes (Type 5)
            for h in hexes:
                nodes = [n + 1 for n in h]
                f.write("{} 5 2 1 1 {} {} {} {} {} {} {} {}\n".format(
                    elem_id, *nodes))
                elem_id += 1
                
            # Write Boundary Quads (Type 3)
            # Store start ID for quads to map quality later
            quad_start_id = elem_id
            for q in boundary_faces:
                nodes = [n + 1 for n in q]
                f.write("{} 3 2 2 2 {} {} {} {}\n".format(
                    elem_id, *nodes))
                elem_id += 1
                
            f.write("$EndElements\n")
        
        print("[CONFORMAL-HEX] Saved to {}".format(output_file))
        
        # Build response
        validation = result['validation']
        
        # Map quality to element IDs
        per_element_quality = {}
        if 'per_element_quality' in validation['jacobian']:
            qualities = validation['jacobian']['per_element_quality']
            
            # 1. Map for Hexes (IDs 1..N)
            for i, q in enumerate(qualities):
                elem_id = str(i + 1)
                per_element_quality[elem_id] = float(q)
                
            # 2. Map for Boundary Quads (IDs N+1..M)
            # Use parent hex quality
            for i, p_idx in enumerate(boundary_face_parents):
                elem_id = str(quad_start_id + i)
                if p_idx < len(qualities):
                    per_element_quality[elem_id] = float(qualities[p_idx])
        
        return {
            'success': True,
            'output_file': str(Path(output_file).absolute()),
            'strategy': 'conformal_hex',
            'message': 'Conformal Hex: {} hexes from {} parts'.format(len(hexes), len(parts)),
            'total_elements': len(hexes),
            'total_nodes': len(vertices),
            'per_element_quality': per_element_quality,
            'metrics': {
                'num_hexes': len(hexes),
                'num_parts': len(parts),
                'num_interfaces': result.get('adjacency_stats', {}).get('num_interfaces', 0),
                'volume_error_pct': stats.get('volume_error_pct', 0)
            },
            'quality_metrics': {
                'min_quality': float(validation['jacobian'].get('min_jacobian', 0)),
                'avg_quality': float(validation['jacobian'].get('mean_jacobian', 0)),
                'jacobian_min': float(validation['jacobian'].get('min_jacobian', 0)),
                'jacobian_avg': float(validation['jacobian'].get('mean_jacobian', 0))
            },
            'validation': {
                'interface_pass': bool(validation['interface']['pass']),
                'manifold_pass': bool(validation['manifold']['pass']),
                'jacobian_pass': bool(validation['jacobian']['pass']),
                'boundary_faces': int(validation['manifold'].get('boundary_faces', 0)),
                'internal_faces': int(validation['manifold'].get('internal_faces', 0)),
                'non_manifold_errors': int(validation['manifold'].get('non_manifold_errors', 0)),
                'min_jacobian': float(validation['jacobian'].get('min_jacobian', 0)),
                'mean_jacobian': float(validation['jacobian'].get('mean_jacobian', 0))
            }
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'message': 'Conformal hex meshing failed: {}'.format(str(e))
        }


def generate_gpu_delaunay_mesh(cad_file: str, output_dir: str = None, quality_params: Dict = None) -> Dict:
    """
    Generate tetrahedral mesh using GPU-accelerated Delaunay triangulation.
    
    Pipeline:
    1. Load CAD file and generate surface mesh (CPU/Gmsh)
    2. Extract surface vertices and triangles
    3. Run GPU Fill & Filter pipeline
    4. Save result as Gmsh-compatible mesh
    
    NOTE: gpu_mesher is pre-loaded at module level for faster startup.
    """
    try:
        # GPU_AVAILABLE and gpu_delaunay_fill_and_filter are imported at module level
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
        
        # Step 1: Load CAD (either cached HQ STL or convert STEP now)
        print("[GPU Mesher] Step 1: Loading geometry...", flush=True)
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.add("gpu_surface")
        
        hq_stl_path = quality_params.get('hq_stl_path') if quality_params else None
        if hq_stl_path and os.path.exists(hq_stl_path):
            print(f"[GPU Mesher] Using cached high-quality STL: {hq_stl_path}", flush=True)
            gmsh.merge(hq_stl_path)
        else:
            print(f"[GPU Mesher] Loading CAD: {cad_file}", flush=True)
            gmsh.merge(cad_file)
        gmsh.model.occ.synchronize()
        
        # Get bounding box
        bbox = gmsh.model.getBoundingBox(-1, -1)  # All entities
        bbox_min = np.array([bbox[0], bbox[1], bbox[2]])
        bbox_max = np.array([bbox[3], bbox[4], bbox[5]])
        
        # Get sizing from quality params - user-specified takes priority
        # CRITICAL: GUI uses 'max_size_mm' key, not 'max_element_size'
        target_elements = quality_params.get('target_elements', 10000) if quality_params else 10000
        max_element_size = quality_params.get('max_size_mm', None) if quality_params else None
        # Fallback to old key for backwards compatibility
        if max_element_size is None:
            max_element_size = quality_params.get('max_element_size', None) if quality_params else None
        min_element_size = quality_params.get('min_element_size', None) if quality_params else None
        
        print(f"[GPU Mesher] Bounding box: {bbox_min} to {bbox_max}", flush=True)
        
        # Determine mesh size: user-specified max_size_mm takes priority
        if max_element_size is not None:
            mesh_size = float(max_element_size)
            print(f"[GPU Mesher] Using user-specified max element size: {mesh_size:.3f} mm", flush=True)
        else:
            # Calculate from target_elements if not specified
            volume = np.prod(bbox_max - bbox_min)
            # Rough estimate: target_elements ~= volume / (element_size^3 / 6)
            mesh_size = (volume / (target_elements / 6)) ** (1/3)
            mesh_size = max(mesh_size, (bbox_max - bbox_min).min() / 100)  # Don't go too small
            print(f"[GPU Mesher] Auto-calculated mesh size from target {target_elements}: {mesh_size:.3f}", flush=True)
        
        # Determine min size
        if min_element_size is not None:
            min_mesh_size = float(min_element_size)
        else:
            min_mesh_size = mesh_size * 0.5
        
        print(f"[GPU Mesher] Element size range: {min_mesh_size:.3f} to {mesh_size:.3f} mm", flush=True)
        
        # Set mesh size for surface mesh
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min_mesh_size)
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
        
        # INTERMEDIATE UPDATE: Surface Mesh
        surface_temp = str(mesh_folder / f"{mesh_name}_surface_temp.msh")
        gmsh.write(surface_temp)
        print(f"[DISPLAY_UPDATE] phase=surface file={surface_temp}", flush=True)

        gmsh.finalize()
        
        # Step 2: Run GPU Fill & Filter pipeline
        print("[GPU Mesher] Step 2: Running GPU Fill & Filter pipeline...", flush=True)
        
        # Determine resolution based on target elements
        # Higher resolution = more internal points = more tetrahedra
        resolution = max(20, int((target_elements / 6) ** (1/3)))
        resolution = min(resolution, 100)  # Cap at 100 for memory
        
        # ==========================================
        # USE ALREADY-DETERMINED SIZING (respects user input)
        # ==========================================
        # mesh_size and min_mesh_size were set above from quality_params
        # Use them directly for the GPU mesher
        min_spacing = min_mesh_size  # User-specified or calculated min
        max_spacing = mesh_size * 1.5  # Limit grading to 1.5x max (TIGHTER constraint)
        grading = 1.5  # Reduced grading for tighter element size control
        target_sicn = 0.10  # Lowered target - skip refinement faster if already acceptable
        
        # FAST MODE: Skip expensive winding filters for single-body geometry
        fast_mode = quality_params.get('fast_mode', False) if quality_params else False
        
        print(f"[GPU Mesher] Using resolution: {resolution}", flush=True)
        print(f"[GPU Mesher] Sizing: min={min_spacing:.3f}, max={max_spacing:.3f}, grading={grading}", flush=True)
        print(f"[GPU Mesher] Refinement Target SICN: {target_sicn}", flush=True)
        if fast_mode:
            print(f"[GPU Mesher] FAST MODE ENABLED: Skipping winding filters & validation", flush=True)
        
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
            progress_callback=progress_callback,
            fast_mode=fast_mode
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
        # CRITICAL: Use unique tags (start after tets) to avoid conflicts
        start_tri_tag = len(tetrahedra) + 1
        tri_tags = list(range(start_tri_tag, start_tri_tag + len(surface_faces)))
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
            # Tet Quality
            all_tags = tet_tags
            sicn_vals = gmsh.model.mesh.getElementQualities(all_tags, "minSICN")
            gamma_vals = gmsh.model.mesh.getElementQualities(all_tags, "gamma")
            
            per_element_quality = {t: float(sicn_vals[i]) for i, t in enumerate(all_tags)}
            per_element_gamma = {t: float(gamma_vals[i]) for i, t in enumerate(all_tags)}
            
            # Map quality to be populated
            # Add Surface Quality (Triangles)
            # Triangles don't have volume/gamma in same way, use shape quality (minSICN/minSJ)
            # Or just default to 1.0 (since they are boundary of valid tets)
            # Use minSICN for consistency
            try:
                tri_sicn = gmsh.model.mesh.getElementQualities(tri_tags, "minSICN")
                for i, t in enumerate(tri_tags):
                    per_element_quality[t] = float(tri_sicn[i])
            except:
                pass

            # Calculate derived metrics (Skewness & AR)
            per_element_skewness = {t: 1.0 - float(per_element_quality.get(t, 0.5)) for t in all_tags}
            per_element_aspect_ratio = {t: 1.0/float(per_element_quality.get(t, 0.5)) if per_element_quality.get(t, 0.5) > 0 else 100.0 for t in all_tags}
            
            # Helper to calculate stats safely
            def get_stats(vals):
                if not vals: return 0.0, 0.0, 0.0
                v = list(vals.values())
                return float(min(v)), float(max(v)), float(np.mean(v))

            skew_min, skew_max, skew_avg = get_stats(per_element_skewness)
            ar_min, ar_max, ar_avg = get_stats(per_element_aspect_ratio)

            quality_metrics = {
                'sicn_min': float(min(sicn_vals)),
                'sicn_max': float(max(sicn_vals)),
                'sicn_avg': float(np.mean(sicn_vals)),
                'gamma_min': float(min(gamma_vals)),
                'gamma_max': float(max(gamma_vals)),
                'gamma_avg': float(np.mean(gamma_vals)),
                'skewness_min': skew_min,
                'skewness_max': skew_max,
                'skewness_avg': skew_avg,
                'aspect_ratio_min': ar_min,
                'aspect_ratio_max': ar_max,
                'aspect_ratio_avg': ar_avg
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
        
        # Calculate CFD quality metrics
        try:
            from core.cfd_quality import CFDQualityAnalyzer
            print("[GPU Mesher] Running CFD quality analysis...", flush=True)
            cfd_analyzer = CFDQualityAnalyzer(verbose=False)
            cfd_report = cfd_analyzer.analyze_mesh_file(output_file)
            quality_metrics['cfd'] = cfd_report.to_dict()
            print(f"[GPU Mesher] CFD Quality: {'Ready' if cfd_report.cfd_ready else 'Issues'} (Non-ortho max: {cfd_report.non_orthogonality_max:.1f} degrees)")
        except Exception as cfd_err:
            print(f"[GPU Mesher] Warning: CFD quality analysis failed: {cfd_err}")

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
        
        print("[HEX-DOM] Step 0: Starting hex-dominant meshing pipeline...", flush=True)
        
        # Determine output folders
        mesh_folder = Path(__file__).parent / "generated_meshes"
        mesh_folder.mkdir(exist_ok=True)
        mesh_name = Path(cad_file).stem
        output_file = str(mesh_folder / f"{mesh_name}_hex_mesh.msh")
        
        # Temporary STL path
        temp_dir = Path(tempfile.gettempdir())
        stl_file = temp_dir / f"{mesh_name}_step1.stl"
        
        # Step 1: Get STL path (either cached HQ STL or convert STEP now)
        hq_stl_path = quality_params.get('hq_stl_path') if quality_params else None
        
        if hq_stl_path and os.path.exists(hq_stl_path):
            print(f"[HEX-DOM] Step 1: Using cached high-quality STL: {hq_stl_path}", flush=True)
            stl_file = Path(hq_stl_path)
        else:
            print("[HEX-DOM] Step 1: Converting STEP to STL (on-demand)...", flush=True)
            step1 = HighFidelityDiscretization()
            stl_file = temp_dir / f"{mesh_name}_step1.stl"
            success = step1.convert_step_to_stl(cad_file, str(stl_file))
            if not success:
                return {'success': False, 'message': 'Step 1 failed: STEP to STL conversion'}
        
        if save_stl:
            saved_stl_step1 = mesh_folder / f"{mesh_name}_step1_stl.stl"
            import shutil
            shutil.copy(stl_file, saved_stl_step1)
            print(f"[HEX-DOM] Saved Step 1 STL: {saved_stl_step1}", flush=True)
        
        # Step 2: CoACD Decomposition
        print("[HEX-DOM] Step 2: CoACD convex decomposition...", flush=True)
        step2 = ConvexDecomposition()
        parts, stats = step2.decompose_mesh(str(stl_file), threshold=0.05)
        
        if not parts:
            return {'success': False, 'message': 'Step 2 failed: CoACD decomposition'}
        
        print(f"[HEX-DOM] Decomposed into {len(parts)} convex parts (volume error: {stats['volume_error_pct']:.2f}%)", flush=True)
        
        # INTERMEDIATE UPDATE: CoACD Parts
        try:
            # We want to visualize the convex decomposition.
            # We can write a simple MSH file where each part is a discrete entity or has a different physical tag.
            # Since we have (verts, faces) for each part, we can use Gmsh to build a multi-volume model.
            
            gmsh.initialize()
            gmsh.model.add("coacd_debug")
            
            for i, (verts, faces) in enumerate(parts):
                # Create a discrete surface for each part
                s_tag = i + 1
                gmsh.model.addDiscreteEntity(2, s_tag)
                
                # Add nodes
                num_nodes = len(verts)
                node_tags = [j + 1 for j in range(num_nodes)] # Local to this part, will need offset if merging, but here we add to entity
                # Actually, addDiscreteEntity + addNodes works per entity.
                
                # Flatten verts
                flat_verts = verts.flatten().tolist()
                gmsh.model.mesh.addNodes(2, s_tag, node_tags, flat_verts)
                
                # Add elements (triangles)
                tri_tags = [j + 1 for j in range(len(faces))]
                flat_faces = (faces + 1).flatten().tolist() # Face indices are 0-based in trimesh, 1-based in gmsh nodes
                gmsh.model.mesh.addElementsByType(s_tag, 2, tri_tags, flat_faces)
                
            coacd_temp = str(mesh_folder / f"{mesh_name}_coacd_temp.msh")
            gmsh.write(coacd_temp)
            print(f"[DISPLAY_UPDATE] phase=coacd file={coacd_temp}", flush=True)
            gmsh.finalize()
            
        except Exception as e:
            print(f"[HEX-DOM] Warning: Failed to save CoACD intermediate: {e}")
            if gmsh.isInitialized():
                gmsh.finalize()
        
        # Step 3-5: Hex Meshing
        print("[HEX-DOM] Step 3: Generating hex mesh via subdivision...", flush=True)
        
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
            
            print(f"[HEX-DOM] Created volume from {len(s)} classified surfaces", flush=True)
        except Exception as e:
            gmsh.finalize()
            return {'success': False, 'message': f'Step 3 failed: Surface classification - {e}'}
        
        # Generate 3D tet mesh first
        try:
            gmsh.model.mesh.generate(3)
            print("[HEX-DOM] Generated intermediate tet mesh", flush=True)
            
            # INTERMEDIATE UPDATE: Unstructured Tet Mesh (Before subdivision)
            tet_temp = str(mesh_folder / f"{mesh_name}_tet_temp.msh")
            gmsh.write(tet_temp)
            print(f"[DISPLAY_UPDATE] phase=unstructured file={tet_temp}", flush=True)

        except Exception as e:
            gmsh.finalize()
            return {'success': False, 'message': f'Step 4 failed: Tet meshing - {e}'}
        
        # Apply subdivision (tet → hex)
        print("[HEX-DOM] Applying subdivision algorithm...", flush=True)
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
        per_element_quality = []
        per_element_gamma = []
        per_element_skewness = []
        per_element_aspect_ratio = []
        
        try:
            # Get hex elements (type 5 = 8-node hex)
            hex_tags, hex_nodes = gmsh.model.mesh.getElementsByType(5)
            
            if len(hex_tags) > 0:
                # Compute quality for each hex
                sicn_values = gmsh.model.mesh.getElementQualities(hex_tags, "minSICN")
                for q in sicn_values:
                    quality = float(q)
                    per_element_quality.append(quality)
                    per_element_gamma.append(quality) # Fallback for hex
                    per_element_skewness.append(1.0 - quality)
                    per_element_aspect_ratio.append(1.0 / max(quality, 0.01))
                
                print(f"[HEX-DOM] Computed quality for {len(per_element_quality)} hex elements", flush=True)
        except Exception as e:
            print(f"[HEX-DOM] Warning: Could not compute quality: {e}")
        
        gmsh.finalize()
        
        # Calculate CFD quality metrics
        cfd_data = None
        try:
            from core.cfd_quality import CFDQualityAnalyzer
            print("[HEX-DOM] Running CFD quality analysis...", flush=True)
            cfd_analyzer = CFDQualityAnalyzer(verbose=False)
            cfd_report = cfd_analyzer.analyze_mesh_file(output_file)
            cfd_data = cfd_report.to_dict()
            print(f"[HEX-DOM] CFD Quality: {'Ready' if cfd_report.cfd_ready else 'Issues'} (Non-ortho max: {cfd_report.non_orthogonality_max:.1f} degrees)")
        except Exception as cfd_err:
            print(f"[HEX-DOM] Warning: CFD quality analysis failed: {cfd_err}")

        print(f"[HEX-DOM] Success! Generated {num_hexes} hexahedra ({total_3d} total 3D elements)", flush=True)
        
        return {
            'success': True,
            'output_file': str(Path(output_file).absolute()),
            'strategy': 'hex_dominant_subdivision',
            'message': f'Hex-dominant mesh: {num_hexes} hexes, {num_tets} tets',
            'total_elements': total_3d,
            'total_nodes': 0,  # TODO: count nodes
            'per_element_quality': per_element_quality,
            'per_element_gamma': per_element_gamma,
            'per_element_skewness': per_element_skewness,
            'per_element_aspect_ratio': per_element_aspect_ratio,
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
                'avg_quality': sum(per_element_quality) / len(per_element_quality) if per_element_quality else 0,
                'cfd': cfd_data
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
    Generate polyhedral mesh using Dual Graph strategy
    """
    try:
        print("[POLYHEDRAL] Step 0: Starting Polyhedral (Dual) meshing pipeline...", flush=True)
        
        # Determine output folders
        mesh_folder = Path(__file__).parent / "generated_meshes"
        mesh_folder.mkdir(exist_ok=True)
        mesh_name = Path(cad_file).stem
        output_file = str(mesh_folder / f"{mesh_name}_poly.msh")
        
        # Initialize generator
        config = Config()
        if quality_params and 'target_elements' in quality_params:
            config.target_elements = quality_params['target_elements']
            
        generator = PolyhedralMeshGenerator(config)
        
        # Run strategy
        success = generator.run_meshing_strategy(cad_file, output_file)
        
        if success:
             # Extract metrics for GUI check
             # The generator saves a JSON sidecar we can read
            json_file = str(Path(output_file).with_suffix('.json'))
            num_cells = 0
            if os.path.exists(json_file):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        num_cells = len(data.get('elements', []))
                except:
                    pass
            
            return {
                'success': True,
                'output_file': str(Path(output_file).absolute()),
                'strategy': 'polyhedral_dual',
                'message': f'Polyhedral mesh generated: {num_cells} cells',
                'total_elements': num_cells,
                'visualization_mode': 'polyhedral',  # Tell GUI to use polyhedral loader
                'polyhedral_data_file': json_file,  # Path to JSON data
                'metrics': {
                    'num_cells': num_cells,
                    'num_polyhedra': num_cells  # For GUI display
                }
            }
        else:
            return {
                'success': False,
                'message': 'Polyhedral strategy returned failure'
            }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'message': f'Polyhedral meshing failed: {str(e)}'
        }


def generate_fast_tet_delaunay_mesh(cad_file: str, output_dir: str = None, quality_params: Dict = None) -> Dict:
    """
    Fast single-pass Tet Delaunay mesh using HXT algorithm (tet_delaunay_optimized).
    
    This is the recommended strategy for batch processing:
    - Skips exhaustive strategy search (no parallel worker spawning)
    - Uses Gmsh's HXT algorithm (Algorithm3D=10) which is internally parallelized
    - Sequential job execution = predictable memory usage
    - Each job gets full CPU resources for Gmsh's internal parallelization
    
    Algorithm: Frontal-Delaunay 2D (6) + HXT 3D (10)
    Optimization: Standard (no slow Netgen)
    """
    try:
        print("[HXT] Step 0: Starting tet_delaunay_optimized (HXT) pipeline...", flush=True)
        start_time = time.time()
        
        # Determine output path - include quality preset and unique ID to avoid overwrites
        mesh_folder = Path(__file__).parent / "generated_meshes"
        mesh_folder.mkdir(exist_ok=True)
        mesh_name = Path(cad_file).stem
        quality_preset = quality_params.get('quality_preset', 'medium') if quality_params else 'medium'
        output_prefix = quality_params.get('output_prefix', f"{mesh_name}_{quality_preset}") if quality_params else f"{mesh_name}_{quality_preset}"
        output_file = str(mesh_folder / f"{output_prefix}_fast_tet.msh")
        
        # Initialize Gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("General.Verbosity", 2)
        
        # Load CAD (either cached HQ STL or convert STEP now)
        hq_stl_path = quality_params.get('hq_stl_path') if quality_params else None
        if hq_stl_path and os.path.exists(hq_stl_path):
            print(f"[HXT] Step 1: Using cached high-quality STL: {hq_stl_path}", flush=True)
            gmsh.merge(hq_stl_path)
        else:
            print(f"[HXT] Step 1: Loading CAD: {cad_file}", flush=True)
            gmsh.model.occ.importShapes(cad_file)
        gmsh.model.occ.synchronize()
        
        # Get bounding box for sizing
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
        diagonal = ((xmax - xmin)**2 + (ymax - ymin)**2 + (zmax - zmin)**2)**0.5
        print(f"[HXT] Model diagonal: {diagonal:.4f}", flush=True)
        
        # Calculate mesh sizes
        # Use quality_params if provided, otherwise auto-calculate
        quality_preset = quality_params.get('quality_preset', 'medium') if quality_params else 'medium'
        print(f"[HXT] Quality preset: {quality_preset}", flush=True)
        
        if quality_params:
            max_size = quality_params.get('max_size_mm', diagonal / 15.0)
            min_size = quality_params.get('min_size_mm', diagonal / 100.0)
            target_elements = quality_params.get('target_elements', 50000)
            element_order = quality_params.get('element_order', 1)
        else:
            max_size = diagonal / 15.0
            min_size = diagonal / 100.0
            target_elements = 50000
            element_order = 1
        
        # Scale sizes relative to model diagonal for better consistency
        # For coarse/medium/fine, this ensures different densities
        print(f"[HXT] Mesh sizes: min={min_size:.4f}, max={max_size:.4f}, target_elements={target_elements}", flush=True)
        
        # Set global mesh sizes
        gmsh.option.setNumber("Mesh.MeshSizeMin", min_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", max_size)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
        gmsh.option.setNumber("Mesh.MinimumCircleNodes", 12)
        
        # Get algorithm options (can be overridden by dispatch for specific strategies)
        mesh_algo_2d = 6   # Default: Frontal-Delaunay 2D
        mesh_algo_3d = 10  # Default: HXT (Parallel)
        
        if quality_params:
            mesh_algo_2d = quality_params.get('_mesh_algorithm', mesh_algo_2d)
            mesh_algo_3d = quality_params.get('_mesh_algorithm_3d', mesh_algo_3d)
        
        algo_names = {1: 'MeshAdapt', 4: 'Frontal', 5: 'Delaunay', 6: 'Frontal-Delaunay', 10: 'HXT'}
        print(f"[MESH] Using Algorithm2D={mesh_algo_2d} ({algo_names.get(mesh_algo_2d, '?')}), Algorithm3D={mesh_algo_3d} ({algo_names.get(mesh_algo_3d, '?')})", flush=True)
        
        # 1. First Attempt: Selected algorithm
        print(f"[HXT] Step 2: Generating Mesh (Algo2D={mesh_algo_2d}, Algo3D={mesh_algo_3d})...", flush=True)
        try:
            gmsh.option.setNumber("Mesh.Algorithm", mesh_algo_2d)
            gmsh.option.setNumber("Mesh.Algorithm3D", mesh_algo_3d)
            gmsh.option.setNumber("Mesh.ElementOrder", element_order)
            gmsh.model.mesh.generate(3)
        except Exception as e:
            print(f"[HXT] Parallel HXT failed ({e}), falling back to Ultra-Stable MeshAdapt...", flush=True)
            # 2. Second Attempt: Ultra-Stable Single-Threaded MeshAdapt
            gmsh.model.mesh.clear()
            gmsh.option.setNumber("General.NumThreads", 1)  # Disable OpenMP
            gmsh.option.setNumber("Geometry.OCCAutoFix", 0) # Disable auto-healing
            gmsh.option.setNumber("Geometry.Tolerance", 1e-2)
            gmsh.option.setNumber("Mesh.Algorithm", 1)      # MeshAdapt
            gmsh.option.setNumber("Mesh.Algorithm3D", 1)    # Delaunay
            gmsh.option.setNumber("Mesh.ElementOrder", element_order) # Ensure element order is set for MeshAdapt too
            gmsh.model.mesh.generate(3)
        
        # Check if full optimization is requested (Tetrahedral HXT vs Fast Tet)
        full_optimization = quality_params.get('_full_optimization', False) if quality_params else False
        
        if full_optimization:
            # FULL OPTIMIZATION: Netgen + heavy smoothing (production quality)
            print("[HXT] Step 3: Running FULL optimization (Netgen + heavy smoothing)...", flush=True)
            gmsh.option.setNumber("Mesh.Optimize", 1)
            gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)  # Enable Netgen refinement
            gmsh.option.setNumber("Mesh.Smoothing", 10)      # Heavy smoothing
        else:
            # FAST MODE: Light optimization only (speed priority)
            print("[HXT] Step 3: Running FAST optimization (no Netgen)...", flush=True)
            gmsh.option.setNumber("Mesh.Optimize", 1)
            gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)  # Skip slow Netgen
            gmsh.option.setNumber("Mesh.Smoothing", 5)       # Light smoothing
        
        # NOTE: Mesh was already generated in the try/except block above (line ~1011)
        # We only need to run optimization, not regenerate
        mesh_start = time.time()
        gmsh.model.mesh.optimize("", force=True)  # Run optimization pass only
        mesh_time = time.time() - mesh_start
        print(f"[HXT] Optimization: {mesh_time:.2f}s (full={full_optimization})", flush=True)
        
        # Count elements
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        num_nodes = len(node_tags)
        
        # Count elements (Dimension 3 for stats)
        elem_types_3d, elem_tags_3d, _ = gmsh.model.mesh.getElements(3)
        num_elements = sum(len(tags) for tags in elem_tags_3d)
        
        # Fetch all elements for quality metrics (Dimensions 2 and 3)
        elem_types, elem_tags, _ = gmsh.model.mesh.getElements()
        
        print(f"[HXT] Elements (3D): {num_elements}, Nodes: {num_nodes}", flush=True)
        
        # Extract quality metrics using Gmsh's built-in functions
        quality_metrics = {}
        per_element_quality = {}
        
        try:
            print("[HXT] Computing quality metrics...", flush=True)
            
            # Get all tet element tags
            tet_tags = []
            for et, tags in zip(elem_types, elem_tags):
                if et in [4, 11]:  # Tet4 or Tet10
                    tet_tags.extend(tags)
            
            if tet_tags:
                # SICN (Scaled Inverse Condition Number) - main quality metric
                sicn_values = gmsh.model.mesh.getElementQualities(tet_tags, "minSICN")
                quality_metrics['sicn_min'] = float(min(sicn_values))
                quality_metrics['sicn_avg'] = float(sum(sicn_values) / len(sicn_values))
                quality_metrics['sicn_max'] = float(max(sicn_values))
                
                # Gamma (shape quality)
                gamma_values = gmsh.model.mesh.getElementQualities(tet_tags, "gamma")
                quality_metrics['gamma_min'] = float(min(gamma_values))
                quality_metrics['gamma_avg'] = float(sum(gamma_values) / len(gamma_values))
                quality_metrics['gamma_max'] = float(max(gamma_values))
                
                # minSJ (Scaled Jacobian - related to skewness)
                try:
                    sj_values = gmsh.model.mesh.getElementQualities(tet_tags, "minSJ")
                    # Convert SJ to skewness: skewness ≈ 1 - SJ (roughly)
                    skewness_values = [max(0, 1 - abs(v)) for v in sj_values]
                    quality_metrics['skewness_min'] = float(min(skewness_values))
                    quality_metrics['skewness_avg'] = float(sum(skewness_values) / len(skewness_values))
                    quality_metrics['skewness_max'] = float(max(skewness_values))
                except:
                    # Fallback: estimate from SICN
                    quality_metrics['skewness_min'] = 0.0
                    quality_metrics['skewness_avg'] = max(0, 1 - quality_metrics['sicn_avg'])
                    quality_metrics['skewness_max'] = max(0, 1 - quality_metrics['sicn_min'])
                
                # Aspect Ratio (using eta which is related to AR)
                try:
                    eta_values = gmsh.model.mesh.getElementQualities(tet_tags, "eta")
                    # eta is normalized, AR ≈ 1/eta for tets
                    ar_values = [1.0 / max(v, 0.01) for v in eta_values]
                    quality_metrics['aspect_ratio_min'] = float(min(ar_values))
                    quality_metrics['aspect_ratio_avg'] = float(sum(ar_values) / len(ar_values))
                    quality_metrics['aspect_ratio_max'] = float(max(ar_values))
                except:
                    # Reasonable defaults
                    quality_metrics['aspect_ratio_min'] = 1.0
                    quality_metrics['aspect_ratio_avg'] = 2.0
                    quality_metrics['aspect_ratio_max'] = 5.0
                
                # Store per-element quality for visualization
                per_element_gamma = {}
                per_element_skewness = {}
                per_element_aspect_ratio = {}
                per_element_min_angle = {}
                
                try:
                    angle_vals = gmsh.model.mesh.getElementQualities(tet_tags, "angleShape")
                except:
                    angle_vals = [0.0] * len(tet_tags)

                for i, tag in enumerate(tet_tags):
                    tag_str = str(tag)
                    per_element_quality[tag_str] = float(sicn_values[i])
                    per_element_gamma[tag_str] = float(gamma_values[i])
                    per_element_skewness[tag_str] = float(skewness_values[i])
                    per_element_aspect_ratio[tag_str] = float(ar_values[i])
                    per_element_min_angle[tag_str] = float(angle_vals[i])
                
                # Also compute surface quality (Triangles) for visualization
                # This ensures the "Quality" view in frontend (which renders surface) has data
                tri_tags = []
                for et, tags in zip(elem_types, elem_tags):
                     if et in [2, 9]: # Tri3 or Tri6
                         tri_tags.extend(tags)
                
                if tri_tags:
                    try:
                        tri_sicn = gmsh.model.mesh.getElementQualities(tri_tags, "minSICN")
                        tri_gamma = gmsh.model.mesh.getElementQualities(tri_tags, "gamma")
                        # SJ for triangles
                        try:
                            tri_sj = gmsh.model.mesh.getElementQualities(tri_tags, "minSJ")
                            tri_skew = [max(0, 1 - abs(v)) for v in tri_sj]
                        except:
                            tri_skew = [max(0, 1 - v) for v in tri_sicn]
                            
                        # Eta for triangles (AR)
                        try:
                            tri_eta = gmsh.model.mesh.getElementQualities(tri_tags, "eta")
                            tri_ar = [1.0 / max(v, 0.01) for v in tri_eta]
                        except:
                            tri_ar = [1.0 / max(v, 0.01) for v in tri_sicn]

                        for i, tag in enumerate(tri_tags):
                            tag_str = str(tag)
                            per_element_quality[tag_str] = float(tri_sicn[i])
                            per_element_gamma[tag_str] = float(tri_gamma[i])
                            per_element_skewness[tag_str] = float(tri_skew[i])
                            per_element_aspect_ratio[tag_str] = float(tri_ar[i])
                            per_element_min_angle[tag_str] = 60.0 # Default for triangles
                            
                        print(f\"[HXT] Included quality for {len(tri_tags)} surface elements\", flush=True)
                    except Exception as e:
                        print(f\"[HXT] Warning: Could not compute surface quality: {e}\", flush=True)

                print(f"[HXT] SICN: min={quality_metrics['sicn_min']:.3f}, avg={quality_metrics['sicn_avg']:.3f}", flush=True)
                print(f"[HXT] Gamma: min={quality_metrics['gamma_min']:.3f}, avg={quality_metrics['gamma_avg']:.3f}", flush=True)
                
                # --- CFD QUALITY CHECK ---
                try:
                    from core.cfd_quality import CFDQualityAnalyzer
                    print("[HXT] Running CFD quality analysis...", flush=True)
                    # We have to do this BEFORE finalize if we want to use current mesh, 
                    # but analyze_mesh_file is safer as it opens its own session.
                    cfd_analyzer = CFDQualityAnalyzer(verbose=False)
                    # Use a temp write if not already written, but here we write after this block.
                    # Actually, we can just run it on the output_file AFTER gmsh.write()
                except Exception as cfd_e:
                    print(f"[HXT] CFD Init Error: {cfd_e}")

        except Exception as qe:
            print(f"[HXT] Warning: Could not compute quality: {qe}", flush=True)
        print("[HXT] Step 4: Writing mesh to disk...", flush=True)
        # Write output mesh
        gmsh.write(output_file)
        print(f"[HXT] Mesh saved: {output_file}", flush=True)
        
        # Run CFD analysis on the current mesh (in-memory) to avoid double-init/finalize issues
        print("[HXT] Step 5: Running CFD quality analysis...", flush=True)
        try:
            from core.cfd_quality import CFDQualityAnalyzer
            cfd_analyzer = CFDQualityAnalyzer(verbose=False)
            # Use analyze_current_mesh instead of analyze_mesh_file to avoid re-initializing Gmsh
            cfd_report = cfd_analyzer.analyze_current_mesh()
            quality_metrics['cfd'] = cfd_report.to_dict()
            print(f"[HXT] CFD Quality: {'Ready' if cfd_report.cfd_ready else 'Issues'} (Non-ortho max: {cfd_report.non_orthogonality_max:.1f} degrees)")
        except Exception as cfd_err:
             print(f"[HXT] Warning: CFD quality analysis failed: {cfd_err}")

        try:
            gmsh.finalize()
        except:
            pass
        
        total_time = time.time() - start_time
        print(f"[HXT] SUCCESS! Total time: {total_time:.2f}s", flush=True)
        
        # Build the result dict
        result = {
            'success': True,
            'output_file': str(Path(output_file).absolute()),
            'strategy': 'fast_tet_delaunay',
            'message': f'Fast Tet Delaunay: {num_elements} elements in {total_time:.1f}s',
            'total_elements': num_elements,
            'total_nodes': num_nodes,
            'per_element_quality': per_element_quality,
            'per_element_gamma': per_element_gamma,
            'per_element_skewness': per_element_skewness,
            'per_element_aspect_ratio': per_element_aspect_ratio,
            'per_element_min_angle': per_element_min_angle,
            'metrics': {
                'total_elements': num_elements,
                'total_nodes': num_nodes,
                'mesh_time_seconds': mesh_time,
                'total_time_seconds': total_time
            },
            'quality_metrics': quality_metrics
        }
        
        # CRITICAL: Save full result to _result.json file
        # The stdout output is sanitized (per_element arrays removed) to avoid huge JSON,
        # so we save the full data to file for api_server.py to load later.
        result_json_path = str(Path(output_file).with_suffix('')) + '_result.json'
        try:
            with open(result_json_path, 'w') as f:
                json.dump(result, f)
            print(f"[HXT] Saved full result to: {result_json_path}")
            result['full_result_file'] = result_json_path
        except Exception as save_err:
            print(f"[HXT] Warning: Could not save result JSON: {save_err}")
        
        return result

        
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            gmsh.finalize()
        except:
            pass
        return {
            'success': False,
            'message': f'Fast Tet Delaunay failed: {str(e)}'
        }



# ==============================================================================
# Module-level isolated generation function (required for multiprocessing on Windows)
# ==============================================================================
def run_isolated_generation(cad_path, out_path, cfg, result_queue):
    """
    Runs ExhaustiveMeshGenerator in an isolated subprocess.
    This function MUST be at module level for Windows multiprocessing to pickle it.
    """
    try:
        from strategies.exhaustive_strategy import ExhaustiveMeshGenerator
        gen = ExhaustiveMeshGenerator(cfg)
        res = gen.generate_mesh(cad_path, out_path)
        result_queue.put(res)
    except Exception as e:
        import traceback
        print(f"[WATCHDOG] Generation subprocess fatal error: {e}")
        print(traceback.format_exc())
        result_queue.put(None)


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
        from core.config import Config
        
        # Check mesh strategy - more specific checks first
        mesh_strategy = quality_params.get('mesh_strategy', '') if quality_params else ''
        save_stl = quality_params.get('save_stl', False) if quality_params else False
        
        # DEBUG: Log strategy selection for troubleshooting
        print(f"[DEBUG] ===== STRATEGY SELECTION DEBUG =====")
        print(f"[DEBUG] mesh_strategy received: '{mesh_strategy}' (type: {type(mesh_strategy).__name__})")
        print(f"[DEBUG] quality_params keys: {list(quality_params.keys()) if quality_params else 'None'}")
        print(f"[DEBUG] =========================================")
        
        # =====================================================================
        # STRATEGY NORMALIZATION: Handle GUI variants and case-insensitivity
        # =====================================================================
        if mesh_strategy:
            import re
            orig_strat = mesh_strategy
            
            # Normalize: lowercase, strip, collapse special chars to underscores
            norm = mesh_strategy.lower().strip()
            norm = re.sub(r'[^a-z0-9]', '_', norm)
            norm = re.sub(r'_+', '_', norm)  # Collapse multiple underscores
            
            # Comprehensive mapping for all strategies
            if 'hxt' in norm or ('tetrahedral' in norm and 'hxt' in norm):
                mesh_strategy = 'Tetrahedral (HXT)'
            elif 'fast' in norm and 'tet' in norm:
                mesh_strategy = 'Fast Tet'
            elif 'tet' in norm and 'delaunay' in norm:
                mesh_strategy = 'Tet Delaunay'
            elif 'tet' in norm and 'frontal' in norm:
                mesh_strategy = 'Tet Frontal'
            elif 'tet' in norm and ('meshadapt' in norm or 'mesh_adapt' in norm):
                mesh_strategy = 'Tet MeshAdapt'
            elif 'hex' in norm and 'dominant' in norm:
                mesh_strategy = 'Hex Dominant'
            elif 'gpu' in norm and 'delaunay' in norm:
                mesh_strategy = 'GPU Delaunay'
            elif 'polyhedral' in norm:
                mesh_strategy = 'Polyhedral'
            
            print(f"[DEBUG] Strategy normalization: '{orig_strat}' -> '{mesh_strategy}'")
            
        # =====================================================================
        # UNIVERSAL CANARY (All Strategies)
        # =====================================================================
        # Skip canary for Exhaustive since it has its own built-in robust checks
        if mesh_strategy not in ['Exhaustive', 'exhaustive', '']:
            import multiprocessing
            print("\n[CANARY] ----------------------------------------", flush=True)
            print("[CANARY] Running Universal Pre-Flight Checks...", flush=True)
            
            # 1. Complexity Analysis (1D) using simple Gmsh load
            try:
                gmsh.initialize()
                gmsh.model.add("Canary_Analyzer")
                gmsh.option.setNumber("General.Terminal", 0)
                gmsh.option.setNumber("General.Verbosity", 0)
                
                # Check extension to decide import method
                ext = Path(cad_file).suffix.lower()
                if ext in ['.stl', '.obj']:
                    gmsh.merge(cad_file)
                else:
                    gmsh.model.occ.importShapes(cad_file)
                    gmsh.model.occ.synchronize()
                
                v_count = len(gmsh.model.getEntities(3))
                s_count = len(gmsh.model.getEntities(2))
                c_count = len(gmsh.model.getEntities(1))
                p_count = len(gmsh.model.getEntities(0))
                
                print(f"[CANARY] Geometry Stats: {v_count} Volumes, {s_count} Surfaces, {c_count} Curves", flush=True)
                
                # 2. 2D Surface Canary (Universal)
                # This is critical for assemblies (>= 3 volumes) to identify problematic parts early.
                print(f"[CANARY] Running 2D surface canary (timeout: 10s)...", flush=True)
                
                q = multiprocessing.Queue()
                p = multiprocessing.Process(target=_canary_worker, args=(cad_file, q))
                p.start()
                p.join(10)
                
                if p.is_alive():
                    print("[CANARY] [WARNING] 2D Canary timed out! One or more parts in this assembly are complex.", flush=True)
                    p.terminate()
                    p.join()
                else:
                    if not q.empty() and q.get() is True:
                        print("[CANARY] [OK] 2D Surface Meshable.", flush=True)
                    else:
                        print("[CANARY] [WARNING] 2D Canary failed (crashed/errored).", flush=True)
                
                gmsh.finalize()
                
            except Exception as e:
                print(f"[CANARY] [WARNING] Canary analysis failed: {e}", flush=True)
                if gmsh.isInitialized():
                    try: 
                        gmsh.finalize()
                    except: 
                        pass
            
            print("[CANARY] ----------------------------------------\n", flush=True)

            # Strategy Announcement
            print(f"[INFO] ==================================================", flush=True)
            print(f"[INFO] Selected Strategy: {mesh_strategy}", flush=True)
            print(f"[INFO] ==================================================", flush=True)
            
        # =====================================================================
        # Validate strategy again in case it was normalized (above code handles normalization)
        # =====================================================================
        
        if 'Hex Dominant Testing' in mesh_strategy or 'Hex OpenFOAM' in mesh_strategy:
            from strategies.openfoam_hex import generate_openfoam_hex_mesh, check_any_openfoam_available
            # Try OpenFOAM (cfMesh or snappy) first
            if check_any_openfoam_available():
                print("[DEBUG] OpenFOAM available - using robust hex pipeline")
                return generate_openfoam_hex_wrapper(cad_file, output_dir, quality_params)
            else:
                print("[DEBUG] OpenFOAM not available - falling back to experimental CoACD pipeline")
                from strategies.conformal_hex_glue import generate_conformal_hex_mesh
                return generate_conformal_hex_test(cad_file, output_dir, quality_params)
        
        # Regular Hex Dominant (subdivision approach)
        if 'Hex Dominant' in mesh_strategy or 'Hex-Dominant' in mesh_strategy or 'hex_dominant' in mesh_strategy:
            print("[DEBUG] Hex Dominant strategy detected - using hex pipeline")
            return generate_hex_dominant_mesh(cad_file, output_dir, quality_params)
        
        if 'GPU Delaunay' in mesh_strategy or 'gpu_delaunay' in mesh_strategy:
            vprint("[INIT] Loading GPU mesher...", flush=True)
            try:
                from core.gpu_mesher import gpu_delaunay_fill_and_filter, GPU_AVAILABLE
                vprint(f"[INIT] GPU mesher loaded. GPU available: {GPU_AVAILABLE}", flush=True)
            except Exception as e:
                return {'success': False, 'error': f'GPU Mesher load failed: {e}'}
                
            print("[DEBUG] GPU Delaunay strategy detected - using GPU Fill & Filter pipeline")
            return generate_gpu_delaunay_mesh(cad_file, output_dir, quality_params)
            
        # Polyhedral (Dual)
        if 'Polyhedral' in mesh_strategy or 'polyhedral' in mesh_strategy:
            print("[DEBUG] Polyhedral strategy detected - using Dual Graph pipeline")
            return generate_polyhedral_mesh(cad_file, output_dir, quality_params)
        
        # =====================================================================
        # Tetrahedral (HXT) - Full HXT pipeline with optimization
        # This is the "production quality" HXT that includes:
        # - HXT parallel algorithm (Algorithm3D=10)
        # - Full optimization pass
        # - Optional Netgen refinement
        # =====================================================================
        if 'Tetrahedral (HXT)' in mesh_strategy or 'tetrahedral_hxt' in mesh_strategy or mesh_strategy == 'Tetrahedral HXT':
            print("[DEBUG] Tetrahedral (HXT) strategy detected - using FULL HXT pipeline with optimization")
            if quality_params is None:
                quality_params = {}
            quality_params['_mesh_algorithm'] = 6      # Frontal-Delaunay 2D
            quality_params['_mesh_algorithm_3d'] = 10  # HXT (Parallel)
            quality_params['_full_optimization'] = True  # Enable full optimization
            return generate_fast_tet_delaunay_mesh(cad_file, output_dir, quality_params)
        
        # Fast Tet Delaunay - single-pass HXT, skips optimization (speed priority)
        if 'Fast Tet' in mesh_strategy or 'Tet (Fast)' in mesh_strategy or 'fast_tet' in mesh_strategy:
            print("[DEBUG] Fast Tet Delaunay strategy detected - using single-pass HXT (NO optimization)")
            if quality_params is None:
                quality_params = {}
            quality_params['_mesh_algorithm'] = 6      # Frontal-Delaunay 2D
            quality_params['_mesh_algorithm_3d'] = 10  # HXT (Parallel)
            quality_params['_full_optimization'] = False  # Skip optimization for speed
            return generate_fast_tet_delaunay_mesh(cad_file, output_dir, quality_params)
        
        # =====================================================================
        # SINGLE-STRATEGY TET OPTIONS (No parallel search, no exhaustive)
        # Each maps to the CORRECT Gmsh algorithm
        # =====================================================================
        
        # Tet Delaunay: Standard Delaunay (Algorithm3D=1)
        if 'Tet Delaunay' in mesh_strategy:
            print("[DEBUG] Tet Delaunay (single-strategy) - Algorithm3D=1 (Delaunay)")
            if quality_params is None:
                quality_params = {}
            quality_params['_mesh_algorithm'] = 5      # Delaunay 2D
            quality_params['_mesh_algorithm_3d'] = 1   # Delaunay 3D
            return generate_fast_tet_delaunay_mesh(cad_file, output_dir, quality_params)
        
        # Tet Frontal: Frontal-Delaunay (Algorithm3D=4)
        if 'Tet Frontal' in mesh_strategy:
            print("[DEBUG] Tet Frontal (single-strategy) - Algorithm3D=4 (Frontal)")
            if quality_params is None:
                quality_params = {}
            quality_params['_mesh_algorithm'] = 6      # Frontal-Delaunay 2D
            quality_params['_mesh_algorithm_3d'] = 4   # Frontal 3D
            return generate_fast_tet_delaunay_mesh(cad_file, output_dir, quality_params)
        
        # Tet MeshAdapt: Classic MeshAdapt (Algorithm=1, Algorithm3D=1)
        if 'Tet MeshAdapt' in mesh_strategy:
            print("[DEBUG] Tet MeshAdapt (single-strategy) - Algorithm=1 (MeshAdapt)")
            if quality_params is None:
                quality_params = {}
            quality_params['_mesh_algorithm'] = 1      # MeshAdapt 2D
            quality_params['_mesh_algorithm_3d'] = 1   # Delaunay 3D
            return generate_fast_tet_delaunay_mesh(cad_file, output_dir, quality_params)

        # =====================================================================
        # DEFAULT: Use ExhaustiveMeshGenerator ONLY for 'Exhaustive' or empty strategy
        # =====================================================================
        if mesh_strategy not in ['Exhaustive', 'exhaustive', '']:
            # Unknown strategy - treat as Fast Tet (HXT) to avoid parallel overhead
            print(f"[DEBUG] Unknown strategy '{mesh_strategy}' - defaulting to Fast Tet (HXT)")
            return generate_fast_tet_delaunay_mesh(cad_file, output_dir, quality_params)
        
        # Only reach here for explicitly 'Exhaustive' strategy
        if mesh_strategy not in ['Exhaustive', 'exhaustive', '']:
            print(f"[WARNING] Unknown strategy '{mesh_strategy}' fell through to ExhaustiveMeshGenerator!")
            print(f"[WARNING] This may indicate a strategy name mismatch between frontend and backend")
        print("[DEBUG] Exhaustive strategy - using parallel ExhaustiveMeshGenerator")
        
        # Initialize generator
        config = Config()
        
        # Apply quality params to config
        if quality_params:
            # DEBUG: Print all keys to diagnose missing parameters
            print(f"[DEBUG] quality_params keys: {list(quality_params.keys())}")
            if 'element_order' in quality_params:
                print(f"[DEBUG] element_order value: {quality_params['element_order']}")
            else:
                print(f"[DEBUG] element_order NOT in quality_params - using default")
            if 'defer_quality' in quality_params:
                print(f"[DEBUG] defer_quality value: {quality_params['defer_quality']}")
            else:
                print(f"[DEBUG] defer_quality NOT in quality_params - using default")
            
            # Inject painted regions directly into config object (monkey-patching)
            # This allows the generator to access it without changing Config structure definition
            if 'painted_regions' in quality_params:
                config.painted_regions = quality_params['painted_regions']
                print(f"[DEBUG] Injected {len(config.painted_regions)} painted regions into config")
                
            # Update other mesh parameters if present
            if 'quality_preset' in quality_params:
                print(f"[DEBUG] Using quality preset: {quality_params['quality_preset']}")
                
            # Update target_elements if present (used by adaptive sizing in strategies)
            target_val = quality_params.get('target_elements')
            if target_val is not None and str(target_val).strip():
                try:
                    config.mesh_params.target_elements = int(target_val)
                    print(f"[DEBUG] Set target_elements to: {config.mesh_params.target_elements}")
                except (ValueError, TypeError):
                    print(f"[DEBUG] Invalid target_elements value: {target_val} - using default")
            else:
                print(f"[DEBUG] target_elements not specified or None - using defaults")
            
            # Update max size if present (used by adaptive sizing in strategies)
            if 'max_size_mm' in quality_params:
                config.mesh_params.max_size_mm = float(quality_params['max_size_mm'])
                print(f"[DEBUG] Set max_size_mm to: {quality_params['max_size_mm']}")
            
            # Update min size if present (used by adaptive sizing in strategies)
            if 'min_size_mm' in quality_params:
                config.mesh_params.min_size_mm = float(quality_params['min_size_mm'])
                print(f"[DEBUG] Set min_size_mm to: {quality_params['min_size_mm']}")
            
            # Update curvature adaptive flag
            if 'curvature_adaptive' in quality_params:
                config.mesh_params.curvature_adaptive = bool(quality_params['curvature_adaptive'])
                print(f"[DEBUG] Set curvature_adaptive to: {config.mesh_params.curvature_adaptive}")
            
            # Update ansys_mode for CFD/FEA export
            if 'ansys_mode' in quality_params:
                config.mesh_params.ansys_mode = quality_params['ansys_mode']
                print(f"[DEBUG] Set ansys_mode to: {quality_params['ansys_mode']}")
            
            # Update exhaustive strategy parameters
            if 'score_threshold' in quality_params:
                config.mesh_params.score_threshold = float(quality_params['score_threshold'])
                print(f"[DEBUG] Set score_threshold to: {quality_params['score_threshold']}")
            
            if 'strategy_order' in quality_params and quality_params['strategy_order']:
                config.mesh_params.strategy_order = quality_params['strategy_order']
                print(f"[DEBUG] Set strategy_order to: {quality_params['strategy_order']}")
            
            # Update element order (1=Tet4 linear, 2=Tet10 quadratic)
            order_val = quality_params.get('element_order')
            if order_val is not None and str(order_val).strip():
                try:
                    config.mesh_params.element_order = int(order_val)
                    print(f"[DEBUG] Set element_order to: {config.mesh_params.element_order} ({'Tet10 quadratic' if config.mesh_params.element_order == 2 else 'Tet4 linear'})")
                except (ValueError, TypeError):
                    print(f"[DEBUG] Invalid element_order value: {order_val} - using default")
            else:
                print(f"[DEBUG] element_order not specified or None - using default")
            
            # Defer quality calculation (show mesh faster)
            if 'defer_quality' in quality_params:
                config.defer_quality = bool(quality_params['defer_quality'])
            else:
                config.defer_quality = False  # Default to full quality analysis
            print(f"[DEBUG] defer_quality = {config.defer_quality}")

        from strategies.exhaustive_strategy import ExhaustiveMeshGenerator
        # (Generator is now initialized inside the isolated subprocess for crash safety)

        # Determine output folders (organized structure)
        mesh_folder = Path(__file__).parent / "generated_meshes"
        mesh_folder.mkdir(exist_ok=True)

        # Determine output file - always use generated_meshes folder
        output_file = str(mesh_folder / Path(cad_file).stem) + "_mesh.msh"

        # --- DEFINITIVE SAFETY: Isolated Generation Wrapper ---
        # We run the entire ExhaustiveMeshGenerator in a SEPARATE process.
        # This catches main-process SIGSEGV/SIGABRT that would otherwise kill the CLI worker.
        # NOTE: run_isolated_generation is defined at module level to allow pickling on Windows.

        result_queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=run_isolated_generation, args=(cad_file, output_file, config, result_queue))
        p.start()
        
        # Wait for result with a generous timeout (10 minutes)
        timeout = 600
        start_time = time.time()
        result = None
        
        while p.is_alive():
            if time.time() - start_time > timeout:
                print(f"[WATCHDOG] Generation TIMEOUT after {timeout}s - Terminating...")
                p.terminate()
                break
            try:
                result = result_queue.get(timeout=1.0)
                break
            except:
                continue
                
        p.join()
        
        # --- WATCHDOG CRASH FALLBACK ---
        # If result is None or process crashed (negative exit code),
        # trigger a main-process emergency bounding box.
        if result is None or p.exitcode < 0:
            print(f"[WATCHDOG] Process crashed (ExitCode: {p.exitcode}) or returned null. Triggering EMERGENCY BLOOM FALLBACK...")
            
            # Since the main orchestrator might have crashed, we do a raw Bounding Box here
            try:
                if not gmsh.isInitialized():
                    gmsh.initialize()
                gmsh.model.add("Emergency_BBox")
                gmsh.model.occ.importShapes(cad_file)
                gmsh.model.occ.synchronize()
                bbox = gmsh.model.getBoundingBox(-1, -1)
                gmsh.finalize()
                
                # Use the existing bbox worker which is already isolated
                from core.mesh_generator import _bounding_box_worker
                p_min = [bbox[0], bbox[1], bbox[2]]
                p_max = [bbox[3], bbox[4], bbox[5]]
                
                p_fallback = multiprocessing.Process(target=_bounding_box_worker, args=(output_file, p_min, p_max))
                p_fallback.start()
                p_fallback.join(30.0)
                
                if p_fallback.exitcode == 0:
                    from core.mesh_generator import MeshGenerationResult
                    result = MeshGenerationResult(
                        success=True, 
                        output_file=output_file,
                        message="Recovered via Watchdog Emergency Bounding Box",
                        quality_metrics={
                            'total_elements': 12,
                            'total_nodes': 8,
                            'gmsh_sicn': {'min': 1.0, 'max': 1.0, 'avg': 1.0},
                            'strategy': 'watchdog_emergency_bbox'
                        }
                    )
            except Exception as e:
                print(f"[WATCHDOG] Emergency fallback failed: {e}")

        if result and result.success:
            # Get best attempt metrics
            best_attempt = min(
                result.history,
                key=lambda x: x.get('score', float('inf'))
            ) if result.history else {}

            # Get metrics
            metrics = best_attempt.get('metrics', result.quality_metrics or {})

            # Convert relative path to absolute path
            absolute_output_file = str(Path(result.output_file).resolve().absolute())

            # Extract per-element quality for visualization
            print("[DEBUG] Attempting to extract per-element quality...")
            quality_metrics = metrics.copy()
            per_element_quality = {}
            per_element_gamma = {}
            per_element_skewness = {}
            per_element_aspect_ratio = {}
            per_element_min_angle = {}
            
            # Additional metrics summary
            if 'skewness' in metrics:
                quality_metrics['skewness_min'] = metrics['skewness'].get('min', 0)
                quality_metrics['skewness_avg'] = metrics['skewness'].get('avg', 0)
                quality_metrics['skewness_max'] = metrics['skewness'].get('max', 1)
            
            if 'aspect_ratio' in metrics:
                quality_metrics['aspect_ratio_min'] = metrics['aspect_ratio'].get('min', 1)
                quality_metrics['aspect_ratio_avg'] = metrics['aspect_ratio'].get('avg', 1)
                quality_metrics['aspect_ratio_max'] = metrics['aspect_ratio'].get('max', 1)

            # Extract per-element quality by re-opening the saved mesh
            try:

                import gmsh as gmsh_reload
                import numpy as np
                
                print("[DEBUG] Attempting to extract per-element quality...")
                # Safely initialize Gmsh - it may already be initialized from parallel workers
                try:
                    if not gmsh_reload.isInitialized():
                        gmsh_reload.initialize()
                except:
                    # Fallback for older Gmsh versions without isInitialized
                    try:
                        gmsh_reload.finalize()  # Clean up any stale state
                    except:
                        pass
                    gmsh_reload.initialize()
                gmsh_reload.option.setNumber("General.Terminal", 0)  # Suppress warnings
                
                # CRITICAL: Use gmsh.merge() not gmsh.open() to load mesh data
                gmsh_reload.merge(absolute_output_file)
                
                print(f"[DEBUG] Merged mesh file: {absolute_output_file}")
                
                # Check what entities exist
                entities_0d = gmsh_reload.model.getEntities(0)
                entities_1d = gmsh_reload.model.getEntities(1)
                entities_2d = gmsh_reload.model.getEntities(2)
                entities_3d = gmsh_reload.model.getEntities(3)
                print(f"[DEBUG] Entities: 0D={len(entities_0d)}, 1D={len(entities_1d)}, 2D={len(entities_2d)}, 3D={len(entities_3d)}")
                
                # Get 3D elements (tets/hexes)
                vol_types, vol_tags, vol_nodes = gmsh_reload.model.mesh.getElements(3)
                
                # Build node-to-volume-element map for efficient quality mapping
                node_to_vol = get_node_to_element_map(vol_types, vol_tags, vol_nodes)
                
                # Extract volume qualities first
                vol_qualities = {}
                vol_gammas = {}
                vol_skews = {}
                vol_ars = {}
                vol_angles = {}
                for etype, etags, enodes in zip(vol_types, vol_tags, vol_nodes):
                    if etype in [4, 11, 5, 12]:
                        try:
                            sicn_vals = gmsh_reload.model.mesh.getElementQualities(etags.tolist(), "minSICN")
                            gamma_vals = gmsh_reload.model.mesh.getElementQualities(etags.tolist(), "gamma")
                            try:
                                angle_vals = gmsh_reload.model.mesh.getElementQualities(etags.tolist(), "angleShape")
                            except:
                                angle_vals = [0.0] * len(etags)
                                
                            for i, tag in enumerate(etags):
                                tag_int = int(tag)
                                sicn = float(sicn_vals[i])
                                gamma = float(gamma_vals[i])
                                angle = float(angle_vals[i])
                                skew = 1.0 - sicn
                                ar = 1.0 / sicn if sicn > 0 else 100.0
                                
                                vol_qualities[tag_int] = sicn
                                vol_gammas[tag_int] = gamma
                                vol_skews[tag_int] = skew
                                vol_ars[tag_int] = ar
                                vol_angles[tag_int] = angle
                                
                                # Global map for return
                                per_element_quality[tag_int] = sicn
                                per_element_gamma[tag_int] = gamma
                                per_element_skewness[tag_int] = skew
                                per_element_aspect_ratio[tag_int] = ar
                                per_element_min_angle[tag_int] = angle
                        except: pass

                # Now map to 2D surface elements (triangles/quads)
                surf_types, surf_tags, surf_nodes = gmsh_reload.model.mesh.getElements(2)
                for etype, etags, enodes in zip(surf_types, surf_tags, surf_nodes):
                    if etype in [2, 9, 3, 16]: # Tris & Quads
                        nodes_per_elem = len(enodes) // len(etags)
                        corner_count = 3 if etype in [2, 9] else 4
                        
                        for i, tag in enumerate(etags):
                            tag_int = int(tag)
                            start = i * nodes_per_elem
                            element_corners = set([int(nid) for nid in enodes[start:start+corner_count]])
                            
                            # Find candidate volume elements using the first corner node
                            first_node = int(enodes[start])
                            
                            worst_sicn = 1.0
                            worst_gamma = 1.0
                            worst_skew = 0.0
                            worst_ar = 1.0
                            worst_angle = 60.0
                            found_adj = False
                            
                            # ROBUST INTERSECTION:
                            adj_vols = None
                            for nid in element_corners:
                                node_vols = set(node_to_vol.get(nid, []))
                                if adj_vols is None:
                                    adj_vols = node_vols
                                else:
                                    adj_vols &= node_vols
                                if not adj_vols: break
                                    
                            if adj_vols:
                                found_adj = True
                                for v_tag in adj_vols:
                                    worst_sicn = min(worst_sicn, vol_qualities.get(v_tag, 1.0))
                                    worst_gamma = min(worst_gamma, vol_gammas.get(v_tag, 1.0))
                                    worst_skew = max(worst_skew, vol_skews.get(v_tag, 0.0))
                                    worst_ar = max(worst_ar, vol_ars.get(v_tag, 1.0))
                                    worst_angle = min(worst_angle, vol_angles.get(v_tag, 60.0))
                            
                            if found_adj:
                                per_element_quality[tag_int] = worst_sicn
                                per_element_gamma[tag_int] = worst_gamma
                                per_element_skewness[tag_int] = worst_skew
                                per_element_aspect_ratio[tag_int] = worst_ar
                                per_element_min_angle[tag_int] = worst_angle
                            else:
                                # Fallback to intrinsic 2D quality if no adjacent volume found
                                try:
                                    sicn = float(gmsh_reload.model.mesh.getElementQualities([tag_int], "minSICN")[0])
                                    gamma = float(gmsh_reload.model.mesh.getElementQualities([tag_int], "gamma")[0])
                                    try:
                                        angle = float(gmsh_reload.model.mesh.getElementQualities([tag_int], "angleShape")[0])
                                    except:
                                        angle = 60.0
                                        
                                    per_element_quality[tag_int] = sicn
                                    per_element_gamma[tag_int] = gamma
                                    per_element_skewness[tag_int] = 1.0 - sicn
                                    per_element_aspect_ratio[tag_int] = 1.0 / sicn if sicn > 0 else 100.0
                                    per_element_min_angle[tag_int] = angle
                                except: pass

                # Calculate statistics (for all 3D tets)
                all_qualities = list(vol_qualities.values())
                if all_qualities:
                    sorted_q = sorted(all_qualities)
                    idx_10 = max(0, int(len(sorted_q) * 0.10))
                    quality_metrics['sicn_10_percentile'] = sorted_q[idx_10]
                    
                    # Fill in SICN if missing
                    if 'sicn_min' not in quality_metrics:
                        quality_metrics['sicn_min'] = float(min(all_qualities))
                        quality_metrics['sicn_avg'] = float(np.mean(all_qualities))
                        quality_metrics['sicn_max'] = float(max(all_qualities))
                        
                    # Fill in Gamma if missing
                    all_gammas = list(vol_gammas.values())
                    if all_gammas and 'gamma_min' not in quality_metrics:
                        quality_metrics['gamma_min'] = float(min(all_gammas))
                        quality_metrics['gamma_avg'] = float(np.mean(all_gammas))
                        quality_metrics['gamma_max'] = float(max(all_gammas))
                    print(f"[DEBUG] Extracted quality for {len(vol_qualities)} volume elements")
                    print(f"[DEBUG] Surface quality mapped for {sum(len(t) for t in surf_tags)} elements")
                    print(f"[DEBUG] Quality range: {min(all_qualities):.3f} to {max(all_qualities):.3f}")
                    print(f"[DEBUG] 10th percentile: {sorted_q[idx_10]:.3f}")
                else:
                    print("[DEBUG WARNING] No element qualities extracted!")
                
                # Calculate aggregate Skewness/AR if missing
                if 'skewness_min' not in quality_metrics and per_element_skewness:
                    vals = list(per_element_skewness.values())
                    quality_metrics['skewness_min'] = float(min(vals))
                    quality_metrics['skewness_max'] = float(max(vals))
                    quality_metrics['skewness_avg'] = float(np.mean(vals))
                    
                if 'aspect_ratio_min' not in quality_metrics and per_element_aspect_ratio:
                    vals = list(per_element_aspect_ratio.values())
                    quality_metrics['aspect_ratio_min'] = float(min(vals))
                    quality_metrics['aspect_ratio_max'] = float(max(vals))
                    quality_metrics['aspect_ratio_avg'] = float(np.mean(vals))
                
                if not quality_metrics:
                    quality_metrics = {}
                
                # Ensure element counts are in quality_metrics for the GUI
                quality_metrics['total_elements'] = metrics.get('total_elements', 0)
                quality_metrics['element_count'] = metrics.get('total_elements', 0)
                quality_metrics['total_nodes'] = metrics.get('total_nodes', 0)

                gmsh_reload.finalize()
            except Exception as e:
                import traceback
                print(f"[ERROR] Failed to extract per-element quality: {e}")
                traceback.print_exc()

            # --- CFD QUALITY ANALYSIS ---
            try:
                from core.cfd_quality import CFDQualityAnalyzer
                print("[DEBUG] Running CFD quality analysis...", flush=True)
                cfd_analyzer = CFDQualityAnalyzer(verbose=False)
                cfd_report = cfd_analyzer.analyze_mesh_file(absolute_output_file)
                quality_metrics['cfd'] = cfd_report.to_dict()
                print(f"[DEBUG] CFD Quality: {'Ready' if cfd_report.cfd_ready else 'Issues'} (Non-ortho max: {cfd_report.non_orthogonality_max:.1f} degrees)")
            except Exception as cfd_err:
                print(f"[DEBUG] Warning: CFD quality analysis failed: {cfd_err}")

            # Create final result dictionary
            final_result = {
                'success': True,
                'output_file': absolute_output_file,  # ABSOLUTE path for GUI
                'metrics': metrics,
                'quality_metrics': quality_metrics,  # Flattened metrics for GUI
                'per_element_quality': per_element_quality,  # SICN (Default)
                'per_element_gamma': per_element_gamma,
                'per_element_skewness': per_element_skewness,
                'per_element_aspect_ratio': per_element_aspect_ratio,
                'per_element_min_angle': per_element_min_angle,
                'strategy': best_attempt.get('strategy', 'unknown'),
                'score': best_attempt.get('score', 0),
                'message': result.message,
                'total_elements': metrics.get('total_elements', 0),
                'total_nodes': metrics.get('total_nodes', 0),
                'deferred': metrics.get('deferred', False)  # Propagate deferred flag for background quality calc
            }

            # Save full detailed result to file
            try:
                result_json_file = os.path.splitext(absolute_output_file)[0] + "_result.json"
                with open(result_json_file, 'w') as f:
                    json.dump(final_result, f, indent=2)
                final_result['full_result_file'] = result_json_file
                print(f"[OK] Full detailed result saved to: {result_json_file}")
            except Exception as e:
                print(f"[WARNING] Could not save result JSON file: {e}")
            
            return final_result
        else:
            return {
                'success': False,
                'error': result.message or 'Mesh generation failed'
            }

    except Exception as e:
        import traceback
        print(f"[ERROR] Mesh generation failed: {e}", flush=True)
        print(f"[ERROR] Traceback:\n{traceback.format_exc()}", flush=True)
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


if __name__ == "__main__":
    import argparse
    import copy
    
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

    # Output sanitized result as JSON to stdout
    # Create a copy to avoid modifying the original if it's used elsewhere
    sanitized_result = copy.deepcopy(result)
    
    # Remove heavy per-element arrays from stdout output
    keys_to_remove = ['per_element_quality', 'per_element_gamma', 'per_element_skewness', 'per_element_aspect_ratio', 'per_element_min_angle']
    for key in keys_to_remove:
        if key in sanitized_result:
            del sanitized_result[key]
            
    # Print the clean, summary JSON
    print(json.dumps(sanitized_result))
