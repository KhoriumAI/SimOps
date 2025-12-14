"""
Mesh Worker Daemon
==================

A long-running subprocess that pre-loads all heavy modules and waits
for mesh generation commands. This eliminates the 3-5s startup delay
when generating meshes.

This script is managed by MeshWorkerPool and should not be run directly.
"""

import sys
import json
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ==============================================================================
# HEAVY IMPORTS - Done at startup to pre-warm the process
# ==============================================================================
print("[WARMING] Loading NumPy...", flush=True)
import numpy as np

print("[WARMING] Loading gmsh...", flush=True)
try:
    import gmsh
    GMSH_AVAILABLE = True
except ImportError:
    GMSH_AVAILABLE = False
    print("[WARMING] gmsh not available")

print("[WARMING] Loading CuPy...", flush=True)
try:
    import cupy as cp
    GPU_AVAILABLE = True
    # Warm the GPU by doing a small computation
    _ = cp.array([1, 2, 3])
    print(f"[WARMING] GPU ready: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
except ImportError:
    GPU_AVAILABLE = False
    cp = None
    print("[WARMING] CuPy not available (CPU mode)")

print("[WARMING] Loading mesh modules...", flush=True)
try:
    from core.gpu_mesher import gpu_delaunay_fill_and_filter, GPU_AVAILABLE as GPU_MESHER_AVAILABLE
    print(f"[WARMING] GPU Mesher: {GPU_MESHER_AVAILABLE}")
except ImportError as e:
    GPU_MESHER_AVAILABLE = False
    print(f"[WARMING] GPU Mesher not available: {e}")

try:
    from core.gpu_adaptive_refinement import AdaptiveGPURefinement
    print("[WARMING] Adaptive refinement loaded")
except ImportError:
    pass

print("[WARMING] Loading scipy...", flush=True)
try:
    from scipy.spatial import cKDTree
except ImportError:
    pass

# ==============================================================================
# SIGNAL READY
# ==============================================================================
print("[READY]", flush=True)


def handle_generate(cad_file: str, config: dict):
    """Handle a mesh generation request."""
    import time
    start_time = time.time()
    
    try:
        # Initialize gmsh
        if not GMSH_AVAILABLE:
            print("[ERROR] gmsh not available")
            return {'success': False, 'error': 'gmsh not available'}
            
        if not gmsh.isInitialized():
            gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("General.Verbosity", 3)
        
        # Load CAD
        print(f"[DAEMON] Loading CAD: {cad_file}", flush=True)
        gmsh.clear()
        gmsh.merge(cad_file)
        
        # Get mesh size from config
        mesh_size = config.get('target_size', 5.0)
        target_elements = config.get('target_elements', 10000)
        strategy = config.get('strategy', 'tetrahedral')
        output_dir = config.get('output_dir', str(Path(cad_file).parent))
        
        # Check if GPU strategy
        if 'gpu' in strategy.lower() and GPU_MESHER_AVAILABLE:
            print("[DAEMON] Using GPU Fill & Filter pipeline", flush=True)
            result = _handle_gpu_mesh(cad_file, config, output_dir)
        else:
            print("[DAEMON] Using standard Gmsh meshing", flush=True)
            result = _handle_gmsh_mesh(cad_file, config, output_dir, mesh_size)
            
        elapsed = time.time() - start_time
        result['elapsed_time'] = elapsed
        print(f"[DAEMON] Mesh completed in {elapsed:.2f}s", flush=True)
        
        return result
        
    except Exception as e:
        import traceback
        error_msg = f"{e}\n{traceback.format_exc()}"
        print(f"[ERROR] {error_msg}", flush=True)
        return {'success': False, 'error': str(e)}


def _handle_gpu_mesh(cad_file: str, config: dict, output_dir: str) -> dict:
    """Handle GPU mesh generation."""
    from core.gpu_mesher import gpu_delaunay_fill_and_filter
    
    # Get surface mesh from gmsh
    gmsh.model.mesh.generate(2)
    
    # Extract surface data
    node_tags, nodes, _ = gmsh.model.mesh.getNodes()
    nodes = np.array(nodes).reshape(-1, 3)
    
    element_types, element_tags, element_nodes = gmsh.model.mesh.getElements(dim=2)
    
    faces = []
    for i, etype in enumerate(element_types):
        if etype == 2:  # Triangle
            tri_nodes = np.array(element_nodes[i]).reshape(-1, 3) - 1
            faces.extend(tri_nodes.tolist())
    faces = np.array(faces)
    
    # Bounding box
    bbox = gmsh.model.getBoundingBox(-1, -1)
    bbox_min = np.array([bbox[0], bbox[1], bbox[2]])
    bbox_max = np.array([bbox[3], bbox[4], bbox[5]])
    
    # Run GPU meshing
    target_sicn = config.get('target_sicn', 0.15)
    target_elements = config.get('target_elements', 10000)
    
    def progress_callback(msg, pct):
        print(f"[GPU Mesher] {msg} ({pct}%)", flush=True)
    
    verts, tets, surf_faces = gpu_delaunay_fill_and_filter(
        nodes, faces, bbox_min, bbox_max,
        target_sicn=target_sicn,
        progress_callback=progress_callback
    )
    
    # Save result
    output_file = str(Path(output_dir) / (Path(cad_file).stem + "_gpu_mesh.msh"))
    _save_mesh_to_gmsh(verts, tets, output_file)
    
    return {
        'success': True,
        'output_file': output_file,
        'total_elements': len(tets),
        'total_nodes': len(verts)
    }


def _handle_gmsh_mesh(cad_file: str, config: dict, output_dir: str, mesh_size: float) -> dict:
    """Handle standard Gmsh meshing."""
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT
    
    gmsh.model.mesh.generate(3)
    
    # Get mesh stats
    node_tags, _, _ = gmsh.model.mesh.getNodes()
    element_types, element_tags, _ = gmsh.model.mesh.getElements(dim=3)
    
    total_elements = sum(len(tags) for tags in element_tags)
    
    # Save
    output_file = str(Path(output_dir) / (Path(cad_file).stem + "_mesh.msh"))
    gmsh.write(output_file)
    
    return {
        'success': True,
        'output_file': output_file,
        'total_elements': total_elements,
        'total_nodes': len(node_tags)
    }


def _save_mesh_to_gmsh(verts, tets, output_file):
    """Save mesh data to Gmsh format."""
    gmsh.clear()
    gmsh.model.add("gpu_mesh")
    
    # Add nodes
    node_tags = list(range(1, len(verts) + 1))
    gmsh.model.mesh.addNodes(3, 1, node_tags, verts.ravel().tolist())
    
    # Create physical group for volume
    gmsh.model.addPhysicalGroup(3, [1], 1)
    gmsh.model.setPhysicalName(3, 1, "Volume")
    
    # Add tetrahedra
    tet_tags = list(range(1, len(tets) + 1))
    tet_nodes = (tets + 1).ravel().tolist()  # 1-indexed
    gmsh.model.mesh.addElements(3, 1, [4], [tet_tags], [tet_nodes])
    
    gmsh.write(output_file)
    print(f"[DAEMON] Saved mesh to: {output_file}", flush=True)


def main():
    """Main daemon loop - read commands from stdin."""
    print("[DAEMON] Ready for commands", flush=True)
    
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
                
            line = line.strip()
            if not line:
                continue
                
            # Parse command
            try:
                command = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[ERROR] Invalid JSON: {e}", flush=True)
                continue
                
            action = command.get('action', '')
            
            if action == 'shutdown':
                print("[DAEMON] Shutting down", flush=True)
                break
                
            elif action == 'generate':
                cad_file = command.get('cad_file')
                config = command.get('config', {})
                
                if not cad_file:
                    print("[ERROR] No cad_file specified", flush=True)
                    continue
                    
                result = handle_generate(cad_file, config)
                
                # Send result as JSON on special line
                print(f"[RESULT]{json.dumps(result)}", flush=True)
                
            elif action == 'ping':
                print("[PONG]", flush=True)
                
            else:
                print(f"[ERROR] Unknown action: {action}", flush=True)
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[ERROR] {e}", flush=True)
            
    print("[DAEMON] Exiting", flush=True)


if __name__ == "__main__":
    main()
