import gmsh
import pyvista as pv
import tetgen
import numpy as np
import os
import time
import sys
import shutil

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
STEP_FILE = os.path.join(PROJECT_ROOT, "cad_files", "00-CM240411_BIANCA-C00_HEATER_BOAR_0704A1.STEP")
TEMP_DIR = os.path.join(PROJECT_ROOT, "temp_volumes")
FINAL_MESH = os.path.join(PROJECT_ROOT, "robust_assembly.vtk")
GAP_SIZE = 1.0 

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def mesh_volume_isolated(vol_idx):
    """Meshes one volume by reloading STEP in a fresh Gmsh session."""
    output_file = os.path.join(TEMP_DIR, f"vol_{vol_idx}.vtk")
    if os.path.exists(output_file):
        return True # Skip already completed
    
    try:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.occ.importShapes(STEP_FILE)
        gmsh.model.occ.synchronize()
        
        volumes = gmsh.model.getEntities(3)
        if vol_idx >= len(volumes):
            gmsh.finalize()
            return False
            
        dim, tag = volumes[vol_idx]
        
        # Mesh settings
        gmsh.option.setNumber("Mesh.MeshSizeMin", GAP_SIZE)
        gmsh.option.setNumber("Mesh.MeshSizeMax", GAP_SIZE * 5.0)
        gmsh.option.setNumber("Mesh.Algorithm", 1) # MeshAdapt
        
        # Mesh only the surfaces of this volume
        # We use visibility to focus Gmsh
        gmsh.model.mesh.generate(2)
        
        # Extract triangles for this volume
        b_dim_tags = gmsh.model.getBoundary([(3, tag)], oriented=False, recursive=False)
        surf_tags = [t for d, t in b_dim_tags]
        
        vol_tri_faces = []
        for s_tag in surf_tags:
            _, _, elem_node_tags = gmsh.model.mesh.getElements(2, s_tag)
            if len(elem_node_tags) > 0:
                tri_tags = np.concatenate(elem_node_tags).reshape(-1, 3)
                vol_tri_faces.append(tri_tags)
        
        if len(vol_tri_faces) == 0:
            gmsh.finalize()
            return False
            
        all_vol_faces_tags = np.concatenate(vol_tri_faces)
        n_tags, n_coords, _ = gmsh.model.mesh.getNodes()
        all_nodes = np.array(n_coords).reshape(-1, 3)
        node_tag_map = {t: k for k, t in enumerate(n_tags)}
        
        unique_tags = np.unique(all_vol_faces_tags)
        local_node_tag_map = {t: j for j, t in enumerate(unique_tags)}
        local_nodes = np.array([all_nodes[node_tag_map[t]] for t in unique_tags])
        local_faces = np.vectorize(local_node_tag_map.get)(all_vol_faces_tags)
        
        gmsh.finalize() # Free Gmsh early
        
        # 2. TetGen (PyVista based)
        threes = np.full((local_faces.shape[0], 1), 3)
        padded_faces = np.hstack((threes, local_faces)).flatten().astype(np.int64)
        surf = pv.PolyData(local_nodes, padded_faces)
        
        tet = tetgen.TetGen(surf)
        res = tet.tetrahedralize(switches='pY')
        
        grid = pv.UnstructuredGrid({pv.CellType.TETRA: res[1]}, res[0])
        grid.cell_data["VolumeID"] = np.full(grid.n_cells, tag)
        grid.save(output_file)
        return True
    except Exception as e:
        if gmsh.isInitialized():
            gmsh.finalize()
        return False

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single_vol", type=int, default=-1)
    args = parser.parse_args()

    t0 = time.time()
    
    if args.single_vol != -1:
        # SINGLE VOLUME MODE
        if mesh_volume_isolated(args.single_vol):
            sys.exit(0)
        else:
            sys.exit(1)

    # FULL ASSEMBLY MODE (Batch)
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    
    # ... existing main logic ...
    log("ROBUST ASSEMBLY COMPLETE.")

if __name__ == "__main__":
    main()
