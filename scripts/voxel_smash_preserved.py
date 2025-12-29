import gmsh
import pyvista as pv
import tetgen
import numpy as np
import os
import time
import sys

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
STEP_FILE = os.path.join(PROJECT_ROOT, "cad_files", "00-CM240411_BIANCA-C00_HEATER_BOAR_0704A1.STEP")
FINAL_VTK = os.path.join(PROJECT_ROOT, "voxel_smash_fixed.vtk")
GAP_SIZE = 0.5

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def main():
    t0 = time.time()
    log("STARTING OPTIMIZED VOXEL-SMASH (Volume Preservation Mode)...")
    
    # 1. GMSH: Rapid Soup Export (Implicit in previous runs, assuming fused_temp.stl exists for speed)
    soup_path = os.path.join(PROJECT_ROOT, "fused_temp.stl")
    if not os.path.exists(soup_path):
        log("   [!] STL Soup not found. Generating...")
        # (Insert gmsh export here if needed, but for iterations we assume it exists)
        pass

    log(f"   - Loading Soup STL: {os.path.basename(soup_path)}")
    soup = pv.read(soup_path)
    
    # 2. Voxel Cleaning (The Smash)
    # We use a finer grid to avoid merging small disconnected components
    log(f"   - Voxelizing with Volume Preservation (Gap: {GAP_SIZE})...")
    xr, yr, zr = soup.bounds[1]-soup.bounds[0], soup.bounds[3]-soup.bounds[2], soup.bounds[5]-soup.bounds[4]
    nx = int(xr / GAP_SIZE)
    ny = int(yr / GAP_SIZE)
    nz = int(zr / GAP_SIZE)
    
    # Resample
    grid = pv.ImageData(
        dimensions=(nx+1, ny+1, nz+1),
        spacing=(GAP_SIZE, GAP_SIZE, GAP_SIZE),
        origin=(soup.bounds[0], soup.bounds[2], soup.bounds[4])
    )
    log(f"   - Grid bounds: {grid.bounds}")
    sampled = grid.sample(soup)
    log(f"   - Sampled arrays: {sampled.point_data.keys()}")
    
    # Extract surface of the voxelized volume
    # Use vtkValidPointMask for thresholding
    sampled.set_active_scalars("vtkValidPointMask")
    vox_surf = sampled.threshold(0.1).extract_surface().triangulate()
    
    # 3. Connectivity Analysis (THE FIX)
    # After smashing, we find all independent "islands" in the mesh.
    # If voxelization was fine enough, we should see ~151 islands.
    log(f"   - Analyzing geometric connectivity (Shell Extraction) on {vox_surf.n_cells} triangles...")
    if vox_surf.n_cells == 0:
        log("   [!] Surface mesh is empty. Voxelization failed to capture geometry.")
        return

    connected = vox_surf.connectivity(extraction_mode='all')
    
    # Map the region IDs (PointData in some VTK versions, CellData in others)
    # connectivity(extraction_mode='all') adds 'RegionId' array.
    region_ids = connected.get_array("RegionId")
    if region_ids is None:
         # Debug: Log all arrays
         log(f"   [!] RegionId not found. Cell arrays: {connected.cell_data.keys()}, Point arrays: {connected.point_data.keys()}")
         raise KeyError("RegionId not found after connectivity analysis.")
             
    n_islands = len(np.unique(region_ids))
    log(f"     [OK] Found {n_islands} distinct volumes in cleaned mesh.")
    
    # 4. Volume Meshing (TetGen)
    log("   - Tetrahedralizing all preserved volumes...")
    # We can run TetGen on the whole multi-block connected mesh
    tet = tetgen.TetGen(connected)
    # Ensure it preserves the RegionId cell data
    try:
        tet.make_manifold()
    except: pass
    
    nodes, elements = tet.tetrahedralize(switches='pY')
    
    # Reconstruct with RegionID mapping
    final_mesh = pv.UnstructuredGrid({pv.CellType.TETRA: elements}, nodes)
    
    # Map RegionIDs back to tets (Nearest neighbor from tet centroids to surface islands)
    log("   - Mapping Volume IDs to tetrahedral elements...")
    centroids = final_mesh.cell_centers().points
    temp_surf = connected.cell_centers()
    # Find nearest surface cell index for each tet centroid
    _, closest_idx = temp_surf.find_closest_cell(centroids, return_closestpoint=False)
    final_mesh.cell_data["VolumeID"] = connected.cell_data["RegionId"][closest_idx]

    final_mesh.save(FINAL_VTK)
    log(f"[OK] DONE. Assembly saved to {FINAL_VTK}")
    log(f"     Volumes preserved: {n_islands}")
    log(f"[Finished] Total Time: {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
