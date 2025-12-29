import pyvista as pv
import tetgen
import numpy as np
import os
import time

# --- CONFIG ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STL_FILE = os.path.join(PROJECT_ROOT, "fixed_soup.stl")
FINAL_VTK = os.path.join(PROJECT_ROOT, "shell_connectivity_mesh.vtk")

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def main():
    t0 = time.time()
    log("STARTING SHELL-PRESERVATION MESHER...")
    
    if not os.path.exists(STL_FILE):
        log(f"[X] STL not found: {STL_FILE}")
        return

    log(f"   - Loading Soup STL: {os.path.basename(STL_FILE)}")
    soup = pv.read(STL_FILE)
    
    # 1. Connectivity on the raw soup
    log("   - Analyzing geometric connectivity (Shell Extraction)...")
    connected = soup.connectivity(extraction_mode='all')
    region_ids = connected.cell_data["RegionId"]
    n_islands = len(np.unique(region_ids))
    log(f"     [OK] Found {n_islands} distinct volumes in STL soup.")
    
    # 2. Independent Tetrahedralization
    log("   - Tetrahedralizing each volume independently for maximum robustness...")
    all_tets = []
    success_count = 0
    
    for i in range(n_islands):
        try:
            # Extract this specific body
            body = connected.threshold([i, i], scalars="RegionId")
            if body.n_cells == 0: continue
            
            # Convert to PolyData for TetGen
            surf = body.extract_surface()
            area = surf.area
            
            tet = tetgen.TetGen(surf)
            res = tet.tetrahedralize(switches='pq') 
            
            grid = pv.UnstructuredGrid({pv.CellType.TETRA: res[1]}, res[0])
            grid.cell_data["VolumeID"] = np.full(grid.n_cells, i)
            
            vol = grid.volume
            if i < 10 or vol > 1000:
                log(f"      - Body {i}: Surface Area={area:.2f}, Mesh Volume={vol:.2f}")
            
            all_tets.append(grid)
            success_count += 1
        except Exception as e:
            log(f"      [!] Failed to mesh Body {i}: {e}")
            # Optional: Fallback to voxel or just keep surface as reference?
            # For now, we skip to maintain volume as much as possible with successes.
            pass

    if not all_tets:
        log("[X] FAILURE: No volumes could be tetrahedralized.")
        return

    log(f"   - Successfully meshed {success_count} / {n_islands} bodies.")
    
    # Debug individual volumes
    total_raw_vol = sum(t.volume for t in all_tets)
    log(f"   - Sum of individual tetrahedral volumes: {total_raw_vol:.2f}")

    log("   - Concatenating all tetrahedral volumes...")
    # Using append() might be more robust for preserving cell data
    final_mesh = all_tets[0]
    for i in range(1, len(all_tets)):
        final_mesh = final_mesh.merge(all_tets[i])

    final_mesh.save(FINAL_VTK)
    log(f"[OK] DONE. Saved to {FINAL_VTK}")
    log(f"     Concatenated Mesh Volume: {final_mesh.volume:.2f}")
    log(f"[Finished] Total Time: {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
