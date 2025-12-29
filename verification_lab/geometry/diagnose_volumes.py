import gmsh
import pyvista as pv
import os
import time

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
STEP_FILE = os.path.join(PROJECT_ROOT, "cad_files", "00-CM240411_BIANCA-C00_HEATER_BOAR_0704A1.STEP")
STL_FILE = os.path.join(PROJECT_ROOT, "fused_temp.stl")
VTK_FILE = os.path.join(PROJECT_ROOT, "voxel_mesh.vtk")
PARALLEL_VTK_FILE = os.path.join(PROJECT_ROOT, "shell_connectivity_mesh.vtk")

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def main():
    import sys
    log("DIAGNOSING VOLUME PRESERVATION...")

    target_mesh = None
    if len(sys.argv) > 1:
        target_mesh = sys.argv[1]
        log(f"   - Target mesh provided: {target_mesh}")

    if target_mesh:
        if os.path.exists(target_mesh):
            log(f"   - Opening Target Mesh: {os.path.basename(target_mesh)}")
            mesh = pv.read(target_mesh)
            
            # Check if it's a surface or volume mesh
            if mesh.n_cells > 0:
                # Try volume first
                try:
                    vol = mesh.volume
                    log(f"     [Target] Total Volume: {vol:.2f}")
                except:
                    log(f"     [Target] Surface Area: {mesh.area:.2f} (Not a volume?)")
                
                # Check regions
                regions = mesh.connectivity()
                # If VolumeID exists, use it
                if "VolumeID" in mesh.cell_data:
                    n_regions = len(np.unique(mesh.cell_data["VolumeID"]))
                else:
                    n_regions = len(np.unique(regions.get_array("RegionId")))
                log(f"     [Target] Found {n_regions} regions/islands.")
        else:
            log(f"   [!] Target mesh not found at {target_mesh}")
        return # Skip the rest if specific mesh is provided
    if os.path.exists(STEP_FILE):
        log(f"   - Opening STEP: {os.path.basename(STEP_FILE)}")
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.occ.importShapes(STEP_FILE)
        gmsh.model.occ.synchronize()
        vols = gmsh.model.getEntities(3)
        
        total_step_vol = 0
        for v in vols:
            try:
                mass = gmsh.model.occ.getMass(v[0], v[1])
                total_step_vol += mass
            except: pass
            
        log(f"     [STEP] Found {len(vols)} volumes. Total Volume: {total_step_vol:.2f}")
        gmsh.finalize()
    else:
        log(f"   [!] STEP not found at {STEP_FILE}")

    # 2. Check Soup STL
    if os.path.exists(STL_FILE):
        log(f"   - Opening STL: {os.path.basename(STL_FILE)}")
        soup = pv.read(STL_FILE)
        shells = soup.connectivity()
        n_shells = len(np.unique(shells.get_array("RegionId")))
        soup_surf_area = soup.area
        log(f"     [STL] Found {n_shells} shells. Total Surface Area: {soup_surf_area:.2f}")
    else:
        log(f"   [!] STL not found at {STL_FILE}")

    # 3. Check Final VTK
    if os.path.exists(VTK_FILE):
        log(f"   - Opening VTK: {os.path.basename(VTK_FILE)}")
        mesh = pv.read(VTK_FILE)
        regions = mesh.connectivity()
        n_regions = len(np.unique(regions.get_array("RegionId")))
        final_vol = mesh.volume
        log(f"     [VTK] Found {n_regions} regions. Total Mesh Volume: {final_vol:.2f}")
        
        if 'total_step_vol' in locals() and total_step_vol > 0:
            survival = (final_vol / total_step_vol) * 100
            log(f"     [!] Volume Retention: {survival:.1f}%")
    # 4. Check Parallel VTK
    if os.path.exists(PARALLEL_VTK_FILE):
        log(f"   - Opening Parallel VTK: {os.path.basename(PARALLEL_VTK_FILE)}")
        pmesh = pv.read(PARALLEL_VTK_FILE)
        # It should already have VolumeID in cell_data
        if "VolumeID" in pmesh.cell_data:
            vids = pmesh.cell_data["VolumeID"]
            n_vids = len(np.unique(vids))
            log(f"     [Parallel VTK] Found {n_vids} tagged volume IDs.")
        else:
            regions = pmesh.connectivity()
            n_vids = len(np.unique(regions.get_array("RegionId")))
            log(f"     [Parallel VTK] Found {n_vids} connected regions.")
            
        p_vol = pmesh.volume
        log(f"     [Parallel VTK] Total Mesh Volume: {p_vol:.2f}")
        if 'total_step_vol' in locals() and total_step_vol > 0:
            survival = (p_vol / total_step_vol) * 100
            log(f"     [Parallel VTK] Volume Retention: {survival:.1f}%")
    else:
        log(f"   [!] Parallel VTK not found at {PARALLEL_VTK_FILE}")

import numpy as np
if __name__ == "__main__":
    main()
