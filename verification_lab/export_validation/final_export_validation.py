import gmsh
import os
import sys
import numpy as np

# Add project root to path for core imports if needed
PROJECT_ROOT = r'C:\Users\markm\Downloads\MeshPackageLean'
sys.path.insert(0, PROJECT_ROOT)

from core.write_fluent_mesh import write_fluent_msh

def log(msg):
    print(f"[CONVERTER] {msg}", flush=True)

def calculate_mesh_volume(points, tets):
    """Calculate total volume of tetrahedral mesh."""
    # Volume of tet = |(a-d) . ((b-d) x (c-d))| / 6
    p0 = points[tets[:, 0]]
    p1 = points[tets[:, 1]]
    p2 = points[tets[:, 2]]
    p3 = points[tets[:, 3]]
    
    v0 = p0 - p3
    v1 = p1 - p3
    v2 = p2 - p3
    
    cross = np.cross(v1, v2)
    dots = np.einsum('ij,ij->i', v0, cross)
    vol = np.sum(np.abs(dots)) / 6.0
    return vol

def run_conversion():
    input_vtk = os.path.join(PROJECT_ROOT, "simulation_ready_defeatured.vtk")
    if not os.path.exists(input_vtk):
        log(f"ERROR: File not found {input_vtk}")
        return

    log(f"Loading VTK: {input_vtk}")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.open(input_vtk)
    
    # 1. VERIFY ORIGINAL
    nodes = gmsh.model.mesh.getNodes()[1].reshape(-1, 3)
    elements = gmsh.model.mesh.getElements(3)
    tet_tags = elements[1][0]
    tet_nodes = elements[2][0].reshape(-1, 4) - 1 # 0-indexed
    
    orig_count = len(tet_tags)
    orig_vol = calculate_mesh_volume(nodes, tet_nodes)
    log(f"ORIGINAL VTK: {orig_count} elements, Volume = {orig_vol:.2f} mm^3")
    
    # 2. EXPORT MSH (ANYS FLUENT COMPATIBLE)
    output_msh = os.path.join(PROJECT_ROOT, "final_assembly_v3.msh")
    if os.path.exists(output_msh):
        log("SKIP: Fluent MSH already exists.")
    else:
        log(f"Exporting to FLUENT MSH: {output_msh}")
        try:
            write_fluent_msh(output_msh, nodes, {'tetra': tet_nodes})
            log("SUCCESS: Fluent MSH Exported.")
        except Exception as e:
            log(f"FAIL: Fluent MSH Export Failed: {e}")

    # 3. EXPORT CGNS
    output_cgns = os.path.join(PROJECT_ROOT, "final_assembly_v3.cgns")
    log(f"Exporting to CGNS: {output_cgns}")
    try:
        gmsh.write(output_cgns)
        log("SUCCESS: CGNS Exported.")
    except Exception as e:
        log(f"FAIL: CGNS Export Failed: {e}")

    # 4. EXPORT BDF (NASTRAN/ANSYS)
    output_bdf = os.path.join(PROJECT_ROOT, "final_assembly_v3.bdf")
    log(f"Exporting to BDF: {output_bdf}")
    try:
        # Nastran BDF settings
        gmsh.option.setNumber("Mesh.BdfFieldFormat", 1)
        gmsh.write(output_bdf)
        log("SUCCESS: BDF Exported.")
    except Exception as e:
        log(f"FAIL: BDF Export Failed: {e}")

    # 5. VALIDATE EXPORTS (Spot check Volume on one)
    log("Validating BDF integrity...")
    gmsh.clear()
    try:
        gmsh.merge(output_bdf)
        bdf_nodes = gmsh.model.mesh.getNodes()[1].reshape(-1, 3)
        bdf_elements = gmsh.model.mesh.getElements(3)
        bdf_tet_nodes = bdf_elements[2][0].reshape(-1, 4) - 1
        bdf_vol = calculate_mesh_volume(bdf_nodes, bdf_tet_nodes)
        
        diff = abs(bdf_vol - orig_vol) / orig_vol * 100
        log(f"VALIDATION BDF: Elements={len(bdf_elements[1][0])}, Volume={bdf_vol:.2f} mm^3 (Diff={diff:.4f}%)")
        
        if diff < 0.1:
            log("VERIFICATION PASSED: Volume conserved.")
        else:
            log("WARNING: Volume mismatch detected!")
            
    except Exception as e:
        log(f"Validation failed: {e}")

    gmsh.finalize()
    log("--- CONVERSION WORKFLOW COMPLETE ---")

if __name__ == "__main__":
    run_conversion()
