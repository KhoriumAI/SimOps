import gmsh
import os

def export_alternatives(input_file):
    print(f"Reading {input_file}...")
    gmsh.initialize()
    gmsh.open(input_file)
    
    base_name = os.path.splitext(input_file)[0]
    
    # 1. Export Nastran BDF
    # Fluent supports this well. 
    # Important: SaveAll=0 (only physical groups)
    gmsh.option.setNumber("Mesh.SaveAll", 0)
    gmsh.option.setNumber("Mesh.BdfFieldFormat", 1) # Standard formatting
    
    bdf_file = base_name + "_nastran.bdf"
    print(f"Exporting to {bdf_file}...")
    gmsh.write(bdf_file)
    
    # 2. Export Abaqus INP
    inp_file = base_name + "_abaqus.inp"
    print(f"Exporting to {inp_file}...")
    gmsh.write(inp_file)
    
    # 3. Export VRML (Just in case, for geometry check)
    wrl_file = base_name + ".wrl"
    print(f"Exporting to {wrl_file}...")
    gmsh.write(wrl_file)

    gmsh.finalize()

if __name__ == "__main__":
    # Use the robust retopo file which we know is clean
    input_f = "C:/Users/Owner/Downloads/MeshPackageLean/apps/cli/generated_meshes/Cube_retopo.msh"
    export_alternatives(input_f)
