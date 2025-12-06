import gmsh
import os

def export_v4(input_file, output_file):
    gmsh.initialize()
    gmsh.open(input_file)
    
    # Ensure Physical Groups are preserved/clean
    # The file input_file (Cube_retopo.msh) should already have them.
    # But let's double check.
    
    # MSH 4.1 ASCII
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.option.setNumber("Mesh.Binary", 0)
    gmsh.option.setNumber("Mesh.SaveAll", 0)
    
    print(f"Writing Gmsh 4.1 to {output_file}...")
    gmsh.write(output_file)
    gmsh.finalize()

if __name__ == "__main__":
    input_f = "C:/Users/Owner/Downloads/MeshPackageLean/apps/cli/generated_meshes/Cube_retopo.msh"
    output_f = "C:/Users/Owner/Downloads/MeshPackageLean/apps/cli/generated_meshes/Cube_v4.msh"
    export_v4(input_f, output_f)
