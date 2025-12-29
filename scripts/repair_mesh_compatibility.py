import gmsh
import os

PROJECT_ROOT = r'C:\Users\markm\Downloads\MeshPackageLean'
input_vtk = os.path.join(PROJECT_ROOT, "simulation_ready_defeatured.vtk")
output_msh = os.path.join(PROJECT_ROOT, "final_assembly_standard.msh")
output_vtk_fallback = os.path.join(PROJECT_ROOT, "final_assembly_v3.vtk")

print(f"Loading VTK: {input_vtk}")
gmsh.initialize()
gmsh.open(input_vtk)

# Export as Standard Gmsh v2.2 (ASCII)
print(f"Exporting to Standard Gmsh v2.2: {output_msh}")
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.option.setNumber("Mesh.Binary", 0)
gmsh.write(output_msh)

# Also copy/save VTK with matching name for viewer fallback
print(f"Saving VTK fallback: {output_vtk_fallback}")
gmsh.write(output_vtk_fallback)

gmsh.finalize()
print("Done.")
