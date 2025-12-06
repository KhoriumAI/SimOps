import gmsh
import os

def generate_simple_box_test():
    gmsh.initialize()
    gmsh.model.add("simple_box")
    
    # 1. Create Geometry (10x10x10 box)
    gmsh.model.occ.addBox(0, 0, 0, 10, 10, 10, 1)
    gmsh.model.occ.synchronize()
    
    # 2. Add Physical Groups
    # Volume
    gmsh.model.addPhysicalGroup(3, [1], 1)
    gmsh.model.setPhysicalName(3, 1, "fluid")
    
    # Surface (all 6 faces)
    faces = gmsh.model.getEntities(2)
    face_tags = [f[1] for f in faces]
    gmsh.model.addPhysicalGroup(2, face_tags, 2)
    gmsh.model.setPhysicalName(2, 2, "wall")
    
    # 3. Mesh Settings
    gmsh.option.setNumber("Mesh.Algorithm3D", 1) # Delaunay
    gmsh.model.mesh.setOrder(1) # Linear
    
    # Generate Mesh
    gmsh.model.mesh.generate(3)
    
    # 4. Export (Try standard 2.2)
    output_file = "C:/Users/Owner/Downloads/MeshPackageLean/apps/cli/generated_meshes/Simple_Box_Test.msh"
    
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.Binary", 0)
    gmsh.option.setNumber("Mesh.SaveAll", 0)
    
    print(f"Exporting MVP test mesh to {output_file}...")
    gmsh.write(output_file)
    
    gmsh.finalize()

if __name__ == "__main__":
    generate_simple_box_test()
