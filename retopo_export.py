import gmsh
import os
import sys

def retopo_and_export(input_file, output_file):
    print(f"Opening {input_file}...")
    gmsh.initialize()
    gmsh.open(input_file)
    
    # 1. Clear existing physical groups (start fresh)
    gmsh.model.removePhysicalGroups(gmsh.model.getPhysicalGroups())
    
    # 2. Linearize (Tet10 -> Tet4)
    # This handles node renumbering and removal of mid-side nodes automatically
    print("Linearizing mesh (setOrder 1)...")
    gmsh.model.mesh.setOrder(1)
    
    # 3. Create Topology
    # This analyzes the discrete elements and creates Volume/Surface entities
    # and ensures connectivity is valid.
    print("Creating topology from discrete mesh...")
    gmsh.model.mesh.createTopology()
    
    # 4. Assign Physical Groups
    # Find Volume entities (there should be one for the cube)
    vols = gmsh.model.getEntities(3)
    if not vols:
        print("Error: No volume entities found after topology creation!")
        sys.exit(1)
    
    vol_tags = [e[1] for e in vols]
    p_vol = gmsh.model.addPhysicalGroup(3, vol_tags)
    gmsh.model.setPhysicalName(3, p_vol, "fluid")
    print(f"Created 'fluid' volume (Tags: {vol_tags})")
    
    # Find Surface entities (should be the boundaries)
    surfs = gmsh.model.getEntities(2)
    if not surfs:
        print("Error: No surface entities found! Topology failed to find skin?")
        # Try to generate specific classification if needed, but createTopology should work.
    else:
        surf_tags = [e[1] for e in surfs]
        p_surf = gmsh.model.addPhysicalGroup(2, surf_tags)
        gmsh.model.setPhysicalName(2, p_surf, "wall")
        print(f"Created 'wall' surface (Tags: {surf_tags})")

    # 5. Export
    # SaveAll=0 ensures we don't write stray points/curves, only what's in Physical Groups
    gmsh.option.setNumber("Mesh.SaveAll", 0)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.Binary", 0)
    
    print(f"Writing to {output_file}...")
    gmsh.write(output_file)
    
    gmsh.finalize()
    print("Success!")

if __name__ == "__main__":
    input_path = "C:/Users/Owner/Downloads/MeshPackageLean/apps/cli/generated_meshes/Cube_mesh.msh"
    output_path = "C:/Users/Owner/Downloads/MeshPackageLean/apps/cli/generated_meshes/Cube_retopo.msh"
    retopo_and_export(input_path, output_path)
