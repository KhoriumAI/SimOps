import gmsh
import sys
import os

def main():
    gmsh.initialize()
    gmsh.model.add("thermal_test")

    # Dimensions
    hs_L, hs_W, hs_H = 0.1, 0.1, 0.01
    chip_L, chip_W, chip_H = 0.02, 0.02, 0.005

    # Volumes
    hs_vol = gmsh.model.occ.addBox(-hs_L/2, -hs_W/2, -hs_H, hs_L, hs_W, hs_H)
    chip_vol = gmsh.model.occ.addBox(-chip_L/2, -chip_W/2, 0, chip_L, chip_W, chip_H)

    # Fragment to ensure conformant mesh at interface
    ov, ov_map = gmsh.model.occ.fragment([(3, hs_vol)], [(3, chip_vol)])
    gmsh.model.occ.synchronize()
    
    # ov contains the resulting entities. We need to identify which is which.
    # Usually, if we pass hs_vol and chip_vol, 'fragment' returns new entities.
    # But since they don't overlap (only touch), they should remain as they were, just faces split.
    # We can identify them by center of mass or bounding box.
    
    heatsink_tags = []
    chip_tags = []
    
    for dim, tag in ov:
        if dim == 3:
            # Check mass center
            com = gmsh.model.occ.getCenterOfMass(dim, tag)
            z = com[2]
            if z < 0:
                heatsink_tags.append(tag)
            else:
                chip_tags.append(tag)
                
    # Physical Groups (Tags 10, 20 to avoid 0/1 confusion)
    gmsh.model.addPhysicalGroup(3, heatsink_tags, tag=10, name="solid_heatsink")
    gmsh.model.addPhysicalGroup(3, chip_tags, tag=20, name="solid_chip")

    # Physical Surfaces
    # Get boundary surfaces
    # We can iterate over surfaces of the volumes
    
    # Heatsink Bottom
    hs_bottom_surfs = gmsh.model.getEntitiesInBoundingBox(-hs_L, -hs_W, -hs_H-1e-6, hs_L, hs_W, -hs_H+1e-6, 2)
    gmsh.model.addPhysicalGroup(2, [s[1] for s in hs_bottom_surfs], name="heatsink_bottom")
    
    # Chip Top
    chip_top_surfs = gmsh.model.getEntitiesInBoundingBox(-chip_L, -chip_W, chip_H-1e-6, chip_L, chip_W, chip_H+1e-6, 2)
    gmsh.model.addPhysicalGroup(2, [s[1] for s in chip_top_surfs], name="chip_top")

    # Mesh Size
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.002)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.01)
    
    # 2D and 3D Mesh
    gmsh.model.mesh.generate(3)
    
    # Write MSH 2.2 (most compatible)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write("test_geom.msh")
    print("Mesh generated: test_geom.msh")
    
    gmsh.finalize()

if __name__ == "__main__":
    main()
