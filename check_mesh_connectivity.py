import gmsh
import sys

def check_connectivity(msh_file):
    gmsh.initialize()
    gmsh.open(msh_file)
    
    # Get all nodes
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    print(f"Total nodes: {len(node_tags)}")
    
    # Get nodes used by Volume (Fluid)
    vol_entities = gmsh.model.getEntities(3)
    if not vol_entities:
        print("ERROR: No volume entities found!")
    else:
        vol_node_tags = set()
        for e in vol_entities:
            # getNodes returns (nodeTags, coord, param)
            nt, _, _ = gmsh.model.mesh.getNodes(3, e[1])
            vol_node_tags.update(nt)
        print(f"Nodes used by Volume elements: {len(vol_node_tags)}")

    # Get nodes used by Surface (Wall)
    surf_entities = gmsh.model.getEntities(2)
    if not surf_entities:
        print("ERROR: No surface entities found!")
    else:
        surf_node_tags = set()
        for e in surf_entities:
            nt, _, _ = gmsh.model.mesh.getNodes(2, e[1])
            surf_node_tags.update(nt)
        print(f"Nodes used by Surface elements: {len(surf_node_tags)}")
        
    # Check intersection
    shared = vol_node_tags.intersection(surf_node_tags)
    print(f"Shared nodes (Volume & Surface): {len(shared)}")
    
    if len(shared) == 0:
        print("CRITICAL ERROR: Surface and Volume describe disjoint meshes! They share 0 nodes.")
        print("This explains 'Null Domain Pointer' - the wall is not attached to the fluid.")
    elif len(shared) < len(surf_node_tags):
        print(f"WARNING: Surface has {len(surf_node_tags)} nodes, but only {len(shared)} match volume nodes.")
        print("This might be a topology mismatch or hanging nodes.")
    else:
        print("SUCCESS: All surface nodes are part of the volume mesh.")

    gmsh.finalize()

if __name__ == "__main__":
    check_connectivity("C:/Users/Owner/Downloads/MeshPackageLean/apps/cli/generated_meshes/Cube_clean_v2.msh")
