import gmsh
import numpy as np
import sys

def generate_cube_mesh(filename="cube.msh"):
    """Generates a simple tetrahedral mesh of a cube."""
    gmsh.initialize()
    gmsh.model.add("cube")

    # Create a cube
    lc = 0.2  # Characteristic length (finer mesh for more internal nodes)
    gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()

    # Mesh settings
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
    
    # Generate 3D mesh
    gmsh.model.mesh.generate(3)
    
    # Save mesh
    gmsh.write(filename)
    
    # Extract data before finalizing
    # Get all nodes
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    nodes = np.array(node_coords).reshape(-1, 3)
    node_map = {tag: i for i, tag in enumerate(node_tags)} # tag -> index in nodes array

    # Get all tetrahedra (element type 4)
    tet_type = 4
    tet_tags, tet_node_tags = gmsh.model.mesh.getElementsByType(tet_type)
    tets = np.array(tet_node_tags).reshape(-1, 4)
    # Map tags to indices
    tets_idx = np.vectorize(node_map.get)(tets)

    gmsh.finalize()
    
    return nodes, tets_idx

def calculate_centroids(nodes, tets):
    """Calculates centroids of all tetrahedra."""
    # nodes: (N, 3) array of coordinates
    # tets: (M, 4) array of node indices
    
    # Get coordinates of all 4 nodes for each tet
    # shape: (M, 4, 3)
    tet_coords = nodes[tets]
    
    # Average over the 4 nodes (axis 1)
    centroids = np.mean(tet_coords, axis=1)
    return centroids

def build_dual_mesh(nodes, tets, centroids):
    """
    Constructs the dual polyhedral mesh.
    
    For a standard dual mesh:
    - Primal Node -> Dual Cell (Polyhedron)
    - Primal Tet -> Dual Node (Vertex of Polyhedron)
    - Primal Edge -> Dual Face
    - Primal Face -> Dual Edge
    """
    print(f"Building dual mesh from {len(nodes)} nodes and {len(tets)} tets...")
    
    # 1. Build Node -> Connected Tets map
    # We need to know which tets share a node
    node_to_tets = {i: [] for i in range(len(nodes))}
    
    for tet_idx, tet_nodes in enumerate(tets):
        for node_idx in tet_nodes:
            node_to_tets[node_idx].append(tet_idx)
            
    # 2. Identify Internal Nodes vs Boundary Nodes
    # A simple heuristic: if a node is on the boundary of the cube [0,1]^3, it's a boundary node.
    # In a general mesh, we'd check if it belongs to a boundary face.
    # For this prototype, let's just check coordinates with a tolerance.
    eps = 1e-5
    is_boundary = np.any((nodes < eps) | (nodes > 1.0 - eps), axis=1)
    
    internal_nodes = np.where(~is_boundary)[0]
    print(f"Found {len(internal_nodes)} internal nodes (will become {len(internal_nodes)} polyhedra)")
    
    polyhedra = []
    
    for node_idx in internal_nodes:
        # The dual cell corresponding to this node is formed by the centroids 
        # of all tets connected to this node.
        connected_tet_indices = node_to_tets[node_idx]
        
        # These centroids form the vertices of the polyhedron
        cell_vertices = centroids[connected_tet_indices]
        
        # To define the faces, we need the connectivity of these tets.
        # Two vertices in the dual cell (centroids of Tet A and Tet B) are connected 
        # if Tet A and Tet B share a face *and* that face contains the central Node.
        # Actually, if they share a face, they definitely share the node if both are in this list.
        
        # Let's build the faces of the polyhedron.
        # The faces of the dual cell correspond to the edges of the primal mesh connected to the central node.
        # For each edge connected to node_idx, we form a face in the dual.
        
        # Simplified output for now: just store the list of dual vertices (centroids)
        # A real polyhedral format requires face definitions.
        polyhedra.append({
            "center_node": node_idx,
            "dual_vertices_indices": connected_tet_indices, # Indices into the 'centroids' array
            "dual_vertices_coords": cell_vertices
        })
        
    return polyhedra

def main():
    print("=== Polyhedral Meshing Prototype ===")
    
    # 1. Generate Input
    try:
        nodes, tets = generate_cube_mesh()
        print(f"Generated mesh: {len(nodes)} nodes, {len(tets)} tets")
    except Exception as e:
        print(f"Error generating mesh: {e}")
        return

    # 2. Calculate Centroids (Dual Nodes)
    centroids = calculate_centroids(nodes, tets)
    print(f"Calculated {len(centroids)} centroids")
    
    # 3. Build Dual Mesh
    polyhedra = build_dual_mesh(nodes, tets, centroids)
    
    print(f"\nGenerated {len(polyhedra)} polyhedral cells.")
    
    # 4. Basic Validation
    if len(polyhedra) > 0:
        avg_verts = np.mean([len(p['dual_vertices_indices']) for p in polyhedra])
        print(f"Average vertices per polyhedron: {avg_verts:.1f}")
        print("Prototype successful!")
    else:
        print("No polyhedra generated (mesh might be too coarse to have internal nodes).")

if __name__ == "__main__":
    main()
