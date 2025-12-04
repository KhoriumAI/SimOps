import numpy as np

def calculate_face_normal(p0, p1, p2):
    """Calculates face normal vector using cross product."""
    v1 = p1 - p0
    v2 = p2 - p0
    return np.cross(v1, v2)

def write_fluent_msh(filename, points, cells):
    print(f"[FluentWriter] Exporting to {filename}...")
    
    # --- 1. EXTRACT & CONVERT TETRAHEDRA ---
    tets = None
    
    # Handle both list and dict formats from meshio
    cell_dict = cells if isinstance(cells, dict) else {c.type: c.data for c in cells}

    if 'tetra' in cell_dict:
        tets = cell_dict['tetra']
    elif 'tetra10' in cell_dict:
        print("[FluentWriter] Found Quadratic (tetra10). Converting to Linear (tetra4) for CFD...")
        # Take only the first 4 columns (corners), ignore mid-side nodes
        tets = cell_dict['tetra10'][:, :4]
    else:
        raise ValueError("[FluentWriter] No tetrahedra found! (Checked 'tetra' and 'tetra10')")

    n_points = len(points)
    n_tets = len(tets)
    print(f"[FluentWriter] Input: {n_points} nodes, {n_tets} tetrahedra (Linearized).")

    # --- 2. CALCULATE TOPOLOGY & WINDING ---
    print("[FluentWriter] Calculating connectivity and enforcing winding...")
    
    # Map: sorted_node_indices -> [nodes_ordered, c0, c1]
    face_map = {}
    
    # Local face definitions for a linear tet [0,1,2,3]
    tet_faces_local = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
    
    for i, tet in enumerate(tets):
        cell_id = i + 1 # Fluent uses 1-based indexing
        
        # Calculate Cell Centroid (for winding check)
        cell_pts = points[tet]
        cell_centroid = np.mean(cell_pts, axis=0)
        
        for f_indices in tet_faces_local:
            face_nodes = tet[f_indices]
            # Sort key to identify unique faces shared between cells
            key = tuple(sorted(face_nodes))
            
            if key not in face_map:
                # First time seeing this face. 
                # This cell is 'c0' (Right Cell).
                # CHECK WINDING: Normal must point INTO c0 (Fluent TGrid convention).
                p0, p1, p2 = points[face_nodes]
                normal = calculate_face_normal(p0, p1, p2)
                face_center = np.mean([p0, p1, p2], axis=0)
                
                # Vector from face to cell (points INWARD)
                vec_to_cell = cell_centroid - face_center
                
                # Dot Product:
                # > 0: Normal points SAME direction as vec (INWARD) -> Correct.
                # < 0: Normal points OPPOSITE (OUTWARD) -> Incorrect, Flip.
                if np.dot(normal, vec_to_cell) < 0:
                    # Flip winding (swap first two nodes)
                    final_nodes = [face_nodes[1], face_nodes[0], face_nodes[2]]
                else:
                    # Keep winding
                    final_nodes = list(face_nodes)
                
                # Store: [ordered_nodes, c0, c1]
                face_map[key] = [final_nodes, cell_id, 0]
            else:
                # Second time seeing this face.
                # This cell is 'c1' (Left Cell).
                face_map[key][2] = cell_id

    # --- 3. SEPARATE ZONES ---
    interior_faces = []
    boundary_faces = []
    
    for nodes, c0, c1 in face_map.values():
        if c0 != 0 and c1 != 0:
            interior_faces.append((nodes, c0, c1))
        else:
            boundary_faces.append((nodes, c0, c1))
            
    n_interior = len(interior_faces)
    n_boundary = len(boundary_faces)
    n_total_faces = n_interior + n_boundary
    
    print(f"[FluentWriter] Classified: {n_interior} Interior Faces, {n_boundary} Boundary Faces.")
    
    if n_interior == 0:
        print("[FluentWriter] WARNING: No interior faces found. Is the mesh exploded?")

    # --- 4. WRITE TGRID FILE ---
    with open(filename, 'w') as f:
        # Header
        f.write("(0 \"Grid written by MeshPackageLean V6\")\n")
        f.write("(2 3)\n")
        
        # Section 10: Nodes
        f.write(f"(10 (0 1 {n_points:x} 0 3))\n") # Declaration
        f.write(f"(10 (1 1 {n_points:x} 1 3)(\n") # Data
        for p in points:
            f.write(f"{p[0]:.10f} {p[1]:.10f} {p[2]:.10f}\n")
        f.write("))\n")
        
        # Section 12: Cells (Zone 2 = Fluid)
        # Element Type 2 = TETRAHEDRON (Fixed from 4)
        f.write(f"(12 (0 1 {n_tets:x} 0 0))\n") # Declaration
        f.write(f"(12 (2 1 {n_tets:x} 1 2))\n") # Data (Mixed/Tet=2)

        # Section 13: Faces
        f.write(f"(13 (0 1 {n_total_faces:x} 0 0))\n") # Declaration

        # Zone 3: Interior Faces (Type 2 = Linear Triangle)
        if n_interior > 0:
            start = 1
            end = n_interior
            f.write(f"(13 (3 {start:x} {end:x} 2 2)(\n")
            for nodes, c0, c1 in interior_faces:
                n0, n1, n2 = [n+1 for n in nodes]
                f.write(f"{n0:x} {n1:x} {n2:x} {c0:x} {c1:x}\n")
            f.write("))\n")
        
        # Zone 4: Boundary Faces (Type 3 = Wall)
        if n_boundary > 0:
            start = n_interior + 1
            end = n_total_faces
            f.write(f"(13 (4 {start:x} {end:x} 3 2)(\n")
            for nodes, c0, c1 in boundary_faces:
                n0, n1, n2 = [n+1 for n in nodes]
                f.write(f"{n0:x} {n1:x} {n2:x} {c0:x} {c1:x}\n")
            f.write("))\n")

    print(f"[FluentWriter] Success! Saved {filename}")

# Wrapper for integration
def convert_gmsh_to_fluent(input_msh, output_fluent):
    import meshio
    import os
    if not os.path.exists(input_msh):
        print(f"Error: Input file {input_msh} not found.")
        return
    try:
        mesh = meshio.read(input_msh)
        write_fluent_msh(output_fluent, mesh.points, mesh.cells)
    except Exception as e:
        print(f"Conversion failed: {e}")