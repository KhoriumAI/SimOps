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
    cell_dict = cells if isinstance(cells, dict) else {c.type: c.data for c in cells}

    if 'tetra' in cell_dict:
        tets = cell_dict['tetra']
    elif 'tetra10' in cell_dict:
        print("[FluentWriter] Found Quadratic (tetra10). Converting to Linear (tetra4) for CFD...")
        tets = cell_dict['tetra10'][:, :4]
    else:
        raise ValueError("[FluentWriter] No tetrahedra found!")

    # --- 2. REMAP NODE INDICES TO BE CONTIGUOUS ---
    print("[FluentWriter] Building node index mapping...")
    unique_nodes = np.unique(tets.flatten())
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_nodes)}
    used_points = points[unique_nodes]
    tets_remapped = np.array([[old_to_new[n] for n in tet] for tet in tets])
    
    n_points = len(used_points)
    n_tets = len(tets_remapped)
    print(f"[FluentWriter] Remapped: {len(points)} -> {n_points} nodes, {n_tets} tetrahedra.")

    # --- 3. CALCULATE TOPOLOGY & WINDING ---
    print("[FluentWriter] Calculating connectivity and enforcing winding...")
    face_map = {}
    tet_faces_local = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
    
    for i, tet in enumerate(tets_remapped):
        cell_id = i + 1
        cell_pts = used_points[tet]
        cell_centroid = np.mean(cell_pts, axis=0)
        
        for f_indices in tet_faces_local:
            face_nodes = tet[f_indices]
            key = tuple(sorted(face_nodes))
            
            if key not in face_map:
                p0, p1, p2 = used_points[face_nodes]
                normal = calculate_face_normal(p0, p1, p2)
                face_center = np.mean([p0, p1, p2], axis=0)
                vec_to_cell = cell_centroid - face_center
                
                if np.dot(normal, vec_to_cell) < 0:
                    final_nodes = [face_nodes[1], face_nodes[0], face_nodes[2]]
                else:
                    final_nodes = list(face_nodes)
                
                face_map[key] = [final_nodes, cell_id, 0]
            else:
                face_map[key][2] = cell_id

    # --- 4. SEPARATE ZONES ---
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

    # --- 5. WRITE TGRID FILE (HEXADECIMAL FORMAT) ---
    # TGrid format uses hexadecimal for indices in section headers
    with open(filename, 'w') as f:
        # Header
        f.write("(0 \"Fluent mesh from MeshPackageLean\")\n")
        f.write("(2 3)\n")  # Dimension = 3D
        
        # Section 10: Nodes
        # Format: (10 (zone-id first-index last-index type ND)(x y z))
        # Use hex for indices
        f.write(f"(10 (0 1 {n_points:x} 0 3))\n")  # Declaration (hex)
        f.write(f"(10 (1 1 {n_points:x} 1 3)(\n")  # Zone 1, nodes 1-n (hex)
        for p in used_points:
            f.write(f" {p[0]:.15e} {p[1]:.15e} {p[2]:.15e}\n")
        f.write("))\n")
        
        # Section 12: Cells (Zone 2 = Fluid)
        # Type 2 = tetrahedral in Section 12
        f.write(f"(12 (0 1 {n_tets:x} 0))\n")  # Declaration (hex)
        f.write(f"(12 (2 1 {n_tets:x} 1 2))\n")  # Zone 2, cells 1-n, active, tet

        # Section 13: Faces
        f.write(f"(13 (0 1 {n_total_faces:x} 0))\n")  # Declaration (hex)

        # Zone 3: Interior Faces (BC Type 2 = Interior)
        if n_interior > 0:
            f.write(f"(13 (3 1 {n_interior:x} 2 3)(\n")  # hex indices
            for nodes, c0, c1 in interior_faces:
                n0, n1, n2 = [n+1 for n in nodes]
                f.write(f" {n0:x} {n1:x} {n2:x} {c0:x} {c1:x}\n")  # hex values
            f.write("))\n")
        
        # Zone 4: Boundary Faces (BC Type 3 = Wall)
        if n_boundary > 0:
            start = n_interior + 1
            end = n_total_faces
            f.write(f"(13 (4 {start:x} {end:x} 3 3)(\n")  # hex indices
            for nodes, c0, c1 in boundary_faces:
                n0, n1, n2 = [n+1 for n in nodes]
                c0_hex = f"{c0:x}" if c0 > 0 else "0"
                c1_hex = f"{c1:x}" if c1 > 0 else "0"
                f.write(f" {n0:x} {n1:x} {n2:x} {c0_hex} {c1_hex}\n")  # hex values
            f.write("))\n")
        
        # Section 45: Zone Types (Required for Fluent to understand zones)
        f.write("(45 (2 fluid fluid)())\n")  # Zone 2 is fluid
        f.write("(45 (3 interior interior-3)())\n")  # Zone 3 is interior
        f.write("(45 (4 wall wall-4)())\n")  # Zone 4 is wall

    print(f"[FluentWriter] Success! Saved {filename}")
    return True

# Wrapper for integration
def convert_gmsh_to_fluent(input_msh, output_fluent):
    import meshio
    import os
    
    if not os.path.exists(input_msh):
        print(f"Error: Input file {input_msh} not found.")
        return False
        
    try:
        print(f"[FluentConverter] Reading {input_msh}...")
        mesh = meshio.read(input_msh)
        
        # --- Extract Boundary Surface Mappings ---
        # We need to map extracted tet faces to the original surface names
        # Key: tuple(sorted(nodes)), Value: zone_name
        boundary_face_map = {}
        
        # 1. Build tag-to-name lookup
        # field_data maps Name -> [tag, dimension]
        tag_to_name = {}
        if mesh.field_data:
            for name, data in mesh.field_data.items():
                # data is usually [tag, dim]
                tag_to_name[data[0]] = name
                
        # 2. Iterate over triangle cells (surfaces)
        if 'triangle' in mesh.cells_dict:
            tris = mesh.cells_dict['triangle']
            
            # Get physical tags for triangles
            # meshio stores generic data in cell_data_dict['gmsh:physical'] 
            # or sometimes just 'physical' depending on version/format
            phys_tags = None
            if 'gmsh:physical' in mesh.cell_data_dict:
                 # Check if we have data for 'triangle' block
                 if 'triangle' in mesh.cell_data_dict['gmsh:physical']:
                     phys_tags = mesh.cell_data_dict['gmsh:physical']['triangle']
            
            if phys_tags is not None:
                for tri, tag in zip(tris, phys_tags):
                    key = tuple(sorted(tri))
                    # Map tag to name, or use "zone_TAG" if name missing
                    zone_name = tag_to_name.get(tag, f"zone_{tag}")
                    boundary_face_map[key] = zone_name
            else:
                print("[FluentConverter] No physical tags found for triangles. Using default 'wall'.")

        print(f"[FluentConverter] Mapped {len(boundary_face_map)} boundary faces from input.")

        # --- Call Exporter ---
        # We need to pass the boundary_face_map to the exporter
        # The exporter will use this to name the zones correctly
        

            
        # Prepare tets (handle quadratic -> linear if needed)
        tets = None
        if 'tetra' in mesh.cells_dict:
            tets = mesh.cells_dict['tetra']
        elif 'tetra10' in mesh.cells_dict:
            print("[FluentConverter] Found Quadratic (tetra10). Converting to Linear (tetra4)...")
            tets = mesh.cells_dict['tetra10'][:, :4]
            
        if tets is None or len(tets) == 0:
            print("[FluentConverter] No tetrahedra found to export!")
            return False

        # Import the advanced exporter
        from core.export_fluent_msh import export_fluent_msh

        return export_fluent_msh(
            output_fluent, 
            mesh.points, 
            tets,
            boundary_classifier=None, 
            boundary_lookup=boundary_face_map,
            verbose=True
        )
        
    except Exception as e:
        print(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False