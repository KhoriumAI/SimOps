import numpy as np

def intersect_edge_with_plane(v1, v2, plane_origin, plane_normal):
    """Calculate intersection of an edge with a plane."""
    v1_arr = np.array(v1)
    v2_arr = np.array(v2)
    origin_arr = np.array(plane_origin)
    normal_arr = np.array(plane_normal)
    
    d1 = np.dot(v1_arr - origin_arr, normal_arr)
    d2 = np.dot(v2_arr - origin_arr, normal_arr)
    
    if d1 * d2 > 0:
        return None  # No intersection
        
    if abs(d1 - d2) < 1e-12:
        return v1_arr.tolist() # Edge lies on plane or points are same
        
    t = d1 / (d1 - d2)
    intersection = v1_arr + t * (v2_arr - v1_arr)
    return intersection.tolist()

def intersect_element_with_plane(element_type, nodes_coords, plane_origin, plane_normal):
    """
    Intersect a single element (Tet4/Tet10/Hex8) with a plane.
    Returns a list of sorted points forming the intersection polygon.
    """
    # Define edges for supported element types
    if element_type in [4, 11]: # Tet4, Tet10
        # For slicing, we only care about linear edges even if quadratic
        edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    elif element_type in [5, 12]: # Hex8, Hex27
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    else:
        return []
        
    intersection_points = []
    for i, j in edges:
        # Check if indices are within bounds
        if i >= len(nodes_coords) or j >= len(nodes_coords):
            continue
            
        point = intersect_edge_with_plane(nodes_coords[i], nodes_coords[j], plane_origin, plane_normal)
        if point is not None:
            intersection_points.append(point)
            
    # Deduplicate points
    unique_points = []
    EPSILON = 1e-8
    for pt in intersection_points:
        is_duplicate = False
        for existing in unique_points:
            if np.linalg.norm(np.array(pt) - np.array(existing)) < EPSILON:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_points.append(pt)
            
    if len(unique_points) < 3:
        return []
        
    # Sort points to form a convex polygon (since elements are convex)
    centroid = np.mean(unique_points, axis=0)
    normal = np.array(plane_normal)
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    
    # Create coordinate system on the plane
    if abs(normal[0]) < 0.9:
        arbitrary = np.array([1, 0, 0])
    else:
        arbitrary = np.array([0, 1, 0])
        
    u = np.cross(normal, arbitrary)
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(normal, u)
    
    def get_angle(pt):
        relative = np.array(pt) - centroid
        x_proj = np.dot(relative, u)
        y_proj = np.dot(relative, v)
        return np.arctan2(y_proj, x_proj)
        
    unique_points.sort(key=get_angle)
    return unique_points

def generate_slice_mesh(mesh_nodes, elements, quality_map, plane_origin, plane_normal):
    """
    Generate a 'crinkle' slice mesh (jaggy look) by identifying all faces
    on the boundary of the volume formed by elements whose centroid is behind the plane.
    """
    import numpy as np
    origin_arr = np.array(plane_origin)
    normal_arr = np.array(plane_normal)
    
    # 1. Determine kept status for all volume elements
    is_kept = []
    element_faces_list = []
    
    for el_type, el_tag, node_ids in elements:
        try:
            coords = np.array([mesh_nodes[nid] for nid in node_ids])
        except KeyError:
            is_kept.append(False)
            element_faces_list.append([])
            continue
            
        centroid = np.mean(coords, axis=0)
        # Keep if centroid is behind the plane (keeping tets on the 'negative' side of normal)
        # to match the clipping direction in the viewer.
        dist = np.dot(centroid - origin_arr, normal_arr)
        # We use a small epsilon to avoid numerical issues
        kept = (dist < 0)
        is_kept.append(kept)
        
        # Define faces (triangles)
        faces = []
        if el_type in [4, 11]: # Tet4, Tet10
            # Order: 0, 1, 2, 3
            # Faces with outward winding: (0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)
            faces = [
                (node_ids[0], node_ids[2], node_ids[1]),
                (node_ids[0], node_ids[1], node_ids[3]),
                (node_ids[0], node_ids[3], node_ids[2]),
                (node_ids[1], node_ids[2], node_ids[3])
            ]
        elif el_type in [5, 12]: # Hex8, Hex27
            f = node_ids
            # Hex faces (6 quads -> 12 triangles)
            faces = [
                (f[0], f[2], f[1]), (f[0], f[3], f[2]), # bottom
                (f[4], f[5], f[6]), (f[4], f[6], f[7]), # top
                (f[0], f[1], f[5]), (f[0], f[5], f[4]), # front
                (f[1], f[2], f[6]), (f[1], f[6], f[5]), # right
                (f[2], f[3], f[7]), (f[2], f[7], f[6]), # back
                (f[3], f[0], f[4]), (f[3], f[4], f[7])  # left
            ]
        element_face_data = faces
        element_faces_list.append(element_face_data)

    # 2. Map faces to their neighbor elements for boundary extraction
    # key is sorted tuple of node IDs to identify shared faces
    face_map = {} 
    
    for i, faces in enumerate(element_faces_list):
        for face in faces:
            key = tuple(sorted(list(face)))
            if key not in face_map:
                face_map[key] = {"nodes": face, "neighbors": [], "tag": elements[i][1]}
            face_map[key]["neighbors"].append(i)
            
    # 3. Identify boundary faces of the KEPT volume
    # A face is on the boundary of the kept volume if exactly one of its neighbors is kept.
    output_vertices = []
    output_colors = []
    output_indices = []
    
    def get_color(q):
        # q is 0 to 1 (0=bad, 1=good)
        r = 1.0 - q
        g = q
        b = 0.2
        return [r, g, b]

    vertex_count = 0
    for key, data in face_map.items():
        neighbors = data["neighbors"]
        kept_neighbors = [n_idx for n_idx in neighbors if is_kept[n_idx]]
        
        # If exactly 1 neighbor is kept, this face is part of the shell of the cut volume
        if len(kept_neighbors) == 1:
            quality = quality_map.get(str(data["tag"]), 1.0)
            color = get_color(quality)
            
            # Add vertices and colors
            for nid in data["nodes"]:
                output_vertices.extend(mesh_nodes[nid])
                output_colors.extend(color)
            
            output_indices.extend([vertex_count, vertex_count + 1, vertex_count + 2])
            vertex_count += 3
            
    return {
        "vertices": output_vertices,
        "colors": output_colors,
        "indices": output_indices
    }


def parse_msh_for_slicing(msh_path):
    """
    Parses a Gmsh 4.1 file specifically for volume elements and nodes.
    Returns: (nodes_dict, vol_elements)
    nodes_dict: {id: [x,y,z]}
    vol_elements: [(type, tag, [node_ids])]
    """
    nodes = {}
    vol_elements = [] # (type, tag, node_ids)
    
    # Volume element types in Gmsh
    VOL_TYPES = {
        4: 4,   # Tet4
        11: 10, # Tet10
        5: 8,   # Hex8
        12: 27, # Hex27
        29: 20, # Prism15
        30: 21, # Prism21
    }
    
    with open(msh_path, 'r') as f:
        content = f.read()
        
    # Parse Nodes
    try:
        nodes_section = content.split('$Nodes')[1].split('$EndNodes')[0].strip().split('\n')
        num_blocks, total_nodes, _, _ = map(int, nodes_section[0].split())
        
        curr_line = 1
        for _ in range(num_blocks):
            dim, tag, parametric, num_nodes_in_block = map(int, nodes_section[curr_line].split())
            curr_line += 1
            node_ids = [int(nodes_section[curr_line + i]) for i in range(num_nodes_in_block)]
            curr_line += num_nodes_in_block
            for i in range(num_nodes_in_block):
                coords = list(map(float, nodes_section[curr_line + i].split()))
                nodes[node_ids[i]] = coords
            curr_line += num_nodes_in_block
    except Exception as e:
        print(f"Error parsing nodes: {e}")

    # Parse Elements
    try:
        elements_section = content.split('$Elements')[1].split('$EndElements')[0].strip().split('\n')
        num_blocks, total_elements, _, _ = map(int, elements_section[0].split())
        
        curr_line = 1
        for _ in range(num_blocks):
            line = elements_section[curr_line].split()
            if not line: break
            entity_dim, entity_tag, el_type, num_els_in_block = map(int, line)
            curr_line += 1
            
            if entity_dim == 3 and el_type in VOL_TYPES:
                nodes_per_el = VOL_TYPES[el_type]
                for i in range(num_els_in_block):
                    el_line = list(map(int, elements_section[curr_line + i].split()))
                    el_tag = el_line[0]
                    node_ids = el_line[1:]
                    vol_elements.append((el_type, el_tag, node_ids))
            curr_line += num_els_in_block
    except Exception as e:
        print(f"Error parsing elements: {e}")
        
    return nodes, vol_elements
