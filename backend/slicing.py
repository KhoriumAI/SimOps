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
    
    # CRITICAL FIX: Create node ID -> sequential index mapping for assemblies
    node_id_to_index = {}
    node_coords_list = []
    # Use sorted IDs to ensure deterministic mapping if needed, 
    # but insertion order (current nodes dict) is generally fine.
    for idx, (node_id, coords) in enumerate(mesh_nodes.items()):
        node_id_to_index[node_id] = idx
        node_coords_list.append(coords)
    
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
        # Keep if centroid is behind the plane
        dist = np.dot(centroid - origin_arr, normal_arr)
        kept = (dist < 0)
        is_kept.append(kept)
        
        # Define faces (triangles) using MAPPED indices instead of raw node IDs
        faces = []
        try:
            m = [node_id_to_index[nid] for nid in node_ids]
            
            if el_type in [4, 11]: # Tet4, Tet10
                # Corner nodes are always 0,1,2,3
                faces = [
                    (m[0], m[2], m[1]), (m[0], m[1], m[3]),
                    (m[0], m[3], m[2]), (m[1], m[2], m[3])
                ]
            elif el_type in [5, 12]: # Hex8, Hex27
                # Hex faces (6 quads -> 12 triangles)
                faces = [
                    (m[0], m[3], m[2]), (m[0], m[2], m[1]), # bottom
                    (m[4], m[5], m[6]), (m[4], m[6], m[7]), # top
                    (m[0], m[1], m[5]), (m[0], m[5], m[4]), # front
                    (m[1], m[2], m[6]), (m[1], m[6], m[5]), # right
                    (m[2], m[3], m[7]), (m[2], m[7], m[6]), # back
                    (m[3], m[0], m[4]), (m[3], m[4], m[7])  # left
                ]
            elif el_type == 6: # Prism6
                # 2 triangles, 3 quads (6 triangles)
                faces = [
                    (m[0], m[1], m[2]), (m[3], m[5], m[4]), # bottom, top
                    (m[0], m[3], m[4]), (m[0], m[4], m[1]), # side 1
                    (m[1], m[4], m[5]), (m[1], m[5], m[2]), # side 2
                    (m[2], m[5], m[3]), (m[2], m[3], m[0])  # side 3
                ]
            elif el_type == 7: # Pyramid5
                # 1 quad (2 tris), 4 triangles
                faces = [
                    (m[0], m[3], m[2]), (m[0], m[2], m[1]), # base
                    (m[0], m[1], m[4]), (m[1], m[2], m[4]),
                    (m[2], m[3], m[4]), (m[3], m[0], m[4])
                ]
        except (KeyError, IndexError):
            pass
            
        element_faces_list.append(faces)

    # 2. Map faces to their neighbor elements for boundary extraction
    face_map = {} 
    
    for i, faces in enumerate(element_faces_list):
        for face in faces:
            key = tuple(sorted(list(face)))
            if key not in face_map:
                face_map[key] = {"nodes": face, "neighbors": [], "tag": elements[i][1]}
            face_map[key]["neighbors"].append(i)
            
    # 3. Identify Interface Elements (Kept elements touching Discarded elements)
    interface_element_indices = set()
    
    for key, data in face_map.items():
        neighbors = data["neighbors"]
        kept_neighbors = [n for n in neighbors if is_kept[n]]
        discarded_neighbors = [n for n in neighbors if not is_kept[n]]
        
        # If a face is shared by Kept and Discarded, the Kept neighbor is an Interface Element
        if len(kept_neighbors) >= 1 and len(discarded_neighbors) >= 1:
            for k in kept_neighbors:
                interface_element_indices.add(k)
                
    # 4. Generate Output Mesh
    # - For Interface Elements: Output ALL faces (Thick/Volumetric look)
    # - For Deep Kept Elements: Output Only Boundary faces (Shell look)
    
    output_vertices = []
    output_colors = []
    output_indices = []
    
    def get_color(q):
        # Vivid quality scale: Red (bad) -> Yellow -> Green (good)
        if q is None: return [0.3, 0.6, 0.9]
        val = max(0.0, min(1.0, q))
        if val <= 0.1: return [0.8, 0.0, 0.0]
        elif val < 0.5:
            t = (val - 0.1) / 0.4
            return [1.0, t, 0.0]
        else:
            t = min(1.0, (val - 0.5) / 0.5)
            return [1.0 - t, 1.0, 0.0]

    vertex_count = 0
    
    # Iterate ALL elements to generate geometry
    for i, faces in enumerate(element_faces_list):
        if not is_kept[i]:
            continue
            
        el_tag = elements[i][1]
        quality = quality_map.get(str(el_tag), quality_map.get(int(el_tag), 1.0))
        color = get_color(quality)
        
        is_interface = (i in interface_element_indices)
        
        for face in faces:
            # Decide whether to show this face
            show_face = False
            
            if is_interface:
                # Interface elements show ALL faces (volumetric plug)
                show_face = True
            else:
                # Deep elements show only Boundary faces
                key = tuple(sorted(list(face)))
                neighbor_indices = face_map[key]["neighbors"]
                # Count kept neighbors for this face
                k_neighbors = sum(1 for n in neighbor_indices if is_kept[n])
                if k_neighbors == 1:
                    show_face = True
            
            if show_face:
                # Add face vertices
                for mapped_idx in face:
                    output_vertices.extend(node_coords_list[mapped_idx])
                    output_colors.extend(color)
                
                output_indices.extend([vertex_count, vertex_count + 1, vertex_count + 2])
                vertex_count += 3
            
    print(f"[SLICE_LOGIC] Interface elements: {len(interface_element_indices)}")
    print(f"[SLICE_LOGIC] Generated {vertex_count} vertices")
            
    return {
        "vertices": output_vertices,
        "colors": output_colors,
        "indices": output_indices
    }


def parse_msh_for_slicing(msh_path):
    """
    Robustly parses Gmsh (2.2 or 4.1) files for volume elements and nodes.
    Supports ASCII files.
    """
    nodes = {}
    vol_elements = [] # (type, tag, node_ids)
    
    # Gmsh volume element types and their node counts
    VOL_TYPES = {
        4: 4,   # Tet4
        11: 10, # Tet10
        29: 20, # Tet20
        5: 8,   # Hex8
        12: 27, # Hex27
        6: 6,   # Prism6
        13: 18, # Prism18
        7: 5,   # Pyramid5
        14: 14, # Pyramid14
    }
    
    try:
        with open(msh_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading Msh: {e}")
        return {}, []

    # Detect Version
    msh_version = "4.1"
    if '$MeshFormat' in content:
        try:
            fmt = content.split('$MeshFormat')[1].split('$EndMeshFormat')[0].strip().split('\n')[0].split()
            msh_version = fmt[0]
        except: pass

    # Parse Nodes
    if '$Nodes' in content:
        try:
            nodes_section = content.split('$Nodes')[1].split('$EndNodes')[0].strip().split('\n')
            if msh_version.startswith('4'):
                # MSH 4.x
                header = nodes_section[0].split()
                num_blocks = int(header[0])
                curr_line = 1
                for _ in range(num_blocks):
                    block_info = nodes_section[curr_line].split()
                    curr_line += 1
                    num_nodes_in_block = int(block_info[3])
                    
                    node_tags = [int(nodes_section[curr_line + i]) for i in range(num_nodes_in_block)]
                    curr_line += num_nodes_in_block
                    
                    for i in range(num_nodes_in_block):
                        coords = list(map(float, nodes_section[curr_line + i].split()))
                        nodes[node_tags[i]] = coords[:3]
                    curr_line += num_nodes_in_block
            else:
                # MSH 2.x
                total_nodes = int(nodes_section[0])
                for i in range(1, total_nodes + 1):
                    parts = nodes_section[i].split()
                    nodes[int(parts[0])] = [float(parts[1]), float(parts[2]), float(parts[3])]
        except Exception as e:
            print(f"Error parsing nodes: {e}")

    # Parse Elements
    if '$Elements' in content:
        try:
            elements_section = content.split('$Elements')[1].split('$EndElements')[0].strip().split('\n')
            if msh_version.startswith('4'):
                # MSH 4.x
                header = elements_section[0].split()
                num_blocks = int(header[0])
                curr_line = 1
                for _ in range(num_blocks):
                    block_info = elements_section[curr_line].split()
                    if not block_info: break
                    entity_dim = int(block_info[0])
                    el_type = int(block_info[2])
                    num_els = int(block_info[3])
                    curr_line += 1
                    
                    if entity_dim == 3 and el_type in VOL_TYPES:
                        for i in range(num_els):
                            parts = list(map(int, elements_section[curr_line + i].split()))
                            vol_elements.append((el_type, parts[0], parts[1:]))
                    curr_line += num_els
            else:
                # MSH 2.x
                total_els = int(elements_section[0])
                for i in range(1, total_els + 1):
                    parts = list(map(int, elements_section[i].split()))
                    el_type = parts[1]
                    num_tags = parts[2]
                    if el_type in VOL_TYPES:
                        el_tag = parts[0]
                        node_ids = parts[3 + num_tags:]
                        vol_elements.append((el_type, el_tag, node_ids))
        except Exception as e:
            print(f"Error parsing elements: {e}")
            
    return nodes, vol_elements

