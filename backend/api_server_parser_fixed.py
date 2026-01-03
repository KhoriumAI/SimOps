# This is a temporary file to prepare the replacement content
# It will contain the three functions that need to be inserted before parse_msh_file

def apply_mesh_quality_colors(element_tags, per_element_quality, per_element_gamma, per_element_skewness, per_element_aspect_ratio, per_element_min_angle, fallback_quality=None):
    """Generate color arrays based on element quality metrics"""
    colors_sicn = []
    colors_gamma = []
    colors_skewness = []
    colors_aspect_ratio = []
    colors_min_angle = []
    
    def get_color(q, metric='sicn'):
        if q is None:
            return 0.29, 0.56, 0.89
        val = q
        if metric == 'skewness':
            val = max(0.0, min(1.0, 1.0 - q))
        elif metric == 'aspect_ratio':
            val = max(0.0, min(1.0, 1.0 - (q - 1.0) / 4.0))
        elif metric == 'minAngle':
            val = max(0.0, min(1.0, q / 60.0))
        else:
            val = max(0.0, min(1.0, q))
            
        if val <= 0.1: return 0.8, 0.0, 0.0
        elif val < 0.3: return 1.0, 0.3 * (val - 0.1)/0.2, 0.0
        elif val < 0.5: return 1.0, 0.3 + 0.7 * (val - 0.3)/0.2, 0.0
        elif val < 0.7: return 1.0 - 0.5 * (val - 0.5)/0.2, 1.0, 0.0
        else: return 0.5 - 0.5 * min(1.0, (val - 0.7)/0.3), 0.8 + 0.2 * min(1.0, (val - 0.7)/0.3), 0.2 * min(1.0, (val - 0.7)/0.3)

    if not fallback_quality:
        fallback_quality = {}

    for el_tag in element_tags:
        tag_key = int(el_tag)
        
        q_sicn = per_element_quality.get(tag_key)
        if q_sicn is None: q_sicn = fallback_quality.get(tag_key)
        if q_sicn is None: q_sicn = fallback_quality.get(str(tag_key))
        
        q_gamma = per_element_gamma.get(tag_key)
        q_skew = per_element_skewness.get(tag_key)
        q_ar = per_element_aspect_ratio.get(tag_key)
        q_ang = per_element_min_angle.get(tag_key)
        
        colors_sicn.extend(get_color(q_sicn, 'sicn') * 3)
        colors_gamma.extend(get_color(q_gamma, 'gamma') * 3)
        colors_skewness.extend(get_color(q_skew, 'skewness') * 3)
        colors_aspect_ratio.extend(get_color(q_ar, 'aspect_ratio') * 3)
        colors_min_angle.extend(get_color(q_ang, 'minAngle') * 3)
        
    return {
        "sicn": colors_sicn,
        "gamma": colors_gamma,
        "skewness": colors_skewness,
        "aspect_ratio": colors_aspect_ratio,
        "min_angle": colors_min_angle
    }


def parse_msh_file_via_gmsh(msh_filepath, per_element_quality, per_element_gamma, per_element_skewness, per_element_aspect_ratio, per_element_min_angle):
    """
    Fallback parser using GMSH Python API for binary MSH files.
    Handles assembly node mapping correctly.
    """
    print(f"[MESH PARSE] Using GMSH fallback parser for: {msh_filepath}")
    
    # GMSH script that handles node mapping correctly for assemblies
    gmsh_script = f'''
import gmsh
import json
import sys

try:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.open("{msh_filepath}")
    
    # Get all nodes
    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    
    # Create node tag -> index mapping (CRITICAL for assemblies!)
    node_map = {{int(tag): i for i, tag in enumerate(nodeTags)}}
    
    # Storage for boundary faces
    face_map = {{}}
    
    def add_face(face_nodes, el_tag):
        key = tuple(sorted(face_nodes))
        if key not in face_map:
            face_map[key] = {{"nodes": face_nodes, "count": 0, "tag": el_tag}}
        face_map[key]["count"] += 1
    
    # Process volume elements to extract boundary
    for dim in [3]:  # 3D elements
        elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim)
        
        for i, etype in enumerate(elemTypes):
            tags = elemTags[i]
            nodes_flat = elemNodeTags[i]
            
            if etype == 4:  # Tet4
                nodes_per_el = 4
                for j in range(0, len(nodes_flat), nodes_per_el):
                    el_nodes_tags = [int(nodes_flat[j+k]) for k in range(nodes_per_el)]
                    try:
                        n = [node_map[t] for t in el_nodes_tags]
                        el_tag = int(tags[j // nodes_per_el])
                        # 4 faces of tet
                        add_face((n[0], n[2], n[1]), el_tag)
                        add_face((n[0], n[1], n[3]), el_tag)
                        add_face((n[0], n[3], n[2]), el_tag)
                        add_face((n[1], n[2], n[3]), el_tag)
                    except (KeyError, IndexError):
                        pass
                        
            elif etype == 5:  # Hex8
                nodes_per_el = 8
                for j in range(0, len(nodes_flat), nodes_per_el):
                    el_nodes_tags = [int(nodes_flat[j+k]) for k in range(nodes_per_el)]
                    try:
                        n = [node_map[t] for t in el_nodes_tags]
                        el_tag = int(tags[j // nodes_per_el])
                        # 6 faces, each split into 2 triangles
                        quads = [
                            (n[0], n[3], n[2], n[1]),
                            (n[4], n[5], n[6], n[7]),
                            (n[0], n[1], n[5], n[4]),
                            (n[2], n[3], n[7], n[6]),
                            (n[1], n[2], n[6], n[5]),
                            (n[4], n[7], n[3], n[0])
                        ]
                        for q in quads:
                            add_face((q[0], q[1], q[2]), el_tag)
                            add_face((q[0], q[2], q[3]), el_tag)
                    except (KeyError, IndexError):
                        pass
    
    # Extract boundary faces (count == 1) and surface elements
    vertices = []
    element_tags = []
    entity_tags = []
    
    for key, data in face_map.items():
        if data["count"] == 1:  # Boundary face
            face_nodes = data["nodes"]
            for idx in face_nodes:
                # idx is already mapped index
                vertices.extend([
                    float(nodeCoords[idx*3]),
                    float(nodeCoords[idx*3 + 1]),
                    float(nodeCoords[idx*3 + 2])
                ])
            element_tags.append(int(data["tag"]))
            entity_tags.append(0)
    
    # Output result
    result = {{
        "vertices": vertices,
        "element_tags": element_tags,
        "entity_tags": entity_tags,
        "num_nodes": len(nodeTags)
    }}
    
    print("GMSH_RESULT:" + json.dumps(result))
    
except Exception as e:
    print("GMSH_ERROR:" + str(e), file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    try:
        gmsh.finalize()
    except:
        pass
'''
    
    # Run GMSH script
    import subprocess
    try:
        result = subprocess.run(
            [sys.executable, '-c', gmsh_script],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            print(f"[GMSH FALLBACK ERROR] {result.stderr}")
            return {"error": f"GMSH subprocess failed: {result.stderr[:200]}"}
        
        # Parse output
        for line in result.stdout.split('\n'):
            if line.startswith('GMSH_RESULT:'):
                import json
                data = json.loads(line[12:])
                vertices = data['vertices']
                element_tags = data['element_tags']
                entity_tags = data['entity_tags']
                num_nodes = data['num_nodes']
                
                # Apply quality colors
                color_data = apply_mesh_quality_colors(
                    element_tags, per_element_quality, per_element_gamma,
                    per_element_skewness, per_element_aspect_ratio, per_element_min_angle
                )
                
                # Compute quality summary
                quality_values = [per_element_quality.get(int(t)) for t in element_tags if per_element_quality.get(int(t)) is not None]
                quality_summary = {}
                if quality_values:
                    quality_summary = {
                        "min": min(quality_values),
                        "max": max(quality_values),
                        "avg": sum(quality_values) / len(quality_values)
                    }
                
                return {
                    "vertices": vertices,
                    "element_tags": element_tags,
                    "entity_tags": entity_tags,
                    "num_nodes": num_nodes,
                    "colors": color_data["sicn"],
                    "qualityColors": color_data,
                    "qualityMetrics": quality_summary,
                    "histogramData": [],
                    "hasQualityData": bool(per_element_quality)
                }
        
        return {"error": "No valid output from GMSH subprocess"}
        
    except subprocess.TimeoutExpired:
        return {"error": "GMSH subprocess timed out"}
    except Exception as e:
        print(f"[GMSH FALLBACK ERROR] {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
