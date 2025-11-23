"""
Correct MSH 4.1 parser - replaces the _parse_msh_file method in gui_final.py
"""

def _parse_msh_file(filepath: str):
    """
    Parse MSH 4.1 format file

    MSH 4.1 format structure:
    $Nodes
    numEntityBlocks numNodes minNodeTag maxNodeTag
    entityDim entityTag parametric numNodesInBlock
    nodeTag(s)
    x y z
    ...
    $EndNodes

    $Elements
    numEntityBlocks numElements minElementTag maxElementTag
    entityDim entityTag elementType numElementsInBlock
    elementTag node1 node2 ... nodeN
    ...
    $EndElements
    """
    nodes = {}
    elements = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line == "$Nodes":
            i += 1
            # Read header: numEntityBlocks numNodes minNodeTag maxNodeTag
            header = lines[i].strip().split()
            num_blocks = int(header[0])
            i += 1

            # Read each node block
            for _ in range(num_blocks):
                # entityDim entityTag parametric numNodesInBlock
                block_header = lines[i].strip().split()
                num_nodes_in_block = int(block_header[3])
                i += 1

                # Read node tags
                node_tags = []
                for _ in range(num_nodes_in_block):
                    node_tags.append(int(lines[i].strip()))
                    i += 1

                # Read coordinates
                for tag in node_tags:
                    coords = lines[i].strip().split()
                    nodes[tag] = [float(coords[0]), float(coords[1]), float(coords[2])]
                    i += 1

            i += 1  # Skip $EndNodes

        elif line == "$Elements":
            i += 1
            # Read header: numEntityBlocks numElements minElementTag maxElementTag
            header = lines[i].strip().split()
            num_blocks = int(header[0])
            i += 1

            # Read each element block
            for _ in range(num_blocks):
                # entityDim entityTag elementType numElementsInBlock
                block_header = lines[i].strip().split()
                element_type = int(block_header[2])
                num_elements_in_block = int(block_header[3])
                i += 1

                # Parse elements based on type
                for _ in range(num_elements_in_block):
                    data = lines[i].strip().split()

                    # Element format: elementTag node1 node2 ... nodeN
                    # Skip elementTag (data[0]), nodes start at data[1]

                    # Linear tetrahedra (4-node)
                    if element_type == 4 and len(data) >= 5:
                        elements.append({
                            "type": "tetrahedron",
                            "nodes": [int(data[1]), int(data[2]), int(data[3]), int(data[4])]
                        })
                    # Quadratic tetrahedra (10-node) - use first 4 corner nodes
                    elif element_type == 11 and len(data) >= 11:
                        elements.append({
                            "type": "tetrahedron",
                            "nodes": [int(data[1]), int(data[2]), int(data[3]), int(data[4])]
                        })
                    # Linear triangles (3-node)
                    elif element_type == 2 and len(data) >= 4:
                        elements.append({
                            "type": "triangle",
                            "nodes": [int(data[1]), int(data[2]), int(data[3])]
                        })
                    # Quadratic triangles (6-node) - use first 3 corner nodes
                    elif element_type == 9 and len(data) >= 7:
                        elements.append({
                            "type": "triangle",
                            "nodes": [int(data[1]), int(data[2]), int(data[3])]
                        })
                    # Skip other element types (lines, points, etc.)

                    i += 1

            i += 1  # Skip $EndElements
        else:
            i += 1

    print(f"[DEBUG] MSH 4.1 parser: {len(nodes)} nodes, {len(elements)} elements")
    return nodes, elements


# Test it
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = "generated_meshes/Cylinder_mesh.msh"

    nodes, elements = _parse_msh_file(filepath)
    print(f"Parsed {len(nodes)} nodes, {len(elements)} elements")

    tet_count = sum(1 for e in elements if e['type'] == 'tetrahedron')
    tri_count = sum(1 for e in elements if e['type'] == 'triangle')
    print(f"Tets: {tet_count}, Triangles: {tri_count}")
