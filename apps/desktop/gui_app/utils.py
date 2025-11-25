"""
GUI App Utilities
==================

Logging setup, helper functions, and file parsers.
"""

import logging
import os
import tempfile
from pathlib import Path


def setup_logging():
    """Set up file logging for GUI debugging"""
    log_file = os.path.join(tempfile.gettempdir(), "meshgen_gui_debug.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'  # Overwrite on each launch
    )
    logging.info("=" * 60)
    logging.info("GUI Starting - Log initialized")
    logging.info("=" * 60)
    return log_file


def hsl_to_rgb(h, s, l):
    """
    Convert HSL color to RGB.
    
    Args:
        h: Hue (0-1)
        s: Saturation (0-1)
        l: Lightness (0-1)
        
    Returns:
        tuple: (r, g, b) as integers 0-255
    """
    def hue_to_rgb(p, q, t):
        if t < 0:
            t += 1
        if t > 1:
            t -= 1
        if t < 1/6:
            return p + (q - p) * 6 * t
        if t < 1/2:
            return q
        if t < 2/3:
            return p + (q - p) * (2/3 - t) * 6
        return p
    
    if s == 0:
        r = g = b = l
    else:
        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q
        r = hue_to_rgb(p, q, h + 1/3)
        g = hue_to_rgb(p, q, h)
        b = hue_to_rgb(p, q, h - 1/3)
    
    return int(r * 255), int(g * 255), int(b * 255)


def parse_msh_file(msh_path):
    """
    Parse a Gmsh .msh file and extract nodes and elements.
    
    Args:
        msh_path: Path to .msh file
        
    Returns:
        tuple: (nodes_dict, elements_list)
            nodes_dict: {node_id: (x, y, z)}
            elements_list: [{id, type, nodes, ...}]
    """
    import re
    
    nodes = {}
    elements = []
    
    with open(msh_path, 'r') as f:
        lines = f.readlines()
    
    # Parse nodes
    in_nodes = False
    for i, line in enumerate(lines):
        if line.strip() == '$Nodes':
            in_nodes = True
            # Next line has format info
            info_line = lines[i + 1].strip()
            continue
        elif line.strip() == '$EndNodes':
            in_nodes = False
            continue
        
        if in_nodes and line.strip():
            # Try to parse as node: ID x y z
            parts = line.strip().split()
            if len(parts) >= 4:
                try:
                    node_id = int(parts[0])
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    nodes[node_id] = (x, y, z)
                except ValueError:
                    pass  # Skip malformed lines
    
    # Parse elements
    in_elements = False
    for i, line in enumerate(lines):
        if line.strip() == '$Elements':
            in_elements = True
            continue
        elif line.strip() == '$EndElements':
            in_elements = False
            continue
        
        if in_elements and line.strip():
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    elem_id = int(parts[0])
                    elem_type = int(parts[1])
                    
                    # Type 2 = triangle (3 nodes)
                    # Type 4 = tetrahedron (4 nodes)
                    if elem_type == 2 and len(parts) >= 6:
                        node_ids = [int(parts[j]) for j in range(3, 6)]
                        elements.append({
                            'id': elem_id,
                            'type': 'triangle',
                            'nodes': node_ids
                        })
                    elif elem_type == 4 and len(parts) >= 7:
                        node_ids = [int(parts[j]) for j in range(3, 7)]
                        elements.append({
                            'id': elem_id,
                            'type': 'tetrahedron',
                            'nodes': node_ids
                        })
                except (ValueError, IndexError):
                    pass
    
    return nodes, elements
