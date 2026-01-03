#!/usr/bin/env python3
"""
Verification script for new native mesh parser.
Tests the parser on core_sample.step and validates quality metrics.
"""
import sys
import json
import subprocess
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def color_text(text, color_code):
    return f"\033[{color_code}m{text}\033[0m"

RED, GREEN, YELLOW, BLUE = "31", "32", "33", "34"

def parse_msh_file_standalone(msh_filepath):
    """Standalone version of the new parser for testing"""
    print(f"[PARSE] Reading: {msh_filepath}")
    
    # Load quality data
    quality_filepath = Path(msh_filepath).with_suffix('.quality.json')
    per_element_quality = {}
    
    if quality_filepath.exists():
        with open(quality_filepath, 'r') as f:
            qdata = json.load(f)
            per_element_quality = {int(k): v for k, v in qdata.get('per_element_quality', {}).items()}
            print(f"[PARSE] Loaded quality for {len(per_element_quality)} elements")
    
    with open(msh_filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Parse Nodes
    if '$Nodes' not in content:
        return {"error": "No $Nodes section"}
    
    nodes_section = content.split('$Nodes')[1].split('$EndNodes')[0].strip().split('\n')
    header = nodes_section[0].split()
    num_blocks = int(header[0])
    
    nodes = {}
    node_id_to_index = {}
    curr_line = 1
    
    for _ in range(num_blocks):
        block_header = nodes_section[curr_line].split()
        curr_line += 1
        num_nodes_in_block = int(block_header[3])
        
        node_tags = [int(nodes_section[curr_line + i]) for i in range(num_nodes_in_block)]
        curr_line += num_nodes_in_block
        
        for i in range(num_nodes_in_block):
            coords = list(map(float, nodes_section[curr_line + i].split()))
            node_tag = node_tags[i]
            nodes[node_tag] = coords[:3]
            node_id_to_index[node_tag] = len(node_id_to_index)
        curr_line += num_nodes_in_block
    
    print(f"[PARSE] Parsed {len(nodes)} nodes")
    
    # Parse Elements
    if '$Elements' not in content:
        return {"error": "No $Elements section"}
    
    elements_section = content.split('$Elements')[1].split('$EndElements')[0].strip().split('\n')
    header = elements_section[0].split()
    num_blocks = int(header[0])
    
    face_map = {}
    curr_line = 1
    
    for _ in range(num_blocks):
        line_parts = elements_section[curr_line].split()
        curr_line += 1
        if not line_parts: continue
        
        entity_tag_block = int(line_parts[1])
        el_type = int(line_parts[2])
        num_els = int(line_parts[3])
        
        for i in range(num_els):
            el_line = list(map(int, elements_section[curr_line + i].split()))
            el_tag = el_line[0]
            node_ids = el_line[1:]
            
            if el_type in [4, 11]:  # Tetrahedra
                try:
                    n = [node_id_to_index[nid] for nid in node_ids[:4]]
                    faces = [(n[0], n[2], n[1]), (n[0], n[1], n[3]), (n[0], n[3], n[2]), (n[1], n[2], n[3])]
                    for face in faces:
                        key = tuple(sorted(face))
                        if key not in face_map:
                            face_map[key] = {'nodes': face, 'count': 0, 'element_tag': el_tag}
                        face_map[key]['count'] += 1
                except KeyError: pass
            
            elif el_type in [5, 12]:  # Hexahedra
                try:
                    n = [node_id_to_index[nid] for nid in node_ids[:8]]
                    qs = [(n[0], n[3], n[2], n[1]), (n[4], n[5], n[6], n[7]), 
                          (n[0], n[1], n[5], n[4]), (n[2], n[3], n[7], n[6]),
                          (n[1], n[2], n[6], n[5]), (n[4], n[7], n[3], n[0])]
                    for q in qs:
                        for tri in [(q[0], q[1], q[2]), (q[0], q[2], q[3])]:
                            key = tuple(sorted(tri))
                            if key not in face_map:
                                face_map[key] = {'nodes': tri, 'count': 0, 'element_tag': el_tag}
                            face_map[key]['count'] += 1
                except KeyError: pass
            
            elif el_type in [2, 9]:  # Triangles
                try:
                    n = [node_id_to_index[nid] for nid in node_ids[:3]]
                    key = tuple(sorted(n))
                    face_map[key] = {'nodes': n, 'count': 1, 'element_tag': el_tag}
                except KeyError: pass
        
        curr_line += num_els
    
    # Extract boundary faces
    indexed_nodes = [None] * len(nodes)
    for nid, idx in node_id_to_index.items():
        indexed_nodes[idx] = nodes[nid]
    
    vertices = []
    element_tags = []
    
    for key, data in face_map.items():
        if data['count'] == 1:
            for idx in data['nodes']:
                vertices.extend(indexed_nodes[idx])
            element_tags.append(data['element_tag'])
    
    print(f"[PARSE] Extracted {len(vertices)//3} vertices ({len(element_tags)} faces)")
    
    # Quality metrics
    quality_values = [per_element_quality.get(tag) for tag in element_tags if tag in per_element_quality]
    quality_values = [q for q in quality_values if q is not None]
    
    if quality_values:
        return {
            "vertices": vertices,
            "qualityMetrics": {
                "total_elements": len(quality_values),
                "sicn_min": min(quality_values),
                "sicn_avg": sum(quality_values) / len(quality_values),
                "sicn_max": max(quality_values),
                "poor_elements": sum(1 for q in quality_values if q < 0.1)
            }
        }
    else:
        return {"vertices": vertices, "qualityMetrics": None}

def run_test():
    print("="*60)
    print("TESTING NEW NATIVE MESH PARSER")
    print("="*60)
    
    cad_file = PROJECT_ROOT / "cad_files" / "core_sample.step"
    if not cad_file.exists():
        print(color_text(f"[ERROR] CAD file not found: {cad_file}", RED))
        return 1
    
    print(f"[1] Found CAD file: {cad_file.name}")
    
    output_mesh = PROJECT_ROOT / "temp_core_sample.msh"
    if output_mesh.exists(): output_mesh.unlink()
    if output_mesh.with_suffix('.quality.json').exists(): 
        output_mesh.with_suffix('.quality.json').unlink()
    
    print("[2] Generating mesh...")
    worker_script = PROJECT_ROOT / "apps" / "cli" / "mesh_worker_subprocess.py"
    config_file = PROJECT_ROOT / "temp_test_config.json"
    
    with open(config_file, 'w') as f:
        json.dump({"element_size": 0.5, "strategy": "hex_dominant", "quality_high": True}, f)
    
    try:
        cmd = [sys.executable, str(worker_script), str(cad_file), str(output_mesh.parent), "--config-file", str(config_file)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(color_text("[ERROR] Mesh generation failed", RED))
            print(result.stderr)
            return 1
        
        print(color_text(f"[2] ✓ Mesh generated", GREEN))
    except subprocess.TimeoutExpired:
        print(color_text("[ERROR] Timeout", RED))
        return 1
    finally:
        if config_file.exists(): config_file.unlink()
    
    
    print("[3] Running parser...")
    
    # The mesh worker saves to apps/cli/generated_meshes/
    actual_mesh = PROJECT_ROOT / "apps" / "cli" / "generated_meshes" / f"{cad_file.stem}_mesh.msh"
    
    if not actual_mesh.exists():
        print(color_text(f"[ERROR] Mesh not found at {actual_mesh}", RED))
        return 1
    
    try:
        mesh_data = parse_msh_file_standalone(str(actual_mesh))
        
        if "error" in mesh_data:
            print(color_text(f"[FAIL] {mesh_data['error']}", RED))
            return 1
        
        print(color_text("[3] ✓ Parser succeeded", GREEN))
        
        metrics = mesh_data.get('qualityMetrics')
        if not metrics:
            print(color_text("[WARN] No quality metrics", YELLOW))
            return 1
        
        print("\n" + "="*60)
        print("QUALITY METRICS")
        print("="*60)
        total = metrics['total_elements']
        poor = metrics['poor_elements']
        percent_poor = (poor / total) * 100
        
        print(f"Total Elements: {total:,}")
        print(f"SICN Min: {metrics['sicn_min']:.4f}")
        print(f"SICN Avg: {metrics['sicn_avg']:.4f}")
        print(f"Poor Elements (< 0.1): {poor}/{total} ({percent_poor:.1f}%)")
        
        print("\n" + "="*60)
        if percent_poor < 10.0:
            print(color_text("✓ PASS: Quality acceptable (< 10% poor)", GREEN))
            return 0
        else:
            print(color_text("✗ FAIL: Too many poor elements", RED))
            return 1
    
    except Exception as e:
        print(color_text(f"[FAIL] Exception: {e}", RED))
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(run_test())
