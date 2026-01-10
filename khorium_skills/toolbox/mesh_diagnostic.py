import os
import json
import sys
from pathlib import Path
import traceback

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import meshio
except ImportError:
    meshio = None

def diagnostic_mesh(msh_path: str, result_json_path: str = None):
    """
    Diagnostic tool to verify mesh tags and quality mapping.
    """
    print(f"\n{'='*70}")
    print(f"MESH DIAGNOSTIC: {os.path.basename(msh_path)}")
    print(f"{'='*70}")
    
    if not os.path.exists(msh_path):
        print(f"[ERROR] MSH file not found: {msh_path}")
        return

    # 1. Check for sidecar JSON
    if result_json_path is None:
        # Try both common extensions
        p1 = Path(msh_path).with_suffix('.json')
        p2 = Path(str(msh_path).replace('.msh', '_result.json'))
        if p1.exists():
            result_json_path = str(p1)
        elif p2.exists():
            result_json_path = str(p2)
            
    qdata = {}
    if result_json_path and os.path.exists(result_json_path):
        print(f"[OK] Found quality sidecar: {os.path.basename(result_json_path)}")
        try:
            with open(result_json_path, 'r') as f:
                qdata = json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load JSON: {e}")
    else:
        print("[WARNING] No sidecar JSON found. Colors will be default (computed on-the-fly).")

    # 2. Inspect MSH with meshio
    print(f"\n--- MSH Structure (meshio) ---")
    if meshio is None:
        print("[SKIP] meshio not installed. Cannot inspect binary tag alignment.")
    else:
        try:
            mesh = meshio.read(msh_path)
            print(f"Nodes: {len(mesh.points)}")
            print(f"Cell Blocks: {len(mesh.cells)}")
            
            # Check for gmsh:element_id
            has_real_ids = False
            if hasattr(mesh, 'cell_data') and 'gmsh:element_id' in mesh.cell_data:
                has_real_ids = True
                print("[OK] gmsh:element_id FOUND in cell_data (Proper tag preservation)")
            else:
                print("[WARNING] gmsh:element_id MISSING - using sequential fallbacks")
            
            for i, cell_block in enumerate(mesh.cells):
                num_cells = len(cell_block.data)
                print(f"  Block {i}: {cell_block.type} ({num_cells} elements)")
                
                if has_real_ids:
                    ids = mesh.cell_data['gmsh:element_id'][i]
                    print(f"    Sample IDs: {ids[:5]}")
        except Exception as e:
            print(f"[ERROR] meshio failed to read mesh: {e}")
            traceback.print_exc()

    # 3. Quality Dictionary Check
    if qdata:
        print(f"\n--- Quality Data Check ---")
        q_dict = qdata.get('per_element_quality', {})
        print(f"Elements in quality dict: {len(q_dict)}")
        if q_dict:
            # Check for 2D vs 3D tags
            sample_keys = [int(k) for k in list(q_dict.keys())[:20]]
            print(f"Sample Keys: {sample_keys}")
            
            # Simple heuristic: often 2D elements are low-index, 3D are higher or disjoint
            # But the real check is if they match the MSH tags
            if meshio and 'mesh' in locals():
                matches = 0
                missing = 0
                for i, cell_block in enumerate(mesh.cells):
                    if cell_block.type in ['triangle', 'quad']:
                        ids = mesh.cell_data['gmsh:element_id'][i] if has_real_ids else range(1, len(cell_block.data)+1)
                        for tag in ids[:100]: # Check first 100
                            if str(tag) in q_dict or int(tag) in q_dict:
                                matches += 1
                            else:
                                missing += 1
                
                print(f"Surface Elements Check (sample 100): {matches} matched, {missing} missing")
                if matches == 0 and missing > 0:
                    print("[FAIL] Mismatch detected. Surface elements have NO quality data.")
                elif matches > 0:
                    print("[PASS] Surface elements have quality data.")

    print(f"{'='*70}")
    print("DIAGNOSTIC COMPLETE")
    print(f"{'='*70}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python mesh_diagnostic.py <path_to_msh>")
    else:
        diagnostic_mesh(sys.argv[1])
