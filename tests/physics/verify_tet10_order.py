import numpy as np
import subprocess
import os
import sys

# STANDARD GMSH TET10 COORDINATES (Standard Unit Tetrahedron)
nodes = {
    1: [0.0, 0.0, 0.0],  # V1
    2: [1.0, 0.0, 0.0],  # V2
    3: [0.0, 1.0, 0.0],  # V3
    4: [0.0, 0.0, 1.0],  # V4
    5: [0.5, 0.0, 0.0],  # E 1-2
    6: [0.5, 0.5, 0.0],  # E 2-3
    7: [0.0, 0.5, 0.0],  # E 3-1
    8: [0.0, 0.0, 0.5],  # E 1-4
    9: [0.5, 0.0, 0.5],  # E 2-4
    10: [0.0, 0.5, 0.5]  # E 3-4
}

# Gmsh native mapping of nodes 0..9 based on documentation:
gmsh_coords = [
    [0.0, 0.0, 0.0], # 0
    [1.0, 0.0, 0.0], # 1
    [0.0, 1.0, 0.0], # 2
    [0.0, 0.0, 1.0], # 3
    [0.5, 0.0, 0.0], # 4 (0-1)
    [0.5, 0.5, 0.0], # 5 (1-2)
    [0.0, 0.5, 0.0], # 6 (2-0)
    [0.0, 0.0, 0.5], # 7 (0-3)
    [0.5, 0.0, 0.5], # 8 (1-3)
    [0.0, 0.5, 0.5]  # 9 (2-3)
]

# Edge lookup table: (NodeA, NodeB) -> Gmsh Edge Index
# Gmsh Nodes: 0,1,2,3
# 0-1 (4), 1-2 (5), 2-0 (6), 0-3 (7), 1-3 (8), 2-3 (9)
gmsh_edges = {
    (0,1): 4, (1,0): 4,
    (1,2): 5, (2,1): 5,
    (2,0): 6, (0,2): 6,
    (0,3): 7, (3,0): 7,
    (1,3): 8, (3,1): 8,
    (2,3): 9, (3,2): 9
}

# CCX Edges (1-based index in C3D10 definition relative to its vertices 1,2,3,4)
# But here we work with 0-based local indices 0,1,2,3 corresponding to CCX nodes 1,2,3,4.
# Edge 1: Nodes 0-1
# Edge 2: Nodes 1-2
# Edge 3: Nodes 2-0
# Edge 4: Nodes 0-3
# Edge 5: Nodes 1-3
# Edge 6: Nodes 2-3
# Wait, CCX C3D10 Definition:
# 5: 1-2
# 6: 2-3
# 7: 3-1
# 8: 1-4
# 9: 2-4
# 10: 3-4
# In 0-based local: (0,1), (1,2), (2,0), (0,3), (1,3), (2,3)

ccx_edge_defs = [
    (0,1), # 5
    (1,2), # 6
    (2,0), # 7 (3-1)
    (0,3), # 8 (1-4)
    (1,3), # 9 (2-4)
    (2,3)  # 10 (3-4)
]

def generate_consistent_perm(vertex_p):
    # vertex_p: list of 4 indices [a,b,c,d] mapping CCX 0,1,2,3 to Gmsh a,b,c,d
    # Start with vertices
    perm = list(vertex_p)
    
    # Calculate Edges
    for (u_loc, v_loc) in ccx_edge_defs:
        # Get Gmsh nodes corresponding to these CCX nodes
        u_gmsh = vertex_p[u_loc]
        v_gmsh = vertex_p[v_loc]
        
        # Find which Gmsh edge connects them
        if (u_gmsh, v_gmsh) in gmsh_edges:
            perm.append(gmsh_edges[(u_gmsh, v_gmsh)])
        else:
            # Should not happen for valid tet perms
            perm.append(-1)
            
    return perm

import itertools
permutations = {}
v_indices = [0, 1, 2, 3]
for p in itertools.permutations(v_indices):
    name = f"Perm_{p[0]}{p[1]}{p[2]}{p[3]}"
    permutations[name] = generate_consistent_perm(p)

def generate_inp(name, perm):
    # Construct node block
    node_str = ""
    for i in range(10):
        # CalculiX nodes are 1-based
        coord = gmsh_coords[perm[i]]
        node_str += f"{i+1}, {coord[0]:.6f}, {coord[1]:.6f}, {coord[2]:.6f}\n"

    # Element definition
    elem_str = "1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10"

    content = f"""*NODE
{node_str}*ELEMENT, TYPE=C3D10, ELSET=E1
{elem_str}
*BOUNDARY
1, 1, 3, 0.0
2, 2, 3, 0.0
3, 3, 3, 0.0
*MATERIAL, NAME=ALUMINUM
*ELASTIC
70000.0, 0.33
*SOLID SECTION, ELSET=E1, MATERIAL=ALUMINUM
*STEP
*STATIC
*CLOAD
4, 3, 100.0
*EL FILE
S
*NODE FILE
U
*END STEP
"""
    with open(f"{name}.inp", "w") as f:
        f.write(content)

def run_ccx(name):
    print(f"Running {name}...")
    
    cmd = "ccx"
    import shutil
    if not shutil.which(cmd):
        cmd = "calculix-ccx"
        if not shutil.which(cmd):
             return False, "CCX binary not found"
    
    try:
        # Capture BOTH stdout and stderr
        res = subprocess.run([cmd, name], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = (res.stdout or b'').decode('utf-8', errors='ignore') + "\n" + (res.stderr or b'').decode('utf-8', errors='ignore')
        
        # Debug output for specific candidates
        if name in ["Identity", "Consistent_Swap01", "Abq_Standard", "Swap_Vert_0_1"]:
             print(f"--- DEBUG OUTPUT FOR {name} (Ret: {res.returncode}) ---\n{output}\n--- END DEBUG ---")

        # Check for semantic errors
        if res.returncode != 0:
            return False, output
            
        if "nonpositive jacobian" in output.lower() or "*error" in output.lower():
             return False, output
             
        return True, "Success"
        
    except Exception as e:
        return False, str(e)

def parse_frd(name):
    frd_file = f"{name}.frd"
    if not os.path.exists(frd_file):
        return None
    
    s33_values = []
    try:
        with open(frd_file, 'r') as f:
            lines = f.readlines()
            
        in_stress_block = False
        for line in lines:
            line_strip = line.strip()
            if "STRESS" in line:
                in_stress_block = True
                continue
            
            if in_stress_block:
                if line_strip.startswith("-3"):
                    in_stress_block = False
                    continue
                
                parts = line.split()
                if len(parts) > 4 and parts[0] == "-1":
                    # -1 node sxx syy szz ...
                    try:
                        szz = float(parts[4])
                        s33_values.append(szz)
                    except ValueError:
                        pass
        
        if not s33_values:
            return None
            
        avg = sum(s33_values) / len(s33_values)
        rng = max(s33_values) - min(s33_values)
        return f"S33_Avg={avg:.2f} S33_Rng={rng:.2f} (Min={min(s33_values):.2f} Max={max(s33_values):.2f})"
        
    except Exception as e:
        return f"ParseError: {e}"

def main():
    print("--- TET10 PERMUTATION VALIDATOR ---")
    
    for name, perm in permutations.items():
        generate_inp(name, perm)
        success, msg = run_ccx(name)
        
        status = "CRASH"
        stress_msg = ""
        if success:
            status = "RAN"
            stress_msg = parse_frd(name) or "NoStress"
            
        print(f"[{name}] -> {status} | {stress_msg}")
        
if __name__ == "__main__":
    main()
