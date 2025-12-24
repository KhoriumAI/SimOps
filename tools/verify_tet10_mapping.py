# tools/verify_tet10_mapping.py
import os
import subprocess
import numpy as np
import shutil

def write_inp(filename, permutation):
    """
    Writes a single Tet10 element input deck with the given node permutation.
    """
    with open(filename, 'w') as f:
        f.write("*HEADING\nTet10 Unit Test\n")
        f.write("*NODE\n")
        # Unit Tetrahedron nodes (10 nodes)
        # Corners (0-3)
        coords = [
            [0,0,0], [1,0,0], [0,1,0], [0,0,1], # 0,1,2,3
            # Midside nodes (Gmsh ordering usually: 0-1, 1-2, 2-0, 0-3, 1-3, 2-3)
            [0.5,0,0], [0.5,0.5,0], [0,0.5,0], # 4,5,6
            [0,0,0.5], [0.5,0,0.5], [0,0.5,0.5] # 7,8,9  
        ] 
        # Note: Gmsh standard ordering for Tet10 (Tag 11):
        # Nodes: 0, 1, 2, 3 (vertices)
        #        4 (0-1), 5 (1-2), 6 (2-0)  (bottom edges)
        #        7 (0-3), 8 (1-3), 9 (2-3)  (vertical edges)
        
        for i, (x,y,z) in enumerate(coords):
            f.write(f"{i+1}, {x:.6f}, {y:.6f}, {z:.6f}\n")
            
        f.write("*ELEMENT, TYPE=C3D10, ELSET=E1\n")
        
        # Apply Permutation (Input is 0-indexed list of indices)
        # CCX expects 1-based node IDs
        p_nodes = [n + 1 for n in permutation]
        
        f.write(f"1, {p_nodes[0]}, {p_nodes[1]}, {p_nodes[2]}, {p_nodes[3]}, {p_nodes[4]}, {p_nodes[5]}, {p_nodes[6]}\n")
        f.write(f"{p_nodes[7]}, {p_nodes[8]}, {p_nodes[9]}\n")
        
        f.write("*MATERIAL, NAME=Steel\n")
        f.write("*ELASTIC\n210000.0, 0.3\n")
        f.write("*SOLID SECTION, ELSET=E1, MATERIAL=Steel\n")
        
        f.write("*BOUNDARY\n")
        # Fix Base (Z=0, nodes 0,1,2, 4,5,6)
        # Fix Node 1 (0,0,0) completely
        f.write("1, 1, 3, 0.0\n")
        # Fix Node 2 (1,0,0) in Y, Z
        f.write("2, 2, 3, 0.0\n")
        # Fix Node 3 (0,1,0) in Z
        f.write("3, 3, 3, 0.0\n")
        
        f.write("*STEP\n*STATIC\n")
        # DISPLACEMENT CONTROL for Uniform Strain/Stress
        # Pull Top Node (Node 4 / Index 3) in Z by 0.001
        f.write("*BOUNDARY\n")
        f.write("4, 3, 3, 0.001\n")
        
        f.write("*EL FILE\nS\n")
        f.write("*EL PRINT, ELSET=E1\nS\n") # Also write to dat
        f.write("*END STEP")

def parse_dat(jobname):
    """
    Parses .dat file for Stress SZZ.
    Actually we used *EL FILE, so we get .dat with specialized format or .frd?
    *EL PRINT gives .dat, *EL FILE gives .frd.
    Let's use *EL PRINT for easier parsing if we want .dat, 
    but *EL FILE is standard. 
    Let's switch to *EL PRINT in write_inp for text parsing.
    """
    pass

def parse_frd(frd_file):
    """
    Parses .frd file for SZZ values.
    """
    if not os.path.exists(frd_file):
        return None
        
    szz_values = []
    
    with open(frd_file, 'r') as f:
        lines = f.readlines()
        
    reading = False
    for line in lines:
        if line.startswith(" -4  STRESS"):
            reading = True
            continue
        if line.startswith(" -3"):
            reading = False
            continue
            
        if reading and line.startswith(" -1"):
            # Format: -1 <nodeID> <SXX> <SYY> <SZZ> <SXY> <SYZ> <SZX>
            parts = line.split()
            if len(parts) >= 5:
                try:
                    szz = float(parts[4]) # 4th val usually? 
                    # Wait, FRD format for STRESS:
                    # columns: node, Sxx, Syy, Szz, Sxy, Syz, Szx
                    # parts[0]=-1, parts[1]=nid, parts[2]=Sxx, parts[3]=Syy, parts[4]=Szz
                    szz_values.append(szz)
                except: pass
                
    return np.array(szz_values)

def check_permutation(perm, name):
    print(f"Testing Permutation: {name} {perm}")
    job = "tet_test"
    inp_file = f"{job}.inp"
    
    write_inp(inp_file, perm)
    
    # Run CCX
    # Check OS for implicit path or use 'ccx'
    ccx = "ccx"
    if os.name == 'nt' and os.path.exists(r"C:\calculix\calculix_2.22_4win\ccx.exe"):
        ccx = r"C:\calculix\calculix_2.22_4win\ccx.exe"
        
    try:
        subprocess.run([ccx, job], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError:
        print(f"  [FAIL] CCX Crashed (Negative Jacobian?)")
        return False

    # Check Results
    frd_file = f"{job}.frd"
    szz = parse_frd(frd_file)
    
    if szz is None or len(szz) == 0:
        print(f"  [FAIL] No Output Data")
        return False
        
    std_dev = np.std(szz)
    mean_val = np.mean(szz)
    
    print(f"  SZZ Mean: {mean_val:.2f}, StdDev: {std_dev:.4f}")
    
    if std_dev < 1.0: # 1 MPa tolerance
        print(f"  [PASS] UNIFORM STRESS FIELD FOUND!")
        return True
    else:
        print(f"  [FAIL] Stress field noisy (StdDev > 1.0)")
        return False

import itertools

def main():
    print("Starting Brute Force Search for Tet10 Mapping...")
    print("Assumed Geometry: Unit Right Tetrahedron")
    print("Corners: 0(0,0,0), 1(1,0,0), 2(0,1,0), 3(0,0,1)")
    print("Target Stress: ~210 MPa (E=210GPa, Strain=0.001)")
    
    # 1. Fix Corners (Identity 0,1,2,3)
    # 2. Permute Edges (Indices 4,5,6,7,8,9)
    edge_indices = [4, 5, 6, 7, 8, 9]
    
    # Generate all 720 permutations
    edge_perms = list(itertools.permutations(edge_indices))
    total = len(edge_perms)
    print(f"Checking {total} edge permutations...")
    
    winner = None
    best_std = 1e9
    best_mean_diff = 1e9
    best_perm = None
    
    for i, p_edges in enumerate(edge_perms):
        if i % 100 == 0:
            print(f"  Progress: {i}/{total}")
            
        # Construct full permutation: [Corners] + [Edges]
        perm = [0, 1, 2, 3] + list(p_edges)
        
        # Run Check
        # Optimization: Don't print every run
        # Inline the check logic to avoid overhead? No, function ok.
        
        job = f"tet_BF_{i}"
        inp_file = f"{job}.inp"
        write_inp(inp_file, perm)
        
        ccx = "ccx"
        if os.name == 'nt' and os.path.exists(r"C:\calculix\calculix_2.22_4win\ccx.exe"):
             ccx = r"C:\calculix\calculix_2.22_4win\ccx.exe"
             
        try:
            subprocess.run([ccx, job], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
            # Parse
            frd_file = f"{job}.frd"
            szz = parse_frd(frd_file)
            
            if szz is not None and len(szz) > 0:
                std = np.std(szz)
                mean = np.mean(szz)
                diff = abs(mean - 210.0) # Target 210
                
                if std < best_std:
                    best_std = std
                    best_perm = perm
                if diff < best_mean_diff:
                    best_mean_diff = diff
                    
                # Strict Pass Criteria
                if std < 1.0 and diff < 10.0: # 1MPa noise, within 10MPa of target
                    print(f"  [FOUND] Permutation {i}: {perm}")
                    print(f"  Stats: Mean={mean:.2f}, Std={std:.4f}")
                    winner = perm
                    break
        except:
            pass
            
        # Cleanup
        try:
            if os.path.exists(inp_file): os.remove(inp_file)
            if os.path.exists(f"{job}.frd"): os.remove(f"{job}.frd")
        except: pass

    if winner:
        print(f"\nWINNER FOUND: {winner}")
    else:
        print(f"\nNo winner found meeting strict criteria.")
        print(f"Best Std: {best_std:.4f}")
        print(f"Best Mean Diff: {best_mean_diff:.2f}")
        if best_perm is not None:
             print(f"Best Candidate Permutation: {best_perm}")
             print("Recommended Action: Verify this permutation manually or accept tolerance.")

if __name__ == "__main__":
    main()
