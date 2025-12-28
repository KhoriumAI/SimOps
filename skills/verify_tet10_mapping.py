import gmsh
import numpy as np
import os
import subprocess
import sys
from pathlib import Path

def create_single_tet10():
    try:
        gmsh.initialize()
    except:
        pass 
        
    gmsh.model.add("test_tet10")
    
    # Create valid Tet geometry (0,0,0) to unitary axes
    p1 = gmsh.model.geo.addPoint(0, 0, 0, 1.0)
    p2 = gmsh.model.geo.addPoint(1, 0, 0, 1.0)
    p3 = gmsh.model.geo.addPoint(0, 1, 0, 1.0)
    p4 = gmsh.model.geo.addPoint(0, 0, 1, 1.0)
    
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p1)
    l4 = gmsh.model.geo.addLine(p1, p4)
    l5 = gmsh.model.geo.addLine(p2, p4)
    l6 = gmsh.model.geo.addLine(p3, p4)
    
    cl1 = gmsh.model.geo.addCurveLoop([l1, l2, l3])
    cl2 = gmsh.model.geo.addCurveLoop([l1, l5, -l4])
    cl3 = gmsh.model.geo.addCurveLoop([l2, l6, -l5])
    cl4 = gmsh.model.geo.addCurveLoop([l3, l4, -l6])
    
    s1 = gmsh.model.geo.addSurfaceFilling([cl1])
    s2 = gmsh.model.geo.addSurfaceFilling([cl2])
    s3 = gmsh.model.geo.addSurfaceFilling([cl3])
    s4 = gmsh.model.geo.addSurfaceFilling([cl4])
    
    sl1 = gmsh.model.geo.addSurfaceLoop([s1, s2, s3, s4])
    v1 = gmsh.model.geo.addVolume([sl1])
    
    gmsh.model.geo.synchronize()
    
    # Mesh settings
    gmsh.option.setNumber("Mesh.ElementOrder", 2)
    # Force coarse mesh to get ~1 element
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 10.0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 10.0)

    try:
        gmsh.model.mesh.generate(3)
    except Exception as e:
        print(f"Mesh generation warning: {e}")
    
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_coords = node_coords.reshape(-1, 3)
    
    elem_types, elem_tags, elem_nodes = gmsh.model.mesh.getElements(dim=3)
    
    tet10_nodes = None
    for i, etype in enumerate(elem_types):
        if etype == 11:
            tet10_nodes = elem_nodes[i].reshape(-1, 10).astype(int)
            break
            
    if tet10_nodes is None:
        print("Failed to generate Tet10 element")
        gmsh.finalize()
        return None, None
        
    print(f"Generated {len(tet10_nodes[0]) if len(tet10_nodes.shape)>1 else 0} Tet10 elements total (showing first).")
    raw_nodes = tet10_nodes[0]
    node_map = dict(zip(node_tags, node_coords))
    
    gmsh.finalize()
    return raw_nodes, node_map

def write_inp(filename, nodes, node_map, permutation):
    permuted_nodes = nodes[permutation]
    
    # Identify nodes to fix (e.g. at Z=0) to prevent rigid body motion
    fixed_tags = []
    for tag in nodes:
        c = node_map[tag]
        if abs(c[2]) < 1e-6: # Z=0
            fixed_tags.append(tag)
            
    with open(filename, 'w') as f:
        f.write("*HEADING\nTest Tet10\n")
        f.write("*NODE\n")
        for tag in nodes:
             c = node_map[tag]
             f.write(f"{int(tag)}, {c[0]:.6f}, {c[1]:.6f}, {c[2]:.6f}\n")
             
        f.write("*ELEMENT, TYPE=C3D10, ELSET=E_Vol\n")
        f.write("1")
        for tag in permuted_nodes:
            f.write(f", {int(tag)}")
        f.write("\n")
        
        f.write("*MATERIAL, NAME=STEEL\n*ELASTIC\n210000, 0.3\n*SOLID SECTION, ELSET=E_Vol, MATERIAL=STEEL\n")
        f.write("*BOUNDARY\n")
        if fixed_tags:
            for tag in fixed_tags:
                f.write(f"{int(tag)}, 1, 3, 0.0\n")
        else:
             # Fallback fix first node
             f.write(f"{int(nodes[0])}, 1, 3, 0.0\n")
            
        f.write("*STEP\n*STATIC\n*END STEP\n")

def run_ccx(jobname):
    paths = [
        r"C:\Users\markm\Downloads\SimOps\calculix_native\CalculiX-2.23.0-win-x64\bin\ccx.exe",
        r"C:\calculix\calculix_2.22_4win\ccx.exe", 
        "ccx"
    ]
    binary = None
    for p in paths:
         if os.path.exists(p) or p == "ccx":
             try:
                 subprocess.run([p, "-v"], capture_output=True)
                 binary = p
                 break
             except:
                 pass
    if not binary:
        print("[ERROR] CCX binary not found.")
        return False
        
    print(f"Running {binary} {jobname}...")
    try:
        proc = subprocess.run([binary, jobname], capture_output=True, text=True, timeout=10)
        output = proc.stdout + proc.stderr
        
        if "Negative Jacobian" in output or "negative volume" in output.lower():
             print("[FAILED] Negative Volume/Jacobian detected")
             return False
        elif "Job finished" in output:
             print("[SUCCESS] Job finished without topology errors")
             return True
        else:
             print(f"[UNKNOWN] Return Code: {proc.returncode}")
             print("Output Sample (Last 1000 chars):")
             print(output[-1000:]) 
             return proc.returncode == 0
    except Exception as e:
        print(f"[ERROR] Execution failed: {e}")
        return False

def main():
    print("Generating single Tet10 element via Gmsh...")
    raw_nodes, node_map = create_single_tet10()
    
    if raw_nodes is None: return
    
    print(f"GMSH Nodes (Tag Order from getElements): {raw_nodes}")
    print(f"Corresponding Coords:")
    for n in raw_nodes:
        print(f"  {n}: {node_map[n]}")
        
    mappings = {
        "Identity": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "Structural_Current": [0, 1, 2, 3, 4, 5, 6, 7, 9, 8],
        "Thermal_Current": [0, 1, 2, 3, 8, 5, 4, 7, 9, 6],
        "Swap01": [1, 0, 2, 3, 4, 6, 5, 8, 7, 9],
        "Swap89": [0, 1, 2, 3, 4, 5, 6, 7, 9, 8],
        "Experimental": [0, 3, 1, 2, 7, 9, 4, 6, 8, 5]
    }
    
    results = {}
    
    for name, perm in mappings.items():
        fname = f"verify_{name}"
        job_path = Path.cwd() / fname
        print(f"\n--- Testing Permutation: {name} ---")
        write_inp(f"{fname}.inp", raw_nodes, node_map, perm)
        success = run_ccx(fname)
        results[name] = success
        
    print("\n--- SUMMARY ---")
    for name, success in results.items():
        print(f"{name}: {'PASS' if success else 'FAIL'}")

if __name__ == "__main__":
    main()
