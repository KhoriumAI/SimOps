import os
import subprocess
import numpy as np
from pathlib import Path

# Config
CCX_PATH = r"C:\Users\markm\Downloads\SimOps\calculix_native\CalculiX-2.23.0-win-x64\bin\ccx.exe"
JOB_NAME = "si_cube_test"
CWD = Path.cwd()

def write_si_inp(filename):
    # a = 0.05m (50mm)
    # E = 68.9e9 Pa
    # rho = 2700 kg/m3
    # G = 9.81 m/s2
    
    with open(filename, 'w') as f:
        f.write("*HEADING\nSI_Sanity_Check_M_KG_PA\n")
        
        # Nodes (0.05m cube)
        f.write("*NODE\n")
        f.write("1, 0.00, 0.00, 0.00\n")
        f.write("2, 0.05, 0.00, 0.00\n")
        f.write("3, 0.05, 0.05, 0.00\n")
        f.write("4, 0.00, 0.05, 0.00\n")
        f.write("5, 0.00, 0.00, 0.05\n")
        f.write("6, 0.05, 0.00, 0.05\n")
        f.write("7, 0.05, 0.05, 0.05\n")
        f.write("8, 0.00, 0.05, 0.05\n")
        
        # Element (Single C3D8)
        f.write("*ELEMENT, TYPE=C3D8, ELSET=Eall\n")
        f.write("1, 1, 2, 3, 4, 5, 6, 7, 8\n")
        
        # Section & Material
        f.write("*SOLID SECTION, ELSET=Eall, MATERIAL=AL6061\n")
        f.write("*MATERIAL, NAME=AL6061\n")
        f.write("*ELASTIC\n")
        f.write("6.89E+10, 0.33\n")
        f.write("*DENSITY\n")
        f.write("2700.0\n")
        
        # Boundary (Fixed Base Z=0)
        f.write("*NSET, NSET=N_Fixed\n")
        f.write("1, 2, 3, 4\n")
        f.write("*BOUNDARY\n")
        f.write("N_Fixed, 1, 3, 0.0\n")
        
        # Step
        f.write("*STEP\n")
        f.write("*STATIC\n")
        # Gravity (1G = 9.81 m/s2 in -Z)
        f.write("*DLOAD\n")
        f.write("Eall, GRAV, 9.81, 0.0, 0.0, -1.0\n")
        
        f.write("*NODE FILE\n")
        f.write("U, RF\n")
        f.write("*EL FILE\n")
        f.write("S\n")
        f.write("*END STEP\n")

def write_point_load_inp(filename):
    with open(filename, 'w') as f:
        f.write("*HEADING\nPoint_Load_Test\n")
        f.write("*NODE\n")
        f.write("1, 0.00, 0.00, 0.00\n")
        f.write("2, 0.05, 0.00, 0.00\n")
        f.write("3, 0.05, 0.05, 0.00\n")
        f.write("4, 0.00, 0.05, 0.00\n")
        f.write("5, 0.00, 0.00, 0.05\n")
        f.write("6, 0.05, 0.00, 0.05\n")
        f.write("7, 0.05, 0.05, 0.05\n")
        f.write("8, 0.00, 0.05, 0.05\n")
        f.write("*ELEMENT, TYPE=C3D8, ELSET=Eall\n")
        f.write("1, 1, 2, 3, 4, 5, 6, 7, 8\n")
        f.write("*SOLID SECTION, ELSET=Eall, MATERIAL=AL6061\n")
        f.write("*MATERIAL, NAME=AL6061\n")
        f.write("*ELASTIC\n")
        f.write("6.89E+10, 0.33\n")
        f.write("*BOUNDARY\n")
        f.write("1, 1, 3, 0.0\n")
        f.write("2, 1, 3, 0.0\n")
        f.write("3, 1, 3, 0.0\n")
        f.write("4, 1, 3, 0.0\n")
        f.write("*STEP\n")
        f.write("*STATIC\n")
        f.write("*CLOAD\n")
        f.write("5, 3, -100.0\n") # 100N on one node
        f.write("*NODE FILE\n")
        f.write("U, RF\n")
        f.write("*EL FILE\n")
        f.write("S\n")
        f.write("*END STEP\n")

def parse_frd_simple(filename, expected_mass_kg=0.3375):
    print(f"\nAnalyzing {filename}...")
    if not os.path.exists(filename):
        print(f"ERROR: {filename} not found.")
        return

    with open(filename, 'r') as f:
        lines = f.readlines()

    mode = None
    node_data = {} # nid -> {mode: [vals]}

    for line in lines:
        ls = line.strip()
        if ls.startswith("-4"):
            if "DISP" in ls: mode = "DISP"
            elif "STRESS" in ls: mode = "STRESS"
            elif "FORC" in ls: mode = "FORC"
            else: mode = None
            continue
        
        if mode and ls.startswith("-1"):
            nid = int(line[3:13])
            vals = []
            
            # Robust Regex for Windows CalculiX Binary (Overlapping columns)
            # Slice line[13:] to exclude Node ID column fully
            import re
            vals_str = re.findall(r'[-+]?\d*\.\d+[Ee][-+]+\d{3}', line[13:])
            
            try:
                vals = [float(v) for v in vals_str]
            except: pass
            
            if nid not in node_data: node_data[nid] = {}
            if vals:
                node_data[nid][mode] = vals

    # Analysis
    max_disp = 0.0
    max_stress = 0.0
    rf_total = np.zeros(3)
    
    for nid, data in node_data.items():
        if "DISP" in data:
            mag = np.linalg.norm(data["DISP"][:3])
            max_disp = max(max_disp, mag)
        if "STRESS" in data:
            s = data["STRESS"]
            if len(s)>=6:
                vm = np.sqrt(0.5 * ((s[0]-s[1])**2 + (s[1]-s[2])**2 + (s[2]-s[0])**2 + 6*(s[3]**2 + s[4]**2 + s[5]**2)))
                max_stress = max(max_stress, vm)
        if "FORC" in data:
            rf_total += np.array(data["FORC"][:3])

    print(f"Max Displacement: {max_disp:.6e}")
    print(f"Max Stress:       {max_stress:.6e}")
    print(f"Total Reaction Force: {rf_total}")
    
    # Detail for specific nodes
    for nid in [1, 5]:
        if nid in node_data:
            print(f"Node {nid}:")
            if "DISP" in node_data[nid]: print(f"  U: {node_data[nid]['DISP'][:3]}")
            if "FORC" in node_data[nid]: print(f"  F: {node_data[nid]['FORC'][:3]}")

    expected_f = expected_mass_kg * 9.81
    if expected_mass_kg > 0:
        print(f"Expected Force ({expected_mass_kg:.4f}kg * 9.81): {expected_f:.4f} N")
        if abs(np.linalg.norm(rf_total) - expected_f) > 0.1:
            print("!!! FORCE MISMATCH DETECTED !!!")

def run_standard_benchmark():
    job_name = "benchmark"
    with open(f"{job_name}.inp", 'w') as f:
        f.write("*HEADING\nVALIDATION_C3D8\n*NODE\n1, 0,0,0\n2, 0,10,0\n3, 0,10,10\n4, 0,0,10\n5, 100,0,0\n6, 100,10,0\n7, 100,10,10\n8, 100,0,10\n")
        f.write("*ELEMENT, TYPE=C3D8, ELSET=Eall\n1, 1,2,3,4, 5,6,7,8\n")
        f.write("*BOUNDARY\n1, 1, 3\n2, 1, 3\n3, 1, 3\n4, 1, 3\n")
        f.write("*MATERIAL, NAME=STEEL\n*ELASTIC\n210000.0, 0.3\n")
        f.write("*SOLID SECTION, ELSET=Eall, MATERIAL=STEEL\n")
        f.write("*STEP\n*STATIC\n*CLOAD\n5, 3, -1000.0\n6, 3, -1000.0\n7, 3, -1000.0\n8, 3, -1000.0\n")
        f.write("*NODE FILE\nU, RF\n*END STEP\n")
    
    print(f"\n--- RUNNING STANDARD BENCHMARK (Cantilever) ---")
    try:
        subprocess.run([CCX_PATH, job_name], capture_output=True, text=True, check=True)
        parse_frd_simple(f"{job_name}.frd", expected_mass_kg=0) # Only concerned with displacement
    except Exception as e:
        print(f"Benchmark failed: {e}")

def write_full_float_inp(filename):
    with open(filename, 'w') as f:
        f.write("*HEADING\nFull_Float_Test\n")
        f.write("*NODE\n1, 0, 0, 0\n2, 50, 0, 0\n3, 50, 50, 0\n4, 0, 50, 0\n5, 0, 0, 50\n6, 50, 0, 50\n7, 50, 50, 50\n8, 0, 50, 50\n")
        f.write("*ELEMENT, TYPE=C3D8, ELSET=Eall\n1, 1, 2, 3, 4, 5, 6, 7, 8\n")
        f.write("*SOLID SECTION, ELSET=Eall, MATERIAL=AL\n")
        f.write("*MATERIAL, NAME=AL\n*ELASTIC\n68900000000.0, 0.33\n*DENSITY\n2700.0\n")
        f.write("*BOUNDARY\n1, 1, 3, 0.0\n2, 1, 3, 0.0\n3, 1, 3, 0.0\n4, 1, 3, 0.0\n")
        f.write("*STEP\n*STATIC\n*DLOAD\nEall, GRAV, 9.81, 0.0, 0.0, -1.0\n")
        f.write("*NODE FILE\nU, RF\n*EL FILE\nS\n*END STEP\n")

def write_user_benchmark_inp(filename):
    with open(filename, 'w') as f:
        f.write("*HEADING\nVerified_Gravity_Test_mm_tonne\n")
        f.write("*NODE\n1, 0.0, 0.0, 0.0\n2, 50.0, 0.0, 0.0\n3, 50.0, 50.0, 0.0\n4, 0.0, 50.0, 0.0\n5, 0.0, 0.0, 50.0\n6, 50.0, 0.0, 50.0\n7, 50.0, 50.0, 50.0\n8, 0.0, 50.0, 50.0\n")
        f.write("*ELEMENT, TYPE=C3D8, ELSET=Eall\n1, 1, 2, 3, 4, 5, 6, 7, 8\n")
        f.write("*SOLID SECTION, ELSET=Eall, MATERIAL=ALUMINUM\n")
        f.write("*MATERIAL, NAME=ALUMINUM\n*ELASTIC\n68900.0, 0.33\n*DENSITY\n2.7E-09\n")
        f.write("*BOUNDARY\n1, 1, 3, 0.0\n2, 2, 3, 0.0\n4, 1, 3, 0.0\n3, 3, 3, 0.0\n")
        f.write("*STEP\n*STATIC\n*DLOAD\nEall, GRAV, 9810.0, 0.0, 0.0, -1.0\n")
        f.write("*NODE FILE\nU, RF\n*EL FILE\nS\n*END STEP\n")

def run_si_test():
    # 0. Full Float Test (Rule out scientific notation bugs)
    job_ff = "si_full_float"
    print(f"\n--- RUNNING FULL FLOAT TEST (SI Units) ---")
    write_full_float_inp(f"{job_ff}.inp")
    try:
        subprocess.run([CCX_PATH, job_ff], capture_output=True, text=True, check=True)
        # Analytical: 50m cube Al. rho=2700. g=9.81.
        # Vol = 125000. Mass = 337500000 kg. Force = 3.3e9 N. 
        # Area = 2500. Stress ~= 1.3e6 Pa.
        parse_frd_simple(f"{job_ff}.frd", expected_mass_kg=2700*125000)
    except Exception as e:
        print(f"Full Float Test failed: {e}")

    # 00. User Benchmark (mm-tonne)
    job_ub = "si_user_benchmark"
    print(f"\n--- RUNNING USER BENCHMARK (mm-tonne) ---")
    write_user_benchmark_inp(f"{job_ub}.inp")
    try:
        subprocess.run([CCX_PATH, job_ub], capture_output=True, text=True, check=True)
        parse_frd_simple(f"{job_ub}.frd", expected_mass_kg=0.3375)
    except Exception as e:
        print(f"User Benchmark failed: {e}")

    # 0. Unity Test (E=1, rho=1, a=1, g=1)
    job_ut = "si_unity_test"
    print(f"\n--- RUNNING UNITY TEST (E=1, rho=1, a=1, g=1) ---")
    with open(f"{job_ut}.inp", 'w') as f:
        f.write("*HEADING\nUnity_Test\n")
        f.write("*NODE\n1,0,0,0\n2,1,0,0\n3,1,1,0\n4,0,1,0\n5,0,0,1\n6,1,0,1\n7,1,1,1\n8,0,1,1\n")
        f.write("*ELEMENT, TYPE=C3D8, ELSET=Eall\n1, 1, 2, 3, 4, 5, 6, 7, 8\n")
        f.write("*SOLID SECTION, ELSET=Eall, MATERIAL=M1\n")
        f.write("*MATERIAL, NAME=M1\n*ELASTIC\n1.0, 0.0\n*DENSITY\n1.0\n")
        f.write("*BOUNDARY\n1,1,3,0\n2,1,3,0\n3,1,3,0\n4,1,3,0\n")
        f.write("*STEP\n*STATIC\n*DLOAD\nEall, GRAV, 1.0, 0, 0, -1\n")
        f.write("*NODE FILE\nU, RF\n*EL FILE\nS\n*END STEP\n")
    try:
        subprocess.run([CCX_PATH, job_ut], capture_output=True, text=True, check=True)
        parse_frd_simple(f"{job_ut}.frd", expected_mass_kg=1.0)
    except Exception as e:
        print(f"Unity Test failed: {e}")

    # 1. Gravity Test
    inp_file = f"{JOB_NAME}.inp"
    frd_file = f"{JOB_NAME}.frd"
    print(f"\n--- RUNNING GRAVITY TEST ---")
    write_si_inp(inp_file)
    print(f"Executing CalculiX...")
    try:
        proc = subprocess.run([CCX_PATH, JOB_NAME], capture_output=True, text=True, check=True)
        print("--- CCX STDOUT ---")
        print(proc.stdout)
        print("--- CCX STDERR ---")
        print(proc.stderr)
        print("CalculiX finished successfully.")
        parse_frd_simple(frd_file)
    except Exception as e:
        print(f"CalculiX failed: {e}")
        if hasattr(e, 'stdout'): print(e.stdout)
        if hasattr(e, 'stderr'): print(e.stderr)
        return

    # 2. Point Load Test
    # ... (skipping for brevity if not needed, but keeping for now) ...

def run_mm_tonne_repro():
    job_name = "repro_mm_tonne"
    with open(f"{job_name}.inp", 'w') as f:
        f.write("*HEADING\nREPRO_MM_TONNE\n")
        f.write("*NODE\n1, 0., 0., 0.\n2, 50., 0., 0.\n3, 50., 50., 0.\n4, 0., 50., 0.\n5, 0., 0., 50.\n6, 50., 0., 50.\n7, 50., 50., 50.\n8, 0., 50., 50.\n")
        f.write("*ELEMENT, TYPE=C3D8, ELSET=Eall\n1, 1,2,3,4, 5,6,7,8\n")
        f.write("*SOLID SECTION, ELSET=Eall, MATERIAL=AL\n")
        f.write("*MATERIAL, NAME=AL\n*ELASTIC\n68900.0, 0.33\n*DENSITY\n2.7E-09\n")
        f.write("*BOUNDARY\n1, 1, 3, 0.0\n2, 1, 3, 0.0\n3, 1, 3, 0.0\n4, 1, 3, 0.0\n")
        f.write("*STEP\n*STATIC\n*DLOAD\nEall, GRAV, 9810.0, 0.0, 0.0, -1.0\n")
        f.write("*NODE FILE\nU, RF\n*END STEP\n")
    
    print(f"\n--- RUNNING MM-TONNE REPRO ---")
    try:
        subprocess.run([CCX_PATH, job_name], capture_output=True, text=True, check=True)
        # Expected: Force ~ 3.31 N. Disp ~ 1e-6 mm.
        parse_frd_simple(f"{job_name}.frd", expected_mass_kg=0.3375)
    except Exception as e:
        print(f"Repro failed: {e}")

if __name__ == "__main__":
    try:
        # run_standard_benchmark()
        run_mm_tonne_repro()
    except Exception as e:
        print(e)
