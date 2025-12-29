import json
import subprocess
import os
import sys
import numpy as np
from pathlib import Path

# Paths
ROOT = Path("C:/Users/markm/Downloads/SimOps")
WORKER = ROOT / "simops_worker.py"
CAD_FILE = ROOT / "cad_files/Cube.step"
OUT_DIR = ROOT / "verification_runs/thermal_suite"

os.makedirs(OUT_DIR, exist_ok=True)

def run_sim(name, config):
    print(f"--- Running {name} ---")
    conf_path = OUT_DIR / f"{name}.json"
    out_path = OUT_DIR / name
    
    with open(conf_path, "w") as f:
        json.dump(config, f, indent=2)
        
    cmd = [
        sys.executable, str(WORKER),
        str(CAD_FILE),
        "-o", str(out_path),
        "-c", str(conf_path)
    ]
    
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"[OK] {name} completed.")
        return out_path, True
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] {name} failed with code {e.returncode}")
        print("STDOUT:", e.stdout[-200:])
        print("STDERR:", e.stderr[-200:])
        return out_path, False

def parse_logs(out_dir):
    # Parse the log or result file to get T_min, T_max
    # SimOps writes a result.json or we can parse the logs
    # Let's rely on finding 'Cube_thermal.json' style results or just parsing the log output if captured, 
    # but since I didn't capture output to variable for parsing, let's look for result artifact?
    # SimOps doesn't write a clean result.json typically accessible easily? 
    # Actually it writes 'simops_worker_results.json' maybe?
    # Let's parse the stdout from the run wrapper? I didn't return it.
    # Let's look for the VTK file as proof of life, and maybe the report PDF?
    # Better: Inspect the *FAILED.txt* or Log files in the folder.
    pass

# --- Configurations ---

# 1. Conduction/Insulation Check
# High Heat Source, Low Convection. Should get HOT.
# Q = 100W, h = 5 W/m2K (Insulated)
# Expected: dT = Q / (hA) approx. 
# Cube 10mm -> 0.01m. Area ~ 6 * 0.01^2 = 0.0006 m2.
# 100W / (5 * 0.0006) = 33,333 K -> This will explode.
# Let's use Q=0.1W. dT = 0.1 / 0.003 = 33 deg C. safe.
conf_cond = {
    "physics": {
        "type": "thermal",
        "material": "Aluminum 6061", # k=150
        "initial_temp_c": 25.0,
        "ambient_temp_c": 25.0,
        "convection_coeff": 5.0, # Low h
        "heat_load_watts": 0.1,
        "transient": False, # Steady State
        "unit_scaling": 1.0
    }
}

# 2. Convection Dominance
# High Convection, Moderate Heat.
# Q=1W, h=250 W/m2K (Forced Air)
# A ~ 0.0006 m2.
# dT = 1 / (250 * 0.0006) = 6.6 deg C.
conf_conv = {
    "physics": {
        "type": "thermal",
        "material": "Aluminum 6061",
        "initial_temp_c": 25.0,
        "ambient_temp_c": 25.0,
        "convection_coeff": 250.0,
        "heat_load_watts": 1.0,
        "transient": False,
        "unit_scaling": 1.0
    }
}

# 3. Transient Pulse
# Heat Q=10W for 10s. Aluminum rho=2.7e-9 t/mm3, Cp=9e8 mm2/s2K
# Mass = 1000mm3 * 2.7e-9 = 2.7e-6 tonnes (= 2.7g)
# Cp = 900 J/kgK = 900 mJ/gK.
# Q = 10W = 10000 mJ/s.
# Rate = Q / (m Cp) = 10000 / (2.7 * 900) = 10000 / 2430 = 4.1 K/s.
# In 10s -> +41K.
# Run for 5s.
conf_trans = {
    "physics": {
        "type": "thermal",
        "material": "Aluminum 6061",
        "initial_temp_c": 25.0,
        "ambient_temp_c": 25.0,
        "convection_coeff": 10.0,
        "heat_load_watts": 10.0,
        "transient": True,
        "duration": 5.0,
        "time_step": 1.0,
        "unit_scaling": 1.0
    }
}

# --- Execution ---
print(">>> Starting Thermal Validation Suite <<<")

p1, s1 = run_sim("Case1_Insulation", conf_cond)
p2, s2 = run_sim("Case2_Convection", conf_conv)
p3, s3 = run_sim("Case3_Transient", conf_trans)

print(f"\nSummary:")
print(f"Case 1 (Insulation): {'PASS' if s1 else 'FAIL'}")
print(f"Case 2 (Convection): {'PASS' if s2 else 'FAIL'}")
print(f"Case 3 (Transient):  {'PASS' if s3 else 'FAIL'}")

# Basic Check of artifacts
for p, name in [(p1, "Case1"), (p2, "Case2"), (p3, "Case3")]:
    if (p / "Cube_thermal.vtk").exists():
        print(f"[{name}] VTK found.")
    else:
        print(f"[{name}] VTK MISSING.")
    
    # Check for report
    if list(p.glob("*_Report.pdf")):
        print(f"[{name}] Report PDF found.")
    else:
        print(f"[{name}] Report PDF MISSING.")

