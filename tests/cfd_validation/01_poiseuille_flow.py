import sys
import os
import json
import shutil
import math
import subprocess
import numpy as np
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

from core.geometry_analyzer import analyze_cad_geometry

def run_poiseuille_test():
    print("========================================")
    print("   CFD VALIDATION: POISEUILLE FLOW")
    print("========================================")

    # 1. Setup Environment
    # --------------------
    # --------------------
    test_name = f"Poiseuille_Test_{int(time.time())}"
    work_dir = Path(__file__).parent / "temp" / test_name
    output_dir = work_dir / "output"
    
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    # Copy CAD file
    cad_src = PROJECT_ROOT / "cad_files" / "Cylinder.step"
    cad_dest = work_dir / "Cylinder.step"
    if not cad_src.exists():
        print(f"[ERROR] Source CAD not found: {cad_src}")
        return False
    shutil.copy(cad_src, cad_dest)
    print(f"[Setup] Copied geometry to {cad_dest}")

    # 2. Analyze Geometry (to get Radius and Length)
    # ----------------------------------------------
    print("[Geo] Analyzing geometry dimensions...")
    geo = analyze_cad_geometry(str(cad_dest))
    
    # geo.bbox is (xmin, ymin, zmin, xmax, ymax, zmax)
    xmin, ymin, zmin, xmax, ymax, zmax = geo.bbox
    
    lx = xmax - xmin
    ly = ymax - ymin
    lz = zmax - zmin
    
    print(f"[Geo] Bounds: {lx:.2f} x {ly:.2f} x {lz:.2f} mm")
    
    # Heuristic: Longest dimension is Length, others are Diameter
    # For a standard cylinder, usually Z is length.
    length_mm = lz
    diameter_mm = (lx + ly) / 2.0
    radius_mm = diameter_mm / 2.0
    
    length_m = length_mm / 1000.0
    radius_m = radius_mm / 1000.0
    
    print(f"[Geo] Length: {length_m:.4f} m, Radius: {radius_m:.4f} m")

    # 3. Configure Simulation
    # -----------------------
    # Physics Parameters
    U_inlet = 1.0       # m/s
    nu = 1e-5           # m^2/s (Viscosity)
    rho = 1.0           # kg/m^3 (assuming standard kinematic viscosity relation)
    
    # Config JSON
    config = {
        "job_name": test_name,
        "physics": {
            "simulation_type": "cfd",
            "inlet_velocity": [0, 0, U_inlet], # Z-direction
            # NOTE: If cylinder is Z-aligned, we need velocity in Z.
            "kinematic_viscosity": nu,
            "density": rho,
            "virtual_wind_tunnel": False
        },
        "virtual_wind_tunnel": False, 
        "tagging_rules": [
            {
                "tag_name": "inlet",
                "entity_type": "surface",
                "selector": {"type": "z_min", "tolerance": 0.1}
            },
            {
                "tag_name": "outlet",
                "entity_type": "surface",
                "selector": {"type": "z_max", "tolerance": 0.1}
            },
            {
                "tag_name": "wall",
                "entity_type": "surface",
                "selector": {"type": "all_remaining"}
            }
        ]
    }
    
    config_path = work_dir / "poiseuille_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # 4. Run Simulation
    # -----------------
    print(f"[Run] Launching SimOps Worker...")
    
    cmd = [
        sys.executable, 
        str(PROJECT_ROOT / "simops_worker.py"),
        str(cad_dest),
        "-o", str(output_dir),
        "-c", str(config_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
        
        # Save logs for debugging
        with open(work_dir / "worker_stdout.log", 'w') as f: f.write(result.stdout)
        with open(work_dir / "worker_stderr.log", 'w') as f: f.write(result.stderr)
        
        if result.returncode != 0:
            print("[Run] Worker failed! Check logs.")
            print(result.stderr[-500:])
            return False
            
    except Exception as e:
        print(f"[Run] Execution Error: {e}")
        return False
        
    print("[Run] Simulation complete.")

    # 5. Validation
    # -------------
    # Analytical Solution (Hagen-Poiseuille)
    # Delta P = (8 * mu * L * Q) / (pi * R^4) 
    # Q = U_avg * Area = U_avg * pi * R^2
    # Delta P = (8 * mu * L * U_avg * pi * R^2) / (pi * R^4)
    # Delta P = (8 * mu * L * U_avg) / R^2
    # Note: mu = nu * rho
    
    mu = nu * rho
    dp_analytical = (8 * mu * length_m * U_inlet) / (radius_m**2)
    
    print(f"\n[Validation] Analytical dP: {dp_analytical:.4f} Pa")
    
    # Parse Result (Need to extract pressure drop from logs or results)
    # We'll check the result.json first
    result_json_path = output_dir / f"{test_name}_result.json"
    if not result_json_path.exists():
         # Fallback to default name if job_name didn't take
         result_json_path = output_dir / "Cylinder_result.json"
    
    if not result_json_path.exists():
        print("[Error] No result.json found.")
        return False
        
    with open(result_json_path) as f:
        res = json.load(f)
        
    # Extract Pressure Drop
    # Need to verify if 'pressure_drop' is in metadata. 
    # If not, we might need to parse simpleFoam logs or VTK.
    # For now, placeholder assuming we add it or read it.
    print(f"[Validation] Analytical dP: {dp_analytical:.6f} Pa")
    
    # 6. Parse VTK for Pressure Drop
    # ------------------------------
    import pyvista as pv
    
    # Find result VTK
    # Usually job_name_cfd.vtk or similar. 
    # simops_worker logic: vtk_file = output_path / f"{job_name}_result.vtk" (Need to verify)
    # Checking logs or dirt listing implies: output/job_name...
    
    # We'll search for *.vtk in output
    vtk_files = list(output_dir.glob("*.vtk"))
    if not vtk_files:
        print("[Error] No VTK files found in output.")
        return False
        
    # Prefer the one with 'result' or 'thermal'? CFD might be 'internal.vtk'?
    # Let's use the largest one or specific name if known.
    # SimOps Worker (CFD) -> result.vtk_file
    
    target_vtk = vtk_files[0]
    print(f"[Validation] processing VTK: {target_vtk.name}")
    
    try:
        mesh = pv.read(target_vtk)
        
        # Check arrays
        # OpenFOAM often exports 'p' or 'p_rgh'.
        p_name = 'p' if 'p' in mesh.point_data else None
        if not p_name and 'p_rgh' in mesh.point_data: p_name = 'p_rgh'
        
        if not p_name:
            print(f"[Error] 'p' or 'p_rgh' not found in VTK. Available: {mesh.point_data.keys()}")
            return False
            
        points = mesh.points
        p_data = mesh.point_data[p_name]
        
        # Extract slices at Z-min and Z-max (approximately)
        # Use zmin + epsilon and zmax - epsilon
        z_min_slice = zmin + 0.01 * lz
        z_max_slice = zmax - 0.01 * lz
        
        mask_inlet = (points[:, 2] < z_min_slice)
        mask_outlet = (points[:, 2] > z_max_slice)
        
        if np.sum(mask_inlet) == 0 or np.sum(mask_outlet) == 0:
             print("[Error] Could not slice Inlet/Outlet coordinates.")
             return False
             
        p_inlet = np.mean(p_data[mask_inlet])
        p_outlet = np.mean(p_data[mask_outlet])
        
        dp_sim = abs(p_inlet - p_outlet)
        
        # OpenFOAM 'p' is usually kinematic pressure (P/rho) for incompressible?
        # Need to check SimOps units. If kinematic, multiply by rho.
        # SimOps often wraps OpenFOAM. Standard OpenFOAM incompressible 'p' is m^2/s^2 (P/rho).
        # We set rho = 1.0, so Magnitude matches.
        
        print(f"[Validation] P_inlet: {p_inlet:.6f}, P_outlet: {p_outlet:.6f}")
        print(f"[Validation] Simulated dP:  {dp_sim:.6f} Pa")
        
    except Exception as e:
        print(f"[Error] VTK analysis failed: {e}")
        return False
    
    error = abs(dp_sim - dp_analytical) / dp_analytical * 100.0
    print(f"[Validation] Error: {error:.2f}%")
    
    if error < 10.0:
        print("\n[PASS] Poiseuille Flow Validation Successful!")
        return True
    else:
        print("\n[FAIL] Error exceeds 10% threshold.")
        return False

if __name__ == "__main__":
    success = run_poiseuille_test()
    sys.exit(0 if success else 1)
