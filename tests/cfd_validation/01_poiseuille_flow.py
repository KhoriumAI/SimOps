import sys
import os

# Force unbuffered output so the IDE catches crashes immediately
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
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


def generate_pipe_geometry(output_path: Path, radius_mm: float, length_mm: float) -> Path:
    """
    Generate a cylindrical pipe geometry using gmsh.
    
    Args:
        output_path: Directory to save the STEP file
        radius_mm: Pipe radius in mm
        length_mm: Pipe length in mm
        
    Returns:
        Path to generated STEP file
    """
    import gmsh
    
    step_file = output_path / "Pipe.step"
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)  # Quiet mode
    
    try:
        gmsh.model.add("Pipe")
        
        # Create cylinder along Z-axis
        # gmsh.model.occ.addCylinder: x, y, z, dx, dy, dz, radius
        # We place inlet at z=0, outlet at z=length
        cylinder_tag = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, length_mm, radius_mm)
        
        gmsh.model.occ.synchronize()
        
        # Export to STEP
        gmsh.write(str(step_file))
        
        print(f"[Geo] Generated pipe: R={radius_mm}mm, L={length_mm}mm (L/D={length_mm/(2*radius_mm):.1f})")
        
    finally:
        gmsh.finalize()
    
    return step_file



def enforce_global_timeout(timeout_sec):
    """Hard kill the process if it exceeds the timeout."""
    def _kill():
        print(f"\n[TIMEOUT] Script exceeded global limit of {timeout_sec}s. Force killing...")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1) # Hard exit to bypass any hung threads/cleanup
        
    import threading
    timer = threading.Timer(timeout_sec, _kill)
    timer.daemon = True
    timer.start()

def run_poiseuille_test():
    # 0. Global Safety Mechanisms
    # ---------------------------
    enforce_global_timeout(180) # 3 minutes hard limit for everything
    
    # Force PyVista to be headless/offscreen
    import pyvista as pv
    pv.set_plot_theme("document")
    pv.OFF_SCREEN = True
    
    print("========================================")
    print("   CFD VALIDATION: POISEUILLE FLOW")
    print("========================================")

    # 1. Setup Environment
    # --------------------
    test_name = f"Poiseuille_Test_{int(time.time())}"
    work_dir = Path(__file__).parent / "temp" / test_name
    output_dir = work_dir / "output"
    
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    # 2. Define Pipe Geometry for Valid Poiseuille Flow
    # --------------------------------------------------
    # Requirements for Poiseuille flow:
    # - L/D > 10 (fully developed flow)
    # - Re < 2000 (laminar regime)
    # - Circular cross-section
    
    radius_mm = 5.0      # 5mm radius = 10mm diameter
    length_mm = 200.0    # 200mm length
    L_D_ratio = length_mm / (2 * radius_mm)
    
    print(f"[Geo] Target L/D ratio: {L_D_ratio:.1f} (need > 10 for valid Poiseuille)")
    
    if L_D_ratio < 10:
        print(f"[WARNING] L/D ratio {L_D_ratio:.1f} is too low for Poiseuille flow validation!")
    
    # Generate pipe geometry
    cad_dest = generate_pipe_geometry(work_dir, radius_mm, length_mm)
    
    # Convert to meters for physics
    length_m = length_mm / 1000.0
    radius_m = radius_mm / 1000.0
    diameter_m = 2 * radius_m

    # 3. Configure Simulation Physics
    # --------------------------------
    # For true laminar: Re = U*D/nu < 2000
    # With D=0.01m (diameter), nu=1e-5: U must be < 2.0 m/s for Re<2000
    # Using U=0.1 m/s gives Re = 0.1 * 0.01 / 1e-5 = 100 (very laminar)
    
    U_inlet = 0.1       # m/s (Gives Re~100)
    nu = 1e-5           # m^2/s (Kinematic viscosity of air at ~20°C)
    rho = 1.0           # kg/m^3 (Simplified for incompressible)
    
    # Calculate Re for verification
    Re_expected = (U_inlet * diameter_m) / nu
    print(f"[Physics] Expected Reynolds Number: {Re_expected:.1f} (must be < 2000 for laminar)")
    
    if Re_expected >= 2000:
        print("[ERROR] Re >= 2000, flow will be turbulent! Reduce U_inlet.")
        return False
    
    # Calculate entrance length requirement
    L_entrance = 0.05 * Re_expected * diameter_m
    print(f"[Physics] Entrance length requirement: {L_entrance*1000:.1f}mm (pipe is {length_mm}mm)")
    
    if L_entrance > length_m * 0.5:
        print(f"[WARNING] Entrance length ({L_entrance*1000:.1f}mm) is > 50% of pipe length!")
        print("          Sampling pressure from interior regions to avoid entrance effects.")
    
    # Config JSON
    config = {
        "job_name": test_name,
        "physics": {
            "simulation_type": "cfd",
            "inlet_velocity": [0, 0, U_inlet],  # Z-direction flow
            "kinematic_viscosity": nu,
            "density": rho,
            "virtual_wind_tunnel": False,
            "mesh_scale_factor": 0.001  # Convert mm to meters
        },
        "L_char": diameter_m,  # Pipe diameter as characteristic length for Re
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
        ],
        "meshing": {
            "mesh_size_multiplier": 0.15  # Finer mesh for better accuracy
        }
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
        # TIMEOUT: 150 seconds max for worker (less than global 180s)
        # We allow a bit of buffer for the global timeout to catch it if subproc hangs
        START_TIME = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=150)
        
        # Save logs for debugging
        with open(work_dir / "worker_stdout.log", 'w') as f: f.write(result.stdout)
        with open(work_dir / "worker_stderr.log", 'w') as f: f.write(result.stderr)
        
        if result.returncode != 0:
            print("[Run] Worker failed! Check logs.")
            # Print explicit error context
            print("STDERR Tail:")
            print(result.stderr[-1000:] if result.stderr else "No Stderr")
            print("STDOUT Tail:")
            print(result.stdout[-1000:] if result.stdout else "No Stdout")
            return False
    
    except subprocess.TimeoutExpired:
        print("[TIMEOUT] Simulation exceeded 150-second worker limit - process killed")
        return False
            
    except Exception as e:
        print(f"[Run] Execution Error: {e}")
        return False
        
    print(f"[Run] Simulation complete (took {time.time() - START_TIME:.1f}s).")

    # 5. Analytical Solution (Hagen-Poiseuille)
    # -----------------------------------------
    # Delta P = (8 * mu * L * Q) / (pi * R^4) 
    # Q = U_avg * Area = U_avg * pi * R^2
    # Delta P = (8 * mu * L * U_avg) / R^2
    # Note: mu = nu * rho (dynamic viscosity)
    
    mu = nu * rho
    dp_analytical = (8 * mu * length_m * U_inlet) / (radius_m**2)
    
    print(f"\n[Validation] Analytical dP (Hagen-Poiseuille): {dp_analytical:.6f} Pa")
    print(f"[Validation] Using: mu={mu:.2e} Pa·s, L={length_m:.4f}m, R={radius_m:.4f}m, U={U_inlet:.4f}m/s")

    # 6. Parse VTK for Simulated Pressure Drop
    # ----------------------------------------
    
    # -------------------------------------------------------------
    # [Fix] Use headless parsing to avoid crashes
    # -------------------------------------------------------------
    
    # Find result VTK
    vtu_files = list(output_dir.rglob("internal.vtu"))
    vtk_root_candidates = []
    case_vtk_dir = next(output_dir.rglob("VTK"), None)
    if case_vtk_dir and case_vtk_dir.is_dir():
        vtk_root_candidates = [f for f in case_vtk_dir.glob("*.vtk") if f.is_file()]
    
    if vtu_files:
        candidates = vtu_files
    elif vtk_root_candidates:
        candidates = vtk_root_candidates
    else:
        candidates = list(output_dir.rglob("*.vtk"))
        
    if not candidates:
        print("[Error] No VTK/VTU files found in output.")
        return False
        
    # Pick the latest modified file
    target_vtk = sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]
    print(f"[Validation] Processing VTK: {target_vtk.name}")
    
    try:
        # Load Mesh - PyVista might crash if display is attempted, but read() should be safe
        mesh = pv.read(target_vtk)
        print(f"[Validation] VTK Bounds: {mesh.bounds}")
        bounds = mesh.bounds
        z_min, z_max = bounds[4], bounds[5]
        
        # Check arrays for pressure field
        p_name = 'p' if 'p' in mesh.point_data else None
        if not p_name and 'p_rgh' in mesh.point_data: 
            p_name = 'p_rgh'
        
        if not p_name:
            print(f"[Error] 'p' or 'p_rgh' not found in VTK. Available: {list(mesh.point_data.keys())}")
            return False
            
        points = mesh.points
        p_data = mesh.point_data[p_name]
        
        # Sample pressure at inlet and outlet regions
        # Use VTK bounds (already scaled by mesh_scale_factor)
        vtk_lz = z_max - z_min
        
        # Sample at 10% from inlet and 10% from outlet to avoid boundary effects
        z_inlet_sample = z_min + 0.10 * vtk_lz  
        z_outlet_sample = z_max - 0.10 * vtk_lz
        
        # Use tolerance for sampling (2% of length)
        tol = 0.02 * vtk_lz
        
        mask_inlet = np.abs(points[:, 2] - z_inlet_sample) < tol
        mask_outlet = np.abs(points[:, 2] - z_outlet_sample) < tol
        
        if np.sum(mask_inlet) == 0 or np.sum(mask_outlet) == 0:
            print("[Error] Could not sample pressure at inlet/outlet planes.")
            print(f"        Z range: [{z_min:.6f}, {z_max:.6f}]")
            print(f"        Sampling at: inlet={z_inlet_sample:.6f}, outlet={z_outlet_sample:.6f}")
            return False
        
        # Average pressure at each plane
        p_inlet = np.mean(p_data[mask_inlet])
        p_outlet = np.mean(p_data[mask_outlet])
        
        # Pressure drop (inlet should be higher than outlet for flow in +Z)
        dp_sim = p_inlet - p_outlet
        
        # OpenFOAM incompressible 'p' is kinematic pressure (P/rho) in m^2/s^2
        # Convert to dynamic pressure (Pa) by multiplying by rho
        dp_sim_pa = dp_sim * rho
        
        # Adjust for the sampling distance (80% of pipe length due to 10% offset on each end)
        sampling_length = 0.80 * length_m
        # Scale analytical to match sampling length
        dp_analytical_sampled = dp_analytical * (sampling_length / length_m)
        
        print(f"\n[Validation] P at 10% from inlet (kinematic): {p_inlet:.6f} m²/s²")
        print(f"[Validation] P at 10% from outlet (kinematic): {p_outlet:.6f} m²/s²")
        print(f"[Validation] Simulated dP over 80% of pipe (dynamic): {dp_sim_pa:.6f} Pa")
        print(f"[Validation] Analytical dP over 80% of pipe: {dp_analytical_sampled:.6f} Pa")
        
        # Explicit clean-up
        del mesh
        pv.close_all()
        
    except Exception as e:
        print(f"[Error] VTK analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 7. Calculate Error and Report
    # ------------------------------
    if abs(dp_analytical_sampled) < 1e-12:
        print("[Error] Analytical dP is effectively zero - check geometry/physics.")
        return False
        
    error = abs(dp_sim_pa - dp_analytical_sampled) / abs(dp_analytical_sampled) * 100.0
    
    print(f"\n{'='*50}")
    print(f"  POISEUILLE FLOW VALIDATION RESULTS")
    print(f"{'='*50}")
    print(f"  Geometry:     R={radius_mm}mm, L={length_mm}mm (L/D={L_D_ratio:.1f})")
    print(f"  Reynolds:     {Re_expected:.1f}")
    print(f"  Analytical:   {dp_analytical_sampled:.6f} Pa")
    print(f"  Simulated:    {dp_sim_pa:.6f} Pa")
    print(f"  Error:        {error:.2f}%")
    print(f"{'='*50}")
    
    if error < 10.0:
        print("\n[PASS] Poiseuille Flow Validation Successful!")
        return True
    else:
        print("\n[FAIL] Error exceeds 10% threshold.")
        return False

if __name__ == "__main__":
    success = run_poiseuille_test()
    sys.exit(0 if success else 1)
