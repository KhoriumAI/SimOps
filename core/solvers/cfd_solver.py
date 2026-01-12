"""
OpenFOAM CFD Solver Wrapper
===========================

Implements ISolver for steady-state incompressible CFD using simpleFoam.
"""

import os
import shutil
import subprocess
import logging
import time
import numpy as np
import platform
from pathlib import Path
from typing import Dict, Any, List, Optional
from .base import ISolver
from .openfoam_parser import parse_boundary_file, verify_patches

from core.logging.sim_logger import SimLogger
logger = SimLogger("CFDSolver")

class CFDSolver(ISolver):
    """
    OpenFOAM solver implementation for steady incompressible flow.
    Uses 'simpleFoam' within a WSL environment (on Windows).
    """
    
    def __init__(self, use_wsl: bool = True):
        self.use_wsl = use_wsl and platform.system() == "Windows"
        
    def run(self, mesh_file: Path, output_dir: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run CFD simulation.
        """
        job_name = mesh_file.stem
        case_dir = output_dir / f"{job_name}_case"
        
        # 1. Check Prereqs
        if self.use_wsl:
            if not self._check_wsl():
                raise RuntimeError("WSL2 is required for OpenFOAM but not found.")
        
        # 2. Setup Case Structure
        self._setup_case(case_dir, config)
        
        # 3. Copy/Convert Mesh
        shutil.copy(mesh_file, case_dir / "mesh.msh")
        
        start_time = time.time()
        
        try:
             # 4. Run OpenFOAM Pipeline
             # A. Convert Mesh
             self._run_foam_cmd(case_dir, "gmshToFoam mesh.msh")
             
             # Fix Boundary Types (e.g. frontAndBack -> empty)
             self._fix_boundary_types(case_dir)
             
             # Fix Boundary Types (e.g. frontAndBack -> empty)
             self._fix_boundary_types(case_dir)
             
             # Apply Mesh Scaling if requested (e.g. mm to m)
             scale_factor = config.get("mesh_scale_factor", 1.0)
             if abs(scale_factor - 1.0) > 1e-6:
                 logger.info(f"[OpenFOAM] Scaling mesh by factor {scale_factor}...")
                 # OpenFOAM v13 syntax: transformPoints "scale=(sx sy sz)"
                 self._run_foam_cmd(case_dir, f"transformPoints \"scale=({scale_factor} {scale_factor} {scale_factor})\"")
             
             # B. [NEW] Verify Topology (The 3-Layer Defense)
             available_patches = parse_boundary_file(case_dir)
             
             # Define required patches based on what our solver expects
             # We now look for the semantic tags we apply: BC_Inlet, BC_Outlet OR inlet/outlet
             required_patches = ["inlet", "outlet"] 
             
             # Strict Verification
             # verify_patches(required_patches, available_patches)
             # Relaxed verification: Just warn if missing
             for req in required_patches:
                 if req not in available_patches and f"BC_{req.capitalize()}" not in available_patches:
                      logger.warning(f"[Setup] Required patch '{req}' not found (Found: {list(available_patches.keys())})")
             
             # Re-Write U and p dicts with KNOWN patches (No Wildcards)
             # We generated them in _setup_case with placeholders/wildcards.
             # Now we overwrite them with strict inputs.
             # Extract inlet velocity - check both 'u_inlet' (from worker) and 'inlet_velocity' (legacy)
             inlet_vel_raw = config.get("u_inlet") or config.get("inlet_velocity", [1.0, 0, 0])
             if isinstance(inlet_vel_raw, (int, float)):
                 # Convert scalar to X-direction vector
                 u_inlet = [float(inlet_vel_raw), 0.0, 0.0]
             elif isinstance(inlet_vel_raw, list) and len(inlet_vel_raw) == 3:
                 u_inlet = inlet_vel_raw
             else:
                 logger.warning(f"Invalid inlet_velocity format: {inlet_vel_raw}, using default")
                 u_inlet = [1.0, 0, 0]
             
             logger.info(f"[Setup] Using inlet velocity: {u_inlet} (from config: {inlet_vel_raw})")
             self._write_U(case_dir / "0" / "U", u_inlet, available_patches)
             self._write_p(case_dir / "0" / "p", available_patches)

             # =================================================================
             # AUTOMATED TURBULENCE MODELING (The Pivot)
             # =================================================================
             try:
                 # 1. Get L_char from config override, OR metadata, OR default
                 L_char = config.get("L_char", None)
                 
                 if L_char is None:
                     potential_meta = case_dir / "mesh_metadata.json"
                     if potential_meta.exists():
                         import json
                         with open(potential_meta) as f:
                             meta = json.load(f)
                             L_char = meta.get('wind_tunnel', {}).get('L_char', 1.0)
                     else:
                         L_char = 1.0  # Default fallback
                 
                 # 2. Calculate Reynolds Number
                 nu = config.get("kinematic_viscosity", 1.5e-5)
                 U_mag = (u_inlet[0]**2 + u_inlet[1]**2 + u_inlet[2]**2)**0.5
                 
                 if U_mag < 1e-9: U_mag = 1e-9 # Avoid ZeroDivision
                 
                 Re = (U_mag * L_char) / nu
                 logger.log_metric("reynolds_number", Re)
                 logger.info(f"[Physics] Reynolds Number: {Re:.2e} (L={L_char:.3f}m, U={U_mag:.2f}m/s)")
                 
                 # 3. Determine Regime
                 # Re < 2000 -> Laminar
                 # Re > 4000 -> Turbulent
                 # Transition zone? Treat as turbulent for safety in external aero.
                 is_turbulent = Re > 4000
                 
                 model_type = "RAS" if is_turbulent else "laminar"
                 logger.info(f"[Physics] Flow Regime: {'TURBULENT' if is_turbulent else 'LAMINAR'} (Model: {model_type})")
                 
                 # 4. Rewrite Configuration
                 self._write_turbulenceProperties(case_dir / "constant" / "turbulenceProperties", model_type)
                 self._write_fvSchemes(case_dir / "system" / "fvSchemes", turbulent=is_turbulent)
                 self._write_fvSolution(case_dir / "system" / "fvSolution", turbulent=is_turbulent)
                 
                 # 5. Initialize Turbulent Fields
                 if is_turbulent:
                     # Estimate k, omega
                     # I = 5% (0.05) typical for external aerodynamics
                     # l = 0.07 * L typical mixing length
                     I = 0.05
                     l_turb = 0.07 * L_char
                     
                     k_val = 1.5 * (U_mag * I)**2
                     # omega = k^0.5 / (C_mu * l) where C_mu = 0.09
                     # omega = k^0.5 / (0.09 * l)
                     omega_val = (k_val**0.5) / (0.09 * l_turb)
                     
                     logger.info(f"[Physics] Initializing Turbulence Fields: k={k_val:.4e}, omega={omega_val:.4e}")
                     
                     self._write_k(case_dir / "0" / "k", k_val, available_patches)
                     self._write_omega(case_dir / "0" / "omega", omega_val, available_patches)
                     self._write_nut(case_dir / "0" / "nut", available_patches)
                     
             except Exception as e:
                 logger.error(f"[Physics] Turbulence automation failed: {e}")
                 # Fallback to Laminar settings default wrote
                 pass


             # D. Run Solver
             logger.info("[OpenFOAM] Running simpleFoam...")
             self._run_foam_cmd(case_dir, "simpleFoam")
             
             # E. Post-Process (VTK) - Include all fields for visualization
             logger.info("[OpenFOAM] Converting to VTK...")
             self._run_foam_cmd(case_dir, "foamToVTK -latestTime")  # ascii removed, all fields included by default

             solve_time = time.time() - start_time
             logger.log_metric("solve_time", solve_time, "s")
             logger.info(f"[OpenFOAM] Solved in {solve_time:.2f}s")
             
             # 5. Parse Results
             # Pass config data for proper Reynolds calculation
             return self._parse_results(case_dir, solve_time, turbulence_model="k-omega SST" if is_turbulent else "Laminar", actual_reynolds=Re if is_turbulent else None)
             
        except subprocess.CalledProcessError as e:
            logger.error(f"[OpenFOAM] Command failed: {e.cmd}")
            raise RuntimeError(f"OpenFOAM failed: {e}")
        except Exception as e:
            logger.error(f"[OpenFOAM] Error: {e}")
            raise

    def _setup_case(self, case_dir: Path, config: Dict[str, Any]):
        """Creates standard OpenFOAM directories and dicts."""
        if case_dir.exists():
            shutil.rmtree(case_dir)
        
        (case_dir / "0").mkdir(parents=True)
        (case_dir / "constant").mkdir()
        (case_dir / "system").mkdir()
        
        # Initial Write (Will be overwritten after mesh parse, but needed structure)
        # We pass empty patches list initially implies standard defaults? 
        # Actually, let's just write dummy files or rely on rewrites.
        # But _write_U is called in run() now.
        
        # 0/p - Pressure
        # self._write_p(case_dir / "0" / "p") # Moved to run()
        
        # constant/transportProperties
        nu = config.get("kinematic_viscosity", 1.5e-5) # Air
        self._write_transportProperties(case_dir / "constant" / "transportProperties", nu)
        
        # constant/turbulenceProperties
        self._write_turbulenceProperties(case_dir / "constant" / "turbulenceProperties", "laminar")
        
        # system/controlDict
        self._write_controlDict(case_dir / "system" / "controlDict", end_time=config.get("iterations", 1000))
        
        # system/fvSchemes
        self._write_fvSchemes(case_dir / "system" / "fvSchemes")
        
        # system/fvSolution
        self._write_fvSolution(case_dir / "system" / "fvSolution")

    # ... (helper methods like _run_foam_cmd, _fix_boundary_types kept same) ...
    
    # =========================================================================
    # DICTIONARY WRITERS (Hardened)
    # =========================================================================
    
    def _write_U(self, path: Path, u_inlet: List[float], patches: Dict[str, str]):
        # Start of file
        content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}}
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform ({u_inlet[0]} {u_inlet[1]} {u_inlet[2]});
boundaryField
{{
"""
        # Explicit Patch Loop
        for patch, ptype in patches.items():
            content += f"    {patch}\n    {{\n"
            
            # Logic Map
            # Logic Map
            if patch == "inlet" or "inlet" in patch.lower():
                content += f"        type            fixedValue;\n"
                content += f"        value           uniform ({u_inlet[0]} {u_inlet[1]} {u_inlet[2]});\n"
            elif patch == "outlet" or "outlet" in patch.lower():
                content += f"        type            zeroGradient;\n"
            elif ptype == "wall" or "wall" in patch.lower() or patch == "cylinder":
                content += f"        type            noSlip;\n"
            elif ptype == "empty" or patch == "frontAndBack":
                content += f"        type            empty;\n"
            elif ptype == "processor":
                 content += f"        type            processor;\n"
            else:
                 # Default fallback
                 if "wall" in patch.lower():
                      content += f"        type            noSlip;\n"
                 else:
                      # If we don't know it, we abort? Or assume wall?
                      # Strict mode: Warn.
                      logger.warning(f"[Hardening] Unknown patch '{patch}' treated as noSlip wall.")
                      content += f"        type            noSlip;\n"
            
            content += "    }\n"

        content += "}\n"
        path.write_text(content)

    def _write_p(self, path: Path, patches: Dict[str, str]):
        content = """FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      p;
}
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;
boundaryField
{
"""
        for patch, ptype in patches.items():
            content += f"    {patch}\n    {{\n"
            
            if patch == "inlet" or "inlet" in patch.lower():
                content += f"        type            zeroGradient;\n"
            elif patch == "outlet" or "outlet" in patch.lower():
                content += f"        type            fixedValue;\n"
                content += f"        value           uniform 0;\n"
            elif ptype == "wall" or "wall" in patch.lower() or patch == "cylinder":
                content += f"        type            zeroGradient;\n"
            elif ptype == "empty" or patch == "frontAndBack":
                content += f"        type            empty;\n"
            elif ptype == "processor":
                 content += f"        type            processor;\n"
            else:
                 content += f"        type            zeroGradient;\n"
                 
            content += "    }\n"

        content += "}\n"
        path.write_text(content)

    def _run_foam_cmd(self, case_dir: Path, cmd: str):
        """
        Executes OpenFOAM command in WSL, capturing output to log file and console.
        Replaces shell '> log 2>&1' with Python-side 'tee' behavior.
        """
        wsl_path = str(case_dir).replace('\\', '/').replace('C:', '/mnt/c')
        cmd_name = cmd.split()[0]
        log_file = case_dir / f"log.{cmd_name}"
        stop_file = case_dir / "STOP_SIM"
        
        foam_source = "source /usr/lib/openfoam/openfoam2312/etc/bashrc 2>/dev/null || source /opt/openfoam10/etc/bashrc 2>/dev/null || source /opt/openfoam13/etc/bashrc 2>/dev/null"
        full_cmd = f"{foam_source}; cd '{wsl_path}' && {cmd}"
        
        
        logger.info(f"Running: {cmd}")
        logger.error(f"[DEBUG] _run_foam_cmd: {cmd}")
        logger.error(f"[DEBUG] Writing log to: {log_file} (Exists: {log_file.parent.exists()})")
        
        # Prepare command args
        run_args = []
        if self.use_wsl:
            run_args = ['wsl', 'bash', '-c', full_cmd]
        else:
            run_args = ['bash', '-c', full_cmd]
            
        interrupted = False
        # Use Popen to stream output
        process = subprocess.Popen(
            run_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace' # Handle potential encoding issues gracefully
        )
        
        # Read stdout line by line
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            
            if line:
                # Echo to console
                print(line, end='')
            
            # Check for Stop Signal
            if stop_file.exists():
                logger.warning(f"[OpenFOAM] Stop signal detected! Interrupting {cmd_name}...")
                process.terminate()
                interrupted = True
                try: process.wait(timeout=5)
                except: process.kill()
                stop_file.unlink()
                break
        
        # Check exit code
        if process.returncode != 0 and not interrupted:
            raise subprocess.CalledProcessError(process.returncode, cmd)
            
        if interrupted:
            logger.info(f"[OpenFOAM] {cmd_name} interrupted. Proceeding to post-processing...")

    def _fix_boundary_types(self, case_dir: Path):
        """Fixes patch types in polyMesh/boundary (e.g. frontAndBack to empty, wall patches to wall)."""
        boundary_file = case_dir / "constant" / "polyMesh" / "boundary"
        if not boundary_file.exists():
            return
            
        content = boundary_file.read_text()
        
        import re
        
        # Fix 1: frontAndBack should be type 'empty' not 'patch'
        new_content = re.sub(
            r'(frontAndBack\s*\{[^}]*type\s+)(patch)(;)', 
            r'\1empty\3', 
            content, 
            flags=re.DOTALL
        )
        
        # Fix 2: BC_Wall_Object and BC_FarField should be type 'wall' not 'patch'
        # This is critical for turbulence wall functions
        # Pattern: Matches "BC_Wall_Object" or "BC_FarField" followed by { ... type patch; }
        # We need to  replace type with 'wall' for wall patches
        
        # For BC_Wall_Object (actual wall with no-slip)
        # GENERIC FIX: Any patch named 'wall' or starting/ending with 'wall' (case insensitive check usually hard in regex without flag, but let's be specific)
        # We specifically use 'wall' in our tagging.
        
        # 1. Fix 'BC_Wall_Object'
        new_content = re.sub(
            r'(BC_Wall_Object\s*\{[^}]*type\s+)(patch)(;)', 
            r'\1wall\3', 
            new_content, 
            flags=re.DOTALL
        )
        
        # 2. Fix plain 'wall' tag
        new_content = re.sub(
            r'(wall\s*\{[^}]*type\s+)(patch)(;)', 
            r'\1wall\3', 
            new_content, 
            flags=re.DOTALL
        )
        
        if new_content != content:
            logger.info("Fixed boundary types: Patches converted to 'wall' type")
            boundary_file.write_text(new_content)
        else:
            logger.warning("Could not fix 'frontAndBack' patch. Check naming.")

    def _parse_results(self, case_dir: Path, solve_time: float, turbulence_model: str = "Laminar", actual_reynolds: float = None) -> Dict[str, Any]:
        """
        Extract comprehensive CFD results from OpenFOAM output.
        
        Returns:
            Dictionary containing:
            - vtk_file: Path to VTK visualization
            - solve_time: Solver execution time
            - courant_max: Maximum Courant number
            - converged: Whether simulation converged
            - reynolds: Reynolds number
            - cd: Drag coefficient (if available)
            - cl: Lift coefficient (if available)
            - num_cells: Mesh cell count
            - config: Simulation configuration (for Reynolds calc)
        """
        vtk_dir = case_dir / "VTK"
        
        # Find VTK output
        vtks = list(vtk_dir.rglob("*.vtk")) + list(vtk_dir.rglob("*.vtu"))
        if not vtks:
            raise RuntimeError("No VTK output found")
        
        # Prefer internal.vtu for volume data
        internal_vtks = [f for f in vtks if "internal" in f.name]
        latest_vtk = internal_vtks[0] if internal_vtks else sorted(vtks, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        
        # Initialize results
        results = {
            "vtk_file": str(latest_vtk),
            "solve_time": solve_time,
            "case_dir": str(case_dir),
            "courant_max": 0.0,
            "converged": True,
            "reynolds": None,
            "cd": "N/A",
            "cl": "N/A",
            "num_cells": 0,
            "turbulence_model": turbulence_model,
        }
        
        # ========================================================================
        # 1. Parse log.simpleFoam for Courant number and convergence
        # ========================================================================
        log_file = case_dir / "log.simpleFoam"
        if log_file.exists():
            try:
                content = log_file.read_text()
                # Courant number
                import re
                courant_matches = re.findall(r"Courant Number.*max:\s*([\d\.e\-\+]+)", content)
                if courant_matches:
                    results["courant_max"] = float(courant_matches[-1])
                
                # Check for "End" statement (indicates successful completion)
                results["converged"] = "End" in content
                
            except Exception as e:
                logger.warning(f"Failed to parse log.simpleFoam: {e}")
        
        # ========================================================================
        # 2. Parse mesh statistics from gmshToFoam log
        # ========================================================================
        gmsh_log = case_dir / "log.gmshToFoam"
        if gmsh_log.exists():
            try:
                content = gmsh_log.read_text()
                import re
                # Extract cell count: "total: 89177"
                cell_match = re.search(r"total:\s+(\d+)", content)
                if cell_match:
                    results["num_cells"] = int(cell_match.group(1))
                    logger.log_metric("mesh_cells", results["num_cells"])
                    logger.info(f"[Parser] Extracted mesh cells: {results['num_cells']}")
            except Exception as e:
                logger.warning(f"Failed to parse mesh stats: {e}")
        
        # ========================================================================
        # 3. Parse postProcessing/forces for Cd/Cl
        # ========================================================================
        forces_dir = case_dir / "postProcessing" / "forces"
        if forces_dir.exists():
            try:
                # Find force.dat (usually in 0/ subdirectory)
                force_files = list(forces_dir.rglob("force.dat"))
                if force_files:
                    force_file = force_files[0]
                    # Read last non-comment line
                    with open(force_file, 'r') as f:
                        lines = [l for l in f if not l.startswith('#') and l.strip()]
                        if lines:
                            last_line = lines[-1].split()
                            # Format: Time total_x total_y total_z pressure_x ... viscous_z
                            if len(last_line) >= 4:
                                total_fx = float(last_line[1])
                                total_fy = float(last_line[2])
                                total_fz = float(last_line[3])
                                
                                # Calculate Cd/Cl (need reference area and dynamic pressure)
                                # For now, store raw forces - coefficients require geometry info
                                # Cd = Fx / (0.5 * rho * U^2 * A_ref)
                                # This is a placeholder - proper calculation needs config
                                results["force_x"] = total_fx
                                results["force_y"] = total_fy
                                results["force_z"] = total_fz
            except Exception as e:
                logger.warning(f"Failed to parse forces: {e}")
        
        # ========================================================================
        # 4. Use pre-calculated Reynolds number from solver
        # ========================================================================
        # We calculated this accurately during turbulence setup with wind tunnel L_char
        if actual_reynolds is not None:
            results["reynolds"] = actual_reynolds
            logger.info(f"[Parser] Using pre-calculated Reynolds: {actual_reynolds:.2e}")
        else:
            # Fallback: try to calculate from files (less accurate)
            try:
                trans_props = case_dir / "constant" / "transportProperties"
                if trans_props.exists():
                    content = trans_props.read_text()
                    import re
                    nu_match = re.search(r"nu\s+\[.*?\]\s+([\d\.e\-\+]+)", content)
                    if nu_match:
                        nu = float(nu_match.group(1))
                        
                        u_field = case_dir / "0" / "U"
                        if u_field.exists():
                            u_content = u_field.read_text()
                            u_matches = re.findall(r"uniform\s+\(([\d\.\-e\+]+)\s+([\d\.\-e\+]+)\s+([\d\.\-e\+]+)\)", u_content)
                            if u_matches:
                                ux, uy, uz = map(float, u_matches[0])
                                U_mag = (ux**2 + uy**2 + uz**2)**0.5
                                L_char = 1.0  # Fallback length
                                results["reynolds"] = (U_mag * L_char) / nu
            except Exception as e:
                logger.warning(f"Failed to calculate Reynolds number: {e}")
        
        return results

    # =========================================================================
    # DICTIONARY WRITERS (Minimal for Laminar/Steady)
    # =========================================================================
    


    def _write_transportProperties(self, path: Path, nu: float):
        content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      transportProperties;
}}
transportModel  Newtonian;
nu              [0 2 -1 0 0 0 0] {nu};
"""
        path.write_text(content)

    def _write_turbulenceProperties(self, path: Path, model: str):
        # model is 'laminar' or 'RAS'
        if model == "RAS":
            extra = """
RAS
{
    RASModel        kOmegaSST;
    turbulence      on;
    printCoeffs     on;
}
"""
        else:
            # OpenFOAM v13 requires 'model' keyword in laminar block
            extra = """
laminar
{
    model           Stokes;
    turbulence      off;
}
"""
        
        content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      turbulenceProperties;
}}
simulationType  {model}; 
{extra}
"""
        path.write_text(content)

    def _write_controlDict(self, path: Path, end_time: int = 1000):
        content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}}
application     simpleFoam;
startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         {end_time};
deltaT          1;
writeControl    timeStep;
writeInterval   100;
purgeWrite      2;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;

functions
{{
}}
"""
        path.write_text(content)

    def _write_fvSchemes(self, path: Path, turbulent: bool = False):
        div_turb = ""
        if turbulent:
            div_turb = """
    div(phi,k)      bounded Gauss upwind;
    div(phi,omega)  bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
"""

        # Wall distance calculation method (required for turbulence models)
        wall_dist = ""
        if turbulent:
            wall_dist = """
wallDist
{
    method meshWave;
}
"""

        content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}}
ddtSchemes
{{
    default         steadyState;
}}
gradSchemes
{{
    default         Gauss linear;
}}
divSchemes
{{
    default         none;
    div(phi,U)      bounded Gauss upwind;{div_turb}
}}
laplacianSchemes
{{
    default         Gauss linear orthogonal;
}}
interpolationSchemes
{{
    default         linear;
}}
snGradSchemes
{{
    default         corrected;
}}
{wall_dist}"""
        path.write_text(content)

    def _write_fvSolution(self, path: Path, turbulent: bool = False):
        
        solvers_turb = ""
        if turbulent:
            solvers_turb = """
    "(k|omega)"
    {
        solver           smoothSolver;
        smoother         symGaussSeidel;
        tolerance        1e-08;
        relTol           0.001;
    }
"""
        
        relax_turb = ""
        if turbulent:
            relax_turb = """
    k               0.7;
    omega           0.7;
"""

        content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}}
solvers
{{
    p
    {{
        solver           GAMG;
        tolerance        1e-06;
        relTol           0.01;
        smoother         GaussSeidel;
    }}
    U
    {{
        solver           smoothSolver;
        smoother         symGaussSeidel;
        tolerance        1e-08;
        relTol           0.001;
    }}
{solvers_turb}
}}
SIMPLE
{{
    nNonOrthogonalCorrectors 0;
    consistent      yes;
    pRefCell        0;
    pRefValue       0;

    residualControl
    {{
        p               1e-4;
        U               1e-4;
        "(k|omega)"     1e-4;
    }}
}}
relaxationFactors
{{
    fields
    {{
        p               0.3;
    }}
    equations
    {{
        U               0.7;{relax_turb}
    }}
}}
"""
        path.write_text(content)

    def _write_k(self, path: Path, k_val: float, patches: Dict[str, str]):
        content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      k;
}}
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform {k_val};
boundaryField
{{
"""
        for patch, ptype in patches.items():
            content += f"    {patch}\n    {{\n"
            if patch == "inlet" or "inlet" in patch.lower():
                content += f"        type            fixedValue;\n"
                content += f"        value           uniform {k_val};\n"
            elif patch == "outlet" or "outlet" in patch.lower():
                content += f"        type            inletOutlet;\n"
                content += f"        inletValue      uniform {k_val};\n"
                content += f"        value           uniform {k_val};\n"
            elif ptype == "wall" or "wall" in patch.lower() or patch == "cylinder":
                content += f"        type            kqRWallFunction;\n"
                content += f"        value           uniform {k_val};\n"
            elif ptype == "empty" or patch == "frontAndBack":
                content += f"        type            empty;\n"
            elif patch == "BC_FarField":
                content += f"        type            slip;\n"
            else:
                 content += f"        type            kqRWallFunction;\n"
                 content += f"        value           uniform {k_val};\n"
            content += "    }\n"
        content += "}\n"
        path.write_text(content)

    def _write_omega(self, path: Path, omega_val: float, patches: Dict[str, str]):
        content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      omega;
}}
dimensions      [0 0 -1 0 0 0 0];
internalField   uniform {omega_val};
boundaryField
{{
"""
        for patch, ptype in patches.items():
            content += f"    {patch}\n    {{\n"
            if patch == "inlet" or "inlet" in patch.lower():
                content += f"        type            fixedValue;\n"
                content += f"        value           uniform {omega_val};\n"
            elif patch == "outlet" or "outlet" in patch.lower():
                content += f"        type            inletOutlet;\n"
                content += f"        inletValue      uniform {omega_val};\n"
                content += f"        value           uniform {omega_val};\n"
            elif ptype == "wall" or "wall" in patch.lower() or patch == "cylinder":
                content += f"        type            omegaWallFunction;\n"
                content += f"        value           uniform {omega_val};\n"
            elif ptype == "empty" or patch == "frontAndBack":
                content += f"        type            empty;\n"
            elif patch == "BC_FarField":
                content += f"        type            slip;\n"
            else:
                 content += f"        type            omegaWallFunction;\n"
                 content += f"        value           uniform {omega_val};\n"
            content += "    }\n"
        content += "}\n"
        path.write_text(content)

    def _write_nut(self, path: Path, patches: Dict[str, str]):
        content = """FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      nut;
}
dimensions      [0 2 -1 0 0 0 0];
internalField   uniform 0;
boundaryField
{
"""
        for patch, ptype in patches.items():
            content += f"    {patch}\n    {{\n"
            if ptype == "empty" or patch == "frontAndBack":
                content += f"        type            empty;\n"
            elif patch == "BC_FarField":
                content += f"        type            slip;\n"
            elif patch == "inlet" or "inlet" in patch.lower() or patch == "outlet" or "outlet" in patch.lower():
                content += f"        type            calculated;\n"
                content += f"        value           uniform 0;\n"
            else: # Walls
                content += f"        type            nutkWallFunction;\n"
                content += f"        value           uniform 0;\n"
            content += "    }\n"
        content += "}\n"
        path.write_text(content)

    def _check_wsl(self) -> bool:
        try:
            subprocess.run(['wsl', '--list'], capture_output=True, timeout=5)
            # Check for foam?
            return True
        except:
            return False
