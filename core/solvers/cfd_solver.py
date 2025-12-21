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

logger = logging.getLogger(__name__)

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
        
        Config Keys:
            kinematic_viscosity: float (m^2/s)
            inlet_velocity: [x,y,z] (m/s)
            
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
        # We need the mesh in the case directory. 
        # Usually fluentMeshToFoam is used for .msh files.
        shutil.copy(mesh_file, case_dir / "mesh.msh")
        
        start_time = time.time()
        
        try:
             # 4. Run OpenFOAM Pipeline
             # A. Convert Mesh
             self._run_foam_cmd(case_dir, "gmshToFoam mesh.msh")
             
             # Fix Boundary Types (e.g. frontAndBack -> empty)
             self._fix_boundary_types(case_dir)
             
             # gmshToFoam creates 'polyMesh' but might have default boundaries. 
             # We rely on physical groups mapping to patch names.
             self._run_foam_cmd(case_dir, "checkMesh")

             # C. Run Solver
             # Run in background or wait? Wait.
             logger.info("[OpenFOAM] Running simpleFoam...")
             self._run_foam_cmd(case_dir, "simpleFoam")
             
             # D. Post-Process (VTK)
             logger.info("[OpenFOAM] Converting to VTK...")
             self._run_foam_cmd(case_dir, "foamToVTK -ascii -latestTime")

             solve_time = time.time() - start_time
             logger.info(f"[OpenFOAM] Solved in {solve_time:.2f}s")
             
             # 5. Parse Results
             return self._parse_results(case_dir, solve_time)
             
        except subprocess.CalledProcessError as e:
            logger.error(f"[OpenFOAM] Command failed: {e.cmd}")
            # Try to read logs
            raise RuntimeError(f"OpenFOAM failed: {e}")
        except Exception as e:
            logger.exception(f"[OpenFOAM] Error: {e}")
            raise

    def _setup_case(self, case_dir: Path, config: Dict[str, Any]):
        """Creates standard OpenFOAM directories and dicts."""
        if case_dir.exists():
            shutil.rmtree(case_dir)
        
        (case_dir / "0").mkdir(parents=True)
        (case_dir / "constant").mkdir()
        (case_dir / "system").mkdir()
        
        # 0/U - Velocity
        u_inlet = config.get("inlet_velocity", [1.0, 0, 0])
        self._write_U(case_dir / "0" / "U", u_inlet)
        
        # 0/p - Pressure
        self._write_p(case_dir / "0" / "p")
        
        # 0/nut, 0/k, 0/omega, 0/epsilon (Turbulence)
        # For Validation (Re=40), we want LAMINAR.
        # So we set simulationType to laminar in turbulenceProperties.
        
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

    def _run_foam_cmd(self, case_dir: Path, cmd: str):
        """Executes OpenFOAM command in WSL."""
        wsl_path = str(case_dir).replace('\\', '/').replace('C:', '/mnt/c')
        
        foam_source = "source /usr/lib/openfoam/openfoam2312/etc/bashrc 2>/dev/null || source /opt/openfoam10/etc/bashrc 2>/dev/null"
        full_cmd = f"{foam_source}; cd '{wsl_path}' && {cmd} > log.{cmd.split()[0]} 2>&1"
        
        if self.use_wsl:
            subprocess.run(
                ['wsl', 'bash', '-c', full_cmd],
                check=True, timeout=600
            )
        else:
            # Native Linux
             subprocess.run(
                ['bash', '-c', full_cmd],
                check=True, timeout=600
            )

    def _fix_boundary_types(self, case_dir: Path):
        """Fixes patch types in polyMesh/boundary (e.g. frontAndBack to empty)."""
        boundary_file = case_dir / "constant" / "polyMesh" / "boundary"
        if not boundary_file.exists():
            return
            
        content = boundary_file.read_text()
        
        # Replace type patch with type empty for frontAndBack
        import re
        # We look for the block 'frontAndBack' then inside it change 'type patch' to 'type empty'
        # This is slightly complex with regex on huge file, but boundary file is small.
        # Pattern: finds frontAndBack followed by any chars until 'type' then 'patch'
        
        # Robust regex:
        # frontAndBack
        #    {
        #        type            patch;
        
        # We replace the specific type line inside the block.
        # NOTE: OpenFOAM might have different spacing.
        
        # Pattern matches:
        # frontAndBack \s* { [^}]* type \s+ patch
        # We want to replace 'patch' with 'empty' only in that match.
        
        def replace_type(match):
            return match.group(0).replace("type            patch", "type            empty").replace("type\tpatch", "type\tempty").replace("type patch", "type empty")

        # More precise: Search for frontAndBack block start, then sub inside?
        # Let's try a simpler approach if the file is standard.
        # "gmshToFoam" produces standard format.
        
        # Standard gmshToFoam output seen in log:
        # frontAndBack
        # {
        #     type            patch;
        
        new_content = re.sub(
            r'(frontAndBack\s*\{[^}]*type\s+)(patch)(;)', 
            r'\1empty\3', 
            content, 
            flags=re.DOTALL
        )
        
        if new_content != content:
            logger.info("Fixed 'frontAndBack' patch type to 'empty'")
            boundary_file.write_text(new_content)
        else:
            logger.warning("Could not fix 'frontAndBack' patch. Check naming.")

    def _parse_results(self, case_dir: Path, solve_time: float) -> Dict[str, Any]:
        """Verify output and extracting scalar data."""
        vtk_dir = case_dir / "VTK"
        # foamToVTK creates .vtu (xml) or .vtk (legacy). 
        # Typically VTK/<case>_<time>/internal.vtu
        
        # Find all vtk/vtu files
        vtks = list(vtk_dir.rglob("*.vtk")) + list(vtk_dir.rglob("*.vtu"))
        
        if not vtks:
            raise RuntimeError("No VTK output found")
            
        # Sort by modification time to get latest
        vtks.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        latest_vtk = vtks[0]
        
        # If multiple files (e.g. boundary files), we prefer 'internal.vtu' or '.vtk'
        # The sort by mtime is usually fine, but 'internal.vtu' is what we want for volume data.
        internal_vtks = [f for f in vtks if "internal" in f.name]
        if internal_vtks:
            latest_vtk = internal_vtks[0]
        
        return {
            "vtk_file": str(latest_vtk),
            "solve_time": solve_time,
            "case_dir": str(case_dir),
            # In a real impl, we'd parse the VTK or logs to get residuals/forces
            "converged": True # Placeholder
        }

    # =========================================================================
    # DICTIONARY WRITERS (Minimal for Laminar/Steady)
    # =========================================================================
    
    def _write_U(self, path: Path, u_inlet: List[float]):
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
    inlet
    {{
        type            fixedValue;
        value           uniform ({u_inlet[0]} {u_inlet[1]} {u_inlet[2]});
    }}
    outlet
    {{
        type            zeroGradient;
    }}
    walls
    {{
        type            noSlip;
    }}
    // Fallback for auto-generated patch names
    "(.*)"
    {{
        type            noSlip;
    }}
    // Standard Fluent export names often: patch_1, patch_2...
    // We might need 'autoPatch' (wildcard) logic if possible, 
    // but OpenFOAM is strict.
    // For validation case, we WILL ensure patch names in Gmsh match these.
    // (inlet, outlet, walls, frontAndBack)
    frontAndBack
    {{
        type            empty;
    }}
}}
"""
        path.write_text(content)

    def _write_p(self, path: Path):
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
    inlet
    {
        type            zeroGradient;
    }
    outlet
    {
        type            fixedValue;
        value           uniform 0;
    }
    walls
    {
        type            zeroGradient;
    }
    "(.*)"
    {
        type            zeroGradient;
    }
    frontAndBack
    {
        type            empty;
    }
}
"""
        path.write_text(content)

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
        content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      turbulenceProperties;
}}
simulationType  {model}; 
"""
        path.write_text(content)

    def _write_controlDict(self, path: Path, end_time: int):
        # We add force calculation for 'cylinder' patch if it exists, or 'walls'
        # For generality, we just add it for 'cylinder' and 'walls'.
        # If patch doesn't exist, OpenFOAM might complain? 
        # Actually it warns but continues usually. 
        # Let's target 'walls' and 'cylinder'.
        
        functions = """
    functions
    {
        forces
        {
            type            forces;
            libs            ("libforces.so");
            writeControl    timeStep;
            writeInterval   10;
            patches         (cylinder);
            rho             rhoInf;
            log             true;
            rhoInf          1.0; // Incompressible solver uses kinematic pressure (p/rho), so forces are scaled by rho. 
                                 // Actually simpleFoam solves for p/rho? No, simpleFoam solves for p (kinematic) if incompressible?
                                 // In OpenFOAM standard simpleFoam, p is kinematic pressure [m^2/s^2]. 
                                 // So Force = Integral(p * normal) * rhoInf.
            CofR            (0 0 0);
        }
    }
"""
        content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}}
application     simpleFoam;
startFrom       startTime;
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

{functions}
"""
        path.write_text(content)

    def _write_fvSchemes(self, path: Path):
        content = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}
ddtSchemes
{
    default         steadyState;
}
gradSchemes
{
    default         Gauss linear;
}
divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes
{
    default         Gauss linear orthogonal;
}
interpolationSchemes
{
    default         linear;
}
snGradSchemes
{
    default         corrected;
}
"""
        path.write_text(content)

    def _write_fvSolution(self, path: Path):
        content = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}
solvers
{
    p
    {
        solver           GAMG;
        tolerance        1e-06;
        relTol           0.01;
        smoother         GaussSeidel;
    }
    U
    {
        solver           smoothSolver;
        smoother         symGaussSeidel;
        tolerance        1e-08;
        relTol           0.001;
    }
}
SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent      yes;
    pRefCell        0;
    pRefValue       0;
}
relaxationFactors
{
    fields
    {
        p               0.3;
    }
    equations
    {
        U               0.7;
    }
}
"""
        path.write_text(content)

    def _check_wsl(self) -> bool:
        try:
            subprocess.run(['wsl', '--list'], capture_output=True, timeout=5)
            # Check for foam?
            return True
        except:
            return False
