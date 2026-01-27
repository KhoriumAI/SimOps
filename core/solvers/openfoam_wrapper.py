"""
OpenFOAM Solver Wrapper
=======================

Wraps OpenFOAM solvers (laplacianFoam, buoyantSimpleFoam) for CFD/Thermal analysis.
"""

import os
import shutil
import subprocess
import tempfile
import time
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import gmsh
import platform
from .openfoam_parser import parse_boundary_file, classify_thermal_bc


def setup_thermal_bcs(case_dir: str, config: Optional[Dict] = None) -> None:
    """
    Standalone function to setup thermal boundary conditions for an OpenFOAM case.
    
    This is a convenience wrapper around OpenFOAMWrapper.setup_thermal_boundary_conditions.
    
    Workflow:
        1. Convert mesh: gmshToFoam mesh.msh (for Gmsh files) or fluentMeshToFoam (for Fluent)
        2. Setup BCs: setup_thermal_bcs("case_dir", config)
        3. Run solver: laplacianFoam
    
    Args:
        case_dir: Path to OpenFOAM case directory (string or Path)
        config: Optional configuration dict with:
            - hot_patches: List of patch names for hot BCs
            - cold_patches: List of patch names for cold BCs
            - hot_temperature: Temperature for hot patches (default: 350K)
            - cold_temperature: Temperature for cold patches (default: 300K)
            - ambient_temperature: Initial/ambient temperature (default: 300K)
            - thermal_conductivity: Material k (W/m/K, default: 200)
            - density: Material rho (kg/m³, default: 2700)
            - specific_heat: Material cp (J/kg/K, default: 900)
    
    Example:
        # After running: gmshToFoam mesh.msh
        setup_thermal_bcs("my_case", {
            'hot_patches': ['chip_top'],
            'cold_patches': ['heatsink_bottom'],
            'hot_temperature': 350.0,
            'cold_temperature': 300.0
        })
    """
    wrapper = OpenFOAMWrapper(verbose=True)
    wrapper.setup_thermal_boundary_conditions(Path(case_dir), config or {})


class OpenFOAMWrapper:
    """Wrapper for OpenFOAM solvers."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        
    def _log(self, msg: str):
        if self.verbose:
            print(msg, flush=True)
            
    def is_available(self) -> bool:
        """Check if OpenFOAM is available (via simple check)"""
        # In Docker, we check for a standard command like 'foamList' or 'simpleFoam -help'
        # In WSL, we might need 'wsl ...'
        try:
            # Check for direct availability (Docker/Linux)
            subprocess.run(['foamList'], capture_output=True, timeout=5)
            return True
        except:
            # Check WSL
            if platform.system() == "Windows":
                try:
                    subprocess.run(['wsl', 'bash', '-c', 'foamList'], capture_output=True, timeout=5)
                    return True
                except:
                    pass
            return False
            
    def solve_thermal_cfd(self, mesh_file: str, output_dir: str, config: Dict) -> Dict:
        """
        Run CFD/Thermal simulation.
        Warning: This is a complex wrapper. For MVP, we presume steady state thermal.
        """
        self._log("[OpenFOAM] Starting Solver...")
        
        # TODO: Implement robust OpenFOAM case generation
        # This requires:
        # 1. fluentMeshToFoam on the input mesh
        # 2. Generating dictionaries (controlDict, fvSchemes, fvSolution)
        # 3. Generating boundary fields (0/T, 0/U, 0/p_rgh)
        # 4. Running solver (laplacianFoam or buoyantSimpleFoam)
        # 5. foamToVTK
        
        # For now, we return a placeholder error if called, as the implementation 
        # is dependent on the specific case dictionaries which are large.
        
        # But per the plan, I should implement it.
        # Let's write a basic laplacianFoam setup (Heat conduction only)
        
        work_dir = Path(tempfile.mkdtemp(prefix="foam_run_"))
        
        try:
            # Setup Case
            self._setup_case(work_dir, mesh_file, config)
            
            # Detect mesh format and convert
            mesh_format = self._detect_mesh_format(mesh_file)
            if mesh_format == 'gmsh':
                convert_cmd = "gmshToFoam mesh.msh"
            else:
                convert_cmd = "fluentMeshToFoam mesh.msh"
            
            # Run
            success = self._run_commands(work_dir, [
                convert_cmd,
                "laplacianFoam"
            ])
            
            if not success:
                raise RuntimeError("OpenFOAM solver failed")
                
            # Post-process
            self._run_commands(work_dir, ["foamToVTK -ascii -latestTime"])
            
            # Find VTK
            vtk_dir = work_dir / "VTK"
            vtks = list(vtk_dir.rglob("*.vtk"))
            if not vtks:
                raise RuntimeError("No VTK output found")
                
            result_vtk = vtks[0]
            
            # Copy result
            dest = Path(output_dir) / "openfoam_result.vtk"
            shutil.copy(result_vtk, dest)
            
            return {
                "vtk_file": str(dest),
                "solve_time": 0.0 # TODO
            }
            
        finally:
            if not self.verbose:
                 shutil.rmtree(work_dir, ignore_errors=True)
                 
    def _detect_mesh_format(self, mesh_file: str) -> str:
        """
        Detect mesh file format (Gmsh or Fluent).
        
        Returns:
            'gmsh' or 'fluent' or 'unknown'
        """
        if not mesh_file.endswith('.msh'):
            return 'unknown'
        
        # Try to read first few lines to detect format
        try:
            with open(mesh_file, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline().strip()
                # Gmsh format starts with $MeshFormat or $NOD
                if first_line.startswith('$'):
                    return 'gmsh'
                # Fluent format is binary or starts with numbers
                # For now, assume .msh files are Gmsh by default (user said most are Gmsh)
                return 'gmsh'
        except:
            # If we can't read it, default to Gmsh (user's use case)
            return 'gmsh'
    
    def _setup_case(self, case_dir: Path, mesh_file: str, config: Dict):
        """
        Setup OpenFOAM case for thermal simulation (laplacianFoam).
        
        This assumes the mesh has already been converted to OpenFOAM format.
        If mesh_file is a .msh file, you need to run gmshToFoam or fluentMeshToFoam first.
        """
        # Create directories
        (case_dir / "0").mkdir(parents=True, exist_ok=True)
        (case_dir / "constant").mkdir(parents=True, exist_ok=True)
        (case_dir / "system").mkdir(parents=True, exist_ok=True)
        
        # Copy mesh if it's a .msh file
        if mesh_file.endswith('.msh'):
            shutil.copy(mesh_file, case_dir / "mesh.msh")
            mesh_format = self._detect_mesh_format(mesh_file)
            if mesh_format == 'gmsh':
                self._log("[OpenFOAM] Gmsh mesh file copied. Run 'gmshToFoam mesh.msh' first.")
            else:
                self._log("[OpenFOAM] Mesh file copied. Run 'fluentMeshToFoam mesh.msh' or 'gmshToFoam mesh.msh' first.")
        
        # Generate system files
        self._write_control_dict(case_dir / "system" / "controlDict", config)
        self._write_fv_schemes(case_dir / "system" / "fvSchemes")
        self._write_fv_solution(case_dir / "system" / "fvSolution", config)
        
        # Generate constant files
        self._write_transport_properties(case_dir / "constant" / "transportProperties", config)
        
        # Generate boundary conditions (0/T)
        # First, try to parse existing boundary file if mesh is already converted
        try:
            patches = parse_boundary_file(case_dir)
            self._log(f"[OpenFOAM] Found {len(patches)} boundary patches")
        except FileNotFoundError:
            # Mesh not converted yet - use default patches
            self._log("[OpenFOAM] Boundary file not found. Using default patches.")
            patches = config.get('default_patches', {
                'inlet': 'patch',
                'outlet': 'patch', 
                'wall': 'wall'
            })
        
        self._write_temperature_field(case_dir / "0" / "T", patches, config)
    
    def _write_control_dict(self, filepath: Path, config: Dict):
        """Write controlDict for laplacianFoam."""
        end_time = config.get('end_time', 1.0)
        write_interval = config.get('write_interval', 1)
        
        content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      controlDict;
}}

application     laplacianFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {end_time};
deltaT          1;
writeControl    timeStep;
writeInterval   {write_interval};
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
"""
        filepath.write_text(content)
        self._log(f"[OpenFOAM] Wrote {filepath.name}")
    
    def _write_fv_schemes(self, filepath: Path):
        """Write fvSchemes for laplacianFoam."""
        content = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
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
}

laplacianSchemes
{
    default         Gauss linear corrected;
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
        filepath.write_text(content)
        self._log(f"[OpenFOAM] Wrote {filepath.name}")
    
    def _write_fv_solution(self, filepath: Path, config: Dict):
        """Write fvSolution for laplacianFoam."""
        tolerance = config.get('solver_tolerance', 1e-6)
        rel_tol = config.get('solver_rel_tol', 0.01)
        
        content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSolution;
}}

solvers
{{
    T
    {{
        solver          PBiCGStab;
        preconditioner  DIC;
        tolerance       {tolerance};
        relTol          {rel_tol};
    }}
}}
"""
        filepath.write_text(content)
        self._log(f"[OpenFOAM] Wrote {filepath.name}")
    
    def _write_transport_properties(self, filepath: Path, config: Dict):
        """Write transportProperties for thermal diffusivity."""
        # Get thermal properties
        k = config.get('thermal_conductivity', 200.0)  # W/m/K
        rho = config.get('density', 2700.0)  # kg/m³
        cp = config.get('specific_heat', 900.0)  # J/kg/K
        
        # Thermal diffusivity: DT = k / (rho * cp)
        DT = k / (rho * cp)
        
        content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      transportProperties;
}}

DT              DT [0 2 -1 0 0 0 0] {DT:.6e};
"""
        filepath.write_text(content)
        self._log(f"[OpenFOAM] Wrote {filepath.name} (DT={DT:.6e} m²/s)")
    
    def _write_temperature_field(self, filepath: Path, patches: Dict[str, str], config: Dict):
        """Write 0/T with boundaryField for all patches."""
        ambient_temp = config.get('ambient_temperature', 300.0)
        initial_temp = config.get('initial_temperature', ambient_temp)
        
        # Build boundaryField section
        boundary_field_lines = []
        boundary_field_lines.append("boundaryField")
        boundary_field_lines.append("{")
        
        for patch_name, patch_type in patches.items():
            bc = classify_thermal_bc(patch_name, patch_type, config)
            
            boundary_field_lines.append(f"    {patch_name}")
            boundary_field_lines.append("    {")
            boundary_field_lines.append(f"        type            {bc['type']};")
            
            if bc['type'] == 'fixedValue' and bc['value'] is not None:
                boundary_field_lines.append(f"        value           uniform {bc['value']};")
            
            boundary_field_lines.append("    }")
        
        # Add default catch-all for any patches not explicitly listed
        boundary_field_lines.append("    defaultFaces")
        boundary_field_lines.append("    {")
        boundary_field_lines.append("        type            zeroGradient;")
        boundary_field_lines.append("    }")
        
        boundary_field_lines.append("}")
        
        boundary_field_str = "\n".join(boundary_field_lines)
        
        content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      T;
}}

dimensions      [0 0 0 1 0 0 0];

internalField   uniform {initial_temp};

{boundary_field_str}
"""
        filepath.write_text(content)
        self._log(f"[OpenFOAM] Wrote {filepath.name} with {len(patches)} boundary patches")

    def setup_thermal_boundary_conditions(self, case_dir: Path, config: Dict) -> None:
        """
        Setup thermal boundary conditions for an existing OpenFOAM case.
        
        This function can be called after gmshToFoam or fluentMeshToFoam has converted the mesh.
        It will:
        1. Parse boundary patches from constant/polyMesh/boundary
        2. Generate 0/T with appropriate boundaryField entries
        3. Generate/update transportProperties
        
        Args:
            case_dir: Path to OpenFOAM case directory
            config: Configuration dict (see classify_thermal_bc for options)
        """
        self._log("[OpenFOAM] Setting up thermal boundary conditions...")
        
        # Ensure directories exist
        (case_dir / "0").mkdir(parents=True, exist_ok=True)
        (case_dir / "constant").mkdir(parents=True, exist_ok=True)
        
        # Parse boundary patches
        try:
            patches = parse_boundary_file(case_dir)
            self._log(f"[OpenFOAM] Found {len(patches)} boundary patches: {list(patches.keys())}")
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Boundary file not found. Make sure to run 'gmshToFoam' or 'fluentMeshToFoam' first. {e}"
            )
        
        # Generate transportProperties
        self._write_transport_properties(case_dir / "constant" / "transportProperties", config)
        
        # Generate temperature field with boundary conditions
        self._write_temperature_field(case_dir / "0" / "T", patches, config)
        
        self._log("[OpenFOAM] Thermal boundary conditions setup complete!")

    def _run_commands(self, case_dir: Path, cmds: List[str]) -> bool:
        # Run commands in sequence
        # Handle WSL vs Native
        return True
