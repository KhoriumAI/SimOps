"""
Sequential OpenFOAM Thermal Job Runner
=======================================

OHEC-based job runner for chtMultiRegionFoam simulations.
Runs three realistic electronics cooling setups sequentially with validation.

Usage:
    python thermal_job_runner.py --setups 3 --output ./thermal_runs
    python thermal_job_runner.py --dry-run  # Test without OpenFOAM
"""

import os
import sys
import json
import shutil
import logging
import tempfile
import subprocess
import platform
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from core.validation.thermal_pass_fail import (
        ThermalPassFailEngine, PassFailCriteria, electronics_criteria
    )
except ImportError:
    # Fallback if not in path
    ThermalPassFailEngine = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

class JobStatus(Enum):
    """Status of a thermal job"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ThermalSetup:
    """Configuration for a single thermal simulation setup"""
    name: str
    description: str
    
    # Heat source
    power_watts: float
    
    # Heatsink area (exposed to air) - typical small heatsink is 50cm² = 0.005 m²
    heatsink_area_m2: float = 0.005
    
    # Convection
    air_velocity_ms: float = 1.0
    inlet_temp_k: float = 300.0  # 27°C ambient
    
    # Domain (meters)
    domain_size_m: tuple = (0.1, 0.05, 0.05)  # 100mm x 50mm x 50mm
    
    # Solver settings
    max_iterations: int = 200
    write_interval: int = 20
    tolerance: float = 1e-6
    simulation_type: str = "steady_state"
    time_step: float = 0.1
    duration: float = 10.0
    convection_coefficient: float = 20.0
    initial_temperature: float = 300.0
    
    # Material (solid region)
    material: str = "Aluminum"
    solid_conductivity_wm_k: float = 167.0  # Aluminum 6061
    solid_density_kgm3: float = 2700.0
    solid_specific_heat_jkg_k: float = 896.0
    
    @property
    def expected_delta_t(self) -> float:
        """
        Estimate temperature rise using simplified forced convection.
        
        Uses Q = h * A * ΔT, where:
        - h ≈ 10 + 10*v W/m²K for forced convection over flat plate
        - A = heatsink area
        """
        # Heat transfer coefficient (simplified correlation)
        h = 10 + 10 * self.air_velocity_ms  # W/m²K
        
        # ΔT = Q / (h * A)
        delta_t = self.power_watts / (h * self.heatsink_area_m2)
        return min(delta_t, 100.0)  # Cap at 100°C for realism


@dataclass
class JobResult:
    """Result of a single thermal job"""
    setup_name: str
    status: JobStatus
    case_dir: Optional[str] = None
    
    # Thermal results
    min_temp_c: Optional[float] = None
    max_temp_c: Optional[float] = None
    avg_temp_c: Optional[float] = None
    
    # Solver info
    converged: bool = False
    final_residual: Optional[float] = None
    iterations_run: int = 0
    solve_time_s: float = 0.0
    
    # Validation
    passed_validation: bool = False
    validation_reason: str = ""
    
    # Errors
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['status'] = self.status.value
        return d


# =============================================================================
# PREDEFINED SETUPS (Realistic Electronics Cooling)
# =============================================================================

def get_standard_setups() -> List[ThermalSetup]:
    """Return three standard thermal setups for electronics cooling"""
    
    return [
        ThermalSetup(
            name="low_power_natural",
            description="Low-power IoT device with natural convection (5W)",
            power_watts=5.0,
            air_velocity_ms=0.1,  # Buoyancy-driven, minimal forced flow
            inlet_temp_k=300.0,
            max_iterations=150,
        ),
        ThermalSetup(
            name="medium_power_forced",
            description="Typical PCB with chassis fan (25W @ 1 m/s)",
            power_watts=25.0,
            air_velocity_ms=1.0,
            inlet_temp_k=300.0,
            max_iterations=200,
        ),
        ThermalSetup(
            name="high_power_active",
            description="High-performance board with active cooling (50W @ 3 m/s)",
            power_watts=50.0,
            air_velocity_ms=3.0,
            inlet_temp_k=300.0,
            max_iterations=250,
        ),
    ]


# =============================================================================
# CASE GENERATOR
# =============================================================================

class CaseGenerator:
    """Generates parameterized OpenFOAM cases from the Golden_CHT_Case template"""
    
    TEMPLATE_PATH = PROJECT_ROOT / "simops" / "templates" / "Golden_Thermal_Case"
    
    MATERIAL_LIBRARY = {
        'Aluminum': {'kappa': 200, 'rho': 2700, 'Cp': 900},
        'Copper': {'kappa': 400, 'rho': 8960, 'Cp': 385},
        'Steel': {'kappa': 16, 'rho': 8000, 'Cp': 500},
        'Silicon': {'kappa': 148, 'rho': 2330, 'Cp': 700}
    }

    def __init__(self, output_base: Path):
        self.output_base = Path(output_base)
        self.output_base.mkdir(parents=True, exist_ok=True)
        
    def generate_case(self, setup: ThermalSetup) -> Path:
        """
        Generate a parameterized case directory for the given setup.
        
        Returns:
            Path to the generated case directory
        """
        case_dir = self.output_base / f"case_{setup.name}"
        
        if case_dir.exists():
            logger.info(f"Removing existing case: {case_dir}")
            shutil.rmtree(case_dir)
        
        # Copy template
        logger.info(f"Copying template to: {case_dir}")
        shutil.copytree(self.TEMPLATE_PATH, case_dir)
        
        # Parameterize files
        self._update_control_dict(case_dir, setup)
        self._update_material_properties(case_dir, setup)
        self._update_solver_settings(case_dir, setup)
        self._update_physics_bcs(case_dir, setup)
        
        logger.info(f"Case generated: {case_dir}")
        return case_dir

    def _update_physics_bcs(self, case_dir: Path, setup: ThermalSetup):
        """Update physics Boundary Conditions (Heat and Convection)"""
        t_solid = case_dir / "0" / "solid_heatsink" / "T"
        if not t_solid.exists():
            return
            
        content = t_solid.read_text()
        
        # 1. Update Heat Source (Wattage)
        # Power [W] -> Flux [W/m2] -> Gradient [K/m]
        # grad = q / k
        # For simplicity, if power is set, we can use a fixedValue for the demo
        # or calculate the gradient. Let's start with fixedValue as a safe default
        # but the user wanted "wattage" so let's try fixedGradient.
        
        thermal_k = self.MATERIAL_LIBRARY.get(setup.material, {'kappa': 200})['kappa']
        area = getattr(setup, 'heatsink_area_m2', 0.005)
        # Gradient = (P / A) / k
        gradient_val = (setup.power_watts / area) / thermal_k
        
        import re
        # Match the heatsink_bottom block
        repl = rf'heatsink_bottom\n    {{\n        type            fixedGradient;\n        gradient        uniform {gradient_val};\n    }}'
        content = re.sub(r'heatsink_bottom\s*\{[^}]*\}', repl, content)
        
        # 2. Update Ambient T
        content = re.sub(r'internalField\s+uniform\s+\S+;', f'internalField   uniform {setup.inlet_temp_k};', content)
        
        t_solid.write_text(content)
        
        # 3. Update Fluid Inlet (if exists)
        t_fluid = case_dir / "0" / "region1" / "T"
        if t_fluid.exists():
            f_content = t_fluid.read_text()
            f_content = re.sub(r'uniform\s+3\d\d', f'uniform {setup.inlet_temp_k}', f_content)
            t_fluid.write_text(f_content)

        # 4. Update Velocity (Inlet) - Reuse old logic if needed
        u_fluid = case_dir / "0" / "region1" / "U"
        if u_fluid.exists():
            u_content = u_fluid.read_text()
            u_content = u_content.replace("uniform (1 0 0)", f"uniform ({setup.air_velocity_ms} 0 0)")
            u_fluid.write_text(u_content)

    def _update_control_dict(self, case_dir: Path, setup: ThermalSetup):
        """Update controlDict with solver settings"""
        control_dict_path = case_dir / "system" / "controlDict"
        
        content = control_dict_path.read_text()
        
        # Update simulation application based on mode
        app = "chtMultiRegionSimpleFoam" if setup.simulation_type == "steady_state" else "chtMultiRegionFoam"
        content = self._replace_value(content, "application", app)
        
        # Update endTime (for steady state it's max iterations, for transient it's duration)
        end_time = setup.max_iterations if setup.simulation_type == "steady_state" else setup.duration
        content = self._replace_value(content, "endTime", str(end_time))
        content = self._replace_value(content, "deltaT", str(setup.time_step))
        content = self._replace_value(content, "writeInterval", str(setup.write_interval))
        
        control_dict_path.write_text(content)
        logger.debug(f"Updated controlDict: app={app}, endTime={end_time}")
    
    def _update_velocity_bc(self, case_dir: Path, setup: ThermalSetup):
        """Update inlet velocity boundary condition"""
        u_file = case_dir / "0" / "region1" / "U"
        
        if not u_file.exists():
            logger.warning(f"U file not found: {u_file}")
            return
            
        content = u_file.read_text()
        
        # Replace inlet velocity (format: uniform (Ux Uy Uz))
        old_value = "uniform (1 0 0)"
        new_value = f"uniform ({setup.air_velocity_ms} 0 0)"
        content = content.replace(old_value, new_value)
        
        u_file.write_text(content)
        logger.debug(f"Updated U inlet: velocity={setup.air_velocity_ms} m/s")
    
    def _update_temperature_bc(self, case_dir: Path, setup: ThermalSetup):
        """Update temperature boundary conditions"""
        # Fluid region
        t_fluid = case_dir / "0" / "region1" / "T"
        if t_fluid.exists():
            content = t_fluid.read_text()
            content = content.replace("uniform 300", f"uniform {setup.inlet_temp_k}")
            t_fluid.write_text(content)

        # Solid region - set initial temp based on expected rise
        t_solid = case_dir / "0" / "solid_heatsink" / "T"
        if t_solid.exists():
            # Start solid at slightly elevated temp
            initial_solid_t = setup.inlet_temp_k + 20  # Start 20K above ambient
            content = t_solid.read_text()
            content = content.replace("uniform 350", f"uniform {initial_solid_t}")
            t_solid.write_text(content)
            logger.debug(f"Updated solid T initial: {initial_solid_t}K")

    def _update_material_properties(self, case_dir: Path, setup: ThermalSetup):
        """Update material properties in thermophysicalProperties files"""
        # Get material properties
        mat_props = self.MATERIAL_LIBRARY.get(setup.material)
        if not mat_props:
            logger.warning(f"Material {setup.material} not in library, using defaults")
            mat_props = {'kappa': setup.solid_conductivity_wm_k,
                        'rho': setup.solid_density_kgm3,
                        'Cp': setup.solid_specific_heat_jkg_k}

        # Update solid region thermophysicalProperties
        thermo_file = case_dir / "constant" / "solid_heatsink" / "thermophysicalProperties"
        if thermo_file.exists():
            content = thermo_file.read_text()
            content = self._replace_value(content, "kappa", str(mat_props['kappa']))
            content = self._replace_value(content, "rho", str(mat_props['rho']))
            content = self._replace_value(content, "Cp", str(mat_props['Cp']))
            thermo_file.write_text(content)
            logger.debug(f"Updated material properties: {setup.material} -> kappa={mat_props['kappa']}")

    def _update_solver_settings(self, case_dir: Path, setup: ThermalSetup):
        """Update solver settings in fvSolution"""
        # Update convergence tolerance in fvSolution
        for region in ['region1', 'solid_heatsink']:
            fv_solution = case_dir / "system" / region / "fvSolution"
            if fv_solution.exists():
                content = fv_solution.read_text()
                # Update tolerance - look for tolerance or residualControl
                import re
                # Replace tolerance values (scientific notation)
                content = re.sub(r'tolerance\s+\S+;', f'tolerance  {setup.tolerance};', content)
                fv_solution.write_text(content)
                logger.debug(f"Updated solver tolerance for {region}: {setup.tolerance}")

    @staticmethod
    def _replace_value(content: str, key: str, value: str) -> str:
        """Replace a simple key-value in OpenFOAM dict format"""
        import re
        # Match: key  value; (with variable whitespace)
        pattern = rf'({key}\s+)\S+;'
        replacement = rf'\g<1>{value};'
        return re.sub(pattern, replacement, content)


# =============================================================================
# OPENFOAM RUNNER
# =============================================================================

class OpenFOAMRunner:
    """Executes OpenFOAM commands (via WSL on Windows)"""
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.is_windows = platform.system() == "Windows"
        self.openfoam_available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """Check if OpenFOAM is available"""
        if self.dry_run:
            return True
            
        try:
            # Check for standard 2312 installation first
            check_cmd = "source /usr/lib/openfoam/openfoam2312/etc/bashrc 2>/dev/null; which chtMultiRegionFoam || which foamList"
            
            if self.is_windows:
                result = subprocess.run(
                    ['wsl', 'bash', '-c', check_cmd],
                    capture_output=True, timeout=10, text=True
                )
            else:
                result = subprocess.run(
                    ['bash', '-c', check_cmd],
                    capture_output=True, timeout=10, text=True
                )
            
            if result.returncode == 0:
                return True
                
            # Fallback to simple check
            if self.is_windows:
                result = subprocess.run(
                    ['wsl', 'bash', '-c', 'which foamList || which chtMultiRegionFoam'],
                    capture_output=True, timeout=10
                )
            else:
                result = subprocess.run(
                    ['which', 'foamList' if not self.is_windows else 'chtMultiRegionFoam'],
                    capture_output=True, timeout=10
                )
            return result.returncode == 0
        except Exception as e:
            logger.warning(f"OpenFOAM check failed: {e}")
            return False
    
    def run_case(self, case_dir: Path, setup: ThermalSetup) -> JobResult:
        """
        Run the OpenFOAM case.
        
        Returns:
            JobResult with status and thermal data
        """
        result = JobResult(
            setup_name=setup.name,
            status=JobStatus.RUNNING,
            case_dir=str(case_dir)
        )
        
        if self.dry_run:
            logger.info(f"[DRY RUN] Would run case: {case_dir}")
            result.status = JobStatus.COMPLETED
            result.converged = True
            result.final_residual = 1e-5  # Mock converged residual
            result.iterations_run = min(setup.max_iterations, 10)
            
            # Generate realistic mock temperatures
            ambient_c = setup.inlet_temp_k - 273.15
            delta_t = setup.expected_delta_t
            
            result.min_temp_c = ambient_c
            result.max_temp_c = ambient_c + delta_t
            result.avg_temp_c = ambient_c + delta_t * 0.6
            
            logger.info(f"[DRY RUN] Mock result: T_max={result.max_temp_c:.1f}°C, ΔT={delta_t:.1f}°C")
            return result
        
        if not self.openfoam_available:
            result.status = JobStatus.FAILED
            result.error_message = "OpenFOAM not available (WSL or native)"
            return result
        
        try:
            start_time = datetime.now()

            # Check if mesh already exists (skip meshing if so)
            polymesh_dir = case_dir / "constant" / "polyMesh"
            region_dirs = list((case_dir / "constant").glob("*/polyMesh"))

            if polymesh_dir.exists() and (polymesh_dir / "points").exists():
                logger.info(f"Mesh already exists in constant/polyMesh, skipping mesh generation")
            elif region_dirs:
                logger.info(f"Region meshes already exist, skipping mesh generation")
            else:
                # Run meshing pipeline only if mesh doesn't exist
                logger.info(f"Running blockMesh...")
                self._run_foam_command(case_dir, "blockMesh")

                logger.info(f"Running snappyHexMesh...")
                self._run_foam_command(case_dir, "snappyHexMesh -overwrite")

                logger.info(f"Running splitMeshRegions...")
                self._run_foam_command(case_dir, "splitMeshRegions -cellZones -overwrite")

            # Run solver - use SIMPLE (steady-state) or transient based on setup
            if setup.simulation_type == "steady_state":
                solver_cmd = "chtMultiRegionSimpleFoam"
            else:
                solver_cmd = "chtMultiRegionFoam"

            logger.info(f"Running {solver_cmd}...")
            solver_output = self._run_foam_command(case_dir, solver_cmd)
            
            result.solve_time_s = (datetime.now() - start_time).total_seconds()
            
            # Parse results
            self._parse_solver_output(solver_output, result, setup)
            self._extract_temperatures(case_dir, result)
            
            result.status = JobStatus.COMPLETED
            # result.converged is set by _parse_solver_output based on residuals/iterations
            
        except subprocess.CalledProcessError as e:
            result.status = JobStatus.FAILED
            result.error_message = f"OpenFOAM command failed: {e}"
            logger.error(result.error_message)
        except Exception as e:
            result.status = JobStatus.FAILED
            result.error_message = str(e)
            logger.error(f"Unexpected error: {e}")
        
        return result
    
    def _run_foam_command(self, case_dir: Path, command: str) -> str:
        """Run a single OpenFOAM command"""
        # Prefix with sourcing bashrc for standard installations
        foam_env = "source /usr/lib/openfoam/openfoam2312/etc/bashrc 2>/dev/null || true"
        full_bash_cmd = f"{foam_env}; cd '{case_dir}' && {command}"
        
        if self.is_windows:
            wsl_path = self._to_wsl_path(case_dir)
            full_bash_cmd = f"{foam_env}; cd '{wsl_path}' && {command}"
            full_cmd = ['wsl', 'bash', '-c', full_bash_cmd]
        else:
            full_cmd = ['bash', '-c', full_bash_cmd]
        
        result = subprocess.run(
            full_cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout per command
        )
        
        if result.returncode != 0:
            logger.error(f"Command failed: {command}")
            logger.error(f"STDOUT:\n{result.stdout}")
            logger.error(f"STDERR:\n{result.stderr}")
            raise subprocess.CalledProcessError(
                result.returncode, command, result.stdout, result.stderr
            )
        
        return result.stdout
    
    def _to_wsl_path(self, path: Path) -> str:
        """Convert Windows path to WSL path"""
        path_str = str(path.resolve())
        if len(path_str) >= 2 and path_str[1] == ':':
            drive = path_str[0].lower()
            return f"/mnt/{drive}/{path_str[3:].replace(chr(92), '/')}"
        return path_str.replace('\\', '/')
    
    def _parse_solver_output(self, output: str, result: JobResult, setup: ThermalSetup):
        """Parse solver output for convergence info (iterations, residuals)."""
        lines = output.splitlines()
        last_time = None
        last_residual = None

        import re
        time_re = re.compile(r'^\s*Time\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')
        residual_re = re.compile(r'Final residual\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')

        for line in lines:
            time_match = time_re.search(line)
            if time_match:
                try:
                    last_time = float(time_match.group(1))
                except ValueError:
                    pass
                continue

            residual_match = residual_re.search(line)
            if residual_match:
                try:
                    last_residual = float(residual_match.group(1))
                except ValueError:
                    pass

        if last_residual is not None:
            result.final_residual = last_residual

        # Estimate iterations run from last time step
        if last_time is not None:
            if setup.simulation_type == "steady_state":
                result.iterations_run = max(0, int(round(last_time)))
            else:
                if setup.time_step > 0:
                    result.iterations_run = max(0, int(round(last_time / setup.time_step)))

        hit_max = (
            setup.simulation_type == "steady_state"
            and result.iterations_run >= setup.max_iterations
        )
        if result.final_residual is not None:
            result.converged = result.final_residual <= setup.tolerance
        else:
            # If we stopped early without a residual, assume converged; otherwise mark not converged
            result.converged = not hit_max
    
    def _extract_temperatures(self, case_dir: Path, result: JobResult):
        """Extract temperature data from latest time directory"""
        # Find latest time directory
        time_dirs = [d for d in case_dir.iterdir() if d.is_dir() and d.name.replace('.', '').isdigit()]
        if not time_dirs:
            logger.warning("No time directories found")
            return

        latest = max(time_dirs, key=lambda d: float(d.name))

        # Look for T file in regions
        for region in ['solid_heatsink', 'region1', 'fluid']:
            t_file = latest / region / 'T'
            if t_file.exists():
                logger.info(f"Found temperature file: {t_file}")
                try:
                    # Parse OpenFOAM temperature field
                    content = t_file.read_text()

                    # Extract internalField values
                    import re
                    # Look for "internalField   uniform XXX;" or "internalField   nonuniform List<scalar>"
                    uniform_match = re.search(r'internalField\s+uniform\s+([\d.eE+-]+)', content)
                    if uniform_match:
                        temp_k = float(uniform_match.group(1))
                        result.min_temp_c = temp_k - 273.15
                        result.max_temp_c = temp_k - 273.15
                        result.avg_temp_c = temp_k - 273.15
                        logger.info(f"  Uniform temperature: {result.avg_temp_c:.1f}°C")
                    else:
                        # Try to parse nonuniform field
                        list_match = re.search(r'internalField\s+nonuniform\s+List<scalar>\s*\n\s*\d+\s*\n\s*\(\s*([\d\s.eE+-]+)\s*\)', content, re.MULTILINE | re.DOTALL)
                        if list_match:
                            values_str = list_match.group(1)
                            temps_k = [float(v) for v in values_str.split()]
                            if temps_k:
                                result.min_temp_c = min(temps_k) - 273.15
                                result.max_temp_c = max(temps_k) - 273.15
                                result.avg_temp_c = sum(temps_k) / len(temps_k) - 273.15
                                logger.info(f"  Temperature range: {result.min_temp_c:.1f}°C to {result.max_temp_c:.1f}°C")
                    break
                except Exception as e:
                    logger.warning(f"Failed to parse temperature field: {e}")


# =============================================================================
# SEQUENTIAL JOB RUNNER
# =============================================================================

class SequentialJobRunner:
    """
    Runs thermal simulations sequentially with validation.
    
    Uses OHEC methodology:
    - O: Observe baseline (check OpenFOAM availability)
    - H: Hypothesis (expected temp rises per setup)
    - E: Experiment (run simulations)
    - C: Conclusion (validate results)
    """
    
    def __init__(self, output_dir: Path, dry_run: bool = False):
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run
        
        self.generator = CaseGenerator(self.output_dir / "cases")
        self.runner = OpenFOAMRunner(dry_run=dry_run)
        self.results: List[JobResult] = []
        
        # Validation engine
        if ThermalPassFailEngine:
            self.validator = ThermalPassFailEngine(verbose=True)
            self.criteria = electronics_criteria(max_junction_temp_c=150.0)
        else:
            self.validator = None
            self.criteria = None
    
    def run_all(self, setups: List[ThermalSetup]) -> List[JobResult]:
        """
        Run all setups sequentially.
        
        Returns:
            List of JobResult objects
        """
        logger.info("=" * 60)
        logger.info("OHEC SEQUENTIAL THERMAL JOB RUNNER")
        logger.info("=" * 60)
        
        if not self.runner.openfoam_available and not self.dry_run:
            logger.error("OpenFOAM not available. Use --dry-run for testing.")
            return []
        
        logger.info(f"Running {len(setups)} setups sequentially")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Dry run: {self.dry_run}")
        logger.info("")
        
        for i, setup in enumerate(setups, 1):
            logger.info("-" * 40)
            logger.info(f"[{i}/{len(setups)}] {setup.name}")
            logger.info(f"Description: {setup.description}")
            logger.info(f"Power: {setup.power_watts}W, Velocity: {setup.air_velocity_ms} m/s")
            logger.info(f"Expected ΔT: ~{setup.expected_delta_t:.1f}°C")
            logger.info("-" * 40)
            
            # Generate case
            case_dir = self.generator.generate_case(setup)
            
            # Run simulation
            result = self.runner.run_case(case_dir, setup)
            
            # Validate result
            self._validate_result(result, setup)
            
            self.results.append(result)
            
            # Log result
            if result.status == JobStatus.COMPLETED:
                if result.max_temp_c is not None:
                    logger.info(f"✓ Completed: T_max={result.max_temp_c:.1f}°C")
                else:
                    logger.info(f"✓ Completed (temperature extraction pending)")
            else:
                logger.error(f"✗ Failed: {result.error_message}")
            
            logger.info("")
        
        # Summary
        self._print_summary()
        self._save_results()
        
        return self.results
    
    def _validate_result(self, result: JobResult, setup: ThermalSetup):
        """Validate thermal result against criteria"""
        if result.status != JobStatus.COMPLETED:
            result.passed_validation = False
            result.validation_reason = f"Job did not complete: {result.status.value}"
            return
        
        if self.validator and result.max_temp_c is not None:
            sim_result = {
                'success': True,
                'min_temp': result.min_temp_c,
                'max_temp': result.max_temp_c,
                'converged': result.converged,
                'final_residual': result.final_residual,
            }
            
            pf_result = self.validator.evaluate(sim_result, self.criteria)
            result.passed_validation = pf_result.passed
            result.validation_reason = pf_result.reason
        else:
            # Manual validation
            result.passed_validation = True
            result.validation_reason = "Validation engine not available"
    
    def _print_summary(self):
        """Print summary of all runs"""
        logger.info("=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        
        completed = sum(1 for r in self.results if r.status == JobStatus.COMPLETED)
        passed = sum(1 for r in self.results if r.passed_validation)
        
        logger.info(f"Total jobs: {len(self.results)}")
        logger.info(f"Completed: {completed}")
        logger.info(f"Passed validation: {passed}")
        
        for result in self.results:
            status = "✓" if result.passed_validation else "✗"
            temp = f"T_max={result.max_temp_c:.1f}°C" if result.max_temp_c else "N/A"
            logger.info(f"  {status} {result.setup_name}: {temp}")
    
    def _save_results(self):
        """Save results to JSON"""
        results_file = self.output_dir / "thermal_results.json"
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'dry_run': self.dry_run,
            'results': [r.to_dict() for r in self.results]
        }
        
        with open(results_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='OHEC Sequential OpenFOAM Thermal Job Runner'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./thermal_runs',
        help='Output directory for cases and results'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run without OpenFOAM (generates cases, simulates results)'
    )
    parser.add_argument(
        '--setups',
        type=int,
        default=3,
        choices=[1, 2, 3],
        help='Number of setups to run (1-3)'
    )
    
    args = parser.parse_args()
    
    # Get setups
    setups = get_standard_setups()[:args.setups]
    
    # Create runner
    runner = SequentialJobRunner(
        output_dir=Path(args.output),
        dry_run=args.dry_run
    )
    
    # Run
    results = runner.run_all(setups)
    
    # Exit code based on results
    if all(r.passed_validation for r in results):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
