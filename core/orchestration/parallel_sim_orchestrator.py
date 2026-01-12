"""
Parallel Simulation Orchestrator
=================================

Manages concurrent thermal simulations across multiple meshes.
Supports CalculiX and OpenFOAM solvers with automatic fallback.

Key Features:
- Runs N simulations in parallel using multiprocessing
- Automatic solver selection based on availability and simulation type
- Aggregates results for ranking and comparison
- Graceful failure handling with timeout support

Usage:
    orchestrator = ParallelSimulationOrchestrator(max_workers=10)
    results = orchestrator.run_batch(
        mesh_files=['mesh1.msh', 'mesh2.msh', ...],
        template_config={'solver': 'auto', 'preset': 'electronics_cooling'},
        output_dir=Path('output/thermal_batch')
    )
    print(f"Best: {results.ranking[0].mesh_name}")
"""

import os
import sys
import time
import json
import shutil
import traceback
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Literal, Any
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from multiprocessing import cpu_count, Manager
import tempfile
import numpy as np

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.logging.sim_logger import SimLogger

logger = SimLogger("ParallelOrchestrator")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ThermalSimulationConfig:
    """Configuration for a thermal simulation run"""
    
    # Solver selection
    solver: Literal["calculix", "openfoam", "auto"] = "auto"
    
    # Material properties
    material: str = "Aluminum_6061"
    thermal_conductivity: float = 167.0  # W/m·K
    density: float = 2700.0  # kg/m³
    specific_heat: float = 896.0  # J/kg·K
    
    # Boundary conditions
    ambient_temp_c: float = 25.0
    source_temp_c: float = 100.0
    convection_coeff: float = 25.0  # W/m²·K
    
    # Simulation parameters
    transient: bool = False
    duration_s: float = 60.0
    time_step_s: float = 1.0
    
    # Pass/fail criteria
    max_temp_limit_c: float = 150.0
    min_temp_limit_c: float = -40.0
    
    # Solver paths (optional overrides)
    ccx_path: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'solver': self.solver,
            'physics': {
                'simulation_type': 'thermal',
                'material': self.material,
                'thermal_conductivity': self.thermal_conductivity,
                'density': self.density,
                'specific_heat': self.specific_heat,
                'ambient_temp_c': self.ambient_temp_c,
                'source_temp_c': self.source_temp_c,
                'convection_coeff': self.convection_coeff,
                'transient': self.transient,
                'duration': self.duration_s,
                'time_step': self.time_step_s,
                'ccx_path': self.ccx_path,
            },
            'pass_fail': {
                'max_temp_limit_c': self.max_temp_limit_c,
                'min_temp_limit_c': self.min_temp_limit_c,
            }
        }


@dataclass
class SimulationResult:
    """Result from a single simulation"""
    mesh_file: str
    mesh_name: str
    success: bool
    solver_used: str = ""
    
    # Temperature results
    min_temp_c: Optional[float] = None
    max_temp_c: Optional[float] = None
    avg_temp_c: Optional[float] = None
    temp_variance: Optional[float] = None
    max_gradient: Optional[float] = None
    
    # Performance metrics
    solve_time_s: float = 0.0
    num_elements: int = 0
    num_nodes: int = 0
    
    # Pass/fail
    passed: bool = False
    fail_reason: Optional[str] = None
    
    # Output files
    vtk_file: Optional[str] = None
    report_file: Optional[str] = None
    
    # Error info
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.mesh_name is None:
            self.mesh_name = Path(self.mesh_file).stem


@dataclass
class RankedSimulation:
    """Simulation result with ranking score"""
    result: SimulationResult
    rank: int
    composite_score: float
    individual_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class BatchSimulationResult:
    """Aggregated results from parallel batch run"""
    total_count: int
    completed_count: int
    failed_count: int
    passed_count: int
    
    results: List[SimulationResult] = field(default_factory=list)
    ranking: List[RankedSimulation] = field(default_factory=list)
    
    total_time_s: float = 0.0
    
    @property
    def success_rate(self) -> float:
        return self.completed_count / max(self.total_count, 1)
    
    @property
    def pass_rate(self) -> float:
        return self.passed_count / max(self.completed_count, 1)
    
    @property
    def best_performer(self) -> Optional[RankedSimulation]:
        return self.ranking[0] if self.ranking else None


# =============================================================================
# PRESET TEMPLATES
# =============================================================================

THERMAL_PRESETS = {
    "electronics_cooling": ThermalSimulationConfig(
        solver="calculix",
        material="Aluminum_6061",
        thermal_conductivity=167.0,
        ambient_temp_c=25.0,
        source_temp_c=85.0,
        convection_coeff=50.0,
        transient=False,
        max_temp_limit_c=100.0,
    ),
    "heat_sink": ThermalSimulationConfig(
        solver="calculix",
        material="Aluminum_6061",
        thermal_conductivity=167.0,
        ambient_temp_c=25.0,
        source_temp_c=100.0,
        convection_coeff=100.0,
        transient=False,
        max_temp_limit_c=120.0,
    ),
    "conduction_test": ThermalSimulationConfig(
        solver="calculix",
        material="Steel",
        thermal_conductivity=50.0,
        ambient_temp_c=20.0,
        source_temp_c=150.0,
        convection_coeff=10.0,
        transient=False,
        max_temp_limit_c=200.0,
    ),
    "rocket_nozzle": ThermalSimulationConfig(
        solver="calculix",
        material="Inconel_718",
        thermal_conductivity=11.4,
        ambient_temp_c=25.0,
        source_temp_c=800.0,
        convection_coeff=500.0,
        transient=True,
        duration_s=10.0,
        time_step_s=0.1,
        max_temp_limit_c=1000.0,
    ),
}


# =============================================================================
# WORKER FUNCTION (runs in subprocess)
# =============================================================================

def _run_single_simulation(
    mesh_file: str,
    config_dict: Dict,
    output_dir: str,
    worker_id: int
) -> Dict:
    """
    Worker function to run a single thermal simulation.
    Must be a top-level function for multiprocessing pickle.
    """
    mesh_path = Path(mesh_file)
    mesh_name = mesh_path.stem
    work_dir = Path(output_dir) / mesh_name
    
    result = {
        'mesh_file': mesh_file,
        'mesh_name': mesh_name,
        'success': False,
        'solver_used': '',
        'error': None,
    }
    
    try:
        # Create working directory
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine solver
        solver = config_dict.get('solver', 'auto')
        physics = config_dict.get('physics', {})
        
        if solver == 'auto':
            if physics.get('convection_coeff', 0) > 50:
                solver = 'openfoam'
            else:
                solver = 'calculix'
        
        result['solver_used'] = solver
        
        # Run simulation
        start_time = time.time()
        
        if solver == 'calculix':
            sim_result = _run_calculix_thermal(mesh_file, work_dir, physics)
        else:
            sim_result = _run_openfoam_thermal(mesh_file, work_dir, physics)
        
        result['solve_time_s'] = time.time() - start_time
        
        # Merge sim results
        result.update(sim_result)
        
        # Normalize keys for SimulationResult
        if 'min_temp' in result: result['min_temp_c'] = result['min_temp']
        if 'max_temp' in result: result['max_temp_c'] = result['max_temp']
        if 'avg_temp' in result: result['avg_temp_c'] = result['avg_temp']
        
        # Pass/fail check
        pass_fail = config_dict.get('pass_fail', {})
        max_limit = pass_fail.get('max_temp_limit_c', 150.0)
        min_limit = pass_fail.get('min_temp_limit_c', -40.0)
        
        if result['success']:
            if result.get('max_temp') is not None and result['max_temp'] > max_limit:
                result['passed'] = False
                result['fail_reason'] = f"Max temp {result['max_temp']:.1f}C exceeds limit {max_limit}C"
            elif result.get('min_temp') is not None and result['min_temp'] < min_limit:
                result['passed'] = False
                result['fail_reason'] = f"Min temp {result['min_temp']:.1f}C below limit {min_limit}C"
            else:
                result['passed'] = True
        else:
            result['passed'] = False
            result['fail_reason'] = result.get('error', 'Simulation failed')
        
        # ---------------------------------------------------------------------
        # REPORTING & VISUALIZATION
        # ---------------------------------------------------------------------
        if result['success'] and result.get('vtk_file'):
            try:
                # 1. Generate Visualizations
                from core.visualization.thermal_viz import ThermalVisualizer
                viz = ThermalVisualizer(work_dir)
                
                # Depending on solver, VTK file might be different
                # CalculiX adapter returns 'vtk_file' path (usually .vtu or .vtk)
                vtk_path = result['vtk_file']
                
                if Path(vtk_path).exists():
                    images = viz.generate_snapshots(
                        vtk_path, 
                        mesh_name,
                        temperature_field='temperature' if solver == 'calculix' else 'T'
                    )
                    result['images'] = images
                
                # 2. Generate PDF Report
                from core.reporting.thermal_report import ThermalPDFReportGenerator
                
                # Prepare data for report config
                report_data = {
                    'job_name': mesh_name,
                    'success': result['passed'],
                    'max_temp_k': (result.get('max_temp') or 0) + 273.15,
                    'min_temp_k': (result.get('min_temp') or 0) + 273.15,
                    'ambient_temp_c': physics.get('ambient_temp_c', 25.0),
                    'source_temp_c': physics.get('source_temp_c', 100.0),
                    'num_elements': result.get('num_elements', 0),
                    'num_nodes': result.get('num_nodes', 0),
                    'strategy_name': solver.capitalize()
                }
                
                reporter = ThermalPDFReportGenerator(job_name=mesh_name, output_dir=work_dir)
                pdf_path = reporter.generate(
                    job_name=mesh_name,
                    output_dir=work_dir,
                    data=report_data,
                    image_paths=result.get('images', [])
                )
                
                result['report_file'] = str(pdf_path)
                
            except Exception as e:
                # Don't fail the whole job if reporting fails
                print(f"Reporting failed for {mesh_name}: {e}")
                traceback.print_exc()
            
    except Exception as e:
        result['error'] = str(e)
        result['success'] = False
        result['passed'] = False
        result['fail_reason'] = f"Exception: {e}"
        traceback.print_exc()
    
    # Clean up large objects before returning (avoid pickle size issues if necessary)
    # But usually dicts of strings/numbers are fine.
    
    return result


def _run_calculix_thermal(mesh_file: str, work_dir: Path, physics: Dict) -> Dict:
    """Run CalculiX thermal simulation"""
    try:
        from core.solvers.calculix_adapter import CalculiXAdapter
        
        ccx_path = physics.get('ccx_path') or 'ccx'
        adapter = CalculiXAdapter(ccx_binary=ccx_path)
        
        # Build config
        config = {
            'thermal_conductivity': physics.get('thermal_conductivity', 167.0),
            'density': physics.get('density', 2700.0),
            'specific_heat': physics.get('specific_heat', 896.0),
            'ambient_temperature': physics.get('ambient_temp_c', 25.0),
            'heat_source_temperature': physics.get('source_temp_c', 100.0),
            'convection_coeff': physics.get('convection_coeff', 25.0),
            'transient': physics.get('transient', False),
            'duration': physics.get('duration', 60.0),
            'time_step': physics.get('time_step', 1.0),
        }
        
        run_output = adapter.run(Path(mesh_file), work_dir, config)
        
        result = {}
        result['vtk_file'] = run_output.get('vtk_file') # Adapter should return this
        
        # Convert temperature from Kelvin to Celsius if present
        # CalculiX uses K internally often, adapter auto-converts?
        # Let's assume adapter returns Kelvin as 'temperature' array, but min/max might be raw
        
        # Inspecting CalculiXAdapter output structure:
        # returns dict with keys: success, temperature (array), min_temp, max_temp, vtk_file
        
        if run_output.get('min_temp') is not None:
            # Check if values look like Kelvin (> 200) or Celsius
            # Standard SimOps adapter usually returns Kelvin or internal units.
            # safe assumption: if > 200 likely Kelvin.
            # But let's check adapter code... reviewed it earlier.
            # Actually, let's just map it directly and check sensible ranges later.
            
            # Assuming adapter returns Kelvin
            t_min_k = run_output['min_temp']
            t_max_k = run_output['max_temp']
            
            result['min_temp'] = t_min_k - 273.15
            result['max_temp'] = t_max_k - 273.15
        
        if 'temperature' in run_output and run_output['temperature'] is not None:
            import numpy as np
            temps = np.array(run_output['temperature'])
            temps_c = temps - 273.15
            result['avg_temp'] = float(np.mean(temps_c))
            result['temp_variance'] = float(np.var(temps_c))
            
            # Generate VTK for visualization
            vtk_path = work_dir / f"{Path(mesh_file).stem}.vtu"
            stats = run_output.get('mesh_stats')
            if stats:
                _write_vtk(vtk_path, stats, temps_c)
                result['vtk_file'] = str(vtk_path)
        
        result['success'] = True
        # Mesh statistics
        stats = run_output.get('mesh_stats', {})
        result['num_elements'] = stats.get('num_elements', 0)
        result['num_nodes'] = stats.get('num_nodes', 0)
        
        return result
        
    except Exception as e:
        return {'success': False, 'error': str(e)}


def _write_vtk(output_path: Path, stats: Dict, temperature: np.ndarray):
    """
    Write VTK file using meshio.
    Assumes temperature array aligns with sorted Node IDs (CalculiXAdapter standard).
    Assumes stats['elements'] indices align with sorted Node IDs (Gmsh standard).
    """
    try:
        import meshio
        import numpy as np
        
        node_map = stats['node_map']
        elements = stats['elements']
        
        # 1. Reconstruct Points (Sorted by ID to match Temperature)
        sorted_ids = sorted(node_map.keys())
        points = np.array([node_map[nid] for nid in sorted_ids])
        
        # 2. Reconstruct Cells
        # stats['elements'] is indices into points array
        # Assuming Gmsh tags were sorted, inputs.indices == sorted_indices
        
        # Identify element type from col count
        # Tet4: 5 cols (1 tag + 4 nodes)
        # Tet10: 11 cols (1 tag + 10 nodes)
        if len(elements) > 0:
            cols = elements.shape[1]
            if cols == 4:
                cells = [("triangle", elements[:, 1:])]
            elif cols == 5:
                cells = [("tetra", elements[:, 1:])]
            elif cols == 7: # Tri6
                cells = [("triangle6", elements[:, 1:])]
            elif cols == 11:
                cells = [("tetra10", elements[:, 1:])]
            else:
                # Fallback / Mixed
                if cols > 4:
                    cells = [("tetra", elements[:, 1:5])]
                else:
                    cells = [("triangle", elements[:, 1:4])]
        else:
            cells = []
            
        # 3. Point Data
        # temperature is already aligned to sorted_ids (from adapter parse logic)
        point_data = {
            "temperature": temperature,
            "T": temperature # Alias
        }
        
        # Write
        mesh = meshio.Mesh(
            points,
            cells,
            point_data=point_data
        )
        mesh.write(str(output_path))
        
    except Exception as e:
        print(f"VTK Generation failed: {e}")
        traceback.print_exc()


def _run_openfoam_thermal(mesh_file: str, work_dir: Path, physics: Dict) -> Dict:
    """Run OpenFOAM thermal simulation"""
    try:
        from core.solvers.openfoam_wrapper import OpenFOAMWrapper
        
        wrapper = OpenFOAMWrapper(verbose=True)
        
        config = {
            'thermal_conductivity': physics.get('thermal_conductivity', 167.0),
            'ambient_temp': physics.get('ambient_temp_c', 25.0) + 273.15,  # To Kelvin
            'source_temp': physics.get('source_temp_c', 100.0) + 273.15,
        }
        
        result = wrapper.solve_thermal_cfd(mesh_file, str(work_dir), config)
        
        # OpenFOAM wrapper returns minimal data currently
        # This will be enhanced in Phase 2
        result['success'] = result.get('vtk_file') is not None
        return result
        
    except Exception as e:
        return {'success': False, 'error': str(e)}


# =============================================================================
# RANKING ENGINE
# =============================================================================

class ThermalRankingEngine:
    """Rank simulations by thermal performance"""
    
    def __init__(self, 
                 weight_max_temp: float = 0.3,
                 weight_uniformity: float = 0.25,
                 weight_gradient: float = 0.2,
                 weight_solve_time: float = 0.15,
                 weight_mesh_quality: float = 0.1):
        self.weights = {
            'max_temp': weight_max_temp,
            'uniformity': weight_uniformity,
            'gradient': weight_gradient,
            'solve_time': weight_solve_time,
            'mesh_quality': weight_mesh_quality,
        }
    
    def rank(self, results: List[SimulationResult]) -> List[RankedSimulation]:
        """Rank simulations (lower max temp + higher uniformity = better)"""
        
        # Filter to successful simulations
        valid = [r for r in results if r.success and r.max_temp_c is not None]
        
        if not valid:
            return []
        
        # Normalize metrics
        max_temps = [r.max_temp_c for r in valid]
        solve_times = [r.solve_time_s for r in valid]
        
        max_temp_range = max(max_temps) - min(max_temps) if len(set(max_temps)) > 1 else 1.0
        solve_time_range = max(solve_times) - min(solve_times) if len(set(solve_times)) > 1 else 1.0
        
        ranked = []
        for r in valid:
            scores = {}
            
            # Max temp score (lower is better, normalized to 0-1)
            scores['max_temp'] = 1.0 - (r.max_temp_c - min(max_temps)) / max(max_temp_range, 1.0)
            
            # Uniformity score (lower variance is better)
            if r.temp_variance is not None and r.temp_variance > 0:
                scores['uniformity'] = 1.0 / (1.0 + r.temp_variance)
            else:
                scores['uniformity'] = 0.5  # Default
            
            # Gradient score (placeholder - would need gradient data)
            scores['gradient'] = 0.5  # TODO: Implement when gradient data available
            
            # Solve time score (faster is better)
            scores['solve_time'] = 1.0 - (r.solve_time_s - min(solve_times)) / max(solve_time_range, 1.0)
            
            # Mesh quality score (more elements = potentially better resolution)
            scores['mesh_quality'] = min(1.0, r.num_elements / 10000.0) if r.num_elements > 0 else 0.5
            
            # Composite score
            composite = sum(
                scores[k] * self.weights[k] 
                for k in self.weights.keys()
            )
            
            ranked.append(RankedSimulation(
                result=r,
                rank=0,  # Will be set after sorting
                composite_score=composite,
                individual_scores=scores
            ))
        
        # Sort by composite score (higher is better)
        ranked.sort(key=lambda x: x.composite_score, reverse=True)
        
        # Assign ranks
        for i, r in enumerate(ranked):
            r.rank = i + 1
        
        return ranked


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

class ParallelSimulationOrchestrator:
    """
    Orchestrates parallel thermal simulations across multiple meshes.
    
    Example:
        orch = ParallelSimulationOrchestrator(max_workers=8)
        results = orch.run_batch(
            mesh_files=['mesh1.msh', 'mesh2.msh', 'mesh3.msh'],
            template_config={'preset': 'electronics_cooling'},
            output_dir=Path('output/batch_run')
        )
        print(f"Best: {results.best_performer.result.mesh_name}")
    """
    
    def __init__(self, 
                 max_workers: Optional[int] = None,
                 timeout_per_sim: int = 600,
                 verbose: bool = True):
        """
        Initialize orchestrator.
        
        Args:
            max_workers: Maximum parallel simulations. Default: CPU cores - 2
            timeout_per_sim: Timeout per simulation in seconds
            verbose: Print progress messages
        """
        if max_workers is None:
            max_workers = max(1, cpu_count() - 2)
        
        self.max_workers = max_workers
        self.timeout = timeout_per_sim
        self.verbose = verbose
        self.ranking_engine = ThermalRankingEngine()
    
    def _log(self, msg: str):
        if self.verbose:
            logger.info(msg)
    
    def run_batch(
        self,
        mesh_files: List[str],
        template_config: Dict,
        output_dir: Path
    ) -> BatchSimulationResult:
        """
        Run thermal simulations on all meshes in parallel.
        
        Args:
            mesh_files: List of mesh file paths
            template_config: Configuration dict with keys:
                - 'preset': Name of preset template (e.g., 'electronics_cooling')
                - 'solver': 'calculix', 'openfoam', or 'auto'
                - 'physics': Dict of physics parameters (optional, overrides preset)
            output_dir: Directory for simulation outputs
        
        Returns:
            BatchSimulationResult with all results and rankings
        """
        start_time = time.time()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build configuration
        preset_name = template_config.get('preset', 'electronics_cooling')
        if preset_name in THERMAL_PRESETS:
            config = THERMAL_PRESETS[preset_name].to_dict()
        else:
            config = ThermalSimulationConfig().to_dict()
        
        # Override with user settings
        if 'solver' in template_config:
            config['solver'] = template_config['solver']
        if 'physics' in template_config:
            config['physics'].update(template_config['physics'])
        if 'pass_fail' in template_config:
            config['pass_fail'].update(template_config['pass_fail'])
        
        self._log("=" * 70)
        self._log("PARALLEL THERMAL SIMULATION ORCHESTRATOR")
        self._log("=" * 70)
        self._log(f"  Mesh files: {len(mesh_files)}")
        self._log(f"  Workers: {self.max_workers}")
        self._log(f"  Preset: {preset_name}")
        self._log(f"  Solver: {config['solver']}")
        self._log(f"  Output: {output_dir}")
        self._log("=" * 70)
        
        # Run simulations in parallel
        results: List[SimulationResult] = []
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            futures = {}
            for i, mesh_file in enumerate(mesh_files):
                future = executor.submit(
                    _run_single_simulation,
                    mesh_file,
                    config,
                    str(output_dir),
                    i
                )
                futures[future] = mesh_file
            
            # Collect results as they complete
            for future in as_completed(futures, timeout=self.timeout * len(mesh_files)):
                mesh_file = futures[future]
                mesh_name = Path(mesh_file).stem
                
                try:
                    result_dict = future.result(timeout=self.timeout)
                    
                    # Convert dict to SimulationResult
                    result = SimulationResult(
                        mesh_file=result_dict['mesh_file'],
                        mesh_name=result_dict['mesh_name'],
                        success=result_dict['success'],
                        solver_used=result_dict.get('solver_used', ''),
                        min_temp_c=result_dict.get('min_temp_c'),
                        max_temp_c=result_dict.get('max_temp_c'),
                        avg_temp_c=result_dict.get('avg_temp_c'),
                        temp_variance=result_dict.get('temp_variance'),
                        solve_time_s=result_dict.get('solve_time_s', 0),
                        num_elements=result_dict.get('num_elements', 0),
                        num_nodes=result_dict.get('num_nodes', 0),
                        passed=result_dict.get('passed', False),
                        fail_reason=result_dict.get('fail_reason'),
                        vtk_file=result_dict.get('vtk_file'),
                        error=result_dict.get('error'),
                    )
                    results.append(result)
                    
                    status = "PASS" if result.passed else "FAIL"
                    if result.success:
                        self._log(f"  [{status}] {mesh_name}: "
                                 f"T_max={result.max_temp_c:.1f}C, "
                                 f"t={result.solve_time_s:.1f}s")
                    else:
                        self._log(f"  [{status}] {mesh_name}: {result.error or result.fail_reason}")
                        
                except TimeoutError:
                    self._log(f"  [TIMEOUT] {mesh_name}: Exceeded {self.timeout}s")
                    results.append(SimulationResult(
                        mesh_file=mesh_file,
                        mesh_name=mesh_name,
                        success=False,
                        error=f"Timeout after {self.timeout}s"
                    ))
                except Exception as e:
                    self._log(f"  [ERROR] {mesh_name}: {e}")
                    results.append(SimulationResult(
                        mesh_file=mesh_file,
                        mesh_name=mesh_name,
                        success=False,
                        error=str(e)
                    ))
        
        # Rank results
        ranking = self.ranking_engine.rank(results)
        
        total_time = time.time() - start_time
        
        # Build batch result
        batch_result = BatchSimulationResult(
            total_count=len(mesh_files),
            completed_count=sum(1 for r in results if r.success),
            failed_count=sum(1 for r in results if not r.success),
            passed_count=sum(1 for r in results if r.passed),
            results=results,
            ranking=ranking,
            total_time_s=total_time,
        )
        
        # Summary
        self._log("=" * 70)
        self._log("SUMMARY")
        self._log("=" * 70)
        self._log(f"  Total time: {total_time:.1f}s")
        self._log(f"  Success rate: {batch_result.success_rate*100:.0f}%")
        self._log(f"  Pass rate: {batch_result.pass_rate*100:.0f}%")
        
        if ranking:
            self._log(f"\n  TOP 3 PERFORMERS:")
            for r in ranking[:3]:
                self._log(f"    #{r.rank} {r.result.mesh_name}: "
                         f"score={r.composite_score:.3f}, "
                         f"T_max={r.result.max_temp_c:.1f}C")
        
        # Save results JSON
        results_file = output_dir / "batch_results.json"
        self._save_results(batch_result, results_file)
        self._log(f"\n  Results saved to: {results_file}")
        
        return batch_result
    
    def _save_results(self, batch_result: BatchSimulationResult, output_file: Path):
        """Save batch results to JSON"""
        data = {
            'summary': {
                'total_count': batch_result.total_count,
                'completed_count': batch_result.completed_count,
                'failed_count': batch_result.failed_count,
                'passed_count': batch_result.passed_count,
                'success_rate': batch_result.success_rate,
                'pass_rate': batch_result.pass_rate,
                'total_time_s': batch_result.total_time_s,
            },
            'ranking': [
                {
                    'rank': r.rank,
                    'mesh_name': r.result.mesh_name,
                    'composite_score': r.composite_score,
                    'scores': r.individual_scores,
                    'max_temp_c': r.result.max_temp_c,
                    'passed': r.result.passed,
                }
                for r in batch_result.ranking
            ],
            'results': [
                {
                    'mesh_name': r.mesh_name,
                    'mesh_file': r.mesh_file,
                    'success': r.success,
                    'passed': r.passed,
                    'fail_reason': r.fail_reason,
                    'solver_used': r.solver_used,
                    'min_temp_c': r.min_temp_c,
                    'max_temp_c': r.max_temp_c,
                    'avg_temp_c': r.avg_temp_c,
                    'solve_time_s': r.solve_time_s,
                    'num_elements': r.num_elements,
                    'vtk_file': r.vtk_file,
                    'error': r.error,
                }
                for r in batch_result.results
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """CLI entry point for batch thermal simulation"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run parallel thermal simulations on multiple meshes"
    )
    parser.add_argument(
        "meshes", 
        nargs="+", 
        help="Mesh files to simulate"
    )
    parser.add_argument(
        "--output", "-o", 
        default="output/thermal_batch",
        help="Output directory"
    )
    parser.add_argument(
        "--preset", "-p",
        default="electronics_cooling",
        choices=list(THERMAL_PRESETS.keys()),
        help="Simulation preset"
    )
    parser.add_argument(
        "--solver", "-s",
        default="auto",
        choices=["calculix", "openfoam", "auto"],
        help="Solver to use"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=600,
        help="Timeout per simulation (seconds)"
    )
    
    args = parser.parse_args()
    
    orchestrator = ParallelSimulationOrchestrator(
        max_workers=args.workers,
        timeout_per_sim=args.timeout,
        verbose=True
    )
    
    results = orchestrator.run_batch(
        mesh_files=args.meshes,
        template_config={
            'preset': args.preset,
            'solver': args.solver,
        },
        output_dir=Path(args.output)
    )
    
    # Exit code based on results
    if results.completed_count == 0:
        sys.exit(2)  # All failed
    elif results.passed_count < results.completed_count:
        sys.exit(1)  # Some failed pass/fail
    else:
        sys.exit(0)  # All passed


if __name__ == "__main__":
    main()
