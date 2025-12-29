#!/usr/bin/env python
"""
SimOps Worker - Cascading Fallback Simulation Engine
=====================================================

This is the worker process that runs simulations with automatic retry.
Implements a "Cascading Fallback" strategy for robustness.

Strategy Cascade:
    1. HighFi_CFD: size=0.5mm, 5 layers, Delaunay
    2. MedFi_Robust: size=1.0mm, 3 layers, Frontal
    3. LowFi_Emergency: size=2.0mm, 0 layers (pure conduction)

Usage:
    rq worker simops --url redis://localhost:6379
"""

import os
import sys
import json
import time
import logging
import traceback
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Optional, List

try:
    from rq import get_current_job
except ImportError:
    get_current_job = lambda: None

import gmsh
import numpy as np
import meshio

# Try importing PyVista for premium rendering
try:
    import pyvista as pv
    import vtk
    HAVE_VTK = True
except ImportError:
    HAVE_VTK = False

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "core"))
sys.path.insert(0, str(Path(__file__).parent / "core" / "strategies"))

# Import ResultDispatcher
from result_dispatcher import ResultDispatcher

# Import PDF Generator
try:
    from core.reporting.pdf_generator import PDFReportGenerator
except ImportError:
    logger.warning("PDFReportGenerator not found, PDF generation will be skipped")
    PDFReportGenerator = None

from core.geometry_analyzer import analyze_cad_geometry, GeometryAnalysis
from core.geometry_healer import AutomaticGeometryHealer
from core.validation.geometry_validator import GeometryValidator
from core.validation.solution_validator import SolutionValidator
from core.solvers.calculix_adapter import CalculiXAdapter
from core.solvers.calculix_structural import CalculiXStructuralAdapter
from core.solvers.cfd_solver import CFDSolver
from core.reporting.cfd_report import CFDPDFReportGenerator
from core.reporting.multi_angle_viz import generate_multi_angle_streamlines, generate_velocity_contour_slices

# Supported Mesh Extensions
MESH_EXTENSIONS = {'.msh', '.vtk', '.vtu', '.unv', '.inp'}

def is_mesh_file(path: str) -> bool:
    return Path(path).suffix.lower() in MESH_EXTENSIONS

try:
    from core.reporting.structural_report import StructuralPDFReportGenerator
    from core.reporting.structural_viz import generate_structural_report
    from core.reporting.thermal_report import ThermalPDFReportGenerator
except ImportError:
    StructuralPDFReportGenerator = None
    generate_structural_report = None
    ThermalPDFReportGenerator = None

try:
    from core.validation.pre_sim_checks import run_pre_simulation_checks
except ImportError:
    run_pre_simulation_checks = None


# Configure logging via SimLogger
from core.logging.sim_logger import SimLogger
logger = SimLogger("Worker")

# -----------------------------------------------------------------------------
# CONSOLE CAPTURE UTILS
# -----------------------------------------------------------------------------
class ConsoleCapturer:
    """
    Context manager that captures stdout/stderr to a file 
    while still printing to the real console (tee).
    """
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.file_handle = None
        self.stdout_orig = sys.stdout
        self.stderr_orig = sys.stderr

    def __enter__(self):
        self.file_handle = open(self.log_file, 'a', encoding='utf-8')
        # Write header
        self.file_handle.write(f"\\n=== LOG START: {datetime.now().isoformat()} ===\\n")
        self.file_handle.write(f"\n=== LOG START: {datetime.now().isoformat()} ===\n")
        
        class Tee:
            def __init__(self, original, file):
                self.original = original
                self.file = file
            def write(self, message):
                self.file.write(message) # Always write to file
                try:
                    self.original.write(message) # Try writing to original stdout/stderr
                except UnicodeEncodeError:
                    # Fallback for consoles that don't support certain characters
                    safe_msg = message.encode('ascii', errors='replace').decode('ascii')
                    self.original.write(safe_msg)
                except Exception:
                    # Catch other potential errors during console write, but don't fail the whole process
                    pass
                self.file.flush() # Ensure realtime persistence for the file
            def flush(self):
                self.original.flush()
                self.file.flush()
                
        sys.stdout = Tee(self.stdout_orig, self.file_handle)
        sys.stderr = Tee(self.stderr_orig, self.file_handle)
        
        # Refresh SimLogger to use the new Tee stream
        logger.set_stream(sys.stdout)
        
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout_orig
        sys.stderr = self.stderr_orig
        
        # Restore SimLogger to the original stdout
        logger.set_stream(sys.stdout)
        
        if self.file_handle:
            self.file_handle.write(f"\\n=== LOG END: {datetime.now().isoformat()} ===\\n")
            self.file_handle.close()

# =============================================================================
# CASCADING FALLBACK STRATEGIES
# =============================================================================
STRATEGIES = [
    {
        "name": "HighFi_Layered",
        "description": "High fidelity mesh with boundary layers (CFD/Thermal)",
        "mesh_size_factor": 1.0, # Relative to auto-diagonal
        "num_layers": 5,
        "growth_rate": 1.2,
        "algorithm_2d": 6,   # Frontal-Delaunay
        "algorithm_3d": 1,   # Delaunay
        "optimize": True,
    },
    {
        "name": "MedFi_Robust", 
        "description": "Medium fidelity with fewer layers",
        "mesh_size_factor": 2.0, # 2x coarser than HighFi
        "num_layers": 3,
        "growth_rate": 1.3,
        "algorithm_2d": 6,   # Frontal-Delaunay
        "algorithm_3d": 4,   # Frontal
        "optimize": True,
    },
    {
        "name": "LowFi_Emergency",
        "description": "Emergency fallback - pure conduction mesh",
        "mesh_size_factor": 4.0, # 4x coarser than HighFi
        "num_layers": 0,     # No boundary layers
        "growth_rate": 1.0,
        "algorithm_2d": 1,   # MeshAdapt
        "algorithm_3d": 1,   # Delaunay
        "optimize": False,
    },
]


@dataclass
class SimulationResult:
    """Result from a simulation attempt"""
    success: bool
    strategy_name: str
    mesh_file: Optional[str] = None
    vtk_file: Optional[str] = None
    report_file: Optional[str] = None
    png_file: Optional[str] = None
    min_temp: Optional[float] = None
    max_temp: Optional[float] = None
    num_elements: int = 0
    solve_time: float = 0
    max_stress_mpa: Optional[float] = None
    max_disp_mm: Optional[float] = None
    error: Optional[str] = None


def generate_mesh_with_strategy(
    cad_file: str, 
    output_dir: Path,
    strategy: Dict,
    sim_config: Optional['SimulationConfig'] = None # Added config
) -> str:
    job_name = Path(cad_file).stem
    strategy_name = strategy.get('name', 'Unknown')
    output_file = output_dir / f"{job_name}_{strategy_name}.msh"
    
    logger.log_stage("Meshing")
    logger.log_metadata("strategy", strategy_name)

    # Determine simulation type
    sim_type = "thermal"
    if sim_config and hasattr(sim_config, 'physics'):
        sim_type = getattr(sim_config.physics, 'simulation_type', "thermal")

    # Define mesh scaling based on strategy (HighFi=1.0, LowFi=Coarser)
    scalings = {
        "HighFi_Layered": 1.0,
        "MedFi_Robust": 2.0,
        "LowFi_Emergency": 4.0
    }
    base_factor = scalings.get(strategy_name, strategy.get('mesh_size_factor', 1.0))

    # Higher quality for structural confirmation (User Request: aim for >5k elements)
    if sim_type == 'structural' and strategy_name == "HighFi_Layered":
        base_factor = 0.4 # Significant refinement to resolve "hotspots"
        logger.info(f"  [Refinement] Increasing structural mesh density (factor={base_factor})")

    # Overrides from Sidecar
    second_order = False
    config_mesh_size = 1.0
    if sim_config and hasattr(sim_config, 'meshing') and sim_config.meshing:
        config_mesh_size = getattr(sim_config.meshing, 'mesh_size_factor', 1.0)
        if hasattr(sim_config.meshing, 'second_order') and sim_config.meshing.second_order:
            second_order = True
            logger.info("  [Override] Enabling Second-Order Elements (Sidecar)")
        if hasattr(sim_config.meshing, 'mesh_size_multiplier') and sim_config.meshing.mesh_size_multiplier != 1.0:
            base_factor *= sim_config.meshing.mesh_size_multiplier
            logger.info(f"  [Override] Scaling mesh size by {sim_config.meshing.mesh_size_multiplier}")

    # Intelligent Wind Tunnel Detection
    enable_wind_tunnel = False
    sim_type = 'cfd'
    if sim_config and hasattr(sim_config, 'physics'):
        sim_type = getattr(sim_config.physics, 'simulation_type', 'thermal')
        if sim_type == 'cfd':
            # Check for explicit override first
            if hasattr(sim_config.physics, 'virtual_wind_tunnel') and sim_config.physics.virtual_wind_tunnel is not None:
                enable_wind_tunnel = sim_config.physics.virtual_wind_tunnel
                if enable_wind_tunnel:
                    logger.info(f"  [Wind Tunnel] Enabled explicitly via config")
                else:
                    logger.info(f"  [Wind Tunnel] Disabled explicitly via config")
            # Auto-detect only if not set
            elif hasattr(sim_config.physics, 'inlet_velocity') and sim_config.physics.inlet_velocity > 0:
                enable_wind_tunnel = True
                logger.info(f"  [Wind Tunnel] Auto-enabled for external flow (Velocity > 0)")

    # ---------------------------------------------------------
    # CHECKS & BALANCES: Load Correct Physics Configuration
    # ---------------------------------------------------------
    
    # If structural, use Structural Strategy
    if sim_type == 'structural':
        from core.strategies.structural_strategy import StructuralMeshConfig, StructuralMeshStrategy
        
        logger.info("[Meshing] Using Structural Strategy (Checks & Balances)")
        
        mesh_config = StructuralMeshConfig(
            mesh_size_factor=config_mesh_size * base_factor,
            second_order=second_order,
            optimize=strategy.get('optimize', True)
        )
        
        runner = StructuralMeshStrategy(verbose=True)
        success, stats = runner.generate_mesh(cad_file, str(output_file), mesh_config)

    else:
        # Default to CFD / Thermal 
        from core.strategies.cfd_strategy import CFDMeshConfig, CFDMeshStrategy
        
        logger.info("[Meshing] Using CFD/Thermal Strategy")
        
        config = CFDMeshConfig(
            num_layers=strategy.get('num_layers', 5),
            growth_rate=strategy.get('growth_rate', 1.2),
            mesh_size_factor=config_mesh_size * base_factor,
            create_prisms=True,
            optimize_netgen=strategy.get('optimize', True),
            smoothing_steps=10,
            virtual_wind_tunnel=enable_wind_tunnel
        )
        
        logger.info(f"  [DEBUG] CFDMeshConfig: virtual_wind_tunnel={config.virtual_wind_tunnel}")
        
        # Pass tagging rules from sidecar if available
        tagging_rules = sim_config.tagging_rules if sim_config else []
        
        runner = CFDMeshStrategy(verbose=True)
        runner.tagging_rules = tagging_rules
        
        success, stats = runner.generate_cfd_mesh(cad_file, str(output_file), config)
    
    if not success:
        raise RuntimeError(f"Mesh generation failed: {stats.get('error')}")
    
    # =========================================================================
    # QUALITY CONTROL: Validate mesh before proceeding to solver
    # =========================================================================
    logger.log_stage("Quality Control")
    
    from core.validation.mesh_quality_validator import validate_mesh_quality
    
    qc_passed, qc_reason, qc_metadata = validate_mesh_quality(Path(output_file))
    
    if not qc_passed:
        # Log detailed failure reason
        logger.error(f"[MeshQC] ❌ {qc_reason}")
        logger.log_metadata("qc_failure", qc_reason)
        
        # Store QC metadata for debugging
        qc_file = output_dir / f"{job_name}_{strategy_name}_qc.json"
        try:
            with open(qc_file, 'w') as f:
                json.dump({
                    'passed': False,
                    'reason': qc_reason,
                    'metadata': qc_metadata,
                    'mesh_file': str(output_file)
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write QC metadata: {e}")
        
        raise RuntimeError(f"Mesh Quality Check Failed: {qc_reason}")
    
    logger.info(f"[MeshQC] [OK] {qc_reason}")
    if qc_metadata:
        logger.log_metadata("qc_cells", qc_metadata.get('n_cells', 0))
        logger.log_metadata("qc_min_jacobian", qc_metadata.get('min_jacobian', 'N/A'))
        
    return str(output_file)
        
    # Export Metadata Sidecar (e.g. for Characteristic Length)
    if runner.wind_tunnel_data:
        try:
            meta_file = output_dir / "mesh_metadata.json"
            meta_data = {
                'wind_tunnel': runner.wind_tunnel_data.get('dimensions', {}),
                'generated_at': time.time(),
                'strategy': strategy['name']
            }
            with open(meta_file, 'w') as f:
                json.dump(meta_data, f, indent=2)
            logger.info(f"  [Metadata] Written to {meta_file.name}")
        except Exception as e:
            logger.warning(f"  [Metadata] Failed to write sidecar: {e}")
        
    return str(output_file)


def run_thermal_solver(mesh_file: Path, output_dir: Path, strategy_name: str, sim_config: Optional['SimulationConfig'] = None) -> Dict:
    """
    Run thermal solver with fallback chain:
    1. Try CalculiX (professional solver)
    2. Fall back to Python/SciPy solver (built-in)
    """
    # Try CalculiX first
    try:
        logger.log_stage("Solving (Thermal)")
        logger.log_metadata("solver", "CalculiX")
        
        if sim_config and hasattr(sim_config, 'physics'):
             if hasattr(sim_config.physics, 'material'):
                 logger.log_metadata("material", sim_config.physics.material)
             if hasattr(sim_config.physics, 'heat_load_watts'):
                 logger.log_metadata("heat_load", f"{sim_config.physics.heat_load_watts}W")
        
        ccx_path = getattr(sim_config.physics, 'ccx_path', 'ccx') if sim_config else 'ccx'
        if not ccx_path: ccx_path = 'ccx'
        adapter = CalculiXAdapter(ccx_binary=ccx_path)
        
        # Build Adapter Config from SimulationConfig
        adapter_config = {}
        if sim_config and hasattr(sim_config, 'physics'):
             phy = sim_config.physics
             
             # --- Material Hydration ---
             from core.materials import get_material
             
             # Helper to prioritise: Explicit > Library > Default
             def get_prop(field_name, attr_name, default_val):
                 val = getattr(phy, field_name, None)
                 if val is not None:
                     return val
                 # Check library
                 if phy.material:
                     try:
                        mat = get_material(phy.material)
                        return getattr(mat, attr_name, default_val)
                     except KeyError:
                        pass
                 return default_val

             adapter_config['thermal_conductivity'] = get_prop('thermal_conductivity', 'conductivity', 150.0)
             adapter_config['density'] = get_prop('density', 'density', 2700.0)
             adapter_config['specific_heat'] = get_prop('specific_heat', 'specific_heat', 900.0)
             
        # Mapping and Unit Normalization
        def get_temp_c(obj, field_c, field_k, default):
            """
            Normalize temperature to Celsius for CalculiX adapter.
            Priority: field_c (Celsius) > field_k (convert from Kelvin) > default
            
            CRITICAL: The CalculiXAdapter ONLY accepts Celsius input.
            All temperatures are internally converted to Kelvin by the adapter.
            """
            val_c = getattr(obj, field_c, None)
            if val_c is not None: 
                return val_c
            val_k = getattr(obj, field_k, None)
            if val_k is not None: 
                return val_k - 273.15
            return default

        adapter_config['ambient_temperature'] = get_temp_c(phy, 'ambient_temp_c', 'ambient_temperature', 25.0)
        adapter_config['heat_source_temperature'] = get_temp_c(phy, 'source_temp_c', 'heat_source_temperature', 100.0)
        adapter_config['initial_temperature'] = get_temp_c(phy, 'initial_temp_c', 'initial_temperature', 25.0)

        # Log temperature configuration (verify Celsius normalization)
        logger.log_metadata("ambient_temp_c", adapter_config['ambient_temperature'])
        logger.log_metadata("source_temp_c", adapter_config['heat_source_temperature'])
        logger.log_metadata("initial_temp_c", adapter_config['initial_temperature'])

        adapter_config['convection_coeff'] = getattr(phy, 'convection_coeff', 25.0)
        adapter_config['transient'] = getattr(phy, 'transient', True)
        adapter_config['duration'] = getattr(phy, 'duration', 60.0)
        adapter_config['time_step'] = getattr(phy, 'time_step', 2.0)
        adapter_config['fix_hot_boundary'] = getattr(phy, 'fix_hot_boundary', True)
        adapter_config['fix_cold_boundary'] = getattr(phy, 'fix_cold_boundary', False)
        adapter_config['heat_load_watts'] = getattr(phy, 'heat_load_watts', 0.0)
        adapter_config['volumetric_heat_wm3'] = getattr(phy, 'volumetric_heat_wm3', 0.0)
        adapter_config['unit_scaling'] = getattr(phy, 'unit_scaling', 1.0)

        result = adapter.run(Path(mesh_file), Path(output_dir), adapter_config)
        
        if 'elements' in result:
            result['num_elements'] = len(result['elements'])
        else:
            result['num_elements'] = 0
            
        # Defensive logging: Handle None temperatures from parser failures
        try:
            if result.get('min_temp') is not None and result.get('max_temp') is not None:
                logger.info(f"  [CalculiX] Solver finished. Temp range: {result['min_temp']:.1f}K - {result['max_temp']:.1f}K")
            else:
                logger.warning(f"  [CalculiX] Solver finished but temperature data is incomplete (parse failure suspected)")
        except Exception as e:
            logger.warning(f"  [CalculiX] Solver finished but failed to log temperature: {e}")
        
        return result
        
    except Exception as e:
        logger.warning(f"CalculiX failed ({e}), falling back to Python solver...")

        
    # Fallback: Python/SciPy solver
    try:
        from simops_pipeline import ThermalSolver, SimOpsConfig
        
        fallback_config = SimOpsConfig()
        
        if sim_config:

             def get_val(path_str, default):
                 # path_str e.g. "physics.heat_source_temperature"
                 obj = sim_config
                 for part in path_str.split('.'):
                     if isinstance(obj, dict):
                         obj = obj.get(part)
                     else:
                         obj = getattr(obj, part, None)
                     if obj is None: return default
                 return obj

             fallback_config.heat_source_temperature = get_val("physics.heat_source_temperature", 373.15)
             fallback_config.ambient_temperature = get_val("physics.ambient_temperature", 298.15)
             fallback_config.thermal_conductivity = get_val("physics.thermal_conductivity", 150.0)
             fallback_config.unit_scaling = get_val("physics.unit_scaling", 1.0)
             
             # If conductivity is None, try material lookup
             if fallback_config.thermal_conductivity is None:
                 mat_name = get_val("physics.material", None)
                 if mat_name:
                     from core.materials import get_material
                     try:
                        mat = get_material(mat_name)
                        fallback_config.thermal_conductivity = mat.conductivity
                     except KeyError:
                        pass
        
        solver = ThermalSolver(fallback_config, verbose=True)
        result = solver.solve(str(mesh_file))

        
        if 'elements' in result:
            result['num_elements'] = len(result['elements'])
        else:
            result['num_elements'] = 0
            
        logger.info(f"  [Python] Solver finished. Temp range: {result['min_temp']:.1f}K - {result['max_temp']:.1f}K")
        return result
        
    except Exception as e:
        logger.error(f"Python solver also failed: {e}")
        raise


def extract_surface_mesh(node_coords, elements):
    """
    Extract boundary surface triangles from tetrahedral mesh.
    Returns: (surf_nodes, surf_tris) for plotting.
    """
    # Faces of a tet: (0,1,2), (0,1,3), (0,2,3), (1,2,3)
    # Vectorized face extraction
    # If elements have tag in column 0, skip it
    if elements.shape[1] in [5, 11]:
        elems_clean = elements[:, 1:5] # Take corner nodes
    else:
        elems_clean = elements[:, :4] # Assume no tags or already cleaned
        
    faces = np.vstack([
        elems_clean[:, [0,1,2]],
        elems_clean[:, [0,1,3]],
        elems_clean[:, [0,2,3]],
        elems_clean[:, [1,2,3]]
    ])
    faces.sort(axis=1)
    
    # Find unique faces and their counts
    unique_faces, return_index, return_inverse, return_counts = np.unique(
        faces, axis=0, return_index=True, return_inverse=True, return_counts=True
    )
    
    # Boundary faces appear exactly once
    boundary_mask = return_counts == 1
    boundary_faces = unique_faces[boundary_mask]
    
    return boundary_faces

def generate_vtk_isometric_view(
    node_coords: np.ndarray,
    surface_tris: np.ndarray, 
    temperature: np.ndarray,
    output_path: Path,
    vmin: float,
    vmax: float
) -> bool:
    """
    Generates a high-quality isometric view using PyVista (VTK).
    Returns True if successful.
    """
    if not HAVE_VTK:
        return False

    # Safety Check: If data is uniform (e.g. parsing failed), skip render
    # A crash is worse than no plot.
    if temperature is None or len(temperature) == 0 or np.all(temperature == temperature[0]):
        logger.warning("VTK Render Skipped: Temperature data is uniform/empty (Parsing likely failed).")
        return False
        
    try:
        # 1. Create PyVista PolyData from surface triangles
        # PyVista/VTK format: [n_pts, p0, p1, ... pk, n_pts, ...]
        # For triangles, n_pts = 3
        
        n_faces = len(surface_tris)
        padding = np.full((n_faces, 1), 3)
        faces_pv = np.hstack([padding, surface_tris]).flatten()
        
        mesh = pv.PolyData(node_coords, faces_pv)
        
        # Add temperature data
        mesh.point_data['Temperature'] = temperature
        
        # 2. Render Offscreen
        # Set window size for high-res output
        pl = pv.Plotter(off_screen=True, window_size=[1024, 800])
        pv.set_plot_theme("document") # Clean white text/bg (usually)
        pl.set_background('white')
        
        pl.add_mesh(mesh, 
                   scalars='Temperature', 
                   cmap='plasma', 
                   clim=[vmin, vmax],
                   smooth_shading=False, # Disabled to prevent LLVMpipe segfaults
                   show_edges=False,
                   show_scalar_bar=False)
                   # specular=0.5,      # Disabled to prevent LLVMpipe segfaults
                   # specular_power=15)
        
        pl.view_isometric()
        pl.camera.zoom(1.2) # Zoom to fill frame
        
        pl.screenshot(str(output_path), transparent_background=True)
        pl.close()
        
        return True
        
    except Exception as e:
        logger.warning(f"VTK Render failed: {e}")
        # traceback.print_exc()
        return False


def generate_report(
    job_name: str,
    output_dir: Path,
    result: Dict,
    strategy_name: str,
    sim_config: Dict = None,
    mesh_file: str = None
) -> Dict[str, str]:
    """
    Generate premium visualizations and PDF report (Celsius).
    Layout: 2x2 Grid (Iso + 3 Sections), Masked for Hollows.
    """
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    try:
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        pass
    try:
        from PIL import Image
    except ImportError:
        Image = None
    import io
    
    node_coords = np.array(result.get('node_coords', []))
    temp_raw = result.get('temperature')
    if temp_raw is None or len(temp_raw) == 0:
        logger.warning("Report Generation: No temperature data found in result.")
        temp_C = np.array([25.0]) # Dummy
        T_min = 25.0
        T_max = 25.0
    else:
        # Filter None values if any exist in the list
        temp_list = [t if t is not None else 298.15 for t in temp_raw]
        temp_K = np.array(temp_list)
        # Metric Switch: Kelvin -> Celsius
        temp_C = temp_K - 273.15
        T_min = np.nanmin(temp_C)
        T_max = np.nanmax(temp_C)
    
    # CRITICAL: Verify data alignment
    if len(node_coords) != len(temp_C):
        logger.warning(f"Data mismatch: {len(node_coords)} nodes vs {len(temp_C)} temps. Attempting recovery...")
        min_len = min(len(node_coords), len(temp_C))
        node_coords = node_coords[:min_len]
        temp_C = temp_C[:min_len]
    
    # Safe Context Extraction
    def to_dict(obj):
        if hasattr(obj, 'model_dump'): return obj.model_dump()
        if hasattr(obj, 'dict'): return obj.dict()
        return obj if isinstance(obj, dict) else {}

    config_dict = to_dict(sim_config) if sim_config else {}
    phy = config_dict.get('physics', {})
        
    def get_safe_temp(val_c, val_k, default_c=25.0):
        if val_c is not None: return float(val_c)
        if val_k is not None: return float(val_k) - 273.15
        return default_c

    init_C = get_safe_temp(phy.get('initial_temp_c'), phy.get('initial_temperature'))
    mat_name = phy.get('material', 'Aluminum 6061')
    amb_C = get_safe_temp(phy.get('ambient_temp_c'), phy.get('ambient_temperature'))
    h_coeff = phy.get('convection_coeff', 25.0)
    
    # Solver Metrics
    flux_w = result.get('heat_flux_watts')
    num_elems = result.get('num_elements', 0)
    solve_time = result.get('solve_time', 0)
    
    # Specific Convergence Details
    conv_threshold = result.get('convergence_threshold', 0.1)  # dT threshold in K
    conv_steps = result.get('convergence_steps', 3)  # steps under threshold
    if result.get('converged'):
        convergence = f"Converged (dT < {conv_threshold}°C)"
    else:
        final_dt = result.get('final_dT', 'N/A')
        convergence = f"Transient (Final dT: {final_dt}°C)"
    
    # Professional Report Title
    clean_title = job_name.replace('_', ' ')
    report_title = f"Sim Results - {clean_title}"
    if h_coeff > 50:
        report_title = f"Sim Results - {clean_title} (Forced Conv.)"
    
    # Build subtitle with simulation params
    _src_val = phy.get('heat_source_temperature')
    src_temp_C = (_src_val if _src_val is not None else 373.15) - 273.15
    param_subtitle = f"Material: {mat_name} | Ambient: {amb_C:.0f}°C | Source: {src_temp_C:.0f}°C | h={h_coeff} W/m²K"
    
    # 1. Generate High-Quality PNG
    png_file = output_dir / f"{job_name}_temperature.png"
    
    # Generate Helper VTK Image first
    vtk_png_path = output_dir / f"{job_name}_iso_vtk.png"
    vtk_success = False
    
    elements = result.get('elements')
    surf_faces = None
    try:
         if elements is not None and len(elements) > 0:
              surf_faces = extract_surface_mesh(node_coords, np.array(elements))
    except Exception as e:
         logger.warning(f"Surface mesh extraction failed: {e}")
         surf_faces = None
         
    if surf_faces is not None and HAVE_VTK:
        logger.info("Attempting VTK Isometric Render...")
        vtk_success = generate_vtk_isometric_view(node_coords, surf_faces, temp_C, vtk_png_path, T_min, T_max)
        if vtk_success:
            logger.info("VTK Render Success")
    
    # Grid for 4 plots (2x2 Layout) - Optimized for PDF A4
    # Using a wider layout to accommodate the sidebar colorbar cleanly
    fig = plt.figure(figsize=(16, 12)) 
    
    # Create 2x2 subplots with enough right margin for the colorbar
    # left, bottom, right, top
    plt.subplots_adjust(left=0.04, bottom=0.1, right=0.88, top=0.88, wspace=0.2, hspace=0.3)
    
    # 2x2 Grid
    gs = fig.add_gridspec(2, 2)
    
    # Note: We always create a 3D axis for the isometric view to avoid "plot_trisurf" errors in fallback.
    # If VTK succeeds, we will use it as a 2D pane essentially by imshowing.
    ax_iso = fig.add_subplot(gs[0, 0], projection='3d')
        
    axes = []
    axes.append(fig.add_subplot(gs[0, 1])) # XY
    axes.append(fig.add_subplot(gs[1, 0])) # XZ
    axes.append(fig.add_subplot(gs[1, 1])) # YZ
    
    # Global Levels - ensure minimum spread to prevent degenerate contours
    temp_spread = T_max - T_min
    if temp_spread < 1.0:  # Less than 1°C spread
        logger.warning(f"Temperature spread very small ({temp_spread:.2f}°C), expanding for visualization.")
        T_min -= 0.5
        T_max += 0.5
    levels = np.linspace(T_min, T_max, 25)

    # --- Subplot 1: 3D Isometric View ---
    ax_iso.set_title("3D Isometric View (Surface)", fontsize=11, fontweight='bold', pad=10)
    
    if vtk_success and vtk_png_path.exists():
        # High Quality VTK Path
        try:
            # We turn off the 3D background for the image
            ax_iso.set_axis_off()
            iso_img = plt.imread(str(vtk_png_path))
            # Create a 2D inset/overlay or just use imshow. 
            # Note: imshow on a 3D axis works but it's cleaner to just replace the axis or use it flat.
            # To avoid 'plot_trisurf' attribute error, we use a boolean check.
            pass # We will handle this in the if/else block below
        except Exception as e:
            logger.warning(f"Failed to prepare VTK image: {e}")

    # Final decision on 3D Render
    if vtk_success and vtk_png_path.exists():
        try:
            iso_img = plt.imread(str(vtk_png_path))
            # Plot on a new 2D axis over the same grid spot to be safe
            ax_iso.remove()
            ax_iso = fig.add_subplot(gs[0, 0])
            ax_iso.imshow(iso_img)
            ax_iso.set_axis_off()
            ax_iso.set_title("3D Isometric View (Surface)", fontsize=11, fontweight='bold', pad=10)
            # Clean up temp file
            try: vtk_png_path.unlink() 
            except: pass
        except Exception as e:
            logger.warning(f"Failed to load VTK image: {e}")
            ax_iso.text(0.5, 0.5, "Image Load Error", ha='center')
    else:
        # VTK didn't work - show placeholder
        # Matplotlib 3D has triangulation issues with some element types
        ax_iso.text(0.5, 0.5, 0.5, "3D Render Unavailable\n(Use VTK path)", 
                   ha='center', va='center', fontsize=12)
        ax_iso.set_axis_off()

    # --- Subplots 2,3,4: Cross Sections ---
    shape_views = [
        (0, 1, 2, 'XY Section (Top)', 'X', 'Y'), 
        (0, 2, 1, 'XZ Section (Front)', 'X', 'Z'), 
        (1, 2, 0, 'YZ Section (Side)', 'Y', 'Z'), 
    ]
    
    # Guard: Ensure temp_C matches node_coords length
    if len(temp_C) != len(node_coords):
        logger.warning(f"[Viz] Temperature array length ({len(temp_C)}) != node count ({len(node_coords)}). Skipping cross-sections.")
        for ax in axes:
            ax.text(0.5, 0.5, "Data Mismatch", ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
    else:
        for ax, (xi, yi, zi, title, xlabel, ylabel) in zip(axes, shape_views):
            try:
                z_coords = node_coords[:, zi]
                z_min_coord, z_max_coord = np.min(z_coords), np.max(z_coords)
                z_mid = (z_min_coord + z_max_coord) / 2.0
                z_range = z_max_coord - z_min_coord
                epsilon = z_range * 0.15 if z_range > 0 else 1.0  # 15% slice for better node capture
                
                mask = np.abs(z_coords - z_mid) < epsilon
                n_points = np.sum(mask)
                
                if n_points < 4:
                    ax.text(0.5, 0.5, f"Slice Empty\n({n_points} pts)", ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(title, fontsize=11, fontweight='bold')
                    ax.set_axis_off()
                    continue
                
                sx = node_coords[mask, xi]
                sy = node_coords[mask, yi]
                st = temp_C[mask]
                
                # Filter NaN/Inf
                valid_mask = np.isfinite(sx) & np.isfinite(sy) & np.isfinite(st)
                sx, sy, st = sx[valid_mask], sy[valid_mask], st[valid_mask]
                
                if len(st) < 4:
                    ax.text(0.5, 0.5, "Insufficient Data", ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(title, fontsize=11, fontweight='bold')
                    ax.set_axis_off()
                    continue
                
                # Deduplicate points (Qhull requires unique vertices)
                # Round to avoid floating-point near-duplicates
                coords_rounded = np.round(np.column_stack([sx, sy]), decimals=6)
                _, unique_idx = np.unique(coords_rounded, axis=0, return_index=True)
                unique_idx = np.sort(unique_idx)  # Preserve order
                sx, sy, st = sx[unique_idx], sy[unique_idx], st[unique_idx]
                
                if len(st) < 4:
                    ax.text(0.5, 0.5, "Too Few Unique Pts", ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(title, fontsize=11, fontweight='bold')
                    ax.set_axis_off()
                    continue
                
                # Method: Grid Interpolation (Robust for coarse meshes)
                # We use linear interpolation for the main field and nearest to fill edges
                try:
                    from scipy.interpolate import griddata
                    
                    # Create dense grid (200x200)
                    xi = np.linspace(sx.min(), sx.max(), 200)
                    yi = np.linspace(sy.min(), sy.max(), 200)
                    Xi, Yi = np.meshgrid(xi, yi)
                    
                    # Log-linear interpolation for smooth fields
                    Zi = griddata((sx, sy), st, (Xi, Yi), method='linear')
                    
                    # Fill NaN edges (extrapolation) with nearest neighbor to "fill up" the shape
                    # This prevents white gaps at the boundaries
                    mask_nan = np.isnan(Zi)
                    if np.any(mask_nan):
                        Zi_nearest = griddata((sx, sy), st, (Xi, Yi), method='nearest')
                        Zi[mask_nan] = Zi_nearest[mask_nan]
                    
                    # Plot filled mesh
                    # Use 'plasma' for better visibility at low end (magma is too dark/black)
                    cntr = ax.pcolormesh(Xi, Yi, Zi, cmap='plasma', vmin=T_min, vmax=T_max, shading='gouraud')
                    
                    # Overlay actual data points as small dots for ground truth
                    ax.scatter(sx, sy, c='k', s=1, alpha=0.3)
                    
                except Exception as grid_err:
                    logger.warning(f"[Viz] Griddata failed: {grid_err}. Fallback to Triangulation.")
                    # Legacy Triangulation Fallback
                    try:
                        triang = mtri.Triangulation(sx, sy)
                        cntr = ax.tricontourf(triang, st, levels=levels, cmap='plasma', extend='both')
                    except Exception as tri_err:
                        # Final Fallback: Scatter
                        cntr = ax.scatter(sx, sy, c=st, cmap='plasma', s=20, vmin=T_min, vmax=T_max)

                ax.set_title(title, fontsize=11, fontweight='bold')
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_aspect('equal')
                ax.grid(True, linestyle=':', alpha=0.4)
                
            except Exception as e:
                logger.error(f"[Viz] Cross-section {title} failed: {e}")
                ax.text(0.5, 0.5, "Plot Error", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title, fontsize=11, fontweight='bold')
                ax.set_axis_off()
                



    # --- Manual Colorbar Placement ---
    # Fixes overlap by defining explicit axis on the right
    # [left, bottom, width, height]
    # We reserved 88% screen width (right=0.88), so we put cbar at 0.90
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7]) 
    if 'cntr' in locals():
        cbar = fig.colorbar(cntr, cax=cbar_ax, label='Temperature (°C)')
        cbar.ax.tick_params(labelsize=10)

    # Title & Subtitle spacing
    fig.suptitle(f"{report_title}", fontsize=20, fontweight='bold', y=0.97)
    fig.text(0.5, 0.93, param_subtitle, ha='center', fontsize=12, color='#57606a', style='italic')

    # Footer
    footer_text = (f"Solver: CalculiX v2.21 (FEM) | Mesh: {num_elems:,} Elements | "
                   f"Solve: {solve_time:.1f}s | {convergence} | Flux: {f'{flux_w:.1f}W' if flux_w else 'N/A'}")
    
    fig.text(0.5, 0.03, footer_text, ha='center', fontsize=10, color='#2f3542', 
             style='italic', bbox=dict(facecolor='#f6f8fa', alpha=0.9, edgecolor='#d0d7de', boxstyle='round,pad=0.5'))
    
    plt.savefig(png_file, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"  PNG saved: {png_file.name}")
    
    # 1.5 Transient Plot (Celsius)
    ts_file = None
    if 'time_series_stats' in result and result['time_series_stats']:
        try:
             stats = result['time_series_stats']
             ts_file = output_dir / f"{job_name}_transient.png"
             
             valid_stats = [s for s in stats if s.get('min') is not None and s.get('max') is not None and s.get('mean') is not None]
             
             if len(valid_stats) == 0:
                 logger.warning("[Transient Debug] No valid stats to plot!")
                 raise ValueError("No valid time series data")
             
             if len(valid_stats) < 2:
                 logger.info("[Transient] Only 1 time step (steady-state). Skipping line plot, creating summary instead.")
                 # Create a simple summary figure instead
                 plt.figure(figsize=(10, 6))
                 s = valid_stats[0]
                 plt.bar(['Min', 'Mean', 'Max'], [s['min']-273.15, s['mean']-273.15, s['max']-273.15], color=['blue', 'green', 'red'])
                 plt.title(f"Steady-State Temperature: {job_name}", fontsize=14, fontweight='bold')
                 plt.ylabel("Temperature (°C)")
                 plt.grid(True, linestyle='--', alpha=0.5, axis='y')
                 plt.savefig(ts_file, dpi=150, bbox_inches='tight')
                 plt.close()
                 logger.info(f"  Steady-State Summary saved: {ts_file.name}")
             else:
                 times = [s['time'] for s in valid_stats]
                 mins = [s['min'] - 273.15 for s in valid_stats]
                 maxs = [s['max'] - 273.15 for s in valid_stats]
                 means = [s['mean'] - 273.15 for s in valid_stats]
             
                 plt.figure(figsize=(10, 6))
                 plt.plot(times, maxs, 'r-', linewidth=2, label='Max Temp', marker='o')
                 plt.plot(times, means, 'g--', linewidth=2, label='Mean Temp', marker='s')
                 plt.plot(times, mins, 'b-', linewidth=2, label='Min Temp', marker='^')
             
                 plt.title(f"Transient Response: {job_name}", fontsize=14, fontweight='bold')
                 plt.xlabel("Time (s)")
                 plt.ylabel("Temperature (°C)")
                 plt.grid(True, linestyle='--', alpha=0.5)
                 plt.legend()
             
                 # Annotation for final Steady State
                 plt.annotate(f"Final Max: {maxs[-1]:.1f}°C", 
                              xy=(times[-1], maxs[-1]), xytext=(-40, 10), 
                              textcoords='offset points', arrowprops=dict(arrowstyle="->"),
                              fontsize=10, fontweight='bold')
             
                 plt.savefig(ts_file, dpi=150, bbox_inches='tight')
                 plt.close()
                 logger.info(f"  Transient PNG saved")
        except Exception as e:
             logger.warning(f"Transient plot failed: {e}")
             print(f"Transient plot failed: {e}")
    else:
         msg = f"[Debug] Transient Graph Skipped: time_series_stats missing or empty. keys={list(result.keys())}"
         logger.info(msg)
         print(msg)

     # 1.6 Animation (Celsius) - Minimal or None for now to avoid bloat
    gif_file = None
    if 'time_series' in result and result['time_series'] and Image:
        try:
             ts_data = result['time_series']
             # Decimate frames if too many (limit to 20 frames)
             max_frames = 20
             if len(ts_data) > max_frames:
                 step_size = len(ts_data) // max_frames
                 full_ts = ts_data # Keep reference to original
                 ts_data = ts_data[::step_size]
                 
                 # Ensure last frame is included (compare times to avoid numpy ambiguous truth value error)
                 last_time = full_ts[-1]['time']
                 if not any(abs(frame['time'] - last_time) < 1e-6 for frame in ts_data):
                     ts_data.append(full_ts[-1])
             
             frames = []
             logger.info(f"  Generating Animation ({len(ts_data)} frames)...")
             
             # Extract surface mesh once to optimize loop
             surf_tris = None
             if 'elements' in result:
                 surf_tris = extract_surface_mesh(node_coords, np.array(result['elements']))
             
             for step_idx, step in enumerate(ts_data):
                 t_val = step['time']
                 temps = step['temperature'] - 273.15
                 
                 # Render Frame (Minimal overhead)
                 fig_anim = plt.figure(figsize=(6, 5))
                 ax_anim = fig_anim.add_subplot(111, projection='3d')
                 ax_anim.set_title(f"Time: {t_val:.2f}s", fontsize=10)
                 
                 # Surface Plot if possible, else scatter
                 tric = ax_anim.plot_trisurf(node_coords[:,0], node_coords[:,1], node_coords[:,2], 
                                      triangles=surf_tris,
                                      cmap='magma', shade=False, vmin=T_min, vmax=T_max)
                 
                 ax_anim.view_init(elev=30, azim=45 + (step_idx * 5)) # Rotating camera!
                 ax_anim.set_axis_off()
                 plt.tight_layout()
                 
                 # Save frame
                 buf = io.BytesIO()
                 plt.savefig(buf, format='png', dpi=80)
                 buf.seek(0)
                 frames.append(Image.open(buf))
                 plt.close(fig_anim)

             
             # Save GIF
             if frames:
                 gif_file = output_dir / f"{job_name}_anim.gif"
                 frames[0].save(
                     gif_file,
                     save_all=True,
                     append_images=frames[1:],
                     duration=150,
                     loop=0
                 )
                 logger.info(f"  Animation saved: {gif_file.name}")
                 
        except Exception as e:
             logger.warning(f"Animation generation failed: {e}")
             print(f"Animation generation failed: {e}")
    else:
        # LOGGING FOR DEBUGGING
        if 'time_series' not in result:
             msg = "[Debug] GIF Skipped: 'time_series' not in result"
             logger.info(msg); print(msg)
        elif not result['time_series']:
             msg = f"[Debug] GIF Skipped: 'time_series' is empty (len={len(result.get('time_series', []))})"
             logger.info(msg); print(msg)
        elif not Image:
             msg = "[Debug] GIF Skipped: PIL.Image is None (Import failed?)"
             logger.warning(msg); print(msg)

    # 2. PDF Report
    pdf_file = None
    if ThermalPDFReportGenerator:
         pdf_name = f"{job_name}_thermal_report.pdf"
         images = [str(png_file)]
         if ts_file: images.append(str(ts_file))
         
         try:
             generator = ThermalPDFReportGenerator()
             
             # Prepare data for thermal report
             t_min_k = T_min + 273.15
             t_max_k = T_max + 273.15
             
             report_data = {
                 'success': True,
                 'job_name': job_name,
                 'strategy_name': strategy_name,
                 'min_temp_k': float(t_min_k),
                 'max_temp_k': float(t_max_k),
                 'ambient_temp_c': float(amb_C),
                 'source_temp_c': float(src_temp_C),
                 'num_elements': result.get('num_elements', 0),
                 'num_nodes': len(node_coords),
                 'solve_time': result.get('solve_time', 0),
                 'heat_flux': result.get('heat_flux_watts'),
                 'convergence': convergence
             }
             
             pdf_path = generator.generate(
                 job_name=job_name,
                 output_dir=output_dir,
                 data=report_data,
                 image_paths=images
             )
             
             pdf_file = pdf_path
             logger.info(f"  Thermal PDF saved: {Path(pdf_file).name}")
         except Exception as e:
             err_msg = f"Thermal PDF failed: {e}\n{traceback.format_exc()}"
             logger.error(err_msg)
             print(err_msg)
             # Write to a special debug file if it fails
             try:
                 with open(output_dir / f"{job_name}_PDF_ERROR.txt", 'w') as f:
                     f.write(err_msg)
             except: pass
            
    return {
        'png': str(png_file), 
        'pdf': str(pdf_file) if pdf_file else None,
        'transient': str(ts_file) if ts_file else None,
        'animation': str(gif_file) if gif_file else None
    }


def export_vtk_with_temperature(result: Dict, output_file: str) -> str:
    """Export mesh with temperature for GUI visualization"""
    node_coords = result['node_coords']
    elements = result['elements']
    temperature = result['temperature']
    
    with open(output_file, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("SimOps Thermal Result\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        
        f.write(f"POINTS {len(node_coords)} float\n")
        for p in node_coords:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            
        num_cells = len(elements)
        f.write(f"\nCELLS {num_cells} {num_cells * 5}\n")
        for elem in elements:
            f.write(f"4 {elem[0]} {elem[1]} {elem[2]} {elem[3]}\n")
            
        f.write(f"\nCELL_TYPES {num_cells}\n")
        for _ in range(num_cells):
            f.write("10\n")
            
        f.write(f"\nPOINT_DATA {len(temperature)}\n")
        f.write("SCALARS Temperature float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for t in temperature:
            f.write(f"{t:.4f}\n")
            
    return output_file


def export_vtk_structural(result: Dict, output_file: str) -> str:
    """Export structural results (Displacement, Stress) to VTK"""
    node_coords = result['node_coords']
    elements = result['elements']
    disp = result.get('displacement') # shape (N, 3)
    stress = result.get('stress') # shape (N, 6) or (N, 1) von mises?
    von_mises = result.get('von_mises')
    
    with open(output_file, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("SimOps Structural Result\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        
        f.write(f"POINTS {len(node_coords)} float\n")
        # Apply displacement scale 1.0 for visualization? Or store deformed?
        # VTK usually stores undeformed and Vectors.
        for p in node_coords:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            
        num_cells = len(elements)
        f.write(f"\nCELLS {num_cells} {num_cells * 5}\n")
        
        # Tag skipping and 0-indexing
        has_tag = elements.shape[1] in [5, 11]
        start_col = 1 if has_tag else 0
        
        # Check if indices are 0-based (min 0) or 1-based (min >= 1)
        # Safe check using a sample
        sample_min = np.min(elements[:, start_col:start_col+4])
        is_one_based = sample_min >= 1
        
        for elem in elements:
            # Get node indices
            nodes = elem[start_col:start_col+4]
            # Convert to 0-based if needed
            if is_one_based:
                nodes = nodes - 1
            e0, e1, e2, e3 = nodes
            f.write(f"4 {int(e0)} {int(e1)} {int(e2)} {int(e3)}\n")
            
        f.write(f"\nCELL_TYPES {num_cells}\n")
        for _ in range(num_cells):
            f.write("10\n") # Tet4
            
        f.write(f"\nPOINT_DATA {len(node_coords)}\n")
        
        if disp is not None:
            f.write("VECTORS Displacement float\n")
            for d in disp:
                f.write(f"{d[0]:.6f} {d[1]:.6f} {d[2]:.6f}\n")
                
        if von_mises is not None:
            f.write("SCALARS VonMisesStress float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for s in von_mises:
                f.write(f"{s:.6f}\n")
                
    return output_file


    return output_file


def generate_structural_viz(
    result: Dict,
    output_path: Path
) -> bool:
    """
    Generates a high-quality isometric view of Von Mises Stress.
    Returns True if successful.
    """
    if not HAVE_VTK:
        return False
        
    try:
        # Create PyVista PolyData
        node_coords = result['node_coords']
        elements = result['elements']
        von_mises = result.get('von_mises')
        
        if von_mises is None:
            return False
            
        n_faces = len(elements) * 4 # Tet4 has 4 faces
        # Actually, simpler to just use UnstructuredGrid for volume rendering? 
        # Or Surface. Let's use the same surface extraction logic as Thermal for consistency?
        # Extract surface mesh
        surf_idx = extract_surface_mesh(node_coords, elements)
        
        # Build PolyData for surface
        # PyVista/VTK format for triangles: [3, p0, p1, p2, 3, ...]
        faces_flat = np.column_stack((
            np.full(len(surf_idx), 3),
            surf_idx
        )).flatten()
        
        cloud = pv.PolyData(node_coords, faces_flat)
        # result['von_mises'] is already in MPa
        cloud.point_data['Von Mises Stress (MPa)'] = von_mises
        
        # Setup Plotter
        pl = pv.Plotter(off_screen=True, window_size=[1024, 768])
        pl.set_background('white')
        
        pl.add_mesh(cloud, 
                   scalars='Von Mises Stress (MPa)',
                   cmap='jet', 
                   smooth_shading=False, 
                   show_edges=False,
                   scalar_bar_args={
                       'title': 'Von Mises (MPa)',
                       'n_labels': 5,
                       'fmt': '%.1f',
                       'color': 'black',
                       'title_font_size': 14,
                       'label_font_size': 10,
                       'position_x': 0.78,
                       'width': 0.18
                   })
        
        # FIX: Ensure scale is correct for engineering units
        pl.enable_parallel_projection()
        pl.view_isometric()
        pl.camera.zoom(1.2)
        
        # Log bounds for debugging aspect ratio
        b = cloud.bounds
        logger.info(f"[Viz] Bounds: X[{b[0]:.3f}, {b[1]:.3f}] Y[{b[2]:.3f}, {b[3]:.3f}] Z[{b[4]:.3f}, {b[5]:.3f}]")
        
        # Add scalar bar title
        # pl.add_scalar_bar(title="Von Mises Stress (MPa)")
        
        pl.screenshot(str(output_path), transparent_background=True)
        pl.close()
        
        return True
        
    except Exception as e:
        logger.warning(f"VTK Structural Render failed: {e}")
        return False


def run_structural_solver(mesh_file: Path, output_dir: Path, strategy_name: str, sim_config: Optional['SimulationConfig'] = None) -> Dict:
    """Run Structural Solver via CalculiX"""
    logger.log_stage("Solving (Structural)")
    logger.log_metadata("solver", "CalculiX")
    
    ccx_path = 'ccx'
    if sim_config and hasattr(sim_config, 'physics'):
        # Log basic physics metadata
        if hasattr(sim_config.physics, 'material'):
            logger.log_metadata("material", sim_config.physics.material)
        if hasattr(sim_config.physics, 'gravity_load_g'):
            logger.log_metadata("gravity", f"{sim_config.physics.gravity_load_g}g")
            
        ccx_path = getattr(sim_config.physics, 'ccx_path', 'ccx')
        ccx_path = getattr(sim_config.physics, 'ccx_path', 'ccx')
        if not ccx_path: ccx_path = 'ccx'
        
        # Ensure absolute path if it exists locally
        if os.path.exists(ccx_path):
            ccx_path = os.path.abspath(ccx_path)
        
        # Check if exists, if not, try relative to worker directory
        if not os.path.exists(ccx_path):
             # Try determining standard locations
             # 1. Relative to this script
             worker_dir = Path(__file__).parent
             candidate = worker_dir / ccx_path
             if candidate.exists():
                 ccx_path = str(candidate)
             elif (worker_dir / "ccx_wsl.bat").exists() and "wsl" in ccx_path:
                  ccx_path = str(worker_dir / "ccx_wsl.bat")
        
        logger.info(f"[Config] Using CalculiX binary from config: {ccx_path}")
    else:
        logger.info(f"[Config] No ccx_path in config, using default: {ccx_path}")
    
    adapter = CalculiXStructuralAdapter(ccx_binary=ccx_path)
    
    # Config - with material library hydration
    adapter_config = {}
    if sim_config and hasattr(sim_config, 'physics'):
        phy = sim_config.physics
        
        # Material hydration (same pattern as thermal solver)
        from core.materials import get_material
        
        def get_structural_prop(field_name, mat_attr, default):
            """Get property from explicit config, material library, or default."""
            val = getattr(phy, field_name, None)
            if val is not None:
                return val
            # Try material library
            if phy.material:
                try:
                    mat = get_material(phy.material)
                    lib_val = getattr(mat, mat_attr, None)
                    if lib_val is not None:
                        return lib_val
                except KeyError:
                    pass
            return default
        
        # Get elastic modulus from library (stored in Pa) and convert to MPa
        E_pa = get_structural_prop('youngs_modulus', 'elastic_modulus', 210000e6)  # Default: Steel in Pa
        # If from config, assume already in MPa; if from library (very large number), convert
        if E_pa is not None and E_pa > 1e6:  # Likely in Pa from library
            adapter_config['youngs_modulus'] = E_pa / 1e6  # Convert Pa to MPa
        else:
            adapter_config['youngs_modulus'] = E_pa if E_pa else 210000.0  # Already MPa or default
            
        adapter_config['poissons_ratio'] = get_structural_prop('poissons_ratio', 'poisson_ratio', 0.3)
        adapter_config['density'] = get_structural_prop('density', 'density', 7850.0)
        adapter_config['gravity_load_g'] = phy.gravity_load_g
        adapter_config['tip_load'] = getattr(phy, 'tip_load', None)
        
        # FIX: Map Clamping Direction (Critical for stability) and Material Name
        if hasattr(phy, 'clamping_direction'):
            adapter_config['clamping_direction'] = phy.clamping_direction
        if hasattr(phy, 'material'):
            adapter_config['material'] = phy.material
            
    # Run
    try:
        # [Fix] Ensure file is flushed and accessible
        import time
        time.sleep(2.0)
        
        if not os.path.exists(mesh_file):
            logger.error(f"Mesh file disappeared: {mesh_file}")
            return {'success': False, 'error': "Mesh file missing"}
            
        size = os.path.getsize(mesh_file)
        logger.info(f"Mesh file ready: {mesh_file} ({size} bytes)")

        result = adapter.run(mesh_file, output_dir, adapter_config)
        return result
    except Exception as e:
        logger.error(f"Structural Solver Failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}




def run_cfd_simulation(
    file_path: str,
    output_dir: Path,
    job_name: str,
    sim_config: Optional['SimulationConfig'],
    update_status_callback=None
) -> SimulationResult:
    """
    Dedicated pipeline for CFD simulations suitable for OpenFOAM.
    """
    # 0. Pre-Flight Validation via Pre-Sim Checks
    if not is_mesh_file(file_path) and run_pre_simulation_checks:
        if update_status_callback: update_status_callback("Validating geometry...")
        # Run comprehensive pre-sim checks including geometry validation
        passed, check_message = run_pre_simulation_checks(
            job_name=job_name,
            output_dir=output_dir,
            mesh_file=Path("placeholder"),  # Mesh doesn't exist yet
            config=sim_config.model_dump() if hasattr(sim_config, 'model_dump') else {},
            case_dir=None,  # Case doesn't exist yet
            expected_patches=None,
            cad_file=file_path  # Pass CAD file for geometry validation
        )
        
        if not passed:
            logger.error(f"Pre-simulation checks failed:\n{check_message}")
            return SimulationResult(
                success=False, 
                strategy_name="CFD_HighFi", 
                error=f"Pre-simulation validation failed:\n{check_message}"
            )
        else:
            logger.info(f"Pre-simulation checks passed:\n{check_message}")
    
    # 1. Mesh
    if is_mesh_file(file_path):
        logger.info(f"  [Direct] Skipping CFD meshing, using {file_path}")
        mesh_file = file_path
    else:
        if update_status_callback: update_status_callback("Meshing (CFD)...")
        
        # We use HighFi_Layered strategy by default for CFD
        strategy = next((s for s in STRATEGIES if s['name'] == 'HighFi_Layered'), STRATEGIES[0])
        
        try:
            mesh_file = generate_mesh_with_strategy(
                file_path, 
                output_dir, 
                strategy, 
                sim_config=sim_config
            )
        except Exception as e:
            logger.error(f"CFD Meshing failed: {e}")
            
            # If meshing failed and we had validation issues, try with healing
            if 'validation_result' in locals() and not validation_result.is_valid:
                logger.info("Meshing failed - retrying with geometry healing...")
                # The healing is handled in generate_mesh_with_strategy via CFDMeshStrategy
                # For now, we just report the original error
            
            return SimulationResult(success=False, strategy_name="CFD_HighFi", error=str(e))

    # 2. Solve
    if update_status_callback: update_status_callback("Solving (OpenFOAM)...")
    
    solver = CFDSolver()
    
    # Config for solver
    solver_config = {
        'u_inlet': [1.0, 0.0, 0.0], # Default, override from config
        'kinematic_viscosity': 1e-5
    }
    
    if sim_config and hasattr(sim_config, 'physics'):
        phy = sim_config.physics
        # Map physics props
        # Looking for fluid props or generic props
        if hasattr(phy, 'inlet_velocity'): 
            # If scalar, assume X direction
            v = phy.inlet_velocity
            solver_config['u_inlet'] = [v, 0.0, 0.0] if isinstance(v, (int, float)) else v
            
        if hasattr(phy, 'kinematic_viscosity'):
            solver_config['kinematic_viscosity'] = phy.kinematic_viscosity
            
        # Log CFD Metadata
        if hasattr(phy, 'inlet_velocity'):
             logger.log_metadata("inlet_velocity", f"{phy.inlet_velocity} m/s")
        if hasattr(phy, 'material'):
             logger.log_metadata("fluid", phy.material)
    
    # =========================================================================
    # PRE-SIMULATION ROBUSTNESS CHECKS
    # =========================================================================
    if run_pre_simulation_checks:
        if update_status_callback: update_status_callback("Running pre-simulation checks...")
        
        # Determine expected patches from tagging rules
        expected_patches = ['inlet', 'outlet']  # Default expectations
        if sim_config and hasattr(sim_config, 'tagging_rules'):
            for rule in sim_config.tagging_rules:
                tag_name = rule.get('tag_name', '') if isinstance(rule, dict) else getattr(rule, 'tag_name', '')
                if 'inlet' in tag_name.lower():
                    expected_patches.append(tag_name)
                elif 'outlet' in tag_name.lower():
                    expected_patches.append(tag_name)
        
        # Get config dict for validation
        config_dict = {}
        if sim_config:
            if hasattr(sim_config, 'model_dump'):
                config_dict = sim_config.model_dump()
            elif hasattr(sim_config, 'dict'):
                config_dict = sim_config.dict()
            elif hasattr(sim_config, '__dict__'):
                config_dict = vars(sim_config)
        
        passed, validation_msg = run_pre_simulation_checks(
            job_name=job_name,
            output_dir=output_dir,
            mesh_file=Path(mesh_file),
            config=config_dict,
            case_dir=None,  # Case dir created by solver, check after setup
            expected_patches=expected_patches
        )
        
        if not passed:
            logger.error(f"Pre-simulation validation FAILED: {validation_msg}")
            return SimulationResult(
                success=False,
                strategy_name=strategy['name'],
                error=f"Pre-simulation validation failed. Check crash log."
            )
        else:
            logger.info(f"[VALIDATION] {validation_msg}")
            
    try:
        # Run Solver
        try:
            result_data = solver.run(Path(mesh_file), output_dir, solver_config)
        except Exception as e:
            error_message = str(e)
            logger.warning(f"CFD Solver failed: {error_message}")
            logger.warning("Proceeding with Mesh-Only visualization/report.")
            
            # Fallback Result Data
            mesh_vtk = Path(mesh_file).with_suffix('.vtk')
            if not mesh_vtk.exists():
                 raise RuntimeError(f"Solver failed and Mesh VTK missing: {e}")
                 
            result_data = {
                 'vtk_file': str(mesh_vtk),
                 'solve_time': 0,
                 'converged': False,
                 'reynolds': 'N/A',
                 'cd': 'N/A',
                 'cl': 'N/A',
                 'turbulence_model': 'N/A (Solver Failed)',
                 'error_message': error_message,
                 'failure_reason': error_message
            }
        
        # Extract VTK path for visualization
        vtk_path = Path(result_data.get('vtk_file', mesh_file)).with_suffix('.vtk')
        
        # [Fix] Check if we can find a better result file (OpenFOAM 'internal.vtu')
        # The default vtk_file might be just the mesh if the solver didn't update it.
        # We look for the case directory and the latest time step.
        case_dir_name = f"{job_name}_{strategy['name']}_case"
        case_dir = output_dir / case_dir_name
        
        if case_dir.exists():
            # Look for VTK/Case_Time/internal.vtu
            vtk_root = case_dir / "VTK"
            if vtk_root.exists():
                candidates = list(vtk_root.rglob("internal.vtu"))
                if candidates:
                    # Sort candidates by time (assumed to be in parent folder name suffix)
                    def extract_time_from_vtu(p):
                        try:
                            # Folder format: {CASE}_{TIME} e.g. Cube_HighFi_Layered_case_560
                            # We grab the part after the last underscore
                            parts = p.parent.name.split('_')
                            return float(parts[-1])
                        except:
                            return -1.0
                    
                    candidates.sort(key=extract_time_from_vtu)
                    best_vtu = candidates[-1] # Last one is latest time
                    
                    logger.info(f"  [Output Recovery] Found OpenFOAM solution at: {best_vtu}")
                    vtk_path = best_vtu
        
        if not vtk_path.exists():
            # Fallback to mesh file converted path
            vtk_path = Path(mesh_file).with_suffix('.vtk')
        
        # 2.5 Post-Simulation Validation
        if update_status_callback: update_status_callback("Validating solution...")
        
        solution_validator = SolutionValidator(verbose=True)
        solution_validation = solution_validator.validate_cfd_solution(
            str(vtk_path),
            expected_fields=['U', 'p']  # Velocity and pressure
        )
        
        if not solution_validation.is_valid:
            logger.error(f"Solution validation failed:\n{solution_validation.get_error_summary()}")
            
            # Continue anyway for now - at least try to generate report with what we have
            # Future enhancement: retry with coarser mesh
            logger.warning("Proceeding with visualization despite validation failures...")
        else:
            logger.info(f"Solution validation passed: {solution_validation.data_quality_score:.1%} data quality")
        
        # Add validation results to result_data for reporting
        result_data['solution_quality'] = solution_validation.data_quality_score
        result_data['validation_issues'] = len(solution_validation.issues)
        
        # 3. Visualization
        if update_status_callback: update_status_callback("Visualizing...")
        
        # We will collect all generated image paths
        all_viz_paths = []
        viz_path = output_dir / f"{job_name}_velocity.png" # Fallback/Primary path name
        viz_success = False

        # Try Multi-Angle Viz First (High Quality)
        try:
            logger.info("Generating High-Quality 3D Streamlines (Multi-Angle)...")
            generated_paths = generate_multi_angle_streamlines(
                vtk_path=str(vtk_path), 
                output_dir=output_dir,
                job_name=job_name
            )
            
            if generated_paths:
                all_viz_paths = generated_paths
                viz_success = True
                # Set primary path to the first one (likely iso) for metadata
                viz_path = Path(generated_paths[0]) 
                logger.info(f"Generated {len(all_viz_paths)} streamline images.")
                
        except Exception as e:
            logger.warning(f"Multi-Angle viz failed: {e}")

        # Also generate velocity contour slices (planar cuts showing velocity heatmap)
        try:
            logger.info("Generating Velocity Contour Slices (Cross-Sectional Heatmaps)...")
            contour_paths = generate_velocity_contour_slices(
                vtk_path=str(vtk_path),
                output_dir=output_dir,
                job_name=job_name,
                planes=['xy', 'xz', 'yz']
            )
            
            if contour_paths:
                all_viz_paths.extend(contour_paths)
                logger.info(f"Generated {len(contour_paths)} contour slice images.")
                
        except Exception as e:
            logger.warning(f"Contour slice viz failed: {e}")

        # Fallback to internal velocity visualization if simple
        if not viz_success:
            logger.info("Falling back to internal velocity visualization...")
            from core.reporting.velocity_viz import generate_velocity_streamlines, generate_mesh_viz
            
            # This generates a single image
            viz_success = generate_velocity_streamlines(str(vtk_path), str(viz_path), title=f"Flow: {job_name}")
            
            if viz_success:
                all_viz_paths = [str(viz_path)]
            else:
                 logger.warning("Velocity visualization failed (or no data). Falling back to Mesh Viz.")
                 generate_mesh_viz(str(vtk_path), str(viz_path), title=f"Mesh: {job_name} (Solver Skipped/Failed)")
                 if viz_path.exists():
                     all_viz_paths = [str(viz_path)]
                     viz_success = True

        # 4. Report

        if update_status_callback: update_status_callback("Generating Report...")
        
        # Prepare report data
        converged = result_data.get('converged', False)  # Default to False (assume failure unless proven)
        # Update status string for clearer reporting if failed
        report_status = "CONVERGED" if converged else ("SKIPPED" if result_data['solve_time']==0 else "DIVERGED")

        report_data = {
            'converged': converged,
            'status_override': report_status,
            'solve_time': result_data.get('solve_time', 0),
            'reynolds_number': result_data.get('reynolds', 'N/A'),
            'drag_coefficient': result_data.get('cd', 'N/A'),
            'lift_coefficient': result_data.get('cl', 'N/A'),
            'viscosity_model': result_data.get('turbulence_model', 'Laminar'),
            'u_inlet': solver_config['u_inlet'][0],
            'mesh_cells': result_data.get('num_cells', 0),  # Now extracted from log.checkMesh
            'strategy_name': strategy['name']
        }
        
        pdf_gen = CFDPDFReportGenerator()
        pdf_file = pdf_gen.generate(
            job_name=job_name,
            output_dir=output_dir,
            data=report_data,
            image_paths=all_viz_paths
        )
        
        return SimulationResult(
            success=True,
            strategy_name=strategy['name'],
            mesh_file=str(mesh_file),
            vtk_file=str(vtk_path),
            png_file=str(viz_path),
            report_file=str(pdf_file),
            solve_time=result_data.get('solve_time', 0)
        )
    except Exception as e:
        logger.exception(f"CFD Solver/Report failed: {e}")
        return SimulationResult(success=False, strategy_name="CFD_OpenFOAM", error=str(e))
    finally:
        # LOG CFD METRICS
        if 'result_data' in locals():
            logger.info("-" * 40)
            logger.info(f"   RESULTS SUMMARY: {job_name}")
            logger.info(f"   Converged: {result_data.get('converged', 'N/A')}")
            # Show failure reason prominently if simulation failed
            if not result_data.get('converged', False) and 'error_message' in result_data:
                logger.error(f"   FAILURE: {result_data['error_message']}")
            logger.info(f"   Solve Time: {result_data.get('solve_time', 0):.2f} s")
            logger.info(f"   Reynolds:   {result_data.get('reynolds', 'N/A')}")
            logger.info(f"   Mesh Cells: {result_data.get('num_cells', 0):,}")
            logger.info(f"   Turbulence: {result_data.get('turbulence_model', 'N/A')}")
            # If using fallback, these will be N/A. If OpenFOAM runs, we might see values if parsed.
            logger.info("-" * 40)



def run_simulation(file_path: str, output_dir: str, config_path: Optional[str] = None) -> SimulationResult:
    """
    Main entry point for SimOps Worker.
    Orchestrates the complete simulation pipeline.
    
    Args:
        file_path: Path to CAD file
        output_dir: Path to directory for results
        config_path: Optional explicit path to sidecar JSON. 
                     If None, looks for [file_path].json
    """
    # 0. Setup Output Directory & Console Capture
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Identfy job name for the log file
    job_name = Path(file_path).stem.replace("USED_", "")
    log_file = output_path / f"{job_name}.log"
    
    # Wrap entire execution in capture
    with ConsoleCapturer(log_file):
        return _run_simulation_internal(file_path, output_path, config_path)

def _run_simulation_internal(file_path: str, output_path: Path, config_path_str: Optional[str]) -> SimulationResult:
    """Internal logic to allow wrapping"""
    job = get_current_job()
    def update_job_status(status: str):
        if job:
            job.meta['status'] = status
            job.save_meta()
            logger.info(f"[RQ] Status: {status}")

    update_job_status("Initializing...")
    start_time = time.time()
    
    # Path handling
    input_path = Path(file_path)
    # output_path is passed in
    
    # Job ID logic
    job_name = input_path.stem
    # If file was renamed to USED_name, strip the prefix for the Job Name for cleanliness
    if job_name.startswith("USED_"):
        job_name = job_name.replace("USED_", "", 1)
    
    # ---------------------------------------------------------
    # 1. Load Configuration
    # ---------------------------------------------------------
    if config_path_str:
        config_path = Path(config_path_str)
        logger.info(f"Using explicit sidecar config: {config_path.name}")
    else:
        config_path = input_path.with_suffix('.json')
    
    from core.config_loader import load_simops_config
    
    sim_config = None
    if config_path.exists():
        logger.info(f"Found sidecar config: {config_path.name}")
        sim_config = load_simops_config(str(config_path))
    else:
        logger.info("No sidecar config found. Using Golden Template defaults.")
        sim_config = load_simops_config(None) 
        
    # FIX: ALWAYS use the filename as job_name to prevent stale sidecar values.
    # The sidecar's job_name is ignored for output naming (was causing cube/cylinder mixups).
    # We just log a warning if it differs, for user awareness.
    if sim_config.job_name and sim_config.job_name != job_name:
        logger.warning(f"Sidecar job_name='{sim_config.job_name}' differs from file '{job_name}'. Using filename.")
    # job_name stays as input_path.stem (set above)

    logger.log_stage("Initializing")
    logger.log_metadata("job_name", job_name)
    if config_path.exists():
        logger.info(f"   [Config] Loaded from: {config_path.absolute()}")
    else:
        logger.info(f"   [Config] No sidecar found. Using internal defaults.")


    # DISPATCH: Check for CFD type
    sim_type = "thermal"
    if sim_config and hasattr(sim_config, 'physics'):
        sim_type = getattr(sim_config.physics, 'simulation_type', 'thermal')
    
    print(f"[DEBUG] Dispatch: Job={job_name}, Config={config_path}, Type={sim_type}")
    logger.info(f"   [Dispatch] Simulation Type: {sim_type.upper()}")
    if not config_path.exists():
        logger.warning(f"   [Dispatch] No sidecar for {job_name}, defaulting to {sim_type}")

    if sim_type == 'cfd':
        logger.info(">>> DISPATCHING TO CFD PIPELINE <<<")
        result = run_cfd_simulation(
            file_path, output_path, job_name, sim_config, update_job_status
        )
        
        # Dispatch result (shared logic)
        metadata = {
            'job_name': job_name,
            'input_file': file_path,
            'strategy': result.strategy_name,
            'success': result.success,
            'vtk_file': result.vtk_file,
            'report_file': result.report_file,
            'completed_at': datetime.now().isoformat(),
            'error': result.error,
        }
        meta_file = output_path / f"{job_name}_result.json"
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        try:
             dispatcher = ResultDispatcher(output_path)
             dispatcher.dispatch_result(metadata)
        except: pass
            
        return result

    # --- Standard Thermal Cascade ---
    logger.info(f"Physics: Material={sim_config.physics.material}, "
                f"Heat={sim_config.physics.heat_load_watts}W")
        
    is_mesh = is_mesh_file(file_path)
    if is_mesh:
        logger.info(f"Direct mesh detected: {input_path.name}. Skipping strategy cascade.")
        strategies_to_try = [{
            "name": "Direct_Mesh", 
            "description": "User-provided mesh",
            "skip_meshing": True
        }]
    else:
        strategies_to_try = STRATEGIES
    
    try:
        # Pre-analyze geometry (Only for CAD)
        if not is_mesh:
            geo_analysis = analyze_cad_geometry(file_path)
            logger.info("[Geometry Analysis]")
            logger.info(f"  Diagonal: {geo_analysis.diagonal:.2f} mm")
            logger.info(f"  Est. Wall: {geo_analysis.estimated_wall_thickness:.2f} mm")
            
            if geo_analysis.is_too_small_for_highfi:
                logger.warning("  [!] Geometry too small/thin for HighFi. Skipping to robust strategies.")
                strategies_to_try = [s for s in STRATEGIES if s['name'] != "HighFi_Layered"]
            elif sim_type in ['thermal', 'structural']:
                logger.warning(f"  [Config] Skipping HighFi_Layered for {sim_type} (Msg Connectivity Risk).")
                strategies_to_try = [s for s in STRATEGIES if s['name'] != "HighFi_Layered"]
        else:
            geo_analysis = None
            
    except Exception as e:
        logger.warning(f"Geometry analysis failed: {e}. Proceeding with full cascade.")
        geo_analysis = None

    failed_strategies = []

    for attempt_num, strategy in enumerate(strategies_to_try, 1):
        logger.info("")
        logger.info(f"ATTEMPT {attempt_num}/{len(STRATEGIES)}: {strategy['name']}")
        logger.info(f"  {strategy['description']}")
        logger.info("-" * 40)
        try:
            # Step 1: Generate mesh
            if strategy.get('skip_meshing'):
                logger.info(f"  [Direct] Using input file as mesh: {file_path}")
                mesh_file = file_path
            else:
                update_job_status(f"Meshing ({strategy['name']})...")
                mesh_file = generate_mesh_with_strategy(
                    file_path, 
                    output_path, 
                    strategy,
                    sim_config=sim_config
                )
            
            # Step 2: Run Solver (Dispatch)
            sim_type = "thermal"
            if sim_config and hasattr(sim_config, 'physics'):
                sim_type = getattr(sim_config.physics, 'simulation_type', "thermal")

            update_job_status(f"Solving ({strategy['name']}) [{sim_type}]...")
            
            if sim_type == "structural":
                # --- STRUCTURAL SIMULATION ---
                result = run_structural_solver(Path(mesh_file), output_path, strategy['name'], sim_config)
                
                if not result.get('success', False):
                    raise RuntimeError(f"Structural Solver failed: {result.get('error')}")
                    
                # Export VTK
                vtk_file = output_path / f"{job_name}_structural.vtk"
                export_vtk_structural(result, str(vtk_file))
                
                # Visualize (Stress)
                png_file = output_path / f"{job_name}_stress.png"
                has_viz = generate_structural_viz(result, png_file)
                
                # Report
                report_file = None
                if generate_structural_report:
                    try:
                        report_data = generate_structural_report(
                            result=result,
                            output_dir=output_path,
                            job_name=job_name,
                            g_factor=getattr(sim_config.physics, 'gravity_load_g', 1.0)
                        )
                        report_file = report_data.get('pdf_file')
                        png_file = report_data.get('png_file')
                        logger.info(f"   [Report] Saved to {report_file}")
                    except Exception as e:
                        logger.error(f"Structural Report Gen Failed: {e}", exc_info=True)
                        with open(output_path / "REPORT_ERROR.txt", "w") as f:
                            f.write(str(e))

                total_time = time.time() - start_time
                
                logger.info("")
                logger.info("=" * 60)
                logger.info(f"   [OK] STRUCTURAL SUCCESS with {strategy['name']}")
                logger.info("=" * 60)
                
                # Write metadata (Required for Dashboard/Traffic Light)
                metadata = {
                    'job_name': job_name,
                    'input_file': file_path,
                    'strategy': strategy['name'],
                    'attempts': attempt_num,
                    'mesh_file': str(mesh_file),
                    'vtk_file': str(vtk_file),
                    'png_file': str(png_file) if has_viz else None,
                    'pdf_file': report_file,
                    'max_stress_pa': float(np.max(result.get('von_mises', [0]))) * 1e6,
                    'max_stress_mpa': float(np.max(result.get('von_mises', [0]))),
                    'max_displacement_mm': float(np.max(result.get('displacement_magnitude', [0]))),
                    'num_elements': result.get('num_elements', 0),
                    'solve_time_s': result.get('solve_time', 0),
                    'total_time_s': total_time,
                    'completed_at': time.strftime("%Y-%m-%dT%H:%M:%S"),
                    'success': True,
                    'sim_type': 'structural'
                }

                meta_file = output_path / f"{job_name}_result.json"
                with open(meta_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # DISPATCH RESULTS
                try:
                    dispatcher = ResultDispatcher(output_path)
                    dispatcher.dispatch_result(metadata)
                    logger.info("   [Dispatch] Result sent to monitor.")
                except Exception as e:
                    logger.warning(f"   [Dispatch] Failed: {e}")

                return SimulationResult(
                    success=True,
                    strategy_name=strategy['name'],
                    mesh_file=str(mesh_file),
                    vtk_file=str(vtk_file),
                    png_file=str(png_file) if png_file else None, 
                    report_file=str(report_file) if report_file else None,
                    solve_time=result.get('solve_time', 0),
                    num_elements=result.get('num_elements', 0),
                    max_stress_mpa=float(np.max(result.get('von_mises', [0]))),
                    max_disp_mm=float(np.max(result.get('displacement_magnitude', [0])))
                )

            else:
                # --- THERMAL SIMULATION (Default) ---
                result = run_thermal_solver(Path(mesh_file), output_path, strategy['name'], sim_config)
                
                if not result.get('success', False):
                     raise RuntimeError(f"Thermal Solver failed: {result.get('error')}")

                # Step 3: Generate report (with error handling to not crash on report failures)
                try:
                    report_files = generate_report(job_name, output_path, result, strategy['name'], sim_config=sim_config, mesh_file=str(mesh_file))
                    png_file = report_files.get('png')
                    pdf_file = report_files.get('pdf')
                except Exception as e:
                    logger.error(f"Report generation failed (solver succeeded, results saved): {e}")
                    report_files = {}  # Empty dict to prevent downstream errors
                    png_file = None
                    pdf_file = None
                
                # Step 4: Export VTK for GUI
                vtk_file = output_path / f"{job_name}_thermal.vtk"
                export_vtk_with_temperature(result, str(vtk_file))
                
                # Success!
                total_time = time.time() - start_time
                
                logger.info("")
                logger.info("=" * 60)
                logger.info(f"   [OK] SUCCESS with {strategy['name']}")
                logger.info(f"   Total time: {total_time:.1f}s")
                logger.info("=" * 60)
                
                # Write metadata
                metadata = {
                    'job_name': job_name,
                    'input_file': file_path,
                    'strategy': strategy['name'],
                    'attempts': attempt_num,
                    'mesh_file': str(mesh_file),
                    'vtk_file': str(vtk_file),
                    'png_file': str(png_file) if png_file else None,
                    'pdf_file': str(pdf_file) if pdf_file else None,
                    'transient_png': report_files.get('transient') if report_files else None,
                    'animation_gif': report_files.get('animation') if report_files else None,
                    'min_temp_K': result.get('min_temp') or 0.0,
                    'max_temp_K': result.get('max_temp') or 0.0,
                    'num_elements': result.get('num_elements', 0),
                    'solve_time_s': result.get('solve_time', 0),
                    'total_time_s': total_time,
                    'completed_at': time.strftime("%Y-%m-%dT%H:%M:%S"),
                    'success': True,
                }

                # LOG METRICS FOR VERIFICATION
                logger.info("-" * 40)
                logger.info(f"   RESULTS SUMMARY: {job_name}")
                if 'max_stress' in metadata:
                    logger.info(f"   Max Stress:       {metadata['max_stress']:.4e} Pa ({(metadata['max_stress']/1e6):.2f} MPa)")
                if 'max_disp' in result:
                     logger.info(f"   Max Displacement: {result['max_disp']:.4e} m") # Check units, usually mm in adapter? Adapter result is mm?
                     # Adapter: 'max_disp': np.max(disp_mag)
                     # Adapter Units: mm usually? "Coordinates... * scale"
                     # Let's assume mm per CalculiX adapter comments found earlier.
                if 'reaction_force_z' in result:
                    logger.info(f"   Reaction Force Z: {result['reaction_force_z']:.4e} N")
                logger.info("-" * 40)

                
                meta_file = output_path / f"{job_name}_result.json"
                with open(meta_file, 'w') as f:
                    json.dump(metadata, f, indent=2)

                # DISPATCH RESULTS
                try:
                    dispatcher = ResultDispatcher(output_path)
                    dispatcher.dispatch_result(metadata)
                except Exception as e:
                    logger.error(f"Dispatch failed (but results are safe): {e}")
                    
                return SimulationResult(
                    success=True,
                    strategy_name=strategy['name'],
                    mesh_file=str(mesh_file),
                    vtk_file=str(vtk_file),
                    png_file=str(png_file),
                    report_file=str(pdf_file) if pdf_file else None,
                    min_temp=result.get('min_temp') or 0.0,
                    max_temp=result.get('max_temp') or 0.0,
                    num_elements=result.get('num_elements', 0),
                    solve_time=result.get('solve_time', 0),
                )
            
        except Exception as e:
            logger.error(f"[X] {strategy['name']} FAILED: {e}")
            failed_strategies.append({
                "name": strategy['name'],
                "error": str(e)
            })
            logger.info("Automatic retry triggered...")
            # traceback.print_exc() # Reduce noise
            continue
            
    # All strategies failed
    total_time = time.time() - start_time
    error_msg = "All automated strategies failed. Manual review needed."
    
    logger.error("")
    logger.error("=" * 60)
    logger.error(f"   [X] ALL STRATEGIES FAILED for {job_name}")
    logger.error(f"   {error_msg}")
    logger.error("=" * 60)
    
    # Write failure report
    failure_file = output_path / f"{job_name}_FAILED.txt"
    report_content = [
        "SIMOPS AUTOMATED PIPELINE - FAILURE REPORT",
        "=" * 50,
        f"Job Name: {job_name}",
        f"Input:    {file_path}",
        f"Time:     {datetime.now().isoformat()}",
        "=" * 50,
        "",
        "GEOMETRY ANALYSIS:",
        str(geo_analysis) if geo_analysis else "Not available",
        "",
        "STRATEGY FAILURES:",
    ]
    
    for fail in failed_strategies:
        report_content.append(f"- {fail['name']}: {fail['error']}")
        
    report_content.append("")
    report_content.append("SUGGESTIONS:")
    report_content.append("1. Check input CAD for validity (self-intersections, gaps).")
    report_content.append("2. Scale up model if units are incorrect (e.g. meters vs mm).")
    report_content.append("3. Try manual cleanup in CAD software.")
    
    with open(failure_file, 'w') as f:
        f.write("\n".join(report_content))

    logger.error(f"   Failure report written to: {failure_file}")

    # WRITE METADATA FOR FAILURE (Ensures result.json exists for UI/Watchers)
    metadata = {
        'job_name': job_name,
        'input_file': file_path,
        'strategy': "ALL_FAILED",
        'success': False,
        'error': error_msg,
        'failed_strategies': failed_strategies,
        'completed_at': datetime.now().isoformat(),
    }
    meta_file = output_path / f"{job_name}_result.json"
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    return SimulationResult(
        success=False,
        strategy_name="ALL_FAILED",
        error=error_msg,
    )


# For direct execution / testing
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="SimOps Worker - Run simulation")
    parser.add_argument("cad_file", help="Input CAD file")
    parser.add_argument("-o", "--output", default="./output", help="Output directory")
    parser.add_argument("-c", "--config", help="Explicit path to sidecar JSON config")
    
    args = parser.parse_args()
    
    result = run_simulation(args.cad_file, args.output, config_path=args.config)
    
    if result.success:
        print(f"\n[OK] SUCCESS: {result.strategy_name}")
        print(f"   PNG: {result.png_file}")
        print(f"   VTK: {result.vtk_file}")
        sys.exit(0)
    else:
        print(f"\n[X] FAILED: {result.error}")
        sys.exit(1)
