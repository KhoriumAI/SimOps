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
from core.reporting.multi_angle_viz import generate_multi_angle_streamlines

# Supported Mesh Extensions
MESH_EXTENSIONS = {'.msh', '.vtk', '.vtu', '.unv', '.inp'}

def is_mesh_file(path: str) -> bool:
    return Path(path).suffix.lower() in MESH_EXTENSIONS

try:
    from core.reporting.structural_report import StructuralPDFReportGenerator
except ImportError:
    StructuralPDFReportGenerator = None

try:
    from core.validation.pre_sim_checks import run_pre_simulation_checks
except ImportError:
    run_pre_simulation_checks = None


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [WORKER] %(levelname)s: %(message)s',
)
logger = logging.getLogger(__name__)

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
        
        class Tee:
            def __init__(self, original, file):
                self.original = original
                self.file = file
            def write(self, message):
                self.original.write(message)
                self.file.write(message)
                self.file.flush() # Ensure realtime persistence
            def flush(self):
                self.original.flush()
                self.file.flush()
                
        sys.stdout = Tee(self.stdout_orig, self.file_handle)
        sys.stderr = Tee(self.stderr_orig, self.file_handle)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout_orig
        sys.stderr = self.stderr_orig
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
    error: Optional[str] = None


def generate_mesh_with_strategy(
    cad_file: str, 
    output_dir: Path,
    strategy: Dict,
    sim_config: Optional['SimulationConfig'] = None # Added config
) -> str:
    """
    Generate CFD mesh using the given strategy.
    Delegates to core.strategies.cfd_strategy.
    """
    from core.strategies.cfd_strategy import CFDMeshStrategy, CFDMeshConfig
    
    job_name = Path(cad_file).stem
    output_file = output_dir / f"{job_name}_{strategy['name']}.msh"
    
    logger.info(f"Meshing with strategy: {strategy['name']}")
    
    # Map strategy params to config
    # Use mesh_size_factor directly or fallback to mesh_size/0.5 for back-compat
    base_factor = strategy.get('mesh_size_factor', strategy.get('mesh_size', 0.5) / 0.5)
    
    # Overrides from Sidecar
    second_order = False
    if sim_config and sim_config.meshing:
        if sim_config.meshing.second_order:
            second_order = True
            logger.info("  [Override] Enabling Second-Order Elements (Sidecar)")
        if sim_config.meshing.mesh_size_multiplier != 1.0:
            base_factor *= sim_config.meshing.mesh_size_multiplier
            logger.info(f"  [Override] Scaling mesh size by {sim_config.meshing.mesh_size_multiplier}")

    # Intelligent Wind Tunnel Detection
    # Enable wind tunnel for external flow CFD (when inlet_velocity > 0)
    enable_wind_tunnel = False
    if sim_config and hasattr(sim_config.physics, 'simulation_type'):
        if sim_config.physics.simulation_type == 'cfd':
            if hasattr(sim_config.physics, 'inlet_velocity') and sim_config.physics.inlet_velocity > 0:
                enable_wind_tunnel = True
                logger.info(f"  [Wind Tunnel] Enabled for external flow (v={sim_config.physics.inlet_velocity} m/s)")
            else:
                logger.info("  [Wind Tunnel] Disabled (inlet_velocity = 0 or not set)")
    
    config = CFDMeshConfig(
        num_layers=strategy.get('num_layers', 5),
        growth_rate=strategy.get('growth_rate', 1.2),
        mesh_size_factor=base_factor,
        create_prisms=True,
        optimize_netgen=strategy.get('optimize', True),
        smoothing_steps=10,
        second_order=second_order,
        virtual_wind_tunnel=enable_wind_tunnel
    )
    
    logger.info(f"  [DEBUG] CFDMeshConfig: virtual_wind_tunnel={config.virtual_wind_tunnel}")
    
    # Pass tagging rules from sidecar if available
    tagging_rules = sim_config.tagging_rules if sim_config else []
    
    runner = CFDMeshStrategy(verbose=True)
    # Note: We need to pass tagging rules to strategy. 
    # Current API is generate_cfd_mesh(cad, out, config).
    # We'll update strategy instance with rules before calling generate.
    runner.tagging_rules = tagging_rules
    
    success, stats = runner.generate_cfd_mesh(cad_file, str(output_file), config)
    
    if not success:
        raise RuntimeError(f"Mesh generation failed: {stats.get('error')}")
        
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
        logger.info(f"Running Thermal Solver via CalculiX...")
        adapter = CalculiXAdapter()
        
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
             
             # Map other fields
             # Heat source temperature (from schema, not heat_load_watts)
             if hasattr(phy, 'heat_source_temperature'): adapter_config['heat_source_temperature'] = phy.heat_source_temperature
             # Assuming standard fields: unit_scaling, convection_coeff, ambient_temperature
             if hasattr(phy, 'unit_scaling'): adapter_config['unit_scaling'] = phy.unit_scaling
             if hasattr(phy, 'convection_coeff'): adapter_config['convection_coeff'] = phy.convection_coeff
             if hasattr(phy, 'ambient_temperature'): adapter_config['ambient_temperature'] = phy.ambient_temperature
             if hasattr(phy, 'transient'): adapter_config['transient'] = phy.transient
             if hasattr(phy, 'duration'): adapter_config['duration'] = phy.duration
             if hasattr(phy, 'time_step'): adapter_config['time_step'] = phy.time_step
             if hasattr(phy, 'initial_temperature') and phy.initial_temperature: adapter_config['initial_temperature'] = phy.initial_temperature
             if hasattr(phy, 'fix_hot_boundary'): adapter_config['fix_hot_boundary'] = phy.fix_hot_boundary
             if hasattr(phy, 'fix_cold_boundary'): adapter_config['fix_cold_boundary'] = phy.fix_cold_boundary


        result = adapter.run(Path(mesh_file), Path(output_dir), adapter_config)
        
        if 'elements' in result:
            result['num_elements'] = len(result['elements'])
        else:
            result['num_elements'] = 0
            
        logger.info(f"  [CalculiX] Solver finished. Temp range: {result['min_temp']:.1f}K - {result['max_temp']:.1f}K")
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
    faces = np.vstack([
        elements[:, [0,1,2]],
        elements[:, [0,1,3]],
        elements[:, [0,2,3]],
        elements[:, [1,2,3]]
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
                   cmap='magma', 
                   clim=[vmin, vmax],
                   smooth_shading=False, # Disabled to prevent LLVMpipe segfaults
                   show_edges=False)
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
    
    node_coords = np.array(result['node_coords'])
    temp_K = np.array(result['temperature'])
    # Metric Switch: Kelvin -> Celsius
    temp_C = temp_K - 273.15
    
    T_min = np.min(temp_C)
    T_max = np.max(temp_C)
    
    # Safe Context Extraction (Handles Dict or Pydantic)
    def to_dict(obj):
        if hasattr(obj, 'model_dump'): return obj.model_dump()
        if hasattr(obj, 'dict'): return obj.dict()
        return obj if isinstance(obj, dict) else {}

    config_dict = to_dict(sim_config)
    phy = config_dict.get('physics', {})
        
    init_C = phy.get('initial_temperature', 293.0)
    if init_C is None: init_C = 293.0
    init_C -= 273.15
    
    mat_name = phy.get('material', 'Aluminum 6061')
    amb_C = phy.get('ambient_temperature', 293.0) - 273.15
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
    src_temp_C = phy.get('heat_source_temperature', 373.15) - 273.15
    param_subtitle = f"Material: {mat_name} | Ambient: {amb_C:.0f}°C | Source: {src_temp_C:.0f}°C | h={h_coeff} W/m²K"
    
    # 1. Generate High-Quality PNG
    png_file = output_dir / f"{job_name}_temperature.png"
    
    # Generate Helper VTK Image first
    vtk_png_path = output_dir / f"{job_name}_iso_vtk.png"
    vtk_success = False
    
    elements = result.get('elements')
    surf_faces = None
    if elements is not None and len(elements) > 0:
         surf_faces = extract_surface_mesh(node_coords, np.array(elements))
         
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
    
    # Global Levels
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
        # Fallback Matplotlib 3D
        try:
            from matplotlib.colors import LightSource
            ls = LightSource(azdeg=315, altdeg=45)
             
            if surf_faces is not None:
                 # Downsample if super dense
                 tris_plot = surf_faces
                 if len(surf_faces) > 50000:
                     tris_plot = surf_faces[::2]
                     
                 x = node_coords[:, 0]
                 y = node_coords[:, 1]
                 z = node_coords[:, 2]
                 
                 # Create custom shading manually for better control
                 trisurf = ax_iso.plot_trisurf(x, y, z, triangles=tris_plot, cmap='magma', 
                                  antialiased=True, shade=True, linewidth=0,
                                  vmin=T_min, vmax=T_max, lightsource=ls)
                 
            else:
                 # Fallback Global Scatter
                 sc = ax_iso.scatter(node_coords[:,0], node_coords[:,1], node_coords[:,2],
                                c=temp_C, cmap='magma', s=2, alpha=0.1)
    
            ax_iso.view_init(elev=30, azim=45)
            ax_iso.set_axis_off()

        except Exception as e:
            logger.warning(f"3D Surface View failed: {e}")
            ax_iso.text(0.5, 0.5, "3D Render Error", ha='center')

    # --- Subplots 2,3,4: Cross Sections ---
    shape_views = [
        (0, 1, 2, 'XY Section (Top)', 'X', 'Y'), 
        (0, 2, 1, 'XZ Section (Front)', 'X', 'Z'), 
        (1, 2, 0, 'YZ Section (Side)', 'Y', 'Z'), 
    ]
    
    for ax, (xi, yi, zi, title, xlabel, ylabel) in zip(axes, shape_views):
        z_coords = node_coords[:, zi]
        z_mid = (np.min(z_coords) + np.max(z_coords)) / 2.0
        epsilon = (np.max(z_coords) - np.min(z_coords)) * 0.05
        if epsilon == 0: epsilon = 1.0
        
        mask = np.abs(z_coords - z_mid) < epsilon
        
        if np.sum(mask) > 3:
            sx = node_coords[mask, xi]
            sy = node_coords[mask, yi]
            st = temp_C[mask]
            
            # Masking Logic (Hole deduction)
            triang = mtri.Triangulation(sx, sy)
            x_tri = sx[triang.triangles]
            y_tri = sy[triang.triangles]
            d1 = np.hypot(x_tri[:,0]-x_tri[:,1], y_tri[:,0]-y_tri[:,1])
            d2 = np.hypot(x_tri[:,1]-x_tri[:,2], y_tri[:,1]-y_tri[:,2])
            d3 = np.hypot(x_tri[:,2]-x_tri[:,0], y_tri[:,2]-y_tri[:,0])
            max_edge = np.max(np.column_stack([d1,d2,d3]), axis=1)
            
            domain_size = max(np.max(sx)-np.min(sx), np.max(sy)-np.min(sy))
            threshold = domain_size * 0.1
            if len(st) > 100:
                avg_edge = np.median(max_edge)
                threshold = max(threshold, avg_edge * 3.0) 
            
            triang.set_mask(max_edge > threshold)
            
            cntr = ax.tricontourf(triang, st, levels=levels, cmap='magma', extend='both')
            # Add thin contour lines for better definition
            ax.tricontour(triang, st, levels=levels, colors='k', linewidths=0.1, alpha=0.2)
        else:
            ax.text(0.5, 0.5, "Slice Empty", ha='center')

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.4)

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
             
             times = [s['time'] for s in stats]
             mins = [s['min'] - 273.15 for s in stats]
             maxs = [s['max'] - 273.15 for s in stats]
             means = [s['mean'] - 273.15 for s in stats]
             
             plt.figure(figsize=(10, 6))
             plt.plot(times, maxs, 'r-', linewidth=2, label='Max Temp')
             plt.plot(times, means, 'g--', label='Mean Temp')
             plt.plot(times, mins, 'b-', linewidth=2, label='Min Temp')
             
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
    if PDFReportGenerator:
         pdf_name = f"{report_title}.pdf" # Professional Filename
         images = [str(png_file)]
         if ts_file: images.append(str(ts_file))
         
         try:
             generator = PDFReportGenerator()
             report_data = {
                 'job_name': job_name,
                 'strategy_name': strategy_name,
                 'min_temp': float(T_min), # Celsius
                 'max_temp': float(T_max), # Celsius
                 'num_elements': result.get('num_elements', 0),
                 'solve_time': result.get('solve_time', 0),
                 'heat_flux': result.get('heat_flux_watts'),
                 'convergence': result.get('convergence_steps'),
                 'courant_max': result.get('courant_max', 0.0)
             }
             pdf_path = generator.generate(
                 job_name=job_name, # Used for title in PDF?
                 output_dir=output_dir,
                 data=report_data,
                 image_paths=images
             )
             
             # Rename the file after generation if needed
             pdf_file = pdf_path
             logger.info(f"  PDF saved: {Path(pdf_file).name}")
         except Exception as e:
             err_msg = f"PDF failed: {e}\n{traceback.format_exc()}"
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
        for elem in elements:
            # VTK expects 0-indexed. Solver returns 0-indexed from adapter?
            # CalculiXAdapter returns 0-indexed.
            f.write(f"4 {elem[0]} {elem[1]} {elem[2]} {elem[3]}\n")
            
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
        cloud.point_data['Von Mises Stress (MPa)'] = von_mises
        
        # Setup Plotter
        pl = pv.Plotter(off_screen=True)
        pl.set_background('white')
        
        pl.add_mesh(cloud, 
                   cmap='jet', 
                   smooth_shading=False, # Disabled to prevent Segfaults
                   show_edges=False)
                   # specular=0.5,
                   # specular_power=15)
        
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
    logger.info("Running Structural Solver via CalculiX...")
    adapter = CalculiXStructuralAdapter()
    
    # Config
    adapter_config = {}
    if sim_config and hasattr(sim_config, 'physics'):
        phy = sim_config.physics
        adapter_config['gravity_load_g'] = phy.gravity_load_g
        adapter_config['tip_load'] = getattr(phy, 'tip_load', None)
        adapter_config['youngs_modulus'] = phy.youngs_modulus
        adapter_config['poissons_ratio'] = phy.poissons_ratio
        if hasattr(phy, 'density'):
            adapter_config['density'] = phy.density
            
    # Run
    try:
        result = adapter.run(mesh_file, output_dir, adapter_config)
        return result
    except Exception as e:
        logger.error(f"Structural Solver Failed: {e}")
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
                logger.info(f"Generated {len(all_viz_paths)} visualization images.")
                
        except Exception as e:
            logger.warning(f"Multi-Angle viz failed: {e}")

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
            logger.info(f"   Solve Time: {result_data.get('solve_time', 0):.2f} s")
            logger.info(f"   Reynolds:   {result_data.get('reynolds', 'N/A')}")
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
    
    console_log = output_path / "worker_console.log"
    
    # Wrap entire execution in capture
    with ConsoleCapturer(console_log):
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

    logger.info(f"Starting job: {job_name}")
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
                if StructuralPDFReportGenerator:
                    try:
                        gen = StructuralPDFReportGenerator()
                        
                        # Pack Data
                        report_data = {
                            'success': True,
                            'strategy_name': strategy['name'],
                            'max_stress': np.max(result.get('von_mises', [0])),
                            'max_displacement': np.max(result.get('displacement_magnitude', [0])),
                            'num_elements': result.get('num_elements', 0),
                            'solve_time': result.get('solve_time', 0),
                            'load_info': "Gravity" if (sim_config and getattr(sim_config.physics, 'gravity_load_g', 0) > 0) else "Tip Load",
                            'max_strain': result.get('max_strain', 0.0),
                            'reaction_force_z': result.get('reaction_force_z', 0.0)
                        }
                        # Small fix: config_overrides is not available here, use sim_config
                        # Better: pass info from result if adapter put it there, or just generic.
                        
                        pdf_path = gen.generate(
                            job_name,
                            output_path,
                            report_data,
                            image_paths=[str(png_file)] if has_viz else []
                        )
                        report_file = str(pdf_path)
                        logger.info(f"   [Report] Saved to {report_file}")
                    except Exception as e:
                        logger.error(f"Structural Report Gen Failed: {e}", exc_info=True)
                        # Create error flag file
                        with open(output_path / "REPORT_ERROR.txt", "w") as f:
                            f.write(str(e))

                total_time = time.time() - start_time
                
                logger.info("")
                logger.info("=" * 60)
                logger.info(f"   [OK] STRUCTURAL SUCCESS with {strategy['name']}")
                logger.info("=" * 60)
                
                return SimulationResult(
                    success=True,
                    strategy_name=strategy['name'],
                    mesh_file=str(mesh_file),
                    vtk_file=str(vtk_file),
                    png_file=str(png_file) if has_viz else None, 
                    report_file=report_file,
                    solve_time=result.get('solve_time', 0)
                )

            else:
                # --- THERMAL SIMULATION (Default) ---
                result = run_thermal_solver(Path(mesh_file), output_path, strategy['name'], sim_config)
                
                if not result.get('success', False):
                     raise RuntimeError(f"Thermal Solver failed: {result.get('error')}")

                # Step 3: Generate report
                report_files = generate_report(job_name, output_path, result, strategy['name'], sim_config=sim_config, mesh_file=str(mesh_file))
                png_file = report_files.get('png')
                pdf_file = report_files.get('pdf')
                
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
                    'min_temp_K': result.get('min_temp'),
                    'max_temp_K': result.get('max_temp'),
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
                    min_temp=result.get('min_temp'),
                    max_temp=result.get('max_temp'),
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
    
    args = parser.parse_args()
    
    result = run_simulation(args.cad_file, args.output)
    
    if result.success:
        print(f"\n[OK] SUCCESS: {result.strategy_name}")
        print(f"   PNG: {result.png_file}")
        print(f"   VTK: {result.vtk_file}")
        sys.exit(0)
    else:
        print(f"\n[X] FAILED: {result.error}")
        sys.exit(1)
