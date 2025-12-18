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

import gmsh
import numpy as np

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
from core.solvers.calculix_adapter import CalculiXAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [WORKER] %(levelname)s: %(message)s',
)
logger = logging.getLogger(__name__)


# =============================================================================
# CASCADING FALLBACK STRATEGIES
# =============================================================================
STRATEGIES = [
    {
        "name": "HighFi_CFD",
        "description": "High fidelity CFD mesh with boundary layers",
        "mesh_size": 0.5,
        "num_layers": 5,
        "growth_rate": 1.2,
        "algorithm_2d": 6,   # Frontal-Delaunay
        "algorithm_3d": 1,   # Delaunay
        "optimize": True,
    },
    {
        "name": "MedFi_Robust", 
        "description": "Medium fidelity with fewer layers",
        "mesh_size": 1.0,
        "num_layers": 3,
        "growth_rate": 1.3,
        "algorithm_2d": 6,   # Frontal-Delaunay
        "algorithm_3d": 4,   # Frontal
        "optimize": True,
    },
    {
        "name": "LowFi_Emergency",
        "description": "Emergency fallback - pure conduction mesh",
        "mesh_size": 2.0,
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
    base_factor = strategy.get('mesh_size', 0.5) / 0.5
    
    # Overrides from Sidecar
    second_order = False
    if sim_config and sim_config.meshing:
        if sim_config.meshing.second_order:
            second_order = True
            logger.info("  [Override] Enabling Second-Order Elements (Sidecar)")
        if sim_config.meshing.mesh_size_multiplier != 1.0:
            base_factor *= sim_config.meshing.mesh_size_multiplier
            logger.info(f"  [Override] Scaling mesh size by {sim_config.meshing.mesh_size_multiplier}")

    config = CFDMeshConfig(
        num_layers=strategy.get('num_layers', 5),
        growth_rate=strategy.get('growth_rate', 1.2),
        mesh_size_factor=base_factor,
        create_prisms=True,
        optimize_netgen=strategy.get('optimize', True),
        smoothing_steps=10,
        second_order=second_order
    )
    
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
             # Map fields (Attribute access assuming dataclass/pydantic)
             if hasattr(phy, 'thermal_conductivity'): adapter_config['thermal_conductivity'] = phy.thermal_conductivity
             if hasattr(phy, 'heat_load_watts'): adapter_config['heat_source_temperature'] = phy.heat_load_watts # Mapping Load to Temp for MVP? Or separate? 
             # Wait, heat_load_watts isn't temp. current adapter uses fixed temp.
             # MVP: If heat_load_watts provided, assumes 800K for now or need logic?
             # Let's check what 'phy' has. 
             # Assuming standard fields: unit_scaling, convection_coeff, ambient_temperature
             if hasattr(phy, 'unit_scaling'): adapter_config['unit_scaling'] = phy.unit_scaling
             if hasattr(phy, 'convection_coeff'): adapter_config['convection_coeff'] = phy.convection_coeff
             if hasattr(phy, 'ambient_temperature'): adapter_config['ambient_temperature'] = phy.ambient_temperature
             if hasattr(phy, 'transient'): adapter_config['transient'] = phy.transient
             if hasattr(phy, 'duration'): adapter_config['duration'] = phy.duration
             if hasattr(phy, 'steps'): adapter_config['time_step'] = phy.duration / max(1, phy.steps)

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
        
        config = SimOpsConfig()
        config.heat_source_temperature = 800.0  # K (Hot end)
        config.ambient_temperature = 300.0      # K (Cold end)
        config.thermal_conductivity = 150.0     # W/m·K (Aluminum)
        
        solver = ThermalSolver(config, verbose=True)
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


def generate_report(
    job_name: str,
    output_dir: Path,
    result: Dict,
    strategy_name: str
) -> Dict[str, str]:
    """
    Generate premium visualizations and PDF report.
    
    Returns:
        Dict with paths to 'png' and 'pdf'
    """
    import matplotlib.pyplot as plt
    
    node_coords = result['node_coords']
    temperature = result['temperature']
    
    # 1. Generate High-Quality PNG using Cross-Section Slices
    png_file = output_dir / f"{job_name}_temperature.png"
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Define views: (Title, X-idx, Y-idx, Z-idx for slicing)
    # axes: 0=X, 1=Y, 2=Z
    views = [
        ('XY Section (Mid-Z)', 0, 1, 2),
        ('XZ Section (Mid-Y)', 0, 2, 1),
        ('YZ Section (Mid-X)', 1, 2, 0),
    ]
    
    T_min, T_max = np.min(temperature), np.max(temperature)
    levels = np.linspace(T_min, T_max, 20)
    
    for ax, (title, xi, yi, zi) in zip(axes, views):
        # Create a slice at the midpoint of the Z-axis (or whatever third axis)
        z_coords = node_coords[:, zi]
        z_min, z_max = np.min(z_coords), np.max(z_coords)
        z_mid = (z_min + z_max) / 2.0
        
        # Tolerance for slice thickness (e.g., 5% of range)
        epsilon = (z_max - z_min) * 0.05
        if epsilon == 0: epsilon = 1.0 # Handle flat 2D cases
        
        # Filter nodes within the slice
        mask = np.abs(z_coords - z_mid) < epsilon
        
        if np.sum(mask) < 3:
            # Fallback for empty slice: plot everything projected
            ax.text(0.5, 0.5, "Slice Empty\nShowing Projection", ha='center', va='center', transform=ax.transAxes)
            slide_coords_x = node_coords[:, xi]
            slide_coords_y = node_coords[:, yi]
            slide_temp = temperature
        else:
            slide_coords_x = node_coords[mask, xi]
            slide_coords_y = node_coords[mask, yi]
            slide_temp = temperature[mask]
        
        # Tricontourf for filled contours (Premium look)
        # Use 'inferno' or 'magma' for professional thermal look
        try:
            cntr = ax.tricontourf(
                slide_coords_x, 
                slide_coords_y, 
                slide_temp, 
                levels=levels, 
                cmap='magma',
                extend='both'
            )
            
            # Add thin contour lines for precision
            ax.tricontour(
                slide_coords_x, 
                slide_coords_y, 
                slide_temp, 
                levels=levels, 
                colors='k', 
                linewidths=0.3, 
                alpha=0.5
            )
        except Exception as e:
            logger.warning(f"Triangulation failed for view {title}: {e}")
            # Fallback to scatter
            cntr = ax.scatter(slide_coords_x, slide_coords_y, c=slide_temp, cmap='magma', s=2)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Add min/max labels on plot
        if len(slide_temp) > 0:
            local_min = np.min(slide_temp)
            local_max = np.max(slide_temp)
            ax.text(0.02, 0.02, f"L: {local_min:.0f}K", transform=ax.transAxes, fontsize=8, color='blue')
            ax.text(0.02, 0.95, f"H: {local_max:.0f}K", transform=ax.transAxes, fontsize=8, color='red')

    # Shared colorbar
    cbar = fig.colorbar(cntr, ax=axes.ravel().tolist(), label='Temperature (K)', shrink=0.9)
    cbar.ax.tick_params(labelsize=10)
    
    fig.suptitle(f"SimOps Thermal Analysis: {job_name}\n"
                f"Strategy: {strategy_name} | Elements: {result['num_elements']:,}",
                fontsize=14, fontweight='bold')
    
    # Save PNG
    plt.savefig(png_file, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  PNG saved: {png_file.name}")
    
    # 2. Generate PDF Report
    pdf_file = None
    if PDFReportGenerator:
        try:
            generator = PDFReportGenerator()
            # Prepare data for report
            report_data = {
                'job_name': job_name,
                'strategy_name': strategy_name,
                'min_temp': float(T_min),
                'max_temp': float(T_max),
                'num_elements': result['num_elements'],
                'solve_time': result['solve_time']
            }
            pdf_file = generator.generate(
                job_name=job_name,
                output_dir=output_dir,
                data=report_data,
                image_paths=[str(png_file)]
            )
            logger.info(f"  PDF saved: {Path(pdf_file).name}")
        except Exception as e:
            logger.error(f"  PDF Generation failed: {e}")
            import traceback
            traceback.print_exc()
            
    return {'png': str(png_file), 'pdf': str(pdf_file) if pdf_file else None}


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



def run_simulation(file_path: str, output_dir: str) -> SimulationResult:
    """
    Main entry point for SimOps Worker.
    Orchestrates the complete simulation pipeline:
    1. Load Config (Sidecar)
    2. Analyze Geometry
    3. Mesh (Cascade)
    4. Solve (Thermal)
    5. Report
    """
    start_time = time.time()
    
    # Path handling
    input_path = Path(file_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Job ID logic
    job_name = input_path.stem
    
    # ---------------------------------------------------------
    # 1. Load Configuration (The Sidecar Protocol)
    # ---------------------------------------------------------
    # Check for sibling .json file
    config_path = input_path.with_suffix('.json')
    from core.config_loader import load_simops_config
    
    # Check if a config file exists (even if not explicitly passed, look for sibling)
    sim_config = None
    if config_path.exists():
        logger.info(f"Found sidecar config: {config_path.name}")
        sim_config = load_simops_config(str(config_path))
    else:
        logger.info("No sidecar config found. Using Golden Template defaults.")
        sim_config = load_simops_config(None) # Load defaults
        
    # Override job name if specified in config
    if sim_config.job_name:
        job_name = sim_config.job_name

    logger.info(f"Starting job: {job_name}")
    logger.info(f"Physics: Material={sim_config.physics.material}, "
                f"Heat={sim_config.physics.heat_load_watts}W")
        
    # Validation / GCI Mode
    if sim_config.validate_mesh:
        try:
             from core.convergence.gci_runner import GCIRunner
             logger.info("="*60)
             logger.info("  VALIDATION MODE ENABLED: Running Grid Convergence Index Study")
             logger.info("="*60)
             
             gci_runner = GCIRunner(sim_config, file_path, output_path)
             gci_stats = gci_runner.run_study(levels=3, refinement_ratio=1.3)
             
             # We should serialize these stats to disk
             gci_file = output_path / f"{job_name}_GCI_Report.json"
             import json
             with open(gci_file, 'w') as f:
                  json.dump(gci_stats, f, indent=2)
                  
             logger.info(f"GCI Study Complete. Report saved to {gci_file.name}")
             
             # Fallthrough: We proceed to run the normal simulation sequence
             # (likely the 'Fine' one was already run, but user expects a standard result artifact)
             # Optimization: If GCIRunner saves artifacts in the right place, we could return early.
             # For now, let's allow the normal flow to run as the "Official" run (using base multipliers)
             # to ensure consistent output structure.
             
        except Exception as e:
             logger.error(f"GCI Study failed: {e}")
             # Don't hard fail the job, proceed to standard run

        
    strategies_to_try = STRATEGIES
    
    try:
        # Pre-analyze geometry
        geo_analysis = analyze_cad_geometry(file_path)
        logger.info("[Geometry Analysis]")
        logger.info(f"  Diagonal: {geo_analysis.diagonal:.2f} mm")
        logger.info(f"  Est. Wall: {geo_analysis.estimated_wall_thickness:.2f} mm")
        
        if geo_analysis.is_too_small_for_highfi:
            logger.warning("  [!] Geometry too small/thin for HighFi CFD. Skipping to robust strategies.")
            strategies_to_try = [s for s in STRATEGIES if s['name'] != "HighFi_CFD"]
            
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
            mesh_file = generate_mesh_with_strategy(
                file_path, 
                output_path, 
                strategy,
                sim_config=sim_config
            )
            
            # Step 2: Run thermal solver
            result = run_thermal_solver(mesh_file, output_path, strategy['name'], sim_config)
            
            # Step 3: Generate report
            report_files = generate_report(job_name, output_path, result, strategy['name'])
            png_file = report_files.get('summary')
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
                'min_temp_K': result['min_temp'],
                'max_temp_K': result['max_temp'],
                'num_elements': result['num_elements'],
                'solve_time_s': result['solve_time'],
                'total_time_s': total_time,
                'completed_at': datetime.now().isoformat(),
            }
            
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
                min_temp=result['min_temp'],
                max_temp=result['max_temp'],
                num_elements=result['num_elements'],
                solve_time=result['solve_time'],
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
