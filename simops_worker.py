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
    strategy: Dict
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
    # strategy['mesh_size'] is % of diagonal in legacy logic.
    # CFDMeshConfig uses mesh_size_factor (multiplier for defaults).
    # Default is diagonal/200 (0.5%).
    # So if strategy['mesh_size'] is 0.5, that matches default (factor=1.0).
    # Factor = strategy_size / 0.5
    
    base_factor = strategy.get('mesh_size', 0.5) / 0.5
    
    config = CFDMeshConfig(
        num_layers=strategy.get('num_layers', 5),
        growth_rate=strategy.get('growth_rate', 1.2),
        mesh_size_factor=base_factor,
        create_prisms=True,
        optimize_netgen=strategy.get('optimize', True),
        smoothing_steps=10
    )
    
    runner = CFDMeshStrategy(verbose=True)
    success, stats = runner.generate_cfd_mesh(cad_file, str(output_file), config)
    
    if not success:
        raise RuntimeError(f"Mesh generation failed: {stats.get('error')}")
        
    return str(output_file)


def run_thermal_solver(mesh_file: str, output_dir: Path, strategy_name: str) -> Dict:
    """
    Run thermal analysis on the mesh.
    
    Uses built-in FEA solver for steady-state heat conduction.
    
    Args:
        mesh_file: Path to mesh file
        output_dir: Output directory
        strategy_name: Name of strategy (for file naming)
        
    Returns:
        Dict with solution data
    """
    from scipy.sparse import lil_matrix, csr_matrix
    from scipy.sparse.linalg import cg, spilu, LinearOperator
    
    logger.info("Running thermal solver...")
    
    # Material properties
    thermal_conductivity = 50.0  # W/(m·K) - steel
    T_hot = 800.0   # K
    T_cold = 300.0  # K
    
    start_time = time.time()
    
    # Load mesh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.open(mesh_file)
    
    try:
        # Extract mesh data
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_coords = node_coords.reshape(-1, 3)
        
        tag_to_idx = {int(tag): i for i, tag in enumerate(node_tags)}
        
        elem_types, elem_tags, elem_nodes = gmsh.model.mesh.getElements(dim=3)
        
        tet_elements = []
        for i, etype in enumerate(elem_types):
            if etype == 4:  # Linear tet
                nodes = elem_nodes[i].reshape(-1, 4)
                tet_elements.append(nodes)
                
        if not tet_elements:
            raise ValueError("No tetrahedral elements found")
            
        tet_elements = np.vstack(tet_elements).astype(int)
        elements = np.array([[tag_to_idx.get(n, 0) for n in elem] for elem in tet_elements])
        
        num_nodes = len(node_tags)
        logger.info(f"  Nodes: {num_nodes:,}, Elements: {len(elements):,}")
        
    finally:
        gmsh.finalize()
    
    # Apply BCs (top = hot, bottom = cold)
    z = node_coords[:, 2]
    z_min, z_max = np.min(z), np.max(z)
    z_range = z_max - z_min
    
    hot_nodes = np.where(z > z_max - 0.1 * z_range)[0]
    cold_nodes = np.where(z < z_min + 0.1 * z_range)[0]
    
    # Assemble system
    K = lil_matrix((num_nodes, num_nodes))
    Q = np.zeros(num_nodes)
    
    for elem_idx in elements:
        coords = node_coords[elem_idx]
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        
        J = np.array([
            [x[1]-x[0], x[2]-x[0], x[3]-x[0]],
            [y[1]-y[0], y[2]-y[0], y[3]-y[0]],
            [z[1]-z[0], z[2]-z[0], z[3]-z[0]],
        ])
        
        detJ = np.linalg.det(J)
        V = abs(detJ) / 6.0
        
        if V < 1e-20:
            continue
            
        try:
            invJ = np.linalg.inv(J)
        except:
            continue
            
        dN_ref = np.array([[-1,-1,-1], [1,0,0], [0,1,0], [0,0,1]])
        dN = dN_ref @ invJ
        ke = thermal_conductivity * V * (dN @ dN.T)
        
        for i, ni in enumerate(elem_idx):
            for j, nj in enumerate(elem_idx):
                K[ni, nj] += ke[i, j]
    
    # Apply BCs
    # Use a lower penalty to avoid ill-conditioning (1e20 is too high for float64)
    # 1e12 gives 12 digits of dominance, leaving sufficient precision for physics
    penalty = 1e12
    for n in hot_nodes:
        K[n, n] += penalty
        Q[n] = penalty * T_hot
    for n in cold_nodes:
        K[n, n] += penalty
        Q[n] = penalty * T_cold
        
    K = K.tocsr()
    
    # Solve
    # Switch to Direct Solver (spsolve) for robustness
    # CG fails with high number of elements/bad conditioning
    from scipy.sparse.linalg import spsolve
    
    logger.info("  Solving linear system (Direct Solver)...")
    try:
        T = spsolve(K, Q)
    except Exception as e:
        logger.error(f"  Direct solver failed: {e}. Falling back to iterative.")
        # Fallback to CG if direct runs out of memory
        ilu = spilu(K.tocsc(), drop_tol=1e-4, fill_factor=10)
        M = LinearOperator(K.shape, matvec=ilu.solve)
        T, info = cg(K, Q, M=M, atol=1e-8, maxiter=2000)
    
    # Safety Check: Clamp negative temperatures (physically impossible in this setup)
    # This handles any minor numerical noise around 0
    T = np.maximum(T, 0.0)
        
    solve_time = time.time() - start_time
    
    logger.info(f"  Temperature range: {np.min(T):.1f} - {np.max(T):.1f} K")
    logger.info(f"  Solve time: {solve_time:.2f}s")
    
    return {
        'temperature': T,
        'node_coords': node_coords,
        'elements': elements,
        'min_temp': float(np.min(T)),
        'max_temp': float(np.max(T)),
        'solve_time': solve_time,
        'num_elements': len(elements),
    }


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


def run_simulation(file_path: str, output_dir: str = None) -> SimulationResult:
    """
    Main entry point called by the RQ Worker.
    
    Implements the 'Cascading Fallback' strategy - tries multiple
    mesh strategies until one succeeds.
    
    Args:
        file_path: Path to input CAD file
        output_dir: Output directory (uses env var if not specified)
        
    Returns:
        SimulationResult with output files and metadata
    """
    job_name = Path(file_path).stem
    output_path = Path(output_dir or os.environ.get('OUTPUT_DIR', './output'))
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info(f"   SIMOPS JOB: {job_name}")
    logger.info("=" * 60)
    logger.info(f"   Input:  {file_path}")
    logger.info(f"   Output: {output_path}")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    # Try each strategy in order
    strategies_to_try = STRATEGIES
    
    # Pre-analyze geometry
    try:
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
            mesh_file = generate_mesh_with_strategy(file_path, output_path, strategy)
            
            # Step 2: Run thermal solver
            result = run_thermal_solver(mesh_file, output_path, strategy['name'])
            
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
