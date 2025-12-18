#!/usr/bin/env python
"""
SimOps Pipeline - Thermal Analysis Vending Machine
===================================================

Complete simulation workflow: CAD → Mesh → Solve → Visualize

Takes a CAD file as input and returns a thermal temperature map,
handling meshing and solving autonomously via open-source tools.

Usage:
    python simops_pipeline.py <cad_file.step> [output_dir]

Workflow:
    1. Import: Load CAD (STEP/IGES)
    2. Mesh: Generate CFD mesh with boundary layers
    3. Solve: Run thermal analysis (built-in FEA or CalculiX)
    4. Result: Export temperature distribution as PNG

Author: SimOps Team
Date: 2024
"""

import os
import sys
import json
import shutil
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
import tempfile
import time

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "core"))
sys.path.insert(0, str(Path(__file__).parent / "core" / "strategies"))

import gmsh


@dataclass
class SimOpsConfig:
    """Configuration for SimOps pipeline"""
    
    # Mesh settings
    num_boundary_layers: int = 5
    growth_rate: float = 1.2
    mesh_size_factor: float = 1.0
    
    # Thermal settings
    heat_source_power: float = 1e6          # W
    ambient_temperature: float = 300.0       # K
    heat_source_temperature: float = 800.0   # K (hot end)
    thermal_conductivity: float = 50.0       # W/(m·K) - default/fallback
    material: str = "Generic_Steel"          # Material name from library
    
    # Solver settings
    solver: str = "builtin"   # "builtin", "calculix", or "openfoam"
    max_iterations: int = 1000
    tolerance: float = 1e-6
    
    # Output settings
    output_format: str = "vtk"   # vtk, msh
    colormap: str = "jet"        # Temperature colormap


class ThermalSolver:
    """
    Built-in thermal FEA solver for steady-state heat conduction.
    
    Solves the heat equation: ∇·(k∇T) = 0
    with Dirichlet boundary conditions.
    """
    
    def __init__(self, config: SimOpsConfig, verbose: bool = True):
        self.config = config
        self.verbose = verbose
        self.node_coords: np.ndarray = None
        self.elements: np.ndarray = None
        self.temperature: np.ndarray = None
        
    def _log(self, msg: str):
        if self.verbose:
            print(msg, flush=True)
            
    def solve(self, mesh_file: str) -> Dict:
        """
        Solve thermal problem on the given mesh.
        
        Args:
            mesh_file: Path to mesh file (.msh)
            
        Returns:
            Dict with solution data
        """
        from scipy.sparse import lil_matrix, csr_matrix
        from scipy.sparse.linalg import cg, spilu, LinearOperator
        
        self._log("=" * 70)
        self._log("THERMAL SOLVER - Steady State Heat Conduction")
        self._log("=" * 70)
        
        start_time = time.time()
        
        # Load mesh
        self._log("\n[1/4] Loading mesh...")
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(mesh_file)
        
        # Extract mesh data
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_coords = node_coords.reshape(-1, 3)
        self.node_coords = node_coords
        
        # Build node tag to index map
        tag_to_idx = {int(tag): i for i, tag in enumerate(node_tags)}
        
        # Get tetrahedral elements
        elem_types, elem_tags, elem_nodes = gmsh.model.mesh.getElements(dim=3)
        
        tet_elements = []
        for i, etype in enumerate(elem_types):
            if etype == 4:  # Linear tet
                nodes = elem_nodes[i].reshape(-1, 4)
                tet_elements.append(nodes)
                
        if not tet_elements:
            gmsh.finalize()
            raise ValueError("No tetrahedral elements found")
            
        tet_elements = np.vstack(tet_elements).astype(int)
        
        # Convert to 0-indexed
        self.elements = np.array([[tag_to_idx.get(n, 0) for n in elem] for elem in tet_elements])
        
        num_nodes = len(node_tags)
        self._log(f"  Nodes: {num_nodes:,}")
        self._log(f"  Elements: {len(self.elements):,}")
        
        gmsh.finalize()
        
        # Apply boundary conditions
        self._log("\n[2/4] Applying boundary conditions...")
        bc_nodes, bc_temps = self._apply_boundary_conditions()
        self._log(f"  Heat source nodes: {np.sum(bc_temps > self.config.ambient_temperature)}")
        self._log(f"  Ambient nodes: {np.sum(bc_temps == self.config.ambient_temperature)}")
        
        # Assemble system
        self._log("\n[3/4] Assembling conductivity matrix...")
        K = lil_matrix((num_nodes, num_nodes))
        Q = np.zeros(num_nodes)
        
        k = self.config.thermal_conductivity
        
        for elem_idx in self.elements:
            coords = self.node_coords[elem_idx]
            ke = self._element_conductivity(coords, k)
            
            for i, ni in enumerate(elem_idx):
                for j, nj in enumerate(elem_idx):
                    K[ni, nj] += ke[i, j]
        
        # Apply Dirichlet BCs (penalty method)
        penalty = 1e20
        for node_idx, temp in zip(bc_nodes, bc_temps):
            K[node_idx, node_idx] += penalty
            Q[node_idx] = penalty * temp
            
        K = K.tocsr()
        self._log(f"  Matrix size: {K.shape[0]:,} DOF")
        self._log(f"  Non-zeros: {K.nnz:,}")
        
        # Solve
        self._log("\n[4/4] Solving thermal system (PCG)...")
        try:
            ilu = spilu(K.tocsc(), drop_tol=1e-4, fill_factor=10)
            M = LinearOperator(K.shape, matvec=ilu.solve)
        except:
            M = None
            
        T, info = cg(K, Q, M=M, atol=self.config.tolerance, maxiter=self.config.max_iterations)
        
        if info == 0:
            self._log("  [OK] Converged!")
        else:
            self._log(f"  [!] CG info: {info}")
            
        self.temperature = T
        
        elapsed = time.time() - start_time
        
        self._log("")
        self._log("=" * 70)
        self._log("RESULTS")
        self._log("=" * 70)
        self._log(f"  Min temperature:  {np.min(T):.2f} K ({np.min(T) - 273.15:.2f} °C)")
        self._log(f"  Max temperature:  {np.max(T):.2f} K ({np.max(T) - 273.15:.2f} °C)")
        self._log(f"  Temperature range: {np.max(T) - np.min(T):.2f} K")
        self._log(f"  Solve time:       {elapsed:.2f} s")
        
        return {
            'temperature': T,
            'node_coords': self.node_coords,
            'elements': self.elements,
            'min_temp': float(np.min(T)),
            'max_temp': float(np.max(T)),
            'solve_time': elapsed,
        }
        
    def _apply_boundary_conditions(self) -> Tuple[np.ndarray, np.ndarray]:
        """Apply temperature BCs based on geometry"""
        z = self.node_coords[:, 2]
        z_min, z_max = np.min(z), np.max(z)
        z_range = z_max - z_min
        
        all_nodes = []
        all_temps = []
        
        # Heat source: top 10% (hot end)
        hot_mask = z > z_max - 0.1 * z_range
        hot_nodes = np.where(hot_mask)[0]
        all_nodes.extend(hot_nodes)
        all_temps.extend([self.config.heat_source_temperature] * len(hot_nodes))
        
        # Ambient: bottom 10% (cold end)
        cold_mask = z < z_min + 0.1 * z_range
        cold_nodes = np.where(cold_mask)[0]
        all_nodes.extend(cold_nodes)
        all_temps.extend([self.config.ambient_temperature] * len(cold_nodes))
        
        return np.array(all_nodes), np.array(all_temps)
        
    def _element_conductivity(self, coords: np.ndarray, k: float) -> np.ndarray:
        """Compute element conductivity matrix for linear tet"""
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        
        # Volume via determinant
        J = np.array([
            [x[1]-x[0], x[2]-x[0], x[3]-x[0]],
            [y[1]-y[0], y[2]-y[0], y[3]-y[0]],
            [z[1]-z[0], z[2]-z[0], z[3]-z[0]],
        ])
        
        detJ = np.linalg.det(J)
        V = abs(detJ) / 6.0
        
        if V < 1e-20:
            return np.zeros((4, 4))
            
        # Shape function gradients (constant for linear tet)
        try:
            invJ = np.linalg.inv(J)
        except:
            return np.zeros((4, 4))
            
        dN_ref = np.array([
            [-1, -1, -1],
            [1,  0,  0],
            [0,  1,  0],
            [0,  0,  1],
        ])
        
        dN = dN_ref @ invJ
        
        # ke = k * V * B^T * B
        ke = k * V * (dN @ dN.T)
        
        return ke


class CalculiXSolver:
    """
    External CalculiX solver wrapper for robust thermal analysis.
    Requires CalculiX (ccx) to be installed.
    """
    
    def __init__(self, config: SimOpsConfig, verbose: bool = True):
        self.config = config
        self.verbose = verbose
        try:
            from core.solvers.calculix_wrapper import CalculiXWrapper
            self._wrapper = CalculiXWrapper(verbose=verbose)
        except ImportError:
            self._wrapper = None
            if verbose:
                print("[!] Could not import CalculiXWrapper from core.solvers")
        
    def _log(self, msg: str):
        if self.verbose:
            print(msg, flush=True)
            
    def is_available(self) -> bool:
        """Check if CalculiX is installed"""
        if not self._wrapper:
            return False
        return self._wrapper.is_available()
            
    def solve(self, mesh_file: str, work_dir: str) -> Dict:
        """Run thermal analysis via CalculiX"""
        self._log("=" * 70)
        self._log("CALCULIX THERMAL SOLVER")
        self._log("=" * 70)
        
        if not self.is_available():
            raise RuntimeError("CalculiX (ccx) not found. Install with: apt install calculix-ccx")
            
        # Resolve Material
        k = self.config.thermal_conductivity
        if self.config.material:
            try:
                from core.mat_lib import get_material_conductivity
                k_lookup = get_material_conductivity(self.config.material)
                if k_lookup is not None:
                    k = k_lookup
                    self._log(f"[CalculiX] Using material '{self.config.material}' properties")
                else:
                    self._log(f"[CalculiX] WARNING: Material '{self.config.material}' not found. Using default k={k}")
            except ImportError:
                pass
                
        return self._wrapper.solve_thermal(
            mesh_file, 
            work_dir,
            k_thermal=k,
            t_ambient=self.config.ambient_temperature,
            t_source=self.config.heat_source_temperature
        )


def generate_temperature_visualization(
    node_coords: np.ndarray,
    temperature: np.ndarray,
    output_file: str,
    colormap: str = "jet",
    title: str = "Temperature Distribution"
) -> str:
    """
    Generate a 2D temperature map visualization.
    
    Args:
        node_coords: Nx3 array of node coordinates
        temperature: N array of temperatures
        output_file: Output PNG path
        colormap: Matplotlib colormap name
        title: Plot title
        
    Returns:
        Path to saved PNG
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Project onto each plane
    views = [
        ('XY (Top)', node_coords[:, 0], node_coords[:, 1], 'X', 'Y'),
        ('XZ (Front)', node_coords[:, 0], node_coords[:, 2], 'X', 'Z'),
        ('YZ (Side)', node_coords[:, 1], node_coords[:, 2], 'Y', 'Z'),
    ]
    
    T_min, T_max = np.min(temperature), np.max(temperature)
    
    for ax, (name, x, y, xlabel, ylabel) in zip(axes, views):
        scatter = ax.scatter(x, y, c=temperature, cmap=colormap, 
                            s=1, vmin=T_min, vmax=T_max)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(name)
        ax.set_aspect('equal')
        
    plt.colorbar(scatter, ax=axes, label='Temperature (K)', shrink=0.8)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_file


def export_vtk_with_temperature(
    node_coords: np.ndarray,
    elements: np.ndarray,
    temperature: np.ndarray,
    output_file: str
) -> str:
    """
    Export mesh with temperature field to VTK format for GUI visualization.
    
    Args:
        node_coords: Nx3 node coordinates
        elements: Mx4 tetrahedral elements (0-indexed)
        temperature: N nodal temperatures
        output_file: Output .vtu path
        
    Returns:
        Path to VTK file
    """
    output_path = Path(output_file)
    
    # Write legacy VTK format (widely compatible)
    with open(output_path, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("SimOps Thermal Analysis Result\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        
        # Points
        f.write(f"POINTS {len(node_coords)} float\n")
        for p in node_coords:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            
        # Cells
        num_cells = len(elements)
        f.write(f"\nCELLS {num_cells} {num_cells * 5}\n")
        for elem in elements:
            f.write(f"4 {elem[0]} {elem[1]} {elem[2]} {elem[3]}\n")
            
        # Cell types (10 = VTK_TETRA)
        f.write(f"\nCELL_TYPES {num_cells}\n")
        for _ in range(num_cells):
            f.write("10\n")
            
        # Point data (temperature)
        f.write(f"\nPOINT_DATA {len(temperature)}\n")
        f.write("SCALARS Temperature float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for t in temperature:
            f.write(f"{t:.4f}\n")
            
    return str(output_path)


def run_simops_pipeline(
    cad_file: str,
    output_dir: str = "simops_output",
    config: Optional[SimOpsConfig] = None,
    verbose: bool = True
) -> Dict:
    """
    Run the complete SimOps thermal analysis pipeline.
    
    Args:
        cad_file: Input CAD file (STEP, IGES)
        output_dir: Output directory
        config: SimOpsConfig settings
        verbose: Print progress
        
    Returns:
        Dict with results and file paths
    """
    config = config or SimOpsConfig()
    
    def log(msg: str):
        if verbose:
            print(msg, flush=True)
            
    log("")
    log("=" * 70)
    log("   SIMOPS - THERMAL ANALYSIS VENDING MACHINE")
    log("=" * 70)
    log(f"   Input:  {cad_file}")
    log(f"   Output: {output_dir}")
    log("=" * 70)
    log("")
    
    start_time = time.time()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Generate CFD mesh with boundary layers
    log("STEP 1: GENERATING CFD MESH WITH BOUNDARY LAYERS")
    log("-" * 70)
    
    from cfd_strategy import CFDMeshStrategy, CFDMeshConfig
    
    mesh_config = CFDMeshConfig(
        num_layers=config.num_boundary_layers,
        growth_rate=config.growth_rate,
        mesh_size_factor=config.mesh_size_factor,
    )
    
    mesh_file = output_path / "mesh.msh"
    strategy = CFDMeshStrategy(verbose=verbose)
    success, mesh_stats = strategy.generate_cfd_mesh(
        cad_file, 
        str(mesh_file),
        mesh_config
    )
    
    if not success:
        raise RuntimeError(f"Mesh generation failed: {mesh_stats.get('error')}")
        
    log("")
    
    # Step 2: Run thermal solver
    log("STEP 2: SOLVING THERMAL PROBLEM")
    log("-" * 70)
    
    if config.solver == "calculix":
        solver = CalculiXSolver(config, verbose=verbose)
        results = solver.solve(str(mesh_file), str(output_path))
    else:
        solver = ThermalSolver(config, verbose=verbose)
        results = solver.solve(str(mesh_file))
        
    log("")
    
    # Step 3: Generate visualizations
    log("STEP 3: GENERATING VISUALIZATIONS")
    log("-" * 70)
    
    # Temperature map PNG
    png_file = output_path / "temperature_map.png"
    generate_temperature_visualization(
        results['node_coords'],
        results['temperature'],
        str(png_file),
        colormap=config.colormap,
        title=f"Thermal Analysis: {Path(cad_file).stem}"
    )
    log(f"  Temperature map: {png_file}")
    
    # VTK with temperature for GUI
    vtk_file = output_path / "thermal_result.vtk"
    export_vtk_with_temperature(
        results['node_coords'],
        results['elements'],
        results['temperature'],
        str(vtk_file)
    )
    log(f"  VTK result:      {vtk_file}")
    
    # Save metadata
    metadata = {
        'input_file': str(cad_file),
        'mesh_file': str(mesh_file),
        'vtk_file': str(vtk_file),
        'png_file': str(png_file),
        'num_nodes': mesh_stats.get('num_nodes', 0),
        'num_elements': mesh_stats.get('num_tets', 0) + mesh_stats.get('num_prisms', 0),
        'min_temperature_K': results['min_temp'],
        'max_temperature_K': results['max_temp'],
        'min_temperature_C': results['min_temp'] - 273.15,
        'max_temperature_C': results['max_temp'] - 273.15,
        'solve_time_s': results['solve_time'],
        'total_time_s': time.time() - start_time,
    }
    
    metadata_file = output_path / "simops_results.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    log(f"  Metadata:        {metadata_file}")
    
    log("")
    log("=" * 70)
    log("   SIMOPS PIPELINE COMPLETE")
    log("=" * 70)
    log(f"   Total time: {metadata['total_time_s']:.2f} s")
    log(f"   Results:    {output_path.absolute()}")
    log("=" * 70)
    log("")
    
    return metadata


# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="SimOps - Thermal Analysis Vending Machine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python simops_pipeline.py model.step
  python simops_pipeline.py model.step --output results/
  python simops_pipeline.py model.step --layers 8 --growth 1.3
        """
    )
    
    parser.add_argument("cad_file", help="Input CAD file (STEP, IGES)")
    parser.add_argument("-o", "--output", default="simops_output", help="Output directory")
    parser.add_argument("--layers", type=int, default=5, help="Number of boundary layers")
    parser.add_argument("--growth", type=float, default=1.2, help="Boundary layer growth rate")
    parser.add_argument("--solver", choices=["builtin", "calculix"], default="builtin",
                       help="Thermal solver to use")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress output")
    
    args = parser.parse_args()
    
    config = SimOpsConfig(
        num_boundary_layers=args.layers,
        growth_rate=args.growth,
        solver=args.solver,
    )
    
    try:
        results = run_simops_pipeline(
            args.cad_file,
            args.output,
            config,
            verbose=not args.quiet
        )
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
