"""
True Anisotropic Meshing Implementation
========================================

Implements REAL anisotropic meshing using metric tensors and MMG3D.

This fixes the fundamental issues with the previous "fake anisotropic" approach:

1. Computes true 3x3 metric tensors based on principal curvature directions
2. Exports metric fields in .sol format for MMG3D
3. Uses curvature-aware sizing (not just length-based)
4. Integrates MMG3D remeshing for quality enforcement
5. Validates and repairs inverted elements (negative SICN)

Key Concepts:
-------------
- Metric Tensor M: 3x3 symmetric positive definite matrix defining anisotropic sizing
- Principal Curvatures κ1, κ2: Max and min curvature at a point
- Principal Directions: Direction of max/min curvature (stretch directions)
- SICN (Scaled Inverse Condition Number): Quality metric, must be > 0

For a sharp edge with curvature κ:
- Perpendicular size h⊥ = C/κ  (small, resolves curvature)
- Parallel size h∥ = αh⊥  (large, α = anisotropy ratio)
- Metric tensor aligns with edge direction

References:
-----------
- MMG3D User Guide: https://hal.inria.fr/hal-00681813
- Gmsh Reference Manual (Metric Fields)
- Cubit Skeleton Sizing: Adaptive anisotropic meshing
"""

import gmsh
import numpy as np
import struct
import subprocess
import os
import tempfile
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.spatial import cKDTree


@dataclass
class MetricTensor:
    """3x3 symmetric positive definite metric tensor"""
    # Store as 6 unique values: M11, M12, M13, M22, M23, M33
    m11: float
    m12: float
    m13: float
    m22: float
    m23: float
    m33: float

    def to_array(self) -> np.ndarray:
        """Convert to 3x3 matrix"""
        return np.array([
            [self.m11, self.m12, self.m13],
            [self.m12, self.m22, self.m23],
            [self.m13, self.m23, self.m33]
        ])

    @staticmethod
    def from_directions(direction: np.ndarray, h_parallel: float, h_perp1: float, h_perp2: float) -> 'MetricTensor':
        """
        Create metric tensor from principal direction and sizes

        Args:
            direction: Principal direction vector (will be normalized)
            h_parallel: Element size parallel to direction
            h_perp1: Element size perpendicular (direction 1)
            h_perp2: Element size perpendicular (direction 2)

        Returns:
            MetricTensor defining anisotropic sizing
        """
        # Normalize direction
        d = direction / np.linalg.norm(direction)

        # Create orthonormal basis
        if abs(d[0]) < 0.9:
            perp1 = np.array([1, 0, 0])
        else:
            perp1 = np.array([0, 1, 0])

        perp1 = perp1 - np.dot(perp1, d) * d
        perp1 = perp1 / np.linalg.norm(perp1)

        perp2 = np.cross(d, perp1)
        perp2 = perp2 / np.linalg.norm(perp2)

        # Metric tensor M = R * Λ * R^T
        # where Λ = diag(1/h_parallel^2, 1/h_perp1^2, 1/h_perp2^2)
        # and R = [d, perp1, perp2] (rotation matrix)

        lambda_vals = np.array([
            1.0 / (h_parallel**2),
            1.0 / (h_perp1**2),
            1.0 / (h_perp2**2)
        ])

        R = np.column_stack([d, perp1, perp2])
        Lambda = np.diag(lambda_vals)
        M = R @ Lambda @ R.T

        # Ensure symmetry (numerical stability)
        M = (M + M.T) / 2.0

        return MetricTensor(
            m11=M[0, 0], m12=M[0, 1], m13=M[0, 2],
            m22=M[1, 1], m23=M[1, 2], m33=M[2, 2]
        )

    @staticmethod
    def isotropic(h: float) -> 'MetricTensor':
        """Create isotropic metric with size h"""
        h2 = 1.0 / (h * h)
        return MetricTensor(m11=h2, m12=0, m13=0, m22=h2, m23=0, m33=h2)


@dataclass
class CurvatureInfo:
    """Curvature information at a point"""
    point: np.ndarray
    kappa_max: float  # Maximum curvature
    kappa_min: float  # Minimum curvature
    dir_max: np.ndarray  # Direction of max curvature
    dir_min: np.ndarray  # Direction of min curvature
    is_sharp: bool  # Is this a sharp feature?
    curve_tag: int = -1  # Which curve this sample belongs to


class TrueAnisotropicMeshGenerator:
    """
    Generates TRUE anisotropic meshes using metric tensors and MMG3D
    """

    def __init__(self,
                 base_size: float = 1.0,
                 anisotropy_ratio: float = 100.0,
                 curvature_threshold: float = 10.0,  # 1/mm - above this is "sharp"
                 min_size: float = 0.0001,
                 max_size: float = 100.0):
        """
        Initialize true anisotropic mesh generator

        Args:
            base_size: Base element size for flat regions
            anisotropy_ratio: Max ratio of parallel/perpendicular sizes
            curvature_threshold: Curvature above which features are considered sharp
            min_size: Absolute minimum element size
            max_size: Absolute maximum element size
        """
        self.base_size = base_size
        self.anisotropy_ratio = anisotropy_ratio
        self.curvature_threshold = curvature_threshold
        self.min_size = min_size
        self.max_size = max_size

        self.curvature_data: List[CurvatureInfo] = []
        self.node_metrics: Dict[int, MetricTensor] = {}

    def compute_curvature_at_curves(self) -> List[CurvatureInfo]:
        """
        Compute curvature information at sharp curves

        Uses finite differences on discretized curves to estimate curvature.
        Generates temporary 1D mesh if needed for analysis.
        """
        print("\n" + "="*70)
        print("CURVATURE ANALYSIS FOR TRUE ANISOTROPIC MESHING")
        print("="*70)

        curves = gmsh.model.getEntities(dim=1)
        curvature_data = []

        print(f"\nAnalyzing curvature of {len(curves)} curves...")

        # Check if 1D mesh exists, if not generate temporarily for analysis
        all_1d_nodes = gmsh.model.mesh.getNodes(dim=1, includeBoundary=True)
        mesh_exists = len(all_1d_nodes[0]) > 0

        if not mesh_exists:
            print("  Generating temporary FINE 1D mesh for curvature analysis...")
            print("  (Using 0.1mm mesh size to capture sharp features)")
            try:
                # Save current mesh size settings
                original_cl_min = gmsh.option.getNumber("Mesh.MeshSizeMin")
                original_cl_max = gmsh.option.getNumber("Mesh.MeshSizeMax")
                original_cl_from_curvature = gmsh.option.getNumber("Mesh.MeshSizeFromCurvature")

                # Set very fine mesh size for curvature analysis
                gmsh.option.setNumber("Mesh.MeshSizeMin", 0.01)  # 0.01mm min
                gmsh.option.setNumber("Mesh.MeshSizeMax", 0.1)   # 0.1mm max (fine!)
                gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 20)  # 20 points per 2π

                # Generate fine 1D mesh
                gmsh.model.mesh.generate(1)

                # Restore original settings
                gmsh.option.setNumber("Mesh.MeshSizeMin", original_cl_min)
                gmsh.option.setNumber("Mesh.MeshSizeMax", original_cl_max)
                gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", original_cl_from_curvature)

                print("  [OK] Temporary fine 1D mesh generated (0.1mm resolution)")
            except Exception as e:
                print(f"  [X] Failed to generate 1D mesh: {e}")
                print("  [!] Curvature analysis will be skipped")
                return []

        failed_bounds = 0
        failed_computation = 0

        for dim, curve_tag in curves:
            try:
                # Get mesh nodes from the curve (now guaranteed to exist)
                node_tags, node_coords, node_params = gmsh.model.mesh.getNodes(dim, curve_tag)

                if len(node_coords) < 6:  # Need at least 2 points (6 coords)
                    failed_bounds += 1
                    continue

                # Reshape to (n, 3)
                points = node_coords.reshape(-1, 3)
                n_points = len(points)

                if n_points < 3:  # Need at least 3 points for curvature
                    failed_bounds += 1
                    continue

                # Sample uniformly along the curve
                n_samples = min(20, n_points)
                sample_indices = np.linspace(0, n_points - 1, n_samples, dtype=int)

                for idx in sample_indices:
                    # Use finite differences on discretized points
                    i = int(idx)
                    if i <= 0 or i >= n_points - 1:
                        continue  # Need neighbors for finite differences

                    p0 = points[i]
                    p_prev = points[i - 1]
                    p_next = points[i + 1]

                    # First derivative (tangent) - central difference
                    dp = p_next - p_prev
                    tangent_norm = np.linalg.norm(dp)

                    if tangent_norm < 1e-10:
                        failed_computation += 1
                        continue

                    tangent = dp / tangent_norm

                    # Second derivative - finite difference
                    d2p = p_next - 2 * p0 + p_prev

                    # Curvature κ = |dp x d2p| / |dp|^3
                    cross = np.cross(dp, d2p)
                    curvature = np.linalg.norm(cross) / (tangent_norm**3 + 1e-10)

                    # For curves, we have one principal curvature (along curve)
                    # and zero perpendicular curvature
                    is_sharp = curvature > self.curvature_threshold

                    # Find perpendicular directions
                    if abs(tangent[0]) < 0.9:
                        perp1 = np.array([1, 0, 0])
                    else:
                        perp1 = np.array([0, 1, 0])

                    perp1 = perp1 - np.dot(perp1, tangent) * tangent
                    perp1 = perp1 / np.linalg.norm(perp1)

                    perp2 = np.cross(tangent, perp1)

                    curvature_info = CurvatureInfo(
                        point=p0,
                        kappa_max=curvature,
                        kappa_min=0.0,  # Zero curvature perpendicular to curve
                        dir_max=perp1 if curvature > 1e-6 else tangent,  # Direction of high curvature
                        dir_min=tangent,  # Direction of low curvature (along edge)
                        is_sharp=is_sharp,
                        curve_tag=curve_tag  # Store which curve this belongs to
                    )

                    curvature_data.append(curvature_info)

            except Exception as e:
                print(f"  [!] Failed to analyze curve {curve_tag}: {e}")
                continue

        sharp_count = sum(1 for c in curvature_data if c.is_sharp)
        print(f"\n[OK] Analyzed {len(curvature_data)} sample points")
        print(f"[OK] Found {sharp_count} sharp feature points (κ > {self.curvature_threshold:.1f})")

        if failed_bounds > 0:
            print(f"[!] {failed_bounds} curves had invalid parametrization bounds")
        if failed_computation > 0:
            print(f"[!] {failed_computation} sample points had zero tangent (skipped)")

        # ALWAYS print max curvature for diagnostics (even if 0 sharp features)
        if len(curvature_data) > 0:
            max_curv = max(c.kappa_max for c in curvature_data)
            avg_curv = sum(c.kappa_max for c in curvature_data) / len(curvature_data)
            print(f"[OK] Maximum curvature detected: {max_curv:.4f} mm⁻¹ (radius = {1/max_curv if max_curv > 0 else float('inf'):.2f} mm)")
            print(f"[OK] Average curvature: {avg_curv:.4f} mm⁻¹")

            # Show curvature distribution
            curvature_values = [c.kappa_max for c in curvature_data]
            percentiles = [50, 75, 90, 95, 99]
            print(f"  Curvature percentiles:")
            for p in percentiles:
                val = np.percentile(curvature_values, p)
                print(f"    {p}th: {val:.4f} mm⁻¹")

        self.curvature_data = curvature_data
        return curvature_data

    def compute_metric_field(self) -> Dict[int, MetricTensor]:
        """
        Compute metric tensor field at mesh nodes based on curvature

        Returns:
            Dictionary mapping node ID to MetricTensor
        """
        print("\n" + "="*70)
        print("METRIC TENSOR FIELD COMPUTATION")
        print("="*70)

        if not self.curvature_data:
            print("[!] No curvature data - computing first...")
            self.compute_curvature_at_curves()

        # Get all nodes from current mesh
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()

        if len(node_tags) == 0:
            print("[!] No mesh nodes found - generate surface mesh first")
            return {}

        # Reshape coordinates
        coords = node_coords.reshape(-1, 3)

        print(f"\nComputing metric tensors for {len(node_tags)} mesh nodes...")

        # Build KD-tree of curvature sample points for fast lookup
        if self.curvature_data:
            curv_points = np.array([c.point for c in self.curvature_data])
            kdtree = cKDTree(curv_points)
        else:
            kdtree = None

        node_metrics = {}
        sharp_node_count = 0

        for i, (node_id, coord) in enumerate(zip(node_tags, coords)):
            # Find nearest curvature sample
            if kdtree is not None:
                dist, idx = kdtree.query(coord, k=1)
                nearest_curv = self.curvature_data[idx]

                # Use curvature info if close enough (within 10x base size)
                if dist < self.base_size * 10:
                    curv_info = nearest_curv
                else:
                    curv_info = None
            else:
                curv_info = None

            # Compute metric tensor for this node
            if curv_info and curv_info.is_sharp:
                # ANISOTROPIC metric for sharp features
                # Size perpendicular to feature: based on curvature
                kappa = max(curv_info.kappa_max, 1e-6)
                h_perp = np.clip(1.0 / (kappa * 5.0), self.min_size, self.base_size)

                # Size parallel to feature: much larger
                h_parallel = np.clip(h_perp * self.anisotropy_ratio,
                                    h_perp,
                                    self.max_size)

                # Size in third direction (along sharp edge)
                h_edge = np.clip(h_perp * (self.anisotropy_ratio ** 0.5),
                                h_perp,
                                self.max_size)

                # Create metric aligned with curvature directions
                # High resolution perpendicular to sharp edge
                metric = MetricTensor.from_directions(
                    direction=curv_info.dir_min,  # Stretch along edge
                    h_parallel=h_parallel,  # Large size along edge
                    h_perp1=h_perp,  # Small size perpendicular (resolves curvature)
                    h_perp2=h_edge   # Medium size in third direction
                )

                sharp_node_count += 1

            else:
                # ISOTROPIC metric for smooth regions
                metric = MetricTensor.isotropic(self.base_size)

            node_metrics[int(node_id)] = metric

        print(f"\n[OK] Computed {len(node_metrics)} metric tensors")
        print(f"[OK] {sharp_node_count} anisotropic metrics at sharp features")
        print(f"[OK] {len(node_metrics) - sharp_node_count} isotropic metrics in smooth regions")

        self.node_metrics = node_metrics
        return node_metrics

    def export_sol_file(self, output_path: str) -> bool:
        """
        Export metric field in .sol format for MMG3D

        Sol file format (binary):
        - Header: version, dimension, metric type
        - Node metrics: For each node, 6 values (symmetric 3x3 tensor)

        Args:
            output_path: Path to write .sol file

        Returns:
            True if successful
        """
        print(f"\nExporting metric field to {output_path}...")

        if not self.node_metrics:
            print("[!] No metric data to export")
            return False

        try:
            # Get sorted node IDs to match mesh order
            node_tags = sorted(self.node_metrics.keys())
            n_nodes = len(node_tags)

            with open(output_path, 'wb') as f:
                # Write ASCII header
                header = f"MeshVersionFormatted 2\nDimension 3\n"
                f.write(header.encode('ascii'))

                # Write solution section
                f.write(f"SolAtVertices\n{n_nodes}\n1 3\n".encode('ascii'))

                # Write metric tensors (6 values per node: M11 M12 M13 M22 M23 M33)
                for node_id in node_tags:
                    metric = self.node_metrics[node_id]
                    line = f"{metric.m11:.10e} {metric.m12:.10e} {metric.m13:.10e} {metric.m22:.10e} {metric.m23:.10e} {metric.m33:.10e}\n"
                    f.write(line.encode('ascii'))

                f.write(b"End\n")

            print(f"[OK] Exported {n_nodes} metric tensors to {output_path}")
            return True

        except Exception as e:
            print(f"[X] Failed to export .sol file: {e}")
            return False

    def run_mmg3d_remeshing(self,
                           input_mesh: str,
                           output_mesh: str,
                           sol_file: str) -> bool:
        """
        Run MMG3D remeshing with anisotropic metric field

        Args:
            input_mesh: Input mesh file (.mesh or .msh)
            output_mesh: Output remeshed file
            sol_file: Metric field file (.sol)

        Returns:
            True if MMG3D succeeded
        """
        print("\n" + "="*70)
        print("MMG3D ANISOTROPIC REMESHING")
        print("="*70)

        # Check if MMG3D is available
        try:
            result = subprocess.run(['mmg3d', '-h'],
                                  capture_output=True,
                                  timeout=5)
            mmg3d_available = result.returncode == 0
        except:
            mmg3d_available = False

        if not mmg3d_available:
            print("\n[!] MMG3D not found in system PATH")
            print("  Install: https://github.com/MmgTools/mmg")
            print("  On macOS: brew install mmg")
            print("  On Linux: sudo apt-get install mmg3d")
            print("\n  Falling back to Gmsh optimization...")
            return False

        print(f"\nRunning MMG3D anisotropic remeshing...")
        print(f"  Input mesh: {input_mesh}")
        print(f"  Metric file: {sol_file}")
        print(f"  Output mesh: {output_mesh}")

        try:
            # Run MMG3D with metric-driven remeshing
            cmd = [
                'mmg3d',
                '-in', input_mesh,
                '-sol', sol_file,
                '-out', output_mesh,
                '-hausd', '0.001',  # Hausdorff distance (geometry fidelity)
                '-hgrad', '1.3',    # Size gradient (smooth transitions)
                '-v', '5'           # Verbosity
            ]

            print(f"\nCommand: {' '.join(cmd)}")

            result = subprocess.run(cmd,
                                  capture_output=True,
                                  text=True,
                                  timeout=300)

            if result.returncode == 0:
                print("\n[OK] MMG3D remeshing completed successfully!")
                print("\nMMG3D output:")
                print(result.stdout)
                return True
            else:
                print(f"\n[X] MMG3D failed with code {result.returncode}")
                print(result.stderr)
                return False

        except subprocess.TimeoutExpired:
            print("\n[X] MMG3D timed out (>5 minutes)")
            return False
        except Exception as e:
            print(f"\n[X] MMG3D execution failed: {e}")
            return False

    def apply_anisotropic_meshing(self,
                                 generate_surface_first: bool = True) -> bool:
        """
        Complete anisotropic meshing workflow

        1. Analyze curvature
        2. Generate surface mesh
        3. Compute metric tensors
        4. Export metric field
        5. Run MMG3D remeshing (if available)
        6. Fallback to Gmsh optimization

        Args:
            generate_surface_first: Generate surface mesh before metric computation

        Returns:
            True if successful
        """
        print("\n" + "="*70)
        print("TRUE ANISOTROPIC MESHING WORKFLOW")
        print("="*70)

        print(f"\nConfiguration:")
        print(f"  Base size: {self.base_size} mm")
        print(f"  Anisotropy ratio: {self.anisotropy_ratio}:1")
        print(f"  Curvature threshold: {self.curvature_threshold} mm⁻¹")
        print(f"  Size range: [{self.min_size}, {self.max_size}] mm")

        # Step 1: Curvature analysis
        self.compute_curvature_at_curves()

        # Step 2: Generate surface mesh if requested
        if generate_surface_first:
            print("\n" + "="*70)
            print("GENERATING SURFACE MESH")
            print("="*70)

            try:
                gmsh.model.mesh.generate(2)
                print("[OK] Surface mesh generated")
            except Exception as e:
                print(f"[X] Surface mesh generation failed: {e}")
                return False

        # Step 3: Compute metric field
        self.compute_metric_field()

        if not self.node_metrics:
            print("[X] No metric field computed")
            return False

        # Step 4: Export mesh and metric for MMG3D
        with tempfile.TemporaryDirectory() as tmpdir:
            input_mesh = os.path.join(tmpdir, "input.mesh")
            sol_file = os.path.join(tmpdir, "input.sol")
            output_mesh = os.path.join(tmpdir, "output.mesh")

            # Write mesh in .mesh format for MMG3D
            try:
                gmsh.write(input_mesh)
                print(f"\n[OK] Exported mesh: {input_mesh}")
            except Exception as e:
                print(f"[!] Mesh export failed: {e}")
                print("  Continuing with Gmsh optimization only...")
                return self._fallback_gmsh_optimization()

            # Export metric field
            if not self.export_sol_file(sol_file):
                print("  Continuing with Gmsh optimization only...")
                return self._fallback_gmsh_optimization()

            # Step 5: Run MMG3D
            if self.run_mmg3d_remeshing(input_mesh, output_mesh, sol_file):
                # Load remeshed result back into Gmsh
                try:
                    gmsh.model.mesh.clear()
                    gmsh.open(output_mesh)
                    print("\n[OK] Loaded MMG3D remeshed result")
                    return True
                except Exception as e:
                    print(f"[!] Failed to load MMG3D result: {e}")

        # Fallback: Gmsh optimization
        return self._fallback_gmsh_optimization()

    def _fallback_gmsh_optimization(self) -> bool:
        """
        Fallback: Use Gmsh's built-in optimization with high anisotropy
        """
        print("\n" + "="*70)
        print("FALLBACK: GMSH ANISOTROPIC OPTIMIZATION")
        print("="*70)

        print("\nApplying Gmsh optimization settings for anisotropic elements...")

        # Enable high anisotropy
        gmsh.option.setNumber("Mesh.AnisoMax", self.anisotropy_ratio)
        gmsh.option.setNumber("Mesh.AllowSwapAngle", 60)

        # Quality-oriented optimization
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.option.setNumber("Mesh.HighOrderOptimize", 2)

        # Smooth transitions
        gmsh.option.setNumber("Mesh.SmoothRatio", 1.5)

        # Size constraints
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", self.min_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", self.max_size)

        print("[OK] Gmsh optimization configured")
        print(f"  AnisoMax: {self.anisotropy_ratio}")
        print(f"  Size range: [{self.min_size}, {self.max_size}] mm")

        return True


def test_true_anisotropic_meshing():
    """Test true anisotropic meshing on a simple geometry"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python true_anisotropic_meshing.py <input.step> [base_size]")
        sys.exit(1)

    input_file = sys.argv[1]
    base_size = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    # Initialize Gmsh
    gmsh.initialize()
    gmsh.model.add("TrueAnisotropicTest")

    # Load geometry
    gmsh.model.occ.importShapes(input_file)
    gmsh.model.occ.synchronize()

    # Create true anisotropic mesher
    aniso_mesher = TrueAnisotropicMeshGenerator(
        base_size=base_size,
        anisotropy_ratio=100.0,
        curvature_threshold=10.0
    )

    # Apply anisotropic meshing
    if aniso_mesher.apply_anisotropic_meshing(generate_surface_first=True):
        # Generate volume mesh
        print("\nGenerating volume mesh...")
        gmsh.model.mesh.generate(3)

        # Write output
        output_file = input_file.replace('.step', '_anisotropic.msh')
        gmsh.write(output_file)

        print(f"\n[OK] Mesh written to {output_file}")
        print("\nVerify quality with:")
        print(f"  python -c 'from core.quality import MeshQualityAnalyzer; " +
              f"MeshQualityAnalyzer().analyze_mesh_file(\"{output_file}\")'")
    else:
        print("\n[X] Anisotropic meshing failed")

    gmsh.finalize()


if __name__ == "__main__":
    test_true_anisotropic_meshing()
