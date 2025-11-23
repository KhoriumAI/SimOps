#!/usr/bin/env python3
"""
Quality Analysis Performance Benchmark
=======================================

Compares performance of different quality analysis methods:
1. GPU-accelerated (CuPy) - Target: 10-100x faster
2. Batch vectorized (NumPy) - Target: 5-10x faster
3. Original element-by-element - Baseline

Usage:
    python benchmark_quality_performance.py <mesh_file.msh>

Example:
    python benchmark_quality_performance.py generated_meshes/Cube_mesh.msh
"""

import sys
import time
from pathlib import Path
import gmsh

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.quality import MeshQualityAnalyzer
from core.quality_accelerated import AcceleratedQualityAnalyzer, GPU_AVAILABLE


def benchmark_quality_analysis(mesh_file: str):
    """
    Benchmark different quality analysis methods

    Args:
        mesh_file: Path to .msh file
    """
    print("=" * 70)
    print("QUALITY ANALYSIS PERFORMANCE BENCHMARK")
    print("=" * 70)
    print(f"Mesh file: {mesh_file}")
    print()

    if not Path(mesh_file).exists():
        print(f"Error: File not found: {mesh_file}")
        return

    # Initialize gmsh and load mesh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    try:
        gmsh.open(mesh_file)

        # Get mesh statistics
        element_types, element_tags, node_tags = gmsh.model.mesh.getElements()
        total_elements = sum(len(tags) for tags in element_tags)
        total_nodes = len(set(node for nodes in node_tags for node in nodes))

        print(f"Mesh Statistics:")
        print(f"  Total Elements: {total_elements:,}")
        print(f"  Total Nodes: {total_nodes:,}")
        print()

        # Benchmark 1: Original implementation
        print("-" * 70)
        print("TEST 1: Original Element-by-Element Analysis (Baseline)")
        print("-" * 70)

        analyzer_original = MeshQualityAnalyzer()

        start_time = time.time()
        metrics_original = analyzer_original.analyze_mesh(include_advanced_metrics=False)
        original_time = time.time() - start_time

        if metrics_original:
            print(f"[OK] Analysis completed in {original_time:.3f} seconds")
            print(f"  SICN min: {metrics_original['gmsh_sicn']['min']:.4f}")
            print(f"  Gamma min: {metrics_original['gmsh_gamma']['min']:.4f}")
        else:
            print("[X] Analysis failed")
            original_time = None

        print()

        # Benchmark 2: Batch vectorized (NumPy)
        print("-" * 70)
        print("TEST 2: Batch Vectorized Analysis (NumPy)")
        print("-" * 70)

        analyzer_batch = AcceleratedQualityAnalyzer(use_gpu=False)

        start_time = time.time()
        metrics_batch = analyzer_batch.analyze_mesh_fast(include_advanced_metrics=False)
        batch_time = time.time() - start_time

        if metrics_batch:
            print(f"[OK] Analysis completed in {batch_time:.3f} seconds")
            print(f"  SICN min: {metrics_batch['gmsh_sicn']['min']:.4f}")
            print(f"  Gamma min: {metrics_batch['gmsh_gamma']['min']:.4f}")

            if original_time:
                speedup = original_time / batch_time
                print(f"  Speedup: {speedup:.1f}x faster than original")
        else:
            print("[X] Analysis failed")
            batch_time = None

        print()

        # Benchmark 3: GPU-accelerated (if available)
        if GPU_AVAILABLE:
            print("-" * 70)
            print("TEST 3: GPU-Accelerated Analysis (CuPy)")
            print("-" * 70)

            analyzer_gpu = AcceleratedQualityAnalyzer(use_gpu=True)

            start_time = time.time()
            metrics_gpu = analyzer_gpu.analyze_mesh_fast(include_advanced_metrics=False)
            gpu_time = time.time() - start_time

            if metrics_gpu:
                print(f"[OK] Analysis completed in {gpu_time:.3f} seconds")
                print(f"  SICN min: {metrics_gpu['gmsh_sicn']['min']:.4f}")
                print(f"  Gamma min: {metrics_gpu['gmsh_gamma']['min']:.4f}")

                if original_time:
                    speedup = original_time / gpu_time
                    print(f"  Speedup: {speedup:.1f}x faster than original")
            else:
                print("[X] Analysis failed")
                gpu_time = None

            print()
        else:
            print("-" * 70)
            print("TEST 3: GPU-Accelerated Analysis (CuPy)")
            print("-" * 70)
            print("[!] GPU acceleration not available (CuPy not installed)")
            print("  Install with: pip install cupy-cuda12x")
            gpu_time = None
            print()

        # Summary
        print("=" * 70)
        print("PERFORMANCE SUMMARY")
        print("=" * 70)

        if original_time:
            print(f"Original (Baseline):    {original_time:.3f}s  (1.0x)")

        if batch_time:
            if original_time:
                speedup = original_time / batch_time
                print(f"Batch Vectorized:       {batch_time:.3f}s  ({speedup:.1f}x faster)")
            else:
                print(f"Batch Vectorized:       {batch_time:.3f}s")

        if gpu_time:
            if original_time:
                speedup = original_time / gpu_time
                print(f"GPU-Accelerated:        {gpu_time:.3f}s  ({speedup:.1f}x faster)")
            else:
                print(f"GPU-Accelerated:        {gpu_time:.3f}s")

        print()

        # Recommendations
        print("=" * 70)
        print("RECOMMENDATIONS")
        print("=" * 70)

        if GPU_AVAILABLE and gpu_time and original_time:
            if original_time / gpu_time > 10:
                print("[OK] GPU acceleration provides excellent speedup (>10x)")
                print("  Recommendation: Enable GPU for all mesh analysis")
            elif original_time / gpu_time > 3:
                print("[OK] GPU acceleration provides good speedup (3-10x)")
                print("  Recommendation: Use GPU for large meshes (>10k elements)")
            else:
                print("[!] GPU overhead dominates for small meshes")
                print("  Recommendation: Use batch CPU for small meshes, GPU for large")

        elif batch_time and original_time:
            speedup = original_time / batch_time
            if speedup > 3:
                print("[OK] Batch vectorization provides significant speedup")
                print("  Recommendation: Use batch analysis for all meshes")
            else:
                print("[!] Limited speedup with current mesh size")
                print("  Recommendation: Batch is best for larger meshes")

        if not GPU_AVAILABLE:
            print()
            print("💡 Install CuPy for GPU acceleration:")
            print("   pip install cupy-cuda12x")
            print("   Expected speedup: 10-100x for large meshes")

        print()

    finally:
        gmsh.finalize()


def run_multiple_iterations(mesh_file: str, iterations: int = 10):
    """
    Run multiple iterations for more accurate timing

    Args:
        mesh_file: Path to mesh file
        iterations: Number of iterations to run
    """
    print("=" * 70)
    print(f"RUNNING {iterations} ITERATIONS FOR ACCURATE TIMING")
    print("=" * 70)
    print()

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    try:
        gmsh.open(mesh_file)

        # Get mesh stats
        element_types, element_tags, node_tags = gmsh.model.mesh.getElements()
        total_elements = sum(len(tags) for tags in element_tags)

        print(f"Mesh: {total_elements:,} elements")
        print(f"Iterations: {iterations}")
        print()

        # Test 1: Original
        print("Testing Original implementation...")
        analyzer_original = MeshQualityAnalyzer()
        times_original = []

        for i in range(iterations):
            start = time.time()
            analyzer_original.analyze_mesh(include_advanced_metrics=False)
            times_original.append(time.time() - start)
            print(f"  Iteration {i+1}/{iterations}: {times_original[-1]:.3f}s")

        avg_original = sum(times_original) / len(times_original)
        print(f"Average: {avg_original:.3f}s")
        print()

        # Test 2: Batch
        print("Testing Batch vectorized implementation...")
        analyzer_batch = AcceleratedQualityAnalyzer(use_gpu=False)
        times_batch = []

        for i in range(iterations):
            start = time.time()
            analyzer_batch.analyze_mesh_fast(include_advanced_metrics=False)
            times_batch.append(time.time() - start)
            print(f"  Iteration {i+1}/{iterations}: {times_batch[-1]:.3f}s")

        avg_batch = sum(times_batch) / len(times_batch)
        speedup_batch = avg_original / avg_batch
        print(f"Average: {avg_batch:.3f}s  ({speedup_batch:.1f}x faster)")
        print()

        # Test 3: GPU (if available)
        if GPU_AVAILABLE:
            print("Testing GPU-accelerated implementation...")
            analyzer_gpu = AcceleratedQualityAnalyzer(use_gpu=True)
            times_gpu = []

            for i in range(iterations):
                start = time.time()
                analyzer_gpu.analyze_mesh_fast(include_advanced_metrics=False)
                times_gpu.append(time.time() - start)
                print(f"  Iteration {i+1}/{iterations}: {times_gpu[-1]:.3f}s")

            avg_gpu = sum(times_gpu) / len(times_gpu)
            speedup_gpu = avg_original / avg_gpu
            print(f"Average: {avg_gpu:.3f}s  ({speedup_gpu:.1f}x faster)")
            print()

        # Final summary
        print("=" * 70)
        print("FINAL RESULTS (AVERAGE OF {} ITERATIONS)".format(iterations))
        print("=" * 70)
        print(f"Original:          {avg_original:.3f}s")
        print(f"Batch Vectorized:  {avg_batch:.3f}s  ({speedup_batch:.1f}x)")
        if GPU_AVAILABLE:
            print(f"GPU-Accelerated:   {avg_gpu:.3f}s  ({speedup_gpu:.1f}x)")
        print()

    finally:
        gmsh.finalize()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python benchmark_quality_performance.py <mesh_file.msh>")
        print()
        print("Example:")
        print("  python benchmark_quality_performance.py generated_meshes/Cube_mesh.msh")
        print()
        print("For multiple iterations (more accurate):")
        print("  python benchmark_quality_performance.py <mesh_file.msh> --iterations 10")
        sys.exit(1)

    mesh_file = sys.argv[1]

    if "--iterations" in sys.argv:
        idx = sys.argv.index("--iterations")
        if idx + 1 < len(sys.argv):
            iterations = int(sys.argv[idx + 1])
            run_multiple_iterations(mesh_file, iterations)
        else:
            print("Error: --iterations requires a number")
            sys.exit(1)
    else:
        benchmark_quality_analysis(mesh_file)
