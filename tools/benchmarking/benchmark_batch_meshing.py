#!/usr/bin/env python3
"""
Batch Meshing Benchmark Script
================================

Meshes multiple STEP files and collects performance/quality metrics.
Tests the automatic geometry healing system on diverse geometries.

Usage:
    python benchmark_batch_meshing.py
"""

import sys
import os
from pathlib import Path
import time
import json
from datetime import datetime
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.config import Config
from strategies.exhaustive_strategy import ExhaustiveMeshGenerator
from core.quality import MeshQualityAnalyzer
import gmsh


class BenchmarkResult:
    """Container for benchmark results"""
    def __init__(self, filename: str):
        self.filename = filename
        self.success = False
        self.time_total = 0.0
        self.time_geometry = 0.0
        self.time_healing = 0.0
        self.time_meshing = 0.0
        self.time_quality = 0.0

        # Geometry stats
        self.num_surfaces = 0
        self.num_curves = 0
        self.num_volumes = 0

        # Healing stats
        self.ultra_thin_surfaces = 0
        self.zero_thickness_edges = 0
        self.size_fields_clamped = 0
        self.healing_applied = False

        # Mesh stats
        self.num_elements = 0
        self.num_nodes = 0

        # Quality stats
        self.sicn_min = 0.0
        self.sicn_avg = 0.0
        self.skew_max = 0.0
        self.skew_avg = 0.0
        self.aspect_max = 0.0
        self.aspect_avg = 0.0

        # Error info
        self.error_message = ""


class BatchMeshBenchmark:
    """Batch meshing benchmark runner"""

    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[BenchmarkResult] = []

    def benchmark_file(self, input_file: str, target_size_mm: float = 3.0) -> BenchmarkResult:
        """
        Benchmark a single STEP file

        Args:
            input_file: Path to STEP file
            target_size_mm: Target mesh size in mm

        Returns:
            BenchmarkResult with all metrics
        """
        result = BenchmarkResult(Path(input_file).name)

        print("\n" + "="*80)
        print(f"BENCHMARKING: {result.filename}")
        print("="*80)

        # Output file
        output_file = str(self.output_dir / f"{Path(input_file).stem}_mesh.msh")

        try:
            # Start total timer
            t_start = time.time()

            # Initialize Gmsh
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 1)

            # Load geometry
            print(f"\n[1/4] Loading geometry...")
            t_geom_start = time.time()
            gmsh.model.add("benchmark")
            gmsh.merge(input_file)

            # Get geometry stats
            surfaces = gmsh.model.getEntities(dim=2)
            curves = gmsh.model.getEntities(dim=1)
            volumes = gmsh.model.getEntities(dim=3)

            result.num_surfaces = len(surfaces)
            result.num_curves = len(curves)
            result.num_volumes = len(volumes)

            result.time_geometry = time.time() - t_geom_start

            print(f"  [OK] Loaded: {result.num_volumes} volumes, "
                  f"{result.num_surfaces} surfaces, {result.num_curves} curves")
            print(f"  [OK] Time: {result.time_geometry:.2f}s")

            # Create config
            config = Config()
            config.default_params = {
                'cl_max': target_size_mm / 1000,  # Convert mm to meters
                'cl_min': target_size_mm / 1000 * 0.1,
                'algorithm_3d': 1  # Delaunay
            }

            # Create mesh generator
            generator = ExhaustiveMeshGenerator(config)

            # Run meshing (this includes automatic healing)
            print(f"\n[2/4] Running meshing with automatic healing...")
            t_mesh_start = time.time()

            success = generator.run_meshing_strategy(input_file, output_file)

            result.time_meshing = time.time() - t_mesh_start
            result.success = success

            if not success:
                result.error_message = "Mesh generation failed"
                print(f"  [X] Meshing failed")
                return result

            print(f"  [OK] Meshing complete: {result.time_meshing:.2f}s")

            # Get mesh stats
            print(f"\n[3/4] Collecting mesh statistics...")
            elements = gmsh.model.mesh.getElements()
            nodes = gmsh.model.mesh.getNodes()

            # Count total elements
            total_elements = 0
            for elem_types, elem_tags_list, _ in zip(elements[0], elements[1], elements[2]):
                total_elements += len(elem_tags_list)

            result.num_elements = total_elements
            result.num_nodes = len(nodes[0])

            print(f"  [OK] Elements: {result.num_elements:,}")
            print(f"  [OK] Nodes: {result.num_nodes:,}")

            # Analyze quality
            print(f"\n[4/4] Analyzing mesh quality...")
            t_quality_start = time.time()

            analyzer = MeshQualityAnalyzer()
            quality = analyzer.analyze_mesh(output_file)

            result.time_quality = time.time() - t_quality_start

            if quality:
                result.sicn_min = quality.get('sicn', {}).get('min', 0.0)
                result.sicn_avg = quality.get('sicn', {}).get('mean', 0.0)
                result.skew_max = quality.get('skew', {}).get('max', 0.0)
                result.skew_avg = quality.get('skew', {}).get('mean', 0.0)
                result.aspect_max = quality.get('aspect_ratio', {}).get('max', 0.0)
                result.aspect_avg = quality.get('aspect_ratio', {}).get('mean', 0.0)

                print(f"  [OK] SICN: min={result.sicn_min:.4f}, avg={result.sicn_avg:.4f}")
                print(f"  [OK] Skewness: max={result.skew_max:.4f}, avg={result.skew_avg:.4f}")
                print(f"  [OK] Aspect: max={result.aspect_max:.1f}, avg={result.aspect_avg:.1f}")

            result.time_total = time.time() - t_start

            print(f"\n[OK] BENCHMARK COMPLETE: {result.filename}")
            print(f"   Total time: {result.time_total:.2f}s")

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.time_total = time.time() - t_start
            print(f"\n[X] BENCHMARK FAILED: {e}")

        finally:
            gmsh.finalize()

        return result

    def run_benchmark_suite(self, files: List[str], target_size_mm: float = 3.0):
        """
        Run benchmark on multiple files

        Args:
            files: List of STEP file paths
            target_size_mm: Target mesh size in mm
        """
        print("\n" + "="*80)
        print("BATCH MESHING BENCHMARK SUITE")
        print("="*80)
        print(f"Files to process: {len(files)}")
        print(f"Target mesh size: {target_size_mm} mm")
        print(f"Output directory: {self.output_dir}")
        print("="*80)

        for i, file_path in enumerate(files, 1):
            print(f"\n{'='*80}")
            print(f"FILE {i}/{len(files)}: {Path(file_path).name}")
            print(f"{'='*80}")

            result = self.benchmark_file(file_path, target_size_mm)
            self.results.append(result)

        # Print summary
        self._print_summary()

        # Export results
        self._export_results()

    def _print_summary(self):
        """Print benchmark summary table"""
        print("\n\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)

        # Header
        print(f"\n{'File':<25} {'Success':<10} {'Time':<10} {'Elements':<12} {'SICN Min':<10} {'Quality':<10}")
        print("-"*80)

        # Results
        for r in self.results:
            success_str = "[OK] YES" if r.success else "[X] NO"
            quality_str = "GOOD" if r.sicn_min > 0.3 else ("OK" if r.sicn_min > 0.1 else "POOR")

            print(f"{r.filename:<25} {success_str:<10} {r.time_total:<10.2f} "
                  f"{r.num_elements:<12,} {r.sicn_min:<10.4f} {quality_str:<10}")

        # Statistics
        print("\n" + "-"*80)
        successful = sum(1 for r in self.results if r.success)
        total_time = sum(r.time_total for r in self.results)
        avg_time = total_time / len(self.results) if self.results else 0

        print(f"Success rate: {successful}/{len(self.results)} ({successful/len(self.results)*100:.1f}%)")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time per file: {avg_time:.2f}s")

        # Quality statistics (successful meshes only)
        successful_results = [r for r in self.results if r.success]
        if successful_results:
            avg_sicn = sum(r.sicn_min for r in successful_results) / len(successful_results)
            avg_skew = sum(r.skew_max for r in successful_results) / len(successful_results)

            print(f"\nQuality metrics (successful meshes):")
            print(f"  Average SICN min: {avg_sicn:.4f}")
            print(f"  Average skew max: {avg_skew:.4f}")

        print("="*80 + "\n")

    def _export_results(self):
        """Export results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"benchmark_{timestamp}.json"

        results_dict = {
            'timestamp': timestamp,
            'num_files': len(self.results),
            'results': []
        }

        for r in self.results:
            results_dict['results'].append({
                'filename': r.filename,
                'success': r.success,
                'time_total': r.time_total,
                'time_geometry': r.time_geometry,
                'time_meshing': r.time_meshing,
                'time_quality': r.time_quality,
                'num_surfaces': r.num_surfaces,
                'num_curves': r.num_curves,
                'num_volumes': r.num_volumes,
                'num_elements': r.num_elements,
                'num_nodes': r.num_nodes,
                'sicn_min': r.sicn_min,
                'sicn_avg': r.sicn_avg,
                'skew_max': r.skew_max,
                'skew_avg': r.skew_avg,
                'aspect_max': r.aspect_max,
                'aspect_avg': r.aspect_avg,
                'error_message': r.error_message
            })

        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"📊 Results exported to: {output_file}")


def main():
    """Main benchmark runner"""

    # Define test files (5 diverse geometries)
    cad_dir = Path(__file__).parent.parent.parent / "cad_files"

    test_files = [
        cad_dir / "Cube.step",           # 1. Simple baseline (fast)
        cad_dir / "Cylinder.step",       # 2. Simple curved (moderate)
        cad_dir / "Airfoil.step",        # 3. Complex with zero-thickness (healing test!)
        cad_dir / "Loft.step",           # 4. Lofted surfaces (sliver test)
        cad_dir / "ChamboRegina.step",   # 5. Very complex geometry
    ]

    # Filter to existing files
    existing_files = [str(f) for f in test_files if f.exists()]

    if not existing_files:
        print("[X] No test files found in cad_files directory!")
        print(f"   Looking for files in: {cad_dir}")
        return 1

    print(f"Found {len(existing_files)} test files:")
    for f in existing_files:
        print(f"  - {Path(f).name}")

    # Create benchmark runner
    benchmark = BatchMeshBenchmark(output_dir="benchmark_results")

    # Run benchmark suite
    benchmark.run_benchmark_suite(existing_files, target_size_mm=3.0)

    return 0


if __name__ == "__main__":
    sys.exit(main())
