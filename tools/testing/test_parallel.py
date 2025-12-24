#!/usr/bin/env python3
"""
Test Script for Parallel Mesh Generation
=========================================

Quick test to verify parallel execution is working and measure speedup.

Usage:
    python test_parallel.py <cad_file.step>

Example:
    python test_parallel.py cad_files/Cube.step
"""

import sys
import time
from pathlib import Path
from strategies.exhaustive_strategy import ExhaustiveMeshGenerator
from core.config import Config

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_parallel_vs_sequential(cad_file: str):
    """
    Test parallel vs sequential execution and measure speedup

    Args:
        cad_file: Path to CAD file to mesh
    """
    print("=" * 70)
    print("PARALLEL EXECUTION PERFORMANCE TEST")
    print("=" * 70)
    print(f"CAD File: {cad_file}")
    print()

    # Test 1: Parallel execution
    print("[ROCKET] Test 1: PARALLEL EXECUTION")
    print("-" * 70)

    config_parallel = Config()
    generator_parallel = ExhaustiveMeshGenerator(config_parallel, use_parallel=True)
    output_parallel = cad_file.replace('.step', '_parallel.msh')

    start_time = time.time()
    success_parallel = generator_parallel.run_meshing_strategy(cad_file, output_parallel)
    parallel_time = time.time() - start_time

    if success_parallel:
        print(f"[OK] Parallel execution completed in {parallel_time:.2f} seconds")
    else:
        print(f"[X] Parallel execution failed")

    print()

    # Test 2: Sequential execution (for comparison)
    print("🐌 Test 2: SEQUENTIAL EXECUTION (for comparison)")
    print("-" * 70)
    print("Note: This will run strategies one at a time (slower)")
    print("Press Ctrl+C to skip this test if you don't want to wait")
    print()

    try:
        input("Press Enter to continue with sequential test (or Ctrl+C to skip)...")

        config_sequential = Config()
        generator_sequential = ExhaustiveMeshGenerator(config_sequential, use_parallel=False)
        output_sequential = cad_file.replace('.step', '_sequential.msh')

        start_time = time.time()
        success_sequential = generator_sequential.run_meshing_strategy(cad_file, output_sequential)
        sequential_time = time.time() - start_time

        if success_sequential:
            print(f"[OK] Sequential execution completed in {sequential_time:.2f} seconds")
        else:
            print(f"[X] Sequential execution failed")

        # Calculate speedup
        print()
        print("=" * 70)
        print("PERFORMANCE COMPARISON")
        print("=" * 70)
        print(f"Parallel time:   {parallel_time:.2f}s")
        print(f"Sequential time: {sequential_time:.2f}s")

        if success_parallel and success_sequential:
            speedup = sequential_time / parallel_time
            print(f"Speedup:         {speedup:.2f}x faster")
            print()

            if speedup >= 3.0:
                print("[OK] EXCELLENT: Achieved 3x+ speedup!")
            elif speedup >= 2.0:
                print("[OK] GOOD: Achieved 2x+ speedup")
            elif speedup >= 1.5:
                print("[!] MODERATE: 1.5x+ speedup (may vary by geometry)")
            else:
                print("[!] LIMITED: Speedup less than expected")
                print("  This can happen with very simple geometries that mesh quickly")

    except KeyboardInterrupt:
        print("\n\nSequential test skipped.")
        print(f"\n[OK] Parallel execution took {parallel_time:.2f} seconds")

    print()
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

    if success_parallel:
        print(f"[OK] Parallel mesh saved to: {output_parallel}")
        print("\nYou can view the mesh with:")
        print(f"  gmsh {output_parallel}")

    return success_parallel


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_parallel.py <cad_file.step>")
        print()
        print("Example:")
        print("  python test_parallel.py cad_files/Cube.step")
        sys.exit(1)

    cad_file = sys.argv[1]

    if not Path(cad_file).exists():
        print(f"Error: File not found: {cad_file}")
        sys.exit(1)

    success = test_parallel_vs_sequential(cad_file)
    sys.exit(0 if success else 1)
