#!/usr/bin/env python3
"""
Compute Backend Benchmark Script

Compares preview generation times across different compute backends
(local GMSH vs SSH tunnel to Threadripper) for CAD files of varying sizes.

Usage:
    python benchmark_compute.py
    python benchmark_compute.py --files path/to/file1.step path/to/file2.step
    python benchmark_compute.py --iterations 5

Output:
    - Console table with timing results
    - JSON file with detailed results (benchmark_results.json)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from backend.compute_backend import (
    LocalGMSHBackend,
    SSHTunnelBackend,
    HTTPRemoteBackend,
    get_available_backends,
    get_preferred_backend
)


# Default test files in order of size
DEFAULT_TEST_FILES = [
    "cad_files/Cube.step",                                    # ~8 KB - trivial
    "cad_files/tesla_valve_benchmark_single.step",            # ~400 KB - small
    "cad_files/00010009_d97409455fa543b3a224250f_step_000.step",  # ~1.7 MB - medium  
    "cad_files/ChamboRegina.step",                            # ~3.7 MB - large
]


def format_time(seconds: float) -> str:
    """Format time in human-readable format"""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"


def format_size(bytes: int) -> str:
    """Format file size in human-readable format"""
    if bytes < 1024:
        return f"{bytes} B"
    elif bytes < 1024 * 1024:
        return f"{bytes/1024:.1f} KB"
    else:
        return f"{bytes/(1024*1024):.2f} MB"


def print_table(results: list):
    """Print results as a formatted table"""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    
    # Header
    print(f"{'File':<30} {'Size':<10} {'Backend':<25} {'Avg Time':<12} {'Status'}")
    print("-" * 80)
    
    for result in results:
        file_name = Path(result['file']).name[:28]
        file_size = format_size(result['file_size'])
        
        for backend_result in result['backends']:
            backend_name = backend_result['backend'][:23]
            
            if 'error' in backend_result:
                status = f"ERROR: {backend_result['error'][:20]}..."
                avg_time = "N/A"
            else:
                status = "OK"
                avg_time = format_time(backend_result['avg_time'])
            
            print(f"{file_name:<30} {file_size:<10} {backend_name:<25} {avg_time:<12} {status}")
        
        print("-" * 80)


def print_recommendation(results: list):
    """Print recommendation based on benchmark results"""
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    
    ssh_times = []
    local_times = []
    
    for result in results:
        for backend_result in result['backends']:
            if 'avg_time' not in backend_result:
                continue
            
            if 'ssh_tunnel' in backend_result['backend']:
                ssh_times.append(backend_result['avg_time'])
            elif 'local' in backend_result['backend']:
                local_times.append(backend_result['avg_time'])
    
    if not ssh_times and not local_times:
        print("❌ No successful benchmark results. Cannot make recommendation.")
        return
    
    if not ssh_times:
        print("⚠️  SSH tunnel backend not available. Using local GMSH by default.")
        print("   To enable SSH tunnel, ensure: ssh -L 8080:localhost:8080 user@aws-instance")
        return
    
    if not local_times:
        print("⚠️  Local GMSH not available. Using SSH tunnel by default.")
        return
    
    avg_ssh = sum(ssh_times) / len(ssh_times)
    avg_local = sum(local_times) / len(local_times)
    diff = avg_ssh - avg_local
    
    print(f"Average SSH Tunnel time: {format_time(avg_ssh)}")
    print(f"Average Local GMSH time: {format_time(avg_local)}")
    print(f"Difference: {format_time(abs(diff))} {'faster' if diff > 0 else 'slower'} with Local GMSH")
    print()
    
    if abs(diff) < 2.0:  # Within 2 seconds
        print("✅ Times are similar (within 2s). Recommend using LOCAL compute for:")
        print("   - Lower latency (no network round-trip)")
        print("   - No SSH tunnel dependency")
        print()
        print("   Set: COMPUTE_BACKEND=local")
    elif diff > 0:  # Local is faster
        print("✅ Local GMSH is faster. Recommend switching all compute to local.")
        print()
        print("   Set: COMPUTE_BACKEND=local")
    else:  # SSH is faster
        print("✅ SSH tunnel (Threadripper) is faster. Keep current architecture.")
        print()
        print("   Set: COMPUTE_BACKEND=ssh_tunnel")


def run_benchmark(test_files: list, iterations: int = 3, output_file: str = None):
    """Run benchmark on all backends with all test files"""
    project_root = Path(__file__).parent.parent
    
    # Get available backends
    backends = [
        SSHTunnelBackend(),
        LocalGMSHBackend()
    ]
    
    print(f"Benchmarking {len(test_files)} files with {len(backends)} backends...")
    print(f"Iterations per test: {iterations}")
    print()
    
    # Check backend availability
    print("Backend availability:")
    for backend in backends:
        available = backend.is_available()
        status = "✅ AVAILABLE" if available else "❌ UNAVAILABLE"
        print(f"  {backend.name}: {status}")
    print()
    
    results = []
    
    for file_path in test_files:
        full_path = project_root / file_path
        
        if not full_path.exists():
            print(f"⚠️  Skipping {file_path} (file not found)")
            continue
        
        file_size = full_path.stat().st_size
        print(f"Testing: {file_path} ({format_size(file_size)})")
        
        file_result = {
            "file": str(file_path),
            "file_size": file_size,
            "backends": []
        }
        
        for backend in backends:
            print(f"  → {backend.name}...", end=" ", flush=True)
            
            if not backend.is_available():
                print("SKIPPED (unavailable)")
                file_result["backends"].append({
                    "backend": backend.name,
                    "error": "Backend unavailable"
                })
                continue
            
            # Run benchmark
            bench_result = backend.benchmark(str(full_path), iterations=iterations)
            file_result["backends"].append(bench_result)
            
            if "error" in bench_result:
                print(f"ERROR: {bench_result['error'][:50]}")
            else:
                print(f"{format_time(bench_result['avg_time'])} avg ({iterations} iterations)")
        
        results.append(file_result)
        print()
    
    # Add metadata
    output = {
        "timestamp": datetime.now().isoformat(),
        "iterations": iterations,
        "results": results
    }
    
    # Save to JSON
    if output_file is None:
        output_file = project_root / "benchmark_results.json"
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    
    # Print summary table
    print_table(results)
    
    # Print recommendation
    print_recommendation(results)
    
    return output


def main():
    parser = argparse.ArgumentParser(description="Benchmark compute backends for preview generation")
    parser.add_argument(
        "--files", 
        nargs="+", 
        default=DEFAULT_TEST_FILES,
        help="CAD files to benchmark (relative to project root)"
    )
    parser.add_argument(
        "--iterations", 
        type=int, 
        default=3,
        help="Number of iterations per test (default: 3)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Output JSON file path (default: benchmark_results.json)"
    )
    
    args = parser.parse_args()
    
    run_benchmark(
        test_files=args.files,
        iterations=args.iterations,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
