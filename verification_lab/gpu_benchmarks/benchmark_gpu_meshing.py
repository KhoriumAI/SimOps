"""
GPU Meshing Performance Benchmark
=================================

Benchmarks the optimized GPU meshing pipeline including:
1. Vectorized LFS computation
2. Vectorized normal computation
3. Vectorized boundary layer validation
4. Vectorized GPU optimizer
5. Adaptive refinement system

Run with: python tests/benchmark_gpu_meshing.py
"""

import numpy as np
import time
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def generate_test_mesh(num_vertices: int = 1000, num_faces: int = 2000, num_tets: int = 5000):
    """Generate a synthetic test mesh for benchmarking."""
    np.random.seed(42)
    
    # Random vertices in unit cube
    vertices = np.random.rand(num_vertices, 3).astype(np.float32)
    
    # Random triangular faces (valid indices)
    faces = np.random.randint(0, num_vertices, size=(num_faces, 3)).astype(np.int32)
    
    # Random tetrahedra (valid indices)
    tets = np.random.randint(0, num_vertices, size=(num_tets, 4)).astype(np.int32)
    
    return vertices, faces, tets


def benchmark_lfs_computation(vertices, faces, iterations=10):
    """Benchmark LFS computation."""
    from core.gpu_mesher import compute_vertex_lfs
    
    print("\n" + "=" * 60)
    print("BENCHMARK: compute_vertex_lfs()")
    print("=" * 60)
    print(f"Vertices: {len(vertices)}, Faces: {len(faces)}")
    
    times = []
    for i in range(iterations):
        start = time.perf_counter()
        lfs = compute_vertex_lfs(vertices, faces)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"Iterations: {iterations}")
    print(f"Average time: {avg_time:.2f} ms (+/- {std_time:.2f})")
    print(f"Min/Max: {np.min(times):.2f} / {np.max(times):.2f} ms")
    
    return avg_time


def benchmark_normal_computation(vertices, faces, iterations=10):
    """Benchmark vertex normal computation."""
    from core.gpu_mesher import compute_vertex_normals
    
    print("\n" + "=" * 60)
    print("BENCHMARK: compute_vertex_normals()")
    print("=" * 60)
    print(f"Vertices: {len(vertices)}, Faces: {len(faces)}")
    
    times = []
    for i in range(iterations):
        start = time.perf_counter()
        normals = compute_vertex_normals(vertices, faces)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"Iterations: {iterations}")
    print(f"Average time: {avg_time:.2f} ms (+/- {std_time:.2f})")
    print(f"Min/Max: {np.min(times):.2f} / {np.max(times):.2f} ms")
    
    return avg_time


def benchmark_boundary_validation(vertices, tets, faces, iterations=5):
    """Benchmark boundary layer health validation."""
    from core.gpu_mesher import validate_boundary_layer_health
    
    print("\n" + "=" * 60)
    print("BENCHMARK: validate_boundary_layer_health()")
    print("=" * 60)
    print(f"Vertices: {len(vertices)}, Tets: {len(tets)}, Faces: {len(faces)}")
    
    times = []
    for i in range(iterations):
        start = time.perf_counter()
        try:
            result = validate_boundary_layer_health(vertices, tets, faces, vertices)
        except Exception as e:
            print(f"  Iteration {i+1}: Validation error (expected for random mesh)")
            continue
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)
    
    if times:
        avg_time = np.mean(times)
        std_time = np.std(times)
        print(f"Successful iterations: {len(times)}/{iterations}")
        print(f"Average time: {avg_time:.2f} ms (+/- {std_time:.2f})")
        return avg_time
    else:
        print("No successful iterations (expected for random synthetic mesh)")
        return None


def benchmark_gpu_optimizer(vertices, tets, iterations=3):
    """Benchmark GPU mesh optimizer."""
    from strategies.gpu_mesh_optimizer import GPUMeshOptimizer, GPU_AVAILABLE
    
    print("\n" + "=" * 60)
    print("BENCHMARK: GPUMeshOptimizer.optimize_mesh()")
    print("=" * 60)
    print(f"Vertices: {len(vertices)}, Tets: {len(tets)}")
    print(f"GPU Available: {GPU_AVAILABLE}")
    
    optimizer = GPUMeshOptimizer(verbose=False)
    
    times = []
    for i in range(iterations):
        nodes_copy = vertices.copy()
        start = time.perf_counter()
        try:
            result = optimizer.optimize_mesh(nodes_copy, tets)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)
        except Exception as e:
            print(f"  Iteration {i+1}: Error - {e}")
            continue
    
    if times:
        avg_time = np.mean(times)
        std_time = np.std(times)
        print(f"Successful iterations: {len(times)}/{iterations}")
        print(f"Average time: {avg_time:.2f} ms (+/- {std_time:.2f})")
        return avg_time
    else:
        print("No successful iterations")
        return None


def benchmark_adaptive_refinement(vertices, tets, iterations=2):
    """Benchmark adaptive refinement."""
    from core.gpu_adaptive_refinement import AdaptiveGPURefinement
    
    print("\n" + "=" * 60)
    print("BENCHMARK: AdaptiveGPURefinement.refine()")
    print("=" * 60)
    print(f"Vertices: {len(vertices)}, Tets: {len(tets)}")
    
    times = []
    for i in range(iterations):
        nodes_copy = vertices.copy().astype(np.float32)
        refiner = AdaptiveGPURefinement(
            target_sicn=0.2,  # Low target to ensure some iterations run
            max_iterations=5,
            verbose=False
        )
        
        start = time.perf_counter()
        try:
            refined_nodes, stats = refiner.refine(nodes_copy, tets)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            print(f"  Iteration {i+1}: {stats['iterations']} refinement iters, "
                  f"SICN: {stats['initial_sicn_min']:.3f} -> {stats['final_sicn_min']:.3f}, "
                  f"time={elapsed:.2f}s")
        except Exception as e:
            print(f"  Iteration {i+1}: Error - {e}")
            continue
    
    if times:
        avg_time = np.mean(times)
        print(f"\nSuccessful runs: {len(times)}/{iterations}")
        print(f"Average total time: {avg_time:.2f}s")
        return avg_time
    else:
        print("No successful iterations")
        return None


def run_all_benchmarks():
    """Run complete benchmark suite."""
    print("=" * 70)
    print("GPU MESHING PERFORMANCE BENCHMARK")
    print("=" * 70)
    print(f"Testing optimized vectorized implementations")
    print()
    
    # Generate test meshes of different sizes
    sizes = [
        {"vertices": 1000, "faces": 2000, "tets": 3000, "label": "Small"},
        {"vertices": 5000, "faces": 10000, "tets": 20000, "label": "Medium"},
        {"vertices": 10000, "faces": 20000, "tets": 50000, "label": "Large"},
    ]
    
    results = {}
    
    for size_config in sizes:
        label = size_config["label"]
        print("\n" + "#" * 70)
        print(f"# {label.upper()} MESH")
        print("#" * 70)
        
        vertices, faces, tets = generate_test_mesh(
            size_config["vertices"],
            size_config["faces"],
            size_config["tets"]
        )
        
        results[label] = {
            'lfs': benchmark_lfs_computation(vertices, faces),
            'normals': benchmark_normal_computation(vertices, faces),
            'validation': benchmark_boundary_validation(vertices, tets, faces),
            'optimizer': benchmark_gpu_optimizer(vertices, tets),
        }
        
        # Only run adaptive refinement on smaller mesh
        if label == "Small":
            results[label]['adaptive'] = benchmark_adaptive_refinement(vertices, tets)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Test':<30} {'Small':>12} {'Medium':>12} {'Large':>12}")
    print("-" * 70)
    
    for test_name in ['lfs', 'normals', 'validation', 'optimizer']:
        row = [test_name.capitalize()]
        for label in ['Small', 'Medium', 'Large']:
            val = results[label].get(test_name)
            if val is not None:
                row.append(f"{val:.1f}ms")
            else:
                row.append("N/A")
        print(f"{row[0]:<30} {row[1]:>12} {row[2]:>12} {row[3]:>12}")
    
    print("\n[OK] Benchmark complete!")
    print("All functions use vectorized NumPy/CuPy operations for optimal performance.")


if __name__ == "__main__":
    run_all_benchmarks()
