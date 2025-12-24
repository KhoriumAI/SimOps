"""
Test Script for Advanced Meshing Features
==========================================

Demonstrates and tests the three advanced meshing features:
1. Virtual topology for sliver faces (fillets/lofts)
2. Metric-driven refinement for small features
3. Boundary layer inflation for thin channels

Usage:
    python test_advanced_features.py
"""

import sys
import gmsh
import time
from pathlib import Path

from core.optimization_config import OptimizationConfig, QualityPreset
from strategies.advanced_strategy import AdvancedMeshingStrategy

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def create_test_geometry_fillet():
    """
    Create test geometry with fillet (sliver faces)

    This creates a cube with a large fillet at one edge,
    creating a problematic sliver surface.
    """
    print("\n" + "="*70)
    print("Creating Test Geometry: Fillet (Sliver Faces)")
    print("="*70)

    gmsh.initialize()
    gmsh.model.add("fillet_test")

    # Create box
    box = gmsh.model.occ.addBox(0, 0, 0, 10, 10, 10)

    # Add fillet to one edge (creates sliver face)
    edges = gmsh.model.occ.getEntities(dim=1)
    if edges:
        # Fillet first edge with large radius
        try:
            gmsh.model.occ.fillet([box], [edges[0][1]], [1.0], removeVolume=True)
            print("  [OK] Created fillet with radius 1.0 mm")
        except:
            print("  [!] Could not create fillet (geometry limitation)")

    gmsh.model.occ.synchronize()

    output_file = "test_fillet.step"
    gmsh.write(output_file)
    gmsh.finalize()

    print(f"  [OK] Saved to: {output_file}")
    return output_file


def create_test_geometry_small_features():
    """
    Create test geometry with small features

    This creates a box with a very small hole, testing
    small feature detection and refinement.
    """
    print("\n" + "="*70)
    print("Creating Test Geometry: Small Features")
    print("="*70)

    gmsh.initialize()
    gmsh.model.add("small_feature_test")

    # Create main box
    box = gmsh.model.occ.addBox(0, 0, 0, 20, 20, 20)

    # Add small cylinder (0.5mm diameter hole)
    small_cylinder = gmsh.model.occ.addCylinder(5, 10, 10, 10, 0, 0, radius=0.25)

    # Cut hole from box
    result, _ = gmsh.model.occ.cut([(3, box)], [(3, small_cylinder)])

    gmsh.model.occ.synchronize()

    output_file = "test_small_feature.step"
    gmsh.write(output_file)
    gmsh.finalize()

    print(f"  [OK] Created box with 0.5mm diameter hole")
    print(f"  [OK] Saved to: {output_file}")
    return output_file


def create_test_geometry_thin_channel():
    """
    Create test geometry with thin channel

    This creates two parallel plates with small gap,
    testing thin channel detection and boundary layers.
    """
    print("\n" + "="*70)
    print("Creating Test Geometry: Thin Channel")
    print("="*70)

    gmsh.initialize()
    gmsh.model.add("thin_channel_test")

    # Create two parallel plates with 2mm gap
    plate1 = gmsh.model.occ.addBox(0, 0, 0, 20, 20, 1)
    plate2 = gmsh.model.occ.addBox(0, 0, 3, 20, 20, 1)

    # Create channel volume between plates
    channel = gmsh.model.occ.addBox(0, 0, 1, 20, 20, 2)

    # Fuse all for single volume with thin channel
    gmsh.model.occ.fuse([(3, plate1)], [(3, plate2), (3, channel)])

    gmsh.model.occ.synchronize()

    output_file = "test_thin_channel.step"
    gmsh.write(output_file)
    gmsh.finalize()

    print(f"  [OK] Created parallel plates with 2mm gap")
    print(f"  [OK] Saved to: {output_file}")
    return output_file


def test_with_existing_cad():
    """
    Test advanced features with existing CAD files

    Uses files from cad_files/ directory if available.
    """
    print("\n" + "="*70)
    print("Testing with Existing CAD Files")
    print("="*70)

    cad_dir = Path("cad_files")
    if not cad_dir.exists():
        print("  [!] cad_files/ directory not found, skipping")
        return []

    test_files = list(cad_dir.glob("*.step")) + list(cad_dir.glob("*.stp"))

    if not test_files:
        print("  [!] No STEP files found in cad_files/, skipping")
        return []

    print(f"  [OK] Found {len(test_files)} STEP files")
    return [str(f) for f in test_files[:3]]  # Test first 3 files


def run_test(test_name: str, input_file: str, config: OptimizationConfig):
    """
    Run a single test case

    Args:
        test_name: Descriptive name for the test
        input_file: Path to STEP file
        config: Optimization configuration
    """
    print("\n" + "="*70)
    print(f"TEST: {test_name}")
    print("="*70)
    print(f"Input: {input_file}")
    print(f"Config: {config.__class__.__name__}")

    try:
        # Create strategy
        strategy = AdvancedMeshingStrategy(config=config, use_gpu=False)

        # Generate mesh
        start_time = time.time()
        result = strategy.generate_mesh(
            input_file=input_file,
            global_mesh_size=5.0
        )
        elapsed = time.time() - start_time

        # Print results
        if result.success:
            print(f"\n[OK] TEST PASSED")
            print(f"  Time: {elapsed:.1f}s")
            print(f"  Output: {result.output_file}")

            if 'sicn_min' in result.quality_metrics:
                print(f"  SICN min: {result.quality_metrics['sicn_min']:.3f}")
                print(f"  SICN mean: {result.quality_metrics['sicn_mean']:.3f}")
                print(f"  Elements: {result.quality_metrics.get('num_elements', 'N/A')}")

            if 'advanced_features' in result.quality_metrics:
                adv = result.quality_metrics['advanced_features']
                print(f"\n  Advanced Features Applied:")
                print(f"    * Virtual faces: {adv['virtual_faces_created']}")
                print(f"    * Size fields: {adv['size_fields_created']}")
                print(f"    * Boundary layers: {adv['boundary_layers_created']}")

            return True
        else:
            print(f"\n[X] TEST FAILED")
            print(f"  Error: {result.message}")
            return False

    except Exception as e:
        print(f"\n[X] TEST FAILED WITH EXCEPTION")
        print(f"  Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all test cases"""
    print("\n" + "="*70)
    print("ADVANCED MESHING FEATURES - TEST SUITE")
    print("="*70)
    print("\nTesting:")
    print("  1. Virtual Topology (Sliver Faces)")
    print("  2. Metric-Driven Refinement (Small Features)")
    print("  3. Boundary Layer Inflation (Thin Channels)")
    print("="*70)

    # Create test configurations
    print("\n[Configuration] Creating test configurations...")

    # Config 1: Virtual topology only
    config_virt = OptimizationConfig.from_preset(QualityPreset.MEDIUM)
    config_virt.advanced_features.enable_virtual_topology = True
    config_virt.advanced_features.enable_adaptive_refinement = False
    config_virt.advanced_features.enable_boundary_layers = False
    print("  [OK] Config 1: Virtual Topology Only")

    # Config 2: Adaptive refinement only
    config_adapt = OptimizationConfig.from_preset(QualityPreset.MEDIUM)
    config_adapt.advanced_features.enable_virtual_topology = False
    config_adapt.advanced_features.enable_adaptive_refinement = True
    config_adapt.advanced_features.enable_boundary_layers = False
    config_adapt.advanced_features.small_feature_threshold = 1.0  # Detect features < 1mm
    print("  [OK] Config 2: Adaptive Refinement Only")

    # Config 3: All features enabled (production)
    config_all = OptimizationConfig.from_preset(QualityPreset.PRODUCTION)
    print("  [OK] Config 3: All Features (Production)")

    # Generate test geometries
    print("\n[Test Geometry] Generating synthetic test cases...")
    test_cases = []

    try:
        fillet_file = create_test_geometry_fillet()
        test_cases.append(("Feature 1: Fillet (Sliver)", fillet_file, config_virt))
    except Exception as e:
        print(f"  [!] Failed to create fillet test: {e}")

    try:
        small_feat_file = create_test_geometry_small_features()
        test_cases.append(("Feature 2: Small Feature", small_feat_file, config_adapt))
    except Exception as e:
        print(f"  [!] Failed to create small feature test: {e}")

    try:
        thin_channel_file = create_test_geometry_thin_channel()
        test_cases.append(("Feature 3: Thin Channel", thin_channel_file, config_all))
    except Exception as e:
        print(f"  [!] Failed to create thin channel test: {e}")

    # Add existing CAD files
    existing_files = test_with_existing_cad()
    for i, file in enumerate(existing_files, 1):
        test_cases.append((f"Real CAD File {i}", file, config_all))

    # Run all tests
    print(f"\n[Testing] Running {len(test_cases)} test cases...")

    results = []
    for test_name, input_file, config in test_cases:
        passed = run_test(test_name, input_file, config)
        results.append((test_name, passed))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    print(f"\nResults: {passed_count}/{total_count} tests passed")

    for test_name, passed in results:
        status = "[OK] PASS" if passed else "[X] FAIL"
        print(f"  {status}: {test_name}")

    print("\n" + "="*70)

    if passed_count == total_count:
        print("[OK] ALL TESTS PASSED!")
    elif passed_count > 0:
        print(f"[!] {total_count - passed_count} test(s) failed")
    else:
        print("[X] ALL TESTS FAILED")

    print("="*70 + "\n")

    return passed_count == total_count


if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║           ADVANCED MESHING FEATURES - TEST SUITE                      ║
║                                                                       ║
║  This script tests the three production-grade meshing features:      ║
║                                                                       ║
║  1. Virtual Topology & Mesh Projection                               ║
║     * Detects sliver faces from fillets/lofts                        ║
║     * Creates composite virtual faces                                ║
║     * Eliminates skewed elements                                     ║
║                                                                       ║
║  2. Metric-Driven Adaptive Refinement                                ║
║     * Detects small features (< 1mm)                                 ║
║     * Creates localized size fields                                  ║
║     * Anisotropic mesh grading                                       ║
║                                                                       ║
║  3. Medial Axis & Boundary Layer Inflation                           ║
║     * Detects thin channels (< 5mm gap)                              ║
║     * Generates structured prism layers                              ║
║     * Resolves boundary layers properly                              ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
""")

    success = main()
    sys.exit(0 if success else 1)
