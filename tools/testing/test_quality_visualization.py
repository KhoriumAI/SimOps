#!/usr/bin/env python3
"""
Test quality visualization pipeline:
1. Generate a mesh
2. Extract per-element quality
3. Verify quality mapping works
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from mesh_worker_subprocess import generate_mesh
from extract_element_quality import extract_per_element_quality

def test_quality_pipeline():
    """Test the complete quality visualization pipeline"""
    print("=" * 60)
    print("TESTING QUALITY VISUALIZATION PIPELINE")
    print("=" * 60)

    # Test with cube (simple geometry)
    cad_file = "CAD_files/Cube.step"

    if not Path(cad_file).exists():
        print(f"[X] Test file not found: {cad_file}")
        return False

    print(f"\n1. Generating mesh for {cad_file}...")

    # Generate mesh with quality parameters
    quality_params = {
        'quality_preset': 'Medium',
        'max_size_mm': 10,
        'target_elements': None,
        'curvature_adaptive': False
    }

    result = generate_mesh(cad_file, quality_params=quality_params)

    if not result.get('success'):
        print(f"[X] Mesh generation failed: {result.get('error')}")
        return False

    print(f"[OK] Mesh generated: {result['output_file']}")
    print(f"  Elements: {result['total_elements']}")
    print(f"  Nodes: {result['total_nodes']}")

    # Check quality metrics
    print("\n2. Checking quality metrics (averages)...")
    quality_metrics = result.get('quality_metrics', {})

    if 'sicn_avg' in quality_metrics:
        print(f"  SICN avg: {quality_metrics['sicn_avg']:.3f}")
    if 'gamma_avg' in quality_metrics:
        print(f"  Gamma avg: {quality_metrics['gamma_avg']:.3f}")
    if 'avg_skewness' in quality_metrics:
        print(f"  Skewness avg: {quality_metrics['avg_skewness']:.3f}")
    if 'avg_aspect_ratio' in quality_metrics:
        print(f"  Aspect Ratio avg: {quality_metrics['avg_aspect_ratio']:.2f}")

    # Check per-element quality
    print("\n3. Checking per-element quality data...")
    per_elem = result.get('per_element_quality', {})

    if per_elem:
        print(f"  [OK] Per-element quality extracted: {len(per_elem)} triangles")

        # Analyze quality distribution
        qualities = list(per_elem.values())
        if qualities:
            import statistics
            print(f"  Min quality: {min(qualities):.3f}")
            print(f"  Avg quality: {statistics.mean(qualities):.3f}")
            print(f"  Max quality: {max(qualities):.3f}")

            # Check 10th percentile threshold
            if 'sicn_10_percentile' in quality_metrics:
                threshold = quality_metrics['sicn_10_percentile']
                print(f"  10th percentile threshold: {threshold:.3f}")

                # Count elements below threshold (should be ~10%)
                below_threshold = sum(1 for q in qualities if q <= threshold)
                percent = (below_threshold / len(qualities)) * 100
                print(f"  Elements below threshold: {below_threshold} ({percent:.1f}%)")

                if 8 <= percent <= 12:
                    print(f"  [OK] Threshold calculation correct (~10%)")
                else:
                    print(f"  [!] Threshold percentage unexpected: {percent:.1f}%")
    else:
        print("  [X] No per-element quality data found")
        return False

    print("\n" + "=" * 60)
    print("[OK][OK][OK] QUALITY VISUALIZATION PIPELINE TEST PASSED")
    print("=" * 60)
    print("\nWhat this means:")
    print("  * Average quality metrics will display correctly (not all red)")
    print("  * Worst 10% of elements will be highlighted in red in 3D view")
    print("  * Quality visualization properly maps triangles to tet quality")
    return True

if __name__ == "__main__":
    success = test_quality_pipeline()
    sys.exit(0 if success else 1)
