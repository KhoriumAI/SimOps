#!/usr/bin/env python3
"""
Test script for Airfoil.step with sharp feature smoothing
"""

import subprocess
import json
import sys
from pathlib import Path

def test_airfoil():
    """Test mesh worker with Airfoil.step"""
    print("=" * 60)
    print("Testing Airfoil.step with Sharp Feature Smoothing")
    print("=" * 60)

    cad_file = str(Path(__file__).parent.parent.parent / "cad_files" / "Airfoil.step")

    if not Path(cad_file).exists():
        print(f"[X] CAD file not found: {cad_file}")
        sys.exit(1)

    print(f"\nInput CAD file: {cad_file}")
    print("\nRunning mesh generation subprocess...")
    print("(This may take a few minutes...)\n")

    # Run subprocess
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent.parent.parent / "apps" / "cli" / "mesh_worker_subprocess.py"), cad_file],
        capture_output=True,
        text=True,
        timeout=600  # 10 minute timeout for complex geometry
    )

    # Parse result
    try:
        # Find the JSON output (last line should be JSON)
        lines = result.stdout.strip().split('\n')
        json_output = None
        for line in reversed(lines):
            try:
                json_output = json.loads(line)
                break
            except:
                continue

        if not json_output:
            print("\n[X] No JSON output found")
            print("\nFull output:")
            print(result.stdout)
            return False

        print("=" * 60)
        print("AIRFOIL MESH QUALITY RESULTS")
        print("=" * 60)

        success = json_output.get('success', False)
        if not success:
            print(f"[X] Mesh generation failed: {json_output.get('error')}")
            return False

        print(f"[OK] Mesh generation SUCCESS")
        print(f"\nStrategy: {json_output.get('strategy')}")
        print(f"Quality Score: {json_output.get('score', 0):.2f}")

        metrics = json_output.get('metrics', {})

        # Check SICN (primary quality metric)
        sicn_min = metrics.get('sicn_min') or (metrics.get('SICN (Gmsh)') or metrics.get('gmsh_sicn', {})).get('min')
        sicn_avg = metrics.get('sicn_avg') or (metrics.get('SICN (Gmsh)') or metrics.get('gmsh_sicn', {})).get('avg')

        print(f"\n--- PRIMARY QUALITY METRICS (Gmsh) ---")
        if sicn_min is not None:
            print(f"SICN min: {sicn_min:.4f}", end="")
            if sicn_min >= 0.3:
                print(" [OK] GOOD (>= 0.3)")
            elif sicn_min >= 0.1:
                print(" [!] FAIR (>= 0.1)")
            else:
                print(f" [X] POOR (< 0.1)")

        if sicn_avg is not None:
            print(f"SICN avg: {sicn_avg:.4f}")

        # Check Gamma
        gamma_min = metrics.get('gamma_min') or (metrics.get('Gamma (Gmsh)') or metrics.get('gmsh_gamma', {})).get('min')
        if gamma_min is not None:
            print(f"Gamma min: {gamma_min:.4f}", end="")
            if gamma_min >= 0.2:
                print(" [OK] GOOD (>= 0.2)")
            else:
                print(f" [!] FAIR (< 0.2)")

        # Check converted metrics
        print(f"\n--- SECONDARY METRICS (converted) ---")
        skewness_max = metrics.get('max_skewness') or (metrics.get('Skewness (converted)') or metrics.get('skewness', {})).get('max')
        if skewness_max is not None:
            print(f"Skewness max: {skewness_max:.4f}", end="")
            if skewness_max <= 0.7:
                print(" [OK] EXCELLENT (<= 0.7)")
            elif skewness_max <= 0.85:
                print(" [!] ACCEPTABLE (<= 0.85)")
            else:
                print(f" [X] POOR (> 0.85)")

        aspect_max = metrics.get('max_aspect_ratio') or (metrics.get('Aspect Ratio (converted)') or metrics.get('aspect_ratio', {})).get('max')
        if aspect_max is not None:
            print(f"Aspect Ratio max: {aspect_max:.2f}", end="")
            if aspect_max <= 5.0:
                print(" [OK] EXCELLENT (<= 5.0)")
            elif aspect_max <= 10.0:
                print(" [!] ACCEPTABLE (<= 10.0)")
            else:
                print(f" [X] POOR (> 10.0)")

        # Element counts
        total_elements = json_output.get('total_elements', 0)
        total_nodes = json_output.get('total_nodes', 0)
        print(f"\n--- MESH SIZE ---")
        print(f"Total elements: {total_elements:,}")
        print(f"Total nodes: {total_nodes:,}")

        # Overall assessment
        print(f"\n{'=' * 60}")
        if sicn_min and sicn_min >= 0.3:
            print("[OK][OK][OK] EXCELLENT QUALITY - Sharp feature smoothing worked!")
        elif sicn_min and sicn_min >= 0.1:
            print("[!] FAIR QUALITY - Improvements made, but room for more")
        else:
            print("[X] POOR QUALITY - Sharp features still problematic")
        print(f"{'=' * 60}")

        return True

    except Exception as e:
        print(f"\n[X] Error parsing result: {e}")
        import traceback
        traceback.print_exc()
        print("\nFull output:")
        print(result.stdout)
        return False

if __name__ == "__main__":
    success = test_airfoil()
    sys.exit(0 if success else 1)
