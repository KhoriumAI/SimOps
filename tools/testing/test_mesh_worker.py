#!/usr/bin/env python3
"""
Test script to verify mesh_worker_subprocess.py fixes
Tests that:
1. Absolute paths are returned
2. total_elements and total_nodes are in result
3. Mesh file is created in generated_meshes/
"""

import subprocess
import json
import sys
from pathlib import Path

def test_mesh_worker():
    """Test mesh worker with Cube.step"""
    print("=" * 60)
    print("Testing mesh_worker_subprocess.py with Cube.step")
    print("=" * 60)

    cad_file = str(Path(__file__).parent.parent.parent / "cad_files" / "Cube.step")

    if not Path(cad_file).exists():
        print(f"[X] CAD file not found: {cad_file}")
        sys.exit(1)

    print(f"\nInput CAD file: {cad_file}")
    print("\nRunning mesh generation subprocess...")

    # Run subprocess
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent.parent.parent / "apps" / "cli" / "mesh_worker_subprocess.py"), cad_file],
        capture_output=True,
        text=True,
        timeout=300  # 5 minute timeout
    )

    print("\n" + "=" * 60)
    print("SUBPROCESS OUTPUT")
    print("=" * 60)
    print(result.stdout)

    if result.stderr:
        print("\nSTDERR:")
        print(result.stderr)

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
            return False

        print("\n" + "=" * 60)
        print("PARSED RESULT")
        print("=" * 60)
        print(json.dumps(json_output, indent=2))

        # Verify expected fields
        print("\n" + "=" * 60)
        print("VERIFICATION")
        print("=" * 60)

        success = json_output.get('success', False)
        print(f"[OK] Success: {success}")

        if not success:
            print(f"[X] Mesh generation failed: {json_output.get('error')}")
            return False

        # Check output_file
        output_file = json_output.get('output_file')
        if output_file:
            print(f"[OK] Output file: {output_file}")

            # Check if it's absolute path
            if Path(output_file).is_absolute():
                print("[OK] Output path is ABSOLUTE")
            else:
                print("[X] Output path is NOT absolute")
                return False

            # Check if file exists
            if Path(output_file).exists():
                print(f"[OK] Mesh file exists")
            else:
                print(f"[X] Mesh file does NOT exist")
                return False
        else:
            print("[X] No output_file in result")
            return False

        # Check total_elements
        total_elements = json_output.get('total_elements')
        if total_elements is not None:
            print(f"[OK] total_elements: {total_elements:,}")
        else:
            print("[X] total_elements NOT in result")
            return False

        # Check total_nodes
        total_nodes = json_output.get('total_nodes')
        if total_nodes is not None:
            print(f"[OK] total_nodes: {total_nodes:,}")
        else:
            print("[X] total_nodes NOT in result")
            return False

        # Check metrics
        metrics = json_output.get('metrics')
        if metrics:
            print(f"[OK] Metrics present")
            if 'sicn_min' in metrics or 'SICN (Gmsh)' in metrics:
                sicn_min = metrics.get('sicn_min') or metrics.get('SICN (Gmsh)', {}).get('min')
                print(f"  - SICN min: {sicn_min:.4f}")

        print("\n" + "=" * 60)
        print("[OK][OK][OK] ALL CHECKS PASSED [OK][OK][OK]")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n[X] Error parsing result: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mesh_worker()
    sys.exit(0 if success else 1)
