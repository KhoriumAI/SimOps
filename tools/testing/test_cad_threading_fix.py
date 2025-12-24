#!/usr/bin/env python3
"""
Test script to verify the Gmsh threading fix in app_fixed.py
Tests the subprocess-based CAD info extraction
"""

import subprocess
import sys
import os
from pathlib import Path
from typing import Optional, Dict

def get_cad_info_without_viz(cad_file: str) -> Optional[Dict]:
    """
    Get CAD file info without visualization
    Uses subprocess to avoid Gmsh threading issues

    This is the same function from app_fixed.py
    """
    try:
        # Create a simple Python script to extract CAD info
        script = f"""
import gmsh
import sys

try:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("info")
    gmsh.model.occ.importShapes("{cad_file}")
    gmsh.model.occ.synchronize()

    surfaces = gmsh.model.getEntities(dim=2)
    curves = gmsh.model.getEntities(dim=1)
    volumes = gmsh.model.getEntities(dim=3)

    print(f"SURFACES: {{len(surfaces)}}")
    print(f"CURVES: {{len(curves)}}")
    print(f"VOLUMES: {{len(volumes)}}")

    bbox = gmsh.model.getBoundingBox(-1, -1)
    print(f"BBOX: {{bbox[0]:.3f}},{{bbox[1]:.3f}},{{bbox[2]:.3f}},{{bbox[3]:.3f}},{{bbox[4]:.3f}},{{bbox[5]:.3f}}")

    gmsh.finalize()
except Exception as e:
    print(f"ERROR: {{e}}", file=sys.stderr)
    sys.exit(1)
"""

        # Run in subprocess to avoid threading issues
        result = subprocess.run(
            ["python3", "-c", script],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            info = {}
            for line in result.stdout.split('\n'):
                if line.startswith('SURFACES:'):
                    info['surfaces'] = int(line.split(':')[1].strip())
                elif line.startswith('CURVES:'):
                    info['curves'] = int(line.split(':')[1].strip())
                elif line.startswith('VOLUMES:'):
                    info['volumes'] = int(line.split(':')[1].strip())
                elif line.startswith('BBOX:'):
                    parts = line.split(':')[1].strip().split(',')
                    info['bbox'] = [float(p) for p in parts]
            return info
        else:
            print(f"[X] Subprocess failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return None

    except subprocess.TimeoutExpired:
        print(f"[X] Subprocess timed out after 10 seconds")
        return None
    except Exception as e:
        print(f"[X] Exception: {e}")
        return None


def test_cad_file(cad_file: str):
    """Test loading a single CAD file"""
    print(f"\n{'='*60}")
    print(f"Testing: {Path(cad_file).name}")
    print(f"{'='*60}")

    if not os.path.exists(cad_file):
        print(f"[X] File not found: {cad_file}")
        return False

    print(f"📂 File path: {cad_file}")
    print(f"📦 File size: {os.path.getsize(cad_file) / 1024:.2f} KB")
    print()

    # Test the subprocess-based extraction
    print("🔧 Testing subprocess-based CAD info extraction...")
    info = get_cad_info_without_viz(cad_file)

    if info:
        print("[OK] SUCCESS! CAD info extracted without threading errors:")
        print(f"   * Surfaces: {info.get('surfaces', 'N/A')}")
        print(f"   * Curves: {info.get('curves', 'N/A')}")
        print(f"   * Volumes: {info.get('volumes', 'N/A')}")
        if 'bbox' in info:
            bbox = info['bbox']
            print(f"   * Bounding box:")
            print(f"     X: [{bbox[0]:.3f}, {bbox[3]:.3f}]")
            print(f"     Y: [{bbox[1]:.3f}, {bbox[4]:.3f}]")
            print(f"     Z: [{bbox[2]:.3f}, {bbox[5]:.3f}]")
        return True
    else:
        print("[X] FAILED: Could not extract CAD info")
        return False


def main():
    """Run tests on multiple CAD files"""
    print("="*60)
    print("GMSH THREADING FIX - TEST SUITE")
    print("="*60)
    print()
    print("This script tests the subprocess-based CAD info extraction")
    print("that avoids Gmsh threading issues in Streamlit.")
    print()

    # Test files
    test_files = [
        "cad_files/Cube.step",
        "cad_files/Cylinder.step",
        "cad_files/Airfoil.step"
    ]

    results = {}

    for test_file in test_files:
        full_path = os.path.join(os.getcwd(), test_file)
        if os.path.exists(full_path):
            results[test_file] = test_cad_file(full_path)
        else:
            print(f"\n[!]️  Skipping {test_file} (not found)")
            results[test_file] = None

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)

    for file, result in results.items():
        status = "[OK] PASS" if result is True else ("[X] FAIL" if result is False else "[!]️  SKIP")
        print(f"{status} - {file}")

    print()
    print(f"Passed:  {passed}/{len(test_files)}")
    print(f"Failed:  {failed}/{len(test_files)}")
    print(f"Skipped: {skipped}/{len(test_files)}")

    if failed == 0 and passed > 0:
        print()
        print("🎉 All tests passed! The threading fix works correctly.")
        print("   You can now use app_fixed.py without Gmsh threading errors.")
        return 0
    elif failed > 0:
        print()
        print("[!]️  Some tests failed. Review the errors above.")
        return 1
    else:
        print()
        print("[!]️  No tests were run. Check that CAD files exist.")
        return 2


if __name__ == "__main__":
    sys.exit(main())
