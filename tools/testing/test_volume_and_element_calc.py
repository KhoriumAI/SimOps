#!/usr/bin/env python3
"""
Test script to verify volume calculation and element count targeting fixes
"""
import gmsh
import json

def test_volume_calculation(filepath):
    """Test that volume is calculated correctly with unit detection"""
    print(f"\n{'='*60}")
    print(f"Testing: {filepath}")
    print('='*60)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    try:
        gmsh.open(filepath)

        # Get bounding box
        bbox = gmsh.model.getBoundingBox(-1, -1)
        bbox_dims = [bbox[3]-bbox[0], bbox[4]-bbox[1], bbox[5]-bbox[2]]

        # Get volume using OCC's built-in method
        volumes_3d = gmsh.model.getEntities(dim=3)
        total_volume_raw = 0.0
        for vol_dim, vol_tag in volumes_3d:
            vol_mass = gmsh.model.occ.getMass(vol_dim, vol_tag)
            total_volume_raw += vol_mass

        print(f"\n📏 Raw Data:")
        print(f"   Bounding box: {bbox_dims[0]:.2f} x {bbox_dims[1]:.2f} x {bbox_dims[2]:.2f}")
        print(f"   Max dimension: {max(bbox_dims):.2f} units")
        print(f"   getMass() value: {total_volume_raw:.2f}")

        # Detect units
        if total_volume_raw > 10000:
            unit_scale = 0.001
            unit_name = "mm"
            print(f"\n[OK] Detected {unit_name} units (volume > 10000)")
        elif max(bbox_dims) > 1000:
            unit_scale = 0.001
            unit_name = "mm"
            print(f"\n[OK] Detected {unit_name} units (bbox > 1000)")
        else:
            unit_scale = 1.0
            unit_name = "m"
            print(f"\n[OK] Detected {unit_name} units")

        # Apply scaling
        total_volume = total_volume_raw * (unit_scale ** 3)

        print(f"\n📊 Final Result:")
        print(f"   Volume: {total_volume:.9f} m³")
        print(f"   Volume: {total_volume * 1e9:.2f} mm³")

        # Test element count calculations
        print(f"\n🔢 Calculated Element Targets:")

        def calc_sizes(target_elements):
            avg_size_m = (total_volume / target_elements) ** (1/3)  # meters
            avg_size_mm = avg_size_m * 1000  # convert to mm
            min_mm = max(0.1, avg_size_mm * 0.7)
            max_mm = max(0.5, avg_size_mm * 1.3)
            return int(target_elements), round(min_mm, 1), round(max_mm, 1)

        # Coarse: ~500-2000 elements
        target_coarse = max(500, min(2000, int(total_volume * 1e6)))
        count_c, min_c, max_c = calc_sizes(target_coarse)
        print(f"   Coarse:    {count_c:>6,} elements -> mesh size {min_c:.1f}-{max_c:.1f} mm")

        # Medium: ~2000-8000 elements
        target_medium = max(2000, min(8000, int(total_volume * 4e6)))
        count_m, min_m, max_m = calc_sizes(target_medium)
        print(f"   Medium:    {count_m:>6,} elements -> mesh size {min_m:.1f}-{max_m:.1f} mm")

        # Fine: ~8000-30000 elements
        target_fine = max(8000, min(30000, int(total_volume * 16e6)))
        count_f, min_f, max_f = calc_sizes(target_fine)
        print(f"   Fine:      {count_f:>6,} elements -> mesh size {min_f:.1f}-{max_f:.1f} mm")

        # Very Fine: ~30000-100000 elements
        target_vfine = max(30000, min(100000, int(total_volume * 64e6)))
        count_vf, min_vf, max_vf = calc_sizes(target_vfine)
        print(f"   Very Fine: {count_vf:>6,} elements -> mesh size {min_vf:.1f}-{max_vf:.1f} mm")

        gmsh.finalize()
        return True

    except Exception as e:
        print(f"\n[X] ERROR: {e}")
        gmsh.finalize()
        return False


if __name__ == "__main__":
    import sys

    test_files = [
        "CAD_files/Cylinder.step",
        "CAD_files/Cube.step",
        "CAD_files/Loft.step"
    ]

    if len(sys.argv) > 1:
        test_files = sys.argv[1:]

    print("\n" + "="*60)
    print("Volume Calculation & Element Count Test")
    print("="*60)

    results = []
    for filepath in test_files:
        try:
            success = test_volume_calculation(filepath)
            results.append((filepath, success))
        except FileNotFoundError:
            print(f"\n[!] File not found: {filepath}")
            results.append((filepath, False))

    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    for filepath, success in results:
        status = "[OK] PASS" if success else "[X] FAIL"
        print(f"{status} - {filepath}")

    print()
