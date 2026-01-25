"""
Regenerate 3D thermal visualizations using matplotlib fallback
"""
import sys
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
from core.reporting.thermal_viz_matplotlib3d import generate_thermal_views_matplotlib

# Test case 1: Original test file
print("=" * 70)
print("TEST 1: Regenerating fddbfe8a visualization")
print("=" * 70)

vtk_file_1 = Path(r"c:\Users\markm\Downloads\Simops\simops_output\fddbfe8a\thermal_result.vtk")
if vtk_file_1.exists():
    output_dir_1 = vtk_file_1.parent
    job_name_1 = "fddbfe8a_test"

    image_paths = generate_thermal_views_matplotlib(
        str(vtk_file_1),
        output_dir_1,
        job_name_1,
        views=['isometric', 'top', 'front'],
        colormap='coolwarm'
    )

    print(f"\nGenerated {len(image_paths)} images:")
    for img in image_paths:
        print(f"  - {img}")
else:
    print("File not found")

# Test case 2: The actual problematic file (daaf205a)
print("\n" + "=" * 70)
print("TEST 2: Regenerating daaf205a_core_sample visualization")
print("=" * 70)

vtk_file_2 = Path(r"c:\Users\markm\Downloads\Simops\.simops\simops_output\0fa95a73\thermal_result.vtk")
if vtk_file_2.exists():
    output_dir_2 = vtk_file_2.parent
    job_name_2 = "daaf205a_core_sample_FIXED"

    image_paths = generate_thermal_views_matplotlib(
        str(vtk_file_2),
        output_dir_2,
        job_name_2,
        views=['isometric', 'top', 'front'],
        colormap='coolwarm'
    )

    print(f"\nGenerated {len(image_paths)} images:")
    for img in image_paths:
        print(f"  - {img}")

    # Check the original for comparison
    original_iso = output_dir_2 / "daaf205a_core_sample_medium_fast_tet_thermal_isometric.png"
    if original_iso.exists():
        print(f"\nCompare with original: {original_iso}")
        print(f"New version: {output_dir_2 / 'daaf205a_core_sample_FIXED_thermal_isometric.png'}")
else:
    print("File not found")

print("\n" + "=" * 70)
print("REGENERATION COMPLETE")
print("=" * 70)
print("\nLook for *_FIXED_thermal_*.png files in the output directories")
