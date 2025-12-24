#!/usr/bin/env python3
"""
Test script to debug 2D meshing freeze issue with cylinder
"""
import gmsh
import time

def test_mesh_with_sizes(filepath, cl_min, cl_max):
    """Test meshing with specific characteristic lengths"""
    print(f"\n{'='*60}")
    print(f"Testing with cl_min={cl_min*1000:.2f}mm, cl_max={cl_max*1000:.2f}mm")
    print('='*60)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)  # Enable terminal output

    try:
        gmsh.open(filepath)

        # Set mesh sizes
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", cl_min)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", cl_max)

        print(f"Set CharacteristicLengthMin: {cl_min} m ({cl_min*1000} mm)")
        print(f"Set CharacteristicLengthMax: {cl_max} m ({cl_max*1000} mm)")

        # Try 1D meshing
        print("\nGenerating 1D mesh...")
        start = time.time()
        gmsh.model.mesh.generate(1)
        elapsed_1d = time.time() - start
        print(f"[OK] 1D mesh complete in {elapsed_1d:.3f}s")

        # Count 1D elements
        nodes_1d = gmsh.model.mesh.getNodes()
        print(f"  Nodes after 1D: {len(nodes_1d[0])}")

        # Try 2D meshing with timeout simulation
        print("\nGenerating 2D mesh...")
        start = time.time()

        # Set a reasonable algorithm
        gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
        gmsh.option.setNumber("Mesh.RecombineAll", 0)  # Don't recombine

        gmsh.model.mesh.generate(2)
        elapsed_2d = time.time() - start
        print(f"[OK] 2D mesh complete in {elapsed_2d:.3f}s")

        # Count 2D elements
        elements_2d = gmsh.model.mesh.getElements(2)
        if elements_2d[0]:
            num_triangles = len(elements_2d[1][0])
            print(f"  2D triangles: {num_triangles}")

        gmsh.finalize()
        return True, elapsed_1d, elapsed_2d

    except Exception as e:
        print(f"\n[X] ERROR: {e}")
        gmsh.finalize()
        return False, 0, 0


def main():
    filepath = "CAD_files/Cylinder.step"

    # Test with different mesh size parameters
    test_cases = [
        # (cl_min_mm, cl_max_mm)
        (5.9, 10.9),   # Coarse preset from our calculation
        (10, 20),      # Even coarser
        (20, 40),      # Very coarse
        (3.7, 6.9),    # Medium preset
    ]

    print("\n" + "="*60)
    print("2D Meshing Performance Test")
    print("="*60)

    results = []
    for min_mm, max_mm in test_cases:
        cl_min = min_mm / 1000.0  # Convert to meters
        cl_max = max_mm / 1000.0

        success, t1d, t2d = test_mesh_with_sizes(filepath, cl_min, cl_max)
        results.append((min_mm, max_mm, success, t1d, t2d))

        if not success:
            print("[!] Skipping remaining tests due to error")
            break

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"{'Min (mm)':<10} {'Max (mm)':<10} {'Status':<10} {'1D (s)':<10} {'2D (s)':<10}")
    print("-"*60)
    for min_mm, max_mm, success, t1d, t2d in results:
        status = "[OK] OK" if success else "[X] FAIL"
        print(f"{min_mm:<10.1f} {max_mm:<10.1f} {status:<10} {t1d:<10.3f} {t2d:<10.3f}")

if __name__ == "__main__":
    main()
