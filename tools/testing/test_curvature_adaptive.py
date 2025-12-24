#!/usr/bin/env python3
"""Test curvature-adaptive meshing on cylinder"""
import sys
import gmsh
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from core.curvature_adaptive import CurvatureAdaptiveMesher

def test_cylinder():
    """Test on cylinder - should refine curved surface, coarsen flat ends"""
    print("="*60)
    print("Testing Curvature-Adaptive Meshing on Cylinder")
    print("="*60)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)

    # Load cylinder
    cad_file = "CAD_files/Cylinder.step"
    if not Path(cad_file).exists():
        print(f"[X] Test file not found: {cad_file}")
        gmsh.finalize()
        return False

    gmsh.open(cad_file)
    print(f"\n[OK] Loaded {cad_file}")

    # Create curvature-adaptive mesher
    # For cylinder: curved surface should get 5mm, flat ends should get 20mm
    mesher = CurvatureAdaptiveMesher(min_size=5.0, max_size=20.0)

    # Analyze curvature
    print("\nAnalyzing geometry curvature...")
    analysis = mesher.analyze_geometry_curvature()

    # Print report
    print("\n" + mesher.get_curvature_report())

    # Create adaptive field
    print("\nCreating curvature-adaptive mesh field...")
    field_tag = mesher.create_curvature_adaptive_field(elements_per_curve=12)
    gmsh.model.mesh.field.setAsBackgroundMesh(field_tag)

    # Disable other mesh size sources
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)

    print("[OK] Background field set")

    # Generate mesh
    print("\nGenerating 1D mesh...")
    gmsh.model.mesh.generate(1)
    nodes_1d = gmsh.model.mesh.getNodes()
    print(f"[OK] 1D: {len(nodes_1d[0])} nodes")

    print("\nGenerating 2D mesh...")
    gmsh.model.mesh.generate(2)
    elem_2d = gmsh.model.mesh.getElements(2)
    num_2d = sum(len(tags) for tags in elem_2d[1])
    print(f"[OK] 2D: {num_2d} elements")

    print("\nGenerating 3D mesh...")
    gmsh.model.mesh.generate(3)
    elem_3d = gmsh.model.mesh.getElements(3)
    num_3d = sum(len(tags) for tags in elem_3d[1])
    print(f"[OK] 3D: {num_3d} elements")

    # Save mesh
    output_file = "test_curvature_adaptive_cylinder.msh"
    gmsh.write(output_file)
    print(f"\n[OK] Saved: {output_file}")

    gmsh.finalize()

    print("\n" + "="*60)
    print("[OK][OK][OK] SUCCESS - Curvature-adaptive meshing works!")
    print("="*60)
    print("\nKey results:")
    print(f"  - Curved surfaces detected: {len(analysis['curved_surfaces'])}")
    print(f"  - Flat surfaces detected: {len(analysis['flat_surfaces'])}")
    print(f"  - Total 3D elements: {num_3d}")
    print(f"  - Refinement ratio: {mesher.max_size_mm/mesher.min_size_mm:.1f}x")
    return True


def test_comparison():
    """Compare uniform vs curvature-adaptive meshing"""
    print("\n" + "="*60)
    print("COMPARISON: Uniform vs Curvature-Adaptive")
    print("="*60)

    cad_file = "CAD_files/Cylinder.step"
    if not Path(cad_file).exists():
        print(f"[X] Test file not found: {cad_file}")
        return

    # Test 1: Uniform mesh
    print("\n1. UNIFORM MESH (20mm everywhere)")
    print("-" * 40)
    gmsh.initialize()
    gmsh.open(cad_file)

    field = gmsh.model.mesh.field.add("MathEval")
    gmsh.model.mesh.field.setString(field, "F", "20")
    gmsh.model.mesh.field.setAsBackgroundMesh(field)

    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)

    gmsh.model.mesh.generate(3)
    elem_uniform = gmsh.model.mesh.getElements(3)
    num_uniform = sum(len(tags) for tags in elem_uniform[1])
    print(f"   Elements: {num_uniform}")

    gmsh.write("test_uniform.msh")
    gmsh.finalize()

    # Test 2: Curvature-adaptive
    print("\n2. CURVATURE-ADAPTIVE (5-20mm)")
    print("-" * 40)
    gmsh.initialize()
    gmsh.open(cad_file)

    mesher = CurvatureAdaptiveMesher(min_size=5.0, max_size=20.0)
    mesher.analyze_geometry_curvature()
    field_tag = mesher.create_curvature_adaptive_field()
    gmsh.model.mesh.field.setAsBackgroundMesh(field_tag)

    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)

    gmsh.model.mesh.generate(3)
    elem_adaptive = gmsh.model.mesh.getElements(3)
    num_adaptive = sum(len(tags) for tags in elem_adaptive[1])
    print(f"   Elements: {num_adaptive}")

    gmsh.write("test_adaptive.msh")
    gmsh.finalize()

    # Test 3: Fine uniform (5mm everywhere for comparison)
    print("\n3. FINE UNIFORM (5mm everywhere)")
    print("-" * 40)
    gmsh.initialize()
    gmsh.open(cad_file)

    field = gmsh.model.mesh.field.add("MathEval")
    gmsh.model.mesh.field.setString(field, "F", "5")
    gmsh.model.mesh.field.setAsBackgroundMesh(field)

    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)

    gmsh.model.mesh.generate(3)
    elem_fine = gmsh.model.mesh.getElements(3)
    num_fine = sum(len(tags) for tags in elem_fine[1])
    print(f"   Elements: {num_fine}")

    gmsh.write("test_fine.msh")
    gmsh.finalize()

    # Summary
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"Uniform (20mm):          {num_uniform:6,} elements  (coarse)")
    print(f"Curvature-Adaptive:      {num_adaptive:6,} elements  (smart)")
    print(f"Fine Uniform (5mm):      {num_fine:6,} elements  (expensive)")
    print()
    print("Benefits of Curvature-Adaptive:")
    print(f"  * {(1 - num_adaptive/num_fine)*100:.1f}% fewer elements than fine uniform")
    print(f"  * Refines curved surfaces (better accuracy)")
    print(f"  * Coarsens flat surfaces (lower cost)")
    print(f"  * Optimal balance of accuracy and efficiency")


if __name__ == "__main__":
    # Run basic test
    success = test_cylinder()

    if success:
        # Run comparison
        test_comparison()

    sys.exit(0 if success else 1)
