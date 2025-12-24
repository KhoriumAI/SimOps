#!/usr/bin/env python3
"""
Test Comprehensive Anisotropic Meshing Fix
===========================================

This script demonstrates the complete fix for anisotropic meshing
and validates that negative SICN values are eliminated.

Usage:
    python test_comprehensive_anisotropic_fix.py <input.step> [base_size]

Example:
    python test_comprehensive_anisotropic_fix.py cad_files/Airfoil.step 3.0
"""

import sys
import os
import gmsh
import time
from pathlib import Path

# Add core directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.true_anisotropic_meshing import TrueAnisotropicMeshGenerator
from core.mesh_quality_validator import MeshQualityValidator


def test_comprehensive_fix(input_file: str, base_size: float = 3.0):
    """
    Complete test of the anisotropic meshing fix

    Args:
        input_file: Path to STEP file
        base_size: Base mesh size in mm
    """
    print("="*80)
    print(" COMPREHENSIVE ANISOTROPIC MESHING FIX - TEST SCRIPT")
    print("="*80)
    print(f"\nInput file: {input_file}")
    print(f"Base size: {base_size} mm")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    if not os.path.exists(input_file):
        print(f"\n[X] ERROR: File not found: {input_file}")
        return False

    # Initialize Gmsh
    gmsh.initialize()
    gmsh.model.add("ComprehensiveAnisotropicTest")

    try:
        # ====================================================================
        # STEP 1: Load Geometry
        # ====================================================================
        print("\n" + "="*80)
        print(" STEP 1: LOADING GEOMETRY")
        print("="*80)

        print(f"\nLoading: {os.path.basename(input_file)}")
        gmsh.model.occ.importShapes(input_file)
        gmsh.model.occ.synchronize()

        # Get geometry info
        volumes = gmsh.model.getEntities(dim=3)
        surfaces = gmsh.model.getEntities(dim=2)
        curves = gmsh.model.getEntities(dim=1)

        print(f"[OK] Geometry loaded:")
        print(f"  Volumes: {len(volumes)}")
        print(f"  Surfaces: {len(surfaces)}")
        print(f"  Curves: {len(curves)}")

        # ====================================================================
        # STEP 2: Apply True Anisotropic Meshing
        # ====================================================================
        print("\n" + "="*80)
        print(" STEP 2: TRUE ANISOTROPIC MESHING")
        print("="*80)

        print("\nInitializing TrueAnisotropicMeshGenerator...")
        aniso_generator = TrueAnisotropicMeshGenerator(
            base_size=base_size,
            anisotropy_ratio=100.0,  # 100:1 aspect ratio
            curvature_threshold=10.0,  # κ > 10 mm⁻¹ is sharp
            min_size=0.0001,  # 0.1 microns minimum
            max_size=base_size * 10.0  # 10x base size maximum
        )

        print("\nApplying anisotropic meshing workflow...")
        start_time = time.time()

        meshing_success = aniso_generator.apply_anisotropic_meshing(
            generate_surface_first=True
        )

        aniso_time = time.time() - start_time

        if not meshing_success:
            print("\n[X] Anisotropic meshing failed")
            return False

        print(f"\n[OK] Anisotropic meshing completed in {aniso_time:.2f}s")

        # Statistics
        n_curvature_samples = len(aniso_generator.curvature_data)
        n_sharp_features = sum(1 for c in aniso_generator.curvature_data if c.is_sharp)
        n_metric_tensors = len(aniso_generator.node_metrics)
        n_anisotropic = sum(1 for m in aniso_generator.node_metrics.values()
                           if abs(m.m11 - m.m22) > 1e-6 or abs(m.m11 - m.m33) > 1e-6)

        print(f"\nStatistics:")
        print(f"  Curvature samples: {n_curvature_samples}")
        print(f"  Sharp features: {n_sharp_features}")
        print(f"  Metric tensors: {n_metric_tensors}")
        print(f"  Anisotropic metrics: {n_anisotropic}")

        # ====================================================================
        # STEP 3: Generate Volume Mesh
        # ====================================================================
        print("\n" + "="*80)
        print(" STEP 3: VOLUME MESH GENERATION")
        print("="*80)

        print("\nGenerating volume mesh...")
        start_time = time.time()

        try:
            gmsh.model.mesh.generate(3)
            volume_time = time.time() - start_time
            print(f"[OK] Volume mesh generated in {volume_time:.2f}s")
        except Exception as e:
            print(f"\n[X] Volume mesh generation failed: {e}")
            return False

        # Get mesh statistics
        tet_type = 4
        tet_tags, tet_nodes = gmsh.model.mesh.getElementsByType(tet_type)
        n_elements = len(tet_tags)

        all_nodes, all_coords, _ = gmsh.model.mesh.getNodes()
        n_nodes = len(all_nodes)

        print(f"\nMesh statistics:")
        print(f"  Nodes: {n_nodes:,}")
        print(f"  Elements (tets): {n_elements:,}")

        # ====================================================================
        # STEP 4: Quality Validation
        # ====================================================================
        print("\n" + "="*80)
        print(" STEP 4: MESH QUALITY VALIDATION")
        print("="*80)

        print("\nInitializing MeshQualityValidator...")
        validator = MeshQualityValidator(
            sicn_threshold=0.0,  # No inverted elements!
            skewness_threshold=0.95,
            aspect_ratio_threshold=1000.0,
            verbose=True
        )

        print("\nValidating mesh quality...")
        validation_result = validator.validate_current_mesh()

        # ====================================================================
        # STEP 5: Quality Repair (if needed)
        # ====================================================================
        if not validation_result['valid']:
            print("\n" + "="*80)
            print(" STEP 5: QUALITY REPAIR")
            print("="*80)

            print(f"\n[!] Found {validation_result['violations']} quality violations")
            print("  Attempting repair...")

            start_time = time.time()
            repair_success = validator.repair_mesh(max_iterations=5)
            repair_time = time.time() - start_time

            if repair_success:
                print(f"\n[OK] Mesh repaired in {repair_time:.2f}s")

                # Re-validate
                print("\nRe-validating after repair...")
                final_validation = validator.validate_current_mesh()

                if final_validation['valid']:
                    print("\n[OK] Final validation: PASSED")
                else:
                    print(f"\n[!] Final validation: {final_validation['violations']} violations remain")
            else:
                print(f"\n[!] Repair unsuccessful after {repair_time:.2f}s")

        else:
            print("\n[OK] Quality validation: PASSED (no repair needed)")

        # ====================================================================
        # STEP 6: Save Mesh
        # ====================================================================
        print("\n" + "="*80)
        print(" STEP 6: SAVING MESH")
        print("="*80)

        output_file = input_file.replace('.step', '_anisotropic.msh').replace('.stp', '_anisotropic.msh')

        print(f"\nWriting mesh to: {output_file}")
        gmsh.write(output_file)
        print(f"[OK] Mesh saved")

        # Also write VTK for visualization
        vtk_file = output_file.replace('.msh', '.vtk')
        print(f"\nWriting VTK for visualization: {vtk_file}")
        gmsh.write(vtk_file)
        print(f"[OK] VTK saved")

        # ====================================================================
        # FINAL SUMMARY
        # ====================================================================
        print("\n" + "="*80)
        print(" FINAL SUMMARY")
        print("="*80)

        print(f"\n[OK] Comprehensive anisotropic meshing: COMPLETE")
        print(f"\nGeometry:")
        print(f"  Volumes: {len(volumes)}, Surfaces: {len(surfaces)}, Curves: {len(curves)}")

        print(f"\nAnisotropic meshing:")
        print(f"  Sharp features detected: {n_sharp_features}")
        print(f"  Anisotropic metrics applied: {n_anisotropic}")
        print(f"  Base size: {base_size} mm")
        print(f"  Anisotropy ratio: 100:1")

        print(f"\nMesh:")
        print(f"  Nodes: {n_nodes:,}")
        print(f"  Elements: {n_elements:,}")

        print(f"\nQuality:")
        sicn_violations = validation_result.get('sicn_violations', 0)
        skewness_violations = validation_result.get('skewness_violations', 0)

        if validation_result['valid']:
            print(f"  [OK] NO QUALITY VIOLATIONS!")
            print(f"  [OK] SICN violations: 0 (no inverted elements)")
            print(f"  [OK] Skewness violations: 0")
        else:
            print(f"  [!] SICN violations: {sicn_violations}")
            print(f"  [!] Skewness violations: {skewness_violations}")
            print(f"  [!] Total violations: {validation_result['violations']}")

        print(f"\nOutput files:")
        print(f"  Mesh: {output_file}")
        print(f"  VTK: {vtk_file}")

        print(f"\nVisualization:")
        print(f"  ParaView: paraview {vtk_file}")
        print(f"  Gmsh: gmsh {output_file}")

        print("\n" + "="*80)

        if validation_result['valid']:
            print("  [OK][OK][OK] SUCCESS: MESH HAS NO NEGATIVE SICN VALUES! [OK][OK][OK]")
        else:
            print("  [!][!][!] WARNING: SOME QUALITY ISSUES REMAIN [!][!][!]")

        print("="*80 + "\n")

        return validation_result['valid']

    except Exception as e:
        print(f"\n[X] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        gmsh.finalize()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_comprehensive_anisotropic_fix.py <input.step> [base_size]")
        print("\nExample:")
        print("  python test_comprehensive_anisotropic_fix.py cad_files/Airfoil.step 3.0")
        print("\nThis will:")
        print("  1. Load the geometry")
        print("  2. Analyze curvature (true curvature, not just length)")
        print("  3. Compute metric tensor field (true anisotropy)")
        print("  4. Apply MMG3D remeshing (if available)")
        print("  5. Generate volume mesh")
        print("  6. Validate quality (SICN, skewness, aspect ratio)")
        print("  7. Repair if needed")
        print("  8. Save mesh with quality report")
        print("\nExpected result: NO negative SICN values!")
        sys.exit(1)

    input_file = sys.argv[1]
    base_size = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0

    success = test_comprehensive_fix(input_file, base_size)

    sys.exit(0 if success else 1)
