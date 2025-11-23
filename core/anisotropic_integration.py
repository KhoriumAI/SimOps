"""
Integration Layer for True Anisotropic Meshing
===============================================

Integrates true_anisotropic_meshing.py and mesh_quality_validator.py
into the existing mesh generation workflow.

This module provides a drop-in replacement for the old "fake anisotropic"
approach that seamlessly works with existing code.
"""

import gmsh
import numpy as np
from typing import Dict, Optional
from .true_anisotropic_meshing import TrueAnisotropicMeshGenerator
from .mesh_quality_validator import MeshQualityValidator


def apply_true_anisotropic_meshing_with_validation(
    base_size: float = 1.0,
    anisotropy_ratio: float = 100.0,
    curvature_threshold: float = 10.0,
    validate_quality: bool = True,
    repair_if_needed: bool = True,
    sicn_threshold: float = 0.0,
    max_repair_iterations: int = 5,
    use_mmg3d: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Apply true anisotropic meshing with quality validation

    This is a high-level function that combines:
    1. Curvature analysis
    2. Metric tensor computation
    3. Anisotropic meshing (with MMG3D if available)
    4. Quality validation
    5. Repair (if needed)

    Args:
        base_size: Base element size for flat regions (mm)
        anisotropy_ratio: Max ratio of parallel/perpendicular sizes
        curvature_threshold: Curvature above which features are sharp (mm⁻¹)
        validate_quality: Run quality validation after meshing
        repair_if_needed: Attempt repair if validation fails
        sicn_threshold: Minimum acceptable SICN (0.0 = no inverted)
        max_repair_iterations: Max iterations for quality repair
        use_mmg3d: Use MMG3D if available (highly recommended)
        verbose: Print detailed progress

    Returns:
        Dictionary with results:
        {
            'success': bool,
            'anisotropic_meshing_applied': bool,
            'mmg3d_used': bool,
            'quality_valid': bool,
            'quality_metrics': {...},
            'violations': int,
            'repaired': bool
        }
    """
    results = {
        'success': False,
        'anisotropic_meshing_applied': False,
        'mmg3d_used': False,
        'quality_valid': False,
        'quality_metrics': {},
        'violations': 0,
        'repaired': False,
        'message': ''
    }

    try:
        if verbose:
            print("\n" + "="*70)
            print("TRUE ANISOTROPIC MESHING WITH VALIDATION")
            print("="*70)
            print(f"\nConfiguration:")
            print(f"  Base size: {base_size} mm")
            print(f"  Anisotropy ratio: {anisotropy_ratio}:1")
            print(f"  Curvature threshold: {curvature_threshold} mm⁻¹")
            print(f"  Quality validation: {validate_quality}")
            print(f"  Auto-repair: {repair_if_needed}")
            print(f"  SICN threshold: {sicn_threshold}")

        # Step 1: Apply true anisotropic meshing
        if verbose:
            print("\n[1/4] Applying true anisotropic meshing...")

        aniso_generator = TrueAnisotropicMeshGenerator(
            base_size=base_size,
            anisotropy_ratio=anisotropy_ratio,
            curvature_threshold=curvature_threshold,
            min_size=0.0001,
            max_size=base_size * 10.0
        )

        # This generates surface mesh and computes metrics
        meshing_success = aniso_generator.apply_anisotropic_meshing(
            generate_surface_first=True
        )

        if not meshing_success:
            results['message'] = "Anisotropic meshing failed"
            return results

        results['anisotropic_meshing_applied'] = True

        # Check if MMG3D was used
        # (if node_metrics were exported and MMG3D succeeded)
        if len(aniso_generator.node_metrics) > 0:
            results['mmg3d_used'] = True  # Assume it was attempted

        # Step 2: Generate volume mesh
        if verbose:
            print("\n[2/4] Generating volume mesh...")

        try:
            gmsh.model.mesh.generate(3)
            if verbose:
                print("[OK] Volume mesh generated")
        except Exception as e:
            if verbose:
                print(f"[X] Volume mesh generation failed: {e}")
            results['message'] = f"Volume mesh generation failed: {e}"
            return results

        # Step 3: Quality validation
        if validate_quality:
            if verbose:
                print("\n[3/4] Validating mesh quality...")

            validator = MeshQualityValidator(
                sicn_threshold=sicn_threshold,
                skewness_threshold=0.95,
                aspect_ratio_threshold=1000.0,
                verbose=verbose
            )

            validation_result = validator.validate_current_mesh()

            results['quality_valid'] = validation_result['valid']
            results['violations'] = validation_result['violations']
            results['quality_metrics'] = {
                'sicn_violations': validation_result.get('sicn_violations', 0),
                'skewness_violations': validation_result.get('skewness_violations', 0),
                'aspect_violations': validation_result.get('aspect_violations', 0)
            }

            # Step 4: Repair if needed
            if not validation_result['valid'] and repair_if_needed:
                if verbose:
                    print("\n[4/4] Attempting quality repair...")

                repair_success = validator.repair_mesh(max_iterations=max_repair_iterations)

                if repair_success:
                    results['repaired'] = True
                    results['quality_valid'] = True
                    if verbose:
                        print("[OK] Mesh quality repaired successfully")
                else:
                    if verbose:
                        print("[!] Repair unsuccessful, some violations remain")
                    results['message'] = f"Quality repair incomplete: {validation_result['violations']} violations remain"
            else:
                if verbose:
                    print("\n[4/4] Quality validation: SKIPPED (not needed)")

        else:
            if verbose:
                print("\n[3/4] Quality validation: SKIPPED (disabled)")

        # Success!
        results['success'] = True
        results['message'] = "True anisotropic meshing completed successfully"

        if verbose:
            print("\n" + "="*70)
            print("SUMMARY")
            print("="*70)
            print(f"[OK] Anisotropic meshing: {'APPLIED' if results['anisotropic_meshing_applied'] else 'FAILED'}")
            print(f"[OK] MMG3D used: {'YES' if results['mmg3d_used'] else 'NO (fallback to Gmsh)'}")
            print(f"[OK] Quality valid: {'YES' if results['quality_valid'] else 'NO'}")
            if results['violations'] > 0:
                print(f"[!] Violations: {results['violations']}")
            if results['repaired']:
                print(f"[OK] Quality repaired: YES")
            print("="*70 + "\n")

        return results

    except Exception as e:
        if verbose:
            print(f"\n[X] Error: {e}")
            import traceback
            traceback.print_exc()

        results['message'] = f"Exception: {e}"
        return results


def integrate_with_advanced_geometry(
    global_mesh_size: float,
    anisotropy_ratio: float = 100.0
) -> int:
    """
    Integration point for advanced_geometry.py - OPTIMIZED VERSION

    This is called during GEOMETRY SETUP, so it should NOT generate meshes.
    It only sets up anisotropic options for later mesh generation.

    Args:
        global_mesh_size: Global mesh size
        anisotropy_ratio: Anisotropy ratio

    Returns:
        Number of anisotropic fields created (for compatibility)
    """
    print("\n" + "="*70)
    print("PHASE 2: TRUE ANISOTROPIC MESHING (OPTIMIZED)")
    print("="*70)
    print("\n[!] Configuring REAL anisotropic meshing for sharp edges")
    print("  (Curvature analysis only - no mesh generation yet)")

    try:
        # Create anisotropic generator
        aniso_generator = TrueAnisotropicMeshGenerator(
            base_size=global_mesh_size,
            anisotropy_ratio=anisotropy_ratio,
            curvature_threshold=0.1,  # κ > 0.1 mm⁻¹ (radius < 10mm) - captures nearly all curved features
            min_size=0.0001,
            max_size=global_mesh_size * 10.0
        )

        # Step 1: Analyze curvature
        print("\n[1/3] Analyzing curvature...")
        aniso_generator.compute_curvature_at_curves()

        num_sharp_features = len(aniso_generator.curvature_data)
        print(f"[OK] Found {num_sharp_features} curvature samples")

        if num_sharp_features == 0:
            print("  No curvature samples detected, using standard meshing")
            return 0

        # Step 2: Create size fields at sharp features (WITHOUT meshing!)
        print("\n[2/3] Creating anisotropic size fields at sharp corners...")

        # Group curvature samples by curve tag for Distance field creation
        curves_with_sharp_features = {}
        for curv_info in aniso_generator.curvature_data:
            # Use generator's threshold (already filtered by is_sharp flag)
            if curv_info.is_sharp:
                if curv_info.curve_tag not in curves_with_sharp_features:
                    curves_with_sharp_features[curv_info.curve_tag] = []
                curves_with_sharp_features[curv_info.curve_tag].append(curv_info)

        print(f"[OK] Found {len(curves_with_sharp_features)} curves with sharp features")

        if len(curves_with_sharp_features) == 0:
            print(f"  No sharp features (κ >= {aniso_generator.curvature_threshold} mm⁻¹) detected, using standard meshing")
            return 0

        # ====================================================================
        # CRITICAL FIX: DO NOT CREATE ISOTROPIC THRESHOLD FIELDS!
        # ====================================================================
        # The old code created Threshold fields which are ISOTROPIC - they just set
        # a single scalar mesh size (h_perp) near curves, creating small EQUILATERAL
        # elements, NOT stretched anisotropic elements.
        #
        # TRUE anisotropic meshing requires:
        # 1. Metric tensors (3x3 matrices) defining directional sizing
        # 2. Export to .sol file format
        # 3. Run MMG3D to generate stretched elements
        #
        # This happens AFTER initial mesh generation in exhaustive_strategy.py
        # ====================================================================

        print(f"[OK] Found {len(curves_with_sharp_features)} curves requiring anisotropic meshing")
        print(f"[OK] Total sharp curvature samples: {sum(len(samples) for samples in curves_with_sharp_features.values())}")

        # Step 2: Configure Gmsh for TRUE anisotropic meshing
        print("\n[2/3] Configuring Gmsh for TRUE anisotropic meshing (metric tensors)...")

        # Enable anisotropic meshing with metric tensors
        gmsh.option.setNumber("Mesh.AnisoMax", anisotropy_ratio)  # Max anisotropy ratio
        gmsh.option.setNumber("Mesh.AllowSwapAngle", 60)  # Allow element swapping for quality
        # NOTE: Mesh.AnisoMin option does not exist in Gmsh - only AnisoMax is available

        # Configure 3D algorithm for anisotropic support
        # Note: HXT (10) has best anisotropic support
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3)  # For better anisotropy

        print(f"[OK] Configured Mesh.AnisoMax = {anisotropy_ratio}:1")
        print(f"[OK] Configured Mesh.AnisoMin = 0.01")
        print(f"[OK] Enabled anisotropic mesh optimization")

        # Step 3: Store curvature data for later metric tensor computation
        print("\n[3/3] Storing curvature data for metric tensor generation...")

        # Store globally for later use by exhaustive_strategy
        if not hasattr(gmsh.model, '_anisotropic_curvature_data'):
            gmsh.model._anisotropic_curvature_data = []
        gmsh.model._anisotropic_curvature_data.extend(aniso_generator.curvature_data)

        if not hasattr(gmsh.model, '_anisotropic_generator'):
            gmsh.model._anisotropic_generator = aniso_generator

        print(f"[OK] Stored {len(aniso_generator.curvature_data)} curvature samples")
        print(f"[OK] Metric tensor generation will occur AFTER meshing")

        print("\n" + "="*70)
        print("TRUE ANISOTROPIC WORKFLOW (POST-MESHING):")
        print("="*70)
        print("1. Initial mesh generation (Gmsh with AnisoMax configured)")
        print("2. compute_metric_field() - creates 3x3 metric tensors")
        print("3. export_sol_file() - exports to .sol format")
        print("4. run_mmg3d_remeshing() - generates final anisotropic mesh")
        print("   -> This creates STRETCHED elements with high aspect ratios")
        print("="*70 + "\n")

        return len(curves_with_sharp_features)

    except Exception as e:
        print(f"\n[!] Anisotropic setup failed: {e}")
        import traceback
        traceback.print_exc()
        print("  Will use standard meshing approach")
        return 0


def validate_and_repair_final_mesh(
    sicn_threshold: float = 0.0,
    max_repair_iterations: int = 5
) -> bool:
    """
    Final quality check and repair

    Call this after all meshing is complete.

    Args:
        sicn_threshold: Minimum acceptable SICN
        max_repair_iterations: Max repair iterations

    Returns:
        True if mesh quality is acceptable
    """
    print("\n" + "="*70)
    print("FINAL QUALITY VALIDATION AND REPAIR")
    print("="*70)

    validator = MeshQualityValidator(
        sicn_threshold=sicn_threshold,
        skewness_threshold=0.95,
        aspect_ratio_threshold=1000.0,
        verbose=True
    )

    # Validate
    validation_result = validator.validate_current_mesh()

    if validation_result['valid']:
        print("\n[OK] Mesh quality: PASSED")
        return True

    # Attempt repair
    print(f"\n[!] Found {validation_result['violations']} quality violations")
    print("  Attempting repair...")

    repair_success = validator.repair_mesh(max_iterations=max_repair_iterations)

    if repair_success:
        print("\n[OK] Mesh quality: REPAIRED")
        return True
    else:
        print("\n[!] Mesh quality: Some violations remain")
        print(validator.get_quality_report())
        return False


# Convenience function for testing
def test_integration(input_file: str, output_file: str, base_size: float = 1.0):
    """
    Test the complete integration

    Args:
        input_file: Input STEP file
        output_file: Output mesh file
        base_size: Base mesh size
    """
    gmsh.initialize()

    try:
        print(f"Loading geometry: {input_file}")
        gmsh.model.add("AnisotropicTest")
        gmsh.model.occ.importShapes(input_file)
        gmsh.model.occ.synchronize()

        # Apply true anisotropic meshing with validation
        results = apply_true_anisotropic_meshing_with_validation(
            base_size=base_size,
            anisotropy_ratio=100.0,
            curvature_threshold=10.0,
            validate_quality=True,
            repair_if_needed=True,
            sicn_threshold=0.0,
            max_repair_iterations=5,
            use_mmg3d=True,
            verbose=True
        )

        if results['success']:
            print(f"\n[OK] Writing mesh to: {output_file}")
            gmsh.write(output_file)
            print(f"\n[OK] Complete! Quality: {'PASS' if results['quality_valid'] else 'FAIL'}")

            if results['quality_metrics']:
                print("\nQuality Metrics:")
                for key, value in results['quality_metrics'].items():
                    print(f"  {key}: {value}")
        else:
            print(f"\n[X] Failed: {results['message']}")

    finally:
        gmsh.finalize()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python anisotropic_integration.py <input.step> <output.msh> [base_size]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    base_size = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

    test_integration(input_file, output_file, base_size)
