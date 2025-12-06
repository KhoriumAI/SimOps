"""
Verify Conformal Hex Mesh Pipeline
===================================

Tests the conformal hex gluing system with a STEP file.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_conformal_hex_pipeline():
    """Test the full conformal hex pipeline."""
    print("=" * 60)
    print("Conformal Hex Mesh Pipeline Test")
    print("=" * 60)
    
    # Find a test STEP file
    step_files = [
        project_root / "cad_files" / "Cube.step",
        project_root / "cad_files" / "Cylinder.step",
    ]
    
    test_file = None
    for f in step_files:
        if f.exists():
            test_file = f
            break
    
    if test_file is None:
        print("[ERROR] No test STEP file found")
        return False
    
    print("[TEST] Using: {}".format(test_file))
    
    # Import the mesh worker function
    try:
        from apps.cli.mesh_worker_subprocess import generate_conformal_hex_test
    except ImportError as e:
        print("[ERROR] Failed to import: {}".format(e))
        return False
    
    # Run the pipeline
    result = generate_conformal_hex_test(
        str(test_file),
        quality_params={
            'hex_divisions': 4,
            'interface_epsilon': 0.5,
            'coacd_threshold': 0.05
        }
    )
    
    # Check results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    if not result['success']:
        print("[FAIL] Pipeline failed: {}".format(result.get('message', 'Unknown error')))
        return False
    
    print("[OK] Generated {} hexes with {} vertices".format(
        result['total_elements'], result['total_nodes']))
    print("[OK] Output: {}".format(result['output_file']))
    
    # Validation results
    validation = result.get('validation', {})
    print("\nValidation:")
    print("  Interface: {}".format('PASS' if validation.get('interface_pass') else 'FAIL'))
    print("  Manifold:  {}".format('PASS' if validation.get('manifold_pass') else 'FAIL'))
    print("  Jacobian:  {}".format('PASS' if validation.get('jacobian_pass') else 'FAIL'))
    print("  Boundary Faces: {}".format(validation.get('boundary_faces', 0)))
    print("  Internal Faces: {}".format(validation.get('internal_faces', 0)))
    print("  Non-Manifold Errors: {}".format(validation.get('non_manifold_errors', 0)))
    print("  Min Jacobian: {:.3f}".format(validation.get('min_jacobian', 0)))
    print("  Mean Jacobian: {:.3f}".format(validation.get('mean_jacobian', 0)))
    
    all_passed = (
        validation.get('interface_pass', False) and
        validation.get('manifold_pass', False) and
        validation.get('jacobian_pass', False)
    )
    
    print("\n" + "=" * 60)
    if all_passed:
        print(">>> ALL VALIDATIONS PASSED <<<")
    else:
        print(">>> SOME VALIDATIONS FAILED <<<")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = test_conformal_hex_pipeline()
    sys.exit(0 if success else 1)
