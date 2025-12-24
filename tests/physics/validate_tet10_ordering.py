"""
Tet10 Node Ordering Validation Test
====================================

This test verifies that the Gmsh→CalculiX node permutation is correct
for 10-node tetrahedral (Tet10/C3D10) elements.

Critical Issue:
    Current code uses: permuted_nodes = raw_nodes[:, [0,1,2,3,4,5,6,7,9,8]]
    This permutation is UNVALIDATED and may be incorrect.

Test Methodology:
    1. Create cantilever beam geometry
    2. Run with Tet4 (4-node, known correct) → Get baseline deflection
    3. Run with Tet10 (10-node, suspect) → Compare to analytical
    4. If Tet10 deflection is WAY OFF → Node ordering is WRONG

Expected Results:
    - Tet4: ~95-98% of analytical (slightly stiffer due to linear shape functions)
    - Tet10: ~99-100% of analytical (quadratic converges faster)
    
    If Tet10 is WORSE than Tet4, the node ordering corrupts the element stiffness matrix.

Usage:
    python tests/physics/validate_tet10_ordering.py
"""

import sys
import subprocess
from pathlib import Path
import json
import numpy as np

# Add SimOps root to path
SIMOPS_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(SIMOPS_ROOT))

from tests.physics.analytical_cantilever import create_cantilever_test_case, CantileverBeamAnalytical


def run_simulation(step_file: Path, config_file: Path) -> dict:
    """
    Run structural simulation and extract results.
    
    Returns:
        dict with 'success', 'tip_deflection' (m), 'max_stress' (Pa)
    """
    output_dir = step_file.parent
    
    # Import worker
    from simops_worker import run_simulation
    
    try:
        result = run_simulation(str(step_file), str(output_dir), config_path=str(config_file))
        
        if not result.success:
            return {'success': False, 'error': result.error}
        
        # Parse results from metadata JSON
        job_name = step_file.stem
        meta_file = output_dir / f"{job_name}_result.json"
        
        if not meta_file.exists():
            return {'success': False, 'error': 'Metadata file not found'}
        
        with open(meta_file, 'r') as f:
            metadata = json.load(f)
        
        return {
            'success': True,
            'tip_deflection': metadata.get('max_disp', 0.0),  # Assumes tip is max
            'max_stress': metadata.get('max_stress', 0.0),
            'num_elements': metadata.get('num_elements', 0),
        }
    
    except Exception as e:
        return {'success': False, 'error': str(e)}


def analyze_results(tet4_result: dict, tet10_result: dict, analytical: CantileverBeamAnalytical):
    """
    Compare FEA results to analytical and print diagnostic report.
    """
    delta_analytical = analytical.tip_deflection() * 1000  # mm
    sigma_analytical = analytical.max_stress() / 1e6  # MPa
    
    print(f"\n{'='*70}")
    print(f"CANTILEVER BEAM VALIDATION RESULTS")
    print(f"{'='*70}")
    
    print(f"\n[GEOM] Analytical Reference:")
    print(f"   Tip Deflection: {delta_analytical:.4f} mm")
    print(f"   Max Stress:     {sigma_analytical:.2f} MPa")
    
    # Tet4 Results
    if tet4_result['success']:
        delta_tet4 = tet4_result.get('tip_deflection', 0) * 1000  # mm
        sigma_tet4 = tet4_result.get('max_stress', 0) / 1e6  # MPa
        
        error_tet4_disp = abs(delta_tet4 - delta_analytical) / delta_analytical * 100
        error_tet4_stress = abs(sigma_tet4 - sigma_analytical) / sigma_analytical * 100
        
        print(f"\n[TET4] Tet4 Results (LINEAR, Known Correct):")
        print(f"   Tip Deflection: {delta_tet4:.4f} mm  (Error: {error_tet4_disp:.1f}%)")
        print(f"   Max Stress:     {sigma_tet4:.2f} MPa (Error: {error_tet4_stress:.1f}%)")
        print(f"   Elements:       {tet4_result.get('num_elements', 0)}")
        
        # Tet4 validation
        if error_tet4_disp < 5:
            print(f"   [PASS] Tet4 PASS: Deflection within 5% tolerance")
        else:
            print(f"   [FAIL] Tet4 FAIL: Excessive error (baseline corrupt?)")
    else:
        print(f"\n[TET4] Tet4 Results:")
        print(f"   [FAIL] FAILED: {tet4_result.get('error', 'Unknown error')}")
    
    # Tet10 Results
    if tet10_result['success']:
        delta_tet10 = tet10_result.get('tip_deflection', 0) * 1000  # mm
        sigma_tet10 = tet10_result.get('max_stress', 0) / 1e6  # MPa
        
        error_tet10_disp = abs(delta_tet10 - delta_analytical) / delta_analytical * 100
        error_tet10_stress = abs(sigma_tet10 - sigma_analytical) / sigma_analytical * 100
        
        print(f"\n[TET10] Tet10 Results (QUADRATIC, SUSPECT):")
        print(f"   Tip Deflection: {delta_tet10:.4f} mm  (Error: {error_tet10_disp:.1f}%)")
        print(f"   Max Stress:     {sigma_tet10:.2f} MPa (Error: {error_tet10_stress:.1f}%)")
        print(f"   Elements:       {tet10_result.get('num_elements', 0)}")
        
        # Critical test: Tet10 should be BETTER than Tet4
        if error_tet10_disp < error_tet4_disp:
            print(f"   [PASS] Tet10 BETTER than Tet4 -> Node ordering likely CORRECT")
        else:
            print(f"   [WARN] Tet10 WORSE than Tet4 -> NODE ORDERING LIKELY WRONG")
        
        # Absolute accuracy check
        if error_tet10_disp < 2:
            print(f"   [PASS] Tet10 PASS: Deflection within 2% (excellent)")
        elif error_tet10_disp < 5:
            print(f"   [WARN] Tet10 MARGINAL: Deflection within 5% (acceptable but suspicious)")
        else:
            print(f"   [FAIL] Tet10 FAIL: Excessive error -> FIX NODE PERMUTATION")
    else:
        print(f"\n[TET10] Tet10 Results:")
        print(f"   [FAIL] FAILED: {tet10_result.get('error', 'Unknown error')}")
    
    print(f"\n{'='*70}")
    
    # Verdict
    if tet4_result['success'] and tet10_result['success']:
        if error_tet10_disp > error_tet4_disp * 1.5:
            print(f"\n[CRITICAL] CRITICAL: TET10 NODE ORDERING IS WRONG")
            print(f"   Current permutation: [0,1,2,3,4,5,6,7,9,8]")
            print(f"   Action Required: Systematically test all permutations of edge nodes")
            print(f"   Location: calculix_adapter.py:125-130, calculix_structural.py:113-118")
            return False
        elif error_tet10_disp < 5:
            print(f"\n[SUCCESS] SUCCESS: Tet10 node ordering appears CORRECT")
            print(f"   Both element types converge to analytical solution")
            return True
        else:
            print(f"\n[UNCERTAIN] UNCERTAIN: Tet10 results are marginal")
            print(f"   Recommend finer mesh re-test or review permutation")
            return None
    else:
        print(f"\n[INCONCLUSIVE] TEST INCONCLUSIVE: One or both simulations failed")
        return None


def main():
    print(f"\n{'#'*70}")
    print(f"# TET10 NODE ORDERING VALIDATION TEST")  
    print(f"# Verifying Gmsh->CalculiX element permutation correctness")
    print(f"{'#'*70}")
    
    # Setup
    test_dir = Path(__file__).parent / "output" / "tet10_validation"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test cases
    print(f"\n[1/4] Step 1: Generating analytical test cases...")
    tet4_case = create_cantilever_test_case(test_dir, element_order=1)
    tet10_case = create_cantilever_test_case(test_dir, element_order=2)
    
    # Run simulations
    print(f"\n[2/4] Step 2: Running Tet4 baseline simulation...")
    tet4_result = run_simulation(tet4_case['step_file'], tet4_case['config_file'])
    
    print(f"\n[3/4] Step 3: Running Tet10 test simulation...")
    tet10_result = run_simulation(tet10_case['step_file'], tet10_case['config_file'])
    
    # Analyze
    print(f"\n[4/4] Step 4: Comparing results...")
    passed = analyze_results(tet4_result, tet10_result, tet4_case['analytical'])
    
    # Exit code
    if passed is True:
        print(f"\n[PASS] ALL TESTS PASSED\n")
        return 0
    elif passed is False:
        print(f"\n[FAIL] TEST FAILED - CRITICAL ISSUE DETECTED\n")
        return 1
    else:
        print(f"\n[WARN] TEST INCONCLUSIVE\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())
