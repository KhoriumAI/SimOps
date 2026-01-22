"""
Verification Script for TASK_05
================================
Runs E2E test suite and validates success rate.
Exit Code 0 if success rate ≥ 80%, otherwise Exit Code 1.
"""

import sys
from pathlib import Path

# Add task root to path
TASK_ROOT = Path(__file__).parent
sys.path.insert(0, str(TASK_ROOT))

from e2e_suite import run_test_suite


def main():
    """Main verification entry point."""
    print("="*70)
    print("TASK_05 E2E Verification Suite")
    print("="*70)
    print("\nThis script validates the complete SimOps pipeline:")
    print("  1. AI Config Generation")
    print("  2. Mesh Generation")
    print("  3. Solver Execution")
    print("  4. Results Parsing")
    print("\nRunning with mocked components for fast validation...")
    print("="*70)
    
    # Paths
    test_cases_file = TASK_ROOT / "test_data" / "test_cases.json"
    output_file = TASK_ROOT / "e2e_results.json"
    
    # Validate test data exists
    if not test_cases_file.exists():
        print(f"\n❌ ERROR: Test cases file not found: {test_cases_file}")
        return 1
    
    # Run test suite
    try:
        success = run_test_suite(test_cases_file, output_file, use_mocks=True)
        
        if success:
            print("\n" + "="*70)
            print("✅ VERIFICATION PASSED - Success rate ≥ 80%")
            print("="*70)
            return 0
        else:
            print("\n" + "="*70)
            print("❌ VERIFICATION FAILED - Success rate < 80%")
            print("="*70)
            return 1
    
    except Exception as e:
        print(f"\n❌ ERROR: Test suite execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
