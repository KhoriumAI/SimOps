"""
Verification script for Assignment 3: Pipeline & Reporting Robustness
Tests defensive coding changes to ensure no NoneType crashes.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.reporting.thermal_report import ThermalPDFReportGenerator

def verify_null_temps():
    """Verify report handles None temperatures."""
    print("\n" + "=" * 70)
    print("VERIFICATION: Null Temperature Handling")
    print("=" * 70)
    
    bad_data = {
        'success': True,
        'max_temp_k': None,
        'min_temp_k': None,
        'num_elements': 5000,
        'num_nodes': 8000,
        'strategy_name': 'HighFi_Layered',
        'ambient_temp_c': 25.0,
        'source_temp_c': 100.0
    }
    
    output_dir = Path("./output/verification_null")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        gen = ThermalPDFReportGenerator(job_name="verify_null", output_dir=str(output_dir))
        pdf_path = gen.generate(data=bad_data, image_paths=[])
        print(f"[PASS] PDF created at {pdf_path}")
        print(f"[PASS] No NoneType crashes with None temperatures")
        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_zero_temps():
    """Verify report handles zero temperatures."""
    print("\n" + "=" * 70)
    print("VERIFICATION: Zero Temperature Handling")
    print("=" * 70)
    
    zero_data = {
        'success': True,
        'max_temp_k': 0.0,
        'min_temp_k': 0.0,
        'num_elements': 5000,
        'num_nodes': 8000,
        'strategy_name': 'LowFi_Emergency'
    }
    
    output_dir = Path("./output/verification_zero")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        gen = ThermalPDFReportGenerator(job_name="verify_zero", output_dir=str(output_dir))
        pdf_path = gen.generate(data=zero_data, image_paths=[])
        print(f"[PASS] PDF created at {pdf_path}")
        print(f"[PASS] No crashes with zero/default temperatures")
        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_valid_temps():
    """Verify report still works with valid data."""
    print("\n" + "=" * 70)
    print("VERIFICATION: Valid Temperature Handling")
    print("=" * 70)
    
    valid_data = {
        'success': True,
        'max_temp_k': 373.15,
        'min_temp_k': 298.15,
        'num_elements': 5000,
        'num_nodes': 8000,
        'strategy_name': 'HighFi_Layered',
        'ambient_temp_c': 25.0,
        'source_temp_c': 100.0
    }
    
    output_dir = Path("./output/verification_valid")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        gen = ThermalPDFReportGenerator(job_name="verify_valid", output_dir=str(output_dir))
        pdf_path = gen.generate(data=valid_data, image_paths=[])
        print(f"[PASS] PDF created at {pdf_path}")
        print(f"[PASS] Report correctly processes valid temperatures")
        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("\nAssignment 3: Pipeline Robustness Verification")
    print("=" * 70)
    
    results = [
        verify_null_temps(),
        verify_zero_temps(),
        verify_valid_temps()
    ]
    
    print("\n" + "=" * 70)
    print(f"SUMMARY: {sum(results)}/{len(results)} Tests Passed")
    print("=" * 70)
    
    if all(results):
        print("\n[SUCCESS] ALL VERIFICATIONS PASSED - Assignment 3 Complete")
        sys.exit(0)
    else:
        print("\n[FAILURE] SOME VERIFICATIONS FAILED")
        sys.exit(1)
