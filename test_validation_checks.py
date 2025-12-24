#!/usr/bin/env python3
"""
Test Pre-Simulation Validation Checks
======================================

Tests the robustness checks and verifies crash log generation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.validation.pre_sim_checks import PreSimulationValidator, run_pre_simulation_checks

def test_physics_bounds():
    """Test physics parameter validation with bad values"""
    print("=" * 60)
    print("TEST 1: Physics Parameter Bounds (should warn/fail)")
    print("=" * 60)
    
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    
    # Bad config with extreme values
    bad_config = {
        "physics": {
            "inlet_velocity": -5.0,  # Negative velocity - ERROR
            "kinematic_viscosity": 0,  # Zero viscosity - ERROR
            "heat_source_temperature": -100  # Below absolute zero - ERROR
        }
    }
    
    passed, msg = run_pre_simulation_checks(
        job_name="TEST_BAD_PHYSICS",
        output_dir=output_dir,
        config=bad_config
    )
    
    print(f"Result: {'PASSED' if passed else 'FAILED'}")
    print(f"Message: {msg}")
    
    # Check if crash log was created
    crash_log = output_dir / "TEST_BAD_PHYSICS_CRASH_LOG.txt"
    if crash_log.exists():
        print(f"\nCrash log created: {crash_log}")
        print("-" * 40)
        print(crash_log.read_text()[:1000])
    else:
        print("\nNo crash log (test passed or error)")
    
    return not passed  # Should fail


def test_disk_space():
    """Test disk space check (should pass unless disk is full)"""
    print("\n" + "=" * 60)
    print("TEST 2: Disk Space Check (should pass)")
    print("=" * 60)
    
    output_dir = Path("./output")
    
    passed, msg = run_pre_simulation_checks(
        job_name="TEST_DISK_SPACE",
        output_dir=output_dir
    )
    
    print(f"Result: {'PASSED' if passed else 'FAILED'}")
    print(f"Message: {msg}")
    
    return passed


def test_good_config():
    """Test with valid configuration (should pass)"""
    print("\n" + "=" * 60)
    print("TEST 3: Good Configuration (should pass)")
    print("=" * 60)
    
    output_dir = Path("./output")
    
    good_config = {
        "physics": {
            "inlet_velocity": 5.0,
            "kinematic_viscosity": 1.5e-5,
            "ambient_temperature": 293.15
        }
    }
    
    passed, msg = run_pre_simulation_checks(
        job_name="TEST_GOOD_CONFIG",
        output_dir=output_dir,
        config=good_config
    )
    
    print(f"Result: {'PASSED' if passed else 'FAILED'}")
    print(f"Message: {msg}")
    
    return passed


if __name__ == "__main__":
    print("Pre-Simulation Validation Tests")
    print("=" * 60 + "\n")
    
    results = []
    results.append(("Bad Physics Detection", test_physics_bounds()))
    results.append(("Disk Space Check", test_disk_space()))
    results.append(("Good Config Pass", test_good_config()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {name}")
    
    all_passed = all(r[1] for r in results)
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
