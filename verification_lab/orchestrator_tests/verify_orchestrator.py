"""
Verification for Parallel Simulation Orchestrator
==================================================

Verifies:
1. Orchestrator can be instantiated
2. Ranking engine works correctly
3. Pass/fail criteria engine works
4. Template system loads correctly

Run with: python verification_lab/orchestrator_tests/verify_orchestrator.py
"""

import sys
from pathlib import Path

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def verify_presets():
    """Verify preset configurations exist and are valid"""
    print("\n=== Testing Thermal Presets ===")
    
    from core.orchestration import THERMAL_PRESETS
    
    assert len(THERMAL_PRESETS) > 0, "No presets found"
    print(f"  [OK] Found {len(THERMAL_PRESETS)} presets")
    
    assert 'electronics_cooling' in THERMAL_PRESETS
    assert 'heat_sink' in THERMAL_PRESETS
    print("  [OK] Key presets exist (electronics_cooling, heat_sink)")
    
    config = THERMAL_PRESETS['electronics_cooling']
    d = config.to_dict()
    
    assert 'solver' in d
    assert 'physics' in d
    assert d['physics']['simulation_type'] == 'thermal'
    print("  [OK] Preset converts to dict correctly")
    
    return True


def verify_material_library():
    """Verify material library is populated correctly"""
    print("\n=== Testing Material Library ===")
    
    from core.templates import get_material, list_materials
    
    materials = list_materials()
    assert len(materials) > 10, f"Only {len(materials)} materials"
    print(f"  [OK] Found {len(materials)} materials")
    
    assert 'Aluminum_6061' in materials
    assert 'Steel_304SS' in materials
    assert 'Copper_Pure' in materials
    print("  [OK] Key materials exist")
    
    al = get_material('Aluminum_6061')
    assert al.thermal_conductivity == 167.0
    assert al.density == 2700.0
    print(f"  [OK] Aluminum_6061: k={al.thermal_conductivity} W/m-K")
    
    cu = get_material('Copper_Pure')
    assert cu.thermal_conductivity > 400.0
    print(f"  [OK] Copper_Pure: k={cu.thermal_conductivity} W/m-K")
    
    # Test error handling
    try:
        get_material('Unobtanium')
        assert False, "Should have raised KeyError"
    except KeyError:
        print("  [OK] Unknown material raises KeyError")
    
    return True


def verify_template_library():
    """Verify template library works"""
    print("\n=== Testing Template Library ===")
    
    from core.templates import get_template, list_templates
    
    templates = list_templates()
    assert len(templates) >= 5
    print(f"  [OK] Found {len(templates)} templates")
    
    assert 'electronics_cooling' in templates
    assert 'rocket_nozzle' in templates
    print("  [OK] Key templates exist")
    
    t = get_template('electronics_cooling')
    assert t.material == 'Aluminum_6061'
    assert t.ambient_temp_c == 25.0
    print(f"  [OK] electronics_cooling: material={t.material}")
    
    config = t.to_config_dict()
    assert 'physics' in config
    assert config['physics']['thermal_conductivity'] == 167.0
    print("  [OK] Template converts to config dict")
    
    return True


def verify_ranking_engine():
    """Verify ranking engine works correctly"""
    print("\n=== Testing Ranking Engine ===")
    
    from core.orchestration import ThermalRankingEngine, SimulationResult
    
    engine = ThermalRankingEngine()
    
    # Test empty
    ranking = engine.rank([])
    assert ranking == []
    print("  [OK] Empty list returns empty ranking")
    
    # Test single result
    result = SimulationResult(
        mesh_file='test.msh',
        mesh_name='test',
        success=True,
        max_temp_c=80.0,
        min_temp_c=25.0,
        solve_time_s=10.0,
        num_elements=5000,
    )
    
    ranking = engine.rank([result])
    assert len(ranking) == 1
    assert ranking[0].rank == 1
    print("  [OK] Single result ranks as #1")
    
    # Test multiple results
    results = [
        SimulationResult(mesh_file='hot.msh', mesh_name='hot', success=True, max_temp_c=100.0, solve_time_s=10.0, num_elements=5000),
        SimulationResult(mesh_file='cool.msh', mesh_name='cool', success=True, max_temp_c=60.0, solve_time_s=10.0, num_elements=5000),
        SimulationResult(mesh_file='warm.msh', mesh_name='warm', success=True, max_temp_c=80.0, solve_time_s=10.0, num_elements=5000),
    ]
    
    ranking = engine.rank(results)
    assert ranking[0].result.mesh_name == 'cool', f"Expected cool, got {ranking[0].result.mesh_name}"
    assert ranking[2].result.mesh_name == 'hot'
    print("  [OK] Lower max temp ranks higher (cool=#1, hot=#3)")
    
    # Test failed exclusion
    results_with_fail = [
        SimulationResult(mesh_file='ok.msh', mesh_name='ok', success=True, max_temp_c=80.0),
        SimulationResult(mesh_file='fail.msh', mesh_name='fail', success=False, error='Crashed'),
    ]
    
    ranking = engine.rank(results_with_fail)
    assert len(ranking) == 1
    print("  [OK] Failed simulations excluded from ranking")
    
    return True


def verify_pass_fail_engine():
    """Verify pass/fail criteria engine"""
    print("\n=== Testing Pass/Fail Engine ===")
    
    from core.validation.thermal_pass_fail import (
        ThermalPassFailEngine,
        PassFailCriteria,
        FailureCategory,
        electronics_criteria,
    )
    
    engine = ThermalPassFailEngine()
    
    # Test pass
    result_ok = {'success': True, 'max_temp': 80.0, 'min_temp': 25.0, 'converged': True}
    criteria = PassFailCriteria(max_temp_limit_c=100.0)
    pf = engine.evaluate(result_ok, criteria)
    assert pf.passed
    print("  [OK] Result within limits passes")
    
    # Test fail - temp exceeded
    result_hot = {'success': True, 'max_temp': 120.0, 'min_temp': 25.0}
    pf = engine.evaluate(result_hot, criteria)
    assert not pf.passed
    assert pf.category == FailureCategory.TEMPERATURE_EXCEEDED
    print("  [OK] Max temp exceeded fails correctly")
    
    # Test fail - solver crash
    result_crash = {'success': False, 'error': 'Segfault'}
    pf = engine.evaluate(result_crash, criteria)
    assert not pf.passed
    assert pf.category == FailureCategory.SOLVER_CRASH
    print("  [OK] Solver crash fails correctly")
    
    # Test electronics criteria
    elec_criteria = electronics_criteria(max_junction_temp_c=85.0)
    
    result_ok = {'success': True, 'max_temp': 80.0, 'min_temp': 25.0}
    assert engine.evaluate(result_ok, elec_criteria).passed
    
    result_hot = {'success': True, 'max_temp': 90.0, 'min_temp': 25.0}
    assert not engine.evaluate(result_hot, elec_criteria).passed
    print("  [OK] Electronics criteria works correctly")
    
    return True


def verify_orchestrator_init():
    """Verify orchestrator initializes correctly"""
    print("\n=== Testing Orchestrator Initialization ===")
    
    from core.orchestration import ParallelSimulationOrchestrator, ThermalRankingEngine
    
    # Default init
    orch = ParallelSimulationOrchestrator()
    assert orch.max_workers >= 1
    assert orch.timeout > 0
    print(f"  [OK] Default init: {orch.max_workers} workers, {orch.timeout}s timeout")
    
    # Custom workers
    orch4 = ParallelSimulationOrchestrator(max_workers=4)
    assert orch4.max_workers == 4
    print("  [OK] Custom worker count works")
    
    # Has ranking engine
    assert hasattr(orch, 'ranking_engine')
    assert isinstance(orch.ranking_engine, ThermalRankingEngine)
    print("  [OK] Has ranking engine attached")
    
    return True


def verify_simulation_config():
    """Verify simulation config dataclass"""
    print("\n=== Testing Simulation Config ===")
    
    from core.orchestration import ThermalSimulationConfig
    
    config = ThermalSimulationConfig()
    assert config.solver == 'auto'
    assert config.material == 'Aluminum_6061'
    assert config.ambient_temp_c == 25.0
    print("  [OK] Default config has sensible values")
    
    config2 = ThermalSimulationConfig(material='Copper_Pure', source_temp_c=150.0)
    d = config2.to_dict()
    assert d['physics']['material'] == 'Copper_Pure'
    assert d['physics']['source_temp_c'] == 150.0
    print("  [OK] Custom config converts to dict")
    
    return True


def main():
    """Run all verification tests"""
    print("=" * 70)
    print("PARALLEL THERMAL ORCHESTRATOR - VERIFICATION SUITE")
    print("=" * 70)
    
    all_passed = True
    tests = [
        ("Thermal Presets", verify_presets),
        ("Material Library", verify_material_library),
        ("Template Library", verify_template_library),
        ("Ranking Engine", verify_ranking_engine),
        ("Pass/Fail Engine", verify_pass_fail_engine),
        ("Orchestrator Init", verify_orchestrator_init),
        ("Simulation Config", verify_simulation_config),
    ]
    
    for name, test_fn in tests:
        try:
            test_fn()
            print(f"\n  [OK] {name}: PASSED")
        except Exception as e:
            print(f"\n  [FAIL] {name}: FAILED - {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED [OK]")
    else:
        print("SOME TESTS FAILED [X]")
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
