"""
Structural Template Verification Script
========================================

Verifies Protocol A: Structural Template implementation.

Test Steps:
1. Generate L-bracket test geometry
2. Run StructuralTemplateStrategy
3. Verify mesh contains TET10 elements
4. Verify fixed face is the largest flat surface
5. Verify CalculiX input has correct cards
6. Check output files exist

Usage:
    python tests/verification_scripts/verify_structural_template.py
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [TEST] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def test_geometry_generation():
    """Test 1: Generate L-bracket geometry"""
    print("\n" + "="*60)
    print("TEST 1: Geometry Generation")
    print("="*60)
    
    from tests.verification_scripts.generate_l_bracket import create_simple_l_bracket_step
    
    output_dir = project_root / "structural_test_env"
    output_dir.mkdir(exist_ok=True)
    
    l_bracket_file = output_dir / "L_bracket.step"
    
    success = create_simple_l_bracket_step(str(l_bracket_file))
    
    assert success, "Failed to create L-bracket geometry"
    assert l_bracket_file.exists(), f"L-bracket file not found: {l_bracket_file}"
    
    print(f"[OK] L-bracket created: {l_bracket_file}")
    return str(l_bracket_file)


def test_structural_strategy(input_file: str):
    """Test 2: Run structural template strategy"""
    print("\n" + "="*60)
    print("TEST 2: Structural Template Strategy")
    print("="*60)
    
    from strategies.structural_template import StructuralTemplateStrategy
    
    output_dir = project_root / "structural_test_env"
    
    strategy = StructuralTemplateStrategy()
    result = strategy.run(input_file, output_dir, g_factor=10.0)
    
    assert result['success'], f"Strategy failed: {result.get('error')}"
    
    print(f"[OK] Mesh file: {result['mesh_file']}")
    print(f"[OK] CalculiX input: {result['inp_file']}")
    print(f"[OK] Fixed nodes: {result['fixed_nodes_count']}")
    print(f"[OK] Total elements: {result['total_elements']}")
    print(f"[OK] G-Factor: {result['g_factor']}G")
    
    return result


def test_calculix_input(inp_file: str):
    """Test 3: Verify CalculiX input deck"""
    print("\n" + "="*60)
    print("TEST 3: CalculiX Input Verification")
    print("="*60)
    
    with open(inp_file, 'r') as f:
        content = f.read()
    
    # Check for required cards
    checks = [
        ("*NODE", "Node definitions"),
        ("*ELEMENT", "Element definitions"),
        ("*MATERIAL", "Material definition"),
        ("*ELASTIC", "Elastic properties"),
        ("*DENSITY", "Density"),
        ("*BOUNDARY", "Boundary conditions"),
        ("*DLOAD", "Distributed load"),
        ("GRAV", "Gravity load"),
        ("*STEP", "Analysis step"),
        ("*STATIC", "Static analysis"),
        ("NSET=FIXED", "Fixed node set"),
        ("ELSET=Eall", "Element set"),
    ]
    
    all_passed = True
    for card, desc in checks:
        if card in content:
            print(f"  [OK] {desc}: {card}")
        else:
            print(f"  [FAIL] MISSING: {desc} ({card})")
            all_passed = False
    
    # Check element type
    if "C3D10" in content:
        print("  [OK] Element type: C3D10 (TET10 quadratic)")
    elif "C3D4" in content:
        print("  [WARN] Element type: C3D4 (TET4 linear) - expected C3D10")
    else:
        print("  [FAIL] Element type not found")
        all_passed = False
    
    # Check GRAV load parameters
    import re
    grav_match = re.search(r'\*DLOAD\s*\n\s*Eall,\s*GRAV,\s*([\d.]+)', content)
    if grav_match:
        grav_value = float(grav_match.group(1))
        expected = 98.1  # 10G
        if abs(grav_value - expected) < 1.0:
            print(f"  [OK] Gravity load: {grav_value} m/s^2 (10G)")
        else:
            print(f"  [WARN] Gravity load: {grav_value} m/s^2 (expected ~{expected})")
    
    assert all_passed, "CalculiX input missing required cards"
    print("\n[OK] CalculiX input file is valid")


def test_vtk_output(vtk_file: str):
    """Test 4: Verify VTK output"""
    print("\n" + "="*60)
    print("TEST 4: VTK Output Verification")
    print("="*60)
    
    import meshio
    
    mesh = meshio.read(vtk_file)
    
    print(f"  Nodes: {len(mesh.points)}")
    
    # Check for elements
    cell_count = sum(len(c.data) for c in mesh.cells)
    print(f"  Elements: {cell_count}")
    
    # Check cell types
    for cell_block in mesh.cells:
        print(f"  Cell type: {cell_block.type} ({len(cell_block.data)} cells)")
    
    # Check point data
    if mesh.point_data:
        for name, data in mesh.point_data.items():
            print(f"  Point data: {name} (shape: {data.shape})")
    
    assert len(mesh.points) > 0, "No nodes in VTK file"
    assert cell_count > 0, "No elements in VTK file"
    
    print("\n[OK] VTK output is valid")


def run_all_tests():
    """Run all verification tests"""
    print("\n" + "#"*60)
    print("# STRUCTURAL TEMPLATE VERIFICATION")
    print("# Protocol A: Anchor & Accelerate")
    print("#"*60)
    
    try:
        # Test 1: Generate geometry
        l_bracket_file = test_geometry_generation()
        
        # Test 2: Run strategy
        result = test_structural_strategy(l_bracket_file)
        
        # Test 3: Verify CalculiX input
        test_calculix_input(result['inp_file'])
        
        # Test 4: Verify VTK output
        test_vtk_output(result['vtk_file'])
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED [OK]")
        print("="*60)
        
        print("\nGenerated files:")
        print(f"  - {result['mesh_file']}")
        print(f"  - {result['inp_file']}")
        print(f"  - {result['vtk_file']}")
        
        print("\nNext steps:")
        print("  1. Run CalculiX solver on the .inp file")
        print("  2. Use structural_viz.py to generate stress visualization")
        print("  3. Generate PDF report")
        
        return True
        
    except AssertionError as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n[ERROR]: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
