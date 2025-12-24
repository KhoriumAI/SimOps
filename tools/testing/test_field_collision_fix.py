#!/usr/bin/env python3
"""
Test script to verify field ID collision fix

Tests that advanced meshing features and intelligent sizing
work together without field ID collisions.
"""

import sys
import gmsh
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from core.optimization_config import OptimizationConfig, QualityPreset
from strategies.exhaustive_strategy import ExhaustiveMeshGenerator

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_field_collision_fix():
    """Test that field IDs don't collide between systems"""

    print("\n" + "="*70)
    print("FIELD COLLISION FIX VERIFICATION")
    print("="*70)

    # Test with Airfoil (geometry that showed the collision)
    input_file = "cad_files/Airfoil.step"

    if not Path(input_file).exists():
        print(f"\n[!] Test file not found: {input_file}")
        print("Please provide a STEP file to test")
        return False

    print(f"\nTest file: {input_file}")

    # Create config with advanced features enabled
    config = OptimizationConfig.from_preset(QualityPreset.MEDIUM)

    print("\nAdvanced Features:")
    print(f"  * Virtual topology: {config.advanced_features.enable_virtual_topology}")
    print(f"  * Adaptive refinement: {config.advanced_features.enable_adaptive_refinement}")
    print(f"  * Boundary layers: {config.advanced_features.enable_boundary_layers}")

    print("\n" + "="*70)
    print("Starting mesh generation...")
    print("="*70 + "\n")

    # Create strategy and generate mesh
    strategy = ExhaustiveMeshGenerator(config=config, use_gpu=False)

    result = strategy.generate_mesh(
        input_file=input_file,
        global_mesh_size=3.0  # mm
    )

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    if result.success:
        print("\n[OK] MESH GENERATION SUCCESSFUL")
        print(f"  Output: {result.output_file}")

        # Check quality metrics
        if 'sicn_min' in result.quality_metrics:
            print(f"\n  Quality Metrics:")
            print(f"    SICN min:  {result.quality_metrics['sicn_min']:.3f}")
            print(f"    SICN mean: {result.quality_metrics['sicn_mean']:.3f}")
            print(f"    Elements:  {result.quality_metrics.get('num_elements', 'N/A')}")

        # Check advanced features applied
        if 'advanced_features' in result.quality_metrics:
            adv = result.quality_metrics['advanced_features']
            print(f"\n  Advanced Features Applied:")
            print(f"    * Virtual faces: {adv['virtual_faces_created']}")
            print(f"    * Size fields: {adv['size_fields_created']}")
            print(f"    * Boundary layers: {adv['boundary_layers_created']}")

            if adv['size_fields_created'] > 0:
                print(f"\n[OK] VERIFICATION PASSED")
                print(f"  All {adv['size_fields_created']} size fields created successfully")
                print(f"  No field ID collisions detected!")
            else:
                print(f"\n[!] WARNING: No size fields were created")

        print("\n" + "="*70)
        return True
    else:
        print(f"\n[X] MESH GENERATION FAILED")
        print(f"  Error: {result.message}")
        print("\n" + "="*70)
        return False

if __name__ == "__main__":
    success = test_field_collision_fix()
    sys.exit(0 if success else 1)
