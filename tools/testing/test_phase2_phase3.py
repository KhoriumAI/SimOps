#!/usr/bin/env python3
"""
Test script for Phase 2 (Anisotropic Metrics) and Phase 3 (Boundary Layers)

Tests the enhanced sharp edge and thin channel meshing on airfoil geometry.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from core.optimization_config import OptimizationConfig, QualityPreset
from strategies.exhaustive_strategy import ExhaustiveMeshGenerator

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_phase2_phase3_airfoil():
    """Test Phase 2 and 3 on airfoil with SICN = -0.3579 (still bad)"""

    print("\n" + "="*80)
    print("TESTING PHASE 2 & 3: ANISOTROPIC METRICS + BOUNDARY LAYERS")
    print("="*80)

    input_file = "cad_files/Airfoil.step"

    if not Path(input_file).exists():
        print(f"\n[X] Test file not found: {input_file}")
        print("Please provide the airfoil STEP file")
        return False

    print(f"\n[FILE] Input: {input_file}")
    print(f"🎯 Goal: Improve SICN from -0.3579 to positive values")
    print(f"📊 Previous SICN: -0.3579 (negative = inverted elements)")

    # Create config with advanced features enabled
    config = OptimizationConfig.from_preset(QualityPreset.MEDIUM)

    print("\n🔧 Configuration:")
    print(f"  * Virtual topology: {config.advanced_features.enable_virtual_topology}")
    print(f"  * Adaptive refinement: {config.advanced_features.enable_adaptive_refinement}")
    print(f"  * Boundary layers: {config.advanced_features.enable_boundary_layers}")
    print(f"  * Phase 2 (Anisotropic): ENABLED")
    print(f"  * Phase 3 (BL fields): ENABLED")

    print("\n" + "="*80)
    print("STARTING MESH GENERATION WITH PHASE 2 & 3...")
    print("="*80 + "\n")

    # Create strategy
    strategy = ExhaustiveMeshGenerator(config=config)

    # Generate mesh
    result = strategy.generate_mesh(
        input_file=input_file,
        global_mesh_size=3.0  # mm
    )

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    if result.success:
        print("\n[OK] MESH GENERATION SUCCESSFUL")
        print(f"\n📄 Output file: {result.output_file}")

        # Display quality metrics
        if 'sicn_min' in result.quality_metrics:
            sicn_min = result.quality_metrics['sicn_min']
            sicn_mean = result.quality_metrics['sicn_mean']

            print(f"\n📊 Quality Metrics:")
            print(f"  * SICN min:  {sicn_min:.4f}")
            print(f"  * SICN mean: {sicn_mean:.4f}")
            print(f"  * Elements:  {result.quality_metrics.get('num_elements', 'N/A')}")

            # Compare with previous
            previous_sicn = -0.3579
            improvement = sicn_min - previous_sicn

            print(f"\n📈 Improvement:")
            print(f"  * Previous SICN: {previous_sicn:.4f}")
            print(f"  * New SICN:      {sicn_min:.4f}")
            print(f"  * Delta:         {improvement:+.4f}")

            if sicn_min > 0:
                print(f"\n🎉 SUCCESS: SICN is now POSITIVE!")
                print(f"   No more inverted elements!")
            elif sicn_min > previous_sicn:
                print(f"\n[OK] IMPROVEMENT: SICN increased by {improvement:.4f}")
                print(f"   Still negative - may need further tuning")
            else:
                print(f"\n[!]️ NO IMPROVEMENT: SICN unchanged or worse")

        # Display advanced features stats
        if 'advanced_features' in result.quality_metrics:
            adv = result.quality_metrics['advanced_features']
            print(f"\n🔧 Advanced Features Applied:")
            print(f"  * Virtual faces:         {adv.get('virtual_faces_created', 0)}")
            print(f"  * Size fields:           {adv.get('size_fields_created', 0)}")
            print(f"  * Anisotropic fields:    {adv.get('anisotropic_fields_created', 0)}")
            print(f"  * Boundary layers:       {adv.get('boundary_layers_created', 0)}")
            print(f"  * BL size fields:        {adv.get('bl_size_fields_created', 0)}")

        print("\n" + "="*80)
        return True

    else:
        print("\n[X] MESH GENERATION FAILED")
        print(f"Error: {result.message}")
        print("\n" + "="*80)
        return False


if __name__ == "__main__":
    try:
        success = test_phase2_phase3_airfoil()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n[!]️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[X] Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
