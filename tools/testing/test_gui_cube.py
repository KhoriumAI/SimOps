#!/usr/bin/env python3
"""Test the updated mesh generator with Cube.step"""
import sys
import os

# Add paths
sys.path.insert(0, os.path.dirname(__file__))

from strategies.exhaustive_strategy import ExhaustiveMeshGenerator
from core.config import Config

def test_cube():
    print("="*60)
    print("Testing Cube.step with updated mesh field approach")
    print("="*60)

    # Create config with user settings
    config = Config()
    config.default_params = {
        'cl_max': 0.1  # 100mm max size
    }

    # Create mesh generator
    generator = ExhaustiveMeshGenerator(config)

    # Generate mesh
    print("\nGenerating mesh...")
    result = generator.generate_mesh("CAD_files/Cube.step", "test_cube_output.msh")

    if result.success:
        print("\n[OK] SUCCESS!")
        print(f"\nMesh metrics:")
        print(f"  Nodes: {result.metrics.get('num_nodes', 0)}")
        print(f"  3D elements: {result.metrics.get('num_3d_elements', 0)}")
        print(f"  Quality: {result.metrics.get('avg_quality', 0):.3f}")
    else:
        print(f"\n[X] FAILED: {result.message}")

    return result.success

if __name__ == "__main__":
    success = test_cube()
    sys.exit(0 if success else 1)
