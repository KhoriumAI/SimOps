#!/usr/bin/env python3
"""
Test script for adaptive refinement hex meshing.
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
import trimesh
from strategies.conformal_hex_glue import generate_adaptive_hex_mesh

def test_adaptive_cylinder():
    """Test adaptive refinement on cylinder."""
    print("="*70)
    print("ADAPTIVE HEX MESHING TEST - CYLINDER")
    print("="*70)
    
    # Create cylinder
    cylinder = trimesh.creation.cylinder(radius=5.0, height=10.0, sections=24)
    temp_file = project_root / "test_cylinder.stl"
    cylinder.export(str(temp_file))
    
    # Create mock CoACD output
    parts = [(cylinder.vertices, cylinder.faces)]
    
    print(f"\nInput: {len(cylinder.vertices)} vertices, {len(cylinder.faces)} faces\n")
    
    # Run ADAPTIVE hex meshing
    print("Running ADAPTIVE hex mesh generation...")
    print("This will automatically refine until quality >= 90% or 10,000 elements\n")
    
    result = generate_adaptive_hex_mesh(
        parts,
        quality_target=0.90,
        max_elements=10000,
        min_divisions=4,
        max_divisions=16,
        reference_stl=str(temp_file),
        verbose=True
    )
    
    if result['success']:
        print("\n" + "="*70)
        print("RESULT")
        print("="*70)
        print(f"Generated: {result['num_hexes']} hexes")
        
        jacobian_val = result['validation']['jacobian']
        num_inverted = jacobian_val.get('inverted_elements', 0)
        total = result['num_hexes']
        quality = 1.0 - (num_inverted / total) if total > 0 else 0.0
        
        print(f"Quality: {quality*100:.1f}%")
        print(f"Valid elements: {total - num_inverted}/{total}")
        print(f"Min Jacobian: {jacobian_val.get('min_jacobian', 'N/A')}")
        print(f"Avg Jacobian: {jacobian_val.get('mean_jacobian', 'N/A')}")
        
        return True
    else:
        print("\n" + "="*70)
        print("FAILED!")
        print(f"Error: {result.get('error', 'Unknown')}")
        return False

if __name__ == "__main__":
    success = test_adaptive_cylinder()
    sys.exit(0 if success else 1)
