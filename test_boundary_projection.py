#!/usr/bin/env python3
"""
Test script for boundary-conforming hex meshing on a cylinder.
"""
import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
import trimesh
from strategies.conformal_hex_glue import AdjacencyGraph, generate_conformal_hex_mesh

def create_test_cylinder(radius=5.0, height=10.0, segments=20):
    """Create a simple cylinder mesh for testing."""
    print(f"Creating test cylinder: r={radius}, h={height}, segments={segments}")
    
    # Create cylinder
    cylinder = trimesh.creation.cylinder(
        radius=radius,
        height=height,
        sections=segments
    )
    
    # Save to temp file
    temp_file = project_root / "test_cylinder.stl"
    cylinder.export(str(temp_file))
    print(f"Saved test cylinder to: {temp_file}")
    
    return cylinder, str(temp_file)

def test_boundary_projection():
    """Test boundary projection on cylinder."""
    print("="*70)
    print("BOUNDARY PROJECTION TEST - CYLINDER")
    print("="*70)
    
    # Create test geometry
    cylinder, stl_file = create_test_cylinder(radius=5.0, height=10.0, segments=24)
    
    # Create mock CoACD output (single part = cylinder itself)
    parts = [(cylinder.vertices, cylinder.faces)]
    
    print(f"\nInput: {len(cylinder.vertices)} vertices, {len(cylinder.faces)} faces")
    
    # Run hex meshing with boundary projection
    print("\nRunning hex mesh generation with boundary projection...")
    result = generate_conformal_hex_mesh(
        parts,
        divisions=4,  # Coarse for testing
        epsilon=0.5,
        reference_stl=stl_file,
        verbose=True
    )
    
    if result['success']:
        print("\n" + "="*70)
        print("SUCCESS!")
        print("="*70)
        print(f"Generated: {result['num_hexes']} hexes, {result['num_vertices']} vertices")
        print(f"Validation: {result['validation']}")
        
        # Check quality
        jac_val = result['validation']['jacobian']
        if 'per_element_quality' in jac_val:
            qualities = jac_val['per_element_quality']
            print(f"\nQuality Metrics:")
            print(f"  Min Jacobian: {jac_val.get('min_jacobian', 'N/A')}")
            print(f"  Avg Jacobian: {jac_val.get('mean_jacobian', 'N/A')}")
            print(f"  Max Jacobian: {jac_val.get('max_jacobian', 'N/A')}")
            print(f"  Inverted elements: {jac_val.get('inverted_elements', 0)}")
        
        return True
    else:
        print("\n" + "="*70)
        print("FAILED!")
        print("="*70)
        print(f"Error: {result.get('error', 'Unknown')}")
        return False

if __name__ == "__main__":
    success = test_boundary_projection()
    sys.exit(0 if success else 1)
