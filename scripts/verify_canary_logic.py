import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import gmsh
from strategies.exhaustive_strategy import ExhaustiveMeshGenerator
from core.config import get_default_config

def test_canary_logic():
    print("Testing Mesh Canary Logic Refinement...")
    
    config = get_default_config()
    generator = ExhaustiveMeshGenerator(config)
    
    # Mock geometry info
    # Case: RocketEngine.step (1 volume, ~2138 edges, ~200mm diagonal)
    generator.geometry_info = {
        'num_volumes': 1,
        'diagonal': 200.0,
        'bounding_box': {'min': [0,0,0], 'max': [150, 100, 50]}
    }
    
    # Mock gmsh functions to simulate edge count
    import unittest.mock as mock
    
    with mock.patch('gmsh.model.getEntities') as mock_entities:
        # Mock getEntities(1) to return 2138 edges
        # Mock getEntities(3) to return 1 volume
        def side_effect(dim):
            if dim == 1: return [(1, i) for i in range(2138)]
            if dim == 3: return [(3, 1)]
            return []
        mock_entities.side_effect = side_effect
        
        # Test check_1d_complexity
        print("\nChecking 1D Complexity for RocketEngine mock...")
        complexity = generator.check_1d_complexity()
        print(f"Is Toxic: {complexity['is_toxic']}")
        print(f"Reason: {complexity['reason']}")
        
        if not complexity['is_toxic']:
            print("[SUCCESS] check_1d_complexity passed for 2138 edges / 200mm!")
        else:
            print("[FAILURE] check_1d_complexity failed unexpectedly.")

    # Test the orchestrator logic bypass
    # We'll need a way to mock run_meshing_strategy or parts of it
    # For now, manually check logic against the printed reasons
    
    num_vols = generator.geometry_info['num_volumes']
    diag = generator.geometry_info['diagonal']
    is_assembly = num_vols > 3
    is_large_part = diag > 20.0
    
    print(f"\nOrchestrator Logic Check:")
    print(f"num_vols: {num_vols}, diag: {diag}")
    print(f"is_assembly: {is_assembly}, is_large_part: {is_large_part}")
    
    should_fail_fast = complexity['is_toxic']
    if complexity['is_toxic']:
        if is_large_part or num_vols <= 3:
            if complexity['edge_count'] < 100000:
                should_fail_fast = False
                print("[SUCCESS] Orchestrator would OVERRIDE toxicity and proceed.")
    else:
        print("[SUCCESS] Orchestrator would PROCEED (not toxic).")

if __name__ == "__main__":
    gmsh.initialize()
    try:
        test_canary_logic()
    finally:
        gmsh.finalize()
