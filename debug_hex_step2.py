
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategies.hex_dominant_strategy import HighFidelityDiscretization, ConvexDecomposition

def test_step2():
    print("=== Testing Hex Dominant Meshing - Step 2 (CoACD) ===")
    
    # Use the model.step from previous tests
    cad_file = "C:/Users/Owner/Downloads/MeshTest/cad_files/model.step"
    
    if not os.path.exists(cad_file):
        print(f"Error: CAD file not found at {cad_file}")
        return
        
    # Create temp output path
    temp_dir = tempfile.gettempdir()
    output_stl = os.path.join(temp_dir, "hex_dom_temp.stl")
    
    # 1. Step 1: High Fidelity Discretization
    print("\n[Step 1] Converting STEP to STL...")
    step1 = HighFidelityDiscretization()
    success = step1.convert_step_to_stl(cad_file, output_stl)
    
    if not success:
        print("Failed Step 1.")
        return
        
    # 2. Step 2: Convex Decomposition
    print("\n[Step 2] Running CoACD Decomposition...")
    step2 = ConvexDecomposition()
    
    # Test with lower threshold for better accuracy
    parts, stats = step2.decompose_mesh(output_stl, threshold=0.01)
    
    print("\n=== Results ===")
    print(f"Num Parts: {stats['num_parts']}")
    print(f"Original Vol: {stats['original_volume']:.4f}")
    print(f"Parts Vol: {stats['parts_volume']:.4f}")
    print(f"Error: {stats['volume_error_pct']:.4f}%")
    
    if stats['volume_error_pct'] < 2.0:
        print("\n[SUCCESS] Step 2 passed! Volume conserved.")
    else:
        print("\n[FAILURE] Step 2 failed: Volume mismatch.")

if __name__ == "__main__":
    test_step2()
