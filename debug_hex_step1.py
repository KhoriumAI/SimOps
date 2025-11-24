
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategies.hex_dominant_strategy import HighFidelityDiscretization

def test_step1():
    print("=== Testing Hex Dominant Meshing - Step 1 ===")
    
    # Use the model.step from previous tests if available, or find one
    cad_file = "C:/Users/Owner/Downloads/MeshTest/cad_files/model.step"
    
    if not os.path.exists(cad_file):
        print(f"Error: CAD file not found at {cad_file}")
        return
        
    # Create temp output path
    temp_dir = tempfile.gettempdir()
    output_stl = os.path.join(temp_dir, "hex_dom_temp.stl")
    
    # Initialize strategy
    strategy = HighFidelityDiscretization()
    
    # 1. Convert STEP to STL
    print("\n[Action] Converting STEP to STL...")
    success = strategy.convert_step_to_stl(cad_file, output_stl)
    
    if not success:
        print("Failed to convert STEP to STL.")
        return
        
    print(f"STL created at: {output_stl}")
    
    # 2. Verify Watertightness
    print("\n[Action] Verifying Watertightness...")
    is_watertight, stats = strategy.verify_watertightness(output_stl)
    
    print("\n=== Results ===")
    print(f"Watertight: {is_watertight}")
    print(f"Stats: {stats}")
    
    if is_watertight:
        print("\n[SUCCESS] Step 1 passed!")
    else:
        print("\n[FAILURE] Step 1 failed: Mesh is not watertight.")

if __name__ == "__main__":
    test_step1()
