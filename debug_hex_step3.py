
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategies.hex_dominant_strategy import HighFidelityDiscretization, ConvexDecomposition, TopologyGlue

def test_step3():
    print("=== Testing Hex Dominant Meshing - Step 3 (Topology Glue) ===")
    
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
    # Use 0.05 threshold for speed in this test, or 0.02 for better quality
    # 0.01 takes too long for a quick debug test (80s)
    parts, stats = step2.decompose_mesh(output_stl, threshold=0.05)
    
    if not parts:
        print("Failed Step 2.")
        return
        
    print(f"Decomposed into {len(parts)} parts.")
    
    # 3. Step 3: Topology Glue
    print("\n[Step 3] Gluing Parts (Boolean Fragment)...")
    step3 = TopologyGlue()
    success, glue_stats = step3.glue_parts(parts)
    
    print("\n=== Results ===")
    print(f"Success: {success}")
    print(f"Stats: {glue_stats}")
    
    if success:
        print("\n[SUCCESS] Step 3 passed! Parts glued.")
    else:
        print("\n[FAILURE] Step 3 failed.")

if __name__ == "__main__":
    test_step3()
