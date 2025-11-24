
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategies.hex_dominant_strategy import (
    HighFidelityDiscretization, 
    ConvexDecomposition, 
    TopologyGlue,
    TetrahedralBaseline
)

def test_step4():
    print("=== Testing Hex Dominant Meshing - Step 4 (Tetrahedral Baseline) ===")
    
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
    # Use balanced settings
    success = step1.convert_step_to_stl(cad_file, output_stl, min_size=0.5, max_size=10.0)
    
    if not success:
        print("Failed Step 1.")
        return
        
    # 2. Step 2: Convex Decomposition
    print("\n[Step 2] Running CoACD Decomposition...")
    step2 = ConvexDecomposition()
    # Use 0.01 threshold (Max tightness)
    parts, stats = step2.decompose_mesh(output_stl, threshold=0.01)
    
    if not parts:
        print("Failed Step 2.")
        return
        
    print(f"Decomposed into {len(parts)} parts.")
    
    # Volume Check Gate
    volume_error = stats.get('volume_error_pct', 100.0)
    if volume_error > 3.0:
        print(f"\n[Error] Volume mismatch {volume_error:.2f}% is too high (>3%). Aborting to prevent crash.")
        return
    
    # 3. Step 3: Topology Glue
    print("\n[Step 3] Gluing Parts (Boolean Fragment)...")
    step3 = TopologyGlue()
    success, glue_stats = step3.glue_parts(parts)
    
    if not success:
        print("Failed Step 3.")
        return
        
    # 4. Step 4: Tetrahedral Baseline
    print("\n[Step 4] Generating Tetrahedral Baseline...")
    step4 = TetrahedralBaseline()
    success, mesh_stats = step4.generate_mesh()
    
    print("\n=== Results ===")
    print(f"Success: {success}")
    print(f"Stats: {mesh_stats}")
    
    if success:
        print("\n[SUCCESS] Step 4 passed! Valid tetrahedral mesh generated.")
    else:
        print("\n[FAILURE] Step 4 failed.")

if __name__ == "__main__":
    test_step4()
