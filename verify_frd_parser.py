import os
import numpy as np
from pathlib import Path
from core.solvers.calculix_adapter import CalculiXAdapter

def verify_frd_parser():
    # Mock data with various whitespaces
    frd_content = """
  100CL  101 1.000000000       11008
 -4  NT
 -1          1  373.00000
 -1          2  293.00000
 -3
 100CL  101 2.000000000       11008
 -4  NT
 -1          1  380.00000
 -1          2  295.00000
 -3
"""
    tmp_frd = Path("verify_mock.frd")
    tmp_frd.write_text(frd_content)
    
    node_map = {1: [0,0,0], 2: [1,0,0]}
    elements = np.array([[1, 1, 2, 2, 2]]) # Dummy elements
    
    adapter = CalculiXAdapter()
    # Mock the elements for parsing (just needs to be iterable or array)
    results = adapter._parse_frd(tmp_frd, node_map, elements)
    
    print(f"Parsed {len(results['time_series'])} steps.")
    for step in results['time_series']:
        print(f"Time: {step['time']}, Temps: {step['temperature']}")
    
    success = True
    if len(results['time_series']) != 2:
        print(f"FAIL: Expected 2 steps, got {len(results['time_series'])}")
        success = False
        
    if abs(results['time_series'][0]['time'] - 1.0) > 1e-6:
        print(f"FAIL: Expected time 1.0, got {results['time_series'][0]['time']}")
        success = False

    if success:
        print("Verification Successful!")
    else:
        print("Verification Failed.")
    
    # Cleanup
    tmp_frd.unlink()
    return success

if __name__ == "__main__":
    verify_frd_parser()
