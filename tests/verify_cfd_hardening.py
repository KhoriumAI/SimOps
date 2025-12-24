import sys
import os
sys.path.append(os.getcwd())
import pytest
from pathlib import Path
from core.solvers.cfd_solver import CFDSolver

def test_dictionary_hardening(tmp_path):
    """
    Verifies that U and p dictionaries are written with explicit patches
    and NO wildcards.
    """
    solver = CFDSolver(use_wsl=False)
    
    # Mock Patch Detection
    patches = {
        "inlet": "patch",
        "outlet": "patch",
        "walls": "wall",
        "frontAndBack": "empty",
        "cylinder": "wall"
    }
    
    # 1. Test U Dictionary
    u_file = tmp_path / "U"
    solver._write_U(u_file, [10.0, 0, 0], patches)
    
    content_u = u_file.read_text()
    
    print("U File Content:\n", content_u)
    
    # Assertions
    assert "inlet" in content_u
    assert "outlet" in content_u
    assert "walls" in content_u
    assert "cylinder" in content_u
    assert "frontAndBack" in content_u
    
    # Check Logic
    assert 'value           uniform (10.0 0 0);' in content_u # Inlet fixedValue
    assert 'type            noSlip;' in content_u # Walls
    assert 'type            empty;' in content_u # Empty
    
    # CRITICAL: Ensure NO wildcards
    assert '"(.*)"' not in content_u
    assert 'autoPatch' not in content_u
    
    # 2. Test p Dictionary
    p_file = tmp_path / "p"
    solver._write_p(p_file, patches)
    
    content_p = p_file.read_text()
    print("p File Content:\n", content_p)
    
    # Assertions
    assert 'type            zeroGradient;' in content_p # Inlet pressure
    assert 'type            fixedValue;' in content_p # Outlet pressure
    assert 'value           uniform 0;' in content_p
    assert '"(.*)"' not in content_p

if __name__ == "__main__":
    # Manual run setup if pytest not present
    class MockPath:
        def __init__(self, p): self.p = p
        def __truediv__(self, other): return MockPath(self.p + "/" + other)
        def write_text(self, text): 
            with open(self.p, 'w') as f: f.write(text)
        def read_text(self):
            with open(self.p, 'r') as f: return f.read()
            self.p = p
            
    import os
    os.makedirs("tmp_test", exist_ok=True)
    test_dictionary_hardening(MockPath("tmp_test"))
    print("\n[PASS] Dictionary Hardening Verified.")
