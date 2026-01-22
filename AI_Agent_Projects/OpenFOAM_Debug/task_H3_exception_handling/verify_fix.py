
import sys
import os
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add the directory containing shadow_pipeline to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the shadow pipeline which patches simops_pipeline
import shadow_pipeline
import simops_pipeline

class TestExceptionPropagation(unittest.TestCase):
    
    def test_openfoam_runner_raises_exception(self):
        """
        Verify that the patched OpenFOAMRunner raises RuntimeError
        when no VTK files are found.
        """
        print("\n[Test] Verifying Exception Propagation in Shadow Pipeline...")
        
        # Setup config
        config = simops_pipeline.SimOpsConfig()
        # Ensure we are using the patched class
        runner = simops_pipeline.OpenFOAMRunner(config, verbose=False)
        
        print(f"  Runner class: {runner.__class__.__name__}")
        print(f"  Runner module: {runner.__class__.__module__}")
        
        fake_case_dir = Path("fake/dir")
        
        # Mock glob to return empty list
        with patch.object(Path, 'glob', return_value=[]):
             with self.assertRaises(RuntimeError) as cm:
                 runner._extract_results(fake_case_dir)
             
             print(f"  Caught expected exception: {cm.exception}")
             self.assertIn("No VTK results found", str(cm.exception))

        print("  [Confirmed] Exception was correctly raised.")

if __name__ == '__main__':
    unittest.main()
