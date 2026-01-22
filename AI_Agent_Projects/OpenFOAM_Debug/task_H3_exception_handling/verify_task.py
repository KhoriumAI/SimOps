
import sys
import os
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[3]))

# Import the module under test
import simops_pipeline

class TestExceptionSwallowing(unittest.TestCase):
    
    def test_openfoam_runner_swallows_exception(self):
        """
        Verify that OpenFOAMRunner._extract_results returns fake success data
        when no VTK files are found (simulating a failed run).
        """
        print("\n[Test] Verifying Exception Swallowing in OpenFOAMRunner...")
        
        # Setup config
        config = simops_pipeline.SimOpsConfig()
        runner = simops_pipeline.OpenFOAMRunner(config, verbose=False)
        
        # Create a temp dir that is empty (no VTK files)
        with patch('pathlib.Path.glob', return_value=[]):
             # Mock the directory structure check in _extract_results to avoid real file component access issues
             # actually _extract_results uses is_dir calls potentially via pathlib, but mainly looks for glob results.
             # real _extract_results takes a case_dir Path object.
             
             fake_case_dir = Path("fake/dir")
             
             # Call _extract_results directly to test the specific logic
             # Use a patch for glob on the path object passed
             with patch.object(Path, 'glob', return_value=[]):
                 result = runner._extract_results(fake_case_dir)
        
        # Check if it returns "Success" data despite failure
        self.assertTrue(result['converged'], "Runner reported convergence despite no results!")
        self.assertEqual(result['num_elements'], 1000, "Runner returned placeholder element count!")
        print(f"  Result observed: Converged={result['converged']}, Elements={result['num_elements']}")
        print("  [Confirmed] Exception was swallowed and fake data returned.")

if __name__ == '__main__':
    unittest.main()
