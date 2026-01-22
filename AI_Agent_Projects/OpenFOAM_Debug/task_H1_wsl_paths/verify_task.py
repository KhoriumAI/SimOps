
import os
import subprocess
import unittest
from pathlib import Path
import tempfile

class TestWSLPathConversion(unittest.TestCase):
    def setUp(self):
        # Create a dummy file with spaces in the name to test escaping
        self.test_dir = tempfile.TemporaryDirectory()
        self.file_path = Path(self.test_dir.name) / "test file with spaces.txt"
        self.file_path.touch()

    def tearDown(self):
        self.test_dir.cleanup()

    def to_wsl_current(self, path_str):
        """Current implementation from simops_pipeline.py"""
        try:
            # This is the implementation we suspect is buggy regarding backslashes/escaping
            # when passed to subprocess.run list without proper raw strings or handling
            res = subprocess.run(["wsl", "wslpath", "-u", str(Path(path_str).absolute())], 
                                    capture_output=True, text=True, check=True)
            return res.stdout.strip()
        except:
            # Fallback to manual
            p = Path(path_str).absolute()
            drive = p.drive.lower().replace(':', '')
            return f"/mnt/{drive}{p.as_posix()[2:]}"

    def to_wsl_robust(self, path_str):
        """Proposed robust implementation"""
        p = Path(path_str).absolute()
        # Use simple manual conversion which is often more robust than wslpath 
        # for windows paths with backslashes when calling from python
        drive = p.drive.lower().replace(':', '')
        # Convert backslashes to forward slashes properly
        path_in_linux = p.as_posix() # This handles the backslash to forward slash conversion
        # Construct /mnt/c/... style path
        wsl_path = f"/mnt/{drive}{path_in_linux[2:]}"
        return wsl_path

    def test_current_impl(self):
        print("\nTesting Current Implementation...")
        win_path = str(self.file_path)
        print(f"Windows Path: {win_path}")
        
        # Test 1: Conversion
        wsl_path = self.to_wsl_current(win_path)
        print(f"Converted:    {wsl_path}")
        
        # Test 2: Verify accessibility in WSL
        # Check if file exists via WSL
        check_cmd = ["wsl", "ls", wsl_path]
        try:
            subprocess.run(check_cmd, check=True, capture_output=True)
            print("Access Check: SUCCESS")
        except subprocess.CalledProcessError:
            print("Access Check: FAILED (File not found in WSL)")
            # This is what we expect to fail if escaping is wrong

    def test_robust_impl(self):
        print("\nTesting Robust Implementation...")
        win_path = str(self.file_path)
        
        # Test 1: Conversion
        wsl_path = self.to_wsl_robust(win_path)
        print(f"Converted:    {wsl_path}")
        
        # Test 2: Verify accessibility in WSL
        check_cmd = ["wsl", "ls", wsl_path]
        try:
            subprocess.run(check_cmd, check=True, capture_output=True)
            print("Access Check: SUCCESS")
        except subprocess.CalledProcessError:
            print("Access Check: FAILED")
            self.fail("Robust implementation failed to access file")

if __name__ == '__main__':
    unittest.main()
