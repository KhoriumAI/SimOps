
import os
import sys
import time
import shutil
import unittest
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SidecarTest")

class TestSidecarLogic(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("./verify_sidecar_env")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir()
        
    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_temporal_matching_and_renaming(self):
        # 1. Create mismatching names but close timestamps
        cad_file = self.test_dir / "my_part.step"
        cad_file.touch()
        
        # Ensure distinct mtime
        time.sleep(0.1)
        
        json_file = self.test_dir / "template_v2.json"
        json_file.touch()
        
        # Set mtimes to within 1s
        now = time.time()
        os.utime(cad_file, (now, now))
        os.utime(json_file, (now, now))
        
        print(f"Created: {cad_file} and {json_file} with synced time.")
        
        # --- REPLICATE WATCHER LOGIC ---
        file_path = cad_file
        config_path = None
        
        # Exact match check
        exact_sidecar = file_path.with_suffix('.json')
        if exact_sidecar.exists():
            config_path = exact_sidecar
        else:
            # Temporal match
            cad_mtime = file_path.stat().st_mtime
            candidates = []
            for f in self.test_dir.glob('*.json'):
                if f.name.startswith('USED_'): continue
                if f == exact_sidecar: continue
                
                delta = abs(f.stat().st_mtime - cad_mtime)
                if delta < 15.0:
                    candidates.append((delta, f))
            
            if candidates:
                candidates.sort(key=lambda x: x[0])
                config_path = candidates[0][1]
                print(f"Match found: {config_path.name} (delta={candidates[0][0]:.4f}s)")
                
        # Assert Match
        self.assertEqual(config_path, json_file, "Failed to identify temporal sidecar match")
        
        # --- REPLICATE RENAME LOGIC ---
        final_cad_path = file_path
        final_config_path = config_path

        try:
            used_cad_name = f"USED_{file_path.name}"
            used_cad_path = file_path.with_name(used_cad_name)
            os.rename(file_path, used_cad_path)
            final_cad_path = used_cad_path
            
            if config_path:
                used_config_name = f"USED_{config_path.name}"
                used_config_path = config_path.with_name(used_config_name)
                os.rename(config_path, used_config_path)
                final_config_path = used_config_path
                
        except Exception as e:
            self.fail(f"Rename failed: {e}")
            
        # Assert Rename
        self.assertTrue(final_cad_path.exists())
        self.assertTrue(final_cad_path.name.startswith("USED_"))
        self.assertTrue(final_config_path.exists())
        self.assertTrue(final_config_path.name.startswith("USED_"))
        
        # Original files should be gone
        self.assertFalse(cad_file.exists())
        self.assertFalse(json_file.exists())
        
        print("Renaming Verified.")

if __name__ == '__main__':
    unittest.main()
