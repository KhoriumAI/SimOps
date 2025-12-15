
import sys
import os
import json
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def verify_polyhedral_integration():
    print("=== Verifying Polyhedral Strategy Integration ===")
    
    # 1. Setup paths
    worker_script = project_root / "apps" / "cli" / "mesh_worker_subprocess.py"
    cad_file = project_root / "cad_files" / "Cube.step"
    
    if not cad_file.exists():
        # Create a dummy step file if needed or use existing one
        # Try to find any step file
        found = list(project_root.rglob("*.step"))
        if found:
            cad_file = found[0]
            print(f"Using CAD file: {cad_file}")
        else:
            print("No STEP file found for testing!")
            return

    # 2. Prepare quality params with Polyhedral strategy
    quality_params = {
        "mesh_strategy": "Polyhedral (Dual)",
        "quality_preset": "Medium"
    }
    
    # 3. Run worker subprocess
    cmd = [sys.executable, str(worker_script), str(cad_file), "--quality-params", json.dumps(quality_params)]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print("\n--- Worker Output ---")
        print(result.stdout)
        print("---------------------")
        
        if result.returncode != 0:
            print(f"Worker failed with code {result.returncode}")
            print(result.stderr)
            return
            
        # 4. Parse JSON result
        # The last line should be the JSON result
        lines = result.stdout.strip().split('\n')
        json_line = lines[-1]
        
        try:
            data = json.loads(json_line)
            if data.get('success'):
                print("\n[SUCCESS] Polyhedral mesh generation successful!")
                print(f"Output file: {data.get('output_file')}")
                print(f"Strategy: {data.get('strategy')}")
                print(f"Visualization Mode: {data.get('visualization_mode')}")
                
                if data.get('visualization_mode') == 'surface':
                    print("[OK] Correct visualization mode returned")
                else:
                    print(f"[FAIL] Expected 'surface' visualization mode, got '{data.get('visualization_mode')}'")
                    
                # Verify file exists
                if os.path.exists(data.get('output_file')):
                    print("[OK] Output mesh file exists")
                else:
                    print("[FAIL] Output mesh file missing")
            else:
                print(f"\n[FAIL] Worker reported failure: {data.get('message')}")
                
        except json.JSONDecodeError:
            print("\n[FAIL] Could not parse JSON result")
            
    except Exception as e:
        print(f"\n[FAIL] Test execution failed: {e}")

if __name__ == "__main__":
    verify_polyhedral_integration()
