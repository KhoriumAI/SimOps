
import subprocess
import os
from pathlib import Path
import sys
import time

# Configuration
shapes = ["Cube", "Cylinder", "L_bracket"]
physics_types = ["structural"]
base_cmd = [sys.executable, "simops_worker.py"]

def cleanup_temp_files():
    """Remove temporary test files before starting batch verification."""
    print("[Batch] Cleaning up temporary test files...")
    try:
        cleanup_script = Path(__file__).parent / "cleanup_test_files.py"
        if cleanup_script.exists():
            subprocess.run([sys.executable, str(cleanup_script)], check=False)
    except Exception as e:
        print(f"[Batch] Warning: Cleanup failed: {e}")

def run_simulation(shape, physics):
    cad_file = f"cad_files/{shape}.step"
    config_file = f"configs_sweep/{shape}_{physics}.json"
    output_dir = f"verification_runs/{physics}/{shape}_{physics}"
    
    print(f"\n[Batch] Starting {shape} ({physics})...")
    
    start_time = time.time()
    cmd = base_cmd + [cad_file, "-o", output_dir, "-c", config_file]
    
    try:
        # Run process
        result = subprocess.run(
            cmd, 
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            encoding='utf-8',       # Force UTF-8 for reading output
            errors='replace'        # Handle any decoding errors gracefully
        )
        
        duration = time.time() - start_time
        
        # Check success logic (look for "[OK] SUCCESS" in output)
        if result.returncode == 0 and "[OK] SUCCESS" in result.stdout:
            print(f"[Batch] [PASS] {shape} {physics} PASSED in {duration:.1f}s")
            return True, result.stdout
        else:
            print(f"[Batch] [FAIL] {shape} {physics} FAILED in {duration:.1f}s")
            print("--- STDERR ---")
            print(result.stderr[-500:]) # Last 500 chars
            print("--- STDOUT ---")
            print(result.stdout[-500:])
            return False, result.stdout
            
    except Exception as e:
        print(f"[Batch] [ERROR] Exception running {shape} {physics}: {e}")
        return False, str(e)

def main():
    print("========================================")
    print("   BATCH VERIFICATION START")
    print("========================================")
    
    # Clean up any leftover temp files from previous runs
    cleanup_temp_files()
    
    results = {}
    
    for phys in physics_types:
        print(f"\n--- Running {phys.upper()} Simulations ---")
        for shape in shapes:
            success, output = run_simulation(shape, phys)
            results[f"{shape}_{phys}"] = success

    print("\n========================================")
    print("   BATCH VERIFICATION SUMMARY")
    print("========================================")
    all_passed = True
    for job, success in results.items():
        status = "[PASS]" if success else "[FAIL]"
        print(f"{job:<25} : {status}")
        if not success: all_passed = False
        
    if all_passed:
        print("\nAll simulations completed successfully!")
        sys.exit(0)
    else:
        print("\nSome simulations failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
