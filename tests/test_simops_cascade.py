
import sys
import os
import shutil
from pathlib import Path

# Add root to python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simops_worker import run_simulation

def test_cascade():
    print("="*60)
    print("STRESS TESTING SIMOPS CASCADE")
    print("="*60)
    
    test_cases = [
        "cad_files/Cube.step",
        "cad_files/Cylinder.step",
        "cad_files/Loft.step"
    ]
    
    output_dir = Path("test_output/cascade_stress_test")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    
    results = {}
    
    for relative_path in test_cases:
        cad_file = str(Path(__file__).parent.parent / relative_path)
        print(f"\nRunning job: {Path(cad_file).name}")
        
        if not os.path.exists(cad_file):
            print(f"  [Skipped] File not found: {cad_file}")
            continue
            
        try:
            result = run_simulation(cad_file, str(output_dir))
            
            if result.success:
                print(f"  [OK] Success with {result.strategy_name}")
                print(f"       Elements: {result.num_elements}")
                results[Path(cad_file).name] = "PASS"
            else:
                print(f"  [FAILED] {result.error}")
                results[Path(cad_file).name] = "FAIL"
                
        except Exception as e:
            print(f"  [CRASH] {e}")
            import traceback
            traceback.print_exc()
            results[Path(cad_file).name] = "CRASH"
            
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, status in results.items():
        print(f"{name:<20} : {status}")
        
    if all(s == "PASS" for s in results.values()):
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed.")
        sys.exit(1)

if __name__ == "__main__":
    test_cascade()
