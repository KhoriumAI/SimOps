
import sys
import numpy as np
import shutil
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from simops_worker import generate_report, extract_surface_mesh

def create_dummy_result():
    # Create a simple 4-node tet
    node_coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    elements = np.array([
        [0, 1, 2, 3] # One tet
    ])
    
    # 4 temperatures
    temperature = np.array([300.0, 350.0, 400.0, 320.0])
    
    return {
        'node_coords': node_coords,
        'elements': elements,
        'temperature': temperature,
        'heat_flux_watts': 50.0,
        'num_elements': 1,
        'solve_time': 0.05,
        'converged': True,
        'convergence_threshold': 1e-4,
        'convergence_steps': 5,
        'time_series_stats': [],
        'time_series': [] 
    }

def run_verification():
    print("="*60)
    print("VERIFYING PREMIUM REPORTING")
    print("="*60)
    
    result = create_dummy_result()
    out_dir = Path("test_output_report")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir()
    
    # Verify Surface Extraction
    print("Testing extract_surface_mesh...")
    faces = extract_surface_mesh(result['node_coords'], result['elements'])
    print(f"  Faces found: {len(faces)}")
    # Should contain 4 faces for a single tet
    if len(faces) == 4:
         print("  [PASS] Surface extraction correct (4 faces for 1 tet)")
    else:
         print(f"  [FAIL] Expected 4 faces, got {len(faces)}")
         
    # Generate Report
    print("\nGenerating Report...")
    sim_config = {
        'physics': {
            'material': 'TestMat',
            'heat_source_temperature': 400.0,
            'ambient_temperature': 300.0,
            'convection_coeff': 25.0
        }
    }
    
    try:
        files = generate_report(
            job_name="Report_Test",
            output_dir=out_dir,
            result=result,
            strategy_name="Unit_Test",
            sim_config=sim_config
        )
        print(f"  [PASS] Report generated: {files['png']}")
    except Exception as e:
        print(f"  [FAIL] Report generation crash: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_verification()
