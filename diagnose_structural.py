"""
Diagnostic script to test structural simulation data flow.
Runs a single L_bracket simulation and prints all intermediate values.
"""
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.solvers.calculix_structural import CalculiXStructuralAdapter

def main():
    # Paths
    mesh_file = Path("output_sweep/L_bracket_structural/L_bracket_HighFi_Layered.msh")
    output_dir = Path("output_sweep/L_bracket_structural")
    config_file = Path("configs_sweep/L_bracket_structural.json")
    
    if not mesh_file.exists():
        print(f"ERROR: Mesh file not found: {mesh_file}")
        return
        
    # Load config
    with open(config_file) as f:
        full_config = json.load(f)
    config = full_config.get("physics", {})
    
    print("="*60)
    print("DIAGNOSTIC: Structural Simulation Data Flow")
    print("="*60)
    print(f"\nConfig material: {config.get('material', 'NOT SET')}")
    print(f"Config gravity: {config.get('gravity_load_g', 'NOT SET')}G")
    
    # Run solver
    ccx_path = r"C:\Users\markm\Downloads\SimOps\calculix_native\CalculiX-2.23.0-win-x64\bin\ccx.exe"
    adapter = CalculiXStructuralAdapter(ccx_binary=ccx_path)
    results = adapter.run(mesh_file, output_dir, config)
    
    print("\n" + "="*60)
    print("SOLVER RESULTS DICT")
    print("="*60)
    for key in ['num_nodes', 'num_elements', 'max_stress', 'max_disp']:
        val = results.get(key, "KEY NOT FOUND")
        print(f"{key}: {val}")
    
    # Check array sizes
    print("\nArray Sizes:")
    for key in ['node_coords', 'displacement', 'von_mises', 'elements']:
        arr = results.get(key)
        if arr is not None:
            print(f"  {key}: shape={arr.shape if hasattr(arr, 'shape') else len(arr)}")
        else:
            print(f"  {key}: None")
    
    # Now call report generator
    print("\n" + "="*60)
    print("REPORT GENERATION")
    print("="*60)
    
    from core.reporting.structural_viz import generate_structural_report
    g_factor = config.get('gravity_load_g', 1.0)
    
    report_output = generate_structural_report(
        result=results,
        output_dir=output_dir,
        job_name="L_bracket",
        g_factor=g_factor
    )
    
    print("\nReport Output Dict:")
    for key in ['max_stress_mpa', 'max_displacement_mm', 'displacement_display', 'passed']:
        print(f"  {key}: {report_output.get(key, 'NOT FOUND')}")
    
    print("\nPDF File:", report_output.get('pdf', 'NOT GENERATED'))
    print("\n" + "="*60)
    print("DONE - Check output above for where data is lost")
    print("="*60)

if __name__ == "__main__":
    main()
