"""
Verify Simulation and Reporting
===============================

Runs a real thermal simulation using the orchestrator and checks for:
1. Successful execution
2. Result data (temps)
3. PDF report generation
4. Snapshot generation
"""

import sys
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.orchestration import ParallelSimulationOrchestrator

def main():
    print("="*60)
    print("VERIFICATION: Full Simulation & Reporting Pipeline")
    print("="*60)
    
    # Setup
    mesh_file = Path("verification_lab/Cube_test.msh")
    if not mesh_file.exists():
        print(f"[FAIL] Test mesh not found at {mesh_file}")
        sys.exit(1)
        
    output_dir = Path("verification_lab/report_test_output")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()
    
    print(f"Mesh: {mesh_file}")
    print(f"Output: {output_dir}")
    
    # Initialize Orchestrator
    try:
        orch = ParallelSimulationOrchestrator(max_workers=2)
        print("[OK] Orchestrator initialized")
        
        # Run Batch (single file)
        print("\nStarting simulation...")
        results = orch.run_batch(
            mesh_files=[str(mesh_file)],
            template_config={
                'preset': 'electronics_cooling',
                'solver': 'calculix' 
            },
            output_dir=output_dir
        )
        
        print("\n--- Results ---")
        print(f"Success: {results.completed_count}/{results.total_count}")
        
        if results.completed_count == 0:
            print("[FAIL] Simulation failed")
            # Print error from first result if avail
            print(f"Errors: {results.ranking}") # Debug print
            sys.exit(1)
            
        ranked_sim = results.best_performer
        res = ranked_sim.result
        
        print(f"Mesh: {res.mesh_name}")
        print(f"Max Temp: {res.max_temp_c:.2f} C")
        print(f"Passed: {res.passed}")
        
        # Verify Report artifacts
        sim_dir = output_dir / res.mesh_name
        pdf_report = list(sim_dir.glob("*.pdf"))
        png_images = list(sim_dir.glob("*.png"))
        
        print(f"\nChecking artifacts in {sim_dir}...")
        
        if pdf_report:
            print(f"  [OK] PDF Report found: {pdf_report[0].name}")
        else:
            print("  [FAIL] No PDF report generated")
            
        if len(png_images) >= 3:
            print(f"  [OK] Images found: {len(png_images)} snapshots")
            for img in png_images:
                print(f"    - {img.name}")
        else:
            print(f"  [WARN] Few images found: {len(png_images)}")
            
        if pdf_report and len(png_images) > 0 and results.completed_count > 0:
            print("\n[SUCCESS] Full pipeline verified!")
            sys.exit(0)
        else:
            print("\n[FAIL] Pipeline incomplete")
            sys.exit(1)

    except Exception as e:
        print(f"\n[ERROR] Exception during verification: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
