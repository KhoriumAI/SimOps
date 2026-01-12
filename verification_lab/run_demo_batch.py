
import os
import sys
from pathlib import Path
import time

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.orchestration import ParallelSimulationOrchestrator

def main():
    print("============================================================")
    print("DEMO: Multi-Model Thermal Simulation & Reporting")
    print("============================================================")
    
    # 1. Setup paths
    meshes_dir = Path("apps/cli/generated_meshes")
    output_dir = Path("verification_lab/demo_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Selection of interesting meshes
    mesh_files = [
        meshes_dir / "Airfoil_mesh.msh",
        meshes_dir / "00040000_fc21414ac8dd4a879b485b51_step_001_mesh.msh"
    ]
    
    # Ensure they exist
    valid_meshes = []
    for m in mesh_files:
        if m.exists():
            valid_meshes.append(str(m))
        else:
            print(f"[WARN] Mesh not found: {m}")
            
    if not valid_meshes:
        print("[ERROR] No valid meshes found to run.")
        return

    # 2. Initialize Orchestrator
    orchestrator = ParallelSimulationOrchestrator(max_workers=2)
    
    print(f"Running batch simulation for {len(valid_meshes)} models...")
    start_time = time.time()
    
    # 3. Run Batch
    # Using 'electronics_cooling' preset
    results = orchestrator.run_batch(
        mesh_files=valid_meshes,
        template_config={'preset': 'electronics_cooling'},
        output_dir=output_dir
    )
    
    end_time = time.time()
    print(f"\nBatch completed in {end_time - start_time:.1f}s")
    
    # 4. Summary of artifacts
    print("\n--- Artifacts Summary ---")
    for rank in results.ranking:
        res = rank.result
        print(f"\nModel: {res.mesh_name}")
        print(f"  Status: {'PASS' if res.passed else 'FAIL'}")
        print(f"  Max Temp: {res.max_temp_c:.1f}C")
        
        # Check files
        sub_dir = output_dir / res.mesh_name
        pdf = sub_dir / f"{res.mesh_name}_thermal_report.pdf"
        
        if pdf.exists():
            print(f"  [OK] PDF Report: {pdf}")
        else:
            print(f"  [MISSING] PDF Report")
            
        pngs = list(sub_dir.glob("*.png"))
        print(f"  [OK] Images: {len(pngs)} snapshots")
        for p in pngs:
            print(f"    - {p.name}")

if __name__ == "__main__":
    main()
