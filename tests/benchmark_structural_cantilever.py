
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[1]))
import numpy as np

from core.validation.cases.cantilever_beam import CantileverBeamCase
from core.strategies.cfd_strategy import CFDMeshStrategy, CFDMeshConfig
from core.solvers.calculix_structural import CalculiXStructuralAdapter
from core.reporting.structural_report import StructuralPDFReportGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StructBenchmark")

def run_test():
    output_dir = Path("output/benchmark_structural")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("STRUCTURAL SIMULATION BENCHMARK: Cantilever Beam")
    logger.info("="*60)
    
    # 1. Setup Case
    case = CantileverBeamCase(output_dir)
    config_overrides = case.get_config_overrides()
    
    # 2. Geometry
    logger.info("[1] Generating Geometry...")
    step_file = case.generate_geometry()
    
    # 3. Meshing
    logger.info("[2] Generating Mesh...")
    mesh_file = output_dir / "beam.msh"
    
    # Use standard CFD strategy (it makes good tets)
    # 10x10x100mm beam.
    # Mesh size 2mm is fine (5 elements across width).
    mesh_config = CFDMeshConfig(
        mesh_size_factor=1.0, 
        min_mesh_size=2.0, 
        max_mesh_size=4.0
    )
    
    mesher = CFDMeshStrategy()
    
    success, stats = mesher.generate_cfd_mesh(str(step_file), str(mesh_file), mesh_config)
    if not success:
        logger.error("Meshing failed!")
        return False
        
    logger.info(f"    Mesh: {stats.get('num_elements', '?')} elements")
    
    # Clean up previous results
    for ext in ['.frd', '.inp', '.dat', '.sta', '.cvg']:
        f = output_dir / f"beam{ext}"
        if f.exists():
            try:
                f.unlink()
            except:
                pass

    # 4. Solve
    logger.info("[3] Running CalculiX Structural...")
    solver = CalculiXStructuralAdapter()
    if not Path(solver.ccx_binary).exists() and solver.ccx_binary != "ccx":
        logger.error(f"CCX Binary not found at {solver.ccx_binary}")
        return False
    
    try:
        result = solver.run(mesh_file, output_dir, config_overrides)
    except Exception as e:
        logger.error(f"Solver failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    # 5. Verify
    logger.info("[4] Verifying Results...")
    check = case.verify(result)
    
    max_disp = np.max(result.get('displacement_magnitude', [0]))
    max_strain = result.get('max_strain', 0.0)
    rf_z = result.get('reaction_force_z', 0.0)
    
    print(f"  Max Strain: {max_strain:.6f}")
    print(f"  Reaction Force Z: {rf_z:.2f} N (Expected ~{1000.0} N if tip load used)")

    # 5b. Verify PDF Generation
    logger.info("[4b] Verifying PDF Generation...")
    try:
        gen = StructuralPDFReportGenerator()
        report_data = {
            'success': True,
            'strategy_name': "Benchmark_Test",
            'max_stress': np.max(result.get('von_mises', [0])),
            'max_displacement': max_disp,
            'max_strain': max_strain,
            'reaction_force_z': rf_z,
            'num_elements': result.get('mesh_stats', {}).get('num_elements', 0),
            'solve_time': result.get('solve_time', 0),
            'load_info': "Tip Load (Test)"
        }
        # Fake visualization path
        pdf_path = gen.generate(
            "Benchmark_Cantilever",
            output_dir,
            report_data,
            image_paths=[]
        )
        print(f"  PDF Generated: {pdf_path}")
        if Path(pdf_path).exists():
            print("  [PASS] PDF Generation")
        else:
            print("  [FAIL] PDF File missing")
    except Exception as e:
        print(f"  [FAIL] PDF Generation Error: {e}")

    # Verify vs Analytical
    print(f"\nANALYTICAL vs SIMULATION:")
    # print(f"  Max Stress:   {max_stress_ana:.2f} vs {max_stress:.2f} MPa") # Removed because max_stress_ana is local to case.verify
    
    metrics = check['metrics']
    logger.info("-" * 40)
    logger.info(f"RESULTS SUMMARY:")
    logger.info(f"Stress (Von Mises):")
    logger.info(f"  Theory: {metrics['stress_theory']:.4f} MPa")
    logger.info(f"  Sim:    {metrics['stress_sim']:.4f} MPa")
    logger.info(f"  Error:  {metrics['stress_error']*100:.2f}%")
    
    logger.info(f"Displacement (Max):")
    logger.info(f"  Theory: {metrics['disp_theory']:.4f} mm")
    logger.info(f"  Sim:    {metrics['disp_sim']:.4f} mm")
    logger.info(f"  Error:  {metrics['disp_error']*100:.2f}%")
    logger.info("-" * 40)
    
    if check['passed']:
        logger.info("[PASS] Validation Successful!")
        return True
    else:
        logger.info("[FAIL] Validation Failed (Error > 5%)")
        return False

if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
