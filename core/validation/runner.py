import logging
import shutil
from pathlib import Path
from typing import List, Type
from .base import BenchmarkCase
from ..strategies.cfd_strategy import CFDMeshStrategy, CFDMeshConfig
from ..solvers.calculix_adapter import CalculiXAdapter

logger = logging.getLogger(__name__)

class ValidationRunner:
    """
    Executes benchmark cases to validate the simulation pipeline.
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def run_suite(self, cases: List[Type[BenchmarkCase]]) -> bool:
        """
        Run a list of benchmark cases.
        Returns True if ALL pass.
        """
        success = True
        results = []
        
        print("="*60)
        print(f"Running Validation Suite ({len(cases)} cases)")
        print("="*60)
        
        for case_cls in cases:
            case_dir = self.output_dir / case_cls.__name__
            if case_dir.exists():
                shutil.rmtree(case_dir)
            case_dir.mkdir(parents=True)
            
            case = case_cls(case_dir)
            print(f"\n[Running {case.name}]")
            
            try:
                # 1. Generate Geometry
                step_file = case.generate_geometry()
                print(f"  Geometry generated: {step_file.name}")
                
                # 2. Generate Mesh
                mesh_file = case_dir / "mesh.msh"
                config = CFDMeshConfig(
                    hex_dominant=False, # Use Tets for robustness in calc
                    min_mesh_size=0.002, # 2mm mesh for 100mm rod
                    max_mesh_size=0.005
                )
                mesher = CFDMeshStrategy()
                ok, mesh_stats = mesher.generate_cfd_mesh(str(step_file), str(mesh_file), config)
                
                if not ok:
                    raise RuntimeError("Meshing failed")
                print(f"  Mesh generated: {mesh_stats['num_tets']} tets")
                
                # 3. Solve (CalculiX)
                solver = CalculiXAdapter()
                # Run solver (CalculiX falls back to Python if needed)
                # But here we call Adapter directly. 
                # Note: Adapter run() raises exception if it fails.
                solve_result = solver.run(mesh_file, case_dir, case.get_config_overrides())
                print(f"  Solver finished. Range: {solve_result['min_temp']:.1f}K - {solve_result['max_temp']:.1f}K")
                
                # 4. Verify
                verify_stats = case.verify(solve_result)
                
                if verify_stats['passed']:
                    print(f"  [PASS] Error: {verify_stats['max_relative_error']*100:.2f}%")
                else:
                    print(f"  [FAIL] Error: {verify_stats['max_relative_error']*100:.2f}%")
                    success = False
                    
                results.append(verify_stats)
                
            except Exception as e:
                print(f"  [ERROR] {e}")
                logger.exception(e)
                success = False
                results.append({'case': case.name, 'status': 'ERROR', 'error': str(e)})
                
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        for r in results:
            status = r.get('status', 'UNKNOWN')
            err = r.get('max_relative_error', 0.0) * 100
            print(f"{r['case']:<20} | {status:<6} | Max Error: {err:.2f}%")
            
        return success
