
import sys
from pathlib import Path
import logging

sys.path.insert(0, "/app")
from core.solvers.calculix_adapter import CalculiXAdapter
from simops_worker import generate_report

logging.basicConfig(level=logging.INFO)

def test_parsing_reporting():
    # Paths
    # We use artifacts from benchmark_transient if available
    bench_dir = Path("/output/benchmark_transient")
    mesh_file = bench_dir / "rod_transient.msh"
    frd_file = bench_dir / "rod_transient.frd"
    
    if not mesh_file.exists() or not frd_file.exists():
        print(f"Missing artifacts in {bench_dir}")
        return
        
    adapter = CalculiXAdapter()
    
    # 1. We need node_map from mesh
    # We can reuse _generate_inp to get it?
    # Or just use gmsh to read it.
    # _generate_inp writes .inp, but returns stats.
    # We can write to a temp inp.
    temp_inp = bench_dir / "temp_parsing.inp"
    # Config irrelevant for just reading stats
    print("Generating stats from mesh...")
    stats = adapter._generate_inp(mesh_file, temp_inp, {}, scale=0.001) # Rod was scale=0.001?
    # Wait, benchmark_transient.py used scale_factor?
    # check benchmark_transient.py logic if possible. 
    # Usually scale=1.0 or 0.001. Rod is usually defined in mm.
    # If we get scale wrong, coords are wrong but parsing works.
    
    # 2. Parse FRD
    print("Parsing FRD...")
    # Helper to get elements from stats
    elements = stats['elements']
    node_map = stats['node_map']
    
    result = adapter._parse_frd(frd_file, node_map, elements)
    
    # Verify Time Series presence
    if 'time_series_stats' in result:
        print(f"Found Time Series Stats: {len(result['time_series_stats'])} steps")
        for s in result['time_series_stats']:
            print(f"  T={s['time']:.2f}, Max={s['max']:.1f}")
    else:
        print("ERROR: No time_series_stats found!")
        
    
    # Inject missing fields expected by generate_report
    result['num_elements'] = len(elements)
    result['solve_time'] = 1.23
    
    # 3. Generate Report
    print("Generating Report...")
    out_dir = Path("/output/test_report")
    out_dir.mkdir(exist_ok=True)
    
    paths = generate_report("Test_Rod_Trans", out_dir, result, "Validation_Strat")
    
    print(f"Report Generated: {paths}")

if __name__ == "__main__":
    test_parsing_reporting()
