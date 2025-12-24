#!/usr/bin/env python3
"""
Regenerate PDF Report from Existing CFD Simulation
===================================================

This script re-generates the PDF report for an existing OpenFOAM case
using the enhanced result parser.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from core.solvers.cfd_solver import CFDSolver
from core.reporting.cfd_report import CFDPDFReportGenerator
from core.reporting.multi_angle_viz import generate_multi_angle_streamlines

def regenerate_cfd_report(case_dir: Path, output_dir: Path, job_name: str):
    """Regenerate PDF for existing CFD case"""
    
    print(f"[Report] Regenerating report for: {job_name}")
    print(f"   Case directory: {case_dir}")
    
    # Use the enhanced parser
    solver = CFDSolver(use_wsl=False)  # Already solved, just parsing
    
    try:
        # Re-parse results with enhanced extractor
        result_data = solver._parse_results(case_dir, solve_time=0)  # solve_time in results already
        
        print(f"\n[SUCCESS] Extracted Results:")
        print(f"   Mesh Cells: {result_data.get('num_cells', 'N/A')}")
        print(f"   Reynolds:   {result_data.get('reynolds', 'N/A')}")
        print(f"   Converged:  {result_data.get('converged', False)}")
        print(f"   Courant:    {result_data.get('courant_max', 'N/A')}")
        
        # Generate multi-angle visualizations
        vtk_file = result_data['vtk_file']
        
        print(f"\n[Viz] Generating multi-angle streamline visualizations...")
        viz_paths = generate_multi_angle_streamlines(
            vtk_file,
            output_dir,
            job_name,
            angles=['iso', 'front', 'side', 'top']
        )
        
        if not viz_paths:
            print("   [WARNING] Streamline generation failed completely")
            return []
        
        print(f"\n[SUCCESS] Generated {len(viz_paths)} visualization(s)")
        
        # Generate PDF
        converged = result_data.get('converged', False)
        report_status = "CONVERGED" if converged else "DIVERGED"
        
        report_data = {
            'converged': converged,
            'status_override': report_status,
            'solve_time': result_data.get('solve_time', 0),
            'reynolds_number': result_data.get('reynolds', 'N/A'),
            'drag_coefficient': result_data.get('cd', 'N/A'),
            'lift_coefficient': result_data.get('cl', 'N/A'),
            'u_inlet': 5.0,  # From config
            'mesh_cells': result_data.get('num_cells', 0),
            'strategy_name': 'HighFi_Layered'
        }
        
        print(f"\n[PDF] Generating PDF with {len(viz_paths)} images...")
        pdf_gen = CFDPDFReportGenerator()
        pdf_file = pdf_gen.generate(
            job_name=job_name,
            output_dir=output_dir,
            data=report_data,
            image_paths=viz_paths  # Pass all visualization paths
        )
        
        print(f"\n[SUCCESS] PDF Generated: {pdf_file}")
        return pdf_file
        
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Target the v12 simulation
    case_dir = Path("c:/Users/Owner/Downloads/Simops/output/USED_cfd_airflow_v12_HighFi_Layered_case")
    output_dir = Path("c:/Users/Owner/Downloads/Simops/output")
    job_name = "CFD_Airflow_Demo_v12_REGENERATED"
    
    if not case_dir.exists():
        print(f"[ERROR] Case directory not found: {case_dir}")
        sys.exit(1)
    
    regenerate_cfd_report(case_dir, output_dir, job_name)
