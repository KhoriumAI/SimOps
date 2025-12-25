
import json
import logging
from pathlib import Path
from core.reporting.multi_angle_viz import generate_multi_angle_streamlines
from core.reporting.cfd_report import CFDPDFReportGenerator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Regenerator")

def regenerate():
    job_name = "cube"
    base_dir = Path(r"c:\Users\markm\Downloads\SimOps")
    output_dir = base_dir / "output"
    case_dir = output_dir / "cube_20251224_001728_OK"
    vtu_path = case_dir / "internal.vtu"
    
    if not vtu_path.exists():
        logger.error(f"VTU not found: {vtu_path}")
        return

    # 1. Generate Multi-Angle Visualizations
    logger.info("Generating multi-angle streamlines...")
    image_paths = generate_multi_angle_streamlines(
        vtk_path=str(vtu_path),
        output_dir=output_dir, # Save images to output dir
        job_name=job_name
    )
    
    # 2. Prepare Report Data
    # Since the original json was minimal, we'll provide reasonable defaults 
    # or leave them as N/A as the generator handles it.
    report_data = {
        'converged': True,
        'solve_time': 0, # Placeholder
        'reynolds_number': 'N/A',
        'drag_coefficient': 'N/A',
        'lift_coefficient': 'N/A',
        'viscosity_model': 'Laminar',
        'u_inlet': 1.0,
        'mesh_cells': 89177, # From manual_viz_test output log
        'strategy_name': 'HighFi_Layered'
    }
    
    # 3. Generate PDF
    logger.info("Generating PDF report...")
    gen = CFDPDFReportGenerator()
    pdf_path = gen.generate(
        job_name=job_name,
        output_dir=output_dir,
        data=report_data,
        image_paths=image_paths
    )
    
    logger.info(f"Report regenerated: {pdf_path}")

if __name__ == "__main__":
    regenerate()
