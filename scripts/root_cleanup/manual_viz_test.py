
import logging
from pathlib import Path
from core.reporting.multi_angle_viz import generate_multi_angle_streamlines

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Reproduction")

def reproduce():
    # Target the internal.vtu from the most recent run
    # Based on file listing: c:\Users\markm\Downloads\SimOps\output\cube_20251224_001728_OK\internal.vtu
    vtu_path = Path(r"c:\Users\markm\Downloads\SimOps\output\cube_20251224_001728_OK\internal.vtu")
    output_dir = Path(r"c:\Users\markm\Downloads\SimOps\temp_test_viz")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    job_name = "Reproduction_Test"
    
    logger.info(f"Targeting VTU: {vtu_path}")
    logger.info(f"Output Dir: {output_dir}")
    
    if not vtu_path.exists():
        logger.error("VTU file not found! Cannot reproduce.")
        return

    logger.info("Running generate_multi_angle_streamlines...")
    try:
        paths = generate_multi_angle_streamlines(
            vtk_path=str(vtu_path),
            output_dir=output_dir,
            job_name=job_name
        )
        
        logger.info(f"Generated {len(paths)} images:")
        for p in paths:
            logger.info(f"  - {p}")
            
        if len(paths) >= 4:
            logger.info("SUCCESS: Generated required number of images.")
        else:
            logger.error(f"FAILURE: Expected 4 images, got {len(paths)}")
            
    except Exception as e:
        logger.exception(f"Exception during generation: {e}")

if __name__ == "__main__":
    reproduce()
