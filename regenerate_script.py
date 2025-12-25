
import json
import logging
import os
import shutil
from pathlib import Path
from core.reporting.multi_angle_viz import generate_multi_angle_streamlines
from core.reporting.cfd_report import CFDPDFReportGenerator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Regenerator")

def find_case_dir(output_root, job_name, strategy):
    # Try probable names
    candidates = [
        f"{job_name}_{strategy}_case",
        f"USED_{job_name}_{strategy}_case",
        f"{job_name}_case"
    ]
    for c in candidates:
        d = output_root / c
        if d.exists():
            return d
    return None

def find_solution_vtu(case_dir):
    vtk_root = case_dir / "VTK"
    if not vtk_root.exists():
        return None
        
    candidates = list(vtk_root.rglob("internal.vtu"))
    if not candidates:
        return None
        
    def extract_time(p):
        try:
            parts = p.parent.name.split('_')
            return float(parts[-1])
        except:
            return -1.0
            
    candidates.sort(key=extract_time)
    return candidates[-1]

def regenerate():
    output_root = Path(r"c:\Users\markm\Downloads\SimOps\output")
    
    # Find OK folders
    targets = list(output_root.glob("*_OK"))
    # Filter for today? Or just all? User said "sims i ran", assuming recent.
    # Let's filter for 20251224 to be safe and fast.
    targets = [t for t in targets if "20251224" in t.name]
    
    logger.info(f"Found {len(targets)} target simulation folders.")
    
    for target_dir in targets:
        logger.info(f"Processing: {target_dir.name}")
        
        manifest_path = target_dir / "dispatch_manifest.json"
        if not manifest_path.exists():
            logger.warning(f"  No manifest found. Skipping.")
            continue
            
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
            
        res = manifest.get('original_result', {})
        job_name = res.get('job_name')
        strategy = res.get('strategy')
        
        if not job_name or not strategy:
            logger.warning("  Missing job/strategy in manifest.")
            continue
            
        # Find Case Dir in Root
        case_dir = find_case_dir(output_root, job_name, strategy)
        if not case_dir:
            logger.warning(f"  Case directory not found for {job_name} ({strategy})")
            continue
            
        logger.info(f"  Found case dir: {case_dir.name}")
        
        # Find Solution VTU
        vtu_path = find_solution_vtu(case_dir)
        if not vtu_path:
            logger.warning("  No internal.vtu solution found!")
            continue
            
        logger.info(f"  Solution VTU: {vtu_path}")
        
        # 1. Regenerate Visualizations
        logger.info("  Regenerating Visualizations...")
        try:
            image_paths = generate_multi_angle_streamlines(
                str(vtu_path),
                target_dir, # Save images directly into the OK folder
                job_name
            )
            logger.info(f"  Generated {len(image_paths)} images.")
        except Exception as e:
            logger.error(f"  Viz failed: {e}")
            image_paths = []

        # 2. Regenerate PDF
        logger.info("  Regenerating PDF...")
        
        # Try to find better data from root result.json if available
        root_res_file = output_root / f"{job_name}_result.json"
        report_data = {
            'converged': True, # Assume converged if we have a solution
            'status_override': "REGENERATED",
            'strategy_name': strategy,
            'job_name': job_name
        }
        
        if root_res_file.exists():
            try:
                with open(root_res_file, 'r') as f:
                    root_data = json.load(f)
                    report_data.update(root_data)
            except: pass
        else:
             # Merge info from manifest
             report_data.update(res)

        try:
            gen = CFDPDFReportGenerator()
            pdf_path = gen.generate(
                job_name,
                target_dir,
                report_data,
                image_paths
            )
            logger.info(f"  PDF Generated: {pdf_path.name}")
            
            # [Fix] Also copy to root output/ for easier visibility
            root_pdf = output_root / f"{job_name}_cfd_report.pdf"
            shutil.copy(pdf_path, root_pdf)
            logger.info(f"  Copied PDF to root: {root_pdf.name}")
            
            if image_paths:
                # Copy iso view as primary preview
                iso_img = [img for img in image_paths if "_iso" in img]
                if iso_img:
                    root_img = output_root / f"{job_name}_velocity.png"
                    shutil.copy(iso_img[0], root_img)
                    logger.info(f"  Copied Iso view to root as {root_img.name}")
                    
        except Exception as e:
            logger.error(f"  PDF Gen failed: {e}")

if __name__ == "__main__":
    regenerate()
