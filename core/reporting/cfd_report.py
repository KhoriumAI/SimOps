
import logging
from pathlib import Path
from typing import Dict, List, Optional
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch, cm
from datetime import datetime

logger = logging.getLogger(__name__)

class CFDPDFReportGenerator:
    def __init__(self):
        self.width, self.height = A4
        
    def generate(self, job_name: str, output_dir: Path, data: Dict, image_paths: List[str]) -> Path:
        output_file = Path(output_dir) / f"{job_name}_cfd_report.pdf"
        
        c = canvas.Canvas(str(output_file), pagesize=A4)
        
        # Header
        c.setFont("Helvetica-Bold", 24)
        c.drawString(2 * cm, self.height - 3 * cm, "SymOps CFD Report")
        
        c.setFont("Helvetica", 14)
        c.drawString(2 * cm, self.height - 4 * cm, f"Job: {job_name}")
        c.drawString(2 * cm, self.height - 4.7 * cm, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        # Status Box
        status_color = (0.2, 0.8, 0.2) if data.get('converged', False) else (0.8, 0.2, 0.2)
        c.setFillColorRGB(*status_color)
        c.rect(14 * cm, self.height - 4 * cm, 4 * cm, 1 * cm, fill=1)
        c.setFillColorRGB(1, 1, 1)
        c.setFont("Helvetica-Bold", 12)
        status_text = data.get('status_override', 'UNKNOWN')
        c.drawCentredString(16 * cm, self.height - 3.65 * cm, status_text)
        
        # Key Metrics Table
        c.setFillColorRGB(0, 0, 0)
        y_start = self.height - 7 * cm
        c.setFont("Helvetica-Bold", 14)
        c.drawString(2 * cm, y_start + 0.5 * cm, "Simulation Metrics")
        
        # Format solve time as HH:MM:SS
        solve_time_sec = data.get('solve_time', 0)
        if isinstance(solve_time_sec, (int, float)) and solve_time_sec > 0:
            hours = int(solve_time_sec // 3600)
            minutes = int((solve_time_sec % 3600) // 60)
            seconds = int(solve_time_sec % 60)
            solve_time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            solve_time_str = "00:00:00"
        
        # Format Reynolds number with thousands separator
        reynolds = data.get('reynolds_number', 'N/A')
        if isinstance(reynolds, (int, float)) and reynolds != 'N/A':
            reynolds_str = f"{reynolds:,.0f}"
        else:
            reynolds_str = str(reynolds) if reynolds else 'N/A'
        
        # Build metrics list - start with always-present items
        metrics = [
            ("Strategy", data.get('strategy_name', 'N/A')),
            ("Solve Time", solve_time_str),
            ("Mesh Cells", f"{data.get('mesh_cells', 0):,}"),
            ("Inlet Velocity", f"{data.get('u_inlet', 0):.2f} m/s"),
            ("Reynolds No.", reynolds_str),
        ]
        
        # Only include coefficients if they have numeric values (not 'N/A' or None)
        cd = data.get('drag_coefficient', 'N/A')
        cl = data.get('lift_coefficient', 'N/A')
        
        # Format if float
        if isinstance(cd, float):
            cd = f"{cd:.4f}"
        if isinstance(cl, float):
            cl = f"{cl:.4f}"
        
        if cd not in ['N/A', None, 'None']:
            metrics.append(("Drag Coeff (Cd)", cd))
        if cl not in ['N/A', None, 'None']:
            metrics.append(("Lift Coeff (Cl)", cl))
        
        # Add turbulence model (dynamic, not hardcoded)
        metrics.append(("Viscosity Model", str(data.get('viscosity_model', 'Laminar'))))
        
        c.setFont("Helvetica", 11)
        row_height = 0.8 * cm
        for i, (label, value) in enumerate(metrics):
            y = y_start - (i * row_height)
            # Label
            c.setFont("Helvetica-Bold", 11)
            c.drawString(2.5 * cm, y, label + ":")
            # Value
            c.setFont("Helvetica", 11)
            c.drawString(7 * cm, y, str(value))
            
            # Draw line
            c.setStrokeColorRGB(0.9, 0.9, 0.9)
            c.line(2 * cm, y - 0.2 * cm, 10 * cm, y - 0.2 * cm)

        # Images
        # Layout: 2x2 grid if we have 4 images, or stacked
        y_img_start = y_start - (len(metrics) * row_height) - 1.5 * cm
        
        c.setFont("Helvetica-Bold", 14)
        c.setFillColorRGB(0, 0, 0)
        c.drawString(2 * cm, y_img_start + 1.0 * cm, "Flow Visualization")
        
        valid_images = [img for img in image_paths if Path(img).exists()]
        
        if valid_images:
            # We put images on a new page if they don't fit, but let's try to fit 1-2 on first page
            # Actually, standard layout: 2 images per page
            
            # Simple layout: One large image on first page (Iso), others on next page
            first_img = valid_images[0]
            try:
                # Aspect ratio preservation is manual in reportlab canvas
                # but drawImage has preserveAspectRatio logic? No, only in Platypus.
                # Just forcing size for now.
                c.drawImage(first_img, 2 * cm, 2 * cm, width=17*cm, height=12*cm, preserveAspectRatio=True)
            except Exception as e:
                logger.error(f"Failed to draw image {first_img}: {e}")
                
            if len(valid_images) > 1:
                c.showPage() # Page 2
                
                # Plot remaining images in 2x2 grid
                remaining = valid_images[1:]
                
                # Grid config
                margin = 2 * cm
                w = (self.width - 2 * margin - 1 * cm) / 2
                h = w * 0.75 # 4:3 aspect
                
                positions = [
                    (margin, self.height - margin - h),
                    (margin + w + 1*cm, self.height - margin - h),
                    (margin, self.height - margin - 2*h - 1*cm),
                    (margin + w + 1*cm, self.height - margin - 2*h - 1*cm),
                ]
                
                for i, img_path in enumerate(remaining[:4]): # Max 4 on this page
                    if i < len(positions):
                        x, y = positions[i]
                        try:
                            # Parse angle from filename for title
                            name = Path(img_path).stem
                            parts = name.split('_')
                            title = parts[-1].capitalize() if parts else "View"
                            
                            c.setFont("Helvetica-Bold", 10)
                            c.drawString(x, y + h + 0.1*cm, title)
                            
                            c.drawImage(img_path, x, y, width=w, height=h, preserveAspectRatio=True)
                        except Exception as e:
                            logger.error(f"Failed to draw grid image {img_path}: {e}")
                            
        c.save()
        logger.info(f"Generated PDF report: {output_file}")
        return output_file
