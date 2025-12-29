
import logging
from pathlib import Path
from typing import Dict, List, Optional
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from datetime import datetime

logger = logging.getLogger(__name__)

class ThermalPDFReportGenerator:
    """
    Generates a PDF report for thermal analysis.
    Designed to work with CalculiX thermal solver output.
    """
    def __init__(self, job_name=None, output_dir=None):
        self.width, self.height = A4
        self.job_name = job_name
        self.output_dir = output_dir

    def generate(self, job_name=None, output_dir=None, data=None, image_paths=None) -> Path:
        # Support both constructor and method-based args
        job_name = job_name or self.job_name
        output_dir = output_dir or self.output_dir
        data = data or {}
        image_paths = image_paths or []
        
        output_file = Path(output_dir) / f"{job_name}_thermal_report.pdf"
        
        c = canvas.Canvas(str(output_file), pagesize=A4)
        
        # Header
        c.setFont("Helvetica-Bold", 24)
        c.drawString(2 * cm, self.height - 3 * cm, "SimOps Thermal Report")
        
        c.setFont("Helvetica", 14)
        c.drawString(2 * cm, self.height - 4 * cm, f"Component: {job_name}")
        c.drawString(2 * cm, self.height - 4.7 * cm, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        # Status Box
        passed = data.get('success', False)
        status_color = (0.2, 0.8, 0.2) if passed else (0.8, 0.2, 0.2)
        c.setFillColorRGB(*status_color)
        c.rect(14 * cm, self.height - 4 * cm, 4 * cm, 1 * cm, fill=1)
        c.setFillColorRGB(1, 1, 1)
        c.setFont("Helvetica-Bold", 12)
        status_text = "PASSED" if passed else "FAILED"
        c.drawCentredString(16 * cm, self.height - 3.65 * cm, status_text)
        
        # Key Metrics Table
        c.setFillColorRGB(0, 0, 0)
        y_start = self.height - 4.8 * cm # Compressed header
        c.setFont("Helvetica-Bold", 14)
        c.drawString(2 * cm, y_start + 0.5 * cm, "Thermal Metrics")
        
        # Format metrics
        max_temp = data.get('max_temp_k') or 0.0
        min_temp = data.get('min_temp_k') or 0.0
        delta_t = max_temp - min_temp
        
        metrics = [
            ("Strategy", data.get('strategy_name', 'Default')),
            ("Ambient Temperature", f"{data.get('ambient_temp_c', 20.0):.1f} °C"),
            ("Heat Source", f"{data.get('source_temp_c', 100.0):.1f} °C"),
            ("Max Temperature", f"{max_temp:.2f} K ({max_temp - 273.15:.2f} °C)"),
            ("Min Temperature", f"{min_temp:.2f} K ({min_temp - 273.15:.2f} °C)"),
            ("Temperature Range (ΔT)", f"{delta_t:.2f} K"),
            ("Elements", f"{data.get('num_elements', 0):,}"),
            ("Nodes", f"{data.get('num_nodes', 0):,}"),
        ]
        
        c.setFont("Helvetica", 10)
        row_height = 0.5 * cm  # Hyper-compressed
        for i, (label, value) in enumerate(metrics):
            y = y_start - (i * row_height)
            c.setFont("Helvetica-Bold", 9)
            c.drawString(2.5 * cm, y, label + ":")
            c.setFont("Helvetica", 9)
            c.drawString(8 * cm, y, str(value))
            c.setStrokeColorRGB(0.9, 0.9, 0.9)
            c.line(2 * cm, y - 0.1 * cm, 12 * cm, y - 0.1 * cm)

        # Physics Validation
        y_validation = y - 0.7 * cm
        c.setFont("Helvetica-Bold", 12) # Smaller header
        c.setFillColorRGB(0, 0, 0)
        c.drawString(2 * cm, y_validation + 0.4 * cm, "Validation") # Shortened title
        
        checks = []
        temp_ok = min_temp > 0 and max_temp < 1000 and min_temp != 0.0 and max_temp != 0.0
        temp_status = "PASS" if temp_ok else "FAIL"
        temp_color = (0.2, 0.8, 0.2) if temp_ok else (0.8, 0.2, 0.2)
        checks.append(("Range", f"{temp_status} ({min_temp:.0f}K-{max_temp:.0f}K)", temp_color))
        
        gradient_ok = delta_t < 500 and delta_t > 0
        grad_status = "PASS" if gradient_ok else "WARN"
        grad_color = (0.2, 0.8, 0.2) if gradient_ok else (0.9, 0.6, 0.0)
        checks.append(("Gradient", f"{grad_status} (dT={delta_t:.0f}K)", grad_color))
        
        c.setFont("Helvetica", 9)
        for i, (label, status, color) in enumerate(checks):
            y_check = y_validation - (i * 0.45 * cm)
            c.setFillColorRGB(0, 0, 0)
            c.drawString(2.5 * cm, y_check, label + ":")
            c.setFillColorRGB(*color)
            c.drawString(6 * cm, y_check, status)
        
        # Material
        y_material = y_validation - (len(checks) * 0.45 * cm) - 0.5 * cm
        c.setFont("Helvetica-Bold", 12) # Smaller header
        c.setFillColorRGB(0, 0, 0)
        c.drawString(2 * cm, y_material + 0.4 * cm, "Material: Al6061")
        
        c.setFont("Helvetica", 9)
        mat_props = ["k: 167 W/mK", "Cp: 896 J/kgK", "Rho: 2700 kg/m3"] # Shortened
        for i, prop in enumerate(mat_props):
            c.drawString(2.5 * cm, y_material - (i * 0.4 * cm), prop)

        # Images Section
        y_img_start_text = y_material - (len(mat_props) * 0.4 * cm) - 0.3 * cm
        
        valid_images = [img for img in image_paths if Path(img).exists()]
        available_height = y_img_start_text - 2 * cm # Margin bottom
        
        if valid_images:
            # Page 1: Primary Image (Thermal Map / 4-Panel)
            # Use all available remaining height (with 2cm bottom margin)
            available_height = y_img_start_text - 2.5 * cm 
            
            # Constraint: Don't make it taller than 13cm or it looks weird
            img1_h = min(13 * cm, available_height)
            
            y_img1 = y_img_start_text - img1_h - 0.2*cm
            try:
                c.drawImage(valid_images[0], 2 * cm, y_img1, width=17*cm, height=img1_h, preserveAspectRatio=True)
            except Exception as e:
                logger.error(f"Failed to draw main image: {e}")
            
            # Page 2+: Secondary Images (Transient Graph, etc.)
            if len(valid_images) > 1:
                c.showPage()
                
                c.setFont("Helvetica-Bold", 14)
                c.drawString(2 * cm, self.height - 3 * cm, "Transient Response & Additional Views")
                
                current_y = self.height - 5 * cm
                
                for i, img_path in enumerate(valid_images[1:]):
                    # Check for space (assume 9cm per image)
                    if current_y < 10 * cm:
                        c.showPage()
                        current_y = self.height - 3 * cm
                    
                    try:
                        # Draw Image centered
                        c.drawImage(img_path, 2 * cm, current_y - 9*cm, width=17*cm, height=9*cm, preserveAspectRatio=True)
                        
                        # Label
                        name = Path(img_path).stem
                        c.setFont("Helvetica", 10)
                        c.drawString(2 * cm, current_y + 0.2*cm, name)
                        
                        current_y -= 10.5 * cm
                    except Exception as e:
                        logger.error(f"Failed to draw secondary image {img_path}: {e}")


        c.save()
        logger.info(f"Generated Thermal PDF report: {output_file}")
        return output_file
