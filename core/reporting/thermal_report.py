
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
        y_start = self.height - 7 * cm
        c.setFont("Helvetica-Bold", 14)
        c.drawString(2 * cm, y_start + 0.5 * cm, "Thermal Metrics")
        
        # Format metrics
        max_temp = data.get('max_temp_k', 0.0)
        min_temp = data.get('min_temp_k', 0.0)
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
        
        c.setFont("Helvetica", 11)
        row_height = 0.8 * cm
        for i, (label, value) in enumerate(metrics):
            y = y_start - (i * row_height)
            # Label
            c.setFont("Helvetica-Bold", 11)
            c.drawString(2.5 * cm, y, label + ":")
            # Value
            c.setFont("Helvetica", 11)
            c.drawString(8 * cm, y, str(value))
            
            # Draw line
            c.setStrokeColorRGB(0.9, 0.9, 0.9)
            c.line(2 * cm, y - 0.2 * cm, 12 * cm, y - 0.2 * cm)

        # Physics Validation Section
        y_validation = y - 1.5 * cm
        c.setFont("Helvetica-Bold", 14)
        c.setFillColorRGB(0, 0, 0)
        c.drawString(2 * cm, y_validation + 0.5 * cm, "Physics Validation")
        
        checks = []
        # Temperature range check
        temp_ok = min_temp > 0 and max_temp < 1000  # Reasonable range
        temp_status = "✓ PASS" if temp_ok else "✗ FAIL"
        temp_color = (0.2, 0.8, 0.2) if temp_ok else (0.8, 0.2, 0.2)
        checks.append(("Temperature Range", f"{temp_status} ({min_temp:.1f}K - {max_temp:.1f}K)", temp_color))
        
        # Gradient check (no unrealistic jumps)
        gradient_ok = delta_t < 500  # Less than 500K range
        grad_status = "✓ PASS" if gradient_ok else "⚠ WARNING"
        grad_color = (0.2, 0.8, 0.2) if gradient_ok else (0.9, 0.6, 0.0)
        checks.append(("Temperature Gradient", f"{grad_status} (ΔT = {delta_t:.1f}K)", grad_color))
        
        c.setFont("Helvetica", 10)
        for i, (label, status, color) in enumerate(checks):
            y_check = y_validation - (i * 0.6 * cm)
            c.setFillColorRGB(0, 0, 0)
            c.drawString(2.5 * cm, y_check, label + ":")
            c.setFillColorRGB(*color)
            c.drawString(6 * cm, y_check, status)
        
        # Material Properties
        y_material = y_validation - (len(checks) * 0.6 * cm) - 1.0 * cm
        c.setFont("Helvetica-Bold", 12)
        c.setFillColorRGB(0, 0, 0)
        c.drawString(2 * cm, y_material + 0.5 * cm, "Material: Al6061-T6")
        
        c.setFont("Helvetica", 9)
        mat_props = [
            "Thermal Conductivity: 167 W/(m·K)",
            "Specific Heat: 896 J/(kg·K)",
            "Density: 2700 kg/m³"
        ]
        for i, prop in enumerate(mat_props):
            c.drawString(2.5 * cm, y_material - (i * 0.5 * cm), prop)

        # Images
        y_img_start = y_material - (len(mat_props) * 0.5 * cm) - 1.0 * cm
        
        c.setFont("Helvetica-Bold", 14)
        c.setFillColorRGB(0, 0, 0)
        c.drawString(2 * cm, y_img_start + 1.0 * cm, "Temperature Distribution")
        
        valid_images = [img for img in image_paths if Path(img).exists()]
        
        if valid_images:
            # Page 1: Main (Isometric) view
            first_img = valid_images[0]
            try:
                # 17x12 cm image
                c.drawImage(first_img, 2 * cm, 2 * cm, width=17*cm, height=12*cm, preserveAspectRatio=True)
            except Exception as e:
                logger.error(f"Failed to draw thermal image: {e}")
                
            # Page 2: Multiple views grid
            if len(valid_images) > 1:
                c.showPage()
                
                # Grid config
                margin = 2 * cm
                w = (self.width - 2 * margin - 1 * cm) / 2
                h = w * 0.75
                
                positions = [
                    (margin, self.height - margin - h),
                    (margin + w + 1*cm, self.height - margin - h),
                    (margin, self.height - margin - 2*h - 1*cm),
                    (margin + w + 1*cm, self.height - margin - 2*h - 1*cm),
                ]
                
                c.setFont("Helvetica-Bold", 14)
                c.drawString(margin, self.height - margin + 0.5*cm, "Additional Views")
                
                remaining_images = valid_images[1:]
                for i, img_path in enumerate(remaining_images[:4]):
                    if i < len(positions):
                        x, y = positions[i]
                        try:
                            name = Path(img_path).stem
                            parts = name.split('_')
                            title = parts[-1].capitalize() if parts else "View"
                            
                            c.setFont("Helvetica-Bold", 10)
                            c.drawString(x, y + h + 0.1*cm, title)
                            
                            c.drawImage(img_path, x, y, width=w, height=h, preserveAspectRatio=True)
                        except Exception as e:
                            logger.error(f"Failed to draw grid image {img_path}: {e}")

        c.save()
        logger.info(f"Generated Thermal PDF report: {output_file}")
        return output_file
