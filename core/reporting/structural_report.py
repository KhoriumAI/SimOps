
import logging
from pathlib import Path
from typing import Dict, List, Optional
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from datetime import datetime

logger = logging.getLogger(__name__)

def format_displacement(disp_mm: float) -> str:
    """Format displacement with appropriate units (μm, nm, mm)."""
    if disp_mm < 1e-6:
        return f"{disp_mm * 1e9:.2f} nm"
    elif disp_mm < 1e-3:
        return f"{disp_mm * 1e6:.2f} μm"
    elif disp_mm < 1.0:
        return f"{disp_mm * 1e3:.2f} μm"
    else:
        return f"{disp_mm:.4f} mm"

class StructuralPDFReportGenerator:
    """
    Generates a PDF report for static structural analysis.
    Designed to work with the data returned by CalculiXStructuralAdapter.
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
        
        output_file = Path(output_dir) / f"{job_name}_structural_report.pdf"
        
        c = canvas.Canvas(str(output_file), pagesize=A4)
        
        # Header
        c.setFont("Helvetica-Bold", 24)
        c.drawString(2 * cm, self.height - 3 * cm, "SimOps Structural Report")
        
        c.setFont("Helvetica", 14)
        c.drawString(2 * cm, self.height - 4 * cm, f"Component: {job_name}")
        c.drawString(2 * cm, self.height - 4.7 * cm, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        # Status Box
        passed = data.get('success', False) # Check for success/passed in data
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
        c.drawString(2 * cm, y_start + 0.5 * cm, "Analysis Metrics")
        
        # Format metrics
        max_stress = data.get('max_stress_mpa', 0.0)
        max_disp_mm = data.get('max_displacement_mm', 0.0)
        disp_display = format_displacement(max_disp_mm)
        
        metrics = [
            ("Strategy", data.get('strategy_name', 'Default')),
            ("Load Case", f"{data.get('g_factor', 1.0):.1f}G Z-Load"),
            ("Max Von Mises Stress", f"{max_stress:.4E} MPa" if max_stress < 0.1 else f"{max_stress:.2f} MPa"),
            ("Max Displacement", disp_display),
            ("Safety Factor", f"{(data.get('yield_strength_mpa', 276.0) / max_stress):.2f}" if max_stress > 1e-6 else "> 1000"),
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
        
        # Validation checks
        yield_strength = data.get('yield_strength_mpa', 276.0)  # Al6061-T6
        safety_factor_target = 1.5
        
        checks = []
        # Stress check
        if max_stress > 1e-6:
            stress_ok = max_stress < (yield_strength / safety_factor_target)
            stress_status = "✓ PASS" if stress_ok else "✗ FAIL"
            stress_color = (0.2, 0.8, 0.2) if stress_ok else (0.8, 0.2, 0.2)
            checks.append(("Stress Margin", f"{stress_status} ({max_stress:.2f} < {yield_strength/safety_factor_target:.1f} MPa)", stress_color))
        
        # Displacement check (< 0.1% of characteristic size)
        char_size = data.get('characteristic_size_mm', 50.0)
        disp_limit = char_size * 0.001  # 0.1%
        disp_ok = max_disp_mm < disp_limit
        disp_status = "✓ PASS" if disp_ok else "⚠ WARNING"
        disp_color = (0.2, 0.8, 0.2) if disp_ok else (0.9, 0.6, 0.0)
        checks.append(("Displacement", f"{disp_status} ({disp_display} < {format_displacement(disp_limit)})", disp_color))
        
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
            f"Yield Strength: {yield_strength:.0f} MPa",
            f"Young's Modulus: 68.9 GPa",
            f"Density: 2700 kg/m³"
        ]
        for i, prop in enumerate(mat_props):
            c.drawString(2.5 * cm, y_material - (i * 0.5 * cm), prop)

        # Images
        y_img_start = y_material - (len(mat_props) * 0.5 * cm) - 1.0 * cm
        
        c.setFont("Helvetica-Bold", 14)
        c.setFillColorRGB(0, 0, 0)
        c.drawString(2 * cm, y_img_start + 1.0 * cm, "Stress Distribution")
        
        valid_images = [img for img in image_paths if Path(img).exists()]
        
        if valid_images:
            # Page 1: Main (Isometric) view
            first_img = valid_images[0]
            try:
                # 17x12 cm image
                c.drawImage(first_img, 2 * cm, 2 * cm, width=17*cm, height=12*cm, preserveAspectRatio=True)
            except Exception as e:
                logger.error(f"Failed to draw structural image: {e}")
                
            # Page 2: Multiple views grid
            if len(valid_images) > 1:
                c.showPage() # Start Page 2
                
                # Grid config
                margin = 2 * cm
                # Use remaining height for 2x2 grid
                w = (self.width - 2 * margin - 1 * cm) / 2
                h = w * 0.75 # 4:3 aspect
                
                positions = [
                    (margin, self.height - margin - h),
                    (margin + w + 1*cm, self.height - margin - h),
                    (margin, self.height - margin - 2*h - 1*cm),
                    (margin + w + 1*cm, self.height - margin - 2*h - 1*cm),
                ]
                
                # Title for page 2
                c.setFont("Helvetica-Bold", 14)
                c.drawString(margin, self.height - margin + 0.5*cm, "Additional Views")
                
                # Loop through remaining images (skip first one which was Iso)
                remaining_images = valid_images[1:]
                for i, img_path in enumerate(remaining_images[:4]):
                    if i < len(positions):
                        x, y = positions[i]
                        try:
                            # Extract view name from filename if possible
                            name = Path(img_path).stem
                            parts = name.split('_')
                            title = parts[-1].capitalize() if parts else "View"
                            
                            c.setFont("Helvetica-Bold", 10)
                            c.drawString(x, y + h + 0.1*cm, title)
                            
                            c.drawImage(img_path, x, y, width=w, height=h, preserveAspectRatio=True)
                        except Exception as e:
                            logger.error(f"Failed to draw grid image {img_path}: {e}")

        c.save()
        logger.info(f"Generated Structural PDF report: {output_file}")
        return output_file
