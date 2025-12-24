import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

class PDFReportGenerator:
    """
    Generates professional thermal analysis reports for SimOps clients.
    """
    
    def __init__(self):
        self.width, self.height = A4
        self.colors = {
            'primary': colors.HexColor('#0d6efd'),    # Bootstrap Blue
            'secondary': colors.HexColor('#6c757d'),  # Gray
            'success': colors.HexColor('#198754'),    # Green
            'danger': colors.HexColor('#dc3545'),     # Red
            'warning': colors.HexColor('#ffc107'),    # Yellow
            'dark': colors.HexColor('#212529'),       # Dark Gray
            'light': colors.HexColor('#f8f9fa')       # Light Gray
        }
        
    def generate(self, 
                 job_name: str, 
                 output_dir: Path, 
                 data: Dict[str, Any], 
                 image_paths: list[str]) -> str:
        """
        Generate PDF report.
        
        Args:
            job_name: Name of the simulation job
            output_dir: Directory to save the PDF
            data: Dictionary containing simulation results and metadata
            image_paths: List of paths to plot images to embed
            
        Returns:
            Path to the generated PDF file
        """
        pdf_path = output_dir / f"{job_name}_report.pdf"
        
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=A4,
            rightMargin=20*mm,
            leftMargin=20*mm,
            topMargin=20*mm,
            bottomMargin=20*mm,
            title=f"SimOps Report - {job_name}"
        )
        
        story = []
        styles = getSampleStyleSheet()
        
        # Custom Styles
        title_style = ParagraphStyle(
            'SimOpsTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=self.colors['primary'],
            spaceAfter=30,
            alignment=1  # Center
        )
        
        heading_style = ParagraphStyle(
            'SimOpsHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=self.colors['dark'],
            spaceBefore=20,
            spaceAfter=12,
            borderWidth=0,
            borderColor=self.colors['secondary']
        )
        
        normal_style = styles['Normal']
        
        # 1. Header
        story.append(Paragraph("SimOps Thermal Analysis", title_style))
        story.append(Paragraph(f"<b>Job:</b> {job_name}", normal_style))
        story.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}", normal_style))
        story.append(Spacer(1, 20))
        
        # 2. Executive Summary / Grade
        # Determine grade (simple logic for now)
        max_temp = data.get('max_temp', 0)
        min_temp = data.get('min_temp', 0)
        delta_t = max_temp - min_temp
        
        grade = "PASS"
        grade_color = self.colors['success']
        
        # Arbitrary criteria for "FAIL" simulation (e.g., exploded solution)
        if max_temp > 5000 or min_temp < 0:
            grade = "CRITICAL FAILURE"
            grade_color = self.colors['danger']
        elif delta_t < 0.1:
            grade = "NO CONDUCTION"
            grade_color = self.colors['warning']
        
        # Stability Check (Courant Number)
        courant = data.get('courant_max', 0.0)
        if courant > 2.0:
            grade = "STABILITY WARNING (Co > 2)"
            grade_color = self.colors['warning']
        elif courant > 10.0:
            grade = "DIVERGED / UNSTABLE (Co > 10)"
            grade_color = self.colors['danger']
            
        grade_style = ParagraphStyle(
            'GradeParams',
            parent=styles['Normal'],
            fontSize=14,
            textColor=grade_color,
            alignment=1,
            spaceAfter=20
        )
        story.append(Paragraph(f"<b>SIMULATION STATUS: {grade}</b>", grade_style))
        story.append(Spacer(1, 10))

        # 3. Key Metrics Table
        metrics_data = [
            ['Metric', 'Value'],
            ['Strategy Used', data.get('strategy_name', 'Unknown')],
            ['Max Temperature', f"{max_temp:.1f} °C"],
            ['Min Temperature', f"{min_temp:.1f} °C"],
            ['Range (Delta T)', f"{delta_t:.1f} °C"],
            ['Max Courant No', f"{data.get('courant_max', 0.0):.3f} (Stability)"],
            ['Mesh Elements', f"{data.get('num_elements', 0):,}"],
            ['Solve Time', f"{data.get('solve_time', 0):.2f} s"],
            ['Total Heat Flux', f"{data.get('heat_flux', 0):.1f} W" if data.get('heat_flux') else "N/A"],
            ['Convergence', f"{data.get('convergence', 'N/A')} Steps"]
        ]
        
        t = Table(metrics_data, colWidths=[2.5*inch, 2.5*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), self.colors['primary']),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), self.colors['light']),
            ('GRID', (0, 0), (-1, -1), 1, self.colors['secondary']),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 11),  # Increased from 10
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, self.colors['light']])
        ]))
        story.append(t)
        story.append(Spacer(1, 30))
        
        # 4. Visualizations
        story.append(Paragraph("Thermal Distribution Maps", heading_style))
        
        for img_path in image_paths:
            if os.path.exists(img_path):
                # Scale image to fit page width
                img = Image(img_path)
                aspect = img.imageHeight / float(img.imageWidth)
                display_width = 6.5 * inch
                display_height = display_width * aspect
                
                # If too tall, scale down to ensured fit on page
                if display_height > 4.5 * inch:
                    display_height = 4.5 * inch
                    display_width = display_height / aspect
                    
                img.drawWidth = display_width
                img.drawHeight = display_height
                
                # If this is NOT the first image, add a page break and heading
                if img_path != image_paths[0]:
                    story.append(PageBreak())
                    story.append(Paragraph("Transient Analysis Results", heading_style))
                
                story.append(img)
                story.append(Spacer(1, 20))
            else:
                story.append(Paragraph(f"<i>Image missing: {Path(img_path).name}</i>", normal_style))

        # Footer / Disclaimer
        story.append(Spacer(1, 30))
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=self.colors['secondary'],
            alignment=1
        )
        story.append(Paragraph("Generated by SimOps Autonomous Pipeline", disclaimer_style))
        story.append(Paragraph(f"Confidential - {datetime.now().year}", disclaimer_style))

        # Build PDF
        doc.build(story)
        return str(pdf_path)
