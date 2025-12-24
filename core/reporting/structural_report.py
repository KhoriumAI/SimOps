import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

class StructuralPDFReportGenerator:
    """
    Generates professional Structural Analysis reports including Stress/Displacement metrics and visualizations.
    """
    
    def __init__(self):
        self.width, self.height = A4
        self.colors = {
            'primary': colors.HexColor('#6610f2'),    # Indigo/Purple for Structural
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
        Generate Structural PDF report.
        
        Args:
            job_name: Name of the simulation job
            output_dir: Directory to save the PDF
            data: Dictionary containing simulation results (Stress, Displacement, etc.)
            image_paths: List of paths to plot images to embed
            
        Returns:
            Path to the generated PDF file
        """
        pdf_path = output_dir / f"{job_name}_structural_report.pdf"
        
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=A4,
            rightMargin=20*mm,
            leftMargin=20*mm,
            topMargin=20*mm,
            bottomMargin=20*mm,
            title=f"SimOps Structural Report - {job_name}"
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
        story.append(Paragraph("SimOps Structural Analysis", title_style))
        story.append(Paragraph(f"<b>Job:</b> {job_name}", normal_style))
        story.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}", normal_style))
        story.append(Spacer(1, 20))
        
        # 2. Executive Summary / Grade
        success = data.get('success', False)
        
        grade = "SUCCESS" if success else "FAILED"
        grade_color = self.colors['success'] if success else self.colors['danger']
            
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
        # Extract metrics
        max_stress = data.get('max_stress', 0.0)
        max_disp = data.get('max_displacement', 0.0)
        
        # Format
        if isinstance(max_stress, float): max_stress = f"{max_stress:.4f}"
        if isinstance(max_disp, float): max_disp = f"{max_disp:.4f}"

        metrics_data = [
            ['Metric', 'Value'],
            ['Strategy Used', data.get('strategy_name', 'CalculiX Structural')],
            ['Max Von Mises Stress', f"{max_stress} MPa"],
            ['Max Displacement', f"{max_disp} mm"],
            ['Max Strain', f"{data.get('max_strain', 0):.6f}"],
            ['Reaction Force (Z)', f"{data.get('reaction_force_z', 0):.2f} N"],
            ['Mesh Elements', f"{data.get('num_elements', 0):,}"],
            ['Solve Time', f"{data.get('solve_time', 0):.2f} s"],
        ]
        
        # Add Load Info if available
        if 'load_info' in data:
            metrics_data.append(['Load Condition', str(data['load_info'])])
        
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
            ('FONTSIZE', (0, 1), (-1, -1), 11),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, self.colors['light']])
        ]))
        story.append(t)
        story.append(Spacer(1, 30))
        
        # 4. Visualizations
        story.append(Paragraph("Structural Response Visualization", heading_style))
        
        if not image_paths:
             story.append(Paragraph("<i>No visualization images available.</i>", normal_style))
        
        for img_path in image_paths:
            if img_path and os.path.exists(img_path):
                # Scale image to fit page width
                try:
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
                    
                    # If this is NOT the first image, add a page break?
                    if img_path != image_paths[0]:
                        story.append(Spacer(1, 20))
                    
                    story.append(img)
                    story.append(Spacer(1, 10))
                    story.append(Paragraph(f"<i>Figure: {Path(img_path).name}</i>", normal_style))
                    story.append(Spacer(1, 20))
                except Exception as e:
                     story.append(Paragraph(f"<i>Image Error: {e}</i>", normal_style))
            elif img_path:
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
