
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime

def create_report(filename):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    # Custom Styles
    title_style = styles['Title']
    heading2_style = styles['Heading2']
    heading3_style = styles['Heading3']
    normal_style = styles['Normal']
    
    # Title
    story.append(Paragraph("Feature Parity & Migration Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", normal_style))
    story.append(Spacer(1, 0.25*inch))

    # Executive Summary
    story.append(Paragraph("Executive Summary", heading2_style))
    story.append(Paragraph(
        "This report outlines the feature gap between the on-premise desktop application and the AWS web frontend. "
        "It categorizes features into Critical Requirements (immediate migration needed), Future/Nice-to-Have features, "
        "and Excluded/Modified features that do not need to be ported to the web environment.", 
        normal_style
    ))
    story.append(Spacer(1, 0.2*inch))

    # SECTION 1: CRITICAL MISSING FEATURES
    story.append(Paragraph("1. Critical Requirements (Must Have)", heading2_style))
    story.append(Paragraph("The following features are present in the desktop app and are critical for the web workflow:", normal_style))
    story.append(Spacer(1, 0.1*inch))

    # Column 2 style for wrapping
    col2_style = ParagraphStyle(
        'Col2Style',
        parent=normal_style,
        fontSize=10,
        leading=12,
    )
    # Header style
    header_style = ParagraphStyle(
        'HeaderStyle',
        parent=normal_style,
        fontSize=10,
        leading=12,
        textColor=colors.whitesmoke,
        fontName='Helvetica-Bold'
    )
    header_style_black = ParagraphStyle(
        'HeaderStyleBlack',
        parent=normal_style,
        fontSize=10,
        leading=12,
        textColor=colors.black,
        fontName='Helvetica-Bold'
    )

    critical_data = [
        [Paragraph("Feature", header_style), Paragraph("Description / Requirement", header_style)],
        [Paragraph("Min Element Size", normal_style), Paragraph("Input field for minimum element size (decoupled from max size).", col2_style)],
        [Paragraph("Element Order Selector", normal_style), Paragraph("Dropdown for Tet4 (Linear) and Tet10 (Quadratic). Critical for FEA.", col2_style)],
        [Paragraph("ANSYS Export", normal_style), Paragraph("Export modes for CFD (Fluent) and FEA (Mechanical).", col2_style)],
        [Paragraph("Hex Strategies", normal_style), Paragraph("Implementation of 'True Hex' Strategy (formerly Hex Dominant Testing).", col2_style)],
        [Paragraph("Refine Mesh Quality", normal_style), Paragraph("Button to trigger optimization. Must include 'Show Pre-refinement' toggle.", col2_style)],
        [Paragraph("Cross-Section Viewer", normal_style), Paragraph("Visual inspection tool with Axis Selection (X/Y/Z) and Offset Slider.", col2_style)],
        [Paragraph("Quality Visualization", normal_style), Paragraph("Visual overlay for metrics: SICN, Gamma, Skewness, Aspect Ratio, Jacobian.", col2_style)],
        [Paragraph("Strategy Fallback", normal_style), Paragraph("Backend must default to HXT. If it fails, fallback to Frontal/Delaunay and ALERT the user.", col2_style)]
    ]

    t_critical = Table(critical_data, colWidths=[2*inch, 4.5*inch])
    t_critical.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dc3545')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_critical)
    story.append(Spacer(1, 0.2*inch))


    # SECTION 2: NICE TO HAVE
    story.append(Paragraph("2. Useful / Future Features", heading2_style))
    story.append(Paragraph("Features that add value but are not blocking deployment:", normal_style))
    story.append(Spacer(1, 0.1*inch))

    nice_data = [
        [Paragraph("Feature", header_style_black), Paragraph("Description", header_style_black)],
        [Paragraph("Surface Mesh Only", normal_style), Paragraph("Mode to generate only 2D surface mesh (useful for validation).", col2_style)]
    ]

    t_nice = Table(nice_data, colWidths=[2*inch, 4.5*inch])
    t_nice.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ffc107')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_nice)
    story.append(Spacer(1, 0.2*inch))


    # SECTION 3: EXCLUDED / MODIFIED
    story.append(Paragraph("3. Excluded or Modified Features", heading2_style))
    story.append(Paragraph("Features intentionally omitted or handled differently on the web:", normal_style))
    story.append(Spacer(1, 0.1*inch))

    excluded_data = [
        [Paragraph("Feature", header_style), Paragraph("Reason / Handling", header_style)],
        [Paragraph("Parallel Workers Slider", normal_style), Paragraph("AWS Managed: Auto-assign max workers with reserve.", col2_style)],
        [Paragraph("GPU / Polyhedral", normal_style), Paragraph("Not ready for deployment.", col2_style)],
        [Paragraph("Score Threshold / Strategy List", normal_style), Paragraph("Not needed. System defaults to best strategy (HXT) automatically.", col2_style)],
        [Paragraph("Defer Quality / Fast Mode", normal_style), Paragraph("Not needed.", col2_style)],
        [Paragraph("Fullscreen / Ghost / Hex Viz", normal_style), Paragraph("Not needed independent controls.", col2_style)],
        [Paragraph("Cell Type Selector", normal_style), Paragraph("Auto-determine based on mesh type.", col2_style)],
        [Paragraph("ETA Display", normal_style), Paragraph("Removed due to inaccuracy.", col2_style)],
        [Paragraph("AI Chat / Paintbrush", normal_style), Paragraph("Not ready for current deployment phase.", col2_style)]
    ]

    t_excluded = Table(excluded_data, colWidths=[2.5*inch, 4.0*inch])
    t_excluded.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6c757d')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_excluded)

    # Build PDF
    doc.build(story)
    print(f"PDF Report generated successfully: {filename}")

if __name__ == "__main__":
    create_report("MeshGen_Feature_Parity_Report.pdf")
