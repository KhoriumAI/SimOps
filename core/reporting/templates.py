"""
PDF Report Layout Templates and Styling
========================================

ReportLab page layout templates and styling constants for SimOps thermal reports.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor

# Page dimensions
PAGE_WIDTH, PAGE_HEIGHT = A4

# SimOps Brand Colors
COLORS = {
    'primary': HexColor('#1E88E5'),      # Blue
    'success': HexColor('#43A047'),      # Green
    'warning': HexColor('#FB8C00'),      # Orange
    'danger': HexColor('#E53935'),       # Red
    'dark': HexColor('#212121'),         # Dark gray
    'light': HexColor('#F5F5F5'),        # Light gray
    'grade_a': HexColor('#4CAF50'),      # Green
    'grade_b': HexColor('#2196F3'),      # Blue
    'grade_c': HexColor('#FF9800'),      # Orange
    'grade_fail': HexColor('#F44336'),   # Red
}

# Typography
FONTS = {
    'title': ('Helvetica-Bold', 24),
    'heading': ('Helvetica-Bold', 16),
    'subheading': ('Helvetica-Bold', 12),
    'body': ('Helvetica', 10),
    'small': ('Helvetica', 8),
    'metric_label': ('Helvetica-Bold', 10),
    'metric_value': ('Helvetica', 10),
}

# Page Layouts
PAGE_LAYOUTS = {
    'cover': {
        'title_y': PAGE_HEIGHT - 3 * cm,
        'metadata_y': PAGE_HEIGHT - 5 * cm,
        'metrics_y': PAGE_HEIGHT - 7 * cm,
        'grade_x': 15 * cm,
        'grade_y': PAGE_HEIGHT - 4 * cm,
        'margin': 2 * cm,
    },
    'full_image': {
        'title_y': PAGE_HEIGHT - 2.5 * cm,
        'image_y': 3 * cm,
        'image_height': PAGE_HEIGHT - 6 * cm,
        'margin': 2 * cm,
    },
    'dual_image': {
        'title_y': PAGE_HEIGHT - 2.5 * cm,
        'image_y': 3 * cm,
        'image_height': (PAGE_HEIGHT - 7 * cm) / 2,
        'image_spacing': 0.5 * cm,
        'margin': 2 * cm,
    }
}


def draw_grade_badge(canvas, x, y, grade: str, size: float = 4 * cm):
    """
    Draw a simulation grade badge (A/B/C/FAIL).
    
    Args:
        canvas: ReportLab canvas
        x: X position (left edge)
        y: Y position (bottom edge)
        grade: Grade letter ('A', 'B', 'C', 'FAIL')
        size: Badge size (width and height)
    """
    # Determine color based on grade
    color_map = {
        'A': COLORS['grade_a'],
        'B': COLORS['grade_b'],
        'C': COLORS['grade_c'],
        'FAIL': COLORS['grade_fail'],
    }
    
    color = color_map.get(grade.upper(), COLORS['light'])
    
    # Draw background rectangle
    canvas.setFillColor(color)
    canvas.rect(x, y, size, size * 0.6, fill=1, stroke=0)
    
    # Draw grade text
    canvas.setFillColorRGB(1, 1, 1)  # White text
    canvas.setFont('Helvetica-Bold', int(size * 0.3))
    
    # Center text
    text_x = x + size / 2
    text_y = y + size * 0.15
    canvas.drawCentredString(text_x, text_y, grade.upper())
    
    # Reset colors
    canvas.setFillColorRGB(0, 0, 0)


def draw_metric_table(canvas, x, y, metrics: list, row_height: float = 0.5 * cm):
    """
    Draw a metrics table with label-value pairs.
    
    Args:
        canvas: ReportLab canvas
        x: X position (left edge)
        y: Y position (top edge)
        metrics: List of (label, value) tuples
        row_height: Height of each row
    """
    canvas.setFont(*FONTS['metric_label'])
    
    for i, (label, value) in enumerate(metrics):
        current_y = y - (i * row_height)
        
        # Draw label
        canvas.setFont(*FONTS['metric_label'])
        canvas.drawString(x, current_y, f"{label}:")
        
        # Draw value
        canvas.setFont(*FONTS['metric_value'])
        canvas.drawString(x + 6 * cm, current_y, str(value))
        
        # Draw separator line
        canvas.setStrokeColorRGB(0.9, 0.9, 0.9)
        canvas.line(x - 0.5 * cm, current_y - 0.1 * cm, 
                   x + 11 * cm, current_y - 0.1 * cm)
    
    # Reset colors
    canvas.setStrokeColorRGB(0, 0, 0)


def draw_header(canvas, title: str, y: float = None):
    """
    Draw a page header.
    
    Args:
        canvas: ReportLab canvas
        title: Header text
        y: Y position (default: top of page)
    """
    if y is None:
        y = PAGE_HEIGHT - 3 * cm
    
    canvas.setFont(*FONTS['title'])
    canvas.drawString(2 * cm, y, title)
