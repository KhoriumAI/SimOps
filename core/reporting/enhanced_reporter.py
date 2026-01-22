"""
Enhanced Thermal PDF Reporter
==============================

Extends ThermalPDFReportGenerator with multi-angle visualizations and simulation grading.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

# Import parent class from core
from core.reporting.thermal_report import ThermalPDFReportGenerator

# Import local modules
from core.reporting.thermal_multi_angle_viz import generate_thermal_views
from core.reporting.templates import (
    COLORS, FONTS, PAGE_LAYOUTS, PAGE_WIDTH, PAGE_HEIGHT,
    draw_grade_badge, draw_metric_table, draw_header
)

logger = logging.getLogger(__name__)


class EnhancedThermalReporter(ThermalPDFReportGenerator):
    """
    Enhanced thermal PDF reporter with multi-angle visualizations and grading.
    
    Backward compatible - falls back to parent's single-view mode if vtu_path not provided.
    """
    
    def __init__(self, job_name: str = None, output_dir: Path = None, vtu_path: str = None, colormap: str = 'coolwarm'):
        """
        Initialize enhanced reporter.

        Args:
            job_name: Job/component name
            output_dir: Output directory for PDF and images
            vtu_path: Path to VTU file for multi-angle rendering (optional)
            colormap: Matplotlib colormap name (default: 'coolwarm')
        """
        super().__init__(job_name, output_dir)
        self.vtu_path = vtu_path
        self.colormap = colormap
    
    def add_simulation_grade(self, max_temp: float, min_temp: float, gradient: float) -> tuple:
        """
        Calculate simulation quality grade based on thermal metrics.
        
        Grading criteria:
        - A: ΔT < 100K, valid range (273-800K)
        - B: ΔT < 200K, valid range  
        - C: ΔT < 400K, valid range
        - FAIL: Invalid range or ΔT > 400K
        
        Args:
            max_temp: Maximum temperature (K)
            min_temp: Minimum temperature (K)
            gradient: Temperature range (ΔT)
            
        Returns:
            Tuple of (grade_letter, grade_color)
        """
        # Check temperature range validity
        valid_range = (273 <= min_temp <= 800) and (273 <= max_temp <= 800)
        
        if not valid_range:
            return ('FAIL', COLORS['grade_fail'])
        
        # Grade based on temperature gradient
        if gradient < 100:
            return ('A', COLORS['grade_a'])
        elif gradient < 200:
            return ('B', COLORS['grade_b'])
        elif gradient < 400:
            return ('C', COLORS['grade_c'])
        else:
            return ('FAIL', COLORS['grade_fail'])
    
    def generate_multi_angle_views(self, vtu_file: str = None) -> List[str]:
        """
        Generate 4 standard thermal views using PyVista.
        
        Args:
            vtu_file: Path to VTU file (uses self.vtu_path if not provided)
            
        Returns:
            List of image paths
        """
        vtu_path = vtu_file or self.vtu_path
        
        if not vtu_path:
            logger.warning("No VTU path provided, cannot generate multi-angle views")
            return []
        
        if not Path(vtu_path).exists():
            logger.error(f"VTU file not found: {vtu_path}")
            return []
        
        # Generate all 4 views
        output_dir = Path(self.output_dir)
        image_paths = generate_thermal_views(
            vtu_path=vtu_path,
            output_dir=output_dir,
            job_name=self.job_name,
            views=['isometric', 'top', 'front', 'section'],
            colormap=self.colormap
        )
        
        logger.info(f"Generated {len(image_paths)} thermal views")
        return image_paths
    
    def generate(self, job_name: str = None, output_dir: Path = None,
                 data: Dict = None, image_paths: List[str] = None,
                 vtu_path: str = None, colormap: str = None) -> Path:
        """
        Generate enhanced multi-page PDF report.

        Args:
            job_name: Job name (uses self.job_name if not provided)
            output_dir: Output directory (uses self.output_dir if not provided)
            data: Metrics dictionary
            image_paths: Pre-generated image paths (optional, will generate if not provided)
            vtu_path: VTU file path for multi-angle views
            colormap: Matplotlib colormap name (uses self.colormap if not provided)

        Returns:
            Path to generated PDF
        """
        # Use instance or provided values
        job_name = job_name or self.job_name
        output_dir = Path(output_dir or self.output_dir)
        data = data or {}
        vtu_path = vtu_path or self.vtu_path
        if colormap:
            self.colormap = colormap
        
        # If no VTU path, fall back to parent's single-view implementation
        if not vtu_path:
            logger.info("No VTU path provided, using standard single-view report")
            return super().generate(job_name, output_dir, data, image_paths)
        
        # Generate multi-angle views if not provided
        if not image_paths:
            image_paths = self.generate_multi_angle_views(vtu_path)
        
        if not image_paths:
            logger.warning("No images generated, falling back to standard report")
            return super().generate(job_name, output_dir, data, image_paths)
        
        # Create output file
        output_file = output_dir / f"{job_name}_thermal_report.pdf"
        c = canvas.Canvas(str(output_file), pagesize=A4)
        
        # ===== PAGE 1: Cover + Summary =====
        self._draw_cover_page(c, job_name, data)
        c.showPage()
        
        # ===== PAGE 2: Isometric View =====
        if len(image_paths) >= 1:
            self._draw_full_image_page(c, job_name, image_paths[0], "Isometric View")
            c.showPage()
        
        # ===== PAGE 3: Top + Front Views =====
        if len(image_paths) >= 3:
            self._draw_dual_image_page(c, job_name, image_paths[1], image_paths[2], 
                                      "Top View", "Front View")
            c.showPage()
        
        # ===== PAGE 4: Cross-Section View =====
        if len(image_paths) >= 4:
            self._draw_full_image_page(c, job_name, image_paths[3], "Cross-Section View")
            c.showPage()
        
        # ===== PAGE 5+: Transient plots (if provided in original image_paths) =====
        # This would be added by the caller if they have transient data
        
        # Save PDF
        c.save()
        logger.info(f"Generated Enhanced Thermal PDF: {output_file}")
        return output_file
    
    def _draw_cover_page(self, c, job_name: str, data: Dict):
        """Draw cover page with metadata, metrics, and grade."""
        layout = PAGE_LAYOUTS['cover']
        
        # Header
        c.setFont(*FONTS['title'])
        c.drawString(layout['margin'], layout['title_y'], "SimOps Thermal Analysis")
        
        # Job metadata
        c.setFont(*FONTS['heading'])
        y = layout['metadata_y']
        c.drawString(layout['margin'], y, f"Component: {job_name}")
        c.setFont(*FONTS['body'])
        c.drawString(layout['margin'], y - 0.7 * cm, 
                    f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        c.drawString(layout['margin'], y - 1.4 * cm,
                    f"Simulation Type: Thermal Conduction")
        
        # Calculate grade
        max_temp = data.get('max_temp_k') or data.get('max_temp_c', 0.0) + 273.15
        min_temp = data.get('min_temp_k') or data.get('min_temp_c', 0.0) + 273.15
        delta_t = max_temp - min_temp
        
        grade, grade_color = self.add_simulation_grade(max_temp, min_temp, delta_t)
        
        # Grade badge
        draw_grade_badge(c, layout['grade_x'], layout['grade_y'], grade, size=3.5 * cm)
        
        # Metrics table
        c.setFont(*FONTS['heading'])
        y_metrics = layout['metrics_y']
        c.drawString(layout['margin'], y_metrics, "Thermal Metrics")
        
        metrics = [
            ("Max Temperature", f"{max_temp:.1f} K ({max_temp - 273.15:.1f} °C)"),
            ("Min Temperature", f"{min_temp:.1f} K ({min_temp - 273.15:.1f} °C)"),
            ("Temperature Range (ΔT)", f"{delta_t:.1f} K"),
            ("Elements", f"{data.get('num_elements', 0):,}"),
            ("Nodes", f"{data.get('num_nodes', 0):,}"),
            ("Solve Time", f"{data.get('solve_time', 0.0):.1f} s"),
        ]
        
        draw_metric_table(c, layout['margin'] + 0.5 * cm, y_metrics - 0.7 * cm, metrics)
    
    def _draw_full_image_page(self, c, job_name: str, image_path: str, title: str):
        """Draw a page with a single full-size image."""
        layout = PAGE_LAYOUTS['full_image']
        
        # Page title
        c.setFont(*FONTS['heading'])
        c.drawString(layout['margin'], layout['title_y'], f"{job_name} - {title}")
        
        # Draw image
        try:
            if Path(image_path).exists():
                c.drawImage(
                    image_path,
                    layout['margin'],
                    layout['image_y'],
                    width=PAGE_WIDTH - 2 * layout['margin'],
                    height=layout['image_height'],
                    preserveAspectRatio=True,
                    anchor='c'
                )
        except Exception as e:
            logger.error(f"Failed to draw image {image_path}: {e}")
            c.setFont(*FONTS['body'])
            c.drawString(layout['margin'], layout['image_y'] + layout['image_height'] / 2,
                        f"Error loading image: {Path(image_path).name}")
    
    def _draw_dual_image_page(self, c, job_name: str, image1_path: str, image2_path: str,
                             title1: str, title2: str):
        """Draw a page with two images side-by-side."""
        layout = PAGE_LAYOUTS['dual_image']
        
        # Page title
        c.setFont(*FONTS['heading'])
        c.drawString(layout['margin'], layout['title_y'], 
                    f"{job_name} - {title1} & {title2}")
        
        # Calculate image dimensions (side by side)
        img_width = (PAGE_WIDTH - 3 * layout['margin'] - layout['image_spacing']) / 2
        img_height = layout['image_height']
        
        # Draw first image (left)
        try:
            if Path(image1_path).exists():
                c.drawImage(
                    image1_path,
                    layout['margin'],
                    layout['image_y'] + img_height + 1 * cm,
                    width=img_width,
                    height=img_height,
                    preserveAspectRatio=True,
                    anchor='c'
                )
                c.setFont(*FONTS['small'])
                c.drawString(layout['margin'], layout['image_y'] + img_height + 0.5 * cm, title1)
        except Exception as e:
            logger.error(f"Failed to draw image {image1_path}: {e}")
        
        # Draw second image (right)
        try:
            if Path(image2_path).exists():
                c.drawImage(
                    image2_path,
                    layout['margin'] + img_width + layout['image_spacing'],
                    layout['image_y'] + img_height + 1 * cm,
                    width=img_width,
                    height=img_height,
                    preserveAspectRatio=True,
                    anchor='c'
                )
                c.setFont(*FONTS['small'])
                c.drawString(layout['margin'] + img_width + layout['image_spacing'],
                           layout['image_y'] + img_height + 0.5 * cm, title2)
        except Exception as e:
            logger.error(f"Failed to draw image {image2_path}: {e}")
