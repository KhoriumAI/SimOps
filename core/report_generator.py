"""
SimOps Report Generator
=======================

Generates professional PDF reports and high-quality visualizations 
for thermal simulation results.

Features:
- Multi-page PDF reports with branding
- Enhanced contour plots (tricontourf) with scientific colormaps
- Mesh statistics summaries
- Temperature distribution analysis

Author: SimOps Team (Agent 5)
Date: 2024
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

class SimOpsReportGenerator:
    """Generates professional PDF reports from simulation results"""
    
    def __init__(self, job_name: str, output_dir: Path, results: Dict):
        """
        Initialize report generator.
        
        Args:
            job_name: Name of the simulation job
            output_dir: Directory to save reports
            results: Simulation results dictionary containing:
                     - node_coords: Nx3 array
                     - temperature: N array
                     - elements: List or array of elements
                     - min_temp, max_temp: floats
                     - solve_time: float
        """
        self.job_name = job_name
        self.output_dir = Path(output_dir)
        self.results = results
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract key data
        self.coords = results['node_coords']
        self.temp = results['temperature']
        self.elements = results.get('elements', [])
        
        # Ensure correct shapes
        if len(self.coords.shape) != 2 or self.coords.shape[1] != 3:
             self.coords = self.coords.reshape(-1, 3)
             
        # Branding colors
        self.colors = {
            'primary': '#0d6efd',    # Bootstrap Blue
            'secondary': '#6c757d',  # Gray
            'success': '#198754',    # Green
            'danger': '#dc3545',     # Red
            'dark': '#212529',       # Dark / Text
            'light': '#f8f9fa'       # Light / Background
        }

    def generate_enhanced_plots(self) -> Dict[str, str]:
        """
        Generate high-quality PNG visualizations.
        
        Returns:
            Dictionary of view names to file paths
        """
        generated_files = {}
        
        # Coordinate projections
        views = [
            ('XY_Top', 0, 1, 'X Position (m)', 'Y Position (m)'),
            ('XZ_Front', 0, 2, 'X Position (m)', 'Z Position (m)'),
            ('YZ_Side', 1, 2, 'Y Position (m)', 'Z Position (m)'),
        ]
        
        T_min = self.results.get('min_temp', np.min(self.temp))
        T_max = self.results.get('max_temp', np.max(self.temp))
        
        # Create a single figure for the summary PNG (3 panels)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')
        
        for ax, (name, xi, yi, xlabel, ylabel) in zip(axes, views):
            x = self.coords[:, xi]
            y = self.coords[:, yi]
            
            # Create triangulation for contour plot
            triang = mtri.Triangulation(x, y)
            
            # Mask triangles with long edges to avoid artifacts in concave shapes
            # (Simple heuristic: mask edges > 20% of bounding box diagonal)
            # This is optional but improves quality for complex shapes
            
            # Filled contour plot
            levels = np.linspace(T_min, T_max, 21)
            cntr = ax.tricontourf(triang, self.temp, levels=levels, cmap='turbo', extend='both')
            
            # Add contour lines
            # ax.tricontour(triang, self.temp, levels=levels[::4], colors='k', linewidths=0.5, alpha=0.5)
            
            ax.set_title(name.replace('_', ' '), fontsize=14, fontweight='bold', color=self.colors['dark'])
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_aspect('equal')
            ax.grid(True, linestyle=':', alpha=0.3)
            
            # Find and mark max temp location
            max_idx = np.argmax(self.temp)
            ax.plot(self.coords[max_idx, xi], self.coords[max_idx, yi], 'r*', markersize=15, markeredgecolor='white')
            ax.text(self.coords[max_idx, xi], self.coords[max_idx, yi], f' {T_max:.0f}K', 
                   color='red', fontweight='bold', ha='left', va='bottom')

        # Add common colorbar
        cbar = fig.colorbar(cntr, ax=axes.ravel().tolist(), pad=0.02, aspect=30)
        cbar.set_label('Temperature (K)', fontsize=12, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)
        
        # Add Header
        fig.suptitle(f"SimOps Thermal Analysis: {self.job_name}\n"
                    f"Range: {T_min:.1f}K - {T_max:.1f}K | Max Temp: {T_max - 273.15:.1f}°C",
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Save summary plot
        summary_file = self.output_dir / f"{self.job_name}_temperature_map.png"
        plt.savefig(summary_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        generated_files['summary'] = str(summary_file)
        return generated_files

    def generate_pdf_report(self) -> str:
        """
        Generate a multi-page PDF report.
        
        Returns:
            Path to the generated PDF file
        """
        pdf_file = self.output_dir / f"{self.job_name}_report.pdf"
        
        with PdfPages(pdf_file) as pdf:
            # --- PAGE 1: Summary Dashboard ---
            fig_summary = plt.figure(figsize=(11.69, 8.27)) # A4 Landscape
            
            # 1. Branding Header
            self._add_header(fig_summary, "Thermal Analysis Report")
            
            # 2. Key Metrics Table (Left)
            stats_text = (
                f"Job Name:     {self.job_name}\n"
                f"Date:         {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                f"Solver Time:  {self.results.get('solve_time', 0):.2f} s\n"
                f"Strategy:     {self.results.get('strategy_name', 'Auto')}\n"
                f"\n"
                f"Nodes:        {len(self.coords):,}\n"
                f"Elements:     {self.results.get('num_elements', len(self.elements)):,}\n"
                f"\n"
                f"Min Temp:     {self.results.get('min_temp', 0):.1f} K\n"
                f"Max Temp:     {self.results.get('max_temp', 0):.1f} K\n"
                f"Delta T:      {self.results.get('max_temp', 0) - self.results.get('min_temp', 0):.1f} K"
            )
            
            fig_summary.text(0.05, 0.65, "Simulation Statistics", fontsize=14, fontweight='bold', color=self.colors['primary'])
            fig_summary.text(0.05, 0.40, stats_text, fontsize=11, fontfamily='monospace', va='top')
            
            # 3. Temperature Distribution Histogram (Bottom Left)
            ax_hist = fig_summary.add_axes([0.05, 0.1, 0.35, 0.25])
            ax_hist.hist(self.temp, bins=50, color=self.colors['primary'], alpha=0.7, edgecolor='white')
            ax_hist.set_title("Temperature Distribution", fontsize=10, fontweight='bold')
            ax_hist.set_xlabel("Temperature (K)", fontsize=9)
            ax_hist.set_ylabel("Node Count", fontsize=9)
            ax_hist.grid(True, linestyle=':', alpha=0.5)
            
            # 4. Main 3D isometric view placeholder (Right)
            # Since we can't easily do 3D in 2D matplotlib efficiently without VTK offscreen,
            # we'll use the generated summary plot from generate_enhanced_plots instead.
            
            # ... but first let's re-generate the 3-view plot *inside* this figure for the PDF
            views = [
                ('XY (Top)', 0, 1, [0.45, 0.55, 0.25, 0.25]),
                ('XZ (Front)', 0, 2, [0.72, 0.55, 0.25, 0.25]),
                ('YZ (Side)', 1, 2, [0.45, 0.20, 0.25, 0.25]),
            ]
            
            T_min = self.results.get('min_temp', np.min(self.temp))
            T_max = self.results.get('max_temp', np.max(self.temp))
            levels = np.linspace(T_min, T_max, 21)
            
            for name, xi, yi, rect in views:
                ax = fig_summary.add_axes(rect)
                x = self.coords[:, xi]
                y = self.coords[:, yi]
                triang = mtri.Triangulation(x, y)
                cntr = ax.tricontourf(triang, self.temp, levels=levels, cmap='turbo', extend='both')
                ax.set_title(name, fontsize=10, fontweight='bold')
                ax.set_aspect('equal')
                ax.set_xticks([])
                ax.set_yticks([])
                
            # Add colorbar for the views
            cax = fig_summary.add_axes([0.72, 0.20, 0.02, 0.25])
            cbar = plt.colorbar(cntr, cax=cax)
            cbar.set_label('Temperature (K)', fontsize=9)
            
            pdf.savefig(fig_summary)
            plt.close(fig_summary)
            
            # --- PAGE 2: Detailed Views ---
            fig_detail = plt.figure(figsize=(11.69, 8.27))
            self._add_header(fig_detail, "Detailed Temperature Maps")
            
            # Large Top View
            ax_main = fig_detail.add_axes([0.1, 0.1, 0.8, 0.75])
            x = self.coords[:, 0]
            y = self.coords[:, 1]
            triang = mtri.Triangulation(x, y)
            cntr = ax_main.tricontourf(triang, self.temp, levels=levels, cmap='turbo', extend='both')
            cbar = plt.colorbar(cntr, ax=ax_main, orientation='horizontal', pad=0.05, aspect=40)
            cbar.set_label('Temperature (K)', fontsize=12)
            ax_main.set_title("Full XY Plan View", fontsize=16, fontweight='bold')
            ax_main.set_aspect('equal')
            ax_main.grid(True, linestyle='--', alpha=0.3)
            
            pdf.savefig(fig_detail)
            plt.close(fig_detail)
            
        return str(pdf_file)

    def _add_header(self, fig, subtitle: str):
        """Add branding header to page"""
        # Blue banner
        fig.patches.extend([
            plt.Rectangle((0, 0.9), 1, 0.1, transform=fig.transFigure, 
                         facecolor=self.colors['primary'], alpha=0.1, zorder=-1)
        ])
        
        # SimOps Logo Text
        fig.text(0.05, 0.94, "SimOps", fontsize=24, fontweight='bold', color=self.colors['primary'])
        fig.text(0.18, 0.94, "|", fontsize=24, color='#ccc')
        fig.text(0.20, 0.95, subtitle, fontsize=14, color=self.colors['secondary'])
        
        # Footer
        fig.text(0.5, 0.02, f"Generated by SimOps AI • {datetime.now().strftime('%Y-%m-%d')}", 
                ha='center', fontsize=8, color='#999')

# --- TEST HARNESS ---
if __name__ == "__main__":
    # Create dummy data for testing
    print("Running SimOpsReportGenerator Test...")
    
    # 1. Generate cylinder mesh points
    theta = np.linspace(0, 2*np.pi, 50)
    z = np.linspace(0, 10, 20)
    theta, z = np.meshgrid(theta, z)
    r = 5.0
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    coords = np.column_stack((x.flatten(), y.flatten(), z.flatten()))
    
    # 2. Assign temperature (hot top, cold bottom)
    temp = 300 + 500 * (coords[:, 2] / 10.0)
    
    # 3. Create mock results dict
    results = {
        'node_coords': coords,
        'temperature': temp,
        'elements': [], # Not needed for 2D plots
        'min_temp': 300.0,
        'max_temp': 800.0,
        'num_elements': 1000,
        'solve_time': 1.23,
        'strategy_name': 'Test_Strategy'
    }
    
    # 4. Generate reports
    gen = SimOpsReportGenerator("Test_Cylinder", ".", results)
    
    try:
        png_files = gen.generate_enhanced_plots()
        print(f"[OK] Generated PNG: {png_files['summary']}")
        
        pdf_file = gen.generate_pdf_report()
        print(f"[OK] Generated PDF: {pdf_file}")
        
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
