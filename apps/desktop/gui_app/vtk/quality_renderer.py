"""
Quality Renderer
================

Handles the generation and display of mesh quality metrics and reports.
"""

from typing import Dict
from PyQt5.QtCore import Qt

class QualityRenderer:
    def __init__(self, viewer):
        self.viewer = viewer

    def show_report(self, metrics: Dict):
        """Show quality metrics overlay in top-right"""
        print(f"[QUALITY_RENDERER DEBUG] show_report called")
        print(f"[QUALITY_RENDERER DEBUG] metrics keys: {list(metrics.keys())}")
        print(f"[QUALITY_RENDERER DEBUG] metrics: {metrics}")
        
        # Extract from nested structure (gmsh_sicn: {min, avg, max})
        sicn_min = None
        if 'gmsh_sicn' in metrics and isinstance(metrics['gmsh_sicn'], dict):
            sicn_min = metrics['gmsh_sicn'].get('min')
        elif 'sicn_min' in metrics:
            sicn_min = metrics['sicn_min']

        gamma_min = None
        if 'gmsh_gamma' in metrics and isinstance(metrics['gmsh_gamma'], dict):
            gamma_min = metrics['gmsh_gamma'].get('min')
        elif 'gamma_min' in metrics:
            gamma_min = metrics['gamma_min']

        max_skew = None
        if 'skewness' in metrics and isinstance(metrics['skewness'], dict):
            max_skew = metrics['skewness'].get('max')
        elif 'max_skewness' in metrics:
            max_skew = metrics['max_skewness']

        max_ar = None
        if 'aspect_ratio' in metrics and isinstance(metrics['aspect_ratio'], dict):
            max_ar = metrics['aspect_ratio'].get('max')
        elif 'max_aspect_ratio' in metrics:
            max_ar = metrics['max_aspect_ratio']

        # Extract badness (element quality measure from Netgen)
        badness = None
        if 'badness' in metrics and isinstance(metrics['badness'], dict):
            badness = metrics['badness'].get('max')
        elif 'max_badness' in metrics:
            badness = metrics['max_badness']

        # Extract worst tet radius ratio
        worst_tet_radius = None
        if 'tet_radius_ratio' in metrics and isinstance(metrics['tet_radius_ratio'], dict):
            worst_tet_radius = metrics['tet_radius_ratio'].get('min')
        elif 'worst_tet_radius_ratio' in metrics:
            worst_tet_radius = metrics['worst_tet_radius_ratio']

        # Get geometric accuracy (most important for shape fidelity)
        geom_accuracy = metrics.get('geometric_accuracy')
        mean_dev = metrics.get('mean_deviation_mm')
        
        print(f"[QUALITY_RENDERER DEBUG] Extracted values:")
        print(f"  sicn_min: {sicn_min}")
        print(f"  gamma_min: {gamma_min}")
        print(f"  max_skew: {max_skew}")
        print(f"  max_ar: {max_ar}")
        print(f"  badness: {badness}")
        print(f"  worst_tet_radius: {worst_tet_radius}")
        print(f"  geom_accuracy: {geom_accuracy}")

        # Determine grade based on geometric accuracy (prioritized)
        grade = "Excellent"
        grade_color = "#198754"
        if geom_accuracy is not None:
            # Use geometric accuracy as primary metric
            if geom_accuracy >= 0.95:
                grade = "Excellent"
                grade_color = "#198754"
            elif geom_accuracy >= 0.85:
                grade = "Good"
                grade_color = "#0d6efd"
            elif geom_accuracy >= 0.75:
                grade = "Fair"
                grade_color = "#ffc107"
            elif geom_accuracy >= 0.60:
                grade = "Poor"
                grade_color = "#fd7e14"
            else:
                grade = "Critical"
                grade_color = "#dc3545"
        if sicn_min is not None:
            # Fallback to SICN if no geometric accuracy
            # THRESHOLD UPDATE: < 0.2 is Critical
            if sicn_min < 0.0001:
                grade = "Critical"
                grade_color = "#dc3545"
            elif sicn_min < 0.2:
                grade = "Critical" # Was Very Poor, now Critical per new threshold
                grade_color = "#dc3545"
            elif sicn_min < 0.3:
                grade = "Poor"
                grade_color = "#fd7e14"
            elif sicn_min < 0.5:
                grade = "Fair"
                grade_color = "#ffc107"
            elif sicn_min < 0.7:
                grade = "Good"
                grade_color = "#0d6efd"

        report_html = f"""
        <div style='font-family: Arial; font-size: 11px;'>
            <div style='font-size: 13px; font-weight: bold; margin-bottom: 8px; color: {grade_color};'>
                Quality: {grade}
            </div>
        """

        # Show geometric accuracy FIRST (most important)
        if geom_accuracy is not None:
            icon = "[OK]" if geom_accuracy >= 0.85 else "[X]"
            color = "#198754" if geom_accuracy >= 0.85 else "#dc3545"
            report_html += f"<div><b>Shape Accuracy:</b> {geom_accuracy:.3f} <span style='color: {color};'>{icon}</span></div>"
            if mean_dev is not None:
                report_html += f"<div style='font-size: 10px; color: #6c757d;'>   (deviation: {mean_dev:.3f}mm)</div>"

        if sicn_min is not None:
            # THRESHOLD: < 0.2 is Bad
            icon = "[OK]" if sicn_min >= 0.2 else "[X]"
            color = "#198754" if sicn_min >= 0.2 else "#dc3545"
            report_html += f"<div><b>SICN (min):</b> {sicn_min:.4f} <span style='color: {color};'>{icon}</span></div>"

        if gamma_min is not None:
            # THRESHOLD: < 0.2 is Bad (same as SICN)
            icon = "[OK]" if gamma_min >= 0.2 else "[X]"
            color = "#198754" if gamma_min >= 0.2 else "#dc3545"
            report_html += f"<div><b>Gamma (min):</b> {gamma_min:.4f} <span style='color: {color};'>{icon}</span></div>"

        if max_skew is not None:
            # THRESHOLD: > 0.7 is Bad
            icon = "[OK]" if max_skew <= 0.7 else "[X]"
            color = "#198754" if max_skew <= 0.7 else "#dc3545"
            report_html += f"<div><b>Max Skewness:</b> {max_skew:.4f} <span style='color: {color};'>{icon}</span></div>"

        if max_ar is not None:
            # THRESHOLD: > 10.0 is Bad
            icon = "[OK]" if max_ar <= 10.0 else "[X]"
            color = "#198754" if max_ar <= 10.0 else "#dc3545"
            report_html += f"<div><b>Max Aspect Ratio:</b> {max_ar:.2f} <span style='color: {color};'>{icon}</span></div>"

        # Check for Jacobian (hex meshes)
        jacobian_min = metrics.get('jacobian_min')
        if jacobian_min is not None:
            icon = "[OK]" if jacobian_min >= 0.3 else "[X]"
            color = "#198754" if jacobian_min >= 0.3 else "#dc3545"
            report_html += f"<div><b>Jacobian (min):</b> {jacobian_min:.4f} <span style='color: {color};'>{icon}</span></div>"

        if badness is not None:
            # Badness: lower is better. <10 is good, >100 is bad
            icon = "[OK]" if badness <= 10 else "[X]"
            color = "#198754" if badness <= 10 else "#dc3545"
            report_html += f"<div><b>Badness (max):</b> {badness:.2f} <span style='color: {color};'>{icon}</span></div>"

        if worst_tet_radius is not None:
            # Tet radius ratio: closer to 1.0 is better (perfect tetrahedron)
            icon = "[OK]" if worst_tet_radius >= 0.5 else "[X]"
            color = "#198754" if worst_tet_radius >= 0.5 else "#dc3545"
            report_html += f"<div><b>Worst Tet Radius:</b> {worst_tet_radius:.4f} <span style='color: {color};'>{icon}</span></div>"

        report_html += "</div>"

        self.viewer.quality_label.setText(report_html)
        self.viewer.quality_label.adjustSize()
        self.viewer.quality_label.move(
            self.viewer.width() - self.viewer.quality_label.width() - 15,
            15
        )
        self.viewer.quality_label.setVisible(True)


    def update_quality_visualization(self, metric='SICN (Min)', opacity=1.0, min_val=None, max_val=None):
        """Update quality visualization based on metric selection"""
        # This is a placeholder - actual implementation would be in the main viewer
        pass
