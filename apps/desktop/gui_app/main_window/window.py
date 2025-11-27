"""
Main Window
===========

The central class for the application window.
"""

import sys
import time
from pathlib import Path
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtCore import Qt, QTimer

from ..workers import MeshWorker
from .ui_builder import UIBuilder
from .workflow import WorkflowManager
from .events import EventHandler

# Paintbrush imports
PAINTBRUSH_AVAILABLE = False
try:
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from paintbrush_widget import PaintbrushWidget
    from core.paintbrush_geometry import PaintbrushSelector
    PAINTBRUSH_AVAILABLE = True
except ImportError:
    pass

class ModernMeshGenGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # State
        self.cad_file = None
        self.mesh_file = None
        self.worker = MeshWorker()
        self.phase_bars = {}
        self.phase_labels = {}
        self.phase_base_names = {}
        self.active_phase = None
        self.dot_count = 0
        
        self.phase_weights = {}
        self.completed_phases = []
        self.phase_completion_times = {}
        self.mesh_start_time = None
        self.master_progress = 0.0

        self.chatbox = None
        self.chatbox_visible = False
        
        self.paintbrush_selector = None
        if PAINTBRUSH_AVAILABLE:
            self.paintbrush_selector = PaintbrushSelector()

        self.quality_data_ranges = {
            'SICN': (0.0, 1.0),
            'Gamma': (0.0, 1.0),
            'Skewness': (0.0, 1.0),
            'Aspect Ratio': (1.0, 10.0)
        }

        # Helpers
        self.ui_builder = UIBuilder(self)
        self.workflow = WorkflowManager(self)
        self.events = EventHandler(self)

        # Signals
        self.worker.signals.log.connect(self.add_log)
        self.worker.signals.progress.connect(self.workflow.update_progress)
        self.worker.signals.phase_complete.connect(self.workflow.mark_phase_complete)
        self.worker.signals.finished.connect(self.workflow.on_mesh_finished)

        # Animation Timer
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_timer.setInterval(400)

        # Initialize UI
        self.ui_builder.init_ui()
        
        # Initialize Chatbox (Optional)
        try:
            from ..ui.chatbox_widget import ChatboxWidget
            self.chatbox = ChatboxWidget()
            self.chatbox.setVisible(False)
            self.chatbox.close_requested.connect(self.events.toggle_chatbox)
            self.left_container.layout().addWidget(self.chatbox)
        except ImportError:
            print("Chatbox not available")

    # Delegate to Helpers
    def load_cad_file(self): self.events.load_cad_file()
    def toggle_chatbox(self): self.events.toggle_chatbox()
    def start_mesh_generation(self): self.workflow.start_mesh_generation()
    def stop_mesh_generation(self): self.workflow.stop_mesh_generation()
    def refine_mesh_quality(self): self.workflow.refine_mesh_quality()
    def on_quality_preset_changed(self, p): self.events.on_quality_preset_changed(p)
    def on_crosssection_toggled(self, s): self.events.on_crosssection_toggled(s)
    def on_clip_axis_changed(self, t): self.events.on_clip_axis_changed(t)
    def on_clip_offset_changed(self, v): self.events.on_clip_offset_changed(v)
    def on_crosssection_element_mode_changed(self, t): self.events.on_crosssection_element_mode_changed(t)
    def on_ghost_visibility_toggled(self, s): self.events.on_ghost_visibility_toggled(s)
    def on_viz_range_slider_changed(self, v): self.events.on_viz_range_slider_changed(v)
    def on_viz_metric_changed(self, t): self.events.on_viz_metric_changed(t)
    def on_viz_opacity_changed(self, v): self.events.on_viz_opacity_changed(v)
    def copy_console_to_clipboard(self): self.events.copy_console_to_clipboard()
    def toggle_hex_visualization(self): self.events.toggle_hex_visualization()

    def add_log(self, message):
        self.console.append(message)
        # Auto-scroll
        sb = self.console.verticalScrollBar()
        sb.setValue(sb.maximum())

    def update_animation(self):
        if not self.active_phase: return
        self.dot_count = (self.dot_count + 1) % 4
        dots = "." * self.dot_count
        if hasattr(self, 'current_process_label'):
            base = self.current_process_label.text().split(':')[0]
            self.current_process_label.setText(f"{base}: Running{dots}")

    def reset_progress_bars(self):
        for bar in self.phase_bars.values():
            bar.setValue(0)
            bar.setStyleSheet(bar.styleSheet().replace("background-color: #198754", "background-color: #0d6efd"))
            
    def calculate_phase_weights(self, element_target, quality_preset):
        base_weights = {'cad': 5, 'surf': 20, 'refine': 15, '3d': 35, 'opt': 20, 'quality': 5}
        complexity = 0.7 if element_target < 5000 else 1.5 if element_target > 50000 else 1.0
        quality_mult = {'Draft': 0.6, 'Standard': 1.0, 'High Fidelity': 1.8}.get(quality_preset, 1.0)
        
        adjusted = base_weights.copy()
        adjusted['opt'] *= quality_mult
        adjusted['refine'] *= quality_mult
        adjusted['3d'] *= complexity
        
        total = sum(adjusted.values())
        return {k: (v/total)*100 for k, v in adjusted.items()}

    def calculate_suggested_element_counts(self, geom_info):
        volume = geom_info.get('volume', 0.001)
        
        def calc_sizes(target):
            avg_size_mm = ((volume / target) ** (1/3)) * 1000
            return int(target), int(round(max(0.1, avg_size_mm * 0.7))), int(round(max(0.5, avg_size_mm * 1.3)))

        presets = {
            "Coarse": {"target": calc_sizes(max(500, min(2000, int(volume * 1e6))))[0], "max": calc_sizes(max(500, min(2000, int(volume * 1e6))))[2]},
            "Medium": {"target": calc_sizes(max(2000, min(8000, int(volume * 4e6))))[0], "max": calc_sizes(max(2000, min(8000, int(volume * 4e6))))[2]},
            "Fine": {"target": calc_sizes(max(8000, min(30000, int(volume * 16e6))))[0], "max": calc_sizes(max(8000, min(30000, int(volume * 16e6))))[2]},
            "Very Fine": {"target": calc_sizes(max(30000, min(100000, int(volume * 64e6))))[0], "max": calc_sizes(max(30000, min(100000, int(volume * 64e6))))[2]}
        }
        self.calculated_presets = presets
        
        current = self.quality_preset.currentText()
        if current in presets:
            self.target_elements.setValue(presets[current]["target"])
            self.max_size.setValue(presets[current]["max"])
            
    def update_eta(self, current_progress):
        if not self.mesh_start_time or current_progress < 5: return
        elapsed = time.time() - self.mesh_start_time
        estimated_total = elapsed / (current_progress / 100)
        remaining = estimated_total - elapsed
        
        eta_text = f"{int(remaining)}s" if remaining < 60 else f"{int(remaining//60)}m {int(remaining%60)}s"
        if hasattr(self, 'master_bar'):
            self.master_bar.setFormat(f"{int(current_progress)}% - ETA: {eta_text}")

    def update_quality_data_ranges(self, q, g, s, a):
        if q: self.quality_data_ranges['SICN'] = (min(q.values()), max(q.values()))
        if g: self.quality_data_ranges['Gamma'] = (min(g.values()), max(g.values()))
        if s: self.quality_data_ranges['Skewness'] = (min(s.values()), max(s.values()))
        if a: self.quality_data_ranges['Aspect Ratio'] = (min(a.values()), min(max(a.values()), 20.0))
        
        metric = self.viz_metric_combo.currentText()
        d_min, d_max = self.quality_data_ranges.get(metric, (0.0, 1.0))
        self.viz_range_min_label.setText(f"Min: {d_min:.2f}")
        self.viz_range_max_label.setText(f"Max: {d_max:.2f}")
