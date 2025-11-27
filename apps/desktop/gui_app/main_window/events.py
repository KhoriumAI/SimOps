"""
Event Handler
=============

Handles UI events and interactions.
"""

from PyQt5.QtWidgets import QFileDialog, QApplication
from PyQt5.QtCore import Qt
from pathlib import Path

class EventHandler:
    def __init__(self, window):
        self.window = window

    def load_cad_file(self):
        if self.window.worker and self.window.worker.is_running:
            self.window.add_log("Stopping previous mesh generation...")
            self.window.worker.stop()

        self.window.reset_progress_bars()
        self.window.viewer.clear_view()

        default_dir = str(Path.home())
        filepath, _ = QFileDialog.getOpenFileName(
            self.window, "Select CAD or Mesh File", default_dir,
            "CAD Files (*.step *.stp *.stl);;Mesh Files (*.msh);;All Files (*)"
        )

        if filepath:
            file_ext = Path(filepath).suffix.lower()

            if file_ext == '.msh':
                self.window.cad_file = None
                self.window.file_label.setText(f"{Path(filepath).name}")
                self.window.generate_btn.setEnabled(False)
                self.window.add_log(f"Loaded mesh: {filepath}")
                self.window.viewer.load_mesh_file(filepath)
                return

            self.window.cad_file = filepath
            self.window.file_label.setText(f"{Path(filepath).name}")
            self.window.generate_btn.setEnabled(True)
            self.window.add_log(f"Loaded CAD: {filepath}")

            geom_info = self.window.viewer.load_step_file(filepath)

            if geom_info and 'volume' in geom_info:
                self.window.calculate_suggested_element_counts(geom_info)

    def toggle_chatbox(self):
        if not self.window.chatbox:
            return

        if self.window.chatbox_visible:
            self.window.chatbox.setVisible(False)
            self.window.left_panel.setVisible(True)
            self.window.chatbox_visible = False
            self.window.chat_toggle_btn.setChecked(False)
            self.window.chat_toggle_btn.setText("💬 AI Chat")
        else:
            self.window.left_panel.setVisible(False)
            self.window.chatbox.setVisible(True)
            self.window.chatbox_visible = True
            self.window.chat_toggle_btn.setChecked(True)
            self.window.chat_toggle_btn.setText("🔙 Settings")

    def on_quality_preset_changed(self, preset: str):
        if hasattr(self.window, 'calculated_presets') and preset in self.window.calculated_presets:
            presets = self.window.calculated_presets
        else:
            presets = {
                "Coarse": {"target": 5000, "max": 200},
                "Medium": {"target": 10000, "max": 100},
                "Fine": {"target": 50000, "max": 50},
                "Very Fine": {"target": 200000, "max": 20}
            }
        presets["Custom"] = None

        if preset in presets and presets[preset]:
            values = presets[preset]
            self.window.target_elements.setValue(values["target"])
            self.window.max_size.setValue(values["max"])

    def on_crosssection_toggled(self, state):
        enabled = bool(state)
        self.window.clip_axis_combo.setEnabled(enabled)
        self.window.clip_offset_slider.setEnabled(enabled)
        if hasattr(self.window, 'crosssection_cell_combo'):
            self.window.crosssection_cell_combo.setEnabled(enabled)
        
        self.window.viewer.set_clipping(
            enabled=enabled,
            axis=self.window.clip_axis_combo.currentText(),
            offset=self.window.clip_offset_slider.value()
        )

    def on_clip_axis_changed(self, text):
        self.window.viewer.set_clipping(
            enabled=self.window.crosssection_enabled.isChecked(),
            axis=text,
            offset=self.window.clip_offset_slider.value()
        )

    def on_clip_offset_changed(self, value):
        self.window.clip_offset_value_label.setText(f"{value}%")
        self.window.viewer.set_clipping(
            enabled=self.window.crosssection_enabled.isChecked(),
            axis=self.window.clip_axis_combo.currentText(),
            offset=value
        )
    
    def on_crosssection_element_mode_changed(self, text: str):
        mode_map = {"Auto": "auto", "Tetrahedra": "tetrahedra", "Hexahedra": "hexahedra"}
        mode = mode_map.get(text, "auto")
        self.window.viewer.set_cross_section_element_mode(mode)

    def on_ghost_visibility_toggled(self, state):
        if hasattr(self.window.viewer, 'above_cut_actor') and self.window.viewer.above_cut_actor:
            if state:
                self.window.viewer.above_cut_actor.VisibilityOn()
            else:
                self.window.viewer.above_cut_actor.VisibilityOff()
            self.window.viewer.vtk_widget.GetRenderWindow().Render()

    def on_viz_range_slider_changed(self, value):
        min_slider, max_slider = value
        metric = self.window.viz_metric_combo.currentText()
        data_min, data_max = self.window.quality_data_ranges.get(metric, (0.0, 1.0))
        
        range_span = data_max - data_min
        min_val = data_min + (min_slider / 100.0) * range_span
        max_val = data_min + (max_slider / 100.0) * range_span
        
        self.window.viz_range_min_label.setText(f"Min: {min_val:.2f}")
        self.window.viz_range_max_label.setText(f"Max: {max_val:.2f}")
        
        self.window.viewer.update_quality_visualization(
            metric=metric,
            opacity=self.window.viz_opacity_spin.value(),
            min_val=min_val,
            max_val=max_val
        )

    def on_viz_metric_changed(self, text):
        data_min, data_max = self.window.quality_data_ranges.get(text, (0.0, 1.0))
        self.window.viz_range_slider.setValue((0, 100))
        self.window.viz_range_min_label.setText(f"Min: {data_min:.2f}")
        self.window.viz_range_max_label.setText(f"Max: {data_max:.2f}")
        
        self.window.viewer.update_quality_visualization(
            metric=text,
            opacity=self.window.viz_opacity_spin.value(),
            min_val=data_min,
            max_val=data_max
        )

    def on_viz_opacity_changed(self, value):
        min_slider, max_slider = self.window.viz_range_slider.value()
        metric = self.window.viz_metric_combo.currentText()
        data_min, data_max = self.window.quality_data_ranges.get(metric, (0.0, 1.0))
        range_span = data_max - data_min
        min_val = data_min + (min_slider / 100.0) * range_span
        max_val = data_min + (max_slider / 100.0) * range_span
        
        self.window.viewer.update_quality_visualization(
            metric=metric,
            opacity=value,
            min_val=min_val,
            max_val=max_val
        )

    def copy_console_to_clipboard(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.window.console.toPlainText())
        self.window.add_log("[INFO] Console output copied to clipboard")

    def toggle_hex_visualization(self):
        # Placeholder for secret hex mode
        pass
