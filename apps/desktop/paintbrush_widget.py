"""
Paintbrush UI Widget
====================

PyQt5 widget for paintbrush mesh refinement controls.

Provides user interface for:
- Enabling/disabling paintbrush mode
- Adjusting brush radius
- Setting refinement level
- Managing painted regions
- Previewing refinement

Usage:
    from paintbrush_widget import PaintbrushWidget

    widget = PaintbrushWidget()
    widget.paintbrush_enabled.connect(on_paintbrush_enabled)
    widget.radius_changed.connect(on_radius_changed)
    widget.refinement_changed.connect(on_refinement_changed)
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QGroupBox, QListWidget, QListWidgetItem, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from typing import List, Optional


class PaintbrushWidget(QWidget):
    """
    Widget for controlling paintbrush mesh refinement.

    Signals:
        paintbrush_enabled: Emitted when paintbrush mode is toggled (bool)
        radius_changed: Emitted when brush radius changes (float)
        refinement_changed: Emitted when refinement level changes (float)
        clear_requested: Emitted when user clicks "Clear All"
        preview_requested: Emitted when user clicks "Preview Refinement"
        region_selected: Emitted when user selects a region in list (int index)
        region_deleted: Emitted when user deletes a region (int index)
    """

    paintbrush_enabled = pyqtSignal(bool)
    radius_changed = pyqtSignal(float)
    refinement_changed = pyqtSignal(float)
    clear_requested = pyqtSignal()
    preview_requested = pyqtSignal()
    region_selected = pyqtSignal(int)
    region_deleted = pyqtSignal(int)

    def __init__(self, parent=None):
        """Initialize paintbrush widget"""
        super().__init__(parent)

        self.is_enabled = False
        self.current_radius = 5.0
        self.current_refinement = 3.0

        self.setup_ui()

    def setup_ui(self):
        """Setup user interface"""
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        # Title
        title = QLabel("Paintbrush Refinement")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        title.setStyleSheet("color: #2c3e50; padding: 5px;")
        layout.addWidget(title)

        # Enable/Disable toggle
        self.enable_btn = QPushButton("Enable Paintbrush")
        self.enable_btn.setCheckable(True)
        self.enable_btn.setStyleSheet(self._get_enable_button_style())
        self.enable_btn.clicked.connect(self._on_toggle_paintbrush)
        layout.addWidget(self.enable_btn)

        # Brush settings group
        settings_group = QGroupBox("Brush Settings")
        settings_layout = QVBoxLayout()

        # Brush radius
        radius_layout = QVBoxLayout()
        radius_label_layout = QHBoxLayout()
        radius_label = QLabel("Brush Radius:")
        self.radius_value_label = QLabel(f"{self.current_radius:.1f} mm")
        self.radius_value_label.setStyleSheet("font-weight: bold; color: #3498db;")
        radius_label_layout.addWidget(radius_label)
        radius_label_layout.addStretch()
        radius_label_layout.addWidget(self.radius_value_label)
        radius_layout.addLayout(radius_label_layout)

        self.radius_slider = QSlider(Qt.Orientation.Horizontal)
        self.radius_slider.setMinimum(1)
        self.radius_slider.setMaximum(50)
        self.radius_slider.setValue(int(self.current_radius))
        self.radius_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.radius_slider.setTickInterval(5)
        self.radius_slider.valueChanged.connect(self._on_radius_changed)
        self.radius_slider.setEnabled(False)
        radius_layout.addWidget(self.radius_slider)

        settings_layout.addLayout(radius_layout)

        # Refinement level
        refinement_layout = QVBoxLayout()
        refinement_label_layout = QHBoxLayout()
        refinement_label = QLabel("Refinement Level:")
        self.refinement_value_label = QLabel(f"{self.current_refinement:.1f}x")
        self.refinement_value_label.setStyleSheet("font-weight: bold; color: #e74c3c;")
        refinement_label_layout.addWidget(refinement_label)
        refinement_label_layout.addStretch()
        refinement_label_layout.addWidget(self.refinement_value_label)
        refinement_layout.addLayout(refinement_label_layout)

        self.refinement_slider = QSlider(Qt.Orientation.Horizontal)
        self.refinement_slider.setMinimum(10)  # 1.0x (10 / 10)
        self.refinement_slider.setMaximum(100)  # 10.0x (100 / 10)
        self.refinement_slider.setValue(int(self.current_refinement * 10))
        self.refinement_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.refinement_slider.setTickInterval(10)
        self.refinement_slider.valueChanged.connect(self._on_refinement_changed)
        self.refinement_slider.setEnabled(False)
        refinement_layout.addWidget(self.refinement_slider)

        # Refinement description
        refinement_desc = QLabel("Higher = finer mesh in painted regions")
        refinement_desc.setStyleSheet("font-size: 9px; color: #7f8c8d; font-style: italic;")
        refinement_layout.addWidget(refinement_desc)

        settings_layout.addLayout(refinement_layout)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # Painted regions list
        regions_group = QGroupBox("Painted Regions")
        regions_layout = QVBoxLayout()

        self.regions_list = QListWidget()
        self.regions_list.setMaximumHeight(120)
        self.regions_list.itemClicked.connect(self._on_region_selected)
        regions_layout.addWidget(self.regions_list)

        # Region list buttons
        region_buttons = QHBoxLayout()

        self.delete_region_btn = QPushButton("Delete")
        self.delete_region_btn.clicked.connect(self._on_delete_region)
        self.delete_region_btn.setEnabled(False)
        self.delete_region_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 4px 8px;
                border-radius: 3px;
            }
            QPushButton:hover { background-color: #c0392b; }
            QPushButton:disabled { background-color: #bdc3c7; }
        """)
        region_buttons.addWidget(self.delete_region_btn)

        self.clear_all_btn = QPushButton("Clear All")
        self.clear_all_btn.clicked.connect(self._on_clear_all)
        self.clear_all_btn.setEnabled(False)
        self.clear_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                color: white;
                border: none;
                padding: 4px 8px;
                border-radius: 3px;
            }
            QPushButton:hover { background-color: #7f8c8d; }
            QPushButton:disabled { background-color: #bdc3c7; }
        """)
        region_buttons.addWidget(self.clear_all_btn)

        regions_layout.addLayout(region_buttons)

        regions_group.setLayout(regions_layout)
        layout.addWidget(regions_group)

        # Preview button
        self.preview_btn = QPushButton("Preview Refinement")
        self.preview_btn.clicked.connect(self._on_preview)
        self.preview_btn.setEnabled(False)
        self.preview_btn.setStyleSheet("""
            QPushButton {
                background-color: #9b59b6;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #8e44ad; }
            QPushButton:disabled { background-color: #bdc3c7; }
        """)
        layout.addWidget(self.preview_btn)

        # Info label
        self.info_label = QLabel("Enable paintbrush to refine mesh in specific areas")
        self.info_label.setStyleSheet("""
            QLabel {
                background-color: #ecf0f1;
                padding: 8px;
                border-radius: 4px;
                color: #2c3e50;
                font-size: 10px;
            }
        """)
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        # Statistics
        self.stats_label = QLabel("No regions painted")
        self.stats_label.setStyleSheet("font-size: 9px; color: #7f8c8d; padding: 5px;")
        layout.addWidget(self.stats_label)

        layout.addStretch()
        self.setLayout(layout)

    def _get_enable_button_style(self) -> str:
        """Get stylesheet for enable button"""
        return """
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
            QPushButton:checked {
                background-color: #e74c3c;
            }
            QPushButton:checked:hover {
                background-color: #c0392b;
            }
        """

    def _on_toggle_paintbrush(self, checked: bool):
        """Handle paintbrush enable/disable"""
        self.is_enabled = checked

        if checked:
            self.enable_btn.setText("Disable Paintbrush")
            self.info_label.setText("PAINT MODE: Left click to paint | Right click to rotate")
            self.info_label.setStyleSheet("""
                QLabel {
                    background-color: #fff9e6;
                    padding: 8px;
                    border-radius: 4px;
                    color: #e67e22;
                    font-size: 10px;
                    font-weight: bold;
                }
            """)
        else:
            self.enable_btn.setText("Enable Paintbrush")
            self.info_label.setText("Enable paintbrush to refine mesh in specific areas")
            self.info_label.setStyleSheet("""
                QLabel {
                    background-color: #ecf0f1;
                    padding: 8px;
                    border-radius: 4px;
                    color: #2c3e50;
                    font-size: 10px;
                }
            """)

        self.radius_slider.setEnabled(checked)
        self.refinement_slider.setEnabled(checked)

        self.paintbrush_enabled.emit(checked)

    def _on_radius_changed(self, value: int):
        """Handle brush radius slider change"""
        self.current_radius = float(value)
        self.radius_value_label.setText(f"{self.current_radius:.1f} mm")
        self.radius_changed.emit(self.current_radius)

    def _on_refinement_changed(self, value: int):
        """Handle refinement level slider change"""
        self.current_refinement = value / 10.0
        self.refinement_value_label.setText(f"{self.current_refinement:.1f}x")

        # Update description based on level
        if self.current_refinement < 2.0:
            desc = "Slight refinement"
        elif self.current_refinement < 4.0:
            desc = "Moderate refinement"
        elif self.current_refinement < 7.0:
            desc = "Strong refinement"
        else:
            desc = "Very strong refinement (slow!)"

        # Find and update description label
        for i in range(self.layout().count()):
            item = self.layout().itemAt(i)
            if item and item.widget():
                if isinstance(item.widget(), QGroupBox):
                    group = item.widget()
                    if group.title() == "Brush Settings":
                        for j in range(group.layout().count()):
                            sub_item = group.layout().itemAt(j)
                            if sub_item and isinstance(sub_item, QVBoxLayout):
                                # This is the refinement layout
                                pass

        self.refinement_changed.emit(self.current_refinement)

    def _on_region_selected(self, item: QListWidgetItem):
        """Handle region selection in list"""
        index = self.regions_list.row(item)
        self.delete_region_btn.setEnabled(True)
        self.region_selected.emit(index)

    def _on_delete_region(self):
        """Handle delete region button"""
        current_row = self.regions_list.currentRow()
        if current_row >= 0:
            self.region_deleted.emit(current_row)
            self.regions_list.takeItem(current_row)
            self.delete_region_btn.setEnabled(False)
            self.update_button_states()

    def _on_clear_all(self):
        """Handle clear all button"""
        self.clear_requested.emit()
        self.regions_list.clear()
        self.delete_region_btn.setEnabled(False)
        self.update_button_states()

    def _on_preview(self):
        """Handle preview button"""
        self.preview_requested.emit()

    def add_region_to_list(self, region_info: str):
        """
        Add a painted region to the list.

        Args:
            region_info: Description string for the region
        """
        self.regions_list.addItem(region_info)
        self.update_button_states()

    def update_button_states(self):
        """Update enabled state of buttons based on region count"""
        has_regions = self.regions_list.count() > 0
        self.clear_all_btn.setEnabled(has_regions)
        self.preview_btn.setEnabled(has_regions)

    def update_statistics(self, num_regions: int, num_surfaces: int, avg_refinement: float):
        """
        Update statistics display.

        Args:
            num_regions: Number of painted regions
            num_surfaces: Total number of surfaces painted
            avg_refinement: Average refinement level
        """
        if num_regions == 0:
            self.stats_label.setText("No regions painted")
        else:
            self.stats_label.setText(
                f"{num_regions} regions, {num_surfaces} surfaces, "
                f"avg {avg_refinement:.1f}x refinement"
            )

    def get_current_radius(self) -> float:
        """Get current brush radius"""
        return self.current_radius

    def get_current_refinement(self) -> float:
        """Get current refinement level"""
        return self.current_refinement

    def is_paintbrush_enabled(self) -> bool:
        """Check if paintbrush mode is enabled"""
        return self.is_enabled

    def set_enabled_state(self, enabled: bool):
        """
        Programmatically set paintbrush enabled state.

        Args:
            enabled: Whether paintbrush should be enabled
        """
        self.enable_btn.setChecked(enabled)
        self._on_toggle_paintbrush(enabled)


if __name__ == "__main__":
    # Test the widget
    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)

    widget = PaintbrushWidget()
    widget.setWindowTitle("Paintbrush Controls")
    widget.resize(300, 600)
    widget.show()

    # Connect test signals
    widget.paintbrush_enabled.connect(lambda x: print(f"Paintbrush enabled: {x}"))
    widget.radius_changed.connect(lambda x: print(f"Radius: {x}"))
    widget.refinement_changed.connect(lambda x: print(f"Refinement: {x}"))
    widget.clear_requested.connect(lambda: print("Clear all requested"))

    # Add test regions
    widget.add_region_to_list("Region 1: 5 surfaces, 3.0x")
    widget.add_region_to_list("Region 2: 12 surfaces, 5.0x")
    widget.update_statistics(2, 17, 4.0)

    sys.exit(app.exec_())
