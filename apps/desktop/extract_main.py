# Extract ModernMeshGenGUI class from gui_final.py
with open('gui_final.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find class boundaries (line 2451 to end of file)
start_idx = 2450  # Line 2451 (0-indexed)
end_idx = len(lines)  # Until end

# Extract the class
class_lines = lines[start_idx:end_idx]

# Create header with imports
header = '''"""
Main GUI Window
================

ModernMeshGenGUI - Main application window with mesh generation controls.
"""

import sys
import os
import json
from pathlib import Path
from typing import Optional, Dict

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QProgressBar, QGroupBox,
    QSplitter, QFileDialog, QFrame, QScrollArea, QGridLayout,
    QCheckBox, QSizePolicy, QSlider, QSpinBox, QComboBox, QDoubleSpinBox
)
from qtrangeslider import QRangeSlider
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor

from .workers import MeshWorker
from .vtk_viewer import VTK3DViewer
from .utils import setup_logging

# Paintbrush imports
PAINTBRUSH_AVAILABLE = False
try:
    # Add project root to path
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from paintbrush_widget import PaintbrushWidget
    from core.paintbrush_geometry import PaintbrushSelector
    from strategies.paintbrush_strategy import PaintbrushStrategy
    PAINTBRUSH_AVAILABLE = True
    print("[OK] Paintbrush feature loaded successfully")
except ImportError as e:
    print(f"[!] Paintbrush feature not available: {e}")


'''

# Write to main.py
with open('gui_app/main.py', 'w', encoding='utf-8') as f:
    f.write(header)
    f.writelines(class_lines)

print('ModernMeshGenGUI extracted successfully!')
print(f'Wrote {len(class_lines)} lines to gui_app/main.py')
