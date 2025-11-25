# Extract VTK3DViewer class from gui_final.py
with open('gui_final.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find class boundaries (lines 445-2448, but Python is 0-indexed so 444-2447)
start_idx = 444  # Line 445
end_idx = 2448    # Line 2449 (exclusive)

# Extract the class
class_lines = lines[start_idx:end_idx]

# Create header with imports
header = '''"""
VTK 3D Viewer
==============

3D mesh visualization with quality overlay, cross-sections, and paintbrush support.
"""

import vtk
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import QFrame, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from .interactor import CustomInteractorStyle
from .utils import hsl_to_rgb


'''

# Write to vtk_viewer.py
with open('gui_app/vtk_viewer.py', 'w', encoding='utf-8') as f:
    f.write(header)
    f.writelines(class_lines)

print('VTK3DViewer extracted successfully!')
print(f'Wrote {len(class_lines)} lines to gui_app/vtk_viewer.py')
