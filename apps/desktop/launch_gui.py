#!/usr/bin/env python3
"""
Launch Khorium MeshGen GUI
===========================

Simple launcher script for the modular GUI application.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent.resolve())
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from PyQt5.QtWidgets import QApplication
from gui_app import ModernMeshGenGUI


def main():
    """Launch the GUI application"""
    app = QApplication(sys.argv)
    
    # Create and show main window
    gui = ModernMeshGenGUI()
    gui.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
