#!/usr/bin/env python3
"""
Launch Khorium MeshGen GUI
===========================

Simple launcher script for the modular GUI application.
"""

import sys
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
