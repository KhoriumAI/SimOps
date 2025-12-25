#!/usr/bin/env python3
"""
Launch Khorium MeshGen GUI
===========================

Simple launcher script for the modular GUI application.
Includes splash screen, enforces light mode on macOS, and
pre-warms the mesh engine for instant mesh generation.
"""

import sys
import os
import time
from pathlib import Path

# Add project root to path FIRST (before any other imports)
project_root = str(Path(__file__).parent.parent.parent.resolve())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import Qt basics first (lightweight)
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

# Import splash screen and palette helper
from splash_screen import SplashScreen, create_light_palette


def main():
    """Launch the GUI application with splash screen."""
    # Enable High DPI scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # Create application
    app = QApplication(sys.argv)
    
    # CRITICAL: Set Fusion style and light palette BEFORE any widgets
    # This fixes the dark mode issue on macOS
    app.setStyle("Fusion")
    app.setPalette(create_light_palette())
    
    # Create and show splash screen
    print("[DEBUG] Creating splash screen...")
    splash = SplashScreen()
    splash.show()
    splash.force_focus()  # Bring to front on macOS
    splash.set_status("Starting up...")
    splash.set_progress(10)
    app.processEvents()
    time.sleep(0.3)  # Ensure splash is visible
    print("[DEBUG] Splash screen created and shown.")
    
    # Heavy imports with progress updates
    splash.set_status("Loading PyQt5 widgets...")
    splash.set_progress(20)
    app.processEvents()
    time.sleep(0.1)
    
    splash.set_status("Loading VTK viewer...")
    splash.set_progress(40)
    app.processEvents()
    time.sleep(0.1)
    
    # Import the main GUI (this triggers heavy VTK imports)
    from gui_app import ModernMeshGenGUI
    
    splash.set_status("Pre-warming mesh engine...")
    splash.set_progress(60)
    app.processEvents()
    
    # Start mesh engine warmup in background
    # This eliminates the 3-5s delay when clicking "Generate Mesh"
    try:
        from core.mesh_worker_pool import warmup_mesh_engine
        
        def warmup_callback(msg):
            splash.set_status(msg)
            app.processEvents()
        
        warmup_mesh_engine(warmup_callback)
        print("[DEBUG] Mesh engine warmup started in background")
    except ImportError as e:
        print(f"[DEBUG] Mesh worker pool not available: {e}")
    except Exception as e:
        print(f"[DEBUG] Mesh warmup error (non-fatal): {e}")
    
    splash.set_status("Initializing mesh engine...")
    splash.set_progress(70)
    app.processEvents()
    time.sleep(0.2)
    
    # Create main window
    splash.set_status("Creating main window...")
    splash.set_progress(85)
    app.processEvents()
    time.sleep(0.1)
    
    gui = ModernMeshGenGUI()
    
    # Finalize and show
    splash.set_status("Ready!")
    splash.set_progress(100)
    app.processEvents()
    time.sleep(0.5)
    
    # Close splash and show main window
    splash.finish(gui)
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

