#!/usr/bin/env python3
"""Test GUI CAD loading to find crash"""
import sys
from pathlib import Path

# Set up paths
sys.path.insert(0, str(Path(__file__).parent))

# Test imports
try:
    from PyQt5.QtWidgets import QApplication
    print("[OK] PyQt5 imported")
except Exception as e:
    print(f"[X] PyQt5 import failed: {e}")
    sys.exit(1)

try:
    from gui_final import ModernMeshGenGUI
    print("[OK] GUI class imported")
except Exception as e:
    print(f"[X] GUI import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Create app
try:
    app = QApplication(sys.argv)
    print("[OK] QApplication created")
except Exception as e:
    print(f"[X] QApplication failed: {e}")
    sys.exit(1)

# Create GUI
try:
    window = ModernMeshGenGUI()
    print("[OK] GUI window created")
except Exception as e:
    print(f"[X] GUI window creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Try to load a CAD file programmatically
try:
    test_file = "CAD_files/Cube.step"
    if Path(test_file).exists():
        print(f"\n Testing load_cad_file with {test_file}...")
        window.cad_file = test_file
        window.file_label.setText(f"[OK] {Path(test_file).name}")
        print("[OK] Set file path")

        # This is where the crash likely happens
        geom_info = window.viewer.load_step_file(test_file)
        print(f"[OK] Viewer loaded file: {geom_info}")

        if geom_info and 'volume' in geom_info:
            window.calculate_suggested_element_counts(geom_info)
            print("[OK] Calculated element counts")

        print("\n[OK][OK][OK] SUCCESS - No crash!")
    else:
        print(f"[X] Test file not found: {test_file}")
except Exception as e:
    print(f"\n[X][X][X] CRASH FOUND: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nTest completed successfully!")
