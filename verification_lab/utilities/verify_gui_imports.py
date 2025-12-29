import sys
import os
from pathlib import Path

# Mimic launch_gui.py path logic (as if running from apps/desktop)
# We are running from root, so we need to simulate the environment
# But simpler: just check if we can import everything from root.
# If launch_gui.py adds root to path, it effectively runs as if from root.

print(f"CWD: {os.getcwd()}")
sys.path.insert(0, os.getcwd())

try:
    import core
    print("[OK] Imported core")
except ImportError as e:
    print(f"[FAIL] Failed to import core: {e}")

try:
    # Try importing the GUI app module
    # It is in apps.desktop.gui_app
    from apps.desktop.gui_app.main import ModernMeshGenGUI
    print("[OK] Imported ModernMeshGenGUI")
except ImportError as e:
    print(f"[FAIL] Failed to import ModernMeshGenGUI: {e}")

try:
    # Test internal imports of ModernMeshGenGUI
    from core.paintbrush_geometry import PaintbrushSelector
    print("[OK] Imported PaintbrushSelector")
except ImportError as e:
    print(f"[FAIL] Failed to import PaintbrushSelector: {e}")
