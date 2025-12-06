import sys
import os
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    # Just try to parse the file to check for SyntaxError
    with open('apps/desktop/gui_app/vtk_viewer.py', 'r') as f:
        compile(f.read(), 'apps/desktop/gui_app/vtk_viewer.py', 'exec')
    print("Syntax OK")
except Exception as e:
    print(f"Syntax Error: {e}")
