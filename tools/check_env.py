
import sys

print("Python Version:", sys.version)

try:
    import PIL
    import PIL.Image
    print(f"PIL Version: {PIL.__version__}")
    print(f"PIL Path: {PIL.__file__}")
except ImportError as e:
    print("PIL (Pillow) NOT FOUND:", e)

try:
    import matplotlib
    print(f"Matplotlib Version: {matplotlib.__version__}")
except ImportError as e:
    print("Matplotlib NOT FOUND:", e)

try:
    import numpy
    print(f"Numpy Version: {numpy.__version__}")
except ImportError as e:
    print("Numpy NOT FOUND:", e)

import subprocess
try:
    result = subprocess.run(["ccx", "-v"], capture_output=True, text=True)
    print(f"CalculiX (ccx) Version: {result.stdout.splitlines()[0] if result.stdout else 'Unknown'}")
except Exception as e:
    print("CalculiX (ccx) NOT FOUND or failed to run:", e)
