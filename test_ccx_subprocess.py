import subprocess
from pathlib import Path
import os

binary = r"C:\calculix\calculix_2.22_4win\ccx.exe"
print(f"Testing binary: {binary}")

try:
    # Try running without args (should hang or print help)
    # CalculiX usually expects an input file.
    # We can try just checking if it launches.
    
    # On Windows, just running it might return immediately if no input
    process = subprocess.Popen([binary], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = process.communicate(timeout=2)
        print(f"Return code: {process.returncode}")
        print(f"Stdout: {stdout}")
        print(f"Stderr: {stderr}")
    except subprocess.TimeoutExpired:
        process.kill()
        print("Process timed out (it launched!)")

except Exception as e:
    print(f"Execution failed: {e}")

print("\nTesting 'ccx' in path:")
try:
    subprocess.run(["ccx"], capture_output=True)
    print("ccx in PATH found (no exception)")
except FileNotFoundError:
    print("ccx NOT in PATH")
except Exception as e:
    print(f"ccx error: {e}")
