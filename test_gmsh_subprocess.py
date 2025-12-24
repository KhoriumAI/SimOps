import subprocess
import sys

# Test if gmsh works in subprocess
test_script = """
import gmsh
gmsh.initialize()
print("GMSH_WORKS")
gmsh.finalize()
"""

result = subprocess.run(
    [sys.executable, "-c", test_script],
    capture_output=True,
    text=True
)

print(f"Python executable: {sys.executable}")
print(f"Stdout: {result.stdout}")
print(f"Stderr: {result.stderr}")
print(f"Return code: {result.returncode}")

if "GMSH_WORKS" in result.stdout:
    print("\n[OK] Gmsh works in subprocess!")
else:
    print("\n[X] Gmsh NOT working in subprocess!")
    print("Install gmsh in your current Python environment:")
    print(f"  {sys.executable} -m pip install gmsh")
