
import gmsh
import json
import os
import sys
import tempfile
import subprocess
from pathlib import Path

def test_file_loading_logic(filepath):
    print(f"Testing logic for file: {filepath}")
    
    # Mirroring the logic inside vtk_viewer.py's load_step_file
    # We want to ensure the generated script runs without error
    
    tmp = tempfile.NamedTemporaryFile(suffix='.stl', delete=False)
    tmp_stl = tmp.name
    tmp.close()

    script = f"""
import gmsh
import json
import os
import sys

try:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.Verbosity", 0)

    input_file = r"{filepath}"
    gmsh.open(input_file)

    # Detect file type
    ext = os.path.splitext(input_file)[1].lower()
    is_mesh = ext in ['.stl', '.obj']
    print(f"File extension: {{ext}}, is_mesh: {{is_mesh}}")

    bbox = gmsh.model.getBoundingBox(-1, -1)
    bbox_dims = [bbox[3]-bbox[0], bbox[4]-bbox[1], bbox[5]-bbox[2]]
    bbox_diag = (bbox_dims[0]**2 + bbox_dims[1]**2 + bbox_dims[2]**2)**0.5
    
    if not is_mesh:
        print("Running tessellation...")
        gmsh.option.setNumber("Mesh.MeshSizeMin", bbox_diag / 100.0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", bbox_diag / 20.0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.model.mesh.generate(2)
    else:
        print("Skipping tessellation (already a mesh)")

    gmsh.write(r"{tmp_stl}")
    gmsh.finalize()
    print("SUCCESS_MARKER")

except Exception as e:
    print("GMSH_ERROR:" + str(e))
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
    
    # Run the script via subprocess
    current_env = os.environ.copy()
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        env=current_env
    )
    
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    if result.returncode == 0 and "SUCCESS_MARKER" in result.stdout:
        print(f"[PASS] Successfully processed {filepath}")
        if os.path.exists(tmp_stl) and os.path.getsize(tmp_stl) > 100:
             print("[PASS] Output STL created and valid")
        else:
             print("[FAIL] Output STL missing or empty")
    else:
        print(f"[FAIL] Failed to process {filepath}")
        
    if os.path.exists(tmp_stl):
        os.unlink(tmp_stl)

if __name__ == "__main__":
    test_file = r"c:\Users\Owner\Downloads\MeshPackageLean\test_cylinder.stl"
    if os.path.exists(test_file):
        test_file_loading_logic(test_file)
    else:
        print(f"Test file not found: {test_file}")
