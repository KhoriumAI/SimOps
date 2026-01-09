"""
OpenFOAM Proof-of-Concept Modal Service

Minimal service to validate OpenFOAM execution on Modal infrastructure.
Runs blockMesh on a simple box geometry and returns the log output.

Usage:
    modal run backend/openfoam_poc_service.py
    
Expected output:
    OpenFOAM header string showing version and blockMesh execution
"""

import modal
import subprocess
from pathlib import Path

# --- IMAGE DEFINITION ---
# Option 1: Use debian base and install OpenFOAM via apt
# This is more reliable than from_registry with add_python
# The openfoam.org PPA provides packages for Ubuntu/Debian

# Option 2: Use a simpler approach with run_commands
# to install OpenFOAM in a Debian container

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "wget",
        "software-properties-common",
        "gnupg2",
        "curl",
    )
    .run_commands(
        # Add OpenFOAM repository (ESI/OpenCFD version)
        "curl -s https://dl.openfoam.com/add-debian-repo.sh | bash",
        "apt-get update",
        "apt-get install -y openfoam2406-default",
    )
    .pip_install("boto3")
)


app = modal.App("khorium-openfoam-poc")


def create_simple_box_case(case_dir: Path) -> None:
    """
    Create a minimal OpenFOAM case with a simple box geometry.
    This tests blockMesh without needing snappyHexMesh.
    """
    # Create directory structure
    (case_dir / "system").mkdir(parents=True, exist_ok=True)
    (case_dir / "constant").mkdir(parents=True, exist_ok=True)
    (case_dir / "0").mkdir(parents=True, exist_ok=True)
    
    # blockMeshDict - Simple 1m x 1m x 1m box with 10x10x10 cells
    block_mesh_dict = '''FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}

convertToMeters 1;

vertices
(
    (0 0 0)
    (1 0 0)
    (1 1 0)
    (0 1 0)
    (0 0 1)
    (1 0 1)
    (1 1 1)
    (0 1 1)
);

blocks
(
    hex (0 1 2 3 4 5 6 7) (10 10 10) simpleGrading (1 1 1)
);

edges
();

boundary
(
    allBoundary
    {
        type patch;
        faces
        (
            (3 7 6 2)
            (0 4 7 3)
            (2 6 5 1)
            (1 5 4 0)
            (0 3 2 1)
            (4 5 6 7)
        );
    }
);
'''
    (case_dir / "system" / "blockMeshDict").write_text(block_mesh_dict)
    
    # controlDict - Minimal required file
    control_dict = '''FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}

application     blockMesh;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1;
deltaT          1;
writeControl    timeStep;
writeInterval   1;
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
'''
    (case_dir / "system" / "controlDict").write_text(control_dict)
    
    # fvSchemes - Required even for meshing
    fv_schemes = '''FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}

ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear orthogonal; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
'''
    (case_dir / "system" / "fvSchemes").write_text(fv_schemes)
    
    # fvSolution - Required even for meshing
    fv_solution = '''FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}

solvers {}
'''
    (case_dir / "system" / "fvSolution").write_text(fv_solution)


@app.function(
    image=image,
    timeout=300,
    cpu=2,
    memory=4096,
)
def hello_openfoam() -> dict:
    """
    Proof-of-concept: Run blockMesh on Modal and return the log.
    
    This validates:
    1. Docker image pulls and starts correctly
    2. OpenFOAM environment can be sourced
    3. blockMesh executes in gVisor sandbox
    4. Logs can be captured and returned
    
    Returns:
        dict with success status, stdout, stderr, and timing
    """
    import time
    
    print("[PoC] Starting OpenFOAM Proof-of-Concept on Modal...")
    
    # Setup case directory
    case_dir = Path("/tmp/openfoam_poc")
    if case_dir.exists():
        import shutil
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[PoC] Creating case directory: {case_dir}")
    create_simple_box_case(case_dir)
    
    # List created files
    print("[PoC] Case directory contents:")
    for f in case_dir.rglob("*"):
        if f.is_file():
            print(f"  - {f.relative_to(case_dir)}")
    
    # Find OpenFOAM bashrc
    # The official image has it in /openfoam or /opt/openfoam*
    bashrc_paths = [
        "/openfoam/bash.rc",  # Docker image convention
        "/usr/lib/openfoam/openfoam2406/etc/bashrc",
        "/opt/openfoam2406/etc/bashrc",
        "/opt/OpenFOAM-v2406/etc/bashrc",
    ]
    
    bashrc = None
    for path in bashrc_paths:
        if Path(path).exists():
            bashrc = path
            break
    
    # If no bashrc found, try to find it
    if not bashrc:
        print("[PoC] Looking for OpenFOAM installation...")
        result = subprocess.run(
            ["find", "/", "-name", "bashrc", "-path", "*openfoam*", "-type", "f"],
            capture_output=True, text=True, timeout=30
        )
        if result.stdout.strip():
            bashrc = result.stdout.strip().split('\n')[0]
            print(f"[PoC] Found bashrc at: {bashrc}")
    
    # Run blockMesh
    print("[PoC] Executing blockMesh...")
    start_time = time.time()
    
    if bashrc:
        cmd = f"source {bashrc} && cd {case_dir} && blockMesh"
    else:
        # Try running blockMesh directly (might be in PATH)
        cmd = f"cd {case_dir} && blockMesh"
    
    result = subprocess.run(
        ["/bin/bash", "-c", cmd],
        capture_output=True,
        text=True,
        timeout=120
    )
    
    duration = time.time() - start_time
    
    print(f"[PoC] blockMesh completed in {duration:.2f}s")
    print(f"[PoC] Return code: {result.returncode}")
    
    # Check if mesh was created
    mesh_dir = case_dir / "constant" / "polyMesh"
    mesh_exists = mesh_dir.exists()
    
    if mesh_exists:
        points_file = mesh_dir / "points"
        if points_file.exists():
            # Count points (rough mesh size estimate)
            content = points_file.read_text()
            num_points = content.count('\n') - 20  # Rough estimate
            print(f"[PoC] Mesh created with ~{num_points} points")
    
    # Prepare result
    response = {
        "success": result.returncode == 0,
        "mesh_created": mesh_exists,
        "duration_seconds": duration,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "bashrc_path": bashrc,
    }
    
    # Print summary
    if result.returncode == 0:
        print("\n" + "="*60)
        print("✅ SUCCESS: OpenFOAM blockMesh executed on Modal!")
        print("="*60)
        # Print first 30 lines of output (contains header)
        for line in result.stdout.split('\n')[:30]:
            print(line)
    else:
        print("\n" + "="*60)
        print("❌ FAILURE: blockMesh did not complete successfully")
        print("="*60)
        print(result.stderr[:2000])
    
    return response


@app.local_entrypoint()
def main():
    """
    Local entrypoint for running the PoC test.
    
    Usage:
        modal run backend/openfoam_poc_service.py
    """
    print("="*60)
    print("OpenFOAM Modal Proof-of-Concept")
    print("="*60)
    print()
    print("This test will:")
    print("1. Pull the opencfd/openfoam2406-run Docker image")
    print("2. Create a simple box case directory")
    print("3. Run blockMesh")
    print("4. Return the log with OpenFOAM header string")
    print()
    print("Starting Modal function...")
    print()
    
    result = hello_openfoam.remote()
    
    print()
    print("="*60)
    print("RESULT SUMMARY")
    print("="*60)
    print(f"Success: {result['success']}")
    print(f"Mesh Created: {result['mesh_created']}")
    print(f"Duration: {result['duration_seconds']:.2f}s")
    print(f"Bashrc Path: {result['bashrc_path']}")
    
    if result['success']:
        print()
        print("✅ PROOF OF CONCEPT VALIDATED")
        print("OpenFOAM executes correctly on Modal infrastructure!")
    else:
        print()
        print("❌ PROOF OF CONCEPT FAILED")
        print("See stderr output above for details.")
    
    return result
