"""
OpenFOAM SnappyHexMesh Modal Service - Full Integration

Runs snappyHexMesh on Modal serverless infrastructure with:
- STEP→STL conversion via Gmsh
- S3 I/O (download CAD, upload mesh)
- mesh_scope support (Internal/External)
- Volume validation

Usage:
    # Deploy to Modal
    modal deploy backend/openfoam_snappy_service.py
    
    # Run local test
    modal run backend/openfoam_snappy_service.py --input-file cad_files/model.step
"""

import modal
import subprocess
import os
import json
import time
from pathlib import Path
from typing import Dict, Optional, List, Tuple

# --- IMAGE DEFINITION ---
# Combine OpenFOAM (apt) + Gmsh (pip) in a single Debian image
# This allows STEP→STL conversion AND snappyHexMesh in one container

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "wget", "software-properties-common", "gnupg2", "curl",
        # OpenGL deps for Gmsh
        "libgl1-mesa-glx", "libglu1-mesa", "libxrender1", "libxcursor1",
        "libxft2", "libfontconfig1", "libfreetype6", "fontconfig", "libxinerama1",
    )
    .run_commands(
        # Add OpenFOAM repository (ESI/OpenCFD version)
        "curl -s https://dl.openfoam.com/add-debian-repo.sh | bash",
        "apt-get update",
        "apt-get install -y openfoam2406-default",
    )
    .pip_install("boto3", "gmsh", "numpy", "trimesh")
)

app = modal.App("khorium-openfoam-snappy")
aws_secret = modal.Secret.from_name("my-aws-secret")


# =============================================================================
# HELPER FUNCTIONS (run inside container, not exposed as Modal endpoints)
# =============================================================================

def convert_step_to_stl(step_path: str, stl_path: str, 
                        deviation: float = 0.01, 
                        min_size: float = 0.5, 
                        max_size: float = 10.0) -> Tuple[bool, Optional[Dict]]:
    """
    Convert STEP to STL using Gmsh with high-fidelity discretization.
    Also extracts volume centroids for multi-region snappyHexMesh support.
    
    Returns:
        Tuple of (success, metadata_dict with volume_centroids and cad_volume)
    """
    import gmsh
    
    print(f"[Gmsh] Converting STEP to STL: {step_path}")
    
    try:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("General.Verbosity", 2)
        
        # Load STEP file
        gmsh.model.occ.importShapes(step_path)
        gmsh.model.occ.synchronize()
        
        # Get volumes and compute centroids for snappyHexMesh insidePoints
        volumes = gmsh.model.getEntities(3)
        volume_centroids = []
        total_cad_volume = 0.0
        
        for dim, tag in volumes:
            # Get mass properties (volume and center of mass)
            mass_props = gmsh.model.occ.getMass(dim, tag)
            total_cad_volume += mass_props
            
            # Get center of mass for this volume
            com = gmsh.model.occ.getCenterOfMass(dim, tag)
            volume_centroids.append(com)
            print(f"[Gmsh] Volume {tag}: {mass_props:.4f} mm³, centroid: {com}")
        
        print(f"[Gmsh] Total CAD volume: {total_cad_volume:.4f} mm³")
        print(f"[Gmsh] Found {len(volume_centroids)} volume(s)")
        
        # Set mesh sizing for STL discretization
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_size)
        gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", deviation)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
        gmsh.option.setNumber("Mesh.MinimumCircleNodes", 12)
        
        # Generate surface mesh (2D) for STL
        gmsh.model.mesh.generate(2)
        
        # Write STL
        gmsh.write(stl_path)
        gmsh.finalize()
        
        print(f"[Gmsh] STL written: {stl_path}")
        
        return True, {
            "volume_centroids": volume_centroids,
            "cad_volume": total_cad_volume,
            "num_volumes": len(volumes)
        }
        
    except Exception as e:
        print(f"[Gmsh] ERROR: {e}")
        try:
            gmsh.finalize()
        except:
            pass
        return False, None


def get_stl_bounds(stl_path: str) -> Tuple[List[float], List[float]]:
    """Get bounding box of STL file using trimesh."""
    import trimesh
    import numpy as np
    
    try:
        mesh = trimesh.load(stl_path)
        if mesh.bounds is not None:
            return mesh.bounds[0].tolist(), mesh.bounds[1].tolist()
    except Exception as e:
        print(f"[WARNING] trimesh bounds failed: {e}")
    
    # Fallback: default bounds
    return [-100, -100, -100], [100, 100, 100]


def create_snappy_case(case_dir: Path, stl_path: str, cell_size: float = 2.0,
                       mesh_scope: str = 'Internal', 
                       volume_centroids: List = None) -> None:
    """
    Create OpenFOAM case directory for snappyHexMesh.
    
    Args:
        case_dir: Path to case directory
        stl_path: Path to input STL
        cell_size: Target cell size in mm
        mesh_scope: 'Internal' (mesh inside solid) or 'External' (mesh air around solid)
        volume_centroids: List of (x,y,z) from CAD for robust insidePoint selection
    """
    import numpy as np
    
    # Create directory structure
    (case_dir / "constant" / "triSurface").mkdir(parents=True, exist_ok=True)
    (case_dir / "system").mkdir(parents=True, exist_ok=True)
    (case_dir / "0").mkdir(parents=True, exist_ok=True)
    
    # Copy STL
    stl_filename = "geometry.stl"
    stl_dest = case_dir / "constant" / "triSurface" / stl_filename
    import shutil
    shutil.copy(stl_path, stl_dest)
    
    # Get bounds for blockMesh
    min_b, max_b = get_stl_bounds(stl_path)
    min_b, max_b = np.array(min_b), np.array(max_b)
    
    center = (min_b + max_b) / 2
    size = max_b - min_b
    
    # Add 50% margin for blockMesh domain
    margin = size * 0.5
    min_pt = min_b - margin
    max_pt = max_b + margin
    
    # Determine locationInMesh based on mesh_scope
    # CRITICAL: This determines whether we mesh INSIDE the solid or OUTSIDE (air)
    if mesh_scope == 'External':
        # External flow: mesh the AIR around the solid
        # Pick a point definitely outside the solid but inside the blockMesh domain
        location_in_mesh = min_b - (margin * 0.25)
        print(f"[Snappy] EXTERNAL mode: meshing air around solid")
        print(f"[Snappy] Location in mesh (air): {location_in_mesh}")
        all_locations = [location_in_mesh.tolist()]
    else:
        # Internal: mesh INSIDE the solid
        # Use CAD-extracted volume centroids if available (robust)
        if volume_centroids and len(volume_centroids) > 0:
            # Round and add a small random-ish offset to avoid being exactly on a cell face
            all_locations = [[round(c + 0.0123, 6) for c in loc] for loc in volume_centroids]
            location_in_mesh = all_locations[0]
            print(f"[Snappy] INTERNAL mode: using {len(all_locations)} CAD volume centroid(s) (with offset)")
        else:
            # Fallback to geometric center
            location_in_mesh = [round(c + 0.0123, 6) for c in center.tolist()]
            all_locations = [location_in_mesh]
            print(f"[Snappy] INTERNAL mode: using bbox center as fallback (with offset)")
        
        print(f"[Snappy] Location in mesh (solid): {location_in_mesh}")
    
    # Calculate block mesh cell count (coarser than final mesh)
    n_cells = ((max_pt - min_pt) / (cell_size * 2.0)).astype(int)
    n_cells = [max(1, int(n)) for n in n_cells]
    
    # --- blockMeshDict ---
    block_mesh_dict = f'''FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}

convertToMeters 1.0;

vertices
(
    ({min_pt[0]} {min_pt[1]} {min_pt[2]})
    ({max_pt[0]} {min_pt[1]} {min_pt[2]})
    ({max_pt[0]} {max_pt[1]} {min_pt[2]})
    ({min_pt[0]} {max_pt[1]} {min_pt[2]})
    ({min_pt[0]} {min_pt[1]} {max_pt[2]})
    ({max_pt[0]} {min_pt[1]} {max_pt[2]})
    ({max_pt[0]} {max_pt[1]} {max_pt[2]})
    ({min_pt[0]} {max_pt[1]} {max_pt[2]})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({n_cells[0]} {n_cells[1]} {n_cells[2]}) simpleGrading (1 1 1)
);

edges ();

boundary
(
    allBoundary
    {{
        type patch;
        faces
        (
            (0 1 5 4) (1 2 6 5) (2 3 7 6)
            (3 0 4 7) (0 3 2 1) (4 5 6 7)
        );
    }}
);
'''
    (case_dir / "system" / "blockMeshDict").write_text(block_mesh_dict)
    
    # Build location clause for snappyHexMeshDict
    if len(all_locations) > 1:
        loc_lines = [f"        ({loc[0]} {loc[1]} {loc[2]})" for loc in all_locations]
        location_clause = "insidePoints\n    (\n" + "\n".join(loc_lines) + "\n    );"
    else:
        loc = all_locations[0]
        location_clause = f"locationInMesh ({loc[0]} {loc[1]} {loc[2]});"
    
    # --- snappyHexMeshDict ---
    # RESTORED FULL ROBUST VERSION
    snappy_dict = f'''FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      snappyHexMeshDict;
}}

castellatedMesh true;
snap            true;
addLayers       false;

geometry
{{
    {stl_filename}
    {{
        type triSurfaceMesh;
        file "{stl_filename}";
        name surface;
    }}
}};

castellatedMeshControls
{{
    maxLocalCells 100000;
    maxGlobalCells 2000000;
    minRefinementCells 10;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels 3;

    features ();

    refinementSurfaces
    {{
        surface
        {{
            level (2 3);
        }}
    }}

    resolveFeatureAngle 30;
    refinementRegions {{}};
    
    {location_clause}
    
    allowFreeStandingZoneFaces true;
}}

snapControls
{{
    nSmoothPatch 3;
    tolerance 2.0;
    nSolveIter 30;
    nRelaxIter 5;
    nFeatureSnapIter 10;
    implicitFeatureSnap false;
    explicitFeatureSnap false;
    multiRegionFeatureSnap false;
}}

addLayersControls
{{
    relativeSizes true;
    layers
    {{
    }}
    expansionRatio 1.2;
    finalLayerThickness 0.5;
    minThickness 0.1;
    nGrow 0;
    featureAngle 60;
    slipFeatureAngle 30;
    nRelaxIter 3;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedianAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
    nRelaxedIter 20;
}}

meshQualityControls
{{
    maxNonOrtho 65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave 80;
    minVol 1e-13;
    minTetQuality 1e-15;
    minArea -1;
    minTwist 0.02;
    minDeterminant 0.001;
    minFaceWeight 0.02;
    minVolRatio 0.01;
    minTriangleTwist -1;
    nSmoothScale 4;
    errorReduction 0.75;
}}

mergeTolerance 1e-6;
'''
    (case_dir / "system" / "snappyHexMeshDict").write_text(snappy_dict)
    
    # --- controlDict ---
    control_dict = '''FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}
application     snappyHexMesh;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1000;
deltaT          1;
writeControl    timeStep;
writeInterval   100;
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
'''
    (case_dir / "system" / "controlDict").write_text(control_dict)
    
    # --- fvSchemes ---
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
    
    # --- fvSolution ---
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


def get_openfoam_env(case_dir: Path) -> Dict:
    """Get environment variables for OpenFOAM execution."""
    of_base = "/usr/lib/openfoam/openfoam2406"
    of_platform = f"{of_base}/platforms/linux64GccDPInt32Opt"
    
    env = os.environ.copy()
    env.update({
        "WM_PROJECT": "OpenFOAM",
        "WM_PROJECT_VERSION": "v2406",
        "WM_PROJECT_DIR": of_base,
        "FOAM_LIBBIN": f"{of_platform}/lib",
        "FOAM_APPBIN": f"{of_platform}/bin",
        "FOAM_USER_APPBIN": "/root/OpenFOAM/-v2406/platforms/linux64GccDPInt32Opt/bin",
        "FOAM_USER_LIBBIN": "/root/OpenFOAM/-v2406/platforms/linux64GccDPInt32Opt/lib",
        "PATH": f"{of_platform}/bin:{of_base}/bin:" + env.get("PATH", ""),
        "LD_LIBRARY_PATH": f"{of_platform}/lib:{of_platform}/lib/dummy:" + env.get("LD_LIBRARY_PATH", ""),
        "PWD": str(case_dir),
        "FOAM_CASE": str(case_dir),
        "FOAM_RUN": str(case_dir.parent),
    })
    return env


def run_openfoam_pipeline(case_dir: Path) -> Tuple[bool, str, float]:
    """
    Run blockMesh + snappyHexMesh + foamMeshToFluent pipeline.
    
    Returns:
        Tuple of (success, stdout, duration_seconds)
    """
    # Instead of sourcing bashrc (which fails in gVisor due to bash function issues),
    # we directly set the required environment variables
    env = get_openfoam_env(case_dir)
    
    print(f"[OpenFOAM] Using direct env vars (avoiding bashrc)")
    
    # Print case directory contents for debugging
    print("[OpenFOAM] Case directory contents:")
    for f in case_dir.rglob("*"):
        if f.is_file():
            print(f"  {f.relative_to(case_dir)} ({f.stat().st_size} bytes)")
    
    start = time.time()
    output_lines = []
    
    # Run each command separately for better error handling
    commands = [
        ("blockMesh", ["blockMesh"], "log.blockMesh"),
        ("snappyHexMesh", ["snappyHexMesh", "-overwrite"], "log.snappy"),
        ("foamMeshToFluent", ["foamMeshToFluent"], "log.fluent"),
        ("listFiles", ["ls", "-R"], "log.ls"),
    ]
    
    for name, cmd, log_file in commands:
        print(f"[OpenFOAM] Running {name}...")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(case_dir),
                env=env,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            # Write log file
            log_path = case_dir / log_file
            log_path.write_text(result.stdout + result.stderr)
            
            output_lines.append(f"[{name}] Return code: {result.returncode}")
            
            if result.returncode != 0:
                print(f"[OpenFOAM] {name} FAILED with code {result.returncode}")
                print(f"[OpenFOAM] STDERR: {result.stderr[-1500:]}")
                print(f"[OpenFOAM] STDOUT: {result.stdout[-1500:]}")
                
                # foamMeshToFluent failure is non-fatal
                if name != "foamMeshToFluent":
                    return False, "\n".join(output_lines) + f"\n{name} failed:\n{result.stderr}", time.time() - start
            else:
                print(f"[OpenFOAM] {name} completed successfully")
                
        except subprocess.TimeoutExpired:
            print(f"[OpenFOAM] {name} TIMEOUT")
            return False, f"{name} timed out", time.time() - start
        except Exception as e:
            print(f"[OpenFOAM] {name} ERROR: {e}")
            return False, f"{name} error: {e}", time.time() - start
    
    duration = time.time() - start
    print(f"[OpenFOAM] Pipeline completed in {duration:.1f}s")
    
    return True, "\n".join(output_lines), duration


def calculate_mesh_volume(case_dir: Path, env: Dict) -> Optional[float]:
    """
    Calculate total volume of generated mesh by summing cell volumes.
    Uses checkMesh to get volume.
    """
    cmd = ["checkMesh"]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(case_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # print(f"[calculate_mesh_volume] checkMesh output:\n{result.stdout}")
        
        if result.returncode != 0:
            print(f"[calculate_mesh_volume] checkMesh failed with code {result.returncode}")
            return None

        # Parse "Total volume" or "Mesh volume" from checkMesh output
        # OpenFOAM versions vary in their wording
        for line in result.stdout.split('\n'):
            line_clean = line.strip().lower()
            if 'total volume' in line_clean or 'mesh volume' in line_clean:
                # Extract number: "Total volume = 1234.56" or "Mesh volume: 1234.56"
                parts = line.replace(':', '=').split('=')
                if len(parts) >= 2:
                    try:
                        vol_str = parts[1].strip().split()[0]
                        val = float(vol_str)
                        if val > 0:
                            return val
                    except:
                        pass
    except Exception as e:
        print(f"[calculate_mesh_volume] Error: {e}")
    
    return None


# =============================================================================
# MODAL FUNCTIONS (exposed endpoints)
# =============================================================================

@app.function(
    image=image,
    timeout=1800,  # 30 min for complex meshes
    cpu=4,
    memory=8192,
    secrets=[aws_secret],
)
def run_snappy_hex_mesh(
    bucket: str,
    input_key: str,
    params: dict = None
) -> dict:
    """
    Run snappyHexMesh on Modal with full pipeline.
    
    Args:
        bucket: S3 bucket name
        input_key: S3 key for input STEP/STL file
        params: Dictionary with:
            - mesh_scope: 'Internal' or 'External'
            - cell_size: Target cell size in mm
            
    Returns:
        dict with success, output paths, volume comparison, timing
    """
    import boto3
    
    params = params or {}
    mesh_scope = params.get('mesh_scope', 'Internal')
    cell_size = params.get('cell_size', 2.0)
    
    print(f"[Modal] Starting snappyHexMesh job")
    print(f"[Modal] Input: s3://{bucket}/{input_key}")
    print(f"[Modal] Mesh scope: {mesh_scope}")
    print(f"[Modal] Cell size: {cell_size}")
    
    s3 = boto3.client("s3")
    
    # Setup directories
    work_dir = Path("/tmp/snappy_work")
    work_dir.mkdir(parents=True, exist_ok=True)
    
    case_dir = work_dir / "case"
    case_dir.mkdir(parents=True, exist_ok=True)
    
    # Download input file
    input_ext = Path(input_key).suffix.lower()
    local_input = work_dir / f"input{input_ext}"
    
    print(f"[Modal] Downloading from S3...")
    s3.download_file(bucket, input_key, str(local_input))
    
    # Step 1: Convert STEP to STL if needed
    cad_metadata = None
    if input_ext in ['.step', '.stp']:
        stl_path = work_dir / "geometry.stl"
        success, cad_metadata = convert_step_to_stl(str(local_input), str(stl_path))
        if not success:
            return {"success": False, "message": "STEP to STL conversion failed"}
    else:
        # Already STL
        stl_path = local_input
        cad_metadata = {"volume_centroids": [], "cad_volume": None}
    
    cad_volume = cad_metadata.get("cad_volume") if cad_metadata else None
    volume_centroids = cad_metadata.get("volume_centroids", []) if cad_metadata else []
    
    # Step 2: Create snappyHexMesh case
    print("[Modal] Creating OpenFOAM case...")
    create_snappy_case(
        case_dir, 
        str(stl_path),
        cell_size=cell_size,
        mesh_scope=mesh_scope,
        volume_centroids=volume_centroids
    )
    
    # Step 3: Run OpenFOAM pipeline
    print("[Modal] Running OpenFOAM pipeline...")
    success, output, duration = run_openfoam_pipeline(case_dir)
    
    if not success:
        return {
            "success": False,
            "message": "OpenFOAM pipeline failed",
            "stdout": output[-3000:],
            "duration_seconds": duration
        }
    
    print(f"[Modal] Pipeline completed in {duration:.1f}s")
    
    # Step 4: Calculate mesh volume for validation
    env = get_openfoam_env(case_dir)
    mesh_volume = calculate_mesh_volume(case_dir, env)
    
    # Volume comparison
    volume_validation = {
        "cad_volume": cad_volume,
        "mesh_volume": mesh_volume,
        "mesh_scope": mesh_scope,
    }
    
    if cad_volume and mesh_volume:
        if mesh_scope == 'Internal':
            # Mesh should be close to CAD volume
            ratio = mesh_volume / cad_volume
            volume_validation["volume_ratio"] = ratio
            volume_validation["volume_match"] = 0.8 < ratio < 1.2  # Within 20%
            print(f"[Modal] Volume check: mesh={mesh_volume:.2f}, CAD={cad_volume:.2f}, ratio={ratio:.2%}")
        else:
            # External: mesh volume >> CAD volume (it's the air + margins)
            volume_validation["external_note"] = "External mesh volume includes air domain"
    
    # Step 5: Find and upload output mesh
    msh_files = list(case_dir.rglob("*.msh"))
    fluent_dir = case_dir / "fluentInterface"
    
    if fluent_dir.exists():
        msh_files.extend(list(fluent_dir.rglob("*.msh")))
    
    output_key = None
    if msh_files:
        # Use the largest .msh file (likely the final mesh)
        msh_file = max(msh_files, key=lambda p: p.stat().st_size)
        output_key = input_key.replace("uploads/", "mesh/").replace(".step", ".msh").replace(".stp", ".msh").replace(".stl", ".msh")
        
        print(f"[Modal] Uploading mesh: {msh_file.name} ({msh_file.stat().st_size / 1024:.1f} KB)")
        s3.upload_file(str(msh_file), bucket, output_key)
    
    # Collect log snippets
    logs = {}
    for log_name in ["log.blockMesh", "log.snappy", "log.fluent", "log.ls"]:
        log_path = case_dir / log_name
        if log_path.exists():
            logs[log_name] = log_path.read_text()[-1000:]  # Last 1KB
    
    return {
        "success": True,
        "mesh_storage_key": output_key,  # The S3 key for the mesh file
        "duration_seconds": duration,
        "mesh_scope": mesh_scope,
        "volume_validation": volume_validation,
        "logs": logs,
        "message": f"snappyHexMesh completed in {duration:.1f}s"
    }


@app.function(
    image=image,
    timeout=1800,
    cpu=4,
    memory=8192,
)
def run_snappy_local(
    step_file_bytes: bytes,
    filename: str,
    params: dict = None
) -> dict:
    """
    Run snappyHexMesh on Modal with file passed directly (no S3).
    Use this for local testing without AWS credentials.
    
    Args:
        step_file_bytes: Raw bytes of the STEP/STL file
        filename: Original filename (to determine extension)
        params: Dictionary with mesh_scope, cell_size
    """
    params = params or {}
    mesh_scope = params.get('mesh_scope', 'Internal')
    cell_size = params.get('cell_size', 2.0)
    
    print(f"[Modal] Starting LOCAL snappyHexMesh job")
    print(f"[Modal] File: {filename} ({len(step_file_bytes) / 1024:.1f} KB)")
    print(f"[Modal] Mesh scope: {mesh_scope}")
    
    # Setup directories
    work_dir = Path("/tmp/snappy_work")
    work_dir.mkdir(parents=True, exist_ok=True)
    
    case_dir = work_dir / "case"
    if case_dir.exists():
        import shutil
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    
    # Write input file
    input_ext = Path(filename).suffix.lower()
    local_input = work_dir / f"input{input_ext}"
    local_input.write_bytes(step_file_bytes)
    
    # Step 1: Convert STEP to STL if needed
    cad_metadata = None
    if input_ext in ['.step', '.stp']:
        stl_path = work_dir / "geometry.stl"
        success, cad_metadata = convert_step_to_stl(str(local_input), str(stl_path))
        if not success:
            return {"success": False, "message": "STEP to STL conversion failed"}
    else:
        stl_path = local_input
        cad_metadata = {"volume_centroids": [], "cad_volume": None}
    
    cad_volume = cad_metadata.get("cad_volume") if cad_metadata else None
    volume_centroids = cad_metadata.get("volume_centroids", []) if cad_metadata else []
    
    # Step 2: Create snappyHexMesh case
    print("[Modal] Creating OpenFOAM case...")
    create_snappy_case(
        case_dir, 
        str(stl_path),
        cell_size=cell_size,
        mesh_scope=mesh_scope,
        volume_centroids=volume_centroids
    )
    
    # Step 3: Run OpenFOAM pipeline
    print("[Modal] Running OpenFOAM pipeline...")
    success, output, duration = run_openfoam_pipeline(case_dir)
    
    if not success:
        # Try to get log files for debugging
        logs = {}
        for log_name in ["log.blockMesh", "log.snappy"]:
            log_path = case_dir / log_name
            if log_path.exists():
                logs[log_name] = log_path.read_text()[-2000:]
        
        return {
            "success": False,
            "message": "OpenFOAM pipeline failed",
            "stdout": output[-3000:],
            "logs": logs,
            "duration_seconds": duration
        }
    
    print(f"[Modal] Pipeline completed in {duration:.1f}s")
    
    # Step 4: Calculate mesh volume for validation
    env = get_openfoam_env(case_dir)
    mesh_volume = calculate_mesh_volume(case_dir, env)
    
    # Volume comparison
    volume_validation = {
        "cad_volume": cad_volume,
        "mesh_volume": mesh_volume,
        "mesh_scope": mesh_scope,
    }
    
    if cad_volume and mesh_volume:
        if mesh_scope == 'Internal':
            ratio = mesh_volume / cad_volume
            volume_validation["volume_ratio"] = ratio
            volume_validation["volume_match"] = 0.8 < ratio < 1.2
            print(f"[Modal] Volume check: mesh={mesh_volume:.2f}, CAD={cad_volume:.2f}, ratio={ratio:.2%}")
        else:
            volume_validation["external_note"] = "External mesh volume includes air domain"
    
    # Collect log snippets
    logs = {}
    for log_name in ["log.blockMesh", "log.snappy", "log.fluent", "log.ls"]:
        log_path = case_dir / log_name
        if log_path.exists():
            logs[log_name] = log_path.read_text()[-1000:]
            
    # DEBUG: List all files in case directory to debug missing .msh
    file_listing = []
    print("[Modal] Listing case directory contents:")
    for f in case_dir.rglob("*"):
        info = f"{f.relative_to(case_dir)} ({f.stat().st_size} bytes)"
        print(f"  {info}")
        file_listing.append(info)
            
    # Upload results to S3
    # Expected location: fluentInterface/case.msh (or similar)
    fluent_dir = case_dir / "fluentInterface"
    expected_msh = fluent_dir / "case.msh"
    
    print(f"[Modal] Checking for mesh file...")
    print(f"[Modal] Expected path: {expected_msh}")
    print(f"[Modal] fluentInterface exists: {fluent_dir.exists()}")
    
    if fluent_dir.exists():
        print(f"[Modal] fluentInterface contents: {list(fluent_dir.iterdir())}")
    
    # Try multiple search strategies
    msh_file = None
    
    # Strategy 1: Check expected location
    if expected_msh.exists():
        msh_file = expected_msh
        print(f"[Modal] Found at expected location: {msh_file}")
    
    # Strategy 2: Search fluentInterface for any .msh
    if not msh_file and fluent_dir.exists():
        msh_files = list(fluent_dir.glob("*.msh"))
        if msh_files:
            msh_file = msh_files[0]
            print(f"[Modal] Found via fluentInterface glob: {msh_file}")
    
    # Strategy 3: Recursive search entire case dir
    if not msh_file:
        msh_files = list(case_dir.rglob("*.msh"))
        if msh_files:
            msh_file = msh_files[0]
            print(f"[Modal] Found via rglob: {msh_file}")
    
    # Strategy 4: Check for .MSH (case sensitivity on Linux)
    if not msh_file:
        msh_files = list(case_dir.rglob("*.MSH"))
        if msh_files:
            msh_file = msh_files[0]
            print(f"[Modal] Found via rglob (uppercase): {msh_file}")
    
    mesh_storage_key = None
    if msh_file:
        base_name = Path(input_key).stem
        mesh_storage_key = f"mesh/modal_{int(time.time())}/{base_name}.msh"
        print(f"[Modal] Uploading result to s3://{bucket}/{mesh_storage_key}...")
        s3.upload_file(str(msh_file), bucket, mesh_storage_key)
    else:
        print(f"[Modal] ERROR: No .msh file found!")
        print(f"[Modal] All files in case_dir:")
        for f in case_dir.rglob("*"):
            if f.is_file():
                print(f"  {f.relative_to(case_dir)}")
    # Upload VTK if exists (useful for vis)
    vtk_files = list(case_dir.rglob("*.vtu")) + list(case_dir.rglob("*.vtk"))
    vtk_storage_key = None
    if vtk_files:
        vtk_file = vtk_files[0]
        base_name = Path(input_key).stem
        vtk_storage_key = f"mesh/modal_{int(time.time())}/{base_name}.vtu"
        print(f"[Modal] Uploading VTK to s3://{bucket}/{vtk_storage_key}...")
        s3.upload_file(str(vtk_file), bucket, vtk_storage_key)
    
    return {
        "success": True,
        "duration_seconds": duration,
        "mesh_scope": mesh_scope,
        "mesh_storage_key": mesh_storage_key,
        "vtk_storage_key": vtk_storage_key,
        "volume_validation": volume_validation,
        "logs": logs,
        "file_listing": file_listing,
        "message": f"snappyHexMesh completed in {duration:.1f}s"
    }


@app.local_entrypoint()
def main(
    input_file: str = "cad_files/model.step",
    mesh_scope: str = "Internal",
    cell_size: float = 2.0,
    use_s3: bool = False,
    bucket: str = None,
):
    """
    Local entrypoint for testing snappyHexMesh on Modal.
    
    Usage (local, no S3):
        modal run backend/openfoam_snappy_service.py --input-file cad_files/model.step
        
    Usage (with S3):
        modal run backend/openfoam_snappy_service.py --input-file cad_files/model.step --use-s3 --bucket my-bucket
    """
    print("="*60)
    print("SnappyHexMesh Modal Service - Local Test")
    print("="*60)
    print(f"Input: {input_file}")
    print(f"Mesh Scope: {mesh_scope}")
    print(f"Cell Size: {cell_size}")
    print(f"Use S3: {use_s3}")
    print()
    
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"ERROR: File not found: {input_file}")
        return {"success": False, "message": f"File not found: {input_file}"}
    
    if use_s3:
        # S3 mode - upload then process
        import boto3
        
        if not bucket:
            bucket = os.environ.get("S3_BUCKET_NAME", "muaz-webdev-assets")
        
        s3 = boto3.client("s3")
        s3_key = f"uploads/modal_test/{input_path.name}"
        
        print(f"Uploading to s3://{bucket}/{s3_key}...")
        s3.upload_file(str(input_path), bucket, s3_key)
        
        result = run_snappy_hex_mesh.remote(
            bucket=bucket,
            input_key=s3_key,
            params={"mesh_scope": mesh_scope, "cell_size": cell_size}
        )
    else:
        # Local mode - pass file bytes directly (no S3 needed)
        print("Running in LOCAL mode (no S3)...")
        file_bytes = input_path.read_bytes()
        
        result = run_snappy_local.remote(
            step_file_bytes=file_bytes,
            filename=input_path.name,
            params={"mesh_scope": mesh_scope, "cell_size": cell_size}
        )
    
    print()
    print("="*60)
    print("RESULT SUMMARY")
    print("="*60)
    print(f"Success: {result['success']}")
    print(f"Duration: {result.get('duration_seconds', 0):.1f}s")
    
    if result.get('s3_output_path'):
        print(f"Output: {result['s3_output_path']}")
    
    if result.get('volume_validation'):
        vv = result['volume_validation']
        print()
        print("VOLUME VALIDATION:")
        print(f"  CAD Volume: {vv.get('cad_volume', 'N/A')}")
        print(f"  Mesh Volume: {vv.get('mesh_volume', 'N/A')}")
        print(f"  Mesh Scope: {vv.get('mesh_scope')}")
        if 'volume_ratio' in vv:
            print(f"  Ratio: {vv['volume_ratio']:.2%}")
            print(f"  Match: {'✅' if vv.get('volume_match') else '❌'}")
    
    if not result['success']:
        print()
        print("ERROR:")
        print(result.get('message', 'Unknown error'))
        if 'stdout' in result:
            print(result['stdout'][-2000:])
        if 'logs' in result:
            for name, content in result['logs'].items():
                print(f"\n--- {name} ---")
                print(content[-1000:])
    
    return result

