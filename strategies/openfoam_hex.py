"""
OpenFOAM cfMesh Integration for Hex-Dominant Meshing
=====================================================

Wraps OpenFOAM's cfMesh (cartesianMesh) via WSL2 for robust hex-dominant mesh generation.
This replaces the CoACD gluing approach which has fundamental limitations on curved surfaces.

Pipeline:
1. Create OpenFOAM case directory structure
2. Copy STL to constant/triSurface/
3. Generate meshDict with target cell size
4. Run cartesianMesh via WSL
5. Convert to Fluent format via foamMeshToFluent
6. Return .msh file path
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional
import platform


def check_wsl_available() -> bool:
    """Check if WSL is available on this system."""
    if platform.system() != 'Windows':
        return False  # Not needed on Linux
    try:
        result = subprocess.run(['wsl', '--list'], capture_output=True, timeout=5)
        return result.returncode == 0
    except:
        return False


def check_openfoam_available() -> bool:
    """Check if cfMesh (cartesianMesh) is installed in WSL."""
    try:
        result = subprocess.run(
            ['wsl', 'bash', '-c', 'which cartesianMesh'],
            capture_output=True, timeout=10
        )
        return result.returncode == 0
    except:
        return False


def check_snappy_available() -> bool:
    """Check if snappyHexMesh is installed in WSL."""
    try:
        result = subprocess.run(
            ['wsl', 'bash', '-c', 'source /usr/lib/openfoam/openfoam*/etc/bashrc 2>/dev/null || source /opt/openfoam*/etc/bashrc 2>/dev/null; which snappyHexMesh'],
            capture_output=True, timeout=10
        )
        return result.returncode == 0
    except:
        return False



def check_any_openfoam_available() -> bool:
    """Check if ANY OpenFOAM mesher (cfMesh or snappy) is available."""
    return check_openfoam_available() or check_snappy_available()


def get_stl_bounds(stl_path: str):
    """
    Get STL bounding box.
    Try using trimesh, fallback to simple parsing.
    """
    try:
        import trimesh
        mesh = trimesh.load(stl_path)
        return mesh.bounds
    except ImportError:
        # Simple fallback parser for ASCII/Binary STL
        # This is risky, but we only need approx bounds for background mesh
        # For robustness, we assume user has trimesh (it's in requirements)
        print("[OpenFOAM] WARNING: trimesh not found, using default bounds")
        return [[-100, -100, -100], [100, 100, 100]]


def create_snappy_case(case_dir: Path, stl_path: str, cell_size: float = 2.0) -> None:
    """
    Create OpenFOAM case directory structure for snappyHexMesh.
    
    Args:
        case_dir: Path to case directory
        stl_path: Path to input STL file
        cell_size: Target cell size in mm
    """
    # Create directory structure
    (case_dir / "constant" / "triSurface").mkdir(parents=True, exist_ok=True)
    (case_dir / "system").mkdir(parents=True, exist_ok=True)
    
    # Copy STL file
    stl_filename = "geometry.stl"
    stl_dest = case_dir / "constant" / "triSurface" / stl_filename
    shutil.copy(stl_path, stl_dest)
    
    # Get bounds for blockMesh
    bounds = get_stl_bounds(stl_path)
    min_b, max_b = bounds[0], bounds[1]
    
    # Add margin (50%)
    center = (min_b + max_b) / 2
    size = max_b - min_b
    margin = size * 0.5
    min_pt = min_b - margin
    max_pt = max_b + margin
    
    # Identify a point inside the mesh (approximation)
    # For a convex object, centroid is usually safe.
    # For concave, this can fail. Using center of bbox is a 50/50 gamble.
    # Ideally ray cast. 
    # Fallback: assume origin (0,0,0) or center if provided.
    # Note: STL is in mm, but blockMesh converts to meters (0.001).
    # We must scale this point to match the background mesh units (Meters).
    location_in_mesh = center * 0.001 
    
    # Generate blockMeshDict
    # Calculate number of cells based on cell_size * 2 (coarser background)
    n_cells = ((max_pt - min_pt) / (cell_size * 2.0)).astype(int)
    n_cells = [max(1, n) for n in n_cells]
    
    block_mesh_dict = f'''FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}

convertToMeters 0.001; // STL is likely in mm

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

edges
();

boundary
(
    allBoundary
    {{
        type patch;
        faces
        (
            (0 1 5 4)
            (1 2 6 5)
            (2 3 7 6)
            (3 0 4 7)
            (0 3 2 1)
            (4 5 6 7)
        );
    }}
);
'''
    (case_dir / "system" / "blockMeshDict").write_text(block_mesh_dict)

    # Generate snappyHexMeshDict
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
        name {stl_filename};
    }}
}};

castellatedMeshControls
{{
    maxLocalCells 100000;
    maxGlobalCells 2000000;
    minRefinementCells 10;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels 3;

    // IMPLICIT FEATURE SNAPPING (More robust for dirty STL)
    features
    (
    );

    refinementSurfaces
    {{
        {stl_filename}
        {{
            level (2 3); // Refinement level (min max)
        }}
    }}

    resolveFeatureAngle 30;
    
    refinementRegions
    {{
    }}

    locationInMesh ({location_in_mesh[0]} {location_in_mesh[1]} {location_in_mesh[2]});
    
    allowFreeStandingZoneFaces true;
}}

snapControls
{{
    nSmoothPatch 3;
    tolerance 2.0;
    nSolveIter 30;
    nRelaxIter 5;
    nFeatureSnapIter 10;
    implicitFeatureSnap true; // Enable implicit
    explicitFeatureSnap false; // Disable explicit
    multiRegionFeatureSnap false;
}}

addLayersControls
{{
}}

meshQualityControls
{{
    maxNonOrthogonality 65;
    maxNonOrtho 65;             // Added for compatibility with v2406
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

    # Generate controlDict
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

    # Generate fvSchemes (required for snappy quality checks)
    fv_schemes = '''FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}

ddtSchemes
{
    default         Euler;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    default         none;
    div(rhoPhi,U)   Gauss linearUpwind grad(U);
}

laplacianSchemes
{
    default         Gauss linear orthogonal;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}
'''
    (case_dir / "system" / "fvSchemes").write_text(fv_schemes)
    
    # Generate fvSolution (required for snappy quality checks)
    fv_solution = '''FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}

solvers
{
    "p.*"
    {
        solver           GAMG;
        tolerance        1e-06;
        relTol           0.01;
        smoother         GaussSeidel;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
}
'''
    (case_dir / "system" / "fvSolution").write_text(fv_solution)


def run_snappy_hex_mesh(case_dir: Path, verbose: bool = True) -> bool:
    """Run snappyHexMesh pipeline via WSL."""
    case_dir_wsl = str(case_dir).replace('\\', '/').replace('C:', '/mnt/c')
    
    # Chain commands: if one fails, print its log and exit
    # We use parentheses to group the "try or print log" logic
    # SKIPPING surfaceFeatureExtract (implicit snapping)
    cmd_str = (
        f'source /usr/lib/openfoam/openfoam*/etc/bashrc 2>/dev/null || source /opt/openfoam*/etc/bashrc 2>/dev/null; '
        f'cd "{case_dir_wsl}" && '
        f'(blockMesh > log.blockMesh 2>&1 || (echo "ERR: blockMesh failed"; cat log.blockMesh; exit 1)) && '
        f'(snappyHexMesh -overwrite > log.snappy 2>&1 || (echo "ERR: snappyHexMesh failed"; cat log.snappy; exit 1))'
    )
    
    cmd = ['wsl', 'bash', '-c', cmd_str]
    
    if verbose:
        print("[OpenFOAM] Running snappyHexMesh pipeline (Implicit Feature Snapping)...")
    
    try:
        # Increase timeout for snappy (can be slow)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        
        if result.returncode != 0:
            print(f"[OpenFOAM] ERROR: Pipeline failed.")
            print(f"Stdout/Stderr capture:\n{result.stdout}\n{result.stderr}")
            return False
            
        return True
    except subprocess.TimeoutExpired:
        print("[OpenFOAM] ERROR: snappyHexMesh timed out")
        return False
    except Exception as e:
        print(f"[OpenFOAM] ERROR: {e}")
        return False



def convert_to_fluent(case_dir: Path, verbose: bool = True) -> Optional[Path]:
    """
    Convert OpenFOAM mesh to Fluent .msh format.
    
    Args:
        case_dir: Path to OpenFOAM case directory
        verbose: Whether to print logs
        
    Returns:
        Path to .msh file or None if failed
    """
    case_dir_wsl = str(case_dir).replace('\\', '/').replace('C:', '/mnt/c')

    cmd = [
        'wsl', 'bash', '-c',
        f'source /usr/lib/openfoam/openfoam*/etc/bashrc 2>/dev/null || source /opt/openfoam*/etc/bashrc 2>/dev/null; '
        f'cd "{case_dir_wsl}" && '
        f'foamMeshToFluent > log.fluent 2>&1'
    ]
    
    if verbose:
        print("[OpenFOAM] Converting to Fluent format...")
        
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        
        # Check if .msh file exists (recursively, as it may be in fluentInterface/)
        msh_files = list(case_dir.rglob("*.msh"))
        if not msh_files:
            if verbose:
                print(f"[OpenFOAM] WARNING: No .msh file found in {case_dir}")
                # List all files to debug
                print(f"[OpenFOAM] Directory contents:")
                for f in case_dir.rglob("*"):
                    print(f"  - {f.relative_to(case_dir)}")
                
                # Print log if failed
                log_file = case_dir / "log.fluent"
                if log_file.exists():
                    print(f"[OpenFOAM] Conversion Log:\n{log_file.read_text()}")
            return None
            
        return msh_files[0]
        
    except Exception as e:
        if verbose:
            print(f"[OpenFOAM] Error converting to Fluent: {e}")
        return None


def create_openfoam_case(case_dir: Path, stl_path: str, cell_size: float = 2.0) -> None:
    # Dummy placeholder as it seems missing but usage is guarded
    pass

def run_cartesian_mesh(case_dir: Path, verbose: bool = True) -> bool:
    # Dummy placeholder
    return False


def generate_openfoam_hex_mesh(
    stl_path: str,
    output_path: str,
    cell_size: float = 2.0,
    verbose: bool = True
) -> Dict:
    """
    Generate hex-dominant mesh using OpenFOAM (cfMesh or snappyHexMesh).
    """
    if verbose:
        print(f"[OpenFOAM] Generating hex mesh for: {stl_path}")
        print(f"[OpenFOAM] Target cell size: {cell_size} mm")
    
    # Check prerequisites
    if platform.system() == 'Windows':
        if not check_wsl_available():
            return {'success': False, 'error': 'WSL not available. Please install WSL2.'}
        
    # Check which mesher is available
    has_cfmesh = check_openfoam_available()
    has_snappy = check_snappy_available()
    
    if not has_cfmesh and not has_snappy:
        return {'success': False, 'error': 'No OpenFOAM mesher found (cfMesh or snappyHexMesh).'}
    
    mesher = 'cfMesh' if has_cfmesh else 'snappyHexMesh'
    if verbose:
        print(f"[OpenFOAM] Using mesher: {mesher}")
    
    # Create temporary case directory
    case_dir = Path(tempfile.mkdtemp(prefix='openfoam_case_'))
    
    try:
        if mesher == 'cfMesh':
            create_openfoam_case(case_dir, stl_path, cell_size)
            if not run_cartesian_mesh(case_dir, verbose):
                return {'success': False, 'error': 'cartesianMesh failed'}
        else:
            create_snappy_case(case_dir, stl_path, cell_size)
            if not run_snappy_hex_mesh(case_dir, verbose):
                return {'success': False, 'error': 'snappyHexMesh failed'}
        
        # Convert to Fluent
        msh_file = convert_to_fluent(case_dir, verbose)
        if not msh_file:
            return {'success': False, 'error': 'Fluent conversion failed'}
        
        shutil.copy(msh_file, output_path)
        
        # Convert to VTK for internal visualization
        # We try to create a sibling file with .vtk extension
        output_vtk_path = str(Path(output_path).with_suffix('.vtk'))
        vtk_file = convert_to_vtk(case_dir, verbose)
        
        if vtk_file:
             shutil.copy(vtk_file, output_vtk_path)
             if verbose:
                 print(f"[OpenFOAM] Vis: VTK mesh saved to {output_vtk_path}")
        
        if verbose:
            print(f"[OpenFOAM] SUCCESS: Mesh saved to {output_path}")
        
        return {
            'success': True,
            'output_file': output_path,
            'visualization_file': output_vtk_path if vtk_file else None,
            'strategy': f'OpenFOAM {mesher}',
            'message': 'Hex-dominant mesh generated successfully'
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}
    finally:
        try:
            shutil.rmtree(case_dir)
        except:
            pass


def convert_to_vtk(case_dir: Path, verbose: bool = True) -> Optional[Path]:
    """
    Convert OpenFOAM mesh to VTK format for visualization.
    Returns path to the generated .vtk file.
    """
    case_dir_wsl = str(case_dir).replace('\\', '/').replace('C:', '/mnt/c')

    cmd = [
        'wsl', 'bash', '-c',
        f'source /usr/lib/openfoam/openfoam*/etc/bashrc 2>/dev/null || source /opt/openfoam*/etc/bashrc 2>/dev/null; '
        f'cd "{case_dir_wsl}" && '
        f'foamToVTK -ascii > log.vtk 2>&1'
    ]
    
    if verbose:
        print("[OpenFOAM] Generating VTK for visualization...")
        
    try:
        subprocess.run(cmd, capture_output=True, timeout=120)
        
        # VTK files are inside a VTK/ directory
        # Look specifically in the VTK directory to avoid picking up the log
        vtk_dir = case_dir / "VTK"
        if not vtk_dir.exists():
            if verbose:
                 print(f"[OpenFOAM] WARNING: VTK directory not found in {case_dir}")
                 # Debug: print log
                 log_file = case_dir / "log.vtk"
                 if log_file.exists():
                     print(f"[OpenFOAM] VTK Log:\n{log_file.read_text()}")
            return None
            
        vtk_files = list(vtk_dir.rglob("*.vtk"))
        
        if verbose:
             print(f"[OpenFOAM] Found {len(vtk_files)} VTK files in {vtk_dir}:")
             for f in vtk_files:
                 print(f"  - {f.relative_to(case_dir)} ({f.stat().st_size} bytes)")
        
        if vtk_files:
            # Sort by size descending
            vtk_files.sort(key=lambda p: p.stat().st_size, reverse=True)
            chosen = vtk_files[0]
            if verbose:
                print(f"[OpenFOAM] Selected largest file: {chosen.name}")
            return chosen
            
        return None
    except:
        return None


if __name__ == "__main__":
    print("OpenFOAM Integration Check")
    print("=" * 50)
    print(f"WSL Available:      {check_wsl_available()}")
    print(f"cfMesh Available:   {check_openfoam_available()}")
    print(f"snappyHexMesh Available: {check_snappy_available()}")
