"""
ANSYS Mechanical Export Module
==============================

Unified API for exporting meshes to ANSYS Mechanical CDB format
using MAPDL's CDWRITE command for guaranteed compatibility.

Usage:
    from core.ansys_export import export_to_ansys_cdb, export_and_open_ansys
    
    # Just export to CDB
    cdb_path = export_to_ansys_cdb(mesh_file, output_dir)
    
    # Export and open in ANSYS Workbench
    export_and_open_ansys(mesh_file, output_dir)
"""

import subprocess
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os

# ANSYS Configuration
ANSYS_VERSIONS = ["v252", "v251", "v241", "v232", "v231"]  # Try these in order
ANSYS_ROOT = Path(r"C:\Program Files\ANSYS Inc")


def find_ansys_installation() -> Tuple[Optional[Path], Optional[str]]:
    """Find ANSYS installation directory and version."""
    for version in ANSYS_VERSIONS:
        root = ANSYS_ROOT / version
        if root.exists():
            return root, version
    return None, None


def find_mapdl_exe() -> Optional[Path]:
    """Find MAPDL executable."""
    root, version = find_ansys_installation()
    if not root:
        return None
    
    # Standard path pattern
    version_num = version[1:]  # Remove 'v' prefix
    mapdl_exe = root / "ansys" / "bin" / "winx64" / f"ANSYS{version_num}.exe"
    
    if mapdl_exe.exists():
        return mapdl_exe
    
    # Try lowercase
    mapdl_exe_lower = root / "ansys" / "bin" / "winx64" / f"ansys{version_num}.exe"
    if mapdl_exe_lower.exists():
        return mapdl_exe_lower
    
    return None


def find_runwb2_exe() -> Optional[Path]:
    """Find ANSYS Workbench executable."""
    root, version = find_ansys_installation()
    if not root:
        return None
    
    runwb2 = root / "Framework" / "bin" / "Win64" / "RunWB2.exe"
    if runwb2.exists():
        return runwb2
    
    return None


def parse_msh_file(msh_path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[int]]]:
    """
    Parse a Gmsh MSH file and extract nodes, elements, and physical groups.
    
    Returns:
        points: (N, 3) array of node coordinates
        elements: (M, 4) or (M, 10) array of element connectivity (0-indexed)
        named_selections: Dict mapping group name -> list of node IDs
    """
    nodes = {}
    elements = []
    physical_groups = {}  # {tag: {'dim': int, 'name': str}}
    
    with open(msh_path, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Parse $PhysicalNames
        if line == "$PhysicalNames":
            i += 1
            if i < len(lines):
                try:
                    num_physical = int(lines[i])
                    i += 1
                    for _ in range(num_physical):
                        if i >= len(lines): break
                        parts = lines[i].split()
                        if len(parts) >= 3:
                            dim = int(parts[0])
                            tag = int(parts[1])
                            name = parts[2].strip('"')
                            physical_groups[tag] = {'dim': dim, 'name': name}
                        i += 1
                except ValueError:
                    pass
            continue
        
        # Parse $Nodes (Gmsh 4.1 format)
        if line == "$Nodes":
            i += 1
            if i < len(lines):
                header = lines[i].split()
                if len(header) >= 2:
                    num_blocks = int(header[0])
                    i += 1
                    for _ in range(num_blocks):
                        if i >= len(lines): break
                        block_header = lines[i].split()
                        if len(block_header) >= 4:
                            num_nodes_in_block = int(block_header[3])
                            i += 1
                            node_ids = []
                            for _ in range(num_nodes_in_block):
                                if i >= len(lines): break
                                node_ids.append(int(lines[i]))
                                i += 1
                            for nid in node_ids:
                                if i >= len(lines): break
                                coords = [float(x) for x in lines[i].split()]
                                if len(coords) >= 3:
                                    nodes[nid] = coords[:3]
                                i += 1
                        else:
                            i += 1
            continue
        
        # Parse $Elements (Gmsh 4.1 format)
        if line == "$Elements":
            i += 1
            if i < len(lines):
                header = lines[i].split()
                if len(header) >= 2:
                    num_blocks = int(header[0])
                    i += 1
                    for _ in range(num_blocks):
                        if i >= len(lines): break
                        block_header = lines[i].split()
                        if len(block_header) >= 4:
                            entity_dim = int(block_header[0])
                            element_type = int(block_header[2])
                            num_elements_in_block = int(block_header[3])
                            i += 1
                            
                            # Only parse 3D elements (tets)
                            if element_type == 4:  # Tet4
                                for _ in range(num_elements_in_block):
                                    if i >= len(lines): break
                                    data = lines[i].split()
                                    if len(data) >= 5:
                                        elem_nodes = [int(x) for x in data[1:5]]
                                        elements.append(elem_nodes)
                                    i += 1
                            elif element_type == 11:  # Tet10
                                for _ in range(num_elements_in_block):
                                    if i >= len(lines): break
                                    data = lines[i].split()
                                    if len(data) >= 11:
                                        elem_nodes = [int(x) for x in data[1:11]]
                                        elements.append(elem_nodes)
                                    i += 1
                            else:
                                i += num_elements_in_block
                        else:
                            i += 1
            continue
        
        i += 1
    
    # Convert to numpy arrays
    if not nodes or not elements:
        raise ValueError(f"Failed to parse mesh file: {msh_path}")
    
    # Create contiguous node array
    node_ids = sorted(nodes.keys())
    node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
    
    points = np.array([nodes[nid] for nid in node_ids])
    
    # Remap element connectivity to 0-indexed
    elements_remapped = []
    for elem in elements:
        remapped = [node_id_to_idx[nid] for nid in elem]
        elements_remapped.append(remapped)
    elements_arr = np.array(elements_remapped)
    
    # For now, create simple named selections from boundary nodes
    named_selections = create_boundary_selections(points)
    
    return points, elements_arr, named_selections


def create_boundary_selections(points: np.ndarray) -> Dict[str, List[int]]:
    """Create named selections based on bounding box faces."""
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    
    tol = 1e-6 * np.linalg.norm(bbox_max - bbox_min)
    
    selections = {}
    
    # X boundaries
    selections['X_MIN'] = np.where(np.abs(points[:, 0] - bbox_min[0]) < tol)[0].tolist()
    selections['X_MAX'] = np.where(np.abs(points[:, 0] - bbox_max[0]) < tol)[0].tolist()
    
    # Y boundaries
    selections['Y_MIN'] = np.where(np.abs(points[:, 1] - bbox_min[1]) < tol)[0].tolist()
    selections['Y_MAX'] = np.where(np.abs(points[:, 1] - bbox_max[1]) < tol)[0].tolist()
    
    # Z boundaries
    selections['Z_MIN'] = np.where(np.abs(points[:, 2] - bbox_min[2]) < tol)[0].tolist()
    selections['Z_MAX'] = np.where(np.abs(points[:, 2] - bbox_max[2]) < tol)[0].tolist()
    
    # Filter out empty selections
    return {k: v for k, v in selections.items() if len(v) > 0}


def generate_apdl_input(
    points: np.ndarray,
    elements: np.ndarray,
    named_selections: Dict[str, List[int]],
    element_type: str = "SOLID187"
) -> str:
    """Generate APDL commands to create mesh."""
    lines = []
    lines.append("/PREP7")
    lines.append("")
    lines.append(f"! Element Type")
    lines.append(f"ET,1,{element_type}")
    lines.append("")
    
    # Nodes
    lines.append("! Nodes")
    for i, (x, y, z) in enumerate(points):
        node_id = i + 1
        lines.append(f"N,{node_id},{x:.10E},{y:.10E},{z:.10E}")
    lines.append("")
    
    # Elements - handle Tet4 and Tet10
    lines.append("! Elements")
    for i, elem_nodes in enumerate(elements):
        elem_id = i + 1
        nodes_1indexed = [n + 1 for n in elem_nodes]
        
        if len(nodes_1indexed) <= 8:
            # EN command supports up to 8 nodes
            node_str = ",".join(str(n) for n in nodes_1indexed)
            lines.append(f"EN,{elem_id},{node_str}")
        else:
            # For Tet10, use E command with TYPE set
            lines.append(f"TYPE,1")
            lines.append(f"*SET,_EN{elem_id}_N1,{nodes_1indexed[0]}")
            lines.append(f"*SET,_EN{elem_id}_N2,{nodes_1indexed[1]}")
            lines.append(f"*SET,_EN{elem_id}_N3,{nodes_1indexed[2]}")
            lines.append(f"*SET,_EN{elem_id}_N4,{nodes_1indexed[3]}")
            lines.append(f"*SET,_EN{elem_id}_N5,{nodes_1indexed[4]}")
            lines.append(f"*SET,_EN{elem_id}_N6,{nodes_1indexed[5]}")
            lines.append(f"*SET,_EN{elem_id}_N7,{nodes_1indexed[6]}")
            lines.append(f"*SET,_EN{elem_id}_N8,{nodes_1indexed[7]}")
            lines.append(f"*SET,_EN{elem_id}_N9,{nodes_1indexed[8]}")
            lines.append(f"*SET,_EN{elem_id}_N10,{nodes_1indexed[9]}")
            lines.append(f"E,_EN{elem_id}_N1,_EN{elem_id}_N2,_EN{elem_id}_N3,_EN{elem_id}_N4,_EN{elem_id}_N5,_EN{elem_id}_N6,_EN{elem_id}_N7,_EN{elem_id}_N8")
            # Continue with remaining nodes
            lines.append(f"EMORE,_EN{elem_id}_N9,_EN{elem_id}_N10")
    lines.append("")
    
    # Named selections
    lines.append("! Named Selections")
    for name, node_ids in named_selections.items():
        lines.append("NSEL,NONE")
        for nid in node_ids:
            lines.append(f"NSEL,A,NODE,,{nid+1}")
        lines.append(f"CM,{name},NODE")
        lines.append("NSEL,ALL")
        lines.append("")
    
    lines.append("FINISH")
    return "\n".join(lines)


def export_to_ansys_cdb(
    msh_file: str,
    output_dir: str,
    output_name: str = None,
    element_type: str = "tet4",
    custom_zones: Dict[str, List[int]] = None,
    verbose: bool = True
) -> Optional[str]:
    """
    Export mesh to ANSYS CDB format using MAPDL's CDWRITE command.
    
    Args:
        msh_file: Path to input .msh file
        output_dir: Directory for output files
        output_name: Base name for output (default: input file name)
        element_type: "tet4" or "tet10"
        custom_zones: Dict of zone_name -> list of node/face IDs (optional)
        verbose: Print progress messages
    
    Returns:
        Path to generated CDB file, or None if failed
    """
    msh_path = Path(msh_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if output_name is None:
        output_name = msh_path.stem
    
    # Find MAPDL
    mapdl_exe = find_mapdl_exe()
    if not mapdl_exe:
        if verbose:
            print("[ANSYS Export] ERROR: Could not find MAPDL installation")
        return None
    
    if verbose:
        print(f"[ANSYS Export] MAPDL: {mapdl_exe}")
    
    # Parse mesh
    if verbose:
        print(f"[ANSYS Export] Parsing {msh_path.name}...")
    
    try:
        points, elements, named_selections = parse_msh_file(str(msh_file))
    except Exception as e:
        if verbose:
            print(f"[ANSYS Export] ERROR parsing mesh: {e}")
        return None
    
    if verbose:
        print(f"[ANSYS Export] Nodes: {len(points)}, Elements: {len(elements)}")
    
    # Merge custom zones with auto-generated boundary selections
    if custom_zones:
        for zone_name, zone_ids in custom_zones.items():
            # Sanitize zone name for ANSYS (max 32 chars, no special chars)
            safe_name = "".join(c if c.isalnum() or c == '_' else '_' for c in zone_name)[:32]
            named_selections[safe_name] = zone_ids
        if verbose:
            print(f"[ANSYS Export] Added {len(custom_zones)} custom zones")
    
    # Determine element type
    ansys_elem = "SOLID285" if element_type.lower() == "tet4" else "SOLID187"
    
    # Generate APDL input
    apdl_content = generate_apdl_input(points, elements, named_selections, ansys_elem)
    
    inp_file = output_path / f"{output_name}_mesh.inp"
    with open(inp_file, 'w') as f:
        f.write(apdl_content)
    
    if verbose:
        print(f"[ANSYS Export] Wrote APDL input: {inp_file}")
    
    # Create batch script for CDWRITE
    batch_content = f"""/PREP7
/INPUT,{output_name}_mesh,inp,'{str(output_path).replace(chr(92), '/')}',, 0
CDWRITE,DB,{output_name},cdb
FINI
/EXIT,NOSAVE
"""
    
    batch_file = output_path / f"{output_name}_convert.inp"
    with open(batch_file, 'w') as f:
        f.write(batch_content)
    
    # Run MAPDL
    if verbose:
        print("[ANSYS Export] Running MAPDL...")
    
    cmd = [
        str(mapdl_exe),
        "-b", "nolist",
        "-i", f"{output_name}_convert.inp",
        "-o", f"{output_name}_convert.out"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(output_path),
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        cdb_file = output_path / f"{output_name}.cdb"
        
        if cdb_file.exists():
            if verbose:
                print(f"[ANSYS Export] SUCCESS: {cdb_file}")
                print(f"[ANSYS Export] File size: {cdb_file.stat().st_size:,} bytes")
            return str(cdb_file)
        else:
            if verbose:
                print("[ANSYS Export] ERROR: CDB file not created")
                out_file = output_path / f"{output_name}_convert.out"
                if out_file.exists():
                    print(out_file.read_text()[-1000:])
            return None
            
    except subprocess.TimeoutExpired:
        if verbose:
            print("[ANSYS Export] ERROR: MAPDL timeout")
        return None
    except Exception as e:
        if verbose:
            print(f"[ANSYS Export] ERROR: {e}")
        return None


def generate_wbjn_import_script(cdb_path: str, output_path: str) -> str:
    """Generate Workbench journal script to import CDB."""
    cdb_path_fwd = cdb_path.replace("\\", "/")
    
    script = f'''# encoding: utf-8
# Auto-generated ANSYS Workbench import script

SetScriptVersion(Version="25.2")

# Create External Model system
template1 = GetTemplate(TemplateName="External Model")
system1 = template1.CreateSystem()

# Add CDB file
setup1 = system1.GetContainer(ComponentName="Setup")
imported_file = setup1.AddDataFile(FilePath="{cdb_path_fwd}")

# Update system
system1.Update(AllDependencies=True)

# Open Mechanical
model_container = system1.GetContainer(ComponentName="Model")
model_container.Edit()

print("CDB file imported successfully!")
'''
    
    wbjn_path = Path(output_path)
    with open(wbjn_path, 'w') as f:
        f.write(script)
    
    return str(wbjn_path)


def export_and_open_ansys(
    msh_file: str,
    output_dir: str,
    output_name: str = None,
    element_type: str = "tet4",
    verbose: bool = True
) -> bool:
    """
    Export mesh to CDB and open directly in ANSYS Workbench.
    
    Args:
        msh_file: Path to input .msh file
        output_dir: Directory for output files
        output_name: Base name for output
        element_type: "tet4" or "tet10"
        verbose: Print progress
    
    Returns:
        True if launched successfully
    """
    # First export to CDB
    cdb_path = export_to_ansys_cdb(msh_file, output_dir, output_name, element_type, verbose)
    
    if not cdb_path:
        return False
    
    # Find Workbench
    runwb2 = find_runwb2_exe()
    if not runwb2:
        if verbose:
            print("[ANSYS Export] ERROR: Could not find ANSYS Workbench")
        return False
    
    # Generate import script
    output_path = Path(output_dir)
    wbjn_path = output_path / f"{output_name or 'mesh'}_import.wbjn"
    generate_wbjn_import_script(cdb_path, str(wbjn_path))
    
    if verbose:
        print(f"[ANSYS Export] Launching ANSYS Workbench...")
    
    # Launch Workbench (non-blocking)
    cmd = [str(runwb2), "-R", str(wbjn_path)]
    
    try:
        subprocess.Popen(cmd, cwd=str(output_path))
        if verbose:
            print("[ANSYS Export] Workbench launched!")
        return True
    except Exception as e:
        if verbose:
            print(f"[ANSYS Export] ERROR launching Workbench: {e}")
        return False


# Quick test
if __name__ == "__main__":
    print("ANSYS Export Module")
    print("=" * 40)
    
    root, version = find_ansys_installation()
    if root:
        print(f"Found ANSYS: {version} at {root}")
    else:
        print("ANSYS not found")
    
    mapdl = find_mapdl_exe()
    if mapdl:
        print(f"MAPDL: {mapdl}")
    
    runwb2 = find_runwb2_exe()
    if runwb2:
        print(f"Workbench: {runwb2}")
