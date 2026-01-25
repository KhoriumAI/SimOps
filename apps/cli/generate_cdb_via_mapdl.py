"""
ANSYS MAPDL-Based CDB Generator
===============================

Uses ANSYS MAPDL's CDWRITE command to generate properly formatted CDB files.
This guarantees compatibility with ANSYS Workbench External Model.

Workflow:
1. Generate an APDL input file (.inp) with mesh data
2. Run MAPDL to execute the file and call CDWRITE,DB
3. MAPDL outputs a valid blocked CDB file

Usage: python generate_cdb_via_mapdl.py
"""

import subprocess
import os
import sys
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from core.export_mechanical_cdb import create_sample_box_mesh_tet4, create_tet10_from_tet4

# Configuration
ANSYS_VERSION = "v252"
ANSYS_ROOT = Path(r"C:\Program Files\ANSYS Inc") / ANSYS_VERSION
MAPDL_EXE = ANSYS_ROOT / "ansys" / "bin" / "winx64" / f"ANSYS{ANSYS_VERSION[1:]}.exe"

OUTPUT_DIR = script_dir / "generated_meshes"


def generate_apdl_input(points, elements, named_selections, element_type="SOLID187"):
    """
    Generate APDL commands to create mesh and export to CDB.
    
    Uses simple N (node) and EN (element by nodes) commands which are 
    universally compatible with MAPDL.
    """
    lines = []
    lines.append("/PREP7")
    lines.append("")
    
    # Element type
    lines.append("! Element Type")
    lines.append(f"ET,1,{element_type}")
    lines.append("")
    
    # Nodes (use simple N command - most compatible)
    lines.append("! Nodes")
    for i, (x, y, z) in enumerate(points):
        node_id = i + 1  # 1-indexed
        lines.append(f"N,{node_id},{x:.10E},{y:.10E},{z:.10E}")
    lines.append("")
    
    # Elements (use EN command - element by node list)
    lines.append("! Elements")
    for i, elem_nodes in enumerate(elements):
        elem_id = i + 1  # 1-indexed
        nodes_1indexed = [n + 1 for n in elem_nodes]  # Convert to 1-indexed
        node_str = ",".join(str(n) for n in nodes_1indexed)
        lines.append(f"EN,{elem_id},{node_str}")
    lines.append("")
    
    # Named selections (component definitions)
    lines.append("! Named Selections")
    for name, node_ids in named_selections.items():
        # Create node component
        lines.append(f"*DIM,_{name}_NODES,ARRAY,{len(node_ids)}")
        for idx, nid in enumerate(node_ids):
            lines.append(f"_{name}_NODES({idx+1})={nid+1}")  # 1-indexed
        
        # Select nodes and create component
        lines.append("NSEL,NONE")
        for nid in node_ids:
            lines.append(f"NSEL,A,NODE,,{nid+1}")  # Add each node
        lines.append(f"CM,{name},NODE")
        lines.append("NSEL,ALL")
        lines.append("")
    
    lines.append("FINISH")
    lines.append("")
    
    return "\n".join(lines)


def generate_mapdl_batch_script(inp_basename, cdb_basename, work_dir):
    """
    Generate the MAPDL batch script that:
    1. Reads our input file
    2. Writes CDB using CDWRITE,DB
    """
    # Use forward slashes for MAPDL path
    work_dir_fwd = str(work_dir).replace("\\", "/")
    
    lines = []
    lines.append("/PREP7")
    lines.append(f"/INPUT,{inp_basename},inp,'{work_dir_fwd}',, 0")
    lines.append(f"CDWRITE,DB,{cdb_basename},cdb")
    lines.append("FINI")
    lines.append("/EXIT,NOSAVE")
    
    return "\n".join(lines)


def run_mapdl_conversion(inp_file, cdb_file, work_dir):
    """Run MAPDL to convert inp to cdb"""
    
    print(f"[MAPDL] MAPDL executable: {MAPDL_EXE}")
    
    if not MAPDL_EXE.exists():
        print(f"[MAPDL] ERROR: MAPDL not found at {MAPDL_EXE}")
        # Try alternative paths
        alt_paths = [
            ANSYS_ROOT / "ansys" / "bin" / "winx64" / "ANSYS252.exe",
            ANSYS_ROOT / "ansys" / "bin" / "winx64" / "ansys252.exe",
        ]
        for alt in alt_paths:
            if alt.exists():
                print(f"[MAPDL] Found at: {alt}")
                mapdl_exe = alt
                break
        else:
            print("[MAPDL] ERROR: Cannot find MAPDL executable")
            return False
    else:
        mapdl_exe = MAPDL_EXE
    
    # Create batch file
    batch_basename = "convert_batch"
    batch_file = work_dir / f"{batch_basename}.inp"
    
    inp_basename = inp_file.stem
    cdb_basename = cdb_file.stem
    
    batch_content = generate_mapdl_batch_script(inp_basename, cdb_basename, work_dir)
    
    with open(batch_file, 'w') as f:
        f.write(batch_content)
    
    print(f"[MAPDL] Batch file: {batch_file}")
    print(f"[MAPDL] Running MAPDL...")
    
    # Run MAPDL
    cmd = [
        str(mapdl_exe),
        "-b", "nolist",  # Batch mode
        "-i", f"{batch_basename}.inp",
        "-o", f"{batch_basename}.out"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=120
        )
        
        print(f"[MAPDL] Exit code: {result.returncode}")
        
        # Check if CDB was created
        expected_cdb = work_dir / f"{cdb_basename}.cdb"
        if expected_cdb.exists():
            print(f"[MAPDL] SUCCESS: Created {expected_cdb}")
            print(f"[MAPDL] File size: {expected_cdb.stat().st_size} bytes")
            return True
        else:
            print(f"[MAPDL] ERROR: CDB file not created")
            # Print output log
            out_file = work_dir / f"{batch_basename}.out"
            if out_file.exists():
                print("[MAPDL] Output log:")
                print(out_file.read_text()[-2000:])  # Last 2000 chars
            return False
            
    except subprocess.TimeoutExpired:
        print("[MAPDL] ERROR: Timeout")
        return False
    except Exception as e:
        print(f"[MAPDL] ERROR: {e}")
        return False


def main():
    print("=" * 60)
    print("ANSYS MAPDL-Based CDB Generator")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create sample mesh
    print("[Test] Creating sample mesh...")
    points, tets, named_selections = create_sample_box_mesh_tet4(
        nx=3, ny=3, nz=3,
        size=(50, 50, 50)
    )
    
    # Convert to Tet10
    print("[Test] Converting to Tet10...")
    points, tets = create_tet10_from_tet4(points, tets)
    print(f"[Test] Mesh: {len(points)} nodes, {len(tets)} elements")
    
    # Generate APDL input
    print("[Test] Generating APDL input file...")
    apdl_content = generate_apdl_input(points, tets, named_selections, "SOLID187")
    
    inp_file = OUTPUT_DIR / "mesh_input.inp"
    with open(inp_file, 'w') as f:
        f.write(apdl_content)
    print(f"[Test] Wrote: {inp_file}")
    print(f"[Test] File size: {inp_file.stat().st_size} bytes")
    
    # Run MAPDL conversion
    cdb_file = OUTPUT_DIR / "mesh_mapdl"  # .cdb extension added by CDWRITE
    success = run_mapdl_conversion(inp_file, cdb_file, OUTPUT_DIR)
    
    print("=" * 60)
    if success:
        print("RESULT: CDB file generated successfully via MAPDL")
        print(f"Output: {OUTPUT_DIR / 'mesh_mapdl.cdb'}")
    else:
        print("RESULT: MAPDL conversion failed")
    print("=" * 60)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
