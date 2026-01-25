"""
ANSYS Mechanical CDB Exporter

Exports tetrahedral meshes to ANSYS Mechanical CDB (APDL archive) format.
This format is directly importable into ANSYS Workbench via "External Model".

Supports:
- SOLID187 (10-node quadratic tetrahedra) - preferred for structural
- SOLID285 (4-node linear tetrahedra)
- Named selections (CMBLOCK) for boundary conditions

Usage:
    from core.export_mechanical_cdb import export_mechanical_cdb
    
    export_mechanical_cdb(
        filename="output.cdb",
        points=node_coords,      # (N, 3) array
        elements=tet_connectivity,  # (M, 4) or (M, 10) array
        named_selections={"wall": [1,2,3], "inlet": [4,5,6]}
    )

Author: MeshPackage Team
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field


# =============================================================================
# CONSTANTS
# =============================================================================

# ANSYS Element Types for structural analysis
ELEMENT_TYPES = {
    "tet4": ("SOLID285", 4),   # 4-node linear tetrahedron
    "tet10": ("SOLID187", 10), # 10-node quadratic tetrahedron (preferred)
    "hex8": ("SOLID185", 8),   # 8-node linear hexahedron
    "hex20": ("SOLID186", 20), # 20-node quadratic hexahedron
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class MechanicalMeshData:
    """Container for mesh data to be exported."""
    points: np.ndarray  # (N, 3) node coordinates
    elements: np.ndarray  # (M, nodes_per_elem) connectivity
    element_type: str = "tet10"  # tet4, tet10, hex8, hex20
    named_selections: Dict[str, List[int]] = field(default_factory=dict)
    material_id: int = 1


# =============================================================================
# TET10 GENERATION (Quadratic from Linear)
# =============================================================================

def create_tet10_from_tet4(points: np.ndarray, tets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert linear tetrahedra (Tet4) to quadratic tetrahedra (Tet10).
    
    For each tet, we add 6 mid-edge nodes:
    - Node 4: midpoint of edge 0-1
    - Node 5: midpoint of edge 1-2
    - Node 6: midpoint of edge 0-2
    - Node 7: midpoint of edge 0-3
    - Node 8: midpoint of edge 1-3
    - Node 9: midpoint of edge 2-3
    
    Args:
        points: (N, 3) array of node coordinates
        tets: (M, 4) array of tet4 connectivity (0-indexed)
        
    Returns:
        new_points: Extended points array with mid-edge nodes
        tet10s: (M, 10) array of tet10 connectivity
    """
    num_orig_nodes = len(points)
    num_tets = len(tets)
    
    # Edge definitions for a tetrahedron (local node indices)
    edges = [
        (0, 1),  # Edge 0 -> mid-node 4
        (1, 2),  # Edge 1 -> mid-node 5
        (0, 2),  # Edge 2 -> mid-node 6
        (0, 3),  # Edge 3 -> mid-node 7
        (1, 3),  # Edge 4 -> mid-node 8
        (2, 3),  # Edge 5 -> mid-node 9
    ]
    
    # Map: (min_node, max_node) -> mid_node_id
    edge_to_midnode = {}
    new_points_list = [points]
    next_node_id = num_orig_nodes
    
    tet10s = np.zeros((num_tets, 10), dtype=np.int32)
    
    for ti, tet in enumerate(tets):
        # Copy corner nodes
        tet10s[ti, :4] = tet
        
        # Create/reuse mid-edge nodes
        for ei, (li, lj) in enumerate(edges):
            ni, nj = tet[li], tet[lj]
            edge_key = (min(ni, nj), max(ni, nj))
            
            if edge_key not in edge_to_midnode:
                # Create new mid-node
                mid_point = (points[ni] + points[nj]) / 2.0
                new_points_list.append(mid_point.reshape(1, 3))
                edge_to_midnode[edge_key] = next_node_id
                next_node_id += 1
            
            tet10s[ti, 4 + ei] = edge_to_midnode[edge_key]
    
    # Concatenate all points
    all_points = np.vstack(new_points_list)
    
    print(f"[Tet10] Converted {num_tets} Tet4 -> Tet10")
    print(f"[Tet10] Added {len(edge_to_midnode)} mid-edge nodes ({num_orig_nodes} -> {len(all_points)})")
    
    return all_points, tet10s


# =============================================================================
# CDB FILE WRITER
# =============================================================================

def export_mechanical_cdb(
    filename: str,
    points: np.ndarray,
    elements: np.ndarray,
    element_type: str = "tet10",
    named_selections: Optional[Dict[str, List[int]]] = None,
    material_id: int = 1,
    verbose: bool = True
) -> bool:
    """
    Export mesh to ANSYS Mechanical CDB (blocked) format.
    
    NOTE: Material properties (MP/MPDATA) are omitted to ensure compatibility 
    with ANSYS Workbench's "External Model" parser, which is strict and often 
    fails on material blocks. Materials should be assigned in ANSYS Mechanical.
    
    Args:
        filename: Output file path (.cdb)
        points: (N, 3) array of node coordinates
        elements: (M, 4) or (M, 10) array of element connectivity (0-indexed)
        element_type: "tet4", "tet10", "hex8", or "hex20"
        named_selections: Dict mapping selection name -> list of node IDs
        material_id: Material ID to assign (links to material in ANSYS)
        verbose: Print progress messages
        
    Returns:
        True if export successful
    """
    if named_selections is None:
        named_selections = {}
    
    # Validate element type
    if element_type not in ELEMENT_TYPES:
        raise ValueError(f"Unknown element type: {element_type}. Use: {list(ELEMENT_TYPES.keys())}")
    
    ansys_elem_name, nodes_per_elem = ELEMENT_TYPES[element_type]
    
    # Convert Tet4 to Tet10 if requested
    if element_type == "tet10" and elements.shape[1] == 4:
        if verbose:
            print("[CDB Export] Converting Tet4 to Tet10...")
        points, elements = create_tet10_from_tet4(points, elements)
    
    # Verify element shape
    if elements.shape[1] != nodes_per_elem:
        raise ValueError(f"Element type {element_type} requires {nodes_per_elem} nodes, got {elements.shape[1]}")
    
    num_nodes = len(points)
    num_elements = len(elements)
    
    if verbose:
        print(f"[CDB Export] Writing {filename}")
        print(f"[CDB Export] Nodes: {num_nodes}, Elements: {num_elements}, Type: {ansys_elem_name}")
    
    with open(filename, 'w') as f:
        # Header
        f.write("/COM,ANSYS CDB file exported by MeshPackage\n")
        f.write("/COM,Element Type: {}\n".format(ansys_elem_name))
        f.write("/PREP7\n")
        f.write("\n")
        
        # Element Type Definition (CRITICAL)
        f.write("! Element Type\n")
        f.write(f"ET,1,{ansys_elem_name}\n")
        f.write("\n")
        
        # NOTE: Materials skipped for "External Model" compatibility.
        # Assign materials in ANSYS Mechanical.
        
        # Node Block (Blocked Format)
        f.write("! Node Definitions (Blocked Format)\n")
        f.write(f"NBLOCK,6,SOLID,{num_nodes},{num_nodes}\n")
        f.write("(3i9,6e20.13)\n")  # Format specification
        
        for i, pt in enumerate(points):
            node_id = i + 1  # 1-indexed for ANSYS
            # Format: node_id, 0, 0, x, y, z
            f.write(f"{node_id:9d}{0:9d}{0:9d}{pt[0]:20.13E}{pt[1]:20.13E}{pt[2]:20.13E}\n")
        
        f.write("-1\n")  # End marker
        f.write("\n")
        
        # Element Block
        f.write("! Element Definitions (Blocked Format)\n")
        
        if element_type in ("tet4", "tet10"):
            # Tetrahedral format
            if element_type == "tet10":
                # SOLID187: 10 attributes + 10 nodes = 20 integers
                # Use (10i9) format: 2 lines of 10 integers each
                f.write(f"EBLOCK,19,SOLID,{num_elements},{num_elements}\n")
                f.write("(10i9)\n")  
                
                for i, elem in enumerate(elements):
                    elem_id = i + 1
                    # Format: mat, type, real, sec, esys, death, solidmodel, shape, num_nodes, elem_id, nodes...
                    # For SOLID187: shape=4 (tet), num_nodes=10
                    nodes_1indexed = elem + 1  # Convert to 1-indexed
                    
                    # Line 1: 10 attributes
                    # mat, type, real, sec, esys, death, solidmodel, shape, num_nodes, elem_id
                    f.write(f"{material_id:9d}{1:9d}{1:9d}{1:9d}{0:9d}{0:9d}{0:9d}{4:9d}{10:9d}{elem_id:9d}\n")
                    
                    # Line 2: 10 nodes
                    for n in nodes_1indexed:
                        f.write(f"{n:9d}")
                    f.write("\n")
            else:
                # Tet4 (SOLID285)
                # 10 attributes + 4 nodes = 14 integers
                # Use (14i9) format: 1 line of 14 integers
                f.write(f"EBLOCK,19,SOLID,{num_elements},{num_elements}\n")
                f.write("(14i9)\n")
                
                for i, elem in enumerate(elements):
                    elem_id = i + 1
                    nodes_1indexed = elem + 1
                    # For SOLID285: shape=4 (tet), num_nodes=4
                    # mat, type, real, sec, esys, death, solidmodel, shape, num_nodes, elem_id, nodes...
                    f.write(f"{material_id:9d}{1:9d}{1:9d}{1:9d}{0:9d}{0:9d}{0:9d}{4:9d}{4:9d}{elem_id:9d}")
                    for n in nodes_1indexed:
                        f.write(f"{n:9d}")
                    f.write("\n")
        
        f.write("-1\n")  # End marker
        f.write("\n")
        
        # Named Selections (Component Blocks)
        if named_selections:
            f.write("! Named Selections (for boundary conditions)\n")
            for name, node_ids in named_selections.items():
                if not node_ids:
                    continue
                    
                # Sanitize name (ANSYS has length limits)
                clean_name = name.upper()[:32].replace(" ", "_")
                
                f.write(f"! Selection: {name}\n")
                f.write(f"CMBLOCK,{clean_name},NODE,{len(node_ids)}\n")
                f.write("(8i10)\n")  # Format: 8 integers per line
                
                # Write node IDs in groups of 8
                nodes_1indexed = [n + 1 for n in node_ids]  # Convert to 1-indexed
                for j in range(0, len(nodes_1indexed), 8):
                    chunk = nodes_1indexed[j:j+8]
                    f.write("".join(f"{n:10d}" for n in chunk) + "\n")
                
                f.write("\n")
        
        # Footer
        f.write("FINISH\n")
    
    if verbose:
        print(f"[CDB Export] Successfully wrote {filename}")
        print(f"[CDB Export] Named selections: {list(named_selections.keys())}")
    
    return True


# =============================================================================
# SAMPLE MESH GENERATOR (for testing)
# =============================================================================

def create_sample_box_mesh_tet4(
    nx: int = 3, ny: int = 3, nz: int = 3,
    size: Tuple[float, float, float] = (10.0, 10.0, 10.0)
) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[int]]]:
    """
    Create a simple structured box mesh of linear tetrahedra.
    Returns boundary node selections for each face.
    
    Args:
        nx, ny, nz: Number of divisions in each direction
        size: Box dimensions (Lx, Ly, Lz)
        
    Returns:
        points: (N, 3) array of node coordinates
        tets: (M, 4) array of tetrahedral connectivity
        named_selections: Dict of boundary face -> node IDs
    """
    Lx, Ly, Lz = size
    
    # Generate grid points
    x = np.linspace(0, Lx, nx + 1)
    y = np.linspace(0, Ly, ny + 1)
    z = np.linspace(0, Lz, nz + 1)
    
    # Create node grid
    points = []
    node_index = {}
    
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                node_index[(i, j, k)] = len(points)
                points.append([x[i], y[j], z[k]])
    
    points = np.array(points)
    
    # Create tetrahedra from hexahedral cells
    # Each hex is split into 6 tetrahedra
    tets = []
    
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                # Get 8 corners of hex cell
                n0 = node_index[(i, j, k)]
                n1 = node_index[(i+1, j, k)]
                n2 = node_index[(i+1, j+1, k)]
                n3 = node_index[(i, j+1, k)]
                n4 = node_index[(i, j, k+1)]
                n5 = node_index[(i+1, j, k+1)]
                n6 = node_index[(i+1, j+1, k+1)]
                n7 = node_index[(i, j+1, k+1)]
                
                # Split into 6 tets (consistent pattern)
                tets.append([n0, n1, n3, n4])
                tets.append([n1, n2, n3, n6])
                tets.append([n1, n3, n4, n6])
                tets.append([n3, n4, n6, n7])
                tets.append([n1, n4, n5, n6])
                tets.append([n1, n3, n6, n4])
    
    tets = np.array(tets)
    
    # Create boundary named selections
    tol = 1e-6
    named_selections = {
        "X_MIN": [],  # Inlet / Fixed support
        "X_MAX": [],  # Outlet / Force application
        "Y_MIN": [],
        "Y_MAX": [],
        "Z_MIN": [],  # Bottom / Fixed
        "Z_MAX": [],  # Top / Pressure
    }
    
    for i, pt in enumerate(points):
        if abs(pt[0]) < tol:
            named_selections["X_MIN"].append(i)
        if abs(pt[0] - Lx) < tol:
            named_selections["X_MAX"].append(i)
        if abs(pt[1]) < tol:
            named_selections["Y_MIN"].append(i)
        if abs(pt[1] - Ly) < tol:
            named_selections["Y_MAX"].append(i)
        if abs(pt[2]) < tol:
            named_selections["Z_MIN"].append(i)
        if abs(pt[2] - Lz) < tol:
            named_selections["Z_MAX"].append(i)
    
    print(f"[Sample Mesh] Created {len(points)} nodes, {len(tets)} Tet4 elements")
    for name, nodes in named_selections.items():
        print(f"[Sample Mesh] Selection '{name}': {len(nodes)} nodes")
    
    return points, tets, named_selections


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Generate a sample mesh with named boundary selections for testing."""
    
    print("=" * 60)
    print("ANSYS Mechanical CDB Export Test")
    print("=" * 60)
    
    # Create sample mesh
    points, tets, named_selections = create_sample_box_mesh_tet4(
        nx=5, ny=5, nz=5,
        size=(100.0, 100.0, 100.0)  # 100mm cube
    )
    
    # Export as Tet10 (quadratic)
    output_file = Path(__file__).parent.parent / "apps" / "cli" / "generated_meshes" / "test_mechanical_tet10.cdb"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    export_mechanical_cdb(
        filename=str(output_file),
        points=points,
        elements=tets,
        element_type="tet10",  # Will auto-convert from Tet4
        named_selections=named_selections,
        material_id=1,
        verbose=True
    )
    
    print("=" * 60)
    print(f"Output: {output_file}")
    print("=" * 60)
    print("\nTo test in ANSYS Workbench:")
    print("1. Create 'External Model' system")
    print("2. Import this .cdb file (uncheck 'Blocked CDB' if needed, though strictly it is blocked)")
    print("3. Connect to 'Static Structural' system")
    print("4. Update Project (Right click Setup -> Update)")
    print("5. Open 'Model' cell in Mechanical")
    print("6. Assign material (e.g. Structural Steel) to the body in 'Geometry' branch")


if __name__ == "__main__":
    main()
