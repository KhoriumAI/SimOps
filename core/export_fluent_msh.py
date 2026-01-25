"""
Fluent MSH Exporter with Named Boundary Layers

Exports tetrahedral meshes to ANSYS Fluent TGrid MSH format with properly
named boundary zones that Fluent can import without errors.

Usage:
    python export_fluent_msh.py                    # Generate sample mesh
    python export_fluent_msh.py input.msh output.msh  # Convert existing mesh

Author: MeshPackageLean
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from pathlib import Path


# =============================================================================
# FLUENT BOUNDARY TYPE CODES
# =============================================================================
# Reference: Fluent MSH file format specification

BOUNDARY_TYPE_CODES = {
    "interior": 2,
    "wall": 3,
    "pressure-inlet": 4,
    "pressure-outlet": 5,
    "symmetry": 7,
    "periodic-shadow": 8,
    "pressure-far-field": 9,
    "velocity-inlet": 10,
    "periodic": 12,
    "fan": 14,
    "porous-jump": 14,
    "radiator": 14,
    "mass-flow-inlet": 20,
    "interface": 24,
    "outflow": 36,
    "axis": 37,
}

# Cell element type codes for Section 12
CELL_TYPE_CODES = {
    "mixed": 0,
    "triangular": 1,
    "tetrahedral": 2,
    "quadrilateral": 3,
    "hexahedral": 4,
    "pyramid": 5,
    "wedge": 6,
    "polyhedral": 7,
}

# Face element type codes for Section 13
FACE_TYPE_CODES = {
    "mixed": 0,
    "linear": 2,
    "triangular": 3,
    "quadrilateral": 4,
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class FluentBoundaryZone:
    """Defines a boundary zone with its type, name, and associated faces."""
    zone_id: int
    zone_type: str  # e.g., "wall", "velocity-inlet", "pressure-outlet"
    zone_name: str  # user-defined name, e.g., "inlet-1", "outlet-main"
    face_indices: List[int] = field(default_factory=list)
    
    @property
    def bc_type_code(self) -> int:
        """Get the integer boundary condition type code for this zone."""
        return BOUNDARY_TYPE_CODES.get(self.zone_type, 3)  # Default to wall


@dataclass
class FluentMeshData:
    """Container for all mesh data to be exported."""
    points: np.ndarray  # (N, 3) array of node coordinates
    cells: np.ndarray   # (M, 4) array of tetrahedral connectivity (0-indexed)
    boundary_zones: List[FluentBoundaryZone] = field(default_factory=list)
    cell_zone_name: str = "fluid"
    
    
# =============================================================================
# FACE EXTRACTION AND CLASSIFICATION
# =============================================================================

def extract_tet_faces(tets: np.ndarray) -> Tuple[Dict, List]:
    """
    Extract all faces from tetrahedra and identify boundary vs interior faces.
    
    Args:
        tets: (N, 4) array of tetrahedral connectivity (0-indexed)
        
    Returns:
        face_map: dict mapping sorted face tuple -> [nodes, [cell_ids]]
        all_faces: list of all faces with their cell adjacency
    """
    # Local face indices for a tetrahedron (using consistent ordering)
    # Face ordering ensures outward normal when viewed from outside cell
    tet_face_indices = [
        [0, 2, 1],  # Face opposite to vertex 3
        [0, 1, 3],  # Face opposite to vertex 2
        [0, 3, 2],  # Face opposite to vertex 1  
        [1, 2, 3],  # Face opposite to vertex 0
    ]
    
    face_map = {}
    
    for cell_id, tet in enumerate(tets):
        for face_idx in tet_face_indices:
            # Get actual node indices for this face
            face_nodes = tuple(tet[i] for i in face_idx)
            
            # Create a canonical key (sorted) for face lookup
            key = tuple(sorted(face_nodes))
            
            if key not in face_map:
                face_map[key] = {
                    'nodes': list(face_nodes),
                    'cells': [cell_id],
                }
            else:
                face_map[key]['cells'].append(cell_id)
    
    return face_map


def compute_face_normal(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Compute face normal using cross product."""
    v1 = p1 - p0
    v2 = p2 - p0
    return np.cross(v1, v2)


def orient_face_outward(
    face_nodes: List[int], 
    cell_id: int, 
    points: np.ndarray, 
    tets: np.ndarray
) -> List[int]:
    """
    Orient face nodes so that the normal points outward from the cell.
    
    Fluent convention: Face normal points FROM owner cell (c0) TOWARD neighbor (c1).
    For boundary faces (c1=0), normal should point outward from c0.
    """
    # Get the cell's centroid
    cell_pts = points[tets[cell_id]]
    cell_centroid = np.mean(cell_pts, axis=0)
    
    # Get face vertices and compute normal
    p0, p1, p2 = points[face_nodes[0]], points[face_nodes[1]], points[face_nodes[2]]
    face_center = (p0 + p1 + p2) / 3.0
    normal = compute_face_normal(p0, p1, p2)
    
    # Vector from face center to cell centroid
    to_cell = cell_centroid - face_center
    
    # Fluent wants normal pointing AWAY from owner cell
    # If normal points toward cell (dot product > 0), keep current winding
    # If normal points away from cell (dot product < 0), reverse winding
    if np.dot(normal, to_cell) < 0:
        return [face_nodes[0], face_nodes[2], face_nodes[1]]  # Reverse
    
    return face_nodes


def classify_boundary_faces(
    face_map: Dict,
    points: np.ndarray,
    classifier: Optional[Callable[[np.ndarray], str]] = None
) -> Dict[str, List[Tuple]]:
    """
    Classify boundary faces into named zones.
    
    Args:
        face_map: Output from extract_tet_faces
        points: Node coordinates
        classifier: Optional function that takes face centroid and returns zone name
                   If None, all boundary faces go to "wall"
    
    Returns:
        Dict mapping zone_name -> list of (face_nodes, owner_cell)
    """
    zones = {}
    
    for key, data in face_map.items():
        cells = data['cells']
        nodes = data['nodes']
        
        if len(cells) == 1:
            # Boundary face
            owner_cell = cells[0]
            
            if classifier is not None:
                # Use classifier to determine zone
                face_pts = points[nodes]
                centroid = np.mean(face_pts, axis=0)
                zone_name = classifier(centroid)
            else:
                zone_name = "wall"
            
            if zone_name not in zones:
                zones[zone_name] = []
            
            zones[zone_name].append((nodes, owner_cell, 0))  # 0 = no neighbor cell
    
    return zones


# =============================================================================
# MSH FILE WRITER
# =============================================================================

def export_fluent_msh(
    filename: str,
    points: np.ndarray,
    tets: np.ndarray,
    boundary_classifier: Optional[Callable[[np.ndarray], str]] = None,
    boundary_zone_types: Optional[Dict[str, str]] = None,
    zone_assignments: Optional[Dict[Tuple[int, ...], str]] = None,
    cell_zone_name: str = "fluid",
    user_zones: Optional[Dict[str, List[int]]] = None,
    boundary_lookup: Optional[Dict[tuple, str]] = None,
    verbose: bool = True
) -> bool:
    """
    Export a tetrahedral mesh to ANSYS Fluent MSH format with named boundary zones.
    
    Args:
        filename: Output file path
        points: (N, 3) array of node coordinates
        tets: (M, 4) array of tetrahedral connectivity (0-indexed)
        boundary_classifier: Function that takes face centroid (3,) and returns zone name string
        boundary_zone_types: Dict mapping zone name -> boundary type (e.g., "inlet" -> "velocity-inlet")
        boundary_zone_types: Dict mapping zone name -> boundary type (e.g., "inlet" -> "velocity-inlet")
        zone_assignments: Optional[Dict[Tuple[int, ...], str]] = None,
        cell_zone_name: Name for the cell zone (default: "fluid")
        user_zones: Dict mapping zone name -> list of face indices (0-indexed based on boundary extraction)
        boundary_lookup: Dict mapping sorted face node tuple -> zone name (Exact matching)
        verbose: Print progress messages
    
    Returns:
        True if export successful
    """
    if verbose:
        print(f"[FluentExport] Exporting to {filename}...")
    
    # --- 1. EXTRACT FACES ---
    face_map = extract_tet_faces(tets)
    
    # Separate interior and boundary faces
    interior_faces = []
    boundary_faces_raw = []
    
    for key, data in face_map.items():
        cells = data['cells']
        nodes = data['nodes']
        
        if len(cells) == 2:
            # Interior face - orient consistently
            oriented_nodes = orient_face_outward(nodes, cells[0], points, tets)
            interior_faces.append((oriented_nodes, cells[0], cells[1]))
        else:
            # Boundary face
            owner_cell = cells[0]
            oriented_nodes = orient_face_outward(nodes, owner_cell, points, tets)
            # Store key for lookup
            boundary_faces_raw.append((oriented_nodes, owner_cell, key))
    
    # --- 2. CLASSIFY BOUNDARY FACES ---
    if boundary_classifier is None and boundary_lookup is None:
        boundary_classifier = default_boundary_classifier(points, tets)
    
    if boundary_zone_types is None:
        boundary_zone_types = {
            "wall": "wall",
            "inlet": "velocity-inlet",
            "outlet": "pressure-outlet",
            "symmetry": "symmetry",
        }
    
    # Group boundary faces by zone
    zone_faces = {}
    # Pre-map user zones for faster lookup
    face_idx_to_zone = {}
    if user_zones:
        for zn, indices in user_zones.items():
            for idx in indices:
                face_idx_to_zone[idx] = zn

    for i, (nodes, owner_cell, key) in enumerate(boundary_faces_raw):
        zone_name = None
        
        # 0. Explicit assignment (Origin feature)
        sorted_nodes = tuple(sorted(nodes))
        if zone_assignments and sorted_nodes in zone_assignments:
            zone_name = zone_assignments[sorted_nodes]

        # 1. Exact lookup (Highest Priority - HEAD feature)
        if zone_name is None and boundary_lookup:
            zone_name = boundary_lookup.get(key)
            
        # 2. User zones (Index based - HEAD feature)
        if zone_name is None:
            zone_name = face_idx_to_zone.get(i)
        
        # 3. Classifier (Geometric)
        if zone_name is None:
            if boundary_classifier:
                face_pts = points[nodes]
                centroid = np.mean(face_pts, axis=0)
                zone_name = boundary_classifier(centroid)
            else:
                zone_name = "wall"
        
        if zone_name not in zone_faces:
            zone_faces[zone_name] = []
        zone_faces[zone_name].append((nodes, owner_cell, 0))
    
    if verbose:
        print(f"[FluentExport] Found {len(interior_faces)} interior faces")
        for zn, faces in zone_faces.items():
            print(f"[FluentExport] Zone '{zn}': {len(faces)} boundary faces")
    
    # --- 3. ASSIGN ZONE IDS ---
    # Zone 1: Nodes
    # Zone 2: Cells (fluid)
    # Zone 3: Interior faces
    # Zone 4+: Boundary zones
    
    zone_id_map = {}
    next_zone_id = 4
    
    # Sort zone names for consistent output
    sorted_zone_names = sorted(zone_faces.keys())
    for zn in sorted_zone_names:
        zone_id_map[zn] = next_zone_id
        next_zone_id += 1
    
    # --- 4. WRITE MSH FILE ---
    n_points = len(points)
    n_cells = len(tets)
    n_interior = len(interior_faces)
    n_boundary_total = sum(len(f) for f in zone_faces.values())
    n_faces_total = n_interior + n_boundary_total
    
    with open(filename, 'w') as f:
        # Header
        f.write('(0 "Fluent mesh exported by MeshPackageLean")\n')
        f.write('(0 "Dimensions:")\n')
        f.write('(2 3)\n')  # 3D
        
        # --- SECTION 10: NODES ---
        f.write(f'\n(0 "Node Section")\n')
        f.write(f'(10 (0 1 {n_points:x} 0 3))\n')  # Declaration
        f.write(f'(10 (1 1 {n_points:x} 1 3)(\n')  # Zone 1, type 1 = boundary nodes
        for p in points:
            f.write(f' {p[0]:.15e} {p[1]:.15e} {p[2]:.15e}\n')
        f.write('))\n')
        
        # --- SECTION 12: CELLS ---
        f.write(f'\n(0 "Cell Section")\n')
        f.write(f'(12 (0 1 {n_cells:x} 0))\n')  # Declaration
        # Zone 2 = fluid, type 1 = active, element type 2 = tet
        f.write(f'(12 (2 1 {n_cells:x} 1 2))\n')
        
        # --- SECTION 13: FACES ---
        f.write(f'\n(0 "Face Section")\n')
        f.write(f'(13 (0 1 {n_faces_total:x} 0))\n')  # Declaration
        
        face_idx = 1
        
        # Interior faces (Zone 3, type 2 = interior)
        if n_interior > 0:
            f.write(f'(13 (3 {face_idx:x} {face_idx + n_interior - 1:x} 2 3)(\n')
            for nodes, c0, c1 in interior_faces:
                n0, n1, n2 = [n + 1 for n in nodes]  # 1-indexed
                f.write(f' {n0:x} {n1:x} {n2:x} {c0 + 1:x} {c1 + 1:x}\n')
            f.write('))\n')
            face_idx += n_interior
        
        # Boundary face zones
        for zone_name in sorted_zone_names:
            faces = zone_faces[zone_name]
            zone_id = zone_id_map[zone_name]
            zone_type = boundary_zone_types.get(zone_name, "wall")
            bc_code = BOUNDARY_TYPE_CODES.get(zone_type, 3)
            
            n_zone_faces = len(faces)
            # CRITICAL: All values in face section must be hexadecimal!
            # bc_code must be :x formatted, otherwise velocity-inlet (10) becomes 0x10=16
            f.write(f'(13 ({zone_id:x} {face_idx:x} {face_idx + n_zone_faces - 1:x} {bc_code:x} 3)(\n')
            for nodes, c0, c1 in faces:
                n0, n1, n2 = [n + 1 for n in nodes]  # 1-indexed
                c0_out = c0 + 1 if c0 >= 0 else 0
                c1_out = c1 + 1 if c1 > 0 else 0
                f.write(f' {n0:x} {n1:x} {n2:x} {c0_out:x} {c1_out:x}\n')
            f.write('))\n')
            face_idx += n_zone_faces
        
        # --- SECTION 39: ZONE LABELS ---
        f.write(f'\n(0 "Zone Labels")\n')
        
        # Zone 1: nodes
        f.write(f'(39 (1 node node-1)())\n')
        
        # Zone 2: cells (fluid)
        f.write(f'(39 (2 fluid {cell_zone_name})())\n')
        
        # Zone 3: interior
        f.write(f'(39 (3 interior default-interior)())\n')
        
        # Boundary zones
        for zone_name in sorted_zone_names:
            zone_id = zone_id_map[zone_name]
            zone_type = boundary_zone_types.get(zone_name, "wall")
            f.write(f'(39 ({zone_id} {zone_type} {zone_name})())\n')
    
    if verbose:
        print(f"[FluentExport] Successfully exported {filename}")
        print(f"[FluentExport]   Nodes: {n_points}")
        print(f"[FluentExport]   Cells: {n_cells}")
        print(f"[FluentExport]   Faces: {n_faces_total} ({n_interior} interior, {n_boundary_total} boundary)")
        print(f"[FluentExport]   Zones: {len(zone_faces)} boundary zones")
    
    return True


def default_boundary_classifier(points: np.ndarray, tets: np.ndarray) -> Callable[[np.ndarray], str]:
    """
    Create a default boundary classifier based on bounding box position.
    
    Classifies faces as:
    - "inlet": faces at minimum X
    - "outlet": faces at maximum X  
    - "wall": all other faces
    """
    # Compute bounding box
    bbox_min = np.min(points, axis=0)
    bbox_max = np.max(points, axis=0)
    bbox_size = bbox_max - bbox_min
    tolerance = np.min(bbox_size) * 0.01
    
    def classifier(centroid: np.ndarray) -> str:
        # Check if at min X (inlet)
        if abs(centroid[0] - bbox_min[0]) < tolerance:
            return "inlet"
        # Check if at max X (outlet)
        if abs(centroid[0] - bbox_max[0]) < tolerance:
            return "outlet"
        # Everything else is wall
        return "wall"
    
    return classifier


# =============================================================================
# SAMPLE MESH GENERATION
# =============================================================================

def create_sample_box_mesh(
    nx: int = 3, ny: int = 3, nz: int = 3,
    size: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a simple structured box mesh of tetrahedra.
    
    Args:
        nx, ny, nz: Number of divisions in each direction
        size: Box dimensions (Lx, Ly, Lz)
    
    Returns:
        points: (N, 3) array of node coordinates
        tets: (M, 4) array of tetrahedral connectivity
    """
    Lx, Ly, Lz = size
    
    # Create structured grid of points
    x = np.linspace(0, Lx, nx + 1)
    y = np.linspace(0, Ly, ny + 1)
    z = np.linspace(0, Lz, nz + 1)
    
    # Generate all points
    points = []
    point_index = {}
    idx = 0
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                points.append([x[i], y[j], z[k]])
                point_index[(i, j, k)] = idx
                idx += 1
    
    points = np.array(points)
    
    # Generate tetrahedra by subdividing each hex cell into 6 tets
    tets = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                # Get the 8 corners of this hex cell
                v = [
                    point_index[(i, j, k)],
                    point_index[(i+1, j, k)],
                    point_index[(i+1, j+1, k)],
                    point_index[(i, j+1, k)],
                    point_index[(i, j, k+1)],
                    point_index[(i+1, j, k+1)],
                    point_index[(i+1, j+1, k+1)],
                    point_index[(i, j+1, k+1)],
                ]
                
                # Subdivide hex into 6 tetrahedra
                # Using the Kuhn triangulation
                tets.extend([
                    [v[0], v[1], v[3], v[4]],
                    [v[1], v[2], v[3], v[6]],
                    [v[1], v[3], v[4], v[6]],
                    [v[1], v[4], v[5], v[6]],
                    [v[3], v[4], v[6], v[7]],
                    [v[4], v[5], v[6], v[7]],
                ])
    
    return points, np.array(tets)


def create_6_zone_classifier(
    points: np.ndarray
) -> Tuple[Callable[[np.ndarray], str], Dict[str, str]]:
    """
    Create a classifier that assigns 6 boundary zones to box faces.
    
    Returns:
        classifier: Function mapping centroid -> zone name
        zone_types: Dict mapping zone name -> boundary type
    """
    bbox_min = np.min(points, axis=0)
    bbox_max = np.max(points, axis=0)
    bbox_size = bbox_max - bbox_min
    tol = np.min(bbox_size) * 0.01
    
    def classifier(centroid: np.ndarray) -> str:
        x, y, z = centroid
        
        # Check each face
        if abs(x - bbox_min[0]) < tol:
            return "inlet"
        if abs(x - bbox_max[0]) < tol:
            return "outlet"
        if abs(y - bbox_min[1]) < tol:
            return "wall-bottom"
        if abs(y - bbox_max[1]) < tol:
            return "wall-top"
        if abs(z - bbox_min[2]) < tol:
            return "symmetry-left"
        if abs(z - bbox_max[2]) < tol:
            return "symmetry-right"
        
        return "wall"
    
    zone_types = {
        "inlet": "velocity-inlet",
        "outlet": "pressure-outlet",
        "wall-bottom": "wall",
        "wall-top": "wall",
        "symmetry-left": "symmetry",
        "symmetry-right": "symmetry",
        "wall": "wall",
    }
    
    return classifier, zone_types


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Generate a sample mesh with named boundary zones."""
    print("=" * 60)
    print("Fluent MSH Export with Named Boundary Layers")
    print("=" * 60)
    
    # Create sample mesh
    print("\n[1] Creating sample box mesh...")
    points, tets = create_sample_box_mesh(nx=4, ny=4, nz=4, size=(1.0, 0.5, 0.5))
    print(f"    Nodes: {len(points)}")
    print(f"    Tetrahedra: {len(tets)}")
    
    # Create classifier for 6 named zones
    classifier, zone_types = create_6_zone_classifier(points)
    
    # Export
    output_file = Path(__file__).parent.parent / "apps" / "cli" / "generated_meshes" / "sample_fluent_named.msh"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[2] Exporting to {output_file}...")
    export_fluent_msh(
        str(output_file),
        points,
        tets,
        boundary_classifier=classifier,
        boundary_zone_types=zone_types,
        cell_zone_name="fluid-domain",
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("EXPORT COMPLETE!")
    print("=" * 60)
    print("\nTo verify in ANSYS Fluent:")
    print("1. Open Fluent in 3D mode")
    print(f"2. File > Read > Mesh > {output_file}")
    print("3. Check Boundary Conditions panel for named zones:")
    print("   - inlet (velocity-inlet)")
    print("   - outlet (pressure-outlet)")
    print("   - wall-bottom, wall-top (wall)")
    print("   - symmetry-left, symmetry-right (symmetry)")
    print("=" * 60)


if __name__ == "__main__":
    main()
