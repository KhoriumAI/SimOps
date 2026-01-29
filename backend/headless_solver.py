"""
Headless thermal solver for SimOps API (Docker / no gmsh).

Uses meshio + numpy + scipy only. No gmsh, no X11.
Produces result dict compatible with export_vtk_with_temperature().
"""
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def _write_vtk_geometry(path: str, points: np.ndarray, cells: np.ndarray, is_triangle: bool) -> None:
    """
    Write legacy VTK POLYDATA (surface mesh) for frontend preview.
    Converts tets to surface triangles if needed.
    """
    npts = len(points)
    
    # If we have triangles, use them directly
    if is_triangle:
        nc = len(cells)
        with open(path, "w") as f:
            f.write("# vtk DataFile Version 3.0\nSimOps Preview\nASCII\n")
            f.write("DATASET POLYDATA\n")
            f.write(f"POINTS {npts} float\n")
            for i in range(npts):
                p = points[i]
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            f.write(f"\nPOLYGONS {nc} {nc * 4}\n")  # 3 vertices + 1 count per triangle
            for i in range(nc):
                c = cells[i]
                f.write(f"3 {' '.join(str(int(x)) for x in c[:3])}\n")
        return
    
    # For tets, extract surface triangles
    # Simple approach: collect all faces and keep only unique ones
    faces = {}
    for tet in cells:
        # Each tet has 4 faces: (0,1,2), (0,1,3), (0,2,3), (1,2,3)
        face_indices = [
            (tet[0], tet[1], tet[2]),
            (tet[0], tet[1], tet[3]),
            (tet[0], tet[2], tet[3]),
            (tet[1], tet[2], tet[3]),
        ]
        for face in face_indices:
            # Normalize face order (smallest index first, then sort remaining)
            sorted_face = tuple(sorted(face))
            # Track both orientations to detect interior faces
            if sorted_face in faces:
                # This face appears twice (interior), remove it
                del faces[sorted_face]
            else:
                # Store original orientation for exterior face
                faces[sorted_face] = face
    
    # Write POLYDATA with surface triangles
    surface_tris = list(faces.values())
    nc = len(surface_tris)
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\nSimOps Preview\nASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {npts} float\n")
        for i in range(npts):
            p = points[i]
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        f.write(f"\nPOLYGONS {nc} {nc * 4}\n")
        for tri in surface_tris:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")


def _parse_msh_ascii_fallback(msh_path: str) -> tuple:
    """
    Minimal MSH 4.1 / 2.2 ASCII parser. Returns (points, cells, is_triangle).
    Used when meshio.read fails (e.g. multi-block / cell_sets index bugs).
    """
    with open(msh_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    if "$Nodes" not in content or "$Elements" not in content:
        raise ValueError("Invalid MSH: missing $Nodes or $Elements")

    # Version
    msh_version = "2.2"
    if "$MeshFormat" in content:
        try:
            fmt = content.split("$MeshFormat")[1].split("$EndMeshFormat")[0].strip().split("\n")[0]
            msh_version = fmt.split()[0]
        except Exception:
            pass

    nodes_section = content.split("$Nodes")[1].split("$EndNodes")[0].strip().split("\n")
    elements_section = content.split("$Elements")[1].split("$EndElements")[0].strip().split("\n")

    nodes = {}
    node_id_to_idx = {}

    # --- Nodes ---
    header = nodes_section[0].split()
    if msh_version.startswith("4"):
        num_blocks = int(header[0])
        curr = 1
        for _ in range(num_blocks):
            # entityDim entityTag parametric numNodesInBlock
            bh = nodes_section[curr].split()
            curr += 1
            n_in_block = int(bh[3])
            tags = [int(nodes_section[curr + i]) for i in range(n_in_block)]
            curr += n_in_block
            for i in range(n_in_block):
                xyz = list(map(float, nodes_section[curr + i].split()))[:3]
                tag = tags[i]
                nodes[tag] = xyz
                node_id_to_idx[tag] = len(node_id_to_idx)
            curr += n_in_block
    else:
        n_nodes = int(header[0])
        for i in range(1, n_nodes + 1):
            p = nodes_section[i].split()
            tag = int(p[0])
            xyz = [float(p[1]), float(p[2]), float(p[3])]
            nodes[tag] = xyz
            node_id_to_idx[tag] = len(node_id_to_idx)

    # --- Elements (tet=4, tet10=11, tri=2, tri6=9) ---
    tets, tris = [], []
    header = elements_section[0].split()
    if msh_version.startswith("4"):
        num_blocks = int(header[0])
        curr = 1
        for _ in range(num_blocks):
            # entityDim entityTag elementType numElementsInBlock
            bh = elements_section[curr].split()
            curr += 1
            el_type = int(bh[2])
            n_el = int(bh[3])
            for i in range(n_el):
                data = [int(x) for x in elements_section[curr + i].split()]
                try:
                    if el_type == 4 and len(data) >= 5:
                        tets.append([node_id_to_idx[n] for n in data[1:5]])
                    elif el_type == 11 and len(data) >= 5:
                        tets.append([node_id_to_idx[n] for n in data[1:5]])
                    elif el_type == 2 and len(data) >= 4:
                        tris.append([node_id_to_idx[n] for n in data[1:4]])
                    elif el_type == 9 and len(data) >= 4:
                        tris.append([node_id_to_idx[n] for n in data[1:4]])
                except KeyError:
                    pass
            curr += n_el
    else:
        n_el = int(header[0])
        for i in range(1, n_el + 1):
            p = elements_section[i].split()
            el_type = int(p[1])
            n_tags = int(p[2])
            node_start = 3 + n_tags
            try:
                if el_type == 4 and len(p) >= node_start + 4:
                    nids = [node_id_to_idx[int(p[k])] for k in range(node_start, node_start + 4)]
                    tets.append(nids)
                elif el_type == 11 and len(p) >= node_start + 4:
                    nids = [node_id_to_idx[int(p[k])] for k in range(node_start, node_start + 4)]
                    tets.append(nids)
                elif el_type == 2 and len(p) >= node_start + 3:
                    nids = [node_id_to_idx[int(p[k])] for k in range(node_start, node_start + 3)]
                    tris.append(nids)
                elif el_type == 9 and len(p) >= node_start + 3:
                    nids = [node_id_to_idx[int(p[k])] for k in range(node_start, node_start + 3)]
                    tris.append(nids)
            except KeyError:
                pass

    n = len(nodes)
    points = np.zeros((n, 3), dtype=np.float64)
    for tag, idx in node_id_to_idx.items():
        points[idx] = nodes[tag]

    if tets:
        cells = np.array(tets, dtype=np.int64)
        return points, cells, False
    if tris:
        cells = np.array(tris, dtype=np.int64)
        return points, cells, True
    raise ValueError("No tetrahedral or triangle elements found for preview")


def msh_to_preview_vtk(msh_path: str, vtk_path: str) -> str:
    """
    Convert .msh to VTK for 3D preview. Prefer meshio read + hand-written VTK;
    on meshio failure (e.g. multi-block index bugs), use ASCII fallback parser.
    """
    points, cells, is_tri = None, None, False
    try:
        import meshio

        mesh = meshio.read(msh_path)
        pts = np.asarray(mesh.points, dtype=np.float64).copy()
        if pts.ndim == 2 and pts.shape[1] == 2:
            pts = np.column_stack([pts, np.zeros(len(pts))])
        tets, tris = [], []
        for block in mesh.cells:
            try:
                raw = np.asarray(block.data, dtype=np.int64).copy()
            except Exception:
                continue
            if block.type == "tetra":
                tets.append(raw)
            elif block.type == "tetra10":
                tets.append(raw[:, :4])
            elif block.type == "triangle":
                tris.append(raw)
            elif block.type == "triangle6":
                tris.append(raw[:, :3])
        if tets:
            cells = np.vstack(tets).astype(np.int64)
            points, is_tri = pts, False
        elif tris:
            cells = np.vstack(tris).astype(np.int64)
            points, is_tri = pts, True
    except Exception:
        points, cells, is_tri = _parse_msh_ascii_fallback(msh_path)

    if points is None or cells is None:
        raise ValueError("No tetrahedral or triangle elements found for preview")

    _write_vtk_geometry(vtk_path, points, cells, is_triangle=is_tri)
    return vtk_path


def load_mesh_msh(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load .msh via meshio. Return (node_coords, elements) for tets only."""
    import meshio

    mesh = meshio.read(path)
    points = np.asarray(mesh.points, dtype=np.float64)

    elements = []
    for block in mesh.cells:
        if block.type == "tetra":
            elements.append(block.data)
        elif block.type == "tetra10":
            # use first 4 nodes for linear tet
            elements.append(block.data[:, :4])
    if not elements:
        raise ValueError("No tetrahedral elements in mesh")
    elems = np.vstack(elements).astype(np.int64)

    return points, elems


def solve_thermal(
    node_coords: np.ndarray,
    elements: np.ndarray,
    heat_source_k: float = 373.15,
    ambient_k: float = 293.15,
    conductivity: float = 200.0,
    max_iter: int = 5000,
    tol: float = 1e-8,
) -> Dict:
    """Steady-state heat conduction. BCs: top 10% z = hot, bottom 10% = ambient."""
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import cg, spilu, LinearOperator
    import time

    n = len(node_coords)
    z = node_coords[:, 2]
    z_min, z_max = float(np.min(z)), float(np.max(z))
    r = z_max - z_min or 1.0

    hot = np.where(z >= z_max - 0.1 * r)[0]
    cold = np.where(z <= z_min + 0.1 * r)[0]
    bc_nodes = np.concatenate([hot, cold])
    bc_temps = np.concatenate([
        np.full(len(hot), heat_source_k),
        np.full(len(cold), ambient_k),
    ])

    K = lil_matrix((n, n))
    Q = np.zeros(n)
    k = conductivity

    for elem in elements:
        coords = node_coords[elem]
        ke = _element_ke(coords, k)
        for i, ni in enumerate(elem):
            for j, nj in enumerate(elem):
                K[ni, nj] += ke[i, j]

    penalty = 1e20
    for idx, temp in zip(bc_nodes, bc_temps):
        K[idx, idx] += penalty
        Q[idx] = penalty * temp

    K = K.tocsr()
    t0 = time.time()
    try:
        ilu = spilu(K.tocsc(), drop_tol=1e-4, fill_factor=10)
        M = LinearOperator(K.shape, matvec=ilu.solve)
    except Exception:
        M = None
    T, info = cg(K, Q, M=M, atol=tol, maxiter=max_iter)
    elapsed = time.time() - t0

    return {
        "node_coords": node_coords,
        "elements": elements,
        "temperature": np.asarray(T, dtype=np.float64),
        "min_temp": float(np.min(T)),
        "max_temp": float(np.max(T)),
        "solve_time": elapsed,
        "num_elements": len(elements),
    }


def _element_ke(coords: np.ndarray, k: float) -> np.ndarray:
    """4x4 conductivity matrix for linear tet."""
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    J = np.array([
        [x[1] - x[0], x[2] - x[0], x[3] - x[0]],
        [y[1] - y[0], y[2] - y[0], y[3] - y[0]],
        [z[1] - z[0], z[2] - z[0], z[3] - z[0]],
    ])
    detJ = np.linalg.det(J)
    V = abs(detJ) / 6.0
    if V < 1e-20:
        return np.zeros((4, 4))
    try:
        invJ = np.linalg.inv(J)
    except Exception:
        return np.zeros((4, 4))
    dN_ref = np.array([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    dN = dN_ref @ invJ
    return k * V * (dN @ dN.T)


def run_headless_solve(
    msh_path: str,
    output_dir: str,
    config: dict,
) -> Dict:
    """Load .msh, solve, return result dict. Optionally write VTK to output_dir."""
    path = Path(msh_path)
    if not path.exists():
        raise FileNotFoundError(msh_path)

    node_coords, elements = load_mesh_msh(str(path))
    hot = float(config.get("heat_source_temperature", 373.15))
    amb = float(config.get("ambient_temperature", 293.15))
    k = 200.0
    mat = config.get("material", "").lower()
    if "alu" in mat or "aluminum" in mat:
        k = 200.0
    elif "steel" in mat:
        k = 50.0

    out = solve_thermal(
        node_coords, elements,
        heat_source_k=hot, ambient_k=amb, conductivity=k,
        max_iter=int(config.get("max_iterations", 5000)),
        tol=float(config.get("tolerance", 1e-8)),
    )

    od = Path(output_dir)
    od.mkdir(parents=True, exist_ok=True)
    vtk_path = od / f"{path.stem}_thermal.vtk"
    export_vtk(out, str(vtk_path))
    out["vtk_path"] = str(vtk_path)
    return out


def export_vtk(result: Dict, path: str) -> str:
    """Write VTK with temperature (same shape as simops_worker.export_vtk_with_temperature)."""
    node_coords = result["node_coords"]
    elements = result["elements"]
    temperature = result["temperature"]

    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\nSimOps Thermal Result\nASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {len(node_coords)} float\n")
        for p in node_coords:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        nc = len(elements)
        f.write(f"\nCELLS {nc} {nc * 5}\n")
        for e in elements:
            f.write(f"4 {e[0]} {e[1]} {e[2]} {e[3]}\n")
        f.write(f"\nCELL_TYPES {nc}\n")
        for _ in range(nc):
            f.write("10\n")
        f.write(f"\nPOINT_DATA {len(temperature)}\nSCALARS Temperature float 1\nLOOKUP_TABLE default\n")
        for t in temperature:
            f.write(f"{t:.4f}\n")
    return path
