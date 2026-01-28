"""
Headless thermal solver for SimOps API (Docker / no gmsh).

Uses meshio + numpy + scipy only. No gmsh, no X11.
Produces result dict compatible with export_vtk_with_temperature().
"""
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


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
