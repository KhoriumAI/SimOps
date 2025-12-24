# core/validators/geometry_check.py
import numpy as np

def check_shell_winding(nodes: np.ndarray, original_normal: np.ndarray, elem_id: int):
    """
    Verifies that the winding order of a shell element (nodes 1-2-3) produces
    a normal vector consistent with the original surface normal.
    
    Args:
        nodes: Float array of shape (N, 3) containing coordinates.
               For Tri3, shape (3, 3).
        original_normal: Float array of shape (3,) derived from geometry.
        elem_id: Integer ID for error reporting.
        
    Raises:
        ValueError: If winding order is inverted (dot product < 0).
    """
    if len(nodes) < 3:
        return # Cannot check line elements
        
    # Compute mesh normal via cross product (Right-Hand Rule)
    # n = (p2 - p1) x (p3 - p1)
    v1 = nodes[1] - nodes[0]
    v2 = nodes[2] - nodes[0]
    mesh_normal = np.cross(v1, v2)
    
    # Normalize for safety (though not strictly needed for dot sign check)
    norm = np.linalg.norm(mesh_normal)
    if norm < 1e-12:
        return # Degenerate element, let solver handle it or skip
        
    mesh_normal /= norm
    
    # Check alignment
    if np.dot(mesh_normal, original_normal) < 0:
        raise ValueError(f"FATAL: Element {elem_id} has inverted winding order! "
                         f"Mesh vs Geo Normal was negative dot product.")
        
if __name__ == "__main__":
    # Self-Test
    print("Testing geometry_check.py...")
    
    # CCW Triangle (Classic Correct) in XY plane
    p1 = np.array([0.,0.,0.])
    p2 = np.array([1.,0.,0.])
    p3 = np.array([0.,1.,0.])
    nodes_ccw = np.array([p1, p2, p3])
    
    # Geo Normal = +Z
    geo_norm = np.array([0.,0.,1.])
    
    try:
        check_shell_winding(nodes_ccw, geo_norm, 1)
        print("PASS: CCW Triangle accepted.")
    except ValueError as e:
        print(f"FAIL: CCW Triangle rejected: {e}")
        
    # CW Triangle (Inverted)
    nodes_cw = np.array([p1, p3, p2]) # Swap 2 and 3
    
    try:
        check_shell_winding(nodes_cw, geo_norm, 2)
        print("FAIL: CW Triangle accepted (Should have raised error).")
    except ValueError:
        print("PASS: CW Triangle rejected.")
