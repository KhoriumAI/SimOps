"""
GPU-accelerated helper functions for hex meshing.
"""
import numpy as np

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None

def closest_point_to_triangles_gpu(surface_vertices, surface_faces, query_points, verbose=False):
    """
    GPU-accelerated closest point search on triangle mesh.
    Falls back to CPU if GPU unavailable.
    
    Args:
        surface_vertices: (N_verts, 3) array
        surface_faces: (N_faces, 3) array of indices
        query_points: (N_query, 3) array
        verbose: Print timing info
        
    Returns:
        closest_points: (N_query, 3) array
    """
    if not GPU_AVAILABLE:
        if verbose:
            print("[GPU] CuPy not available - using CPU fallback")
        return _closest_point_cpu_fallback(surface_vertices, surface_faces, query_points)
    
    try:
        import time
        start = time.time()
        
        # Transfer to GPU
        triangles = surface_vertices[surface_faces]  # (N_faces, 3, 3)
        triangles_gpu = cp.asarray(triangles, dtype=cp.float32)
        queries_gpu = cp.asarray(query_points, dtype=cp.float32)
        
        N_query = len(query_points)
        N_triangles = len(triangles)
        
        if verbose:
            print(f"[GPU] Processing {N_query} queries against {N_triangles} triangles")
        
        # FULLY VECTORIZED: Compute distances to ALL triangles for ALL queries
        # Shape: (N_query, N_triangles, 3)
        queries_broadcast = queries_gpu[:, None, :]  # (N_query, 1, 3)
        triangles_broadcast = triangles_gpu[None, :, :, :]  # (1, N_triangles, 3, 3)
        
        # Extract triangle vertices
        v0 = triangles_broadcast[:, :, 0, :]  # (1, N_triangles, 3)
        v1 = triangles_broadcast[:, :, 1, :]
        v2 = triangles_broadcast[:, :, 2, :]
        
        # Vectorized barycentric projection (simplified for speed)
        # For each query-triangle pair, project onto triangle plane
        edge0 = v1 - v0
        edge1 = v2 - v0
        v0_to_p = queries_broadcast - v0
        
        # Dot products (vectorized)
        a = cp.sum(edge0 * edge0, axis=2)  # (1, N_triangles)
        b = cp.sum(edge0 * edge1, axis=2)
        c = cp.sum(edge1 * edge1, axis=2)
        d = cp.sum(edge0 * v0_to_p, axis=2)  # (N_query, N_triangles)
        e = cp.sum(edge1 * v0_to_p, axis=2)
        
        # Barycentric coordinates (clamped to triangle)
        det = a * c - b * b
        det = cp.maximum(det, 1e-12)  # Avoid division by zero
        
        s = (b * e - c * d) / det
        t = (b * d - a * e) / det
        
        # Clamp to [0,1] and ensure s+t <= 1
        s = cp.clip(s, 0, 1)
        t = cp.clip(t, 0, 1)
        s_plus_t = s + t
        mask = s_plus_t > 1
        s = cp.where(mask, s / s_plus_t, s)
        t = cp.where(mask, t / s_plus_t, t)
        
        # Compute closest points on all triangles
        # Shape: (N_query, N_triangles, 3)
        closest_on_triangles = v0 + s[:, :, None] * edge0 + t[:, :, None] * edge1
        
        # Compute distances (N_query, N_triangles)
        distances_sq = cp.sum((queries_broadcast - closest_on_triangles) ** 2, axis=2)
        
        # Find minimum distance triangle for each query
        min_indices = cp.argmin(distances_sq, axis=1)  # (N_query,)
        
        # Extract closest points
        closest_points_gpu = closest_on_triangles[cp.arange(N_query), min_indices]
        
        # Transfer back to CPU
        result = cp.asnumpy(closest_points_gpu)
        
        elapsed = time.time() - start
        if verbose:
            print(f"[GPU] Closest point search completed in {elapsed:.3f}s")
        
        return result
        
    except Exception as e:
        if verbose:
            print(f"[GPU] GPU computation failed ({e}), falling back to CPU")
        return _closest_point_cpu_fallback(surface_vertices, surface_faces, query_points)

def _closest_point_cpu_fallback(surface_vertices, surface_faces, query_points):
    """CPU fallback for closest point search."""
    # Import the brute-force implementation from conformal_hex_glue
    # This duplicates the logic we already have
    triangles = surface_vertices[surface_faces]
    closest_points = np.zeros_like(query_points)
    
    for i, query_point in enumerate(query_points):
        min_dist_sq = np.inf
        closest_pt = query_point
        
        for tri in triangles:
            v0, v1, v2 = tri[0], tri[1], tri[2]
            edge0 = v1 - v0
            edge1 = v2 - v0
            v0_to_p = query_point - v0
            
            a = np.dot(edge0, edge0)
            b = np.dot(edge0, edge1)
            c = np.dot(edge1, edge1)
            d = np.dot(edge0, v0_to_p)
            e = np.dot(edge1, v0_to_p)
            
            det = a * c - b * b
            if det < 1e-12:
                continue
            
            s = b * e - c * d
            t = b * d - a * e
            
            # Simplified barycentric clamping
            s = np.clip(s / det, 0, 1)
            t = np.clip(t / det, 0, 1)
            if s + t > 1:
                s = s / (s + t)
                t = t / (s + t)
            
            closest_on_tri = v0 + s * edge0 + t * edge1
            dist_sq = np.sum((query_point - closest_on_tri) ** 2)
            
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_pt = closest_on_tri
        
        closest_points[i] = closest_pt
    
    return closest_points
