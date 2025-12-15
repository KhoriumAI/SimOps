"""
Fast Winding Number - Fully Vectorized Barnes-Hut
==================================================

Achieves microsecond-range queries by:
1. Processing ALL query points simultaneously against each octree node
2. Using NumPy broadcasting for massive parallelism
3. Dipole approximation for distant clusters

Target: < 1ms per 1000 points (vs 5ms per point in naive version)
"""

import numpy as np


def compute_triangle_properties(vertices, faces):
    """Vectorized triangle property computation."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    centroids = (v0 + v1 + v2) / 3.0
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    area_vectors = cross * 0.5  # Area-weighted normals (dipole term)
    
    return centroids, areas, area_vectors, v0, v1, v2


def solid_angle_vectorized(points, v0, v1, v2):
    """
    Compute solid angles for ALL points against ALL triangles simultaneously.
    
    Args:
        points: (P, 3) query points
        v0, v1, v2: (T, 3) triangle vertices
        
    Returns:
        (P, T) matrix of solid angles
    """
    P = len(points)
    T = len(v0)
    
    # Expand dimensions for broadcasting: (P, 1, 3) - (1, T, 3) = (P, T, 3)
    points_exp = points[:, np.newaxis, :]  # (P, 1, 3)
    v0_exp = v0[np.newaxis, :, :]  # (1, T, 3)
    v1_exp = v1[np.newaxis, :, :]
    v2_exp = v2[np.newaxis, :, :]
    
    a = v0_exp - points_exp  # (P, T, 3)
    b = v1_exp - points_exp
    c = v2_exp - points_exp
    
    # Norms: (P, T)
    a_norm = np.linalg.norm(a, axis=2)
    b_norm = np.linalg.norm(b, axis=2)
    c_norm = np.linalg.norm(c, axis=2)
    
    # Cross product b x c: (P, T, 3)
    bc_cross = np.cross(b, c)
    
    # Numerator: a · (b × c)
    num = np.sum(a * bc_cross, axis=2)  # (P, T)
    
    # Dot products
    ab_dot = np.sum(a * b, axis=2)
    ac_dot = np.sum(a * c, axis=2)
    bc_dot = np.sum(b * c, axis=2)
    
    den = a_norm * b_norm * c_norm + ab_dot * c_norm + ac_dot * b_norm + bc_dot * a_norm
    
    # Handle degenerate cases
    result = 2.0 * np.arctan2(num, den)
    degenerate = (a_norm < 1e-12) | (b_norm < 1e-12) | (c_norm < 1e-12)
    result[degenerate] = 0.0
    
    return result  # (P, T)


def dipole_approximation_vectorized(points, centers, area_vectors):
    """
    Compute dipole approximation for ALL points against ALL clusters.
    
    Args:
        points: (P, 3) query points
        centers: (C, 3) cluster centers
        area_vectors: (C, 3) cluster area vectors (dipole)
        
    Returns:
        (P, C) solid angle approximations
    """
    # r_vec = center - point: (P, C, 3)
    points_exp = points[:, np.newaxis, :]  # (P, 1, 3)
    centers_exp = centers[np.newaxis, :, :]  # (1, C, 3)
    
    r_vec = centers_exp - points_exp  # (P, C, 3)
    
    # |r|^3
    r_sq = np.sum(r_vec * r_vec, axis=2)  # (P, C)
    r = np.sqrt(r_sq)
    r_cubed = r_sq * r
    r_cubed = np.where(r_cubed < 1e-30, 1e-30, r_cubed)  # Avoid div by zero
    
    # A · r / |r|^3
    area_vectors_exp = area_vectors[np.newaxis, :, :]  # (1, C, 3)
    dot = np.sum(area_vectors_exp * r_vec, axis=2)  # (P, C)
    
    return dot / r_cubed  # (P, C)


def build_octree_flat(centroids, areas, area_vectors, v0, v1, v2, max_tris=32):
    """
    Build a flattened list of leaf nodes by recursively splitting.
    
    Returns lists of:
    - leaf_tri_indices: List of triangle index arrays for each leaf
    - node_centers: (N, 3) cluster centers
    - node_area_vectors: (N, 3) cluster dipoles
    - node_sizes: (N,) cluster sizes
    """
    leaf_groups = []
    node_centers = []
    node_area_vectors = []
    node_sizes = []
    
    # Initial bounds
    bbox_min = np.min(centroids, axis=0)
    bbox_max = np.max(centroids, axis=0)
    
    def recurse(indices, min_b, max_b):
        n_tris = len(indices)
        
        # Compute cluster properties
        oct_centroids = centroids[indices]
        oct_areas = areas[indices]
        oct_area_vectors = area_vectors[indices]
        
        total_area = np.sum(oct_areas)
        if total_area > 1e-12:
            weighted_center = np.average(oct_centroids, axis=0, weights=oct_areas)
        else:
            weighted_center = np.mean(oct_centroids, axis=0)
            
        total_area_vector = np.sum(oct_area_vectors, axis=0)
        
        # Size
        curr_min = np.min(oct_centroids, axis=0)
        curr_max = np.max(oct_centroids, axis=0)
        size = np.max(curr_max - curr_min)
        
        # Leaf condition
        if n_tris <= max_tris:
            leaf_groups.append(indices)
            node_centers.append(weighted_center)
            node_area_vectors.append(total_area_vector)
            node_sizes.append(size)
            return

        # Split
        center = (min_b + max_b) / 2
        diff = oct_centroids - center
        # Calculate octant index for each triangle
        octant_idx = ((diff[:, 0] > 0).astype(int) | 
                      ((diff[:, 1] > 0).astype(int) << 1) | 
                      ((diff[:, 2] > 0).astype(int) << 2))
        
        for i in range(8):
            mask = octant_idx == i
            if not np.any(mask):
                continue
            
            child_indices = indices[mask]
            
            # Calculate child bounds
            child_min = min_b.copy()
            child_max = max_b.copy()
            if i & 1: child_min[0] = center[0]
            else:     child_max[0] = center[0]
            if i & 2: child_min[1] = center[1]
            else:     child_max[1] = center[1]
            if i & 4: child_min[2] = center[2]
            else:     child_max[2] = center[2]
            
            recurse(child_indices, child_min, child_max)

    # Start recursion
    recurse(np.arange(len(centroids)), bbox_min, bbox_max)
    
    return (leaf_groups, 
            np.array(node_centers), 
            np.array(node_area_vectors), 
            np.array(node_sizes))


def compute_fast_winding_grid(vertices, faces, query_points, verbose=True, theta=1.5):
    """
    Fast winding number using vectorized Barnes-Hut.
    
    Args:
        vertices: (V, 3) mesh vertices
        faces: (F, 3) triangle indices
        query_points: (P, 3) query points
        verbose: Print progress
        theta: Approximation threshold (higher = faster, less accurate)
        
    Returns:
        (P,) winding numbers normalized to [0, 1]
    """
    n_points = len(query_points)
    n_faces = len(faces)
    
    if verbose:
        print(f"[Fast Winding] {n_points} points x {n_faces} faces", flush=True)
    
    # Compute triangle properties
    centroids, areas, area_vectors, v0, v1, v2 = compute_triangle_properties(vertices, faces)
    
    # For very small meshes, just use brute force (fully vectorized)
    if n_faces <= 500:
        if verbose:
            print(f"[Fast Winding] Using vectorized brute force...", flush=True)
        
        # Process in batches to avoid memory issues
        batch_size = min(1000, n_points)
        results = np.zeros(n_points)
        
        for i in range(0, n_points, batch_size):
            end = min(i + batch_size, n_points)
            batch_points = query_points[i:end]
            
            # (batch, faces) solid angles
            angles = solid_angle_vectorized(batch_points, v0, v1, v2)
            results[i:end] = np.sum(angles, axis=1)
            
            if verbose:
                pct = int(100 * end / n_points)
                print(f"  [Fast Winding] {end}/{n_points} ({pct}%)", flush=True)
        
        return results / (4 * np.pi)
    
    # Build octree for larger meshes
    if verbose:
        print(f"[Fast Winding] Building spatial index...", flush=True)
    
    leaf_groups, node_centers, node_area_vectors, node_sizes = build_octree_flat(
        centroids, areas, area_vectors, v0, v1, v2
    )
    
    n_nodes = len(leaf_groups)
    if verbose:
        print(f"[Fast Winding] {n_nodes} spatial clusters", flush=True)
    
    # Process in point batches (increased from 500 to 10k for large meshes)
    batch_size = min(10000, n_points)
    results = np.zeros(n_points)
    
    for bi in range(0, n_points, batch_size):
        end = min(bi + batch_size, n_points)
        batch_points = query_points[bi:end]
        batch_size_actual = end - bi
        
        batch_angles = np.zeros(batch_size_actual)
        
        # For each cluster, decide: approximate or exact
        for ci, (indices, center, area_vec, size) in enumerate(
            zip(leaf_groups, node_centers, node_area_vectors, node_sizes)
        ):
            # Distance from each point to cluster center
            r_vec = center - batch_points  # (batch, 3)
            r = np.linalg.norm(r_vec, axis=1)  # (batch,)
            
            # Barnes-Hut criterion: size/distance < 1/theta
            far_enough = (r > 1e-10) & ((size / r) < (1.0 / theta))
            
            # Points that can use approximation
            if np.any(far_enough):
                r_cubed = r[far_enough] ** 3
                r_cubed = np.maximum(r_cubed, 1e-30)
                dot = np.sum(area_vec * r_vec[far_enough], axis=1)
                batch_angles[far_enough] += dot / r_cubed
            
            # Points that need exact calculation
            need_exact = ~far_enough
            if np.any(need_exact):
                exact_points = batch_points[need_exact]
                # Get triangles for this cluster
                cluster_v0 = v0[indices]
                cluster_v1 = v1[indices]
                cluster_v2 = v2[indices]
                
                # Vectorized solid angles: (n_exact_points, n_cluster_tris)
                angles = solid_angle_vectorized(exact_points, cluster_v0, cluster_v1, cluster_v2)
                batch_angles[need_exact] += np.sum(angles, axis=1)
        
        results[bi:end] = batch_angles
        
        if verbose:
            pct = int(100 * end / n_points)
            print(f"  [Fast Winding] {end}/{n_points} ({pct}%)", flush=True)
    
    if verbose:
        print(f"[Fast Winding] Complete!", flush=True)
    
    return results / (4 * np.pi)


if __name__ == "__main__":
    import time
    
    print("=== Fast Winding Number Test ===")
    
    # Create test cube
    cube_verts = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ], dtype=np.float64)
    
    cube_faces = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
        [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
    ])
    
    # Test with many points
    np.random.seed(42)
    test_pts = np.random.rand(1000, 3) * 2 - 0.5  # Some inside, some outside
    
    start = time.time()
    results = compute_fast_winding_grid(cube_verts, cube_faces, test_pts, verbose=True)
    elapsed = (time.time() - start) * 1000
    
    inside = np.sum(results > 0.5)
    print(f"Inside: {inside}, Outside: {1000 - inside}")
    print(f"Time: {elapsed:.2f} ms ({elapsed/1000:.3f} ms/point)")
    print("Done!")
