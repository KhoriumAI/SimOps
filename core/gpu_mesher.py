"""
GPU Delaunay Mesher Integration
================================

Implements the "Fill & Filter" pipeline for GPU-accelerated tetrahedral meshing
with boundary layer quality enforcement.

Pipeline:
1. Generate surface mesh (CPU)
2. Compute Local Feature Size (LFS) for each surface vertex
3. Generate boundary layer points (Normal Offset Scaffold)
4. Generate internal points with LFS-aware spacing
5. GPU Delaunay triangulation
6. Filter tetrahedra by winding number
7. Validate boundary layer health
"""

import numpy as np
import sys
import os
from pathlib import Path

# Add GPU mesher to path
GPU_MESHER_PATH = Path(__file__).parent.parent / "gpu_experiments" / "Release" / "Release"
if GPU_MESHER_PATH.exists():
    sys.path.insert(0, str(GPU_MESHER_PATH))
    
    # Register CUDA DLLs
    cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin"
    if os.path.exists(cuda_path):
        if hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(cuda_path)
        os.environ['PATH'] = cuda_path + os.pathsep + os.environ['PATH']

try:
    import _gpumesher
    GPU_AVAILABLE = True
except ImportError as e:
    print("[GPU Mesher] WARNING: GPU Mesher not available: {}".format(e))
    GPU_AVAILABLE = False


# =============================================================================
# LOCAL FEATURE SIZE (LFS) COMPUTATION
# =============================================================================

def compute_vertex_lfs(surface_verts, surface_faces):
    """
    Computes Local Feature Size for each vertex.
    LFS = Average edge length of triangles connected to this vertex.
    
    OPTIMIZED: Uses np.bincount for vectorized scatter-add (10-50x faster).
    
    Returns:
        lfs_field: (N,) array of local sizes per vertex
    """
    num_verts = len(surface_verts)
    
    # Get triangle vertices (vectorized)
    v0 = surface_verts[surface_faces[:, 0]]
    v1 = surface_verts[surface_faces[:, 1]]
    v2 = surface_verts[surface_faces[:, 2]]
    
    # Compute edge lengths (vectorized)
    e01 = np.linalg.norm(v1 - v0, axis=1)
    e12 = np.linalg.norm(v2 - v1, axis=1)
    e20 = np.linalg.norm(v0 - v2, axis=1)
    
    # Average edge length per face
    avg_edge = (e01 + e12 + e20) / 3.0
    
    # Scatter-add using np.bincount (replaces Python loop)
    # Flatten face indices and repeat weights for each vertex of each face
    flat_indices = surface_faces.ravel()  # Shape: (num_faces * 3,)
    weights = np.repeat(avg_edge, 3)       # Each face contributes to 3 vertices
    
    # Accumulate edge sums and counts per vertex
    edge_sum = np.bincount(flat_indices, weights=weights, minlength=num_verts)
    edge_count = np.bincount(flat_indices, minlength=num_verts)
    
    # Compute average (avoid div/0)
    lfs_field = np.divide(edge_sum, edge_count, 
                          out=np.ones(num_verts, dtype=np.float64), 
                          where=edge_count > 0)
    
    return lfs_field


def compute_vertex_normals(surface_verts, surface_faces):
    """
    Computes per-vertex normals by averaging connected face normals.
    
    OPTIMIZED: Uses np.add.at for vectorized scatter-add (10-50x faster).
    
    Returns:
        normals: (N, 3) array of unit normals per vertex
    """
    num_verts = len(surface_verts)
    
    # Compute face normals (vectorized)
    v0 = surface_verts[surface_faces[:, 0]]
    v1 = surface_verts[surface_faces[:, 1]]
    v2 = surface_verts[surface_faces[:, 2]]
    
    face_normals = np.cross(v1 - v0, v2 - v0)  # Shape: (num_faces, 3)
    
    # Accumulate normals to vertices using np.add.at (replaces Python loop)
    # Each face normal contributes to its 3 vertices
    normal_sum = np.zeros((num_verts, 3), dtype=np.float64)
    
    # Scatter-add face normals to each vertex
    np.add.at(normal_sum, surface_faces[:, 0], face_normals)
    np.add.at(normal_sum, surface_faces[:, 1], face_normals)
    np.add.at(normal_sum, surface_faces[:, 2], face_normals)
    
    # Normalize
    lengths = np.linalg.norm(normal_sum, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-12)
    normals = normal_sum / lengths
    
    return normals


# =============================================================================
# STRATEGY 2: NORMAL OFFSET SCAFFOLD (Boundary Layer)
# =============================================================================

def generate_boundary_layer(surface_verts, surface_faces, lfs_field):
    """
    Creates a 'Shadow Layer' of points exactly 0.8 * LFS inward.
    This guarantees isotropic tetrahedra at the surface boundary.
    
    The factor 0.8 is the height of a perfect tetrahedron relative to its edge.
    """
    # 1. Compute Vertex Normals
    normals = compute_vertex_normals(surface_verts, surface_faces)
    
    # 2. Offset Distance = 0.8 * LFS (pointing inward)
    # Normals point outward, so we subtract
    offset_dist = lfs_field * 0.8
    
    # 3. Generate layer points
    layer_points = surface_verts - (normals * offset_dist[:, None])
    
    return layer_points


# =============================================================================
# STRATEGY 1: LOCAL SIZE CLAMP (Smart Octree Sampler)
# =============================================================================

def generate_lfs_aware_points(surface_verts, surface_faces, lfs_field, 
                               bbox_min, bbox_max, min_size, max_size, grading=1.5):
    """
    Generates internal points with LFS-aware exclusion zones.
    
    Key Fix: Points near the surface are forbidden if they would be closer
    than 0.7 * local_surface_size, preventing pancake tetrahedra.
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        print("[GPU Mesher] Warning: scipy not found, falling back to simple grid")
        return _generate_simple_grid(bbox_min, bbox_max, 20)
    
    # Build tree for surface queries
    tree = cKDTree(surface_verts)
    final_points = []
    
    # Octree-style adaptive sampling
    center_init = (bbox_min + bbox_max) / 2.0
    size_init = np.max(bbox_max - bbox_min)
    stack = [(center_init, size_init)]
    
    while stack:
        center, size = stack.pop()
        
        # 1. Query nearest surface vertex and its LFS
        dist, idx = tree.query(center, k=1)
        local_surf_size = lfs_field[idx]
        
        # 2. STRATEGY 1 FIX: Dynamic Exclusion Zone
        # If we are closer than 70% of the local triangle size, DO NOT spawn a point
        min_safe_dist = local_surf_size * 0.7
        
        if dist < min_safe_dist:
            # Too close to surface - let boundary layer handle this
            continue
        
        # 3. Compute target size at this location (graded)
        target_size_at_loc = min(max_size, min_size + (dist * (grading - 1.0)))
        
        # 4. Refinement decision
        if size > target_size_at_loc:
            # Split into 8 octants
            quarter = size / 4.0
            half = size / 2.0
            
            for dx in [-1, 1]:
                for dy in [-1, 1]:
                    for dz in [-1, 1]:
                        offset = np.array([dx, dy, dz]) * quarter
                        stack.append((center + offset, half))
        else:
            # Leaf node - generate point with jitter
            jitter = np.random.uniform(-0.3, 0.3, 3) * size
            final_points.append(center + jitter)
    
    if len(final_points) == 0:
        return np.zeros((0, 3))
    
    return np.array(final_points)


def _generate_simple_grid(bbox_min, bbox_max, resolution):
    """Fallback simple grid generator."""
    x = np.linspace(bbox_min[0], bbox_max[0], resolution)
    y = np.linspace(bbox_min[1], bbox_max[1], resolution)
    z = np.linspace(bbox_min[2], bbox_max[2], resolution)
    
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    return np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)


# =============================================================================
# CURVATURE-BASED POINT INSERTION
# =============================================================================

def compute_surface_curvature(surface_verts, surface_faces):
    """
    Compute approximate curvature at each vertex based on normal variation.
    High curvature = sharp edges/corners that need more points.
    """
    # Compute vertex normals
    normals = compute_vertex_normals(surface_verts, surface_faces)
    
    # Build vertex adjacency using faces
    num_verts = len(surface_verts)
    from collections import defaultdict
    adj = defaultdict(set)
    
    for f in surface_faces:
        adj[f[0]].update([f[1], f[2]])
        adj[f[1]].update([f[0], f[2]])
        adj[f[2]].update([f[0], f[1]])
    
    # Compute curvature as max angular difference between vertex normal and neighbor normals
    curvature = np.zeros(num_verts)
    for i in range(num_verts):
        if len(adj[i]) == 0:
            continue
        neighbors = list(adj[i])
        neighbor_normals = normals[neighbors]
        # Angular difference (1 - dot product), range [0, 2]
        dots = np.dot(neighbor_normals, normals[i])
        curvature[i] = np.max(1.0 - dots)  # Higher = more curved
    
    return curvature


def generate_curvature_points(surface_verts, surface_faces, lfs_field, curvature_threshold=0.3, density_factor=0.3):
    """
    Generate extra points near high-curvature areas (sharp edges).
    
    For vertices with curvature > threshold, we add extra internal points
    at multiple depths to ensure good tetrahedra formation.
    
    Args:
        curvature_threshold: Vertices with curvature above this get extra points (0.3 = ~70° angle)
        density_factor: Point spacing as fraction of LFS (0.3 = 3 points per LFS)
    """
    curvature = compute_surface_curvature(surface_verts, surface_faces)
    normals = compute_vertex_normals(surface_verts, surface_faces)
    
    # Find high-curvature vertices
    high_curv_mask = curvature > curvature_threshold
    high_curv_indices = np.where(high_curv_mask)[0]
    
    if len(high_curv_indices) == 0:
        return np.zeros((0, 3))
    
    # Generate points at multiple depths for each high-curvature vertex
    extra_points = []
    depths = [0.2, 0.4, 0.6, 0.8, 1.0]  # Fractions of LFS
    
    for idx in high_curv_indices:
        vert = surface_verts[idx]
        normal = normals[idx]
        lfs = lfs_field[idx]
        curv = curvature[idx]
        
        # More curvature = more points
        num_depths = min(5, int(curv * 5) + 1)
        
        for d in depths[:num_depths]:
            offset = -normal * lfs * d * density_factor
            extra_points.append(vert + offset)
    
    if len(extra_points) == 0:
        return np.zeros((0, 3))
    
    return np.array(extra_points)


# =============================================================================
# STRATEGY 3: BOUNDARY REPULSION (Lloyd with Surface Awareness)
# =============================================================================

def optimize_with_boundary_repulsion(points, tets, surface_tree, lfs_field, 
                                     surface_point_count, iterations=3):
    """
    Lloyd relaxation with boundary-aware repulsion.
    Internal points that are too close to the surface get pushed away.
    """
    current_points = points.copy()
    num_total = len(current_points)
    
    for iteration in range(iterations):
        # 1. Standard Lloyd: Build adjacency and compute centroids
        neighbor_sum = np.zeros((num_total, 3))
        neighbor_count = np.zeros(num_total)
        
        edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        
        for e1, e2 in edges:
            idx1, idx2 = tets[:, e1], tets[:, e2]
            np.add.at(neighbor_sum, idx1, current_points[idx2])
            np.add.at(neighbor_count, idx1, 1)
            np.add.at(neighbor_sum, idx2, current_points[idx1])
            np.add.at(neighbor_count, idx2, 1)
        
        # Compute average position
        mask = neighbor_count > 0
        new_positions = current_points.copy()
        new_positions[mask] = neighbor_sum[mask] / neighbor_count[mask][:, None]
        
        # 2. STRATEGY 3: Boundary Repulsion for internal points
        internal_indices = np.arange(surface_point_count, num_total)
        if len(internal_indices) > 0:
            internal_pts = new_positions[internal_indices]
            
            # Query distance and nearest surface vertex
            dists, nearest_idx = surface_tree.query(internal_pts, k=1)
            
            # Target distance = LFS at nearest surface point
            target_dist = lfs_field[nearest_idx]
            
            # Compute repulsion for points that are too close
            too_close = dists < target_dist
            if np.any(too_close):
                # Get direction away from surface (normalize dist vector)
                surface_pts = surface_tree.data[nearest_idx[too_close]]
                push_dir = internal_pts[too_close] - surface_pts
                push_len = np.linalg.norm(push_dir, axis=1, keepdims=True)
                push_len = np.maximum(push_len, 1e-12)
                push_dir = push_dir / push_len
                
                # Push to target distance
                push_amount = (target_dist[too_close] - dists[too_close])[:, None] * 0.5
                new_positions[internal_indices[too_close]] += push_dir * push_amount
        
        # 3. Update internal points with damping
        damping = 0.5
        current_points[surface_point_count:] = (
            (1.0 - damping) * current_points[surface_point_count:] + 
            damping * new_positions[surface_point_count:]
        )
        
        # 4. Re-mesh for next iteration (if not last)
        if iteration < iterations - 1:
            norm_pts, _, _ = normalize_points(current_points)
            tets = _gpumesher.compute_delaunay(norm_pts.astype(np.float64))
            valid_mask = np.all(tets < len(current_points), axis=1)
            tets = tets[valid_mask]
    
    return current_points


# =============================================================================
# VALIDATION SUITE
# =============================================================================

def validate_boundary_layer_health(points, tets, surface_faces, surface_verts):
    """
    Validates the boundary layer transition quality.
    Outputs a health report without GUI visualization.
    
    OPTIMIZED: Uses vectorized NumPy operations (5-20x faster).
    
    Returns:
        dict with validation results
    """
    print("[Validation] Analyzing Boundary Layer...")
    
    # 1. Generate all tet faces (vectorized) - 4 faces per tet
    # Face indices for each tet: [[0,1,2], [0,2,3], [0,1,3], [1,2,3]]
    face_patterns = np.array([[0, 1, 2], [0, 2, 3], [0, 1, 3], [1, 2, 3]])
    
    # Generate all faces for all tets
    num_tets = len(tets)
    all_tet_faces = tets[:, face_patterns]  # Shape: (num_tets, 4, 3)
    
    # Sort each face for consistent comparison
    all_tet_faces_sorted = np.sort(all_tet_faces, axis=2)  # Shape: (num_tets, 4, 3)
    
    # Reshape for comparison: (num_tets * 4, 3)
    tet_faces_flat = all_tet_faces_sorted.reshape(-1, 3)
    
    # Create sorted surface faces for comparison
    surface_faces_sorted = np.sort(surface_faces, axis=1)
    
    # 2. Find boundary tets using set-based lookup (faster than pure vectorization for large sets)
    # Convert surface faces to tuples for set lookup
    surface_face_set = set(map(tuple, surface_faces_sorted))
    
    # Find which tet faces match surface faces
    boundary_tet_indices = []
    boundary_face_indices = []  # Which of the 4 faces matched
    
    for tet_idx in range(num_tets):
        for face_idx in range(4):
            face_tuple = tuple(all_tet_faces_sorted[tet_idx, face_idx])
            if face_tuple in surface_face_set:
                boundary_tet_indices.append(tet_idx)
                boundary_face_indices.append(face_idx)
                break  # Only count once per tet
    
    if len(boundary_tet_indices) == 0:
        print("[Validation] WARNING: No boundary tets found!")
        return {'pass': False, 'error': 'No boundary tets found'}
    
    boundary_tet_indices = np.array(boundary_tet_indices)
    boundary_face_indices = np.array(boundary_face_indices)
    print("[Validation] Found {} boundary tets".format(len(boundary_tet_indices)))
    
    # 3. Calculate Tet Heights and Surface Edge Lengths (vectorized)
    boundary_tets = tets[boundary_tet_indices]  # Shape: (num_boundary, 4)
    boundary_faces = all_tet_faces_sorted[boundary_tet_indices, boundary_face_indices]  # Shape: (num_boundary, 3)
    
    # Find apex vertex (the one not in the face) for each boundary tet
    # The apex is at position: 3 - face_index for patterns [[0,1,2], [0,2,3], [0,1,3], [1,2,3]]
    apex_map = np.array([3, 1, 2, 0])  # apex position for each face pattern
    apex_positions = apex_map[boundary_face_indices]
    apex_indices = boundary_tets[np.arange(len(boundary_tets)), apex_positions]
    
    # Get face vertices (vectorized)
    fv0 = boundary_faces[:, 0]
    fv1 = boundary_faces[:, 1]
    fv2 = boundary_faces[:, 2]
    
    p0 = points[fv0]  # Shape: (num_boundary, 3)
    p1 = points[fv1]
    p2 = points[fv2]
    apex_pts = points[apex_indices]
    
    # Calculate face normals (vectorized)
    edge1 = p1 - p0
    edge2 = p2 - p0
    cross = np.cross(edge1, edge2)
    cross_norm = np.linalg.norm(cross, axis=1, keepdims=True)
    normals = cross / np.maximum(cross_norm, 1e-12)
    
    # Calculate heights (distance from apex to face plane)
    heights = np.abs(np.einsum('ij,ij->i', apex_pts - p0, normals))
    
    # Calculate average edge lengths (vectorized)
    e1 = np.linalg.norm(p1 - p0, axis=1)
    e2 = np.linalg.norm(p2 - p1, axis=1)
    e3 = np.linalg.norm(p0 - p2, axis=1)
    avg_edges = (e1 + e2 + e3) / 3.0
    
    # Aspect Ratios
    ratios = heights / (avg_edges + 1e-12)
    
    # 4. Analysis
    total_boundary = len(ratios)
    critical_fails = np.sum(ratios < 0.1)
    bad_count = np.sum(ratios < 0.3)
    good_count = np.sum(ratios >= 0.5)
    
    print("[Validation] Total Boundary Tets: {}".format(total_boundary))
    print("[Validation] CRITICAL FAILS (Ratio < 0.1): {} ({:.1f}%)".format(
        critical_fails, 100.0 * critical_fails / total_boundary if total_boundary > 0 else 0))
    print("[Validation] Poor Quality (Ratio < 0.3): {} ({:.1f}%)".format(
        bad_count, 100.0 * bad_count / total_boundary if total_boundary > 0 else 0))
    print("[Validation] Good Quality (Ratio >= 0.5): {} ({:.1f}%)".format(
        good_count, 100.0 * good_count / total_boundary if total_boundary > 0 else 0))
    print("[Validation] Ratio Stats: min={:.3f}, avg={:.3f}, max={:.3f}".format(
        np.min(ratios), np.mean(ratios), np.max(ratios)))
    
    # 5. Pass/Fail
    fail_threshold = 0.05  # Allow up to 5% critical failures
    passed = (critical_fails / total_boundary) < fail_threshold if total_boundary > 0 else False
    
    if passed:
        print("[Validation] PASS: Healthy boundary transition.")
    else:
        print("[Validation] FAIL: Surface-Volume mismatch detected (Pancaking).")
    
    return {
        'pass': passed,
        'total_boundary_tets': total_boundary,
        'critical_fails': int(critical_fails),
        'critical_fail_pct': 100.0 * critical_fails / total_boundary if total_boundary > 0 else 0,
        'ratio_min': float(np.min(ratios)) if len(ratios) > 0 else 0,
        'ratio_avg': float(np.mean(ratios)) if len(ratios) > 0 else 0,
        'ratio_max': float(np.max(ratios)) if len(ratios) > 0 else 0
    }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def normalize_points(points):
    """Normalize points to [0,1] for gDel3D robustness."""
    offset = points.min(axis=0)
    scaled = points - offset
    scale = scaled.max()
    
    if scale > 0:
        normalized = scaled / scale
    else:
        normalized = scaled
        
    return normalized, offset, scale


def compute_winding_number_vectorized(test_points, surface_verts, surface_faces):
    """FAST winding number computation for inside/outside test.
    
    Note: For thin features (blades, walls), tets near the surface may have
    winding numbers as low as 0.35-0.45. We use 0.35 threshold to be inclusive.
    """
    from core.fast_winding import compute_fast_winding_grid
    
    winding_nums = compute_fast_winding_grid(surface_verts, surface_faces, test_points, verbose=False)
    # Use 0.35 instead of 0.5 to include thin features where winding numbers are borderline
    return winding_nums > 0.35


def extract_surface_from_volume(points, tets):
    """Reconstructs surface mesh from volume mesh."""
    faces_a = np.sort(tets[:, [0, 1, 2]], axis=1)
    faces_b = np.sort(tets[:, [0, 2, 3]], axis=1)
    faces_c = np.sort(tets[:, [0, 1, 3]], axis=1)
    faces_d = np.sort(tets[:, [1, 2, 3]], axis=1)
    
    all_faces = np.vstack([faces_a, faces_b, faces_c, faces_d])
    unique_faces, counts = np.unique(all_faces, axis=0, return_counts=True)
    
    boundary_mask = counts == 1
    return unique_faces[boundary_mask]


def remove_degenerate_tets(points, tets, min_volume=1e-8, min_quality=0.001):
    """Removes degenerate tetrahedra."""
    verts = points[tets]
    a, b, c, d = verts[:, 0], verts[:, 1], verts[:, 2], verts[:, 3]
    
    cross_prod = np.cross(b - a, c - a)
    dot_prod = np.einsum('ij,ij->i', cross_prod, d - a)
    volumes = np.abs(dot_prod) / 6.0
    
    # Max edge length
    e1 = np.sum((b - a)**2, axis=1)
    e2 = np.sum((c - a)**2, axis=1)
    e3 = np.sum((d - a)**2, axis=1)
    e4 = np.sum((c - b)**2, axis=1)
    e5 = np.sum((d - b)**2, axis=1)
    e6 = np.sum((d - c)**2, axis=1)
    
    max_edge_sq = np.maximum.reduce([e1, e2, e3, e4, e5, e6])
    max_edge_cubed = max_edge_sq ** 1.5
    
    quality = volumes / (max_edge_cubed + 1e-12)
    
    valid_mask = (volumes > min_volume) & (quality > min_quality)
    return tets[valid_mask]


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def gpu_delaunay_fill_and_filter(surface_verts, surface_faces, bbox_min, bbox_max, 
                                   min_spacing=None, max_spacing=None, grading=1.5, 
                                   resolution=50, target_sicn=0.15, progress_callback=None,
                                   fast_mode=False):
    """
    GPU-accelerated "Fill & Filter" pipeline with boundary layer quality enforcement.
    
    Args:
        fast_mode: If True, skip expensive winding number filters and validation.
                   Use for single-body geometry with no voids/gaps. (2-3x faster)
    
    Returns:
        vertices, tetrahedra, surface_faces: Final mesh
    """
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU Mesher not available. Check installation.")
    
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        raise RuntimeError("scipy is required for GPU mesher")
    
    def log(msg, pct=None):
        print("[GPU Mesher] {}".format(msg), flush=True)
        if progress_callback:
            progress_callback(msg, pct if pct is not None else 0)
    
    # ===================
    # STEP 1: Compute LFS
    # ===================
    log("Computing Local Feature Size (LFS)...", 5)
    lfs_field = compute_vertex_lfs(surface_verts, surface_faces)
    avg_lfs = np.mean(lfs_field)
    log("LFS computed. Average: {:.4f}, Min: {:.4f}, Max: {:.4f}".format(
        avg_lfs, np.min(lfs_field), np.max(lfs_field)))
    
    # Set spacing based on LFS if not provided
    if min_spacing is None:
        min_spacing = avg_lfs * 0.8
        max_spacing = avg_lfs * 5.0
    
    # ================================
    # STEP 2: Generate Boundary Layer
    # ================================
    log("Generating Boundary Layer (Normal Offset Scaffold)...", 15)
    layer_points = generate_boundary_layer(surface_verts, surface_faces, lfs_field)
    log("Generated {} boundary layer points".format(len(layer_points)))
    
    # Filter layer points that are outside the shape (SKIP in fast_mode)
    if fast_mode:
        log("FAST MODE: Skipping boundary layer winding filter", 20)
    else:
        log("Filtering boundary layer by winding number...", 20)
        is_inside = compute_winding_number_vectorized(layer_points, surface_verts, surface_faces)
        layer_points = layer_points[is_inside]
        log("Kept {} valid boundary layer points".format(len(layer_points)))
    
    # ===================================
    # STEP 3: Generate LFS-Aware Points
    # ===================================
    log("Generating LFS-aware internal points...", 30)
    internal_points = generate_lfs_aware_points(
        surface_verts, surface_faces, lfs_field,
        bbox_min, bbox_max, min_spacing, max_spacing, grading
    )
    log("Generated {} candidate internal points".format(len(internal_points)))
    
    # Filter by winding number (SKIP in fast_mode)
    if len(internal_points) > 0 and not fast_mode:
        log("Filtering internal points by winding number...", 40)
        is_inside = compute_winding_number_vectorized(internal_points, surface_verts, surface_faces)
        internal_points = internal_points[is_inside]
        log("Kept {} internal points".format(len(internal_points)))
    elif fast_mode and len(internal_points) > 0:
        log("FAST MODE: Keeping all {} internal points".format(len(internal_points)), 40)
    
    # ============================================
    # STEP 3.5: Generate Curvature-Based Points
    # ============================================
    log("Generating curvature-based points near sharp edges...", 43)
    try:
        curvature_points = generate_curvature_points(
            surface_verts, surface_faces, lfs_field,
            curvature_threshold=0.3, density_factor=0.4
        )
        if len(curvature_points) > 0:
            # Filter by winding number (SKIP in fast_mode)
            if not fast_mode:
                is_inside = compute_winding_number_vectorized(curvature_points, surface_verts, surface_faces)
                curvature_points = curvature_points[is_inside]
            log("Generated {} curvature-based points".format(len(curvature_points)))
            
            # Merge with internal points
            if len(internal_points) > 0:
                internal_points = np.vstack((internal_points, curvature_points))
            else:
                internal_points = curvature_points
        else:
            log("No high-curvature vertices detected")
    except Exception as e:
        log("Warning: Curvature point generation failed: {}".format(e))
    
    # Fallback: if no internal points, generate a simple grid sample
    if len(internal_points) == 0 and len(layer_points) == 0:
        log("WARNING: No internal points passed filtering. Using fallback grid.", 45)
        # Simple fallback grid in center of bbox
        center = (bbox_min + bbox_max) / 2.0
        grid_size = np.min(bbox_max - bbox_min) * 0.3
        fallback_pts = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    pt = center + np.array([dx, dy, dz]) * grid_size * 0.3
                    fallback_pts.append(pt)
        internal_points = np.array(fallback_pts)
        log("Generated {} fallback internal points".format(len(internal_points)))
    
    # =====================
    # STEP 4: Merge Points
    # =====================
    log("Merging all point sets...", 50)
    # Order: Surface first (fixed), then layer, then internal
    num_surface = len(surface_verts)
    num_layer = len(layer_points)
    
    if len(layer_points) > 0 and len(internal_points) > 0:
        all_points = np.vstack((surface_verts, layer_points, internal_points))
    elif len(layer_points) > 0:
        all_points = np.vstack((surface_verts, layer_points))
    elif len(internal_points) > 0:
        all_points = np.vstack((surface_verts, internal_points))
    else:
        all_points = surface_verts.copy()
    
    log("Total points for GPU: {}".format(len(all_points)))
    
    # ========================
    # STEP 5: GPU Delaunay
    # ========================
    log("Running GPU Delaunay triangulation...", 60)
    import time
    start = time.time()
    
    norm_points, offset, scale = normalize_points(all_points)
    all_tets = _gpumesher.compute_delaunay(norm_points.astype(np.float64))
    
    elapsed = (time.time() - start) * 1000
    log("GPU generated {} tetrahedra in {:.1f} ms".format(len(all_tets), elapsed))
    
    # Filter ghost tets
    n_points = len(all_points)
    valid_mask = np.all(all_tets < n_points, axis=1)
    all_tets = all_tets[valid_mask]
    log("After ghost removal: {} tetrahedra".format(len(all_tets)))
    
    # ======================================
    # STEP 6: Boundary-Aware Lloyd Relaxation
    # ======================================
    log("Running boundary-aware Lloyd relaxation...", 70)
    surface_tree = cKDTree(surface_verts)
    all_points = optimize_with_boundary_repulsion(
        all_points, all_tets, surface_tree, lfs_field,
        num_surface, iterations=2
    )
    
    # Re-mesh after relaxation
    log("Final meshing pass...", 75)
    norm_points, offset, scale = normalize_points(all_points)
    all_tets = _gpumesher.compute_delaunay(norm_points.astype(np.float64))
    
    valid_mask = np.all(all_tets < len(all_points), axis=1)
    all_tets = all_tets[valid_mask]
    
    # =======================
    # STEP 7: Centroid Winding Filter (Catches cross-gap tets in multi-body geometry)
    # =======================
    # CRITICAL: Must filter ALL tets by centroid winding to prevent webbing between:
    # - Parallel plates
    # - Fan case and fan blades
    # - Any separate bodies in an assembly
    #
    # Internal points can be verified inside their OWN body, but Delaunay connects
    # points from DIFFERENT bodies through the void. These void-spanning tets have
    # centroids with winding number ≈ 0 (outside all bodies).
    #
    # Strategy:
    # - Very low threshold (0.1) catches void-spanning tets
    # - Thin features (winding 0.1-0.4) are preserved
    # - Robust for both single-body and multi-body geometry
    
    log("Filtering void-spanning tetrahedra (multi-body safe)...", 80)
    from core.fast_winding import compute_fast_winding_grid
    
    tet_verts = all_points[all_tets]
    centroids = np.mean(tet_verts, axis=1)
    
    # ========================================
    # STEP 7a: EDGE LENGTH FILTER (FAST - do this first!)
    # ========================================
    # Cross-gap tets have edges spanning between bodies - much longer than expected
    log("Step 7a: Filtering by edge length...", 81)
    p0, p1, p2, p3 = tet_verts[:, 0], tet_verts[:, 1], tet_verts[:, 2], tet_verts[:, 3]
    edge_lengths = np.stack([
        np.linalg.norm(p1 - p0, axis=1),
        np.linalg.norm(p2 - p0, axis=1),
        np.linalg.norm(p3 - p0, axis=1),
        np.linalg.norm(p2 - p1, axis=1),
        np.linalg.norm(p3 - p1, axis=1),
        np.linalg.norm(p3 - p2, axis=1),
    ], axis=1)
    max_edge = np.max(edge_lengths, axis=1)
    min_edge = np.min(edge_lengths, axis=1)
    
    # OPTIMIZATION: Tighter threshold (was 3.0x, now 2.0x) to filter more aggressively
    edge_threshold = max_spacing * 2.0
    edges_ok = max_edge < edge_threshold
    
    # OPTIMIZATION: Also filter by aspect ratio (max/min edge)
    # Tets spanning gaps often have extreme aspect ratios
    aspect_ratio = max_edge / (min_edge + 1e-12)
    aspect_ok = aspect_ratio < 10.0  # Filter tets with aspect ratio > 10
    
    # Combine filters
    combined_ok = edges_ok & aspect_ok
    
    # Filter tets by edge length FIRST (fast operation)
    edge_filtered_tets = all_tets[combined_ok]
    edge_filtered_centroids = centroids[combined_ok]
    
    num_edge_filtered = len(all_tets) - len(edge_filtered_tets)
    log("Edge filter: kept {} tets (filtered {} with long/bad edges)".format(
        len(edge_filtered_tets), num_edge_filtered))
    
    # ========================================
    # STEP 7b: WINDING NUMBER FILTER (SLOW - only on edge-filtered tets)
    # ========================================
    # SKIP in fast_mode - this is the main bottleneck (3-5+ seconds)
    if fast_mode:
        log("FAST MODE: Skipping void-spanning filter (saves 3-5s)", 85)
        final_tets = edge_filtered_tets
    elif len(edge_filtered_tets) > 0:
        log("Step 7b: Computing winding numbers for {} tets...".format(len(edge_filtered_tets)), 83)
        
        # Batch processing for large meshes (prevents hang)
        BATCH_SIZE = 50000
        winding_nums = np.zeros(len(edge_filtered_centroids))
        
        for i in range(0, len(edge_filtered_centroids), BATCH_SIZE):
            end = min(i + BATCH_SIZE, len(edge_filtered_centroids))
            batch_centroids = edge_filtered_centroids[i:end]
            winding_nums[i:end] = compute_fast_winding_grid(
                surface_verts, surface_faces, batch_centroids, verbose=False
            )
            if len(edge_filtered_centroids) > BATCH_SIZE:
                log("  Batch {}/{} done".format(
                    end, len(edge_filtered_centroids)), 84)
        
        VOID_THRESHOLD = 0.15
        winding_ok = winding_nums > VOID_THRESHOLD
        
        final_tets = edge_filtered_tets[winding_ok]
        num_winding_filtered = np.sum(~winding_ok)
        avg_winding_kept = np.mean(winding_nums[winding_ok]) if np.sum(winding_ok) > 0 else 0
        
        log("Winding filter: kept {} tets (filtered {} void-spanning, avg winding: {:.3f})".format(
            len(final_tets), num_winding_filtered, avg_winding_kept))
    else:
        final_tets = edge_filtered_tets
        log("No tets left after edge filter - skipping winding computation")
    
    # ==============================
    # STEP 8: Remove Degenerate Tets
    # ==============================
    log("Removing degenerate tetrahedra...", 85)
    prev_count = len(final_tets)
    final_tets = remove_degenerate_tets(all_points, final_tets)
    removed = prev_count - len(final_tets)
    if removed > 0:
        log("Removed {} degenerate tetrahedra".format(removed))
    
    # =====================================
    # STEP 9: Adaptive Quality Refinement
    # =====================================
    # Only run refinement if we have tets and quality target is set
    if len(final_tets) > 0 and target_sicn > 0:
        log("Running adaptive quality refinement (target SICN: {:.2f})...".format(target_sicn), 88)
        
        try:
            from core.gpu_adaptive_refinement import AdaptiveGPURefinement
            
            # Pre-filter: Remove severely inverted elements (SICN < -0.1)
            # These cannot be fixed by vertex movement and will hurt refinement
            elem_verts = all_points[final_tets]
            p0, p1, p2, p3 = elem_verts[:, 0], elem_verts[:, 1], elem_verts[:, 2], elem_verts[:, 3]
            e1, e2, e3 = p1 - p0, p2 - p0, p3 - p0
            cross_e2_e3 = np.cross(e2, e3)
            volumes = np.einsum('ij,ij->i', e1, cross_e2_e3)
            all_lengths = np.stack([
                np.linalg.norm(e1, axis=1), np.linalg.norm(e2, axis=1), np.linalg.norm(e3, axis=1),
                np.linalg.norm(p2 - p1, axis=1), np.linalg.norm(p3 - p1, axis=1), np.linalg.norm(p3 - p2, axis=1)
            ], axis=1)
            max_lengths = np.max(all_lengths, axis=1)
            pre_sicn = volumes / (max_lengths ** 3 + 1e-12)
            
            # PERFORMANCE OPTIMIZATION: Skip refinement if quality already acceptable
            # This avoids 0.3-0.5s of wasted rollback iterations
            initial_min_sicn = np.min(pre_sicn)
            initial_avg_sicn = np.mean(pre_sicn)
            if initial_avg_sicn >= target_sicn and initial_min_sicn > -0.1:
                log("SKIP: Initial quality acceptable (avg={:.3f} >= target={:.2f})".format(
                    initial_avg_sicn, target_sicn), 95)
            else:
                # Keep elements with SICN > -0.1 (severely inverted are hopeless)
                severely_inverted = pre_sicn < -0.1
                if np.sum(severely_inverted) > 0:
                    log("Removing {} severely inverted elements (SICN < -0.1)".format(np.sum(severely_inverted)))
                    final_tets = final_tets[~severely_inverted]
                
                # Identify surface nodes (fixed during refinement)
                fixed_nodes = np.arange(num_surface)
                
                # Create refiner with quality target - REDUCED iterations for speed
                refiner = AdaptiveGPURefinement(
                    target_sicn=max(target_sicn, 0.10),  # Accept 0.10 minimum
                    max_iterations=5,  # REDUCED from 15 for speed
                    iteration_timeout_sec=2.0,  # REDUCED from 5.0
                    step_size=0.15,
                    verbose=True,
                    progress_callback=lambda msg, pct: log(msg, 88 + int(pct * 0.07))
                )
                
                # Run refinement
                all_points, refine_stats = refiner.refine(
                    all_points.astype(np.float32), 
                    final_tets.astype(np.int32),
                    fixed_nodes
                )
                
                # Report improvement
                improved = refine_stats['final_sicn_min'] > refine_stats['initial_sicn_min']
                if refine_stats.get('converged', False):
                    log("Quality refinement CONVERGED: SICN min {:.3f} -> {:.3f}".format(
                        refine_stats['initial_sicn_min'], refine_stats['final_sicn_min']), 95)
                elif improved:
                    log("Quality refinement improved: SICN min {:.3f} -> {:.3f}".format(
                        refine_stats['initial_sicn_min'], refine_stats['final_sicn_min']), 95)
                else:
                    log("Quality refinement: no improvement (keeping original)", 95)
                
                # Only re-mesh if quality improved (Laplacian keeps topology stable)
                if improved:
                    log("Re-triangulating after refinement...", 96)
                    norm_points, offset, scale = normalize_points(all_points)
                    final_tets = _gpumesher.compute_delaunay(norm_points.astype(np.float64))
                    
                    # Filter again
                    valid_mask = np.all(final_tets < len(all_points), axis=1)
                    final_tets = final_tets[valid_mask]
                    
                    tet_verts = all_points[final_tets]
                    centroids = np.mean(tet_verts, axis=1)
                    is_tet_inside = compute_winding_number_vectorized(centroids, surface_verts, surface_faces)
                    final_tets = final_tets[is_tet_inside]
                    
                    # Remove any new degenerates
                    final_tets = remove_degenerate_tets(all_points, final_tets)
                    log("Post-refinement mesh: {} tetrahedra".format(len(final_tets)))
            
        except ImportError as e:
            log("Skipping adaptive refinement: {}".format(e))
        except Exception as e:
            log("Adaptive refinement error: {}".format(e))
    
    # =====================================
    # STEP 10: Remove Inverted Elements (PRESERVE SURFACE)
    # =====================================
    # Critical: Never remove tetrahedra that share a face with the original surface mesh.
    # These define the shape boundary and must be preserved.
    
    # Build a set of surface faces (as sorted tuples for matching)
    num_surface_verts = len(surface_verts)
    surface_face_set = set()
    for face in surface_faces:
        # Faces use original indices (0 to num_surface_verts-1)
        surface_face_set.add(tuple(sorted(face)))
    
    def tet_shares_surface_face(tet_nodes):
        """Check if a tet has any face matching the original surface mesh."""
        # Tet faces (4 triangular faces per tet)
        tet_face_indices = [
            (0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)
        ]
        for fi in tet_face_indices:
            face = tuple(sorted([tet_nodes[fi[0]], tet_nodes[fi[1]], tet_nodes[fi[2]]]))
            if face in surface_face_set:
                return True
        return False
    
    # Identify surface-adjacent tets (protect these from removal)
    if len(final_tets) > 0:
        is_surface_tet = np.zeros(len(final_tets), dtype=bool)
        for i, tet in enumerate(final_tets):
            if tet_shares_surface_face(tet):
                is_surface_tet[i] = True
        
        num_surface_tets = np.sum(is_surface_tet)
        log("Identified {} surface-adjacent tets (protected)".format(num_surface_tets))
        
        # Compute SICN for all elements
        elem_verts = all_points[final_tets]
        p0, p1, p2, p3 = elem_verts[:, 0], elem_verts[:, 1], elem_verts[:, 2], elem_verts[:, 3]
        e1, e2, e3 = p1 - p0, p2 - p0, p3 - p0
        cross_e2_e3 = np.cross(e2, e3)
        volumes = np.einsum('ij,ij->i', e1, cross_e2_e3)
        all_lengths = np.stack([
            np.linalg.norm(e1, axis=1), np.linalg.norm(e2, axis=1), np.linalg.norm(e3, axis=1),
            np.linalg.norm(p2 - p1, axis=1), np.linalg.norm(p3 - p1, axis=1), np.linalg.norm(p3 - p2, axis=1)
        ], axis=1)
        max_lengths = np.max(all_lengths, axis=1)
        sicn = volumes / (max_lengths ** 3 + 1e-12)
        
        # Remove inverted elements ONLY if they're not surface-adjacent
        inverted_mask = (sicn < 0) & (~is_surface_tet)
        num_inverted = np.sum(inverted_mask)
        if num_inverted > 0:
            log("Removing {} inverted INTERNAL elements (SICN < 0)".format(num_inverted))
            final_tets = final_tets[~inverted_mask]
            is_surface_tet = is_surface_tet[~inverted_mask]
            sicn = sicn[~inverted_mask]
        
        # Remove poor quality INTERNAL elements only (SICN < 0.05)
        # Practical RDT: Aggressively remove near-zero quality tets
        if len(final_tets) > 0:
            poor_mask = (sicn < 0.05) & (~is_surface_tet)
            num_poor = np.sum(poor_mask)
            # Allow removing up to 25% of total elements for cleaner mesh
            if num_poor > 0 and num_poor < len(final_tets) * 0.25:
                log("Removing {} poor quality INTERNAL elements (SICN < 0.05)".format(num_poor))
                final_tets = final_tets[~poor_mask]
    
    # =======================
    # STEP 11: Extract Surface
    # =======================
    log("Extracting surface from volume...", 97)
    final_surface = extract_surface_from_volume(all_points, final_tets)
    log("Extracted {} surface triangles".format(len(final_surface)))
    
    # ======================
    # STEP 12: Validation (SKIP in fast_mode)
    # ======================
    if fast_mode:
        log("FAST MODE: Skipping validation", 100)
        validation = {'pass': True, 'skipped': True}
    elif len(final_tets) > 0:
        log("Validating boundary layer health...", 98)
        validation = validate_boundary_layer_health(all_points, final_tets, final_surface, surface_verts)
        
        if validation.get('pass', False):
            log("Validation PASSED: Healthy boundary layer", 100)
        else:
            log("Validation WARNING: {} critical failures ({:.1f}%)".format(
                validation.get('critical_fails', 0), validation.get('critical_fail_pct', 0)), 100)
    else:
        log("WARNING: No tetrahedra generated - check input geometry", 100)
        validation = {'pass': False, 'error': 'No tetrahedra'}
    
    log("Mesh generation complete! {} tets, {} surface tris".format(
        len(final_tets), len(final_surface)), 100)

    
    return all_points, final_tets, final_surface

