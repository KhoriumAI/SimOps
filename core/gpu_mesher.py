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
    
    Returns:
        lfs_field: (N,) array of local sizes per vertex
    """
    num_verts = len(surface_verts)
    edge_sum = np.zeros(num_verts)
    edge_count = np.zeros(num_verts)
    
    # For each triangle, compute edge lengths
    v0 = surface_verts[surface_faces[:, 0]]
    v1 = surface_verts[surface_faces[:, 1]]
    v2 = surface_verts[surface_faces[:, 2]]
    
    e01 = np.linalg.norm(v1 - v0, axis=1)
    e12 = np.linalg.norm(v2 - v1, axis=1)
    e20 = np.linalg.norm(v0 - v2, axis=1)
    
    # Sum edge lengths per vertex
    for i, face in enumerate(surface_faces):
        avg_len = (e01[i] + e12[i] + e20[i]) / 3.0
        edge_sum[face[0]] += avg_len
        edge_sum[face[1]] += avg_len
        edge_sum[face[2]] += avg_len
        edge_count[face[0]] += 1
        edge_count[face[1]] += 1
        edge_count[face[2]] += 1
    
    # Compute average (avoid div/0)
    lfs_field = np.where(edge_count > 0, edge_sum / edge_count, 1.0)
    
    return lfs_field


def compute_vertex_normals(surface_verts, surface_faces):
    """
    Computes per-vertex normals by averaging connected face normals.
    
    Returns:
        normals: (N, 3) array of unit normals per vertex
    """
    num_verts = len(surface_verts)
    normal_sum = np.zeros((num_verts, 3))
    
    # Compute face normals
    v0 = surface_verts[surface_faces[:, 0]]
    v1 = surface_verts[surface_faces[:, 1]]
    v2 = surface_verts[surface_faces[:, 2]]
    
    face_normals = np.cross(v1 - v0, v2 - v0)
    
    # Accumulate to vertices
    for i, face in enumerate(surface_faces):
        normal_sum[face[0]] += face_normals[i]
        normal_sum[face[1]] += face_normals[i]
        normal_sum[face[2]] += face_normals[i]
    
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
    
    Returns:
        dict with validation results
    """
    print("[Validation] Analyzing Boundary Layer...")
    
    # 1. Identify Boundary Tets
    # A tet is a boundary tet if any of its 4 faces matches a surface face
    tet_faces_list = []
    for tet in tets:
        # Generate 4 faces per tet (sorted for comparison)
        f0 = tuple(sorted([tet[0], tet[1], tet[2]]))
        f1 = tuple(sorted([tet[0], tet[2], tet[3]]))
        f2 = tuple(sorted([tet[0], tet[1], tet[3]]))
        f3 = tuple(sorted([tet[1], tet[2], tet[3]]))
        tet_faces_list.append([f0, f1, f2, f3])
    
    # Create set of surface faces for fast lookup
    surface_face_set = set()
    for face in surface_faces:
        surface_face_set.add(tuple(sorted(face)))
    
    # Find boundary tets and their associated surface face
    boundary_tet_indices = []
    boundary_surface_faces = []
    
    for i, tet_faces in enumerate(tet_faces_list):
        for face in tet_faces:
            if face in surface_face_set:
                boundary_tet_indices.append(i)
                boundary_surface_faces.append(face)
                break
    
    if len(boundary_tet_indices) == 0:
        print("[Validation] WARNING: No boundary tets found!")
        return {'pass': False, 'error': 'No boundary tets found'}
    
    boundary_tet_indices = np.array(boundary_tet_indices)
    print("[Validation] Found {} boundary tets".format(len(boundary_tet_indices)))
    
    # 2. Calculate Tet Heights and Surface Edge Lengths
    boundary_tets = tets[boundary_tet_indices]
    
    heights = []
    edge_lengths = []
    ratios = []
    
    for i, tet_idx in enumerate(boundary_tet_indices):
        tet = tets[tet_idx]
        face = boundary_surface_faces[i]
        
        # Get the apex (vertex not in the surface face)
        apex = None
        for v in tet:
            if v not in face:
                apex = v
                break
        
        if apex is None:
            continue
        
        # Get face vertices
        fv0, fv1, fv2 = face
        p0, p1, p2 = points[fv0], points[fv1], points[fv2]
        apex_pt = points[apex]
        
        # Calculate face area
        cross = np.cross(p1 - p0, p2 - p0)
        face_area = np.linalg.norm(cross) / 2.0
        
        # Calculate height (distance from apex to face plane)
        normal = cross / (np.linalg.norm(cross) + 1e-12)
        height = abs(np.dot(apex_pt - p0, normal))
        
        # Calculate average edge length of surface face
        e1 = np.linalg.norm(p1 - p0)
        e2 = np.linalg.norm(p2 - p1)
        e3 = np.linalg.norm(p0 - p2)
        avg_edge = (e1 + e2 + e3) / 3.0
        
        heights.append(height)
        edge_lengths.append(avg_edge)
        
        # Aspect Ratio = Height / EdgeLength
        # Ideal: ~0.8 for equilateral
        # Pancake: < 0.1
        ratio = height / (avg_edge + 1e-12)
        ratios.append(ratio)
    
    ratios = np.array(ratios)
    heights = np.array(heights)
    edge_lengths = np.array(edge_lengths)
    
    # 3. Analysis
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
    
    # 4. Pass/Fail
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
    """FAST winding number computation."""
    from core.fast_winding import compute_fast_winding_grid
    
    winding_nums = compute_fast_winding_grid(surface_verts, surface_faces, test_points, verbose=False)
    return winding_nums > 0.5


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
                                   resolution=50, target_sicn=0.15, progress_callback=None):
    """
    GPU-accelerated "Fill & Filter" pipeline with boundary layer quality enforcement.
    
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
    
    # Filter layer points that are outside the shape
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
    
    # Filter by winding number
    if len(internal_points) > 0:
        log("Filtering internal points by winding number...", 40)
        is_inside = compute_winding_number_vectorized(internal_points, surface_verts, surface_faces)
        internal_points = internal_points[is_inside]
        log("Kept {} internal points".format(len(internal_points)))
    
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
    # STEP 7: Winding Filter
    # =======================
    log("Filtering tetrahedra by winding number...", 80)
    tet_verts = all_points[all_tets]
    centroids = np.mean(tet_verts, axis=1)
    
    is_tet_inside = compute_winding_number_vectorized(centroids, surface_verts, surface_faces)
    final_tets = all_tets[is_tet_inside]
    log("Kept {} tetrahedra (filtered {} outside)".format(
        len(final_tets), len(all_tets) - len(final_tets)))
    
    # ==============================
    # STEP 8: Remove Degenerate Tets
    # ==============================
    log("Removing degenerate tetrahedra...", 90)
    prev_count = len(final_tets)
    final_tets = remove_degenerate_tets(all_points, final_tets)
    removed = prev_count - len(final_tets)
    if removed > 0:
        log("Removed {} degenerate tetrahedra".format(removed))
    
    # =======================
    # STEP 9: Extract Surface
    # =======================
    log("Extracting surface from volume...", 95)
    final_surface = extract_surface_from_volume(all_points, final_tets)
    log("Extracted {} surface triangles".format(len(final_surface)))
    
    # ======================
    # STEP 10: Validation
    # ======================
    log("Validating boundary layer health...", 98)
    if len(final_tets) > 0:
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
