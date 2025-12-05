"""
GPU Delaunay Mesher Integration
================================

Implements the "Fill & Filter" pipeline for GPU-accelerated tetrahedral meshing.

Pipeline:
1. Generate surface mesh (CPU)
2. Sample internal points using winding number
3. GPU Delaunay triangulation on all points
4. Filter tetrahedra by centroid winding number
"""

import numpy as np
import sys
import os
from pathlib import Path

# Add GPU mesher to path
# core/gpu_mesher.py -> parent = core/ -> parent = MeshPackageLean/
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
    print(f"WARNING: GPU Mesher not available: {e}")
    GPU_AVAILABLE = False


def generate_jittered_points(bbox_min, bbox_max, resolution=50):
    """Generate jittered grid points within bounding box."""
    # Create regular grid
    x = np.linspace(bbox_min[0], bbox_max[0], resolution)
    y = np.linspace(bbox_min[1], bbox_max[1], resolution)
    z = np.linspace(bbox_min[2], bbox_max[2], resolution)
    
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    
    # Add jitter (10% of grid spacing)
    spacing = (bbox_max - bbox_min) / resolution
    jitter = np.random.uniform(-0.1, 0.1, grid_points.shape) * spacing
    
    return grid_points + jitter


def normalize_points(points):
    """Normalize points to [0,1] for gDel3D robustness.
    
    Returns:
        normalized_points, offset, scale
    """
    offset = points.min(axis=0)
    scaled = points - offset
    scale = scaled.max()
    
    if scale > 0:
        normalized = scaled / scale
    else:
        normalized = scaled
        
    return normalized, offset, scale


def denormalize_points(normalized_points, offset, scale):
    """Reverse normalization."""
    return normalized_points * scale + offset


def compute_winding_number_vectorized(test_points, surface_verts, surface_faces):
    """FAST winding number using Barnes-Hut octree implementation.
    
    Args:
        test_points: (N, 3) array of points to test
        surface_verts: (M, 3) array of surface vertices
        surface_faces: (K, 3) array of triangle indices
        
    Returns:
        (N,) boolean array: True if inside, False if outside
    """
    from core.fast_winding import compute_fast_winding_grid
    
    # Get winding numbers (normalized by 4π, so inside ≈ 1, outside ≈ 0)
    winding_nums = compute_fast_winding_grid(surface_verts, surface_faces, test_points, verbose=True)
    
    # Threshold: > 0.5 is inside
    return winding_nums > 0.5


def filter_internal_points_by_proximity(internal_points, surface_points, min_dist):
    """
    Removes internal points that are too close to the surface.
    This prevents 'slivers' (flat tetrahedra) at the boundary.
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        print("[GPU Mesher] Warning: scipy not found, skipping proximity filter")
        return internal_points
    
    if len(internal_points) == 0:
        return internal_points

    # 1. Build fast lookup tree for surface points
    tree = cKDTree(surface_points)
    
    # 2. Query distance for every internal point
    # k=1 returns (distances, indices) to the nearest surface point
    distances, _ = tree.query(internal_points, k=1)
    
    # 3. Keep only points that are far enough away
    mask = distances > min_dist
    
    filtered_points = internal_points[mask]
    
    return filtered_points


def remove_degenerate_tets(points, tets, min_volume=1e-5, min_quality=0.002):
    """
    Removes tetrahedra that are either:
    1. Too small (Volume < min_volume)
    2. Too flat (Volume / max_edge_length^3 < min_quality)
    
    This acts as a 'Mesh Erosion' step to remove the bad surface skin.
    """
    # 1. Get vertices: (N, 4, 3)
    verts = points[tets]
    a, b, c, d = verts[:, 0], verts[:, 1], verts[:, 2], verts[:, 3]
    
    # 2. Calculate Signed Volume
    # Vol = dot(a-d, cross(b-d, c-d)) / 6
    cross_prod = np.cross(b - a, c - a)
    dot_prod = np.einsum('ij,ij->i', cross_prod, d - a)
    volumes = np.abs(dot_prod) / 6.0
    
    # 3. Calculate Max Edge Length (Squared) for every tet
    # Edges: AB, AC, AD, BC, BD, CD
    e1 = np.sum((b - a)**2, axis=1)
    e2 = np.sum((c - a)**2, axis=1)
    e3 = np.sum((d - a)**2, axis=1)
    e4 = np.sum((c - b)**2, axis=1)
    e5 = np.sum((d - b)**2, axis=1)
    e6 = np.sum((d - c)**2, axis=1)
    
    # We want max edge length cubed
    max_edge_sq = np.maximum.reduce([e1, e2, e3, e4, e5, e6])
    max_edge_cubed = max_edge_sq ** 1.5
    
    # 4. Compute 'Collapse Ratio' (Simplified Quality)
    # A perfect tet has high volume relative to its edges.
    # A sliver has tiny volume relative to long edges.
    quality = volumes / (max_edge_cubed + 1e-12) # Add epsilon to prevent div/0
    
    # 5. Filter
    # Condition: Keep if Volume is decent AND Quality is decent
    valid_mask = (volumes > min_volume) & (quality > min_quality)
    
    clean_tets = tets[valid_mask]
    
    return clean_tets


def extract_surface_from_volume(points, tets):
    """
    Reconstructs the surface mesh from the volume mesh.
    A face is on the surface if it belongs to EXACTLY ONE tetrahedron.
    """
    # 1. Create a list of all faces from all tetrahedra
    # Each tet has 4 faces: (0,1,2), (0,2,3), (0,1,3), (1,2,3)
    # We sort the indices of each face so (1,2,0) becomes (0,1,2) for matching
    faces_a = np.sort(tets[:, [0, 1, 2]], axis=1)
    faces_b = np.sort(tets[:, [0, 2, 3]], axis=1)
    faces_c = np.sort(tets[:, [0, 1, 3]], axis=1)
    faces_d = np.sort(tets[:, [1, 2, 3]], axis=1)
    
    all_faces = np.vstack([faces_a, faces_b, faces_c, faces_d])
    
    # 2. Find unique faces and count occurrences
    # 'return_counts=True' tells us how many tets share this face
    unique_faces, counts = np.unique(all_faces, axis=0, return_counts=True)
    
    # 3. Filter: Boundary faces appear exactly ONCE
    boundary_mask = counts == 1
    surface_faces = unique_faces[boundary_mask]
    
    return surface_faces


def generate_hessian_sized_points(surface_points, min_b, max_b, min_size, max_size, grading=1.5):
    """
    Generates points based on a 'Sizing Function' relative to the boundary.
    Simulates a Hessian-based sizing field for isotropic meshers.
    
    Args:
        min_size: Target element size at the surface.
        max_size: Target element size in the core.
        grading: How fast elements grow. 
                 1.2 = High quality, slow transition.
                 2.0 = Low count, fast transition (Aggressive).
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        print("[GPU Mesher] Warning: scipy not found, falling back to uniform grid")
        return generate_jittered_points(min_b, max_b, 20)
    
    tree = cKDTree(surface_points)
    final_points = []
    
    # Using a stack instead of recursion to avoid depth limits
    center_init = (min_b + max_b) / 2.0
    size_init = np.max(max_b - min_b)
    stack = [(center_init, size_init)]
    
    while stack:
        center, size = stack.pop()
        
        # 1. Distance Query (Hessian Metric)
        dist, _ = tree.query(center, k=1)
        
        # 2. Compute Target Size at this location
        # S(x) = min_size + (grading * distance)
        target_size_at_loc = min(max_size, min_size + (dist * (grading - 1.0)))
        
        # 3. Refinement Decision
        if size > target_size_at_loc:
            # Split
            quarter = size / 4.0
            half = size / 2.0
            
            for dx in [-1, 1]:
                for dy in [-1, 1]:
                    for dz in [-1, 1]:
                        offset = np.array([dx, dy, dz]) * quarter
                        stack.append((center + offset, half))
        else:
            # 4. Point Generation (Leaf Node)
            # Exclusion Zone: Only generate if far enough from surface
            if dist > (size * 0.7):
                jitter = np.random.uniform(-0.4, 0.4, 3) * size
                final_points.append(center + jitter)

    return np.array(final_points)


    return np.array(final_points)


def optimize_points_lloyd(points, tets, surface_point_count, iterations=3):
    """
    Iteratively smoothes the internal points to improve mesh quality.
    (Lloyd's Relaxation via Laplacian Smoothing)
    """
    current_points = points.copy()
    num_total = len(current_points)
    
    for i in range(iterations):
        # 1. Build Adjacency Graph
        neighbor_sum = np.zeros((num_total, 3))
        neighbor_count = np.zeros(num_total)
        
        edges = [
            (0, 1), (0, 2), (0, 3), 
            (1, 2), (1, 3), (2, 3)
        ]
        
        for e1, e2 in edges:
            idx1, idx2 = tets[:, e1], tets[:, e2]
            np.add.at(neighbor_sum, idx1, current_points[idx2])
            np.add.at(neighbor_count, idx1, 1)
            np.add.at(neighbor_sum, idx2, current_points[idx1])
            np.add.at(neighbor_count, idx2, 1)
            
        # 2. Compute Average Position
        mask = neighbor_count > 0
        new_positions = current_points.copy()
        new_positions[mask] = neighbor_sum[mask] / neighbor_count[mask][:, None]
        
        # 3. Update ONLY Internal Points
        damping = 0.5
        current_points[surface_point_count:] = (
            (1.0 - damping) * current_points[surface_point_count:] + 
            damping * new_positions[surface_point_count:]
        )
        
        # 4. RE-MESH
        if i < iterations - 1:
            norm_pts, _, _ = normalize_points(current_points)
            tets = _gpumesher.compute_delaunay(norm_pts.astype(np.float64))
            # Filter ghosts for next iteration
            valid_mask = np.all(tets < len(current_points), axis=1)
            tets = tets[valid_mask]
            
    return current_points


def iterative_refinement_loop(surface_verts, surface_faces, initial_points, target_sicn=0.15, max_iters=5):
    """
    Refines the mesh by iteratively inserting points into bad tetrahedra.
    This breaks 'slivers' and improves minimum quality.
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return initial_points, None
        
    current_points = initial_points.copy()
    final_tets = None
    
    for i in range(max_iters):
        # 1. Mesh
        norm_pts, offset, scale = normalize_points(current_points)
        tets = _gpumesher.compute_delaunay(norm_pts.astype(np.float64))
        
        # Filter ghosts
        valid_mask = np.all(tets < len(current_points), axis=1)
        tets = tets[valid_mask]
        final_tets = tets
        
        # 2. Compute Quality (SICN approximation)
        verts = current_points[tets]
        a, b, c, d = verts[:,0], verts[:,1], verts[:,2], verts[:,3]
        
        # Vol
        cross = np.cross(b-a, c-a)
        vol = np.abs(np.einsum('ij,ij->i', cross, d-a)) / 6.0
        
        # Max Edge Sq
        e1 = np.sum((b-a)**2, axis=1); e2 = np.sum((c-a)**2, axis=1)
        e3 = np.sum((d-a)**2, axis=1); e4 = np.sum((c-b)**2, axis=1)
        e5 = np.sum((d-b)**2, axis=1); e6 = np.sum((d-c)**2, axis=1)
        max_edge_sq = np.maximum.reduce([e1,e2,e3,e4,e5,e6])
        
        # Quality Proxy
        max_edge_sq = np.maximum(max_edge_sq, 1e-12)
        qual = (vol / (max_edge_sq**1.5)) * 8.48 
        
        min_q = np.min(qual)
        
        if min_q >= target_sicn:
            break
            
        # 3. Identify Bad Elements
        bad_mask = (qual < target_sicn) & (vol > 1e-9)
        bad_indices = np.where(bad_mask)[0]
        
        if len(bad_indices) == 0:
            break
            
        # 4. Generate New Points (Steiner Points)
        bad_verts = verts[bad_indices]
        new_candidates = np.mean(bad_verts, axis=1)
        
        # 5. Filter Candidates
        # A. Winding Check (Must be inside)
        # Only check if we have candidates
        if len(new_candidates) > 0:
            is_inside = compute_winding_number_vectorized(new_candidates, surface_verts, surface_faces)
            new_candidates = new_candidates[is_inside]
        
        if len(new_candidates) == 0:
            break
            
        # B. Density Check
        tree = cKDTree(current_points)
        dists, _ = tree.query(new_candidates, k=1)
        
        # Heuristic: distance > 0.5
        density_mask = dists > 0.5 
        valid_new_points = new_candidates[density_mask]
        
        if len(valid_new_points) == 0:
            break
            
        current_points = np.vstack((current_points, valid_new_points))
        
    return current_points, final_tets


def gpu_delaunay_fill_and_filter(surface_verts, surface_faces, bbox_min, bbox_max, 
                                   min_spacing=None, max_spacing=None, grading=1.5, resolution=50, 
                                   target_sicn=0.15, progress_callback=None):
    """
    GPU-accelerated "Fill & Filter" pipeline.
    
    Returns:
        vertices, tetrahedra, surface_faces: Final mesh
    """
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU Mesher not available. Check installation.")
    
    def log(msg, pct=None):
        print(f"[GPU Mesher] {msg}", flush=True)
        if progress_callback:
            progress_callback(msg, pct if pct is not None else 0)
    
    log("Generating candidate points (Hessian-Proxy)...", 10)
    
    if min_spacing is None:
        # Infer from resolution if not provided
        dims = bbox_max - bbox_min
        min_spacing = np.max(dims) / resolution
        max_spacing = min_spacing * 10 # Default grading
        
    candidates = generate_hessian_sized_points(
        surface_verts, bbox_min, bbox_max, 
        min_spacing, max_spacing, grading
    )
    log(f"Generated {len(candidates)} smart points")
    
    # Pre-filter safety check
    log("Pre-filtering points near surface...", 15)
    if len(candidates) > 0:
        from scipy.spatial import cKDTree
        tree = cKDTree(surface_verts)
        dists, _ = tree.query(candidates, k=1)
        safe_mask = dists > (min_spacing * 0.1)
        candidates = candidates[safe_mask]
    log(f"Kept {len(candidates)} safe points")
    
    log("Filtering internal points with winding number...", 20)
    is_inside = compute_winding_number_vectorized(candidates, surface_verts, surface_faces)
    internal_points = candidates[is_inside]
    log(f"Kept {len(internal_points)} internal points", 30)
    
    # Proximity Filter
    log("Filtering points too close to surface...", 32)
    grid_spacing = min_spacing
    safety_margin = grid_spacing * 0.9
    log(f"Min spacing: {grid_spacing:.4f}, Safety margin: {safety_margin:.4f}")
    internal_points = filter_internal_points_by_proximity(internal_points, surface_verts, safety_margin)
    log(f"Kept {len(internal_points)} points after proximity filter", 34)
    
    log("Merging surface and internal points...", 35)
    all_points_raw = np.vstack((surface_verts, internal_points))
    log(f"Total points for GPU: {len(all_points_raw)}")
    
    # Delaunay Refinement
    log(f"Running Delaunay Refinement (Target SICN: {target_sicn})...", 40)
    refined_points, all_tets = iterative_refinement_loop(
        surface_verts, surface_faces, all_points_raw, 
        target_sicn=target_sicn, max_iters=5
    )
    all_points_raw = refined_points
    log(f"Refinement complete. Total points: {len(all_points_raw)}", 60)
    
    log("Normalizing coordinates...", 65)
    norm_points, offset, scale = normalize_points(all_points_raw)
    
    # Final Mesh (if not returned by refinement, or just to be safe)
    if all_tets is None:
        log(f"Running GPU Delaunay on {len(norm_points)} points...", 70)
        import time
        start = time.time()
        all_tets = _gpumesher.compute_delaunay(norm_points.astype(np.float64))
        elapsed = (time.time() - start) * 1000
        log(f"GPU generated {len(all_tets)} tetrahedra in {elapsed:.1f} ms!", 70)
    
    # Filter ghosts
    n_points = len(all_points_raw)
    valid_tets_mask = np.all(all_tets < n_points, axis=1)
    all_tets = all_tets[valid_tets_mask]
    
    # Lloyd Relaxation (Optional - run after refinement to smooth new points)
    log("Optimizing mesh quality (Lloyd Relaxation)...", 72)
    num_surface = len(surface_verts)
    refined_points = optimize_points_lloyd(all_points_raw, all_tets, num_surface, iterations=2)
    
    # Final Meshing
    log("Final High-Quality Meshing pass...", 74)
    norm_points, offset, scale = normalize_points(refined_points)
    all_tets = _gpumesher.compute_delaunay(norm_points.astype(np.float64))
    all_points_raw = refined_points
    
    log("Sculpting mesh (filtering by winding number)...", 75)
    
    # gDel3D may include "ghost" tetrahedra with vertex index >= N
    # Filter these out before processing
    n_points = len(all_points_raw)
    valid_tets_mask = np.all(all_tets < n_points, axis=1)
    all_tets = all_tets[valid_tets_mask]
    log(f"After ghost removal: {len(all_tets)} tetrahedra")
    
    # Calculate centroids
    tet_verts = all_points_raw[all_tets]  # (NumTets, 4, 3)
    centroids = np.mean(tet_verts, axis=1)  # (NumTets, 3)
    
    # Jitter centroids to avoid divide-by-zero
    epsilon = 1e-6
    jitter = np.random.uniform(-epsilon, epsilon, centroids.shape)
    safe_centroids = centroids + jitter
    
    log("Computing winding numbers for tet centroids...", 80)
    is_tet_inside = compute_winding_number_vectorized(safe_centroids, surface_verts, surface_faces)
    
    final_tets = all_tets[is_tet_inside]
    log(f"Final mesh: {len(final_tets)} tetrahedra (filtered from {len(all_tets)})", 95)
    
    # Remove degenerate (zero-volume) tetrahedra
    log("Removing degenerate tetrahedra...", 98)
    prev_count = len(final_tets)
    # Relaxed quality filter (0.0001) to preserve connectivity while removing zero-volume tets
    final_tets = remove_degenerate_tets(all_points_raw, final_tets, min_quality=0.0001)
    removed = prev_count - len(final_tets)
    if removed > 0:
        log(f"Removed {removed} degenerate tetrahedra")
    
    # Extract surface from volume
    log("Extracting surface topology from volume...", 99)
    final_surface_faces = extract_surface_from_volume(all_points_raw, final_tets)
    log(f"Extracted {len(final_surface_faces)} surface triangles")
    
    log("Mesh generation complete!", 100)
    
    return all_points_raw, final_tets, final_surface_faces
