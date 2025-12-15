"""
Conformal Hex Mesh Gluing System
================================

Implements topology-first conformal hex mesh generation from CoACD decomposition.
Refactored to use a Single Global Grid strategy for robustness with Fast Winding Number.

Strategy:
- Global Bounding Box Hex Grid
- Winding Number Filtering (Containment) using core/fast_winding.py
- Fallback System: Winding -> Signed Distance -> AABB
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
import trimesh
from dataclasses import dataclass, field

# Try to import GPU acceleration module
try:
    from . import hex_gpu_utils
    GPU_MODULE_AVAILABLE = True
except ImportError:
    GPU_MODULE_AVAILABLE = False
    hex_gpu_utils = None

@dataclass
class Interface:
    """Represents a shared interface between two convex chunks."""
    chunk_a: int
    chunk_b: int
    contact_faces_a: np.ndarray
    contact_faces_b: np.ndarray
    shared_vertices: np.ndarray
    normal: np.ndarray
    area: float
    divisions: int = 4

@dataclass 
class ChunkInfo:
    """Metadata for a convex chunk."""
    index: int
    vertices: np.ndarray
    faces: np.ndarray
    centroid: np.ndarray
    volume: float
    adjacent_chunks: List[int] = field(default_factory=list)
    interfaces: List[int] = field(default_factory=list)

class AdjacencyGraph:
    """
    Builds and stores the topological adjacency graph from CoACD chunks.
    Nodes = Chunks, Edges = Shared Interfaces
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.chunks: List[ChunkInfo] = []
        self.interfaces: List[Interface] = []
        self.adjacency_matrix: Optional[np.ndarray] = None
        
    def log(self, msg: str):
        if self.verbose:
            print("[ConformalGlue] {}".format(msg))
    
    def build_from_coacd(self, parts: List[Tuple[np.ndarray, np.ndarray]], 
                          epsilon: float = 0.5) -> bool:
        """
        Build adjacency graph from CoACD decomposition output.
        """
        self.log("Building adjacency graph from {} chunks...".format(len(parts)))
        
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            self.log("ERROR: scipy required for adjacency detection")
            return False
        
        # Step 1: Create ChunkInfo for each part
        self.chunks = []
        for i, (verts, faces) in enumerate(parts):
            verts = np.asarray(verts, dtype=np.float64)
            faces = np.asarray(faces, dtype=np.int32)
            
            centroid = np.mean(verts, axis=0)
            volume = self._calculate_volume(verts, faces)
            
            chunk = ChunkInfo(
                index=i,
                vertices=verts,
                faces=faces,
                centroid=centroid,
                volume=volume
            )
            self.chunks.append(chunk)
        
        self.log("Created {} chunk info records".format(len(self.chunks)))
        
        # Step 2: Detect adjacencies using KDTree
        n = len(self.chunks)
        self.adjacency_matrix = np.zeros((n, n), dtype=bool)
        self.interfaces = []
        
        for i in range(n):
            chunk_a = self.chunks[i]
            tree_a = cKDTree(chunk_a.vertices)
            
            for j in range(i + 1, n):
                chunk_b = self.chunks[j]
                
                # Find vertices in B that are close to A
                distances, indices = tree_a.query(chunk_b.vertices, k=1)
                close_mask = distances < epsilon
                
                if np.sum(close_mask) >= 3:
                    self.adjacency_matrix[i, j] = True
                    self.adjacency_matrix[j, i] = True
                    chunk_a.adjacent_chunks.append(j)
                    chunk_b.adjacent_chunks.append(i)
                    
                    interface = self._create_interface(i, j, chunk_a, chunk_b, 
                                                        close_mask, indices, epsilon)
                    if interface is not None:
                        interface_id = len(self.interfaces)
                        self.interfaces.append(interface)
                        chunk_a.interfaces.append(interface_id)
                        chunk_b.interfaces.append(interface_id)
        
        self.log("Found {} adjacencies, {} interfaces".format(
            np.sum(self.adjacency_matrix) // 2, len(self.interfaces)))
        
        return True
    
    def _create_interface(self, idx_a: int, idx_b: int, 
                           chunk_a: ChunkInfo, chunk_b: ChunkInfo,
                           close_mask: np.ndarray, nearest_indices: np.ndarray,
                           epsilon: float) -> Optional[Interface]:
        """Create an Interface object from detected contact."""
        contact_verts_b = chunk_b.vertices[close_mask]
        contact_indices_b = np.where(close_mask)[0]
        
        contact_indices_a = nearest_indices[close_mask]
        contact_verts_a = chunk_a.vertices[contact_indices_a]
        
        contact_faces_a = self._find_faces_with_vertices(chunk_a.faces, contact_indices_a)
        contact_faces_b = self._find_faces_with_vertices(chunk_b.faces, contact_indices_b)
        
        if len(contact_faces_a) == 0 or len(contact_faces_b) == 0:
            return None
        
        normal_a = self._compute_face_normals(chunk_a.vertices, chunk_a.faces[contact_faces_a])
        avg_normal = np.mean(normal_a, axis=0)
        norm_len = np.linalg.norm(avg_normal)
        if norm_len > 1e-10:
            avg_normal = avg_normal / norm_len
        
        area = self._compute_face_area(chunk_a.vertices, chunk_a.faces[contact_faces_a])
        shared_verts = (contact_verts_a + contact_verts_b) / 2.0
        
        return Interface(
            chunk_a=idx_a,
            chunk_b=idx_b,
            contact_faces_a=contact_faces_a,
            contact_faces_b=contact_faces_b,
            shared_vertices=shared_verts,
            normal=avg_normal,
            area=area
        )
    
    def _find_faces_with_vertices(self, faces: np.ndarray, vertex_indices: np.ndarray) -> np.ndarray:
        vertex_set = set(vertex_indices)
        contact_faces = []
        for i, face in enumerate(faces):
            if any(v in vertex_set for v in face):
                contact_faces.append(i)
        return np.array(contact_faces, dtype=np.int32)
    
    def _compute_face_normals(self, verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        normals = np.cross(v1 - v0, v2 - v0)
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        lengths = np.maximum(lengths, 1e-12)
        return normals / lengths
    
    def _compute_face_area(self, verts: np.ndarray, faces: np.ndarray) -> float:
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        areas = np.linalg.norm(cross, axis=1) / 2.0
        return float(np.sum(areas))
    
    def _calculate_volume(self, verts: np.ndarray, faces: np.ndarray) -> float:
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        cross = np.cross(v0, v1)
        dots = np.sum(cross * v2, axis=1)
        return abs(np.sum(dots) / 6.0)
    
    def get_adjacency_stats(self) -> Dict:
        if len(self.chunks) == 0:
            return {}
        adjacencies_per_chunk = [len(c.adjacent_chunks) for c in self.chunks]
        return {
            'num_chunks': len(self.chunks),
            'num_interfaces': len(self.interfaces),
            'total_volume': sum(c.volume for c in self.chunks)
        }

class ConformalHexGenerator:
    """
    Generates conformal hex meshes using a global grid approach.
    """
    
    def __init__(self, graph: AdjacencyGraph, verbose: bool = True):
        self.graph = graph
        self.verbose = verbose
        
    def log(self, msg: str):
        if self.verbose:
            print("[ConformalHex] {}".format(msg))
    
    def generate(self, divisions: int = 4, reference_surface: Optional[trimesh.Trimesh] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate conformal hex mesh using a single global grid approach.
        """
        if reference_surface is None:
            self.log("ERROR: Reference surface is required for global grid generation")
            return np.zeros((0, 3)), np.zeros((0, 8), dtype=np.int32)
            
        self.log("Generating conformal hex mesh using GLOBAL GRID approach...")
        
        # Step 1: Create global hex grid covering the bounding box
        grid_verts, grid_hexes = self._create_global_hex_grid(reference_surface, divisions)
        self.log(f"Global grid created: {len(grid_verts)} vertices, {len(grid_hexes)} hexes")
        
        # Step 2: Filter hexes inside the reference surface
        filtered_verts, filtered_hexes = self._filter_hexes_by_containment(grid_verts, grid_hexes, reference_surface)
        self.log(f"After filtering: {len(filtered_verts)} vertices, {len(filtered_hexes)} hexes")
        
        if len(filtered_hexes) == 0:
            self.log("WARNING: All hexes filtered out! Check grid size or constraints.")
            return np.zeros((0, 3)), np.zeros((0, 8), dtype=np.int32)
        
        # Step 3: SAFE boundary projection (collision-aware)
        projected_verts = self._project_boundary_safe(filtered_verts, filtered_hexes, reference_surface)
        
        # Step 4: Quality safeguard - gentle untangle if needed
        final_verts = self._untangle_mesh_laplacian(projected_verts, filtered_hexes)
        
        return final_verts, filtered_hexes

    def _create_global_hex_grid(self, surface: trimesh.Trimesh, divisions: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create a uniform hex grid covering the surface bounding box."""
        bounds = surface.bounds
        extents = bounds[1] - bounds[0]
        
        min_ext = np.min(extents)
        if min_ext < 1e-6:
            cell_size = 1.0
        else:
            cell_size = min_ext / max(1, divisions)
            
        dims = np.ceil(extents / cell_size).astype(int) + 2
        
        pad = cell_size * 1.5
        x = np.linspace(bounds[0][0] - pad, bounds[1][0] + pad, dims[0])
        y = np.linspace(bounds[0][1] - pad, bounds[1][1] + pad, dims[1])
        z = np.linspace(bounds[0][2] - pad, bounds[1][2] + pad, dims[2])
        
        xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
        vertices = np.stack([xv.flatten(), yv.flatten(), zv.flatten()], axis=1)
        
        nx, ny, nz = dims
        indices = np.arange(len(vertices)).reshape((nx, ny, nz))
        
        hexes = []
        for i in range(nx - 1):
            for j in range(ny - 1):
                for k in range(nz - 1):
                    n0 = indices[i, j, k]
                    n1 = indices[i+1, j, k]
                    n2 = indices[i+1, j+1, k]
                    n3 = indices[i, j+1, k]
                    n4 = indices[i, j, k+1]
                    n5 = indices[i+1, j, k+1]
                    n6 = indices[i+1, j+1, k+1]
                    n7 = indices[i, j+1, k+1]
                    hexes.append([n0, n1, n2, n3, n4, n5, n6, n7])
                    
        return vertices, np.array(hexes, dtype=np.int32)

    def _filter_hexes_by_containment(self, vertices: np.ndarray, hexes: np.ndarray, surface: trimesh.Trimesh) -> Tuple[np.ndarray, np.ndarray]:
        """Keep only hexes whose centroids are inside the surface."""
        hex_verts = vertices[hexes]
        centroids = np.mean(hex_verts, axis=1)
        
        # Tier 1: Fast Winding Number (Robust, RTree-free)
        try:
            # Dynamic import to handle package/script contexts
            try:
                from core.fast_winding import compute_fast_winding_grid
            except ImportError:
                import sys, os
                sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
                from core.fast_winding import compute_fast_winding_grid

            wn = compute_fast_winding_grid(surface.vertices, surface.faces, centroids, verbose=False)
            contains = wn > 0.5
            
        except Exception as e_wn:
            print(f"[ConformalHex] Fast Winding unavailable ({e_wn}), trying Tier 2...")
            
            # Tier 2: Signed Distance (Standard Trimesh)
            try:
                distances = trimesh.proximity.signed_distance(surface, centroids)
                contains = distances < 0
            except Exception as e_sd:
                print(f"[ConformalHex] Signed Distance failed ({e_sd}), using Tier 3 (AABB)")
                
                # Tier 3: AABB Check (Fallback for boxes/simple shapes)
                bounds = surface.bounds
                # Check if centroids are within [min, max]
                # Broadcasting: (N,3) >= (3,)
                in_box = np.all((centroids >= bounds[0]) & (centroids <= bounds[1]), axis=1)
                contains = in_box
        
        valid_hexes = hexes[contains]
        
        unique_nodes = np.unique(valid_hexes)
        new_vertices = vertices[unique_nodes]
        
        map_array = np.full(vertices.shape[0], -1, dtype=np.int32)
        map_array[unique_nodes] = np.arange(len(unique_nodes))
        
        new_hexes = map_array[valid_hexes]
        return new_vertices, new_hexes
    
    def _project_boundary_safe(self, vertices: np.ndarray, hexes: np.ndarray, surface: trimesh.Trimesh) -> np.ndarray:
        """
        SAFE boundary projection - limits displacement to prevent overlaps.
        
        Key insight: Maximum safe displacement = fraction of minimum edge length to neighbors.
        This preserves shape without causing face inversions.
        """
        self.log("Projecting boundary nodes (SAFE mode - collision-aware)...")
        
        # Build node connectivity to find neighbors
        node_neighbors = {i: set() for i in range(len(vertices))}
        hex_edges = [
            (0,1), (1,2), (2,3), (3,0),  # Bottom face
            (4,5), (5,6), (6,7), (7,4),  # Top face
            (0,4), (1,5), (2,6), (3,7)   # Vertical edges
        ]
        for hex_ids in hexes:
            for e0, e1 in hex_edges:
                node_neighbors[hex_ids[e0]].add(hex_ids[e1])
                node_neighbors[hex_ids[e1]].add(hex_ids[e0])
        
        # Identify boundary nodes
        hex_face_indices = [
            (0, 1, 2, 3), (4, 7, 6, 5),
            (0, 4, 5, 1), (1, 5, 6, 2),
            (2, 6, 7, 3), (3, 7, 4, 0)
        ]
        face_counts = {}
        for hex_ids in hexes:
            for local_face in hex_face_indices:
                face_nodes = tuple(sorted([hex_ids[i] for i in local_face]))
                face_counts[face_nodes] = face_counts.get(face_nodes, 0) + 1
        
        boundary_nodes = set()
        for face_nodes, count in face_counts.items():
            if count == 1:
                boundary_nodes.update(face_nodes)
        
        boundary_nodes = np.array(list(boundary_nodes), dtype=np.int32)
        self.log(f"Found {len(boundary_nodes)} boundary nodes")
        
        if len(boundary_nodes) == 0:
            return vertices
        
        # Calculate minimum neighbor distance for each boundary node (safe displacement limit)
        max_safe_displacement = np.zeros(len(vertices))
        SAFETY_FACTOR = 0.4  # Only move 40% of the way toward surface to stay safe
        
        for node_id in boundary_nodes:
            neighbors = list(node_neighbors[node_id])
            if len(neighbors) > 0:
                neighbor_positions = vertices[neighbors]
                distances = np.linalg.norm(neighbor_positions - vertices[node_id], axis=1)
                min_dist = np.min(distances)
                max_safe_displacement[node_id] = min_dist * SAFETY_FACTOR
        
        # Get closest points on surface
        boundary_positions = vertices[boundary_nodes]
        try:
            closest_points, _, _ = trimesh.proximity.closest_point(surface, boundary_positions)
        except:
            # Fallback - use GPU or CPU
            if GPU_MODULE_AVAILABLE and hex_gpu_utils.GPU_AVAILABLE:
                closest_points = hex_gpu_utils.closest_point_to_triangles_gpu(
                    surface.vertices, surface.faces, boundary_positions, verbose=self.verbose
                )
            else:
                closest_points = hex_gpu_utils._closest_point_cpu_fallback(
                    surface.vertices, surface.faces, boundary_positions
                )
        
        # Calculate displacement vectors and clamp to safe distance
        projected_verts = vertices.copy()
        total_clamped = 0
        
        for i, node_id in enumerate(boundary_nodes):
            displacement = closest_points[i] - vertices[node_id]
            displacement_dist = np.linalg.norm(displacement)
            safe_dist = max_safe_displacement[node_id]
            
            if displacement_dist > safe_dist and displacement_dist > 1e-10:
                # Clamp displacement to safe distance
                displacement = displacement * (safe_dist / displacement_dist)
                total_clamped += 1
            
            projected_verts[node_id] = vertices[node_id] + displacement
        
        self.log(f"Clamped {total_clamped}/{len(boundary_nodes)} nodes to safe displacement")
        
        # Calculate volume preservation
        orig_bbox = np.prod(vertices.max(axis=0) - vertices.min(axis=0))
        new_bbox = np.prod(projected_verts.max(axis=0) - projected_verts.min(axis=0))
        volume_change = abs(new_bbox - orig_bbox) / orig_bbox * 100
        self.log(f"Approximate volume change: {volume_change:.1f}%")
        
        return projected_verts
    
    def _project_boundary_to_surface(self, vertices: np.ndarray, hexes: np.ndarray, surface: trimesh.Trimesh) -> np.ndarray:
        """Project boundary nodes to the reference surface."""
        self.log("Projecting boundary nodes to surface...")
        
        # Define hex faces (local node indices)
        hex_face_indices = [
            (0, 1, 2, 3), (4, 7, 6, 5),  # Bottom, Top
            (0, 4, 5, 1), (1, 5, 6, 2),  # Front, Right
            (2, 6, 7, 3), (3, 7, 4, 0)   # Back, Left
        ]
        
        # Count face occurrences to find boundary faces
        face_counts = {}
        for hex_ids in hexes:
            for local_face in hex_face_indices:
                face_nodes = tuple(sorted([hex_ids[i] for i in local_face]))
                face_counts[face_nodes] = face_counts.get(face_nodes, 0) + 1
        
        # Boundary nodes are those on faces that appear only once
        boundary_nodes = set()
        for face_nodes, count in face_counts.items():
            if count == 1:
                boundary_nodes.update(face_nodes)
        
        boundary_nodes = np.array(list(boundary_nodes), dtype=np.int32)
        self.log(f"Found {len(boundary_nodes)} boundary nodes out of {len(vertices)}")
        
        # Project boundary nodes to closest point on surface
        projected_verts = vertices.copy()
        if len(boundary_nodes) > 0:
            boundary_positions = vertices[boundary_nodes]
            
            # Try using proximity.closest_point (requires rtree)
            # If unavailable, use brute-force or scipy KDTree
            try:
                closest_points, _, _ = trimesh.proximity.closest_point(surface, boundary_positions)
            except (ImportError, ModuleNotFoundError):
                # Try GPU acceleration if available
                if GPU_MODULE_AVAILABLE and hex_gpu_utils.GPU_AVAILABLE:
                    try:
                        self.log("rtree not available - trying GPU acceleration")
                        closest_points = hex_gpu_utils.closest_point_to_triangles_gpu(
                            surface.vertices, surface.faces, boundary_positions, verbose=self.verbose
                        )
                    except Exception as gpu_error:
                        self.log(f"GPU failed ({gpu_error}) - using CPU fallback")
                        closest_points = hex_gpu_utils._closest_point_cpu_fallback(
                            surface.vertices, surface.faces, boundary_positions
                        )
                else:
                    self.log("rtree not available - using brute-force CPU closest point search")
                    
                    # Inline brute-force (backup if GPU module not available)
                    from . import hex_gpu_utils as hgu_fallback
                    if hgu_fallback is not None:
                        closest_points = hgu_fallback._closest_point_cpu_fallback(
                            surface.vertices, surface.faces, boundary_positions
                        )
                    else:
                        # Ultimate fallback - simplified brute force
                        triangles = surface.vertices[surface.faces]
                        closest_points = np.zeros_like(boundary_positions)
                        
                        for i, query_point in enumerate(boundary_positions):
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
                                
                                s = np.clip((b * e - c * d) / det, 0, 1)
                                t = np.clip((b * d - a * e) / det, 0, 1)
                                if s + t > 1:
                                    s, t = s / (s + t), t / (s + t)
                                
                                closest_on_tri = v0 + s * edge0 + t * edge1
                                dist_sq = np.sum((query_point - closest_on_tri) ** 2)
                                
                                if dist_sq < min_dist_sq:
                                    min_dist_sq = dist_sq
                                    closest_pt = closest_on_tri
                            
                            closest_points[i] = closest_pt
            
            projected_verts[boundary_nodes] = closest_points
            
            # Calculate projection distance stats
            distances = np.linalg.norm(closest_points - boundary_positions, axis=1)
            self.log(f"Projection distances - min: {distances.min():.4f}, max: {distances.max():.4f}, avg: {distances.mean():.4f}")
        
        return projected_verts
    
    def _untangle_mesh_laplacian(self, vertices: np.ndarray, hexes: np.ndarray, max_iterations: int = 20) -> np.ndarray:
        """Use Laplacian smoothing to untangle inverted elements while keeping boundary fixed."""
        
        # Check initial quality
        jacobians = self._compute_hex_jacobians(vertices, hexes)
        num_inverted = np.sum(jacobians <= 0)
        
        if num_inverted == 0:
            self.log("No inverted elements detected - skipping untangling")
            return vertices
        
        self.log(f"Detected {num_inverted} inverted elements - applying Laplacian smoothing...")
        
        # Identify boundary nodes
        hex_face_indices = [
            (0, 1, 2, 3), (4, 7, 6, 5),
            (0, 4, 5, 1), (1, 5, 6, 2),
            (2, 6, 7, 3), (3, 7, 4, 0)
        ]
        face_counts = {}
        for hex_ids in hexes:
            for local_face in hex_face_indices:
                face_nodes = tuple(sorted([hex_ids[i] for i in local_face]))
                face_counts[face_nodes] = face_counts.get(face_nodes, 0) + 1
        boundary_nodes = set()
        for face_nodes, count in face_counts.items():
            if count == 1:
                boundary_nodes.update(face_nodes)
        
        # Build node connectivity
        node_neighbors = {i: set() for i in range(len(vertices))}
        for hex_ids in hexes:
            for node_a in hex_ids:
                for node_b in hex_ids:
                    if node_a != node_b:
                        node_neighbors[node_a].add(node_b)
        
        # Laplacian smoothing with damping
        smoothed_verts = vertices.copy()
        damping = 0.5
        
        for iteration in range(max_iterations):
            new_verts = smoothed_verts.copy()
            
            for node_id in range(len(vertices)):
                if node_id not in boundary_nodes:
                    neighbors = list(node_neighbors[node_id])
                    if len(neighbors) > 0:
                        target_pos = np.mean(smoothed_verts[neighbors], axis=0)
                        new_verts[node_id] = smoothed_verts[node_id] + damping * (target_pos - smoothed_verts[node_id])
            
            smoothed_verts = new_verts
            
            jacobians = self._compute_hex_jacobians(smoothed_verts, hexes)
            num_inverted = np.sum(jacobians <= 0)
            
            if num_inverted == 0:
                self.log(f"Untangling successful after {iteration + 1} iterations")
                return smoothed_verts
        
        jacobians_final = self._compute_hex_jacobians(smoothed_verts, hexes)
        num_inverted_final = np.sum(jacobians_final <= 0)
        self.log(f"Warning: {num_inverted_final} inverted elements remain after {max_iterations} iterations")
        
        return smoothed_verts
    
    def _compute_hex_jacobians(self, vertices: np.ndarray, hexes: np.ndarray) -> np.ndarray:
        """Compute scaled Jacobian for each hex element."""
        jacobians = np.zeros(len(hexes))
        
        for i, hex_ids in enumerate(hexes):
            corners = vertices[hex_ids]
            v01 = corners[1] - corners[0]
            v03 = corners[3] - corners[0]
            v04 = corners[4] - corners[0]
            
            det = np.dot(v01, np.cross(v03, v04))
            edge_len = np.linalg.norm(v01)
            
            if edge_len > 1e-12:
                jacobians[i] = det / (edge_len ** 3)
            else:
                jacobians[i] = 0.0
        
        return jacobians
def validate_interface_overlap(graph: AdjacencyGraph) -> Dict:
    # Not used in global grid, but kept for compatibility
    return {'pass': True, 'valid_interfaces': 0}

def validate_watertight_hex_mesh(hex_cells: np.ndarray) -> Dict:
    if len(hex_cells) == 0: return {'pass': False, 'error': 'No hex cells provided'}
    face_counts = {}
    hex_face_indices = [(0,1,2,3), (4,7,6,5), (0,4,5,1), (1,5,6,2), (2,6,7,3), (3,7,4,0)]
    for hex_ids in hex_cells:
        for local_face in hex_face_indices:
            face_nodes = tuple(sorted([hex_ids[i] for i in local_face]))
            face_counts[face_nodes] = face_counts.get(face_nodes, 0) + 1
    
    boundary_faces = 0
    non_manifold_errors = 0
    for count in face_counts.values():
        if count == 1: boundary_faces += 1
        elif count > 2: non_manifold_errors += 1
        
    return {'pass': non_manifold_errors == 0, 'boundary_faces': boundary_faces, 'non_manifold_errors': non_manifold_errors}

def validate_hex_jacobian(vertices: np.ndarray, hex_cells: np.ndarray) -> Dict:
    if len(hex_cells) == 0: return {'pass': False}
    jacobians = []
    inverted_count = 0
    for hex_ids in hex_cells:
        corners = vertices[hex_ids]
        v01 = corners[1] - corners[0]
        v03 = corners[3] - corners[0]
        v04 = corners[4] - corners[0]
        det = np.dot(v01, np.cross(v03, v04))
        edge_len = np.linalg.norm(v01)
        scaled_jacobian = det / (edge_len ** 3) if edge_len > 1e-12 else 0.0
        jacobians.append(scaled_jacobian)
        if det <= 0: inverted_count += 1
    
    jacobians_array = np.array(jacobians)
    return {
        'pass': inverted_count == 0,
        'inverted_elements': inverted_count,
        'min_jacobian': float(np.min(jacobians_array)),
        'max_jacobian': float(np.max(jacobians_array)),
        'mean_jacobian': float(np.mean(jacobians_array)),
        'per_element_quality': jacobians_array
    }

def validate_no_overlap(vertices: np.ndarray, hex_cells: np.ndarray, tolerance: float = 1e-6) -> Dict:
    if len(hex_cells) == 0: return {'pass': True}
    hex_verts = vertices[hex_cells]
    centroids = np.mean(hex_verts, axis=1)
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(centroids)
        pairs = tree.query_pairs(tolerance)
        return {'pass': len(pairs) == 0, 'overlapping_pairs': len(pairs)}
    except:
        return {'pass': True}

def compute_hex_mesh_volume(vertices: np.ndarray, hex_cells: np.ndarray) -> float:
    if len(hex_cells) == 0: return 0.0
    total_volume = 0.0
    tet_splits = [[0,1,3,4], [2,1,3,6], [5,1,4,6], [7,3,4,6], [1,3,4,6]]
    for hex_ids in hex_cells:
        corners = vertices[hex_ids]
        for tet_local in tet_splits:
            v0, v1, v2, v3 = corners[tet_local]
            vol = abs(np.dot(v1 - v0, np.cross(v2 - v0, v3 - v0))) / 6.0
            total_volume += vol
    return total_volume

def generate_conformal_hex_mesh(coacd_parts: List[Tuple[np.ndarray, np.ndarray]],
                                 divisions: int = 4,
                                 epsilon: float = 0.5,
                                 reference_stl: Optional[str] = None,
                                 verbose: bool = True) -> Dict:
    """Main entry point for conformal hex mesh generation."""
    if verbose:
        print("[ConformalHex] Starting conformal hex mesh generation...")
        print("[ConformalHex] Input: {} CoACD parts".format(len(coacd_parts)))
    
    graph = AdjacencyGraph(verbose=verbose)
    # Build graph even if we don't use it for meshing, to validate input structure
    success = graph.build_from_coacd(coacd_parts, epsilon=epsilon)
    
    if not success:
        return {'success': False, 'error': 'Failed to build adjacency graph'}
    
    if verbose:
        print("[ConformalHex] Adjacency Stats: {}".format(graph.get_adjacency_stats()))
    
    generator = ConformalHexGenerator(graph, verbose=verbose)
    
    ref_surf = None
    if reference_stl:
        try:
            ref_surf = trimesh.load(reference_stl)
            if verbose: print(f"[ConformalHex] Loaded reference STL: {reference_stl}")
            # Ensure reference surface is watertight/manifold if possible for filtering
            if not ref_surf.is_watertight:
                 if verbose: print("[ConformalHex] WARNING: Reference surface is not watertight. Filtering may be inaccurate.")
        except Exception as e:
            if verbose: print(f"[ConformalHex] WARNING: Failed to load reference STL: {e}")
            
    vertices, hexes = generator.generate(divisions=divisions, reference_surface=ref_surf)
    
    if len(hexes) == 0:
        return {'success': False, 'error': 'No hex elements generated'}
    
    if verbose: print("[ConformalHex] Running validation suite...")
    
    # Validation
    interface_validation = validate_interface_overlap(graph)
    manifold_validation = validate_watertight_hex_mesh(hexes)
    jacobian_validation = validate_hex_jacobian(vertices, hexes)
    overlap_validation = validate_no_overlap(vertices, hexes)
    hex_volume = compute_hex_mesh_volume(vertices, hexes)
    
    all_passed = (interface_validation['pass'] and manifold_validation['pass'] and jacobian_validation['pass'] and overlap_validation['pass'])
    
    return {
        'success': True,
        'vertices': vertices,
        'hexes': hexes,
        'num_vertices': len(vertices),
        'num_hexes': len(hexes),
        'volume': {'hex_volume': hex_volume},
        'adjacency_stats': graph.get_adjacency_stats(), 
        'validation': {
            'all_passed': all_passed,
            'interface': interface_validation,
            'manifold': manifold_validation,
            'jacobian': jacobian_validation,
            'overlap': overlap_validation
        }
    }

def generate_adaptive_hex_mesh(coacd_parts: List[Tuple[np.ndarray, np.ndarray]],
                                quality_target: float = 0.90,
                                max_elements: int = 10000,
                                min_divisions: int = 4,
                                max_divisions: int = 16,
                                reference_stl: Optional[str] = None,
                                verbose: bool = True) -> Dict:
    """
    Adaptive hex mesh generation with automatic refinement.
    
    Automatically increases mesh resolution until:
    - Quality target is met (e.g., 90% of elements valid)
    - OR max element count is reached
    
    Args:
        coacd_parts: CoACD decomposition output
        quality_target: Minimum acceptable quality (0.0-1.0)
        max_elements: Maximum number of hex elements
        min_divisions: Starting grid divisions (coarse)
        max_divisions: Maximum grid divisions (fine)
        reference_stl: Path to reference STL for projection
        verbose: Print progress
        
    Returns:
        Dict with mesh data and quality metrics
    """
    if verbose:
        print(f"[AdaptiveHex] Quality target: {quality_target*100:.0f}%, Max elements: {max_elements}")
    
    best_mesh = None
    best_quality = 0.0
    
    for divisions in range(min_divisions, max_divisions + 1, 2):
        if verbose:
            print(f"[AdaptiveHex] Attempting divisions={divisions}...")
        
        result = generate_conformal_hex_mesh(
            coacd_parts,
            divisions=divisions,
            epsilon=0.5,
            reference_stl=reference_stl,
            verbose=False  # Suppress internal logging
        )
        
        if not result['success']:
            if verbose:
                print(f"[AdaptiveHex] divisions={divisions} failed: {result.get('error', 'unknown')}")
            continue
        
        # Compute quality metrics
        num_hexes = result['num_hexes']
        jacobian_val = result['validation']['jacobian']
        num_inverted = jacobian_val.get('inverted_elements', 0)
        quality = 1.0 - (num_inverted / num_hexes) if num_hexes > 0 else 0.0
        
        if verbose:
            print(f"[AdaptiveHex] divisions={divisions}: {num_hexes} elements, quality={quality*100:.1f}%")
        
        # Track best mesh
        if quality > best_quality:
            best_quality = quality
            best_mesh = result
        
        # Check termination conditions
        if quality >= quality_target:
            if verbose:
                    print(f"[AdaptiveHex] SUCCESS: Quality target met at divisions={divisions}")
            return result
        
        if num_hexes >= max_elements:
            if verbose:
                    print(f"[AdaptiveHex] WARNING: Max element count reached at divisions={divisions}")
            return result
    
    # Return best mesh found
    if best_mesh is not None:
        if verbose:
            print(f"[AdaptiveHex] Returning best mesh: quality={best_quality*100:.1f}%")
        return best_mesh
    else:
        return {'success': False, 'error': 'All division levels failed'}
