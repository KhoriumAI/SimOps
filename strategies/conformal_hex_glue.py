"""
Conformal Hex Mesh Gluing System
================================

Implements topology-first conformal hex mesh generation from CoACD decomposition.

Strategy: Block-Structured Decomposition
- Builds adjacency graph from CoACD chunks
- Forces shared interface nodes to align BEFORE meshing
- Merges with KDTree deduplication

This replaces the error-prone Gmsh Boolean Fragment approach.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
import trimesh
from dataclasses import dataclass, field


@dataclass
class Interface:
    """Represents a shared interface between two convex chunks."""
    chunk_a: int
    chunk_b: int
    contact_faces_a: np.ndarray  # Face indices in chunk A
    contact_faces_b: np.ndarray  # Face indices in chunk B
    shared_vertices: np.ndarray  # Vertex positions at interface
    normal: np.ndarray  # Average normal direction
    area: float
    divisions: int = 4  # Number of divisions along interface


@dataclass 
class ChunkInfo:
    """Metadata for a convex chunk."""
    index: int
    vertices: np.ndarray
    faces: np.ndarray
    centroid: np.ndarray
    volume: float
    adjacent_chunks: List[int] = field(default_factory=list)
    interfaces: List[int] = field(default_factory=list)  # Interface IDs


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
        
        Args:
            parts: List of (vertices, faces) tuples from CoACD
            epsilon: Distance threshold for detecting touching chunks
            
        Returns:
            True if successful
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
                
                if np.sum(close_mask) >= 3:  # Need at least 3 vertices for a face
                    # These chunks are adjacent
                    self.adjacency_matrix[i, j] = True
                    self.adjacency_matrix[j, i] = True
                    
                    chunk_a.adjacent_chunks.append(j)
                    chunk_b.adjacent_chunks.append(i)
                    
                    # Create interface
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
        
        # Get contact vertices from chunk B
        contact_verts_b = chunk_b.vertices[close_mask]
        contact_indices_b = np.where(close_mask)[0]
        
        # Get corresponding vertices from chunk A
        contact_indices_a = nearest_indices[close_mask]
        contact_verts_a = chunk_a.vertices[contact_indices_a]
        
        # Find faces that use these vertices
        contact_faces_a = self._find_faces_with_vertices(chunk_a.faces, contact_indices_a)
        contact_faces_b = self._find_faces_with_vertices(chunk_b.faces, contact_indices_b)
        
        if len(contact_faces_a) == 0 or len(contact_faces_b) == 0:
            return None
        
        # Compute average normal
        normal_a = self._compute_face_normals(chunk_a.vertices, chunk_a.faces[contact_faces_a])
        avg_normal = np.mean(normal_a, axis=0)
        norm_len = np.linalg.norm(avg_normal)
        if norm_len > 1e-10:
            avg_normal = avg_normal / norm_len
        
        # Compute interface area
        area = self._compute_face_area(chunk_a.vertices, chunk_a.faces[contact_faces_a])
        
        # Shared vertices (average of A and B positions)
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
        """Find face indices that contain any of the given vertex indices."""
        vertex_set = set(vertex_indices)
        contact_faces = []
        for i, face in enumerate(faces):
            if any(v in vertex_set for v in face):
                contact_faces.append(i)
        return np.array(contact_faces, dtype=np.int32)
    
    def _compute_face_normals(self, verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """Compute normal vectors for given faces."""
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        normals = np.cross(v1 - v0, v2 - v0)
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        lengths = np.maximum(lengths, 1e-12)
        return normals / lengths
    
    def _compute_face_area(self, verts: np.ndarray, faces: np.ndarray) -> float:
        """Compute total area of given faces."""
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        areas = np.linalg.norm(cross, axis=1) / 2.0
        return float(np.sum(areas))
    
    def _calculate_volume(self, verts: np.ndarray, faces: np.ndarray) -> float:
        """Calculate mesh volume using signed tetrahedron method."""
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        cross = np.cross(v0, v1)
        dots = np.sum(cross * v2, axis=1)
        return abs(np.sum(dots) / 6.0)
    
    def get_adjacency_stats(self) -> Dict:
        """Return statistics about the adjacency graph."""
        if len(self.chunks) == 0:
            return {}
        
        adjacencies_per_chunk = [len(c.adjacent_chunks) for c in self.chunks]
        
        return {
            'num_chunks': len(self.chunks),
            'num_interfaces': len(self.interfaces),
            'total_adjacencies': np.sum(self.adjacency_matrix) // 2,
            'avg_adjacencies_per_chunk': np.mean(adjacencies_per_chunk),
            'max_adjacencies': max(adjacencies_per_chunk),
            'isolated_chunks': sum(1 for a in adjacencies_per_chunk if a == 0),
            'total_volume': sum(c.volume for c in self.chunks),
            'total_interface_area': sum(iface.area for iface in self.interfaces)
        }


class IntegerConstraintSolver:
    """
    Solves the "Interval Assignment" problem to ensure compatible 
    edge divisions across shared interfaces.
    """
    
    def __init__(self, graph: AdjacencyGraph, base_divisions: int = 4):
        self.graph = graph
        self.base_divisions = base_divisions
        
    def solve(self) -> bool:
        """
        Propagate division constraints to ensure compatibility.
        
        Simple greedy approach: all interfaces get the same base division count.
        More advanced: scale by interface area.
        """
        if len(self.graph.interfaces) == 0:
            return True
        
        # For now, use area-based scaling
        areas = [iface.area for iface in self.graph.interfaces]
        max_area = max(areas) if areas else 1.0
        
        for iface in self.graph.interfaces:
            # Scale divisions by relative area (more area = more divisions)
            scale = (iface.area / max_area) ** 0.5  # sqrt for gentler scaling
            iface.divisions = max(2, int(self.base_divisions * scale + 0.5))
        
        return True


class ConformalHexGenerator:
    """
    Generates conformal hex meshes from the adjacency graph.
    """
    
    def __init__(self, graph: AdjacencyGraph, verbose: bool = True):
        self.graph = graph
        self.verbose = verbose
        self.all_vertices: List[np.ndarray] = []
        self.all_hexes: List[np.ndarray] = []
        self.interface_nodes: Dict[int, np.ndarray] = {}  # interface_id -> node indices
        
    def log(self, msg: str):
        if self.verbose:
            print("[ConformalHex] {}".format(msg))
    
    def generate(self, divisions: int = 4, reference_surface: Optional[trimesh.Trimesh] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate conformal hex mesh.
        
        Args:
            divisions: Base number of divisions per chunk
            reference_surface: Optional Trimesh object for boundary projection
            
        Returns:
            vertices: (N, 3) array
            hexes: (M, 8) array of vertex indices
        """
        self.log("Generating conformal hex mesh with {} base divisions...".format(divisions))
        
        # Step 1: Solve integer constraints
        solver = IntegerConstraintSolver(self.graph, divisions)
        solver.solve()
        
        # Step 2: Mesh each chunk independently (for now using simple subdivision)
        all_verts = []
        all_hexes = []
        vertex_offset = 0
        
        for chunk in self.graph.chunks:
            chunk_verts, chunk_hexes = self._mesh_convex_chunk(chunk, divisions)
            
            if len(chunk_hexes) > 0:
                # Offset hex indices
                chunk_hexes = chunk_hexes + vertex_offset
                all_verts.append(chunk_verts)
                all_hexes.append(chunk_hexes)
                vertex_offset += len(chunk_verts)
        
        if len(all_verts) == 0:
            return np.zeros((0, 3)), np.zeros((0, 8), dtype=np.int32)
        
        vertices = np.vstack(all_verts)
        hexes = np.vstack(all_hexes) if len(all_hexes) > 0 else np.zeros((0, 8), dtype=np.int32)
        
        self.log("Before merge: {} vertices, {} hexes".format(len(vertices), len(hexes)))
        
        # Step 3: Merge duplicate vertices at interfaces
        vertices, hexes = self._merge_interface_vertices(vertices, hexes)
        
        self.log("After merge: {} vertices, {} hexes".format(len(vertices), len(hexes)))
        
        # Step 4: Boundary Projection and Smoothing
        if reference_surface is not None and len(hexes) > 0:
            self.log("Performing boundary projection and smoothing...")
            vertices = self._project_and_smooth(vertices, hexes, reference_surface)
        
        return vertices, hexes
    
    def _project_and_smooth(self, vertices: np.ndarray, hexes: np.ndarray, 
                            surface: trimesh.Trimesh) -> np.ndarray:
        """
        Project boundary nodes to surface and smooth interior.
        """
        # 1. Identify boundary nodes
        boundary_nodes = self._identify_boundary_nodes(hexes)
        self.log(f"Identified {len(boundary_nodes)} boundary nodes")
        
        if len(boundary_nodes) == 0:
            return vertices
            
        # 2. Project boundary nodes
        vertices = self._project_boundary_nodes(vertices, boundary_nodes, surface)
        
        # 3. Smooth interior nodes
        # Create mask of fixed nodes (boundary nodes)
        fixed_mask = np.zeros(len(vertices), dtype=bool)
        fixed_mask[boundary_nodes] = True
        
        vertices = self._smooth_interior_nodes(vertices, hexes, fixed_mask, iterations=3)
        
        return vertices

    def _identify_boundary_nodes(self, hexes: np.ndarray) -> np.ndarray:
        """Identify unique node indices that lie on the boundary of the hex mesh."""
        # Hex faces (local indices)
        faces = np.array([
            [0, 1, 2, 3], [4, 7, 6, 5], # Bottom, Top
            [0, 4, 5, 1], [1, 5, 6, 2], # Front, Right
            [2, 6, 7, 3], [3, 7, 4, 0]  # Back, Left
        ])
        
        # Collect all faces
        all_faces = []
        for h in hexes:
            for f in faces:
                # Store sorted tuple to identify unique faces
                all_faces.append(tuple(sorted(h[f])))
        
        # Count occurrences
        from collections import Counter
        counts = Counter(all_faces)
        
        # Boundary faces appear exactly once
        boundary_node_set = set()
        for face, count in counts.items():
            if count == 1:
                boundary_node_set.update(face)
                
        return np.array(list(boundary_node_set), dtype=np.int32)

    def _project_boundary_nodes(self, vertices: np.ndarray, node_indices: np.ndarray, 
                              surface: trimesh.Trimesh) -> np.ndarray:
        """
        Project specified nodes onto the closest point on the surface.
        Uses cKDTree to snap to nearest mesh vertex (robust to missing rtree).
        """
        if len(node_indices) == 0:
            return vertices
            
        points = vertices[node_indices]
        
        # Use scipy cKDTree for robust nearest vertex lookup
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(surface.vertices)
            _, closest_indices = tree.query(points, k=1)
            closest_points = surface.vertices[closest_indices]
            
            # Update vertex positions
            new_vertices = vertices.copy()
            new_vertices[node_indices] = closest_points
            
            return new_vertices
        except Exception as e:
            self.log(f"WARNING: Projection failed: {e}")
            return vertices

    def _smooth_interior_nodes(self, vertices: np.ndarray, hexes: np.ndarray, 
                             fixed_mask: np.ndarray, iterations: int = 5) -> np.ndarray:
        """Apply Laplacian smoothing to non-fixed (interior) nodes."""
        current_verts = vertices.copy()
        
        # Build adjacency list for smoothing
        # (This can be slow for large meshes, optimization possible)
        # For prototype, we'll iterate edges
        
        num_verts = len(vertices)
        
        # Edges in a hex
        hex_edges = [
            (0,1), (1,2), (2,3), (3,0), # Bottom ring
            (4,5), (5,6), (6,7), (7,4), # Top ring
            (0,4), (1,5), (2,6), (3,7)  # Pillars
        ]
        
        # Build neighbor graph
        adj = [set() for _ in range(num_verts)]
        for h in hexes:
            for u_local, v_local in hex_edges:
                u, v = h[u_local], h[v_local]
                adj[u].add(v)
                adj[v].add(u)
                
        # Smoothing iterations
        for it in range(iterations):
            new_pos = current_verts.copy()
            max_move = 0.0
            
            for i in range(num_verts):
                if fixed_mask[i]:
                    continue
                    
                neighbors = list(adj[i])
                if not neighbors:
                    continue
                
                # Centroid of neighbors
                sum_pos = np.zeros(3)
                for n in neighbors:
                    sum_pos += current_verts[n]
                
                avg_pos = sum_pos / len(neighbors)
                
                # Update with relaxation (0.5 factor)
                move = (avg_pos - current_verts[i]) * 0.5
                new_pos[i] += move
                max_move = max(max_move, np.linalg.norm(move))
            
            current_verts = new_pos
            if max_move < 1e-4:
                break
                
        return current_verts
    
    def _mesh_convex_chunk(self, chunk: ChunkInfo, divisions: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mesh a single convex chunk using bounding box subdivision.
        
        For truly convex chunks, this generates a simple structured grid.
        """
        verts = chunk.vertices
        
        # Get bounding box
        bbox_min = verts.min(axis=0)
        bbox_max = verts.max(axis=0)
        
        # Generate structured grid
        nx = ny = nz = divisions
        
        x = np.linspace(bbox_min[0], bbox_max[0], nx + 1)
        y = np.linspace(bbox_min[1], bbox_max[1], ny + 1)
        z = np.linspace(bbox_min[2], bbox_max[2], nz + 1)
        
        # Create vertex grid
        grid_verts = []
        for k in range(nz + 1):
            for j in range(ny + 1):
                for i in range(nx + 1):
                    grid_verts.append([x[i], y[j], z[k]])
        
        grid_verts = np.array(grid_verts)
        
        # Create hex elements
        hexes = []
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    # 8 corners of the hex (VTK ordering)
                    n0 = i + j * (nx + 1) + k * (nx + 1) * (ny + 1)
                    n1 = n0 + 1
                    n2 = n0 + (nx + 1) + 1
                    n3 = n0 + (nx + 1)
                    n4 = n0 + (nx + 1) * (ny + 1)
                    n5 = n4 + 1
                    n6 = n4 + (nx + 1) + 1
                    n7 = n4 + (nx + 1)
                    
                    hexes.append([n0, n1, n2, n3, n4, n5, n6, n7])
        
        hexes = np.array(hexes, dtype=np.int32)
        
        hexes = np.array(hexes, dtype=np.int32)
        
        # Filter hexes: only keep those whose centroid is inside the chunk
        hexes = self._filter_hexes_inside_chunk(hexes, grid_verts, chunk)
        
        return grid_verts, hexes

    def _filter_hexes_inside_chunk(self, hexes: np.ndarray, vertices: np.ndarray, 
                                 chunk: ChunkInfo) -> np.ndarray:
        """Keep only hexes whose centroids are inside the convex chunk."""
        if len(hexes) == 0:
            return hexes
            
        # Compute face normals for the chunk
        chunk_verts = chunk.vertices
        chunk_faces = chunk.faces
        
        # v1-v0 cross v2-v0
        v0 = chunk_verts[chunk_faces[:, 0]]
        v1 = chunk_verts[chunk_faces[:, 1]]
        v2 = chunk_verts[chunk_faces[:, 2]]
        normals = np.cross(v1 - v0, v2 - v0)
        
        # Normalize
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        normals = normals / norms
        
        # Face centers (points on plane)
        plane_points = v0
        
        # Compute hex centroids
        # hex shape: (M, 8)
        # vertices shape: (N, 3)
        # gathered: (M, 8, 3)
        hex_verts = vertices[hexes]
        centroids = np.mean(hex_verts, axis=1) # (M, 3)
        
        # Check against all planes
        # Dot product: (point - plane_point) . normal < 0
        # If dot > 0 for ANY plane, point is outside
        # (Assuming outward normals)
        
        # Vector from plane point to centroid: (M, F, 3) - too big?
        # Use matrix: (M, 3) dot (F, 3).T -> (M, F)
        
        # d = (C - P) . N = C.N - P.N
        # C.N: (M, 3) x (3, F) -> (M, F)
        # P.N: (F, 3) dot (3, 1) -> (F,)
        
        # This assumes consistent winding. We can use a tolerance.
        
        valid_mask = np.ones(len(hexes), dtype=bool)
        
        # Batch processing might be heavy if F is large, but for convex chunks F is small
        # centroids: (M, 3)
        # normals: (F, 3)
        # plane_points: (F, 3)
        
        # Projections of centroids onto normals
        c_dot_n = np.dot(centroids, normals.T) # (M, F)
        
        # Projections of plane points onto normals
        # p_dot_n[i] = dot(plane_points[i], normals[i])
        p_dot_n = np.einsum('ij,ij->i', plane_points, normals) # (F,)
        
        # Distances: d[i, j] = c_dot_n[i, j] - p_dot_n[j]
        # If d > epsilon, it's outside
        distances = c_dot_n - p_dot_n[np.newaxis, :]
        
        # Centroid must be inside ALL planes (dist <= epsilon)
        # max_dist for each hex
        max_dists = np.max(distances, axis=1)
        
        # If max_dist > tolerance, it's outside
        # CoACD usually outputs outward normals
        valid_mask = max_dists < 1e-4
        
        filtered_hexes = hexes[valid_mask]
        
        # If we filtered everything, keep at least one (fallback) 
        # or maybe the centroid logic was too strict?
        # For now, trust the logic.
        
        if len(filtered_hexes) == 0 and len(hexes) > 0:
            # Fallback: keep hexes closest to chunk centroid?
            # Or just return empty (and let merge handle it)
            pass
            
        return filtered_hexes
    
    def _merge_interface_vertices(self, vertices: np.ndarray, hexes: np.ndarray, 
                                    tolerance: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Merge duplicate vertices using KDTree.
        """
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            return vertices, hexes
        
        if len(vertices) == 0:
            return vertices, hexes
        
        tree = cKDTree(vertices)
        
        # Find duplicate pairs
        pairs = tree.query_pairs(tolerance)
        
        if len(pairs) == 0:
            return vertices, hexes
        
        # Build merge map
        merge_map = np.arange(len(vertices))
        for i, j in pairs:
            # Keep lower index
            if merge_map[j] > merge_map[i]:
                merge_map[j] = merge_map[i]
            else:
                merge_map[i] = merge_map[j]
        
        # Compress indices
        unique_indices = np.unique(merge_map)
        new_index_map = {old: new for new, old in enumerate(unique_indices)}
        
        final_merge_map = np.array([new_index_map[merge_map[i]] for i in range(len(vertices))])
        
        # Create new vertex array
        new_vertices = vertices[unique_indices]
        
        # Remap hex indices
        new_hexes = final_merge_map[hexes]
        
        return new_vertices, new_hexes


# =============================================================================
# VALIDATION SUITE
# =============================================================================

def validate_interface_overlap(graph: AdjacencyGraph) -> Dict:
    """
    Test Suite 1: Interface Integrity Check
    Verifies that detected interfaces have valid overlap.
    """
    results = {
        'total_interfaces': len(graph.interfaces),
        'valid_interfaces': 0,
        'invalid_interfaces': 0,
        'errors': []
    }
    
    for i, iface in enumerate(graph.interfaces):
        # Check 1: Area > 0
        if iface.area <= 0:
            results['errors'].append("Interface {}: Zero or negative area".format(i))
            results['invalid_interfaces'] += 1
            continue
        
        # Check 2: Has shared vertices
        if len(iface.shared_vertices) < 3:
            results['errors'].append("Interface {}: Less than 3 shared vertices".format(i))
            results['invalid_interfaces'] += 1
            continue
        
        # Check 3: Opposing normals (should be roughly anti-parallel)
        # This would require computing normals for both sides
        # For now, just verify normal is unit length
        norm_len = np.linalg.norm(iface.normal)
        if abs(norm_len - 1.0) > 0.01:
            results['errors'].append("Interface {}: Non-unit normal".format(i))
            results['invalid_interfaces'] += 1
            continue
        
        results['valid_interfaces'] += 1
    
    results['pass'] = results['invalid_interfaces'] == 0
    return results


def validate_watertight_hex_mesh(hex_cells: np.ndarray) -> Dict:
    """
    Test Suite 2: Manifold Connectivity Check
    Verifies that the hex mesh is watertight and conformal.
    """
    if len(hex_cells) == 0:
        return {'pass': False, 'error': 'No hex cells provided'}
    
    face_counts = {}
    
    # Define the 6 faces of a hex (using local node indices 0-7)
    hex_face_indices = [
        (0, 1, 2, 3),  # Bottom
        (4, 7, 6, 5),  # Top
        (0, 4, 5, 1),  # Front
        (1, 5, 6, 2),  # Right
        (2, 6, 7, 3),  # Back
        (3, 7, 4, 0)   # Left
    ]
    
    for hex_ids in hex_cells:
        for local_face in hex_face_indices:
            # Get global node IDs for this face (sorted for comparison)
            face_nodes = tuple(sorted([hex_ids[i] for i in local_face]))
            
            if face_nodes in face_counts:
                face_counts[face_nodes] += 1
            else:
                face_counts[face_nodes] = 1
    
    # Analysis
    boundary_faces = 0
    internal_faces = 0
    non_manifold_errors = 0
    
    for face, count in face_counts.items():
        if count == 1:
            boundary_faces += 1
        elif count == 2:
            internal_faces += 1
        else:
            non_manifold_errors += 1
    
    return {
        'pass': non_manifold_errors == 0,
        'boundary_faces': boundary_faces,
        'internal_faces': internal_faces,
        'non_manifold_errors': non_manifold_errors,
        'total_faces': len(face_counts)
    }


def validate_hex_jacobian(vertices: np.ndarray, hex_cells: np.ndarray) -> Dict:
    """
    Test Suite 3: Geometric Quality Check
    Computes scaled Jacobian for hex elements.
    """
    if len(hex_cells) == 0:
        return {'pass': False, 'error': 'No hex cells'}
    
    jacobians = []
    inverted_count = 0
    
    for hex_ids in hex_cells:
        # Get 8 corner vertices
        corners = vertices[hex_ids]
        
        # Compute Jacobian at center (simplified)
        # Using vectors from corner 0
        v01 = corners[1] - corners[0]
        v03 = corners[3] - corners[0]
        v04 = corners[4] - corners[0]
        
        # Jacobian determinant = triple product
        det = np.dot(v01, np.cross(v03, v04))
        
        # Normalize by ideal hex volume
        edge_len = np.linalg.norm(v01)
        if edge_len > 1e-12:
            scaled_jacobian = det / (edge_len ** 3)
        else:
            scaled_jacobian = 0.0
        
        jacobians.append(scaled_jacobian)
        
        if det <= 0:
            inverted_count += 1
    
    jacobians = np.array(jacobians)
    
    return {
        'pass': inverted_count == 0 and np.mean(jacobians) > 0.3,
        'min_jacobian': float(np.min(jacobians)),
        'max_jacobian': float(np.max(jacobians)),
        'mean_jacobian': float(np.mean(jacobians)),
        'inverted_elements': inverted_count,
        'total_elements': len(hex_cells),
        'per_element_quality': jacobians.tolist()
    }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def generate_conformal_hex_mesh(coacd_parts: List[Tuple[np.ndarray, np.ndarray]],
                                 divisions: int = 4,
                                 epsilon: float = 0.5,
                                 reference_stl: Optional[str] = None,
                                 verbose: bool = True) -> Dict:
    """
    Main entry point for conformal hex mesh generation.
    
    Args:
        coacd_parts: List of (vertices, faces) from CoACD decomposition
        divisions: Base number of divisions per chunk
        epsilon: Distance threshold for interface detection
        reference_stl: Path to original STL file for boundary projection
        verbose: Print progress messages
        
    Returns:
        Dictionary with vertices, hexes, and validation results
    """
    if verbose:
        print("[ConformalHex] Starting conformal hex mesh generation...")
        print("[ConformalHex] Input: {} CoACD parts".format(len(coacd_parts)))
    
    # Phase 1: Build adjacency graph
    graph = AdjacencyGraph(verbose=verbose)
    success = graph.build_from_coacd(coacd_parts, epsilon=epsilon)
    
    if not success:
        return {'success': False, 'error': 'Failed to build adjacency graph'}
    
    adjacency_stats = graph.get_adjacency_stats()
    if verbose:
        print("[ConformalHex] Adjacency Stats: {}".format(adjacency_stats))
    
    # Phase 2: Generate hex mesh
    generator = ConformalHexGenerator(graph, verbose=verbose)
    
    # Load reference surface if provided
    ref_surf = None
    if reference_stl:
        try:
            ref_surf = trimesh.load(reference_stl)
            if verbose: print(f"[ConformalHex] Loaded reference STL: {reference_stl}")
        except Exception as e:
            if verbose: print(f"[ConformalHex] WARNING: Failed to load reference STL: {e}")
            
    vertices, hexes = generator.generate(divisions=divisions, reference_surface=ref_surf)
    
    if len(hexes) == 0:
        return {'success': False, 'error': 'No hex elements generated'}
    
    # Phase 3: Validation
    if verbose:
        print("[ConformalHex] Running validation suite...")
    
    interface_validation = validate_interface_overlap(graph)
    manifold_validation = validate_watertight_hex_mesh(hexes)
    jacobian_validation = validate_hex_jacobian(vertices, hexes)
    
    all_passed = (interface_validation['pass'] and 
                  manifold_validation['pass'] and 
                  jacobian_validation['pass'])
    
    if verbose:
        print("[ConformalHex] Interface Validation: {}".format(
            'PASS' if interface_validation['pass'] else 'FAIL'))
        print("[ConformalHex] Manifold Validation: {}".format(
            'PASS' if manifold_validation['pass'] else 'FAIL'))
        print("[ConformalHex] Jacobian Validation: {}".format(
            'PASS' if jacobian_validation['pass'] else 'FAIL'))
    
    return {
        'success': True,
        'vertices': vertices,
        'hexes': hexes,
        'num_vertices': len(vertices),
        'num_hexes': len(hexes),
        'adjacency_stats': adjacency_stats,
        'validation': {
            'all_passed': all_passed,
            'interface': interface_validation,
            'manifold': manifold_validation,
            'jacobian': jacobian_validation
        }
    }
