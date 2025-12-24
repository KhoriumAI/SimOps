
import numpy as np
from typing import Dict, List, Set, Optional, Tuple, Callable
from dataclasses import dataclass, field
import collections

@dataclass
class ZoneData:
    """Data for a single named zone."""
    name: str
    zone_type: str
    face_ids: Set[int] = field(default_factory=set)
    color: Tuple[float, float, float] = (1.0, 0.0, 0.0)

class FluentZoneManager:
    """
    Manages boundary zone assignments and face selection logic.
    Designed to be independent of the GUI for testing.
    """
    
    def __init__(self):
        # State
        self.zone_registry: Dict[str, ZoneData] = {}    # zone_name -> ZoneData
        self.face_to_zone: Dict[int, str] = {}          # face_id -> zone_name
        self.selected_faces: Set[int] = set()           # Currently selected face IDs
        self.highlighted_face: Optional[int] = None     # Currently hovered face
        
        # Geometry Data for Spill Logic
        self.face_adjacency: Dict[int, List[int]] = {}  # face_id -> [neighbor_face_ids]
        self.face_normals: Dict[int, np.ndarray] = {}   # face_id -> normal vector
        self.face_centers: Dict[int, np.ndarray] = {}   # face_id -> centroid
        self.points: Optional[np.ndarray] = None        # Nx3 points array
        
        # Curvature & Normals
        self.vertex_curvature: Dict[int, float] = {}    # vertex_idx -> mean curvature
        self.vertex_normals: Dict[int, np.ndarray] = {} # vertex_idx -> angle-weighted normal
        self.break_edges: Set[Tuple[int, int]] = set()  # edges that are curvature breaks
        
        # Segmentation Map (Bonus Face Coloring)
        self.element_to_face_partition: Dict[int, int] = {} # face_id -> partition_id
        
        # Constants
        self.spill_angle_threshold = 15.0 
        self.curvature_break_threshold = 0.5 

    def set_mesh_data(self, points: np.ndarray, elements: List[Dict]):
        """
        Initialize mesh connectivity for spill selection.
        Expects elements to be dictionaries with 'type', 'id', 'nodes'.
        Only processes surface elements (triangles/quads).
        """
        print(f"[ZoneManager] Building adjacency for {len(elements)} elements...")
        self.points = points
        
        # 1. Extract surface faces and build edge map
        # Edge -> List[face_id]
        edge_map = collections.defaultdict(list)
        
        self.face_adjacency.clear()
        self.face_normals.clear()
        self.face_centers.clear()
        
        count = 0
        for elem in elements:
            etype = elem.get('type')
            eid = elem.get('id')
            nodes = elem.get('nodes')
            
            if etype not in ('triangle', 'quadrilateral'):
                continue
                
            count += 1
            
            # Compute Normal & Centroid
            elem_pts = points[nodes] # Assuming nodes are 0-indexed indices into points
            
            # Centroid
            center = np.mean(elem_pts, axis=0)
            self.face_centers[eid] = center
            
            # Normal (using first 3 points)
            v1 = elem_pts[1] - elem_pts[0]
            v2 = elem_pts[2] - elem_pts[0]
            normal = np.cross(v1, v2).astype(float)
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                normal /= norm
            self.face_normals[eid] = normal
            
            # Edges
            num_nodes = len(nodes)
            for i in range(num_nodes):
                n1 = nodes[i]
                n2 = nodes[(i + 1) % num_nodes]
                edge = tuple(sorted((n1, n2)))
                edge_map[edge].append(eid)
                
        # 2. Build Adjacency Graph from Edge Map
        for edge, face_ids in edge_map.items():
            # A manifold internal edge has 2 faces. Boundary has 1. Non-manifold >2.
            # We connect all faces sharing an edge.
            for i in range(len(face_ids)):
                f1 = face_ids[i]
                if f1 not in self.face_adjacency:
                    self.face_adjacency[f1] = []
                    
                for j in range(len(face_ids)):
                    if i == j: continue
                    f2 = face_ids[j]
                    if f2 not in self.face_adjacency[f1]:
                        self.face_adjacency[f1].append(f2)
                        
        print(f"[ZoneManager] Processed {count} surface faces. Adjacency graph built.")
        
        # Compute curvature for fallback segmentation
        self._compute_vertex_curvature(points, elements)
        self._detect_break_edges()
        print(f"[ZoneManager] Curvature computed. {len(self.break_edges)} break edges detected.")
        
        # Compute Angle-Weighted Normals (Requested for Boundary Layers)
        self._compute_angle_weighted_normals(points, elements)
        
        # Segment entire mesh into logical partitions (Requested for Face Coloring Mode)
        self.segment_all_faces()
        
    def _compute_angle_weighted_normals(self, points: np.ndarray, elements: List[Dict]):
        """
        Compute Angle-Weighted Normals for all vertices.
        This provides a superior extrusion vector for boundary layers compared to simple averaging,
        as it accounts for the geometry of the triangles at the vertex.
        """
        self.vertex_normals.clear()
        
        # Accumulate weighted normals: vertex_idx -> [nx, ny, nz]
        weighted_normals = collections.defaultdict(lambda: np.zeros(3))
        
        for elem in elements:
            etype = elem.get('type')
            eid = elem.get('id')
            nodes = elem.get('nodes')
            
            if etype != 'triangle': # TODO: Add Quad support if needed, but triangles are 99%
                continue
                
            # Get face normal
            face_normal = self.face_normals.get(eid)
            if face_normal is None:
                continue
                
            # Get coords
            p0 = points[nodes[0]]
            p1 = points[nodes[1]]
            p2 = points[nodes[2]]
            
            # Compute angles at each vertex
            # Angle at P0 (between P0->P1 and P0->P2)
            v01 = (p1 - p0).astype(float)
            v02 = (p2 - p0).astype(float)
            v01 /= (np.linalg.norm(v01) + 1e-12)
            v02 /= (np.linalg.norm(v02) + 1e-12)
            angle0 = np.arccos(np.clip(np.dot(v01, v02), -1.0, 1.0))
            
            # Angle at P1
            v10 = (p0 - p1).astype(float)
            v12 = (p2 - p1).astype(float)
            v10 /= (np.linalg.norm(v10) + 1e-12)
            v12 /= (np.linalg.norm(v12) + 1e-12)
            angle1 = np.arccos(np.clip(np.dot(v10, v12), -1.0, 1.0))
            
            # Angle at P2
            v20 = (p0 - p2).astype(float)
            v21 = (p1 - p2).astype(float)
            v20 /= (np.linalg.norm(v20) + 1e-12)
            v21 /= (np.linalg.norm(v21) + 1e-12)
            angle2 = np.arccos(np.clip(np.dot(v20, v21), -1.0, 1.0))
            
            # Accumulate: Normal * Angle
            weighted_normals[nodes[0]] += face_normal * angle0
            weighted_normals[nodes[1]] += face_normal * angle1
            weighted_normals[nodes[2]] += face_normal * angle2
            
        # Normalize
        count = 0
        for vid, dynamic_n in weighted_normals.items():
            norm = np.linalg.norm(dynamic_n)
            if norm > 1e-12:
                dynamic_n /= norm
            self.vertex_normals[vid] = dynamic_n
            count += 1
            
        print(f"[ZoneManager] Computed angle-weighted normals for {count} vertices.")

    def segment_all_faces(self):
        """
        Pre-compute logical face partitions for the entire mesh using the spill logic.
        This enables the "Face Coloring Mode" visualization.
        """
        self.element_to_face_partition.clear()
        
        visited_global = set()
        partition_id = 0
        
        # Iterate all known faces
        all_faces = sorted(list(self.face_normals.keys()))
        
        for start_face in all_faces:
            if start_face in visited_global:
                continue
                
            partition_id += 1
            
            # Run a constrained flood fill for this partition
            # Re-using logic similar to spill_select but purely topological/geometric
            queue = collections.deque([start_face])
            partition_faces = set([start_face])
            visited_global.add(start_face)
            
            # Start normal for drift check
            start_normal = self.face_normals[start_face]
            start_threshold_cos = np.cos(np.radians(45.0)) # Relaxed slightly for global partitioning
            threshold_cos = np.cos(np.radians(self.spill_angle_threshold))
            
            while queue:
                current_id = queue.popleft()
                current_normal = self.face_normals.get(current_id)
                
                neighbors = self.face_adjacency.get(current_id, [])
                for nid in neighbors:
                    if nid in visited_global:
                        continue
                        
                    # 1. Break Edge Check (Strict Dam)
                    edge_key = tuple(sorted([current_id, nid]))
                    if edge_key in self.break_edges:
                        continue
                        
                    # 2. Angle Check
                    neighbor_normal = self.face_normals.get(nid)
                    if neighbor_normal is None: 
                        continue
                        
                    # Local smoothness
                    if np.dot(current_normal, neighbor_normal) < threshold_cos:
                        continue
                        
                    # Global drift
                    if np.dot(start_normal, neighbor_normal) < start_threshold_cos:
                        continue
                        
                    # Add to partition
                    visited_global.add(nid)
                    partition_faces.add(nid)
                    queue.append(nid)
            
            # Assign partition ID to these faces
            for fid in partition_faces:
                self.element_to_face_partition[fid] = partition_id
                
        print(f"[ZoneManager] Segmented mesh into {partition_id} logical face partitions.")

    def _compute_vertex_curvature(self, points: np.ndarray, elements: List[Dict]):
        """
        Compute approximate mean curvature at each vertex using angle defect method.
        This is simpler and more robust than cotangent weights for triangle meshes.
        """
        self.vertex_curvature.clear()
        
        # Track which vertices belong to which faces
        vertex_faces = collections.defaultdict(list)
        for elem in elements:
            if elem.get('type') not in ('triangle', 'quadrilateral'):
                continue
            nodes = elem.get('nodes', [])
            for n in nodes:
                vertex_faces[n].append(elem.get('id'))
        
        # Compute curvature at each vertex
        for vertex_idx, face_ids in vertex_faces.items():
            if len(face_ids) < 2:
                continue
                
            # Get normals of all faces touching this vertex
            normals = []
            for fid in face_ids:
                if fid in self.face_normals:
                    normals.append(self.face_normals[fid])
            
            if len(normals) < 2:
                continue
            
            # Curvature approximation: variance of normal directions
            # High variance = high curvature
            normals = np.array(normals)
            mean_normal = np.mean(normals, axis=0)
            norm = np.linalg.norm(mean_normal)
            if norm > 1e-9:
                mean_normal /= norm
            
            # Variance from mean (1 - dot product)
            deviations = [1.0 - abs(np.dot(n, mean_normal)) for n in normals]
            curvature = np.mean(deviations)
            self.vertex_curvature[vertex_idx] = curvature
    
    def _detect_break_edges(self):
        """
        Detect edges with high curvature difference - these are natural face boundaries.
        """
        self.break_edges.clear()
        
        if not self.vertex_curvature:
            return
        
        # Check all edges in adjacency
        checked = set()
        for fid1, neighbors in self.face_adjacency.items():
            for fid2 in neighbors:
                edge_key = tuple(sorted([fid1, fid2]))
                if edge_key in checked:
                    continue
                checked.add(edge_key)
                
                # Get curvature at shared vertices (simplified: use face centers)
                c1 = self.face_normals.get(fid1)
                c2 = self.face_normals.get(fid2)
                
                if c1 is not None and c2 is not None:
                    # Dihedral angle as curvature proxy
                    dot = np.dot(c1, c2)
                    angle_diff = np.arccos(np.clip(dot, -1, 1))
                    
                    # If angle > threshold, it's a break edge
                    if angle_diff > self.curvature_break_threshold:
                        self.break_edges.add(edge_key)

    def select_face(self, face_id: int, toggle: bool = False, multi_select: bool = False):
        """
        Handle a click on a face.
        
        Args:
            face_id: The ID of the clicked face.
            toggle: If True, toggle selection state of this face (Ctrl behavior).
            multi_select: If True, add to existing selection (Shift behavior). 
                          If False, clear existing selection first.
        """
        if face_id is None:
            return

        print(f"[ZoneManager] select_face id={face_id}, toggle={toggle}, multi={multi_select}")

        if not multi_select and not toggle:
            # Standard click: Select only this one (and its region), clear others
            self.selected_faces.clear()
            self.spill_select(face_id)
            
        elif multi_select and not toggle:
            # Shift+Click: Add region to selection
            self.spill_select(face_id)
            
        elif toggle:
            # Ctrl+Click: Toggle (Tricky with spill, for now just toggle single or spill add?)
            # Standard behavior: If start face is selected, deselect region. Else select region.
            if face_id in self.selected_faces:
                self.selected_faces.remove(face_id)
                # Ideally we'd remove the whole region, but for now remove single is safer to avoid removing unintended parts
            else:
                self.spill_select(face_id)

    def spill_select(self, start_face_id: int):
        """
        Perform BFS flood fill selection based on angle threshold.
        Extends the current selection.
        """
        if start_face_id not in self.face_normals:
            print(f"[ZoneManager] WARNING: start_face_id {start_face_id} not in face_normals (len={len(self.face_normals)})")
            # Fallback: just add this single face
            self.selected_faces.add(start_face_id)
            return
            
        print(f"[ZoneManager] Starting spill select from {start_face_id}... (threshold={self.spill_angle_threshold}°)")
        
        # Use a queue for BFS
        queue = collections.deque([start_face_id])
        visited = set([start_face_id])
        
        # Add start face to selection immediately
        self.selected_faces.add(start_face_id)
        
        # Store starting normal to prevent drift around curved surfaces
        start_normal = self.face_normals[start_face_id]
        start_threshold_cos = np.cos(np.radians(30.0))  # 30° from starting normal
        
        threshold_cos = np.cos(np.radians(self.spill_angle_threshold))
        
        while queue:
            current_id = queue.popleft()
            current_normal = self.face_normals.get(current_id)
            if current_normal is None:
                continue
            
            # Check neighbors
            neighbors = self.face_adjacency.get(current_id, [])
            if not neighbors and len(visited) < 5:  # Only print for first few if empty
                print(f"[ZoneManager] Face {current_id} has NO neighbors! (adjacency graph may be empty)")
                
            for neighbor_id in neighbors:
                if neighbor_id not in visited:
                    # Check if this edge is a curvature break (high dihedral angle)
                    edge_key = tuple(sorted([current_id, neighbor_id]))
                    if edge_key in self.break_edges:
                        # Don't cross break edges - this is a face boundary
                        continue
                    
                    # Check angle threshold
                    neighbor_normal = self.face_normals.get(neighbor_id)
                    if neighbor_normal is None:
                        continue
                    
                    # ANTI-DRIFT CHECK: Compare neighbor to STARTING normal
                    # This prevents gradual drift around curved surfaces
                    dot_to_start = np.dot(start_normal, neighbor_normal)
                    if dot_to_start < start_threshold_cos:
                        # Too far from starting orientation - don't include
                        continue
                        
                    dot_prod = np.dot(current_normal, neighbor_normal)
                    
                    # Dot product close to 1.0 means parallel
                    if dot_prod >= threshold_cos:
                        # Match!
                        visited.add(neighbor_id)
                        queue.append(neighbor_id)
                        self.selected_faces.add(neighbor_id)
                        
        print(f"[ZoneManager] Spill selected {len(visited)} faces.")

    def create_zone(self, name: str, zone_type: str) -> bool:
        """
        Create a new zone from currently selected faces.
        Returns True if successful.
        """
        if not name or not self.selected_faces:
            return False
            
        print(f"[ZoneManager] Creating zone '{name}' ({zone_type}) with {len(self.selected_faces)} faces")
        
        # Remove these faces from any existing zones
        for fid in self.selected_faces:
            if fid in self.face_to_zone:
                old_zone = self.face_to_zone[fid]
                if old_zone in self.zone_registry:
                    self.zone_registry[old_zone].face_ids.discard(fid)
        
        # Create new zone data
        # Assign a random color (or cycled) - for now just hash based
        import hashlib
        h = hashlib.md5(name.encode()).hexdigest()
        r = int(h[0:2], 16) / 255.0
        g = int(h[2:4], 16) / 255.0
        b = int(h[4:6], 16) / 255.0
        
        new_zone = ZoneData(name=name, zone_type=zone_type, color=(r,g,b))
        new_zone.face_ids = set(self.selected_faces) # Copy
        
        self.zone_registry[name] = new_zone
        
        # Update inverse map
        for fid in self.selected_faces:
            self.face_to_zone[fid] = name
            
        # Clear selection after creation? User usually expects this.
        self.selected_faces.clear()
        return True

    def delete_zone(self, name: str):
        if name in self.zone_registry:
            # Update inverse map
            for fid in self.zone_registry[name].face_ids:
                if self.face_to_zone.get(fid) == name:
                    del self.face_to_zone[fid]
            del self.zone_registry[name]

    def get_zone_info_for_face(self, face_id: int) -> Optional[ZoneData]:
        zone_name = self.face_to_zone.get(face_id)
        if zone_name:
            return self.zone_registry.get(zone_name)
        return None
        
    def get_boundary_classifier(self) -> Callable[[np.ndarray], str]:
        """
        Returns a classifier function compatible with export_fluent_msh.py
        Note: The exporter uses centroids to classify. We need to be careful.
        Ideally we should pass IDs, but the exporter is currently geometry-based.
        
        We will modify the exporter to accept explicit zone maps, but for backward
        compatibility or if we have to use the centroid classifier:
        """
        # This is tricky because centroid matching is float-imprecise.
        # Ideally we pass the zone_assignments map directly to the exporter.
        pass
