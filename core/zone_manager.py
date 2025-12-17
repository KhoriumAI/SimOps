
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
        
        # Constants
        self.spill_angle_threshold = 15.0 # degrees - Tight threshold to prevent spill across sharp edges

    def set_mesh_data(self, points: np.ndarray, elements: List[Dict]):
        """
        Initialize mesh connectivity for spill selection.
        Expects elements to be dictionaries with 'type', 'id', 'nodes'.
        Only processes surface elements (triangles/quads).
        """
        print(f"[ZoneManager] Building adjacency for {len(elements)} elements...")
        
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
            normal = np.cross(v1, v2)
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
                    # Check angle
                    neighbor_normal = self.face_normals.get(neighbor_id)
                    if neighbor_normal is None:
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
