
import gmsh
import numpy as np
import sys
import os
from pathlib import Path
from typing import Tuple, Optional, Dict, List

# Add parent directory to path to import core modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.mesh_generator import BaseMeshGenerator
from core.config import Config

class PolyhedralMeshGenerator(BaseMeshGenerator):
    """
    Polyhedral Mesh Generator (Dual Graph Method)
    
    Converts a tetrahedral mesh into a polyhedral mesh by computing the dual graph.
    - Primal Nodes -> Dual Cells (Polyhedra)
    - Primal Edges -> Dual Faces (Polygons)
    - Primal Faces -> Dual Edges
    - Primal Cells -> Dual Vertices
    
    For visualization purposes, this strategy outputs the FACES of the polyhedral mesh
    as a surface mesh (triangulated polygons).
    """
    
    def __init__(self, config: Optional[Config] = None):
        super().__init__(config)
        
    def run_meshing_strategy(self, input_file: str, output_file: str) -> bool:
        """
        Run the polyhedral meshing strategy.
        1. Generate/Load Tet Mesh
        2. Compute Dual Graph (Polyhedral Faces)
        3. Save as surface mesh for visualization
        """
        self.log_message("\n" + "=" * 60)
        self.log_message("POLYHEDRAL MESH GENERATION (DUAL GRAPH)")
        self.log_message("=" * 60)
        
        try:
            # 1. Generate Base Tetrahedral Mesh
            self.log_message("Step 1: Generating Base Tetrahedral Mesh...")
            
            # Initialize Gmsh
            if not gmsh.is_initialized():
                gmsh.initialize()
                
            gmsh.model.add("polyhedral_base")
            
            # Load input file (CAD or Mesh)
            file_ext = Path(input_file).suffix.lower()
            if file_ext == '.msh':
                gmsh.open(input_file)
            else:
                # It's a CAD file, generate tet mesh
                gmsh.open(input_file)
                
                # Calculate bounding box to set appropriate mesh size
                try:
                    # Get bounding box of the model
                    # (xmin, ymin, zmin, xmax, ymax, zmax)
                    bbox = gmsh.model.getBoundingBox(-1, -1)
                    dx = bbox[3] - bbox[0]
                    dy = bbox[4] - bbox[1]
                    dz = bbox[5] - bbox[2]
                    diagonal = np.sqrt(dx*dx + dy*dy + dz*dz)
                    
                    self.log_message(f"Model Bounding Box: {dx:.2f} x {dy:.2f} x {dz:.2f}")
                    self.log_message(f"Model Diagonal: {diagonal:.2f}")
                    
                    # Set lc to be coarse (e.g. diagonal / 10)
                    # For a 1000mm cube, diagonal is ~1732. lc = 173.
                    # For a 1mm cube, diagonal is ~1.732. lc = 0.173.
                    lc = diagonal / 10.0 
                    
                except Exception as e:
                    self.log_message(f"[WARNING] Could not calculate bounding box: {e}")
                    lc = 0.5 # Fallback
                    
                gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc * 0.5)
                gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
                gmsh.option.setNumber("Mesh.Algorithm", 6) # Frontal-Delaunay
                gmsh.option.setNumber("Mesh.Algorithm3D", 1) # Delaunay
                
                self.log_message(f"Meshing with target size ~{lc:.3f}...")
                gmsh.model.mesh.generate(3)
                
            # 2. Extract Mesh Data
            self.log_message("Step 2: Extracting Mesh Data...")
            
            # Get Nodes
            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            nodes = np.array(node_coords).reshape(-1, 3)
            # Map tag -> index
            node_map = {tag: i for i, tag in enumerate(node_tags)}
            
            # Get Tets
            tet_type = 4
            tet_tags, tet_node_tags = gmsh.model.mesh.getElementsByType(tet_type)
            if len(tet_tags) == 0:
                self.log_message("[!] No tetrahedra found in base mesh!")
                return False
                
            tets = np.array(tet_node_tags).reshape(-1, 4)
            # Map tags to indices
            tets_idx = np.vectorize(node_map.get)(tets)
            
            self.log_message(f"Base mesh: {len(nodes)} nodes, {len(tets)} tets")
            
            # 2.5 Extract Surface Topology for Boundary Faces
            self.log_message("Step 2.5: Extracting surface triangles for boundary closure...")
            
            # Get surface triangles (2D elements on 3D boundary)
            surface_tris = []
            tri_type = 2  # Linear triangle
            
            try:
                # Correct Gmsh API usage: Iterate over surface entities (dim=2)
                surface_entities = gmsh.model.getEntities(2)
                all_surface_tris = []
                
                for entity in surface_entities:
                    tag = entity[1]
                    # Get elements for this surface entity
                    # getElements returns: elementTypes, elementTags, nodeTags
                    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2, tag)
                    
                    for e_type, e_tags, e_nodes in zip(elem_types, elem_tags, elem_node_tags):
                        if e_type == tri_type:
                            # Reshape nodes to (num_elements, 3)
                            tris = np.array(e_nodes).reshape(-1, 3)
                            all_surface_tris.append(tris)
                
                if all_surface_tris:
                    # Combine all triangles from all surfaces
                    combined_tris = np.vstack(all_surface_tris)
                    # Map node tags to indices
                    tris_idx = np.vectorize(node_map.get)(combined_tris)
                    surface_tris = tris_idx
                    self.log_message(f"Found {len(surface_tris)} surface triangles")
                else:
                    self.log_message("No surface triangles found.")
                    
            except Exception as e:
                self.log_message(f"Warning: Could not extract surface triangles: {e}")
                import traceback
                self.log_message(traceback.format_exc())
            
            # 3. Compute Dual Graph (Polyhedral Faces)
            self.log_message("Step 3: Computing Dual Polyhedral Faces...")
            
            # Calculate Centroids (Dual Vertices)
            tet_coords = nodes[tets_idx]
            centroids = np.mean(tet_coords, axis=1)
            
            # We need to identify internal edges to create dual faces.
            # An internal edge in the primal mesh corresponds to a face in the dual mesh.
            # The face connects the centroids of the tets sharing that edge.
            
            # Build Edge -> Tets adjacency
            # An edge is defined by sorted pair of node indices (n1, n2)
            edge_to_tets = {}
            
            for t_idx, tet in enumerate(tets_idx):
                # 6 edges per tet
                edges = [
                    tuple(sorted((tet[0], tet[1]))),
                    tuple(sorted((tet[0], tet[2]))),
                    tuple(sorted((tet[0], tet[3]))),
                    tuple(sorted((tet[1], tet[2]))),
                    tuple(sorted((tet[1], tet[3]))),
                    tuple(sorted((tet[2], tet[3])))
                ]
                for edge in edges:
                    if edge not in edge_to_tets:
                        edge_to_tets[edge] = []
                    edge_to_tets[edge].append(t_idx)
            
            self.log_message(f"Found {len(edge_to_tets)} total edges")
            
            # Filter for internal edges (shared by 3+ tets usually, but at least 1)
            # Actually, boundary edges (on surface) also create dual faces (clipped).
            # But for "Polyhedral" visualization, we mainly care about the internal structure.
            # Let's generate faces for ALL edges that have > 1 tet.
            # If an edge has only 1 tet, it's on a sharp boundary or something weird? 
            # Actually, surface edges have > 1 tet usually? No, a surface edge might have only 1 tet if it's on the boundary volume.
            # Wait, an edge on the boundary surface is shared by some number of tets inside the volume.
            # The dual face for a boundary edge would be perpendicular to the boundary.
            
            dual_faces = [] # List of polygons (list of centroid indices)
            edge_to_face_idx = {} # Map edge tuple -> index in dual_faces
            
            for edge, connected_tets in edge_to_tets.items():
                # Include ALL edges, even boundary ones
                # Boundary edges (< 3 tets) still form valid dual faces
                if len(connected_tets) < 1:
                    continue
                    
                # For edges with only 1-2 tets, just use them as-is (no need for ordering)
                if len(connected_tets) <= 2:
                    ordered_tet_indices = list(connected_tets)
                else:
                    # We need to order the tets around the edge to form a valid polygon.
                    # The centroids must be ordered.
                    # Project centroids onto a plane perpendicular to the edge.
                    
                    # Edge vector
                    p1 = nodes[edge[0]]
                    p2 = nodes[edge[1]]
                    edge_vec = p2 - p1
                    edge_len = np.linalg.norm(edge_vec)
                    if edge_len < 1e-9:
                        continue
                    edge_vec /= edge_len
                    
                    # Get centroids of connected tets
                    c_points = centroids[connected_tets]
                    
                    # Project points onto plane perpendicular to edge_vec
                    # v_proj = v - (v . edge_vec) * edge_vec
                    # We use the first point p1 as origin for projection
                    vecs = c_points - p1
                    dots = np.sum(vecs * edge_vec, axis=1)[:, np.newaxis]
                    projected = vecs - dots * edge_vec
                    
                    # Now we have 3D points on a plane. We need 2D coords to sort by angle.
                    # Create a local basis.
                    # u = normalized first projected vector (if not zero)
                    u = projected[0]
                    if np.linalg.norm(u) < 1e-9:
                        # All centroids are collinear with edge? Degenerate case
                        ordered_tet_indices = list(connected_tets)
                    else:
                        u /= np.linalg.norm(u)
                        
                        # v = edge_vec x u
                        v = np.cross(edge_vec, u)
                        
                        # Calculate angles
                        x = np.sum(projected * u, axis=1)
                        y = np.sum(projected * v, axis=1)
                        angles = np.arctan2(y, x)
                        
                        # Sort indices by angle
                        sorted_indices = np.argsort(angles)
                        ordered_tet_indices = [connected_tets[i] for i in sorted_indices]
                
                # Store the face
                face_idx = len(dual_faces)
                dual_faces.append(ordered_tet_indices)
                edge_to_face_idx[edge] = face_idx
                
            self.log_message(f"Generated {len(dual_faces)} dual faces (internal edges)")
            
            # 3.5 Add Boundary Closure Faces for Surface Nodes
            self.log_message("Step 3.5: Adding boundary closure faces...")
            
            boundary_faces_added = 0
            for node_idx in node_surface_tris.keys():
                # Get surface triangles sharing this boundary node
                tri_indices = node_surface_tris[node_idx]
                
                # For each triangle, create a closure face using:
                # - The tet centroid adjacent to the triangle
                # - The triangle vertices (in dual space, we need the primal triangle)
                
                # Find tets adjacent to this node
                if node_idx in node_to_edges:
                    # Get all tets connected via edges from this node
                    connected_tets_set = set()
                    for edge in node_to_edges[node_idx]:
                        if edge in edge_to_tets:
                            connected_tets_set.update(edge_to_tets[edge])
                    
                    # Create a boundary closure face from connected tet centroids
                    # This "caps" the open side of the boundary polyhedron
                    if len(connected_tets_set) >= 3:
                        # Sort centroids to form a valid closure polygon
                        boundary_tet_list = list(connected_tets_set)
                        
                        # Use the first connected tet centroid to project others
                        if len(boundary_tet_list) >= 3:
                            face_idx = len(dual_faces)
                            dual_faces.append(boundary_tet_list)
                            # Don't add to edge_to_face_idx since this isn't from an edge
                            boundary_faces_added += 1
            
            self.log_message(f"Added {boundary_faces_added} boundary closure faces")
            self.log_message(f"Total dual faces: {len(dual_faces)} (internal + boundary)")
            
            # ---------------------------------------------------------
            # 4. Construct Full Polyhedral Data Structure
            # ---------------------------------------------------------
            self.log_message("Step 4: Constructing Polyhedral Cells...")
            
            # Build Node -> Edges adjacency to find faces for each primal node
            node_to_edges = {}
            for edge in edge_to_face_idx.keys():
                n1, n2 = edge
                if n1 not in node_to_edges: node_to_edges[n1] = []
                if n2 not in node_to_edges: node_to_edges[n2] = []
                node_to_edges[n1].append(edge)
                node_to_edges[n2].append(edge)
            
            # Construct Polyhedra
            # Each primal node becomes a dual polyhedron
            polyhedral_elements = []
            
            # Debug counters
            skipped_no_edges = 0
            skipped_too_few_faces = 0
            face_count_histogram = {}
            
            for p_node_idx in range(len(nodes)):
                if p_node_idx not in node_to_edges:
                    skipped_no_edges += 1
                    continue
                    
                connected_edges = node_to_edges[p_node_idx]
                cell_faces = []
                
                for edge in connected_edges:
                    if edge in edge_to_face_idx:
                        face_idx = edge_to_face_idx[edge]
                        # The face is a list of dual node indices (centroids)
                        face_nodes = dual_faces[face_idx]
                        cell_faces.append(face_nodes)
                
                # Track face count distribution
                num_faces = len(cell_faces)
                face_count_histogram[num_faces] = face_count_histogram.get(num_faces, 0) + 1
                
                # Lowered threshold to 1 to include boundary polyhedra
                # VTK can handle open/partial polyhedra on boundaries
                if len(cell_faces) >= 1:
                    polyhedral_elements.append({
                        'id': p_node_idx + 1,
                        'type': 'polyhedron',
                        'faces': cell_faces
                    })
                else:
                    skipped_too_few_faces += 1
            
            self.log_message(f"Constructed {len(polyhedral_elements)} polyhedral cells")
            self.log_message(f"Skipped: {skipped_no_edges} (no edges), {skipped_too_few_faces} (too few faces)")
            self.log_message(f"Face count histogram: {sorted(face_count_histogram.items())}")
            
            # ---------------------------------------------------------
            # 5. Save Data
            # ---------------------------------------------------------
            import json
            
            # Save JSON for the viewer
            json_output_file = str(Path(output_file).with_suffix('.json'))
            
            poly_data = {
                "nodes": {i: c.tolist() for i, c in enumerate(centroids)},
                "elements": polyhedral_elements
            }
            
            with open(json_output_file, 'w') as f:
                json.dump(poly_data, f)
                
            self.log_message(f"Saved polyhedral data to {json_output_file}")
            
            # Also save the surface mesh (triangulated) for legacy/fallback
            gmsh.model.add("dual_mesh_viz")
            surf_tag = gmsh.model.addDiscreteEntity(2)
            c_tags = list(range(1, len(centroids) + 1))
            c_coords = centroids.flatten()
            gmsh.model.mesh.addNodes(2, surf_tag, c_tags, c_coords)
            
            tri_tags = []
            tri_nodes = []
            tag_counter = 1
            
            for face_indices in dual_faces:
                face_tags = [i + 1 for i in face_indices]
                v0 = face_tags[0]
                for i in range(1, len(face_tags) - 1):
                    v1 = face_tags[i]
                    v2 = face_tags[i+1]
                    tri_tags.append(tag_counter)
                    tri_nodes.extend([v0, v1, v2])
                    tag_counter += 1
            
            if tri_tags:
                gmsh.model.mesh.addElements(2, surf_tag, [2], [tri_tags], [tri_nodes])
            
            gmsh.write(output_file)
            self.log_message(f"Saved surface visualization to {output_file}")
            
            return True
            
        except Exception as e:
            self.log_message(f"[!] Polyhedral generation failed: {e}")
            import traceback
            self.log_message(traceback.format_exc())
            return False


