
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
            
            # Map node_index -> list of surface triangle indices
            node_surface_tris = {} 
            # Map edge (n1, n2) -> list of surface triangle indices
            edge_surface_tris = {}
            
            surface_centroids = []
            
            try:
                # Get surface entities (dim=2)
                surface_entities = gmsh.model.getEntities(2)
                all_surface_tris = []
                
                tri_count = 0
                
                for entity in surface_entities:
                    tag = entity[1]
                    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2, tag)
                    
                    for e_type, e_tags, e_nodes in zip(elem_types, elem_tags, elem_node_tags):
                        if e_type == 2: # Triangle
                            tris = np.array(e_nodes).reshape(-1, 3)
                            
                            for tri in tris:
                                # Map tags to indices
                                try:
                                    tri_idx = [node_map[n] for n in tri]
                                except KeyError:
                                    continue # Skip if node not found (shouldn't happen)
                                
                                # Store Surface Triangle
                                all_surface_tris.append(tri_idx)
                                current_tri_idx = tri_count
                                tri_count += 1
                                
                                # Calculate Centroid
                                coords = nodes[tri_idx]
                                center = np.mean(coords, axis=0)
                                surface_centroids.append(center)
                                
                                # Map Nodes to this Tri
                                for n in tri_idx:
                                    if n not in node_surface_tris: node_surface_tris[n] = []
                                    node_surface_tris[n].append(current_tri_idx)
                                
                                # Map Edges to this Tri
                                edges = [
                                    tuple(sorted((tri_idx[0], tri_idx[1]))),
                                    tuple(sorted((tri_idx[1], tri_idx[2]))),
                                    tuple(sorted((tri_idx[2], tri_idx[0])))
                                ]
                                for edge in edges:
                                    if edge not in edge_surface_tris: edge_surface_tris[edge] = []
                                    edge_surface_tris[edge].append(current_tri_idx)
                
                surface_centroids = np.array(surface_centroids)
                self.log_message(f"Found {len(surface_centroids)} surface triangles")
                
            except Exception as e:
                self.log_message(f"Warning: Could not extract surface triangles: {e}")
                import traceback
                self.log_message(traceback.format_exc())
                surface_centroids = np.zeros((0, 3))

            # 3. Compute Dual Graph
            self.log_message("Step 3: Computing Dual Polyhedral Faces...")
            
            # ALL Dual Nodes = [Tet Centroids] + [Surface Triangle Centroids] + [Original Surface Nodes]
            # We need Original Surface Nodes to form the "Peaks" of the boundary cells, 
            # recovering volume at corners/curvature.
            
            tet_centroids = np.mean(nodes[tets_idx], axis=1)
            num_tets = len(tet_centroids)
            num_surf_tris = len(surface_centroids)
            
            # Identify surface nodes (we need their coordinates)
            # We can just add ALL nodes or just surface nodes? 
            # Adding only surface nodes breaks indexing unless handled carefully.
            # Simpler: Add ALL original nodes? No, that's wasteful for internal nodes.
            # Let's add ONLY surface nodes.
            
            surface_node_indices = list(node_surface_tris.keys())
            surface_node_map = {old_idx: i for i, old_idx in enumerate(surface_node_indices)}
            surface_node_coords = nodes[surface_node_indices]
            
            offset_surf_tris = num_tets
            offset_surf_nodes = num_tets + num_surf_tris
            
            if num_surf_tris > 0 and len(surface_node_indices) > 0:
                all_dual_nodes = np.vstack([tet_centroids, surface_centroids, surface_node_coords])
            else:
                all_dual_nodes = tet_centroids

            self.log_message(f"Total Dual Nodes: {len(all_dual_nodes)} " 
                             f"({num_tets} tets + {num_surf_tris} surf_tris + {len(surface_node_indices)} surf_nodes)")

            # Build Edge -> Tets adjacency
            edge_to_tets = {}
            for t_idx, tet in enumerate(tets_idx):
                edges = [
                    tuple(sorted((tet[0], tet[1]))),
                    tuple(sorted((tet[0], tet[2]))),
                    tuple(sorted((tet[0], tet[3]))),
                    tuple(sorted((tet[1], tet[2]))),
                    tuple(sorted((tet[1], tet[3]))),
                    tuple(sorted((tet[2], tet[3])))
                ]
                for edge in edges:
                    if edge not in edge_to_tets: edge_to_tets[edge] = []
                    edge_to_tets[edge].append(t_idx)
            
            dual_faces = [] 
            edge_to_face_idx = {}
            
            # --- 3A. Generate Faces for Primal Edges (Internal & Side-Walls) ---
            for edge, connected_tets in edge_to_tets.items():
                if len(connected_tets) < 1: continue
                
                is_boundary_edge = edge in edge_surface_tris
                
                # Case 1: Purely Internal Edge
                if not is_boundary_edge:
                    if len(connected_tets) <= 2:
                        ordered_indices = list(connected_tets)
                    else:
                        ordered_indices = self._order_angular(edge, connected_tets, nodes, tet_centroids)
                    
                    dual_faces.append(ordered_indices)
                    edge_to_face_idx[edge] = len(dual_faces) - 1
                    
                # Case 2: Boundary Edge (Needs Closure)
                else:
                    # Side Walls connect [Tet Centers] + [Surface Tri Centers]
                    s_tris = edge_surface_tris[edge]
                    s_indices = [idx + offset_surf_tris for idx in s_tris]
                    
                    combined_indices = list(connected_tets) + s_indices
                    ordered_indices = self._order_angular(edge, combined_indices, nodes, all_dual_nodes)
                    
                    dual_faces.append(ordered_indices)
                    edge_to_face_idx[edge] = len(dual_faces) - 1

            self.log_message(f"Generated {len(dual_faces)} dual faces from edges")
            
            # --- 3B. Generate Boundary Polygon Faces (Proper Voronoi) ---
            self.log_message("Step 3.5: Creating boundary polygon faces...")
            boundary_faces_added = 0
            
            # Track which boundary edges we've already processed
            processed_boundary_edges = set()
            
            # For each boundary edge, create a quadrilateral face
            # connecting the two surface triangle centroids and the two edge nodes
            for edge, s_tris in edge_surface_tris.items():
                if edge in processed_boundary_edges:
                    continue
                processed_boundary_edges.add(edge)
                
                # Boundary edges should have exactly 2 adjacent surface triangles
                if len(s_tris) != 2:
                    self.log_message(f"[WARN] Boundary edge {edge} has {len(s_tris)} adjacent triangles (expected 2)")
                    continue
                
                n1, n2 = edge  # Primal node indices
                t1_idx, t2_idx = s_tris  # Surface triangle indices
                
                # Convert to dual node indices
                # Surface triangle centroids are at offset_surf_tris + tri_idx
                c1 = t1_idx + offset_surf_tris
                c2 = t2_idx + offset_surf_tris
                
                # Surface nodes are at offset_surf_nodes + surface_node_map[node_idx]
                dn1 = offset_surf_nodes + surface_node_map[n1]
                dn2 = offset_surf_nodes + surface_node_map[n2]
                
                # Create quadrilateral face: [dn1, c1, dn2, c2]
                # Order matters for correct winding
                # We want: node1 -> centroid1 -> node2 -> centroid2
                face = [dn1, c1, dn2, c2]
                dual_faces.append(face)
                boundary_faces_added += 1
            
            self.log_message(f"Added {boundary_faces_added} boundary polygon faces")
            
            # Track face indices for cell construction
            # Map: boundary_edge -> face_index
            edge_to_boundary_face = {}
            face_start_idx = len(dual_faces) - boundary_faces_added
            
            idx = 0
            for edge in processed_boundary_edges:
                edge_to_boundary_face[edge] = face_start_idx + idx
                idx += 1


            # 4. Construct Cells
            self.log_message("Step 4: Constructing Polyhedral Cells...")
            
            node_to_edges = {}
            for edge in edge_to_face_idx.keys():
                n1, n2 = edge
                if n1 not in node_to_edges: node_to_edges[n1] = []
                if n2 not in node_to_edges: node_to_edges[n2] = []
                node_to_edges[n1].append(edge)
                node_to_edges[n2].append(edge)
            
            polyhedral_elements = []
            
            total_nodes = len(nodes)
            progress_step = max(1, total_nodes // 20)  # 5% increments
            
            for p_node_idx in range(total_nodes):
                cell_faces = []
                
                # Check if this is a surface node
                is_surface_node = p_node_idx in node_surface_tris
                
                # 1. Faces from Edges (Internal faces)
                if p_node_idx in node_to_edges:
                    for edge in node_to_edges[p_node_idx]:
                        if edge in edge_to_face_idx:
                            f_idx = edge_to_face_idx[edge]
                            cell_faces.append(dual_faces[f_idx])
                
                # 2. Boundary Faces (Quad faces if on boundary)
                # ONLY add boundary faces to SURFACE nodes (the "first" node of every edge)
                # This prevents duplication - each boundary face belongs to exactly one cell
                if is_surface_node and p_node_idx in node_to_edges:
                    for edge in node_to_edges[p_node_idx]:
                        if edge in edge_to_boundary_face:
                            # Only add if this node is the "first" node in the edge (or both are surface)
                            n1, n2 = edge
                            other_node = n2 if n1 == p_node_idx else n1
                            
                            # Add to this cell only if:
                            # - Other node is NOT a surface node, OR
                            # - This node has smaller index (tiebreaker)
                            if other_node not in node_surface_tris or p_node_idx < other_node:
                                f_idx = edge_to_boundary_face[edge]
                                cell_faces.append(dual_faces[f_idx])

                if len(cell_faces) >= 4:
                    polyhedral_elements.append({
                        'id': p_node_idx + 1,
                        'type': 'polyhedron',
                        'faces': cell_faces
                    })
                
                # Log progress in 5% increments
                if (p_node_idx + 1) % progress_step == 0 or p_node_idx == total_nodes - 1:
                    pct = int(((p_node_idx + 1) / total_nodes) * 100)
                    self.log_message(f"[Polyhedral] Conversion progress: {pct}% ({p_node_idx + 1}/{total_nodes} nodes processed)")
            
            self.log_message(f"Constructed {len(polyhedral_elements)} polyhedral cells")
            
            # 5. Save Data (JSON)
            import json
            json_output_file = str(Path(output_file).with_suffix('.json'))
            
            poly_data = {
                "nodes": {i: c.tolist() for i, c in enumerate(all_dual_nodes)},
                "elements": polyhedral_elements
            }
            
            with open(json_output_file, 'w') as f:
                json.dump(poly_data, f)
            
            self.log_message(f"Saved polyhedral data to {json_output_file}")
            
            # Save Debug Surface Mesh
            self._save_debug_surface(output_file, all_dual_nodes, dual_faces)
            
            return True
            
        except Exception as e:
            self.log_message(f"[!] Polyhedral generation failed: {e}")
            import traceback
            self.log_message(traceback.format_exc())
            return False

    def _order_angular(self, edge, indices, primal_nodes, dual_nodes):
        """Helper to order dual nodes angularly around a primal edge"""
        p1 = primal_nodes[edge[0]]
        p2 = primal_nodes[edge[1]]
        edge_vec = p2 - p1
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 1e-9: return list(indices)
        edge_vec /= edge_len
        
        points = dual_nodes[indices]
        
        # Project onto plane
        vecs = points - p1
        dots = np.sum(vecs * edge_vec, axis=1)[:, np.newaxis]
        projected = vecs - dots * edge_vec
        
        # Basis
        if len(projected) == 0: return list(indices)
        u = projected[0]
        if np.linalg.norm(u) < 1e-9: return list(indices)
        u /= np.linalg.norm(u)
        v = np.cross(edge_vec, u)
        
        x = np.sum(projected * u, axis=1)
        y = np.sum(projected * v, axis=1)
        angles = np.arctan2(y, x)
        
        sorted_idx = np.argsort(angles)
        return [indices[i] for i in sorted_idx]

    def _sort_points_angular(self, center, normal, indices, dual_nodes):
        """Sort points around a center with given normal"""
        points = dual_nodes[indices]
        
        # Basis
        if np.linalg.norm(normal) < 1e-9: return list(indices)
        normal /= np.linalg.norm(normal)
        
        u = points[0] - center
        u = u - np.dot(u, normal) * normal
        if np.linalg.norm(u) < 1e-9: return list(indices)
        u /= np.linalg.norm(u)
        
        v = np.cross(normal, u)
        
        vecs = points - center
        x = np.sum(vecs * u, axis=1)
        y = np.sum(vecs * v, axis=1)
        angles = np.arctan2(y, x)
        
        sorted_idx = np.argsort(angles)
        return [indices[i] for i in sorted_idx]

    def _save_debug_surface(self, filename, nodes, faces):
        gmsh.model.add("dual_debug")
        tag = gmsh.model.addDiscreteEntity(2)
        
        # Nodes
        n_tags = list(range(1, len(nodes)+1))
        gmsh.model.mesh.addNodes(2, tag, n_tags, nodes.flatten())
        
        # Elements (Triangulate faces)
        tri_tags = []
        tri_nodes = []
        t_id = 1
        for face in faces:
            fl = len(face)
            if fl < 3: continue
            f_tags = [i+1 for i in face]
            v0 = f_tags[0]
            for i in range(1, fl-1):
                tri_tags.append(t_id)
                tri_nodes.extend([v0, f_tags[i], f_tags[i+1]])
                t_id += 1
        
        if tri_tags:
            gmsh.model.mesh.addElements(2, tag, [2], [tri_tags], [tri_nodes])
            
        gmsh.write(filename)



