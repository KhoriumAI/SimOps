#!/usr/bin/env python3
"""
Polyhedral and Hexahedral Mesh Converters

Converts tetrahedral meshes to:
1. Polyhedral meshes (via node-based dual/agglomeration)
2. Hexahedral meshes (via THex tet-to-hex splitting)

Based on industry-standard algorithms used in ANSYS Fluent, OpenFOAM, etc.
"""

import numpy as np
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
import gmsh


@dataclass
class PolyCell:
    """Polyhedral cell with arbitrary number of faces"""
    center: Tuple[float, float, float]  # Original node position
    vertices: List[Tuple[float, float, float]]  # Centroids of adjacent tets
    faces: List['PolyFace']

    def get_volume(self) -> float:
        """Calculate volume using divergence theorem"""
        volume = 0.0
        for face in self.faces:
            # Volume contribution from each face
            centroid = face.get_centroid()
            area_vector = face.get_area_vector()
            volume += np.dot(centroid, area_vector)
        return abs(volume) / 3.0


@dataclass
class PolyFace:
    """Polygonal face of a polyhedral cell"""
    vertices: List[Tuple[float, float, float]]

    def get_centroid(self) -> np.ndarray:
        """Calculate face centroid"""
        return np.mean(self.vertices, axis=0)

    def get_area_vector(self) -> np.ndarray:
        """Calculate area vector (normal * area)"""
        # Use cross product sum for arbitrary polygon
        center = self.get_centroid()
        area_vec = np.zeros(3)
        n = len(self.vertices)

        for i in range(n):
            v1 = np.array(self.vertices[i]) - center
            v2 = np.array(self.vertices[(i+1) % n]) - center
            area_vec += np.cross(v1, v2)

        return area_vec / 2.0


class TetToPolyConverter:
    """
    Converts tetrahedral mesh to polyhedral mesh via node-based dual/agglomeration.

    Algorithm:
    1. Calculate centroids of all tetrahedra
    2. For each node N in tet mesh:
       - Create new polyhedral cell P centered at N
       - Vertices of P = centroids of tets adjacent to N
       - Faces of P = polygons formed by tets around each edge connected to N

    Result: Robust polyhedral mesh suitable for CFD (ANSYS Fluent, OpenFOAM)
    """

    def __init__(self):
        self.tet_centroids: Dict[int, np.ndarray] = {}
        self.node_to_tets: Dict[int, Set[int]] = defaultdict(set)
        self.edge_to_tets: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        self.nodes: Dict[int, np.ndarray] = {}
        self.tets: Dict[int, List[int]] = {}

    def load_from_gmsh(self, mesh_file: str):
        """Load tetrahedral mesh from Gmsh file"""
        gmsh.initialize()
        gmsh.open(mesh_file)

        # Get nodes
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        for i, tag in enumerate(node_tags):
            self.nodes[int(tag)] = np.array([
                node_coords[3*i],
                node_coords[3*i + 1],
                node_coords[3*i + 2]
            ])

        # Get tetrahedra (element type 4)
        tet_type = 4
        tet_tags, tet_nodes = gmsh.model.mesh.getElementsByType(tet_type)

        for i, tet_tag in enumerate(tet_tags):
            nodes = [
                int(tet_nodes[4*i]),
                int(tet_nodes[4*i + 1]),
                int(tet_nodes[4*i + 2]),
                int(tet_nodes[4*i + 3])
            ]
            self.tets[int(tet_tag)] = nodes

            # Build adjacency maps
            for node in nodes:
                self.node_to_tets[node].add(int(tet_tag))

            # Build edge-to-tet map
            edges = [
                (nodes[0], nodes[1]),
                (nodes[0], nodes[2]),
                (nodes[0], nodes[3]),
                (nodes[1], nodes[2]),
                (nodes[1], nodes[3]),
                (nodes[2], nodes[3])
            ]
            for edge in edges:
                edge_key = tuple(sorted(edge))
                self.edge_to_tets[edge_key].append(int(tet_tag))

        gmsh.finalize()
        print(f"[OK] Loaded {len(self.nodes)} nodes, {len(self.tets)} tets")

    def calculate_tet_centroids(self):
        """Calculate centroid of each tetrahedron"""
        for tet_id, node_ids in self.tets.items():
            points = [self.nodes[nid] for nid in node_ids]
            centroid = np.mean(points, axis=0)
            self.tet_centroids[tet_id] = centroid

        print(f"[OK] Calculated {len(self.tet_centroids)} tet centroids")

    def order_vertices_circularly(self, vertices: List[np.ndarray],
                                   axis_start: np.ndarray,
                                   axis_end: np.ndarray) -> List[np.ndarray]:
        """
        Order vertices circularly around an edge (axis) to form valid polygon.

        This is critical for non-intersecting faces.
        """
        if len(vertices) < 3:
            return vertices

        # Edge direction
        axis = axis_end - axis_start
        axis = axis / np.linalg.norm(axis)

        # Project vertices onto plane perpendicular to axis
        midpoint = (axis_start + axis_end) / 2.0

        # Calculate angles around the axis
        angles = []
        for v in vertices:
            vec = v - midpoint
            # Remove component along axis
            vec = vec - np.dot(vec, axis) * axis

            if np.linalg.norm(vec) < 1e-10:
                angles.append(0.0)
            else:
                # Use arbitrary reference vector perpendicular to axis
                ref = np.array([axis[1], -axis[0], 0.0])
                if np.linalg.norm(ref) < 1e-6:
                    ref = np.array([0.0, axis[2], -axis[1]])
                ref = ref / np.linalg.norm(ref)

                # Calculate angle
                angle = np.arctan2(
                    np.dot(np.cross(ref, vec), axis),
                    np.dot(ref, vec)
                )
                angles.append(angle)

        # Sort by angle
        sorted_indices = np.argsort(angles)
        return [vertices[i] for i in sorted_indices]

    def convert(self) -> List[PolyCell]:
        """
        Main conversion: tetrahedral mesh -> polyhedral mesh

        Returns list of polyhedral cells.
        """
        print("\n" + "="*70)
        print("TETRAHEDRAL -> POLYHEDRAL CONVERSION")
        print("="*70)

        self.calculate_tet_centroids()

        poly_cells = []

        print(f"\nBuilding {len(self.nodes)} polyhedral cells...")

        for node_id, node_pos in self.nodes.items():
            # Create new polyhedral cell centered at this node
            cell = PolyCell(
                center=tuple(node_pos),
                vertices=[],
                faces=[]
            )

            # Get adjacent tets
            adjacent_tets = list(self.node_to_tets[node_id])

            # Vertices = centroids of adjacent tets
            cell.vertices = [
                tuple(self.tet_centroids[tet_id])
                for tet_id in adjacent_tets
            ]

            # Build faces based on edges connected to this node
            adjacent_edges = [
                edge for edge in self.edge_to_tets.keys()
                if node_id in edge
            ]

            for edge in adjacent_edges:
                # Get tets that share this edge
                tets_around_edge = self.edge_to_tets[edge]

                # Filter to only tets that also contain our node
                tets_around_edge = [
                    t for t in tets_around_edge
                    if node_id in self.tets[t]
                ]

                if len(tets_around_edge) < 2:
                    # Boundary edge, skip or handle specially
                    continue

                # Get centroids for face vertices
                face_vertices = [
                    self.tet_centroids[tet_id]
                    for tet_id in tets_around_edge
                ]

                # Order circularly around edge
                edge_nodes = list(edge)
                axis_start = self.nodes[edge_nodes[0]]
                axis_end = self.nodes[edge_nodes[1]]

                ordered_vertices = self.order_vertices_circularly(
                    face_vertices, axis_start, axis_end
                )

                # Create face
                face = PolyFace(vertices=[tuple(v) for v in ordered_vertices])
                cell.faces.append(face)

            poly_cells.append(cell)

        print(f"[OK] Created {len(poly_cells)} polyhedral cells")

        # Statistics
        face_counts = [len(c.faces) for c in poly_cells]
        vertex_counts = [len(c.vertices) for c in poly_cells]

        print(f"\nStatistics:")
        print(f"  Average faces per cell: {np.mean(face_counts):.1f}")
        print(f"  Average vertices per cell: {np.mean(vertex_counts):.1f}")
        print(f"  Min/Max faces: {min(face_counts)}/{max(face_counts)}")
        print(f"  Min/Max vertices: {min(vertex_counts)}/{max(vertex_counts)}")

        return poly_cells

    def export_to_vtk(self, poly_cells: List[PolyCell], output_file: str):
        """Export polyhedral mesh to VTK format"""
        with open(output_file, 'w') as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write("Polyhedral Mesh\n")
            f.write("ASCII\n")
            f.write("DATASET UNSTRUCTURED_GRID\n")

            # Collect all unique vertices
            vertex_set = set()
            for cell in poly_cells:
                vertex_set.update(cell.vertices)
                for face in cell.faces:
                    vertex_set.update(face.vertices)

            vertices = list(vertex_set)
            vertex_to_id = {v: i for i, v in enumerate(vertices)}

            # Write points
            f.write(f"\nPOINTS {len(vertices)} float\n")
            for v in vertices:
                f.write(f"{v[0]} {v[1]} {v[2]}\n")

            # Write cells (VTK polyhedron type = 42)
            f.write(f"\nCELLS {len(poly_cells)} ")

            # Calculate total size
            total_size = 0
            for cell in poly_cells:
                total_size += 1  # num faces
                for face in cell.faces:
                    total_size += 1 + len(face.vertices)  # num verts + verts

            f.write(f"{total_size}\n")

            for cell in poly_cells:
                f.write(f"{len(cell.faces)} ")  # Number of faces
                for face in cell.faces:
                    f.write(f"{len(face.vertices)} ")
                    for v in face.vertices:
                        f.write(f"{vertex_to_id[v]} ")
                f.write("\n")

            # Write cell types
            f.write(f"\nCELL_TYPES {len(poly_cells)}\n")
            for _ in poly_cells:
                f.write("42\n")  # VTK_POLYHEDRON

        print(f"[OK] Exported polyhedral mesh to {output_file}")


class TetToHexConverter:
    """
    Converts tetrahedral mesh to hexahedral mesh via THex splitting.

    Algorithm:
    Each tet is split into 4 hexahedra by adding:
    - 1 cell center node
    - 4 face center nodes
    - 6 edge midpoint nodes

    Total: 11 new nodes per tet -> 4 hexes per tet

    Warning: Resulting hexes may have poor quality (high skewness).
    Better for element count than structural analysis.
    """

    def __init__(self):
        self.nodes: Dict[int, np.ndarray] = {}
        self.tets: Dict[int, List[int]] = {}
        self.mid_edge_nodes: Dict[Tuple[int, int], int] = {}
        self.mid_face_nodes: Dict[Tuple[int, ...], int] = {}
        self.cell_center_nodes: Dict[int, int] = {}
        self.next_node_id = 1

    def load_from_gmsh(self, mesh_file: str):
        """Load tetrahedral mesh from Gmsh file"""
        gmsh.initialize()
        gmsh.open(mesh_file)

        # Get nodes
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        for i, tag in enumerate(node_tags):
            self.nodes[int(tag)] = np.array([
                node_coords[3*i],
                node_coords[3*i + 1],
                node_coords[3*i + 2]
            ])
            self.next_node_id = max(self.next_node_id, int(tag) + 1)

        # Get tetrahedra
        tet_type = 4
        tet_tags, tet_nodes = gmsh.model.mesh.getElementsByType(tet_type)

        for i, tet_tag in enumerate(tet_tags):
            nodes = [
                int(tet_nodes[4*i]),
                int(tet_nodes[4*i + 1]),
                int(tet_nodes[4*i + 2]),
                int(tet_nodes[4*i + 3])
            ]
            self.tets[int(tet_tag)] = nodes

        gmsh.finalize()
        print(f"[OK] Loaded {len(self.nodes)} nodes, {len(self.tets)} tets")

    def get_or_create_midpoint(self, n1: int, n2: int) -> int:
        """Get or create edge midpoint node"""
        edge_key = tuple(sorted([n1, n2]))

        if edge_key not in self.mid_edge_nodes:
            p1 = self.nodes[n1]
            p2 = self.nodes[n2]
            midpoint = (p1 + p2) / 2.0

            new_id = self.next_node_id
            self.next_node_id += 1

            self.nodes[new_id] = midpoint
            self.mid_edge_nodes[edge_key] = new_id

        return self.mid_edge_nodes[edge_key]

    def get_or_create_face_center(self, n1: int, n2: int, n3: int) -> int:
        """Get or create face center node"""
        face_key = tuple(sorted([n1, n2, n3]))

        if face_key not in self.mid_face_nodes:
            p1 = self.nodes[n1]
            p2 = self.nodes[n2]
            p3 = self.nodes[n3]
            center = (p1 + p2 + p3) / 3.0

            new_id = self.next_node_id
            self.next_node_id += 1

            self.nodes[new_id] = center
            self.mid_face_nodes[face_key] = new_id

        return self.mid_face_nodes[face_key]

    def get_or_create_cell_center(self, tet_id: int, n1: int, n2: int, n3: int, n4: int) -> int:
        """Get or create tet center node"""
        if tet_id not in self.cell_center_nodes:
            p1 = self.nodes[n1]
            p2 = self.nodes[n2]
            p3 = self.nodes[n3]
            p4 = self.nodes[n4]
            center = (p1 + p2 + p3 + p4) / 4.0

            new_id = self.next_node_id
            self.next_node_id += 1

            self.nodes[new_id] = center
            self.cell_center_nodes[tet_id] = new_id

        return self.cell_center_nodes[tet_id]

    def convert(self) -> List[List[int]]:
        """
        Main conversion: tetrahedral mesh -> hexahedral mesh

        Returns list of hexahedra (each is 8 node IDs).
        """
        print("\n" + "="*70)
        print("TETRAHEDRAL -> HEXAHEDRAL CONVERSION (THex)")
        print("="*70)

        hexes = []

        print(f"\nSplitting {len(self.tets)} tets into hexes...")

        for tet_id, tet_nodes in self.tets.items():
            v1, v2, v3, v4 = tet_nodes

            # Create 11 new nodes:
            # 6 edge midpoints
            n_12 = self.get_or_create_midpoint(v1, v2)
            n_13 = self.get_or_create_midpoint(v1, v3)
            n_14 = self.get_or_create_midpoint(v1, v4)
            n_23 = self.get_or_create_midpoint(v2, v3)
            n_24 = self.get_or_create_midpoint(v2, v4)
            n_34 = self.get_or_create_midpoint(v3, v4)

            # 4 face centers
            n_123 = self.get_or_create_face_center(v1, v2, v3)
            n_124 = self.get_or_create_face_center(v1, v2, v4)
            n_134 = self.get_or_create_face_center(v1, v3, v4)
            n_234 = self.get_or_create_face_center(v2, v3, v4)

            # 1 cell center
            n_c = self.get_or_create_cell_center(tet_id, v1, v2, v3, v4)

            # Create 4 hexahedra
            # Hex vertex ordering: bottom 4 CCW, top 4 CCW

            # Hex 1: anchored at v1
            hex1 = [v1, n_12, n_123, n_13, n_14, n_124, n_c, n_134]

            # Hex 2: anchored at v2
            hex2 = [v2, n_23, n_123, n_12, n_24, n_234, n_c, n_124]

            # Hex 3: anchored at v3
            hex3 = [v3, n_13, n_123, n_23, n_34, n_134, n_c, n_234]

            # Hex 4: anchored at v4
            hex4 = [v4, n_14, n_124, n_24, n_34, n_134, n_c, n_234]

            hexes.extend([hex1, hex2, hex3, hex4])

        print(f"[OK] Created {len(hexes)} hexahedra from {len(self.tets)} tets")
        print(f"[OK] Total nodes: {len(self.nodes)}")

        return hexes

    def export_to_gmsh(self, hexes: List[List[int]], output_file: str):
        """Export hexahedral mesh to Gmsh .msh format"""
        gmsh.initialize()
        gmsh.model.add("hex_mesh")

        # Add nodes
        for node_id, coords in self.nodes.items():
            gmsh.model.geo.addPoint(coords[0], coords[1], coords[2], tag=node_id)

        gmsh.model.geo.synchronize()

        # Add hexes manually to mesh
        # This is a bit hacky, but Gmsh API for volume elements is tricky

        gmsh.finalize()

        # Write custom .msh file
        with open(output_file, 'w') as f:
            f.write("$MeshFormat\n")
            f.write("4.1 0 8\n")
            f.write("$EndMeshFormat\n")

            # Nodes
            f.write("$Nodes\n")
            f.write(f"1 {len(self.nodes)} 1 {len(self.nodes)}\n")
            f.write(f"3 1 0 {len(self.nodes)}\n")

            for node_id in sorted(self.nodes.keys()):
                f.write(f"{node_id}\n")

            for node_id in sorted(self.nodes.keys()):
                coords = self.nodes[node_id]
                f.write(f"{coords[0]} {coords[1]} {coords[2]}\n")

            f.write("$EndNodes\n")

            # Elements (hex type = 5 in Gmsh)
            f.write("$Elements\n")
            f.write(f"1 {len(hexes)} 1 {len(hexes)}\n")
            f.write(f"3 1 5 {len(hexes)}\n")

            for i, hex_nodes in enumerate(hexes, 1):
                f.write(f"{i} " + " ".join(map(str, hex_nodes)) + "\n")

            f.write("$EndElements\n")

        print(f"[OK] Exported hexahedral mesh to {output_file}")


def main():
    """Test both converters"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python poly_hex_converter.py <input.msh> [--poly|--hex]")
        sys.exit(1)

    input_file = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "--poly"

    if mode == "--poly":
        print("\n🌍 POLYHEDRAL MESH CONVERSION")
        converter = TetToPolyConverter()
        converter.load_from_gmsh(input_file)
        poly_cells = converter.convert()

        output_file = input_file.replace(".msh", "_poly.vtk")
        converter.export_to_vtk(poly_cells, output_file)

    elif mode == "--hex":
        print("\n🧊 HEXAHEDRAL MESH CONVERSION (THex)")
        converter = TetToHexConverter()
        converter.load_from_gmsh(input_file)
        hexes = converter.convert()

        output_file = input_file.replace(".msh", "_hex.msh")
        converter.export_to_gmsh(hexes, output_file)

    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)

    print("\n" + "="*70)
    print("CONVERSION COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
