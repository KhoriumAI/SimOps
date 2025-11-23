"""
Quality-Based Mesh Coloring
============================

Color mesh elements based on quality metrics to visually identify problem areas.
This is a critical feature for mesh debugging - ANSYS does this beautifully.

Key concept: Map tetrahedral element quality to surface triangle colors
so users can see where the mesh quality is poor.
"""

import gmsh
import numpy as np
from typing import Dict, List, Tuple


class QualityColorMapper:
    """Maps mesh element quality to colors for visualization"""

    def __init__(self, mesh_file: str):
        """
        Initialize with a mesh file

        Args:
            mesh_file: Path to .msh file
        """
        self.mesh_file = mesh_file
        self.element_qualities = {}
        self.worst_elements = []

    def extract_element_qualities(self) -> Dict:
        """
        Extract per-element quality metrics from gmsh

        Returns:
            Dictionary with element_id -> quality mapping
        """
        gmsh.initialize()
        gmsh.open(self.mesh_file)

        # Get all tetrahedral elements
        elem_types, elem_tags, node_tags = gmsh.model.mesh.getElements(dim=3)

        qualities = {}
        all_qualities = []

        for elem_type, tags in zip(elem_types, elem_tags):
            # Get quality for these elements
            try:
                element_qualities = gmsh.model.mesh.getElementQualities(tags.tolist(), "minSICN")

                for elem_id, quality in zip(tags, element_qualities):
                    qualities[int(elem_id)] = float(quality)
                    all_qualities.append(float(quality))

            except Exception as e:
                print(f"Warning: Could not get qualities for element type {elem_type}: {e}")

        gmsh.finalize()

        self.element_qualities = qualities

        # Find worst 10% threshold
        if all_qualities:
            sorted_quals = sorted(all_qualities)
            threshold_idx = int(len(sorted_quals) * 0.1)  # Worst 10%
            threshold_quality = sorted_quals[threshold_idx] if threshold_idx < len(sorted_quals) else sorted_quals[0]

            self.worst_elements = [eid for eid, q in qualities.items() if q <= threshold_quality]

            return {
                'element_qualities': qualities,
                'worst_10_percent': self.worst_elements,
                'threshold': threshold_quality,
                'min_quality': min(all_qualities),
                'max_quality': max(all_qualities),
                'avg_quality': np.mean(all_qualities)
            }

        return {}

    def get_color_for_quality(self, quality: float, threshold: float) -> Tuple[float, float, float]:
        """
        Map quality value to RGB color

        Args:
            quality: Element quality (SICN, 0-1)
            threshold: Threshold for "bad" elements

        Returns:
            (R, G, B) tuple, values 0-1
        """
        if quality <= threshold:
            # Worst 10% = RED
            return (1.0, 0.0, 0.0)
        elif quality < 0.3:
            # Poor quality = ORANGE
            return (1.0, 0.5, 0.0)
        elif quality < 0.5:
            # Moderate = YELLOW
            return (1.0, 1.0, 0.0)
        elif quality < 0.7:
            # Good = YELLOW-GREEN
            return (0.5, 1.0, 0.0)
        else:
            # Excellent = GREEN
            return (0.0, 1.0, 0.0)

    def map_surface_to_volume_quality(self) -> Dict[int, float]:
        """
        Map surface triangles to adjacent tetrahedral quality

        Returns:
            Dictionary mapping triangle element ID to quality
        """
        gmsh.initialize()
        gmsh.open(self.mesh_file)

        # Get surface triangles
        surf_types, surf_tags, surf_nodes = gmsh.model.mesh.getElements(dim=2)

        # Get volume tets
        vol_types, vol_tags, vol_nodes = gmsh.model.mesh.getElements(dim=3)

        triangle_qualities = {}

        # For each surface triangle, find adjacent tetrahedron
        # and use its quality
        for surf_type, s_tags, s_nodes in zip(surf_types, surf_tags, surf_nodes):
            if surf_type in [2, 9]:  # Triangle types
                # Build node sets for each triangle
                nodes_per_tri = 3 if surf_type == 2 else 6

                for i, tri_id in enumerate(s_tags):
                    start_idx = i * nodes_per_tri
                    tri_nodes = set(s_nodes[start_idx:start_idx+3])  # Use first 3 nodes

                    # Find tet that contains all these nodes
                    best_quality = 1.0  # Default to good

                    for vol_type, v_tags, v_nodes in zip(vol_types, vol_tags, vol_nodes):
                        if vol_type in [4, 11]:  # Tet types
                            nodes_per_tet = 4 if vol_type == 4 else 10

                            for j, tet_id in enumerate(v_tags):
                                v_start = j * nodes_per_tet
                                tet_nodes = set(v_nodes[v_start:v_start+4])  # Use first 4 nodes

                                # If triangle nodes are subset of tet nodes, they're adjacent
                                if tri_nodes.issubset(tet_nodes):
                                    if int(tet_id) in self.element_qualities:
                                        quality = self.element_qualities[int(tet_id)]
                                        best_quality = min(best_quality, quality)  # Use worst adjacent

                    triangle_qualities[int(tri_id)] = best_quality

        gmsh.finalize()
        return triangle_qualities

    def write_quality_visualization_file(self, output_file: str):
        """
        Write a VTK file with per-element colors based on quality

        Args:
            output_file: Output .vtk file path
        """
        # Extract qualities
        quality_data = self.extract_element_qualities()
        if not quality_data:
            return

        threshold = quality_data['threshold']

        # Get triangle-to-quality mapping
        tri_qualities = self.map_surface_to_volume_quality()

        gmsh.initialize()
        gmsh.open(self.mesh_file)

        # Get surface mesh
        surf_types, surf_tags, surf_nodes = gmsh.model.mesh.getElements(dim=2)

        # Get node coordinates
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()

        # Build node map
        nodes = {}
        for i, tag in enumerate(node_tags):
            coord_idx = i * 3
            nodes[int(tag)] = (
                node_coords[coord_idx],
                node_coords[coord_idx + 1],
                node_coords[coord_idx + 2]
            )

        gmsh.finalize()

        # Write VTK file
        with open(output_file, 'w') as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write("Mesh Quality Visualization\n")
            f.write("ASCII\n")
            f.write("DATASET POLYDATA\n")

            # Write points
            f.write(f"POINTS {len(nodes)} float\n")
            for tag in sorted(nodes.keys()):
                coord = nodes[tag]
                f.write(f"{coord[0]} {coord[1]} {coord[2]}\n")

            # Count triangles
            num_tris = sum(len(tags) for t, tags in zip(surf_types, surf_tags) if t in [2, 9])

            # Write triangles
            f.write(f"\nPOLYGONS {num_tris} {num_tris * 4}\n")
            node_map = {tag: idx for idx, tag in enumerate(sorted(nodes.keys()))}

            for surf_type, s_tags, s_nodes in zip(surf_types, surf_tags, surf_nodes):
                if surf_type in [2, 9]:
                    nodes_per_tri = 3 if surf_type == 2 else 6
                    for i in range(len(s_tags)):
                        start = i * nodes_per_tri
                        n1, n2, n3 = s_nodes[start:start+3]
                        f.write(f"3 {node_map[n1]} {node_map[n2]} {node_map[n3]}\n")

            # Write cell colors based on quality
            f.write(f"\nCELL_DATA {num_tris}\n")
            f.write("COLOR_SCALARS Quality 3\n")

            for surf_type, s_tags in zip(surf_types, surf_tags):
                if surf_type in [2, 9]:
                    for tri_id in s_tags:
                        quality = tri_qualities.get(int(tri_id), 1.0)
                        color = self.get_color_for_quality(quality, threshold)
                        f.write(f"{color[0]} {color[1]} {color[2]}\n")
