"""
TetGen Meshing Strategy (Supplementary)
========================================

Uses TetGen as an alternative tetrahedral mesher when Gmsh fails.
TetGen is known for robust meshing of complex geometries.

Requires: pip install tetgen

This is a SUPPLEMENTARY strategy - use when Gmsh approaches fail.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.mesh_generator import BaseMeshGenerator
from core.config import Config
import gmsh

# Check if tetgen is available
TETGEN_AVAILABLE = False
try:
    import tetgen
    import pyvista as pv
    import numpy as np
    TETGEN_AVAILABLE = True
except ImportError:
    pass


class TetGenMeshGenerator(BaseMeshGenerator):
    """
    TetGen-based mesh generator (supplementary strategy)

    Uses TetGen when Gmsh strategies fail. TetGen is particularly
    good at handling complex/problematic geometries.

    Installation:
        pip install tetgen pyvista
    """

    def __init__(self, config: Optional[Config] = None):
        super().__init__(config)

        if not TETGEN_AVAILABLE:
            self.log_message("[!] TetGen not available. Install with: pip install tetgen pyvista")

    def run_meshing_strategy(self, input_file: str, output_file: str) -> bool:
        """
        Run TetGen meshing strategy

        Args:
            input_file: Input CAD file (must be already loaded in Gmsh)
            output_file: Output mesh file

        Returns:
            True if successful
        """
        if not TETGEN_AVAILABLE:
            self.log_message("[X] TetGen not installed - skipping")
            return False

        self.log_message("\n" + "=" * 60)
        self.log_message("TETGEN SUPPLEMENTARY STRATEGY")
        self.log_message("Using TetGen as fallback mesher")
        self.log_message("=" * 60)

        try:
            # Step 1: Generate surface mesh in Gmsh (2D only)
            self.log_message("Generating surface mesh with Gmsh...")
            gmsh.model.mesh.clear()
            gmsh.model.mesh.generate(2)  # Only 2D surface mesh

            # Step 2: Extract surface mesh as PyVista PolyData
            self.log_message("Extracting surface mesh...")
            surface_mesh = self._extract_surface_mesh_pyvista()

            if surface_mesh is None:
                self.log_message("[X] Failed to extract surface mesh")
                return False

            # Step 3: Use TetGen to mesh the volume
            self.log_message("Running TetGen volume mesher...")
            tet_mesh = self._tetgen_mesh(surface_mesh)

            if tet_mesh is None:
                return False

            # Step 4: Convert back to Gmsh format and save
            self.log_message("Converting to Gmsh format...")
            if self._convert_tetgen_to_gmsh(tet_mesh, output_file):
                # Analyze quality
                metrics = self.analyze_current_mesh()
                if metrics:
                    self.log_message("\n[OK] TetGen meshing completed successfully")
                    return True

            return False

        except Exception as e:
            self.log_message(f"[X] TetGen strategy failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _extract_surface_mesh_pyvista(self) -> Optional[object]:
        """Extract surface mesh from Gmsh as PyVista PolyData"""
        try:
            # Get 2D mesh elements (triangles on surfaces)
            element_types, element_tags, node_tags = gmsh.model.mesh.getElements(dim=2)

            if not element_tags:
                return None

            # Get all nodes
            node_ids, coords, _ = gmsh.model.mesh.getNodes()
            points = np.array(coords).reshape(-1, 3)

            # Build connectivity for triangles
            all_triangles = []
            for elem_type, tags, nodes in zip(element_types, element_tags, node_tags):
                if elem_type == 2:  # 3-node triangle
                    triangles = np.array(nodes).reshape(-1, 3)
                    all_triangles.append(triangles)
                elif elem_type == 9:  # 6-node triangle (use only corner nodes)
                    triangles = np.array(nodes).reshape(-1, 6)[:, :3]
                    all_triangles.append(triangles)

            if not all_triangles:
                return None

            triangles = np.vstack(all_triangles)

            # Create node ID mapping (Gmsh IDs to 0-based indices)
            node_map = {nid: idx for idx, nid in enumerate(node_ids)}
            triangles_remapped = np.array([[node_map[n] for n in tri] for tri in triangles])

            # Create PyVista PolyData
            faces = np.hstack([[3, *tri] for tri in triangles_remapped])
            surface = pv.PolyData(points, faces)

            return surface

        except Exception as e:
            self.log_message(f"Surface extraction failed: {e}")
            return None

    def _tetgen_mesh(self, surface_mesh: object) -> Optional[object]:
        """Run TetGen on surface mesh"""
        try:
            tet = tetgen.TetGen(surface_mesh)

            # TetGen options
            order = self.config.mesh_params.element_order

            # Quality settings
            min_dihedral = 10.0  # Minimum dihedral angle
            max_ratio = 2.0      # Maximum radius-edge ratio

            self.log_message(f"TetGen settings: order={order}, min_angle={min_dihedral}, max_ratio={max_ratio}")

            # Run tetrahedralization
            tet.tetrahedralize(
                order=1,  # Start with linear, can upgrade later
                mindihedral=min_dihedral,
                minratio=max_ratio,
                quality=True,
                verbose=1
            )

            grid = tet.grid

            self.log_message(f"[OK] TetGen generated {grid.n_cells} tets, {grid.n_points} nodes")

            return grid

        except Exception as e:
            self.log_message(f"TetGen meshing failed: {e}")
            return None

    def _convert_tetgen_to_gmsh(self, tet_mesh: object, output_file: str) -> bool:
        """Convert TetGen mesh back to Gmsh and save"""
        try:
            # Clear Gmsh mesh
            gmsh.model.mesh.clear()

            # Add nodes
            points = tet_mesh.points
            for i, point in enumerate(points):
                gmsh.model.mesh.addNode(point[0], point[1], point[2], i + 1)

            # Add tetrahedral elements
            cells = tet_mesh.cells_dict.get('tetra', None)
            if cells is None:
                self.log_message("[X] No tetrahedral elements in TetGen output")
                return False

            element_type = 4  # 4-node tetrahedron
            for i, tet in enumerate(cells):
                # TetGen uses 0-based, Gmsh uses 1-based
                nodes = [int(n) + 1 for n in tet]
                gmsh.model.mesh.addElement(element_type, nodes, i + 1)

            # Save mesh
            gmsh.write(output_file)
            self.log_message(f"[OK] Mesh saved to {output_file}")

            return True

        except Exception as e:
            self.log_message(f"Conversion to Gmsh failed: {e}")
            return False


def main():
    """Command-line interface"""
    if len(sys.argv) > 1:
        cad_file = sys.argv[1]
    else:
        cad_file = input("Enter CAD file path: ").strip()

    try:
        generator = TetGenMeshGenerator()
        result = generator.generate_mesh(cad_file)

        if result.success:
            print(f"\n[OK] TetGen meshing completed!")
            print(f"Output file: {result.output_file}")
        else:
            print(f"\n[X] TetGen meshing failed")
            sys.exit(1)

    except Exception as e:
        print(f"\n[X] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
