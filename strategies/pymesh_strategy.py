"""
PyMesh Meshing Strategy (Supplementary)
========================================

Uses PyMesh as an alternative mesh processing and repair tool when Gmsh fails.
PyMesh excels at repairing problematic surface meshes and creating robust volumes.

Requires: pip install pymesh2 (or pymeshfix)

This is a SUPPLEMENTARY strategy - use when Gmsh and TetGen approaches fail.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.mesh_generator import BaseMeshGenerator
from core.config import Config
import gmsh
import numpy as np

# Check if pymesh is available
PYMESH_AVAILABLE = False
PYMESH_TYPE = None

try:
    import pymesh
    PYMESH_AVAILABLE = True
    PYMESH_TYPE = "pymesh2"
except ImportError:
    try:
        import pymeshfix
        PYMESH_AVAILABLE = True
        PYMESH_TYPE = "pymeshfix"
    except ImportError:
        pass


class PyMeshMeshGenerator(BaseMeshGenerator):
    """
    PyMesh-based mesh generator (supplementary strategy)

    Uses PyMesh for mesh repair and processing when Gmsh strategies fail.
    PyMesh is particularly good at:
    - Repairing self-intersections
    - Removing degenerate elements
    - Removing duplicate faces
    - Fixing manifold issues

    Installation:
        pip install pymesh2
        # OR
        pip install pymeshfix
    """

    def __init__(self, config: Optional[Config] = None):
        super().__init__(config)

        if not PYMESH_AVAILABLE:
            self.log_message("[!] PyMesh not available. Install with: pip install pymesh2")
        else:
            self.log_message(f"[OK] PyMesh available (using {PYMESH_TYPE})")

    def run_meshing_strategy(self, input_file: str, output_file: str) -> bool:
        """
        Run PyMesh meshing strategy

        Args:
            input_file: Input CAD file (must be already loaded in Gmsh)
            output_file: Output mesh file

        Returns:
            True if successful
        """
        if not PYMESH_AVAILABLE:
            self.log_message("[X] PyMesh not installed - skipping")
            return False

        self.log_message("\n" + "=" * 60)
        self.log_message("PYMESH SUPPLEMENTARY STRATEGY")
        self.log_message(f"Using PyMesh ({PYMESH_TYPE}) for mesh repair/generation")
        self.log_message("=" * 60)

        try:
            if PYMESH_TYPE == "pymesh2":
                return self._run_pymesh2_strategy(output_file)
            elif PYMESH_TYPE == "pymeshfix":
                return self._run_pymeshfix_strategy(output_file)
            else:
                return False

        except Exception as e:
            self.log_message(f"[X] PyMesh strategy failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _run_pymesh2_strategy(self, output_file: str) -> bool:
        """Run strategy using pymesh2 library"""
        import pymesh

        # Step 1: Generate surface mesh in Gmsh (2D only)
        self.log_message("Generating surface mesh with Gmsh...")
        gmsh.model.mesh.clear()
        gmsh.model.mesh.generate(2)  # Only 2D surface mesh

        # Step 2: Extract surface mesh as PyMesh format
        self.log_message("Extracting surface mesh...")
        surface_mesh = self._extract_surface_mesh_pymesh()

        if surface_mesh is None:
            self.log_message("[X] Failed to extract surface mesh")
            return False

        # Step 3: Repair surface mesh
        self.log_message("Repairing surface mesh...")
        repaired_mesh = self._repair_mesh_pymesh2(surface_mesh)

        if repaired_mesh is None:
            self.log_message("[!] Mesh repair failed, using original surface")
            repaired_mesh = surface_mesh

        # Step 4: Generate tetrahedral mesh
        self.log_message("Generating tetrahedral mesh with PyMesh...")
        tet_mesh = self._tetrahedralize_pymesh2(repaired_mesh)

        if tet_mesh is None:
            return False

        # Step 5: Convert back to Gmsh format and save
        self.log_message("Converting to Gmsh format...")
        if self._convert_pymesh_to_gmsh(tet_mesh, output_file):
            # Analyze quality
            metrics = self.analyze_current_mesh()
            if metrics:
                self.log_message("\n[OK] PyMesh meshing completed successfully")
                return True

        return False

    def _run_pymeshfix_strategy(self, output_file: str) -> bool:
        """Run strategy using pymeshfix library"""
        import pymeshfix

        # Step 1: Generate surface mesh in Gmsh (2D only)
        self.log_message("Generating surface mesh with Gmsh...")
        gmsh.model.mesh.clear()
        gmsh.model.mesh.generate(2)  # Only 2D surface mesh

        # Step 2: Extract surface mesh as arrays
        self.log_message("Extracting surface mesh...")
        vertices, faces = self._extract_surface_mesh_arrays()

        if vertices is None or faces is None:
            self.log_message("[X] Failed to extract surface mesh")
            return False

        # Step 3: Repair mesh with pymeshfix
        self.log_message("Repairing surface mesh with pymeshfix...")
        mfix = pymeshfix.MeshFix(vertices, faces)

        # Repair mesh (removes self-intersections, duplicates, etc.)
        mfix.repair(verbose=True, joincomp=True, remove_smallest_components=True)

        repaired_vertices = mfix.v
        repaired_faces = mfix.f

        self.log_message(f"[OK] Mesh repaired: {len(repaired_vertices)} vertices, {len(repaired_faces)} faces")

        # Step 4: Convert repaired surface back to Gmsh
        self.log_message("Loading repaired surface into Gmsh...")
        if not self._load_surface_to_gmsh(repaired_vertices, repaired_faces):
            return False

        # Step 5: Generate volume mesh with Gmsh
        self.log_message("Generating volume mesh with Gmsh...")
        gmsh.model.mesh.generate(3)

        # Step 6: Save mesh
        gmsh.write(output_file)
        self.log_message(f"[OK] Mesh saved to {output_file}")

        # Analyze quality
        metrics = self.analyze_current_mesh()
        if metrics:
            self.log_message("\n[OK] PyMeshFix repair + Gmsh volume meshing completed")
            return True

        return False

    def _extract_surface_mesh_pymesh(self) -> Optional[object]:
        """Extract surface mesh from Gmsh as PyMesh object"""
        import pymesh

        try:
            # Get 2D mesh elements (triangles on surfaces)
            element_types, element_tags, node_tags = gmsh.model.mesh.getElements(dim=2)

            if not element_tags:
                return None

            # Get all nodes
            node_ids, coords, _ = gmsh.model.mesh.getNodes()
            vertices = np.array(coords).reshape(-1, 3)

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
            faces_remapped = np.array([[node_map[n] for n in tri] for tri in triangles])

            # Create PyMesh mesh object
            mesh = pymesh.form_mesh(vertices, faces_remapped)

            self.log_message(f"[OK] Extracted surface: {len(vertices)} vertices, {len(faces_remapped)} faces")

            return mesh

        except Exception as e:
            self.log_message(f"Surface extraction failed: {e}")
            return None

    def _extract_surface_mesh_arrays(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract surface mesh from Gmsh as numpy arrays"""
        try:
            # Get 2D mesh elements (triangles on surfaces)
            element_types, element_tags, node_tags = gmsh.model.mesh.getElements(dim=2)

            if not element_tags:
                return None, None

            # Get all nodes
            node_ids, coords, _ = gmsh.model.mesh.getNodes()
            vertices = np.array(coords).reshape(-1, 3)

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
                return None, None

            triangles = np.vstack(all_triangles)

            # Create node ID mapping (Gmsh IDs to 0-based indices)
            node_map = {nid: idx for idx, nid in enumerate(node_ids)}
            faces = np.array([[node_map[n] for n in tri] for tri in triangles])

            self.log_message(f"[OK] Extracted surface: {len(vertices)} vertices, {len(faces)} faces")

            return vertices, faces

        except Exception as e:
            self.log_message(f"Surface extraction failed: {e}")
            return None, None

    def _repair_mesh_pymesh2(self, mesh: object) -> Optional[object]:
        """Repair mesh using pymesh2 utilities"""
        import pymesh

        try:
            original_face_count = len(mesh.faces)

            # Remove degenerate triangles
            mesh, _ = pymesh.remove_degenerated_triangles(mesh, num_iterations=5)
            self.log_message(f"  - Removed degenerate triangles")

            # Remove duplicated faces
            mesh, _ = pymesh.remove_duplicated_faces(mesh)
            self.log_message(f"  - Removed duplicate faces")

            # Resolve self-intersections (if any)
            try:
                mesh = pymesh.resolve_self_intersection(mesh)
                self.log_message(f"  - Resolved self-intersections")
            except Exception:
                self.log_message(f"  - No self-intersections to resolve")

            # Remove isolated vertices
            mesh, _ = pymesh.remove_isolated_vertices(mesh)

            final_face_count = len(mesh.faces)
            removed = original_face_count - final_face_count

            self.log_message(f"[OK] Mesh repair complete: removed {removed} problematic faces")

            return mesh

        except Exception as e:
            self.log_message(f"[!] Mesh repair failed: {e}")
            return None

    def _tetrahedralize_pymesh2(self, mesh: object) -> Optional[object]:
        """Generate tetrahedral mesh using pymesh2"""
        import pymesh

        try:
            # Tetrahedralize with quality constraints
            tet_mesh = pymesh.tetrahedralize(
                mesh,
                max_tet_volume=None,  # Auto-determined
                max_radius_edge_ratio=2.0,
                min_dihedral_angle=10.0,
                verbose=1
            )

            self.log_message(f"[OK] PyMesh generated {len(tet_mesh.voxels)} tets, {len(tet_mesh.vertices)} nodes")

            return tet_mesh

        except Exception as e:
            self.log_message(f"PyMesh tetrahedralization failed: {e}")
            return None

    def _load_surface_to_gmsh(self, vertices: np.ndarray, faces: np.ndarray) -> bool:
        """Load repaired surface mesh into Gmsh"""
        try:
            # Clear existing mesh
            gmsh.model.mesh.clear()

            # Add nodes
            for i, vertex in enumerate(vertices):
                gmsh.model.mesh.addNode(vertex[0], vertex[1], vertex[2], i + 1)

            # Add triangles
            element_type = 2  # 3-node triangle
            for i, face in enumerate(faces):
                nodes = [int(n) + 1 for n in face]  # 1-based indexing
                gmsh.model.mesh.addElement(element_type, nodes, i + 1)

            self.log_message(f"[OK] Loaded {len(vertices)} nodes, {len(faces)} triangles into Gmsh")

            return True

        except Exception as e:
            self.log_message(f"Failed to load surface to Gmsh: {e}")
            return False

    def _convert_pymesh_to_gmsh(self, tet_mesh: object, output_file: str) -> bool:
        """Convert PyMesh tetrahedral mesh to Gmsh format"""
        try:
            # Clear Gmsh mesh
            gmsh.model.mesh.clear()

            # Add nodes
            vertices = tet_mesh.vertices
            for i, vertex in enumerate(vertices):
                gmsh.model.mesh.addNode(vertex[0], vertex[1], vertex[2], i + 1)

            # Add tetrahedral elements
            voxels = tet_mesh.voxels
            element_type = 4  # 4-node tetrahedron

            for i, tet in enumerate(voxels):
                # PyMesh uses 0-based, Gmsh uses 1-based
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
        generator = PyMeshMeshGenerator()
        result = generator.generate_mesh(cad_file)

        if result.success:
            print(f"\n[OK] PyMesh meshing completed!")
            print(f"Output file: {result.output_file}")
        else:
            print(f"\n[X] PyMesh meshing failed")
            sys.exit(1)

    except Exception as e:
        print(f"\n[X] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
