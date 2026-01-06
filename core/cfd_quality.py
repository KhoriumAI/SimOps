"""
CFD Quality Analyzer - OpenFOAM checkMesh Equivalent
=====================================================

Native Python implementation of CFD-specific mesh quality metrics,
matching OpenFOAM's checkMesh output without requiring OpenFOAM.

Metrics implemented:
- Non-orthogonality: angle between face normal and cell-center vector
- CFD Skewness: distance from face center to cell-to-cell intersection
- Face Pyramids: positive volume check for cell-face pyramids
- Boundary Openness: sum of boundary face area vectors
- Cell Openness: sum of face area vectors per cell
- Aspect Ratio: max cell dimension / min cell dimension

All calculations are vectorized with NumPy for performance.

Usage:
    from core.cfd_quality import CFDQualityAnalyzer
    analyzer = CFDQualityAnalyzer()
    report = analyzer.analyze_mesh_file("mesh.msh")
    print(report.to_json())
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import json
from pathlib import Path

# CFD Quality Thresholds (OpenFOAM defaults)
NON_ORTHO_WARN = 65.0   # degrees - warning threshold
NON_ORTHO_FAIL = 70.0   # degrees - failure threshold  
SKEWNESS_WARN = 4.0     # CFD skewness warning
SKEWNESS_FAIL = 20.0    # CFD skewness failure
ASPECT_RATIO_MAX = 1000.0
OPENNESS_TOLERANCE = 1e-10  # For boundary/cell openness checks


@dataclass
class CFDQualityReport:
    """Structured CFD quality report matching checkMesh output format"""
    
    # Mesh statistics
    points: int = 0
    faces: int = 0
    internal_faces: int = 0
    cells: int = 0
    boundary_patches: int = 0
    
    # Geometry checks
    non_orthogonality_max: float = 0.0
    non_orthogonality_avg: float = 0.0
    non_orthogonality_ok: bool = True
    
    skewness_max: float = 0.0
    skewness_ok: bool = True
    
    aspect_ratio_max: float = 1.0
    aspect_ratio_ok: bool = True
    
    face_pyramids_ok: bool = True
    face_pyramids_failed: int = 0
    
    boundary_openness: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    boundary_openness_ok: bool = True
    
    cell_openness_max: float = 0.0
    cell_openness_ok: bool = True
    
    # Overall
    cfd_ready: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary matching OpenFOAM checkMesh JSON format"""
        # Helper to convert numpy types to Python native types for JSON
        def to_native(val):
            if isinstance(val, (np.bool_, np.integer)):
                return bool(val) if isinstance(val, np.bool_) else int(val)
            elif isinstance(val, np.floating):
                return float(val)
            return val
        
        return {
            "mesh_stats": {
                "points": int(self.points),
                "faces": int(self.faces),
                "internal_faces": int(self.internal_faces),
                "cells": int(self.cells),
                "boundary_patches": int(self.boundary_patches)
            },
            "geometry_checks": {
                "non_orthogonality": {
                    "max": float(self.non_orthogonality_max),
                    "avg": float(self.non_orthogonality_avg),
                    "ok": bool(self.non_orthogonality_ok)
                },
                "skewness": {
                    "max": float(self.skewness_max),
                    "ok": bool(self.skewness_ok)
                },
                "aspect_ratio": {
                    "max": float(self.aspect_ratio_max),
                    "ok": bool(self.aspect_ratio_ok)
                },
                "face_pyramids": {
                    "ok": bool(self.face_pyramids_ok),
                    "failed_count": int(self.face_pyramids_failed)
                },
                "boundary_openness": {
                    "value": [float(v) for v in self.boundary_openness],
                    "ok": bool(self.boundary_openness_ok)
                },
                "cell_openness": {
                    "max": float(self.cell_openness_max),
                    "ok": bool(self.cell_openness_ok)
                }
            },
            "cfd_ready": bool(self.cfd_ready),
            "warnings": list(self.warnings),
            "errors": list(self.errors)
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)
    
    def print_report(self):
        """Print human-readable report matching checkMesh format"""
        print("\n" + "=" * 70)
        print("CFD MESH QUALITY CHECK (OpenFOAM checkMesh Equivalent)")
        print("=" * 70)
        
        print(f"\nMesh stats:")
        print(f"    points:           {self.points}")
        print(f"    faces:            {self.faces}")
        print(f"    internal faces:   {self.internal_faces}")
        print(f"    cells:            {self.cells}")
        
        print(f"\nChecking geometry...")
        
        status = "OK" if self.non_orthogonality_ok else "FAILED"
        print(f"    Mesh non-orthogonality Max: {self.non_orthogonality_max:.6g} "
              f"average: {self.non_orthogonality_avg:.6g}")
        print(f"    Non-orthogonality check {status}.")
        
        status = "OK" if self.face_pyramids_ok else f"FAILED ({self.face_pyramids_failed} cells)"
        print(f"    Face pyramids {status}.")
        
        status = "OK" if self.skewness_ok else "FAILED"
        print(f"    Max skewness = {self.skewness_max:.6g} {status}.")
        
        status = "OK" if self.aspect_ratio_ok else "FAILED"
        print(f"    Max aspect ratio = {self.aspect_ratio_max:.6g} {status}.")
        
        bo = self.boundary_openness
        status = "OK" if self.boundary_openness_ok else "FAILED"
        print(f"    Boundary openness ({bo[0]:.6e} {bo[1]:.6e} {bo[2]:.6e}) {status}.")
        
        status = "OK" if self.cell_openness_ok else "FAILED"
        print(f"    Max cell openness = {self.cell_openness_max:.6e} {status}.")
        
        if self.warnings:
            print(f"\nWarnings:")
            for w in self.warnings:
                print(f"    [!] {w}")
        
        if self.errors:
            print(f"\nErrors:")
            for e in self.errors:
                print(f"    [X] {e}")
        
        print("\n" + "-" * 70)
        if self.cfd_ready:
            print("[OK] Mesh CFD quality check PASSED")
        else:
            print("[X] Mesh CFD quality check FAILED")
        print("-" * 70)


class CFDQualityAnalyzer:
    """
    OpenFOAM checkMesh-equivalent quality analyzer.
    
    All calculations are vectorized with NumPy for performance.
    Works with MSH 4.x format files loaded via Gmsh.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._nodes: Optional[np.ndarray] = None  # (N, 3) node coordinates
        self._cells: Optional[np.ndarray] = None  # (M, 4) for tets, connectivity
        self._faces: Optional[List] = None  # Face definitions
        self._cell_centers: Optional[np.ndarray] = None
        self._face_centers: Optional[np.ndarray] = None
        self._face_normals: Optional[np.ndarray] = None
        self._face_areas: Optional[np.ndarray] = None
        
    def log(self, msg: str):
        if self.verbose:
            print(msg)
    
    def analyze_mesh_file(self, mesh_path: str) -> CFDQualityReport:
        """
        Analyze a mesh file and return CFD quality report.
        
        Args:
            mesh_path: Path to .msh file (MSH 4.x format)
            
        Returns:
            CFDQualityReport with all quality metrics
        """
        import gmsh
        
        report = CFDQualityReport()
        
        try:
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.open(mesh_path)
            
            # Extract mesh data
            self._extract_mesh_data(gmsh)
            
            if self._nodes is None or len(self._nodes) == 0:
                report.errors.append("No nodes found in mesh")
                report.cfd_ready = False
                return report
            
            if self._cells is None or len(self._cells) == 0:
                report.errors.append("No volume elements found in mesh")
                report.cfd_ready = False
                return report
            
            # Populate mesh stats - count actual nodes used, not padded array size
            report.points = len(np.unique(self._cells.flatten()))
            report.cells = len(self._cells)
            report.faces = len(self._faces) if (self._faces is not None and len(self._faces) > 0) else 0
            
            # Compute cell centers (needed for most checks)
            self._compute_cell_centers()
            
            # Run all quality checks
            self._check_non_orthogonality(report)
            self._check_face_pyramids(report)
            self._check_skewness(report)
            self._check_aspect_ratio(report)
            self._check_boundary_openness(report)
            self._check_cell_openness(report)
            
            # Determine overall CFD readiness
            report.cfd_ready = (
                report.non_orthogonality_ok and
                report.face_pyramids_ok and
                report.skewness_ok and
                report.aspect_ratio_ok and
                report.boundary_openness_ok and
                report.cell_openness_ok
            )
            
        except Exception as e:
            report.errors.append(f"Analysis failed: {str(e)}")
            report.cfd_ready = False
            
        finally:
            try:
                gmsh.finalize()
            except:
                pass
        
        return report
    
    def analyze_current_mesh(self) -> CFDQualityReport:
        """
        Analyze the currently loaded Gmsh mesh (assumes gmsh is already initialized).
        
        Returns:
            CFDQualityReport with all quality metrics
        """
        import gmsh
        
        report = CFDQualityReport()
        
        try:
            # Extract mesh data from current Gmsh state
            self._extract_mesh_data(gmsh)
            
            if self._nodes is None or len(self._nodes) == 0:
                report.errors.append("No nodes found in mesh")
                report.cfd_ready = False
                return report
            
            if self._cells is None or len(self._cells) == 0:
                report.errors.append("No volume elements found in mesh")
                report.cfd_ready = False
                return report
            
            # Populate mesh stats - count actual nodes used, not padded array size
            report.points = len(np.unique(self._cells.flatten()))
            report.cells = len(self._cells)
            report.faces = len(self._faces) if (self._faces is not None and len(self._faces) > 0) else 0
            
            # Compute cell centers
            self._compute_cell_centers()
            
            # Run all quality checks
            self._check_non_orthogonality(report)
            self._check_face_pyramids(report)
            self._check_skewness(report)
            self._check_aspect_ratio(report)
            self._check_boundary_openness(report)
            self._check_cell_openness(report)
            
            # Determine overall CFD readiness
            report.cfd_ready = (
                report.non_orthogonality_ok and
                report.face_pyramids_ok and
                report.skewness_ok and
                report.aspect_ratio_ok and
                report.boundary_openness_ok and
                report.cell_openness_ok
            )
            
        except Exception as e:
            report.errors.append(f"Analysis failed: {str(e)}")
            report.cfd_ready = False
        
        return report
    
    def _extract_mesh_data(self, gmsh):
        """Extract nodes, cells, and faces from Gmsh mesh"""
        
        # Get all nodes - batch fetch for performance
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        if len(node_tags) == 0:
            return
        
        # Build node coordinate array (indexed by node tag)
        max_tag = int(max(node_tags))
        self._nodes = np.zeros((max_tag + 1, 3))
        coords = np.array(coords).reshape(-1, 3)
        for i, tag in enumerate(node_tags):
            self._nodes[int(tag)] = coords[i]
        
        # Get tetrahedral elements (type 4 = linear tet, type 11 = quadratic tet)
        cells_list = []
        for tet_type in [4, 11]:
            try:
                tags, nodes = gmsh.model.mesh.getElementsByType(tet_type)
                if len(tags) > 0:
                    nodes_per_elem = 4 if tet_type == 4 else 10
                    nodes = np.array(nodes).reshape(-1, nodes_per_elem)
                    # Use only corner nodes (first 4)
                    cells_list.append(nodes[:, :4].astype(int))
            except:
                pass
        
        if cells_list:
            self._cells = np.vstack(cells_list)
        
        # Extract face data for internal faces
        # For tets, faces are triangles formed by cell vertices
        self._extract_faces_from_cells()
    
    def _extract_faces_from_cells(self):
        """Extract unique internal faces from tetrahedral cells"""
        if self._cells is None:
            return
        
        # Each tet has 4 triangular faces
        # Face vertex indices within tet: (0,1,2), (0,1,3), (0,2,3), (1,2,3)
        face_patterns = np.array([
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3]
        ])
        
        # Build all faces with cell ownership
        n_cells = len(self._cells)
        all_faces = []  # (sorted_nodes, original_nodes, cell_id)
        
        for cell_id, cell in enumerate(self._cells):
            for pattern in face_patterns:
                face_nodes = cell[pattern]
                sorted_nodes = tuple(sorted(face_nodes))
                all_faces.append((sorted_nodes, face_nodes, cell_id))
        
        # Find internal faces (shared by 2 cells) and boundary faces (owned by 1 cell)
        from collections import defaultdict
        face_dict = defaultdict(list)
        for sorted_nodes, original_nodes, cell_id in all_faces:
            face_dict[sorted_nodes].append((original_nodes, cell_id))
        
        self._faces = []
        self._internal_face_owners = []  # (owner_cell, neighbor_cell)
        self._boundary_faces = []
        
        for sorted_nodes, owners in face_dict.items():
            if len(owners) == 2:
                # Internal face
                self._faces.append(np.array(owners[0][0]))
                self._internal_face_owners.append((owners[0][1], owners[1][1]))
            elif len(owners) == 1:
                # Boundary face
                self._boundary_faces.append(np.array(owners[0][0]))
        
        self._faces = np.array(self._faces) if self._faces else np.array([]).reshape(0, 3)
        self._boundary_faces = np.array(self._boundary_faces) if self._boundary_faces else np.array([]).reshape(0, 3)
    
    def _compute_cell_centers(self):
        """Compute cell centers as average of vertex positions (vectorized)"""
        if self._cells is None:
            return
        
        # For each cell, average its 4 vertex positions
        # cells shape: (n_cells, 4), values are node indices
        cell_coords = self._nodes[self._cells]  # (n_cells, 4, 3)
        self._cell_centers = np.mean(cell_coords, axis=1)  # (n_cells, 3)
    
    def _compute_face_geometry(self, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute face centers, normals, and areas (vectorized).
        
        Args:
            faces: (n_faces, 3) array of node indices for triangle faces
            
        Returns:
            centers: (n_faces, 3) face center coordinates
            normals: (n_faces, 3) unit face normals
            areas: (n_faces,) face areas
        """
        if len(faces) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Get vertex coordinates for all faces
        v0 = self._nodes[faces[:, 0]]  # (n_faces, 3)
        v1 = self._nodes[faces[:, 1]]
        v2 = self._nodes[faces[:, 2]]
        
        # Face centers
        centers = (v0 + v1 + v2) / 3.0
        
        # Face normals via cross product
        e1 = v1 - v0
        e2 = v2 - v0
        normals = np.cross(e1, e2)
        
        # Areas = 0.5 * |normal|
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        areas = (norms.flatten() / 2.0)
        
        # Normalize (avoid division by zero)
        norms = np.maximum(norms, 1e-15)
        unit_normals = normals / norms
        
        return centers, unit_normals, areas
    
    def _check_non_orthogonality(self, report: CFDQualityReport):
        """
        Calculate non-orthogonality for internal faces.
        
        Non-orthogonality = angle between face normal and cell-center-to-cell-center vector.
        Target: < 70° for CFD stability.
        
        This is vectorized for performance.
        """
        if len(self._faces) == 0 or not self._internal_face_owners:
            report.non_orthogonality_max = 0.0
            report.non_orthogonality_avg = 0.0
            report.non_orthogonality_ok = True
            return
        
        # Compute face geometry
        face_centers, face_normals, _ = self._compute_face_geometry(self._faces)
        
        # Get owner and neighbor cell centers
        owners = np.array([o for o, n in self._internal_face_owners])
        neighbors = np.array([n for o, n in self._internal_face_owners])
        
        owner_centers = self._cell_centers[owners]      # (n_internal, 3)
        neighbor_centers = self._cell_centers[neighbors]
        
        # Cell-to-cell vectors (d)
        d = neighbor_centers - owner_centers
        d_norms = np.linalg.norm(d, axis=1, keepdims=True)
        d_norms = np.maximum(d_norms, 1e-15)
        d_unit = d / d_norms
        
        # Non-orthogonality = angle between face normal and d
        # cos(angle) = dot(n, d_unit)
        cos_angles = np.abs(np.sum(face_normals * d_unit, axis=1))
        cos_angles = np.clip(cos_angles, -1.0, 1.0)
        
        # Angle in degrees (complement for non-orthogonality measure)
        angles_rad = np.arccos(cos_angles)
        non_ortho_degrees = np.degrees(angles_rad)
        
        report.non_orthogonality_max = float(np.max(non_ortho_degrees))
        report.non_orthogonality_avg = float(np.mean(non_ortho_degrees))
        report.internal_faces = len(self._faces)
        
        if report.non_orthogonality_max > NON_ORTHO_FAIL:
            report.non_orthogonality_ok = False
            report.errors.append(f"Non-orthogonality {report.non_orthogonality_max:.1f}° exceeds limit {NON_ORTHO_FAIL}°")
        elif report.non_orthogonality_max > NON_ORTHO_WARN:
            report.warnings.append(f"Non-orthogonality {report.non_orthogonality_max:.1f}° exceeds warning threshold {NON_ORTHO_WARN}°")
    
    def _check_face_pyramids(self, report: CFDQualityReport):
        """
        Check that all face pyramids have positive volume.
        
        For each cell face, the pyramid from face to cell center must have positive volume
        (face normal should point outward from cell center).
        
        This detects inverted or negative-volume cells.
        """
        if self._cells is None or len(self._cells) == 0:
            report.face_pyramids_ok = True
            return
        
        failed_count = 0
        face_patterns = np.array([
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3]
        ])
        
        for cell_id, cell in enumerate(self._cells):
            cell_center = self._cell_centers[cell_id]
            
            for pattern in face_patterns:
                v0 = self._nodes[cell[pattern[0]]]
                v1 = self._nodes[cell[pattern[1]]]
                v2 = self._nodes[cell[pattern[2]]]
                
                # Face center
                face_center = (v0 + v1 + v2) / 3.0
                
                # Face normal (cross product)
                e1 = v1 - v0
                e2 = v2 - v0
                normal = np.cross(e1, e2)
                
                # Vector from face center to cell center
                to_center = cell_center - face_center
                
                # Pyramid volume sign: positive if normal points away from cell center
                # (dot product should be negative for outward normal convention)
                vol_sign = np.dot(normal, to_center)
                
                # OpenFOAM convention: face normal points from owner to neighbor
                # For consistency, we check if pyramid volume is non-zero
                # A truly inverted cell would have inconsistent orientations
                if abs(vol_sign) < 1e-20:
                    failed_count += 1
        
        # For tet meshes, also check overall cell volume
        for cell_id, cell in enumerate(self._cells):
            v0 = self._nodes[cell[0]]
            v1 = self._nodes[cell[1]]
            v2 = self._nodes[cell[2]]
            v3 = self._nodes[cell[3]]
            
            # Tet volume = (1/6) * |det([v1-v0, v2-v0, v3-v0])|
            e1 = v1 - v0
            e2 = v2 - v0
            e3 = v3 - v0
            vol = np.dot(np.cross(e1, e2), e3) / 6.0
            
            if vol <= 0:
                failed_count += 1
        
        report.face_pyramids_failed = failed_count
        report.face_pyramids_ok = (failed_count == 0)
        
        if failed_count > 0:
            report.errors.append(f"Face pyramids check failed: {failed_count} issues detected")
    
    def _check_skewness(self, report: CFDQualityReport):
        """
        Calculate CFD skewness for internal faces.
        
        CFD skewness = distance from face center to the intersection point
        of the line connecting cell centers with the face plane,
        divided by the distance between cell centers.
        
        This is different from FEA equiangular skewness!
        Target: < 4 for CFD stability
        """
        if len(self._faces) == 0 or not self._internal_face_owners:
            report.skewness_max = 0.0
            report.skewness_ok = True
            return
        
        # Compute face geometry
        face_centers, face_normals, _ = self._compute_face_geometry(self._faces)
        
        # Get owner and neighbor cell centers
        owners = np.array([o for o, n in self._internal_face_owners])
        neighbors = np.array([n for o, n in self._internal_face_owners])
        
        owner_centers = self._cell_centers[owners]
        neighbor_centers = self._cell_centers[neighbors]
        
        # Cell-to-cell vector
        d = neighbor_centers - owner_centers
        d_lengths = np.linalg.norm(d, axis=1)
        d_lengths = np.maximum(d_lengths, 1e-15)
        
        # Find intersection of line (owner_center + t*d) with face plane
        # Plane equation: dot(n, x - face_center) = 0
        # Substituting: dot(n, owner_center + t*d - face_center) = 0
        # t = dot(n, face_center - owner_center) / dot(n, d)
        
        numerator = np.sum(face_normals * (face_centers - owner_centers), axis=1)
        denominator = np.sum(face_normals * d, axis=1)
        denominator = np.where(np.abs(denominator) < 1e-15, 1e-15, denominator)
        
        t = numerator / denominator
        
        # Intersection point
        intersection = owner_centers + t[:, np.newaxis] * d
        
        # Skewness = distance from face center to intersection / d_length
        skew_vectors = face_centers - intersection
        skew_distances = np.linalg.norm(skew_vectors, axis=1)
        skewness = skew_distances / d_lengths
        
        report.skewness_max = float(np.max(skewness))
        report.skewness_ok = (report.skewness_max < SKEWNESS_FAIL)
        
        if report.skewness_max > SKEWNESS_FAIL:
            report.errors.append(f"Skewness {report.skewness_max:.2f} exceeds limit {SKEWNESS_FAIL}")
        elif report.skewness_max > SKEWNESS_WARN:
            report.warnings.append(f"Skewness {report.skewness_max:.2f} exceeds warning threshold {SKEWNESS_WARN}")
    
    def _check_aspect_ratio(self, report: CFDQualityReport):
        """
        Calculate cell aspect ratio.
        
        Aspect ratio = longest edge / shortest edge for each cell.
        Target: < 1000 for numerical stability
        """
        if self._cells is None or len(self._cells) == 0:
            report.aspect_ratio_max = 1.0
            report.aspect_ratio_ok = True
            return
        
        # Edge pairs for tetrahedron: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        edge_pairs = np.array([[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]])
        
        max_ar = 1.0
        
        for cell in self._cells:
            coords = self._nodes[cell]  # (4, 3)
            edges = []
            for i, j in edge_pairs:
                edge_len = np.linalg.norm(coords[i] - coords[j])
                edges.append(edge_len)
            
            min_edge = min(edges)
            max_edge = max(edges)
            
            if min_edge > 1e-15:
                ar = max_edge / min_edge
                max_ar = max(max_ar, ar)
        
        report.aspect_ratio_max = float(max_ar)
        report.aspect_ratio_ok = (max_ar < ASPECT_RATIO_MAX)
        
        if not report.aspect_ratio_ok:
            report.errors.append(f"Aspect ratio {max_ar:.1f} exceeds limit {ASPECT_RATIO_MAX}")
    
    def _check_boundary_openness(self, report: CFDQualityReport):
        """
        Check boundary openness (mesh closure).
        
        Sum of all boundary face area vectors should equal zero
        for a properly closed mesh.
        """
        if self._boundary_faces is None or len(self._boundary_faces) == 0:
            report.boundary_openness = (0.0, 0.0, 0.0)
            report.boundary_openness_ok = True
            return
        
        # Compute boundary face geometry
        _, normals, areas = self._compute_face_geometry(self._boundary_faces)
        
        # Sum of area vectors (area * normal)
        area_vectors = normals * areas[:, np.newaxis]
        total = np.sum(area_vectors, axis=0)
        
        report.boundary_openness = tuple(total)
        
        # Check if within tolerance (relative to total boundary area)
        total_area = np.sum(areas)
        relative_openness = np.linalg.norm(total) / max(total_area, 1e-15)
        
        report.boundary_openness_ok = (relative_openness < OPENNESS_TOLERANCE)
        
        if not report.boundary_openness_ok:
            report.errors.append(f"Boundary openness check failed: mesh may not be watertight")
    
    def _check_cell_openness(self, report: CFDQualityReport):
        """
        Check cell openness.
        
        Sum of face area vectors for each cell should equal zero.
        This validates proper cell closure.
        """
        if self._cells is None or len(self._cells) == 0:
            report.cell_openness_max = 0.0
            report.cell_openness_ok = True
            return
        
        face_patterns = np.array([
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3]
        ])
        
        max_openness = 0.0
        
        for cell in self._cells:
            cell_area_sum = np.zeros(3)
            cell_total_area = 0.0
            
            for pattern in face_patterns:
                v0 = self._nodes[cell[pattern[0]]]
                v1 = self._nodes[cell[pattern[1]]]
                v2 = self._nodes[cell[pattern[2]]]
                
                e1 = v1 - v0
                e2 = v2 - v0
                area_vec = np.cross(e1, e2) / 2.0
                
                cell_area_sum += area_vec
                cell_total_area += np.linalg.norm(area_vec)
            
            if cell_total_area > 1e-15:
                relative_openness = np.linalg.norm(cell_area_sum) / cell_total_area
                max_openness = max(max_openness, relative_openness)
        
        report.cell_openness_max = float(max_openness)
        report.cell_openness_ok = (max_openness < OPENNESS_TOLERANCE)
        
        if not report.cell_openness_ok:
            report.warnings.append(f"Cell openness check: max relative openness = {max_openness:.2e}")


def analyze_mesh_cfd(mesh_path: str, verbose: bool = True) -> CFDQualityReport:
    """
    Convenience function to analyze a mesh file for CFD quality.
    
    Args:
        mesh_path: Path to mesh file (.msh format)
        verbose: Print progress messages
        
    Returns:
        CFDQualityReport with all metrics
    """
    analyzer = CFDQualityAnalyzer(verbose=verbose)
    return analyzer.analyze_mesh_file(mesh_path)


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description="CFD Mesh Quality Analyzer (OpenFOAM checkMesh equivalent)"
    )
    parser.add_argument("mesh_file", help="Path to mesh file (.msh)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    
    args = parser.parse_args()
    
    report = analyze_mesh_cfd(args.mesh_file, verbose=not args.quiet)
    
    if args.json:
        print(report.to_json())
    else:
        report.print_report()
    
    sys.exit(0 if report.cfd_ready else 1)
