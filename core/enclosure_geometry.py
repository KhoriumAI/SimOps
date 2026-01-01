"""
Enclosure Geometry Generator for External Flow CFD
===================================================

Creates parametric wind tunnel domains around CAD parts for external flow meshing.
Supports box, cylindrical, and spherical enclosures with named boundary regions.

Usage:
    from core.enclosure_geometry import EnclosureGenerator
    
    gen = EnclosureGenerator()
    result = gen.create_box_enclosure('part.stl', multiplier=5.0, flow_direction='+X')
    # result contains enclosure STL path and boundary patch definitions
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class BoundaryPatch:
    """Definition of a boundary patch for CFD."""
    name: str
    patch_type: str  # 'inlet', 'outlet', 'wall', 'symmetry'
    face_indices: List[int]  # Triangle indices in the enclosure mesh
    

@dataclass  
class EnclosureResult:
    """Result of enclosure generation."""
    enclosure_stl_path: str
    part_stl_path: str  # Original part (may be copied)
    boundary_patches: Dict[str, BoundaryPatch]
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    enclosure_type: str


class EnclosureGenerator:
    """
    Generates wind tunnel enclosures around CAD parts.
    
    Enclosures are sized as multiples of the part's bounding box,
    which is standard practice in aerospace CFD.
    """
    
    def __init__(self, output_dir: str = None):
        """
        Initialize enclosure generator.
        
        Args:
            output_dir: Directory for generated STL files. 
                       Defaults to temp_geometry in project root.
        """
        if output_dir is None:
            project_root = Path(__file__).parent.parent
            self.output_dir = project_root / "temp_geometry"
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_stl_bounds(self, stl_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get bounding box of STL file.
        
        Returns:
            Tuple of (min_bounds, max_bounds) as numpy arrays
        """
        try:
            import trimesh
            mesh = trimesh.load(stl_path)
            return mesh.bounds[0], mesh.bounds[1]
        except Exception as e:
            print(f"[EnclosureGenerator] WARNING: trimesh failed: {e}")
            # Fallback: parse STL manually for bounds
            return self._parse_stl_bounds(stl_path)
    
    def _parse_stl_bounds(self, stl_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback STL bounds parser for binary STL."""
        try:
            with open(stl_path, 'rb') as f:
                # Skip 80-byte header
                f.read(80)
                # Read triangle count
                num_tris = np.frombuffer(f.read(4), dtype=np.uint32)[0]
                
                mins = np.array([np.inf, np.inf, np.inf])
                maxs = np.array([-np.inf, -np.inf, -np.inf])
                
                for _ in range(num_tris):
                    # Skip normal (12 bytes)
                    f.read(12)
                    # Read 3 vertices (36 bytes)
                    verts = np.frombuffer(f.read(36), dtype=np.float32).reshape(3, 3)
                    mins = np.minimum(mins, verts.min(axis=0))
                    maxs = np.maximum(maxs, verts.max(axis=0))
                    # Skip attribute (2 bytes)
                    f.read(2)
                
                return mins, maxs
        except Exception as e:
            print(f"[EnclosureGenerator] ERROR: Failed to parse STL: {e}")
            return np.array([-100, -100, -100]), np.array([100, 100, 100])
    
    def _create_box_mesh(self, min_pt: np.ndarray, max_pt: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Create a box mesh with face groups for boundary patches.
        
        Returns:
            vertices, faces, face_groups dict
        """
        # 8 vertices of the box
        vertices = np.array([
            [min_pt[0], min_pt[1], min_pt[2]],  # 0: ---
            [max_pt[0], min_pt[1], min_pt[2]],  # 1: +--
            [max_pt[0], max_pt[1], min_pt[2]],  # 2: ++-
            [min_pt[0], max_pt[1], min_pt[2]],  # 3: -+-
            [min_pt[0], min_pt[1], max_pt[2]],  # 4: --+
            [max_pt[0], min_pt[1], max_pt[2]],  # 5: +-+
            [max_pt[0], max_pt[1], max_pt[2]],  # 6: +++
            [min_pt[0], max_pt[1], max_pt[2]],  # 7: -++ 
        ], dtype=np.float32)
        
        # 12 triangles (2 per face), with normals pointing INWARD for external flow
        # (we mesh the air inside the box, not the box itself)
        faces = np.array([
            # -X face (left wall): normal points +X (inward)
            [0, 3, 7], [0, 7, 4],
            # +X face (right wall): normal points -X (inward)  
            [1, 5, 6], [1, 6, 2],
            # -Y face (bottom wall): normal points +Y (inward)
            [0, 4, 5], [0, 5, 1],
            # +Y face (top wall): normal points -Y (inward)
            [3, 2, 6], [3, 6, 7],
            # -Z face (back wall): normal points +Z (inward)
            [0, 1, 2], [0, 2, 3],
            # +Z face (front wall): normal points -Z (inward)
            [4, 7, 6], [4, 6, 5],
        ], dtype=np.int32)
        
        # Face groups: indices into faces array
        face_groups = {
            'left':   [0, 1],    # -X
            'right':  [2, 3],    # +X
            'bottom': [4, 5],    # -Y
            'top':    [6, 7],    # +Y
            'back':   [8, 9],    # -Z
            'front':  [10, 11],  # +Z
        }
        
        return vertices, faces, face_groups
    
    def _write_stl(self, vertices: np.ndarray, faces: np.ndarray, filepath: str):
        """Write binary STL file."""
        try:
            import trimesh
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            mesh.export(filepath)
        except ImportError:
            # Fallback: write binary STL manually
            with open(filepath, 'wb') as f:
                # 80-byte header
                f.write(b'\x00' * 80)
                # Triangle count
                f.write(np.array([len(faces)], dtype=np.uint32).tobytes())
                
                for face in faces:
                    v0, v1, v2 = vertices[face]
                    # Compute normal
                    normal = np.cross(v1 - v0, v2 - v0)
                    norm = np.linalg.norm(normal)
                    if norm > 0:
                        normal /= norm
                    # Write normal + 3 vertices + attribute
                    f.write(normal.astype(np.float32).tobytes())
                    f.write(v0.astype(np.float32).tobytes())
                    f.write(v1.astype(np.float32).tobytes())
                    f.write(v2.astype(np.float32).tobytes())
                    f.write(np.array([0], dtype=np.uint16).tobytes())
    
    def create_box_enclosure(self, stl_path: str, 
                             multiplier: float = 5.0,
                             flow_direction: str = '+X',
                             asymmetric_factors: Dict[str, float] = None) -> EnclosureResult:
        """
        Creates rectangular wind tunnel domain around a part.
        
        The enclosure is sized as a multiple of the part's bounding box.
        Standard CFD practice: 5x for general, 10x+ for wake-sensitive studies.
        
        Args:
            stl_path: Path to part STL file
            multiplier: Size multiplier for the enclosure (e.g., 5.0 = 5x larger)
            flow_direction: Main flow axis ('+X', '-X', '+Y', '-Y', '+Z', '-Z')
            asymmetric_factors: Optional dict to make enclosure asymmetric
                               e.g., {'downstream': 2.0} extends outlet side by 2x
                               
        Returns:
            EnclosureResult with paths and boundary definitions
        """
        print(f"[EnclosureGenerator] Creating box enclosure (multiplier={multiplier}x)")
        
        # Get part bounds
        part_min, part_max = self._get_stl_bounds(stl_path)
        part_center = (part_min + part_max) / 2
        part_size = part_max - part_min
        
        print(f"[EnclosureGenerator] Part bounds: {part_min} to {part_max}")
        print(f"[EnclosureGenerator] Part size: {part_size}")
        
        # Calculate enclosure bounds
        half_enclosure = part_size * multiplier / 2
        enc_min = part_center - half_enclosure
        enc_max = part_center + half_enclosure
        
        # Apply asymmetric factors if specified (e.g., longer wake region)
        if asymmetric_factors:
            # Parse flow direction
            axis = {'X': 0, 'Y': 1, 'Z': 2}[flow_direction[-1].upper()]
            positive = flow_direction[0] == '+'
            
            if 'downstream' in asymmetric_factors:
                factor = asymmetric_factors['downstream']
                if positive:
                    enc_max[axis] = part_center[axis] + half_enclosure[axis] * factor
                else:
                    enc_min[axis] = part_center[axis] - half_enclosure[axis] * factor
                    
            if 'upstream' in asymmetric_factors:
                factor = asymmetric_factors['upstream']
                if positive:
                    enc_min[axis] = part_center[axis] - half_enclosure[axis] * factor
                else:
                    enc_max[axis] = part_center[axis] + half_enclosure[axis] * factor
        
        print(f"[EnclosureGenerator] Enclosure bounds: {enc_min} to {enc_max}")
        
        # Create box mesh
        vertices, faces, face_groups = self._create_box_mesh(enc_min, enc_max)
        
        # Map flow direction to boundary patches
        axis_map = {'X': ('left', 'right'), 'Y': ('bottom', 'top'), 'Z': ('back', 'front')}
        axis = flow_direction[-1].upper()
        negative_face, positive_face = axis_map[axis]
        
        if flow_direction[0] == '+':
            inlet_face, outlet_face = negative_face, positive_face
        else:
            inlet_face, outlet_face = positive_face, negative_face
        
        # Wall faces are all non-inlet/outlet faces
        wall_faces = [name for name in face_groups.keys() 
                      if name not in [inlet_face, outlet_face]]
        
        # Create boundary patches
        boundary_patches = {
            'inlet': BoundaryPatch(
                name='inlet',
                patch_type='inlet',
                face_indices=face_groups[inlet_face]
            ),
            'outlet': BoundaryPatch(
                name='outlet', 
                patch_type='outlet',
                face_indices=face_groups[outlet_face]
            ),
            'walls': BoundaryPatch(
                name='walls',
                patch_type='wall',
                face_indices=[idx for name in wall_faces 
                             for idx in face_groups[name]]
            ),
        }
        
        # Write enclosure STL
        base_name = Path(stl_path).stem
        enc_stl_path = str(self.output_dir / f"{base_name}_enclosure.stl")
        self._write_stl(vertices, faces, enc_stl_path)
        
        print(f"[EnclosureGenerator] Enclosure saved to: {enc_stl_path}")
        
        return EnclosureResult(
            enclosure_stl_path=enc_stl_path,
            part_stl_path=stl_path,
            boundary_patches=boundary_patches,
            bounds_min=enc_min,
            bounds_max=enc_max,
            enclosure_type='box'
        )
    
    def create_cylindrical_enclosure(self, stl_path: str,
                                     length_multiplier: float = 5.0,
                                     radius_multiplier: float = 3.0,
                                     axis: str = 'X',
                                     segments: int = 32) -> EnclosureResult:
        """
        Creates cylindrical domain for axisymmetric bodies.
        
        Ideal for rockets, missiles, fuselages, pipes.
        
        Args:
            stl_path: Path to part STL file
            length_multiplier: Length as multiple of part length
            radius_multiplier: Radius as multiple of part max cross-section
            axis: Cylinder axis ('X', 'Y', 'Z')
            segments: Number of circumferential segments
            
        Returns:
            EnclosureResult with paths and boundary definitions
        """
        print(f"[EnclosureGenerator] Creating cylindrical enclosure (L={length_multiplier}x, R={radius_multiplier}x)")
        
        # Get part bounds
        part_min, part_max = self._get_stl_bounds(stl_path)
        part_center = (part_min + part_max) / 2
        part_size = part_max - part_min
        
        # Determine dimensions based on axis
        axis_idx = {'X': 0, 'Y': 1, 'Z': 2}[axis.upper()]
        other_axes = [i for i in range(3) if i != axis_idx]
        
        length = part_size[axis_idx] * length_multiplier
        max_cross_section = max(part_size[other_axes[0]], part_size[other_axes[1]])
        radius = max_cross_section * radius_multiplier / 2
        
        print(f"[EnclosureGenerator] Cylinder: length={length:.2f}, radius={radius:.2f}")
        
        # Create cylinder mesh
        vertices = []
        faces = []
        
        # Generate vertices
        angles = np.linspace(0, 2 * np.pi, segments + 1)[:-1]  # Remove duplicate endpoint
        
        # Front and back circle vertices
        for z_offset, z_idx in [(-length/2, 0), (length/2, 1)]:
            for angle in angles:
                if axis_idx == 0:  # X axis
                    vertices.append([part_center[0] + z_offset,
                                   part_center[1] + radius * np.cos(angle),
                                   part_center[2] + radius * np.sin(angle)])
                elif axis_idx == 1:  # Y axis
                    vertices.append([part_center[0] + radius * np.cos(angle),
                                   part_center[1] + z_offset,
                                   part_center[2] + radius * np.sin(angle)])
                else:  # Z axis
                    vertices.append([part_center[0] + radius * np.cos(angle),
                                   part_center[1] + radius * np.sin(angle),
                                   part_center[2] + z_offset])
        
        # Add center vertices for end caps
        back_center_idx = len(vertices)
        if axis_idx == 0:
            vertices.append([part_center[0] - length/2, part_center[1], part_center[2]])
        elif axis_idx == 1:
            vertices.append([part_center[0], part_center[1] - length/2, part_center[2]])
        else:
            vertices.append([part_center[0], part_center[1], part_center[2] - length/2])
            
        front_center_idx = len(vertices)
        if axis_idx == 0:
            vertices.append([part_center[0] + length/2, part_center[1], part_center[2]])
        elif axis_idx == 1:
            vertices.append([part_center[0], part_center[1] + length/2, part_center[2]])
        else:
            vertices.append([part_center[0], part_center[1], part_center[2] + length/2])
        
        vertices = np.array(vertices, dtype=np.float32)
        
        # Generate faces
        wall_faces = []
        inlet_faces = []
        outlet_faces = []
        
        # Cylinder wall (quads split into triangles)
        for i in range(segments):
            i_next = (i + 1) % segments
            # Back ring indices: 0 to segments-1
            # Front ring indices: segments to 2*segments-1
            v0, v1 = i, i_next
            v2, v3 = i + segments, i_next + segments
            
            # Two triangles per quad (inward normals)
            wall_faces.extend([len(faces), len(faces) + 1])
            faces.append([v0, v3, v1])  # CCW from inside
            faces.append([v0, v2, v3])
        
        # Back cap (inlet)
        for i in range(segments):
            i_next = (i + 1) % segments
            inlet_faces.append(len(faces))
            faces.append([back_center_idx, i, i_next])  # CCW from inside
        
        # Front cap (outlet)
        for i in range(segments):
            i_next = (i + 1) % segments
            outlet_faces.append(len(faces))
            faces.append([front_center_idx, i + segments + (i_next - i), i + segments])  # CCW from inside
        
        faces = np.array(faces, dtype=np.int32)
        
        # Create boundary patches
        boundary_patches = {
            'inlet': BoundaryPatch(name='inlet', patch_type='inlet', face_indices=inlet_faces),
            'outlet': BoundaryPatch(name='outlet', patch_type='outlet', face_indices=outlet_faces),
            'walls': BoundaryPatch(name='walls', patch_type='wall', face_indices=wall_faces),
        }
        
        # Calculate bounds
        bounds_min = np.array([vertices[:, i].min() for i in range(3)])
        bounds_max = np.array([vertices[:, i].max() for i in range(3)])
        
        # Write enclosure STL
        base_name = Path(stl_path).stem
        enc_stl_path = str(self.output_dir / f"{base_name}_cylinder_enclosure.stl")
        self._write_stl(vertices, faces, enc_stl_path)
        
        print(f"[EnclosureGenerator] Cylinder enclosure saved to: {enc_stl_path}")
        
        return EnclosureResult(
            enclosure_stl_path=enc_stl_path,
            part_stl_path=stl_path,
            boundary_patches=boundary_patches,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            enclosure_type='cylinder'
        )
    
    def create_spherical_enclosure(self, stl_path: str,
                                   radius_multiplier: float = 5.0,
                                   subdivisions: int = 3) -> EnclosureResult:
        """
        Creates spherical domain for omnidirectional flow.
        
        Ideal for objects with flow from all directions (e.g., hovering drones).
        
        Args:
            stl_path: Path to part STL file
            radius_multiplier: Radius as multiple of part diagonal
            subdivisions: Icosphere subdivision level (more = smoother)
            
        Returns:
            EnclosureResult with paths and boundary definitions
        """
        print(f"[EnclosureGenerator] Creating spherical enclosure (R={radius_multiplier}x)")
        
        # Get part bounds
        part_min, part_max = self._get_stl_bounds(stl_path)
        part_center = (part_min + part_max) / 2
        part_diagonal = np.linalg.norm(part_max - part_min)
        radius = part_diagonal * radius_multiplier / 2
        
        print(f"[EnclosureGenerator] Sphere: center={part_center}, radius={radius:.2f}")
        
        # Create icosphere
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
        verts = np.array([
            [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
            [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
            [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1]
        ], dtype=np.float64)
        
        faces = np.array([
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ], dtype=np.int32)
        
        # Normalize to unit sphere
        verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
        
        # Subdivide
        for _ in range(subdivisions):
            new_faces = []
            edge_midpoints = {}
            verts_list = list(verts)
            
            for face in faces:
                v0, v1, v2 = face
                
                def get_midpoint(i0, i1):
                    key = (min(i0, i1), max(i0, i1))
                    if key not in edge_midpoints:
                        mid = (verts_list[i0] + verts_list[i1]) / 2.0
                        mid = mid / np.linalg.norm(mid)
                        edge_midpoints[key] = len(verts_list)
                        verts_list.append(mid)
                    return edge_midpoints[key]
                
                m01 = get_midpoint(v0, v1)
                m12 = get_midpoint(v1, v2)
                m20 = get_midpoint(v2, v0)
                
                new_faces.extend([
                    [v0, m01, m20],
                    [v1, m12, m01],
                    [v2, m20, m12],
                    [m01, m12, m20]
                ])
            
            verts = np.array(verts_list)
            faces = np.array(new_faces, dtype=np.int32)
        
        # Flip normals inward and scale/translate
        faces = faces[:, ::-1]  # Reverse winding for inward normals
        vertices = (verts * radius + part_center).astype(np.float32)
        
        # For sphere, entire surface is "farfield" boundary
        boundary_patches = {
            'farfield': BoundaryPatch(
                name='farfield',
                patch_type='wall',  # Or 'freestream' for CFD
                face_indices=list(range(len(faces)))
            ),
        }
        
        bounds_min = vertices.min(axis=0)
        bounds_max = vertices.max(axis=0)
        
        # Write enclosure STL
        base_name = Path(stl_path).stem
        enc_stl_path = str(self.output_dir / f"{base_name}_sphere_enclosure.stl")
        self._write_stl(vertices, faces.astype(np.int32), enc_stl_path)
        
        print(f"[EnclosureGenerator] Sphere enclosure saved to: {enc_stl_path}")
        
        return EnclosureResult(
            enclosure_stl_path=enc_stl_path,
            part_stl_path=stl_path,
            boundary_patches=boundary_patches,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            enclosure_type='sphere'
        )


# === CLI ===
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python enclosure_geometry.py <stl_path> [type] [multiplier]")
        print("  type: box (default), cylinder, sphere")
        print("  multiplier: size factor (default 5.0)")
        sys.exit(1)
    
    stl_path = sys.argv[1]
    enc_type = sys.argv[2] if len(sys.argv) > 2 else 'box'
    multiplier = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
    
    gen = EnclosureGenerator()
    
    if enc_type == 'box':
        result = gen.create_box_enclosure(stl_path, multiplier=multiplier)
    elif enc_type == 'cylinder':
        result = gen.create_cylindrical_enclosure(stl_path, length_multiplier=multiplier)
    elif enc_type == 'sphere':
        result = gen.create_spherical_enclosure(stl_path, radius_multiplier=multiplier)
    else:
        print(f"Unknown enclosure type: {enc_type}")
        sys.exit(1)
    
    print(f"\nResult:")
    print(f"  Enclosure: {result.enclosure_stl_path}")
    print(f"  Bounds: {result.bounds_min} to {result.bounds_max}")
    print(f"  Patches: {list(result.boundary_patches.keys())}")
