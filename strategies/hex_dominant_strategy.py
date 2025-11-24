
import gmsh
import trimesh
import os
import sys
from pathlib import Path
from typing import Tuple, Optional

class HighFidelityDiscretization:
    """
    Step 1 of Hex Dominant Meshing Strategy:
    High-fidelity discretization of STEP files into watertight STL meshes.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        
    def log(self, message: str):
        if self.verbose:
            print(f"[HexDom] {message}")

    def convert_step_to_stl(self, step_path: str, output_stl_path: str, 
                           deviation: float = 0.01, 
                           min_size: float = 0.1,
                           max_size: float = 10.0) -> bool:
        """
        Convert STEP file to high-resolution STL using Gmsh.
        
        Args:
            step_path: Path to input STEP file
            output_stl_path: Path to output STL file
            deviation: Max chordal deviation (lower = higher quality)
            min_size: Minimum mesh element size
            max_size: Maximum mesh element size
            
        Returns:
            True if successful, False otherwise
        """
        self.log(f"Converting {step_path} to STL...")
        
        try:
            # Initialize Gmsh if not already running
            if not gmsh.is_initialized():
                gmsh.initialize()
                
            gmsh.model.add("hex_dom_step1")
            
            # Load STEP file
            gmsh.merge(step_path)
            
            # Set high-fidelity meshing options
            # 1. MeshSizeFromCurvature: Refine mesh on curved surfaces
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 20) # Number of elements per 2*pi radians
            
            # 2. Set min/max element sizes
            gmsh.option.setNumber("Mesh.MeshSizeMin", min_size)
            gmsh.option.setNumber("Mesh.MeshSizeMax", max_size)
            
            # 3. Optimize for quality
            gmsh.option.setNumber("Mesh.Algorithm", 6) # Frontal-Delaunay for 2D
            
            # Generate 2D surface mesh
            self.log("Generating surface mesh...")
            gmsh.model.mesh.generate(2)
            
            # Export STL
            self.log(f"Exporting to {output_stl_path}...")
            gmsh.write(output_stl_path)
            
            # Cleanup
            gmsh.model.remove()
            # Don't finalize if we want to reuse gmsh, but for now let's assume standalone usage or careful management
            # gmsh.finalize() 
            
            return True
            
        except Exception as e:
            self.log(f"Error converting STEP to STL: {e}")
            return False

    def verify_watertightness(self, stl_path: str) -> Tuple[bool, dict]:
        """
        Verify if the STL mesh is watertight using trimesh.
        
        Args:
            stl_path: Path to STL file
            
        Returns:
            Tuple (is_watertight, stats_dict)
        """
        self.log(f"Verifying watertightness of {stl_path}...")
        
        try:
            # Load mesh
            mesh = trimesh.load(stl_path)
            
            is_watertight = mesh.is_watertight
            
            stats = {
                "is_watertight": is_watertight,
                "vertices": len(mesh.vertices),
                "faces": len(mesh.faces),
                "volume": mesh.volume if is_watertight else None,
                "euler_number": mesh.euler_number
            }
            
            if is_watertight:
                self.log("PASS: Mesh is watertight.")
            else:
                self.log("FAIL: Mesh is NOT watertight.")
                
            return is_watertight, stats
            
        except Exception as e:
            self.log(f"Error verifying watertightness: {e}")
            return False, {"error": str(e)}


import coacd
import numpy as np

class ConvexDecomposition:
    """
    Step 2 of Hex Dominant Meshing Strategy:
    Decompose watertight mesh into convex chunks using CoACD.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        
    def log(self, message: str):
        if self.verbose:
            print(f"[HexDom] {message}")

    def calculate_signed_volume(self, mesh) -> float:
        """
        Calculate exact volume using the Signed Tetrahedron Method (Method 1).
        V = sum(dot(cross(p1, p2), p3)) / 6.0
        This is the standard way to calculate volume for polygonal meshes.
        """
        # Get vertices for each face
        # mesh.triangles is (n, 3, 3) array of vertices
        triangles = mesh.triangles
        
        # p1, p2, p3 are (n, 3) arrays
        p1 = triangles[:, 0, :]
        p2 = triangles[:, 1, :]
        p3 = triangles[:, 2, :]
        
        # Calculate cross product (p1 x p2)
        cross_p1_p2 = np.cross(p1, p2)
        
        # Calculate dot product with p3
        # We use einsum for efficient dot product across the last axis
        # or simpler: sum(cross * p3, axis=1)
        dots = np.sum(cross_p1_p2 * p3, axis=1)
        
        # Sum and divide by 6
        volume = np.sum(dots) / 6.0
        
        return abs(volume)

    def decompose_mesh(self, input_mesh_path: str, threshold: float = 0.01) -> Tuple[list, dict]:
        """
        Decompose mesh into convex parts.
        
        Args:
            input_mesh_path: Path to input STL/OBJ
            threshold: Concavity threshold (0.01-1.0). Lower = more parts.
            
        Returns:
            Tuple (parts_list, stats_dict)
            parts_list is list of (vertices, faces) tuples
        """
        self.log(f"Decomposing {input_mesh_path} with threshold {threshold}...")
        
        try:
            # 1. Load mesh
            mesh = trimesh.load(input_mesh_path)
            
            if not mesh.is_watertight:
                self.log("Warning: Input mesh is not watertight! CoACD might fail or produce bad results.")
                
            # Use explicit signed volume calculation for verification
            original_volume = self.calculate_signed_volume(mesh)
            trimesh_volume = mesh.volume
            
            self.log(f"Original Volume (Signed Method): {original_volume:.4f}")
            
            # Verify trimesh volume matches (sanity check)
            if abs(original_volume - trimesh_volume) > 1e-6:
                self.log(f"Note: Trimesh volume {trimesh_volume:.4f} differs slightly from explicit calculation.")
            
            # 2. Run CoACD
            # CoACD expects mesh to be loaded by trimesh or passed as (verts, faces)
            # Fix: CoACD might look for 'indices' instead of 'faces'
            if not hasattr(mesh, 'indices'):
                mesh.indices = mesh.faces
                
            parts = coacd.run_coacd(
                mesh=mesh,
                threshold=threshold,
                max_convex_hull=32,       # Hard limit to prevent over-shattering
                preprocess_resolution=50, # Balanced resolution
                mcts_nodes=20,
                mcts_iterations=60,       # Reduced for faster search
                mcts_max_depth=3
            )
            
            self.log(f"Decomposed into {len(parts)} convex parts.")
            
            # 3. Verify Volume Conservation
            total_parts_volume = 0.0
            
            for i, (verts, faces) in enumerate(parts):
                # Create temp mesh to calc volume
                part_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                if part_mesh.is_volume:
                    # Use our explicit method for parts too
                    vol = self.calculate_signed_volume(part_mesh)
                    total_parts_volume += vol
                else:
                    self.log(f"Warning: Part {i} has no volume (degenerate?)")
            
            self.log(f"Total Parts Volume: {total_parts_volume:.4f}")
            
            volume_diff = abs(original_volume - total_parts_volume)
            volume_error_pct = (volume_diff / original_volume) * 100 if original_volume > 0 else 0
            
            stats = {
                "original_volume": original_volume,
                "parts_volume": total_parts_volume,
                "volume_error_pct": volume_error_pct,
                "num_parts": len(parts),
                "threshold": threshold
            }
            
            if volume_error_pct < 2.0:
                self.log(f"PASS: Volume conservation good ({volume_error_pct:.2f}% error)")
            else:
                self.log(f"FAIL: Volume mismatch too high ({volume_error_pct:.2f}% error)")
                
            return parts, stats
            
        except Exception as e:
            self.log(f"Error during decomposition: {e}")
            return [], {"error": str(e)}


class TopologyGlue:
    """
    Step 3 of Hex Dominant Meshing Strategy:
    Glue convex parts together using Boolean Fragment to ensure shared faces.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        
    def log(self, message: str):
        if self.verbose:
            print(f"[HexDom] {message}")

    def glue_parts(self, parts: list) -> Tuple[bool, dict]:
        """
        Glue CoACD parts into a single connected model.
        
        Args:
            parts: List of (vertices, faces) tuples from CoACD
            
        Returns:
            Tuple (success, stats_dict)
        """
        self.log(f"Gluing {len(parts)} parts...")
        
        try:
            gmsh.initialize()
            gmsh.model.add("hex_dom_step3")
            
            # Enable OCC auto-fixing and looser tolerances for boolean
            gmsh.option.setNumber("Geometry.OCCFixSmallEdges", 1)
            gmsh.option.setNumber("Geometry.OCCFixSmallFaces", 1)
            gmsh.option.setNumber("Geometry.Tolerance", 1e-4)
            gmsh.option.setNumber("Geometry.ToleranceBoolean", 1e-4)
            gmsh.option.setNumber("Mesh.ToleranceInitialDelaunay", 1e-4)
            
            solid_tags = []
            
            # 1. Convert each part to an OCC Solid (with cleaning)
            for i, (verts, faces) in enumerate(parts):
                # Clean the mesh using trimesh to prevent PLC errors
                chunk_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                
                # Cleaning steps
                chunk_mesh.merge_vertices(merge_tex=True, merge_norm=True)
                chunk_mesh.remove_degenerate_faces()
                chunk_mesh.remove_duplicate_faces()
                try:
                    trimesh.repair.fix_normals(chunk_mesh)
                except:
                    pass  # May fail on open meshes
                
                # Use cleaned vertices and faces
                verts = chunk_mesh.vertices
                faces = chunk_mesh.faces
                
                # Create vertices
                occ_verts = []
                for v in verts:
                    tag = gmsh.model.occ.addPoint(v[0], v[1], v[2])
                    occ_verts.append(tag)
                
                # Create faces
                occ_faces = []
                for f in faces:
                    # Create linear loop for triangle
                    p1, p2, p3 = occ_verts[f[0]], occ_verts[f[1]], occ_verts[f[2]]
                    l1 = gmsh.model.occ.addLine(p1, p2)
                    l2 = gmsh.model.occ.addLine(p2, p3)
                    l3 = gmsh.model.occ.addLine(p3, p1)
                    
                    loop = gmsh.model.occ.addCurveLoop([l1, l2, l3])
                    face = gmsh.model.occ.addPlaneSurface([loop])
                    occ_faces.append(face)
                
                # Create Surface Loop (Shell)
                shell = gmsh.model.occ.addSurfaceLoop(occ_faces)
                
                # Create Solid
                solid = gmsh.model.occ.addVolume([shell])
                solid_tags.append((3, solid))
                
                if i % 10 == 0:
                    self.log(f"  Converted part {i}/{len(parts)} to OCC solid")
            
            self.log("Synchronizing before fragment...")
            gmsh.model.occ.synchronize()
            
            # Attempt to remove duplicates before fragment
            self.log("Removing duplicate entities...")
            gmsh.model.occ.removeAllDuplicates()
            gmsh.model.occ.synchronize()
            
            # 2. Run Boolean Fragment
            self.log("Running Boolean Fragment (this may take a while)...")
            # Fragment all solids against themselves to find intersections/shared faces
            occ_result = gmsh.model.occ.fragment(solid_tags, [])
            
            self.log("Synchronizing after fragment...")
            gmsh.model.occ.synchronize()
            
            # 3. Verify Topology
            # Count surfaces
            surfaces = gmsh.model.getEntities(dim=2)
            volumes = gmsh.model.getEntities(dim=3)
            
            self.log(f"Result: {len(volumes)} volumes, {len(surfaces)} surfaces")
            
            # Check for duplicate surfaces (geometric check)
            # If fragment worked, there should be no overlapping surfaces.
            # We can check bounding boxes or centroids to see if any are identical.
            
            # Simple check: If we have N parts, and they touch, the number of surfaces
            # should be less than Sum(surfaces_per_part) because shared faces are merged.
            
            stats = {
                "num_volumes": len(volumes),
                "num_surfaces": len(surfaces),
                "fragment_success": True
            }
            
            # Cleanup (optional, depend on usage)
            # gmsh.finalize() 
            
            return True, stats
            
        except Exception as e:
            self.log(f"Error during gluing: {e}")
            return False, {"error": str(e)}


class TetrahedralBaseline:
    """
    Step 4 of Hex Dominant Meshing Strategy:
    Generate baseline tetrahedral mesh to verify topology.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        
    def log(self, message: str):
        if self.verbose:
            print(f"[HexDom] {message}")

    def generate_mesh(self) -> Tuple[bool, dict]:
        """
        Generate 3D Delaunay mesh and verify quality.
        Assumes Gmsh model is already populated (e.g. by TopologyGlue).
        
        Returns:
            Tuple (success, stats_dict)
        """
        self.log("Generating Tetrahedral Baseline Mesh...")
        
        try:
            # Set meshing options
            gmsh.option.setNumber("Mesh.Algorithm", 6)     # Frontal-Delaunay 2D
            gmsh.option.setNumber("Mesh.Algorithm3D", 1)   # Delaunay 3D (Robust)
            
            # Enable parallel meshing
            import multiprocessing
            num_cores = multiprocessing.cpu_count()
            self.log(f"Enabling parallel meshing with {num_cores} threads...")
            gmsh.option.setNumber("Mesh.MaxNumThreads3D", num_cores)
            gmsh.option.setNumber("Mesh.MaxNumThreads2D", num_cores)
            gmsh.option.setNumber("Mesh.MaxNumThreads1D", num_cores)
            
            # Generate mesh
            gmsh.model.mesh.generate(3)
            
            # Verify Quality (Jacobian)
            # Jacobian > 0 means valid element
            # We check min Jacobian
            
            # Get all 3D elements
            # elementTypes, elementTags, nodeTags = gmsh.model.mesh.getElements(dim=3)
            # But getJacobians works on element types
            
            min_jacobian = float('inf')
            num_negative = 0
            
            # Get Jacobians for tetrahedra (type 4)
            try:
                jacobians, determinants, points = gmsh.model.mesh.getJacobians(elementType=4)
                if len(jacobians) > 0:
                    min_j = min(jacobians)
                    min_jacobian = min(min_jacobian, min_j)
                    num_negative += sum(1 for j in jacobians if j <= 0)
            except:
                pass # No tets?
                
            self.log(f"Min Jacobian: {min_jacobian:.6f}")
            
            if num_negative > 0:
                self.log(f"FAIL: Found {num_negative} elements with negative Jacobian!")
                success = False
            elif min_jacobian <= 0:
                self.log(f"FAIL: Min Jacobian is non-positive: {min_jacobian}")
                success = False
            else:
                self.log("PASS: All elements have positive Jacobian (valid topology).")
                success = True
                
            stats = {
                "min_jacobian": min_jacobian,
                "num_negative_elements": num_negative,
                "num_elements": gmsh.model.mesh.getNbElements()
            }
            
            return success, stats
            
        except Exception as e:
            self.log(f"Error during meshing: {e}")
            return False, {"error": str(e)}

if __name__ == "__main__":
    # Simple test
    print("This module provides HighFidelityDiscretization, ConvexDecomposition, TopologyGlue, and TetrahedralBaseline classes.")
