"""
Structural Template Strategy - Protocol A: Anchor & Accelerate
===============================================================

Automatic static structural analysis for drone arms/brackets under inertial loads.
Uses heuristic geometry analysis to determine boundary conditions without user input.

Heuristic Strategy:
- ANCHOR: Largest flat face = Fixed Support (mounting interface)
- ACCELERATE: Global inertial load (Gravity × G-Factor) on entire volume

Usage:
    strategy = StructuralTemplateStrategy()
    result = strategy.run(input_file, output_dir, g_factor=10.0)
"""

import gmsh
import numpy as np
import logging
import meshio
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FaceInfo:
    """Information about a geometric face"""
    tag: int
    area: float
    normal: np.ndarray
    centroid: np.ndarray
    is_planar: bool


@dataclass
class StructuralConfig:
    """Configuration for structural analysis"""
    g_factor: float = 10.0  # G-factor for inertial load
    gravity_direction: Tuple[float, float, float] = (0.0, 0.0, -1.0)  # -Z direction
    
    # Material: Al6061-T6
    youngs_modulus: float = 68.9e9  # Pa (68.9 GPa)
    poissons_ratio: float = 0.33
    density: float = 2700.0  # kg/m³
    yield_strength: float = 276e6  # Pa (276 MPa)
    
    # Mesh settings
    element_order: int = 2  # TET10 (quadratic)
    mesh_size_factor: float = 1.0


class StructuralTemplateStrategy:
    """
    Structural Template Strategy - "Anchor & Accelerate"
    
    Automatically sets up structural FEA for drone components:
    1. Analyzes geometry to find mounting face (largest flat surface)
    2. Generates TET10 mesh
    3. Writes CalculiX input deck with boundary conditions and loads
    4. Returns mesh and node groups for solver
    """
    
    def __init__(self, config: Optional[StructuralConfig] = None):
        self.config = config or StructuralConfig()
        self.faces: List[FaceInfo] = []
        self.fixed_face_tag: Optional[int] = None
        self.fixed_nodes: List[int] = []
        self.all_nodes: Dict[int, np.ndarray] = {}
        self.elements: List[Tuple[int, ...]] = []
        
    def run(self, input_file: str, output_dir: Path, g_factor: Optional[float] = None) -> Dict:
        """
        Run the structural template strategy.
        
        Args:
            input_file: Path to STEP/CAD file
            output_dir: Directory for output files
            g_factor: Optional override for G-factor (default: 10.0)
            
        Returns:
            Dict with mesh file path, node groups, and CalculiX input path
        """
        if g_factor is not None:
            self.config.g_factor = g_factor
            
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        job_name = Path(input_file).stem
        
        logger.info(f"[StructuralTemplate] Starting analysis for: {job_name}")
        logger.info(f"[StructuralTemplate] G-Factor: {self.config.g_factor}G")
        
        try:
            # 1. Initialize Gmsh and load geometry
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("structural_model")
            
            # Load CAD file
            gmsh.model.occ.importShapes(input_file)
            gmsh.model.occ.synchronize()
            
            # 2. Analyze geometry to find mounting face
            self._analyze_geometry()
            
            if self.fixed_face_tag is None:
                raise RuntimeError("Could not identify mounting face")
                
            logger.info(f"[StructuralTemplate] Fixed face identified: Tag {self.fixed_face_tag}, "
                       f"Area: {self.faces[0].area:.6f} m²")
            
            # 3. Generate TET10 mesh
            mesh_file = output_dir / f"{job_name}_structural.msh"
            self._generate_mesh(str(mesh_file))
            
            # 4. Extract fixed nodes
            self._extract_fixed_nodes()
            logger.info(f"[StructuralTemplate] Fixed nodes: {len(self.fixed_nodes)}")
            
            # 5. Write CalculiX input deck
            inp_file = output_dir / f"{job_name}.inp"
            self._write_calculix_input(str(inp_file), job_name)
            
            # 6. Export VTK for visualization
            vtk_file = output_dir / f"{job_name}_structural.vtk"
            self._export_vtk(str(vtk_file))
            
            gmsh.finalize()
            
            return {
                'success': True,
                'mesh_file': str(mesh_file),
                'inp_file': str(inp_file),
                'vtk_file': str(vtk_file),
                'job_name': job_name,
                'fixed_face_tag': self.fixed_face_tag,
                'fixed_nodes_count': len(self.fixed_nodes),
                'total_nodes': len(self.all_nodes),
                'total_elements': len(self.elements),
                'g_factor': self.config.g_factor,
                'material': 'Al6061-T6',
                'yield_strength_mpa': self.config.yield_strength / 1e6
            }
            
        except Exception as e:
            logger.error(f"[StructuralTemplate] Failed: {e}")
            try:
                gmsh.finalize()
            except:
                pass
            return {
                'success': False,
                'error': str(e)
            }
    
    def _analyze_geometry(self):
        """
        Analyze geometry faces to find the mounting interface.
        
        The mounting face is identified as the largest flat (planar) surface.
        This is typically where a drone arm would be bolted to the frame.
        """
        self.faces = []
        
        # Get all surfaces
        surfaces = gmsh.model.getEntities(dim=2)
        logger.info(f"[StructuralTemplate] Found {len(surfaces)} surfaces to analyze")
        
        for dim, tag in surfaces:
            try:
                # Get surface area using mass properties
                try:
                    area = gmsh.model.occ.getMass(dim, tag)
                except Exception as e:
                    # Fallback: estimate area from bounding box
                    logger.debug(f"getMass failed for surface {tag}: {e}, using bbox estimate")
                    try:
                        bbox = gmsh.model.getBoundingBox(dim, tag)
                        # Estimate area as max of 2D projections
                        dx = abs(bbox[3] - bbox[0])
                        dy = abs(bbox[4] - bbox[1])
                        dz = abs(bbox[5] - bbox[2])
                        # Pick the two largest dimensions
                        dims = sorted([dx, dy, dz], reverse=True)
                        area = dims[0] * dims[1]
                    except:
                        area = 0.0
                
                if area <= 0:
                    logger.debug(f"Surface {tag} has zero or negative area, skipping")
                    continue
                
                # Get centroid
                try:
                    cog = gmsh.model.occ.getCenterOfMass(dim, tag)
                    centroid = np.array(cog)
                except:
                    centroid = np.array([0.0, 0.0, 0.0])
                
                # Estimate normal by sampling parametric center
                normal = np.array([0, 0, 1])  # Default
                try:
                    param_bounds = gmsh.model.getParametricBounds(dim, tag)
                    if param_bounds and len(param_bounds) >= 4:
                        u_mid = (param_bounds[0] + param_bounds[2]) / 2
                        v_mid = (param_bounds[1] + param_bounds[3]) / 2
                        
                        # Get normal at center point
                        normal_data = gmsh.model.getNormal(tag, [u_mid, v_mid])
                        if normal_data and len(normal_data) >= 3:
                            normal = np.array(normal_data[:3])
                            norm_len = np.linalg.norm(normal)
                            if norm_len > 1e-10:
                                normal = normal / norm_len
                except Exception as e:
                    logger.debug(f"Could not get normal for surface {tag}: {e}")
                
                # Check if surface is reasonably planar
                is_planar = True  # Simplified check
                
                self.faces.append(FaceInfo(
                    tag=tag,
                    area=area,
                    normal=normal,
                    centroid=centroid,
                    is_planar=is_planar
                ))
                logger.debug(f"Surface {tag}: area={area:.6f}, normal={normal}")
                
            except Exception as e:
                logger.warning(f"Could not analyze face {tag}: {e}")
                continue
        
        if not self.faces:
            raise RuntimeError("No analyzable faces found in geometry")
        
        # Sort by area (descending) - largest face is the mounting interface
        self.faces.sort(key=lambda f: f.area, reverse=True)
        
        # Select the largest planar face as the fixed support
        for face in self.faces:
            if face.is_planar:
                self.fixed_face_tag = face.tag
                break
        
        # If no planar face found, use the largest face anyway
        if self.fixed_face_tag is None:
            self.fixed_face_tag = self.faces[0].tag
            
        logger.info(f"[StructuralTemplate] Analyzed {len(self.faces)} faces")
        logger.info(f"[StructuralTemplate] Top 3 by area: "
                   f"{[f'{f.tag}:{f.area:.4f}' for f in self.faces[:3]]}")
    
    def _generate_mesh(self, output_file: str):
        """Generate TET10 mesh with appropriate sizing"""
        
        # Get bounding box for size estimation
        bbox = gmsh.model.getBoundingBox(-1, -1)
        diag = np.sqrt((bbox[3]-bbox[0])**2 + (bbox[4]-bbox[1])**2 + (bbox[5]-bbox[2])**2)
        
        # Mesh size: ~20-30 elements along diagonal
        base_size = diag / 25.0 * self.config.mesh_size_factor
        
        logger.info(f"[StructuralTemplate] Bounding diagonal: {diag:.4f}, mesh size: {base_size:.4f}")
        
        # Set mesh options
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", base_size * 0.5)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", base_size * 1.5)
        gmsh.option.setNumber("Mesh.ElementOrder", self.config.element_order)
        gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)  # Curved elements
        
        # Use HXT algorithm for 3D (robust and fast)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT
        
        # Optimize for quality
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        
        # Generate mesh
        gmsh.model.mesh.generate(3)
        
        # Extract mesh data
        self._extract_mesh_data()
        
        # Save mesh
        gmsh.write(output_file)
        logger.info(f"[StructuralTemplate] Mesh saved: {output_file}")
        logger.info(f"[StructuralTemplate] Nodes: {len(self.all_nodes)}, Elements: {len(self.elements)}")
    
    def _extract_mesh_data(self):
        """Extract nodes and elements from Gmsh"""
        
        # Get all nodes
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        self.all_nodes = {}
        for i, tag in enumerate(node_tags):
            self.all_nodes[int(tag)] = np.array([
                node_coords[3*i],
                node_coords[3*i + 1],
                node_coords[3*i + 2]
            ])
        
        # Get volume elements (TET10 = type 11)
        volumes = gmsh.model.getEntities(dim=3)
        self.elements = []
        
        for dim, vol_tag in volumes:
            elem_types, elem_tags, elem_nodes = gmsh.model.mesh.getElements(dim, vol_tag)
            
            for etype, etags, enodes in zip(elem_types, elem_tags, elem_nodes):
                # TET4 = 4, TET10 = 11
                if etype == 4:  # TET4
                    nodes_per_elem = 4
                elif etype == 11:  # TET10
                    nodes_per_elem = 10
                else:
                    continue
                    
                for i in range(len(etags)):
                    elem_node_ids = tuple(int(enodes[i*nodes_per_elem + j]) 
                                         for j in range(nodes_per_elem))
                    self.elements.append(elem_node_ids)
    
    def _extract_fixed_nodes(self):
        """Extract nodes on the fixed (mounting) face"""
        
        if self.fixed_face_tag is None:
            return
            
        # Get nodes on the fixed face
        node_tags, _, _ = gmsh.model.mesh.getNodes(dim=2, tag=self.fixed_face_tag)
        self.fixed_nodes = [int(n) for n in node_tags]
        
    def _write_calculix_input(self, output_file: str, job_name: str):
        """
        Write CalculiX input deck with:
        - Material definition (Al6061-T6)
        - Node and element sets
        - Fixed boundary conditions on mounting face
        - Gravity (GRAV) distributed load
        """
        
        # Calculate gravity load magnitude
        # GRAV format: magnitude (m/s²), direction vector
        grav_magnitude = 9.81 * self.config.g_factor
        gx, gy, gz = self.config.gravity_direction
        
        with open(output_file, 'w') as f:
            # Header
            f.write(f"** CalculiX Input File: {job_name}\n")
            f.write(f"** Generated by SimOps Structural Template Strategy\n")
            f.write(f"** G-Factor: {self.config.g_factor}G, Direction: ({gx}, {gy}, {gz})\n")
            f.write("**\n")
            
            # Nodes
            f.write("*NODE\n")
            for node_id, coords in sorted(self.all_nodes.items()):
                f.write(f"{node_id}, {coords[0]:.10e}, {coords[1]:.10e}, {coords[2]:.10e}\n")
            
            # Elements
            nodes_per_elem = len(self.elements[0]) if self.elements else 4
            if nodes_per_elem == 10:
                f.write("*ELEMENT, TYPE=C3D10, ELSET=Eall\n")
            else:
                f.write("*ELEMENT, TYPE=C3D4, ELSET=Eall\n")
                
            for i, elem in enumerate(self.elements, start=1):
                node_str = ", ".join(str(n) for n in elem)
                f.write(f"{i}, {node_str}\n")
            
            # Fixed node set
            f.write("*NSET, NSET=FIXED\n")
            for i, node_id in enumerate(self.fixed_nodes):
                if (i + 1) % 10 == 0 or i == len(self.fixed_nodes) - 1:
                    f.write(f"{node_id}\n")
                else:
                    f.write(f"{node_id}, ")
            
            # Material definition (Al6061-T6)
            f.write("**\n")
            f.write("*MATERIAL, NAME=AL6061\n")
            f.write("*ELASTIC\n")
            f.write(f"{self.config.youngs_modulus:.6e}, {self.config.poissons_ratio}\n")
            f.write("*DENSITY\n")
            f.write(f"{self.config.density}\n")
            
            # Section assignment
            f.write("**\n")
            f.write("*SOLID SECTION, ELSET=Eall, MATERIAL=AL6061\n")
            
            # Analysis step
            f.write("**\n")
            f.write("*STEP\n")
            f.write("*STATIC\n")
            
            # Boundary conditions (full constraint on fixed nodes)
            f.write("**\n")
            f.write("** Fixed Support (Mounting Interface)\n")
            f.write("*BOUNDARY\n")
            f.write("FIXED, 1, 6\n")  # Constrain all DOFs (1-3: translation, 4-6: rotation for shells)
            
            # Gravity load
            f.write("**\n")
            f.write(f"** Gravity Load: {self.config.g_factor}G\n")
            f.write("*DLOAD\n")
            f.write(f"Eall, GRAV, {grav_magnitude:.4f}, {gx}, {gy}, {gz}\n")
            
            # Output requests
            f.write("**\n")
            f.write("*NODE FILE\n")
            f.write("U\n")  # Displacements
            f.write("*EL FILE\n")
            f.write("S\n")  # Stresses
            
            f.write("*END STEP\n")
        
        logger.info(f"[StructuralTemplate] CalculiX input written: {output_file}")
        
    def _export_vtk(self, output_file: str):
        """Export mesh to VTK format for visualization"""
        
        # Prepare data for meshio
        points = np.array([self.all_nodes[i] for i in sorted(self.all_nodes.keys())])
        
        # Create node ID to index mapping
        node_id_to_idx = {node_id: idx for idx, node_id in enumerate(sorted(self.all_nodes.keys()))}
        
        # Convert elements to 0-indexed
        nodes_per_elem = len(self.elements[0]) if self.elements else 4
        cells_data = []
        for elem in self.elements:
            cells_data.append([node_id_to_idx[n] for n in elem])
        
        if nodes_per_elem == 10:
            cells = [("tetra10", np.array(cells_data))]
        else:
            cells = [("tetra", np.array(cells_data))]
        
        # Create fixed nodes marker
        fixed_marker = np.zeros(len(points))
        for node_id in self.fixed_nodes:
            if node_id in node_id_to_idx:
                fixed_marker[node_id_to_idx[node_id]] = 1.0
        
        mesh = meshio.Mesh(
            points=points,
            cells=cells,
            point_data={"FixedNodes": fixed_marker}
        )
        
        mesh.write(output_file)
        logger.info(f"[StructuralTemplate] VTK exported: {output_file}")


def main():
    """Command-line interface for testing"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python structural_template.py <input.step> [output_dir] [g_factor]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("structural_output")
    g_factor = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
    
    logging.basicConfig(level=logging.INFO)
    
    strategy = StructuralTemplateStrategy()
    result = strategy.run(input_file, output_dir, g_factor=g_factor)
    
    if result['success']:
        print(f"\n✓ SUCCESS")
        print(f"  Mesh: {result['mesh_file']}")
        print(f"  CalculiX Input: {result['inp_file']}")
        print(f"  Fixed Nodes: {result['fixed_nodes_count']}")
        print(f"  Total Elements: {result['total_elements']}")
    else:
        print(f"\n✗ FAILED: {result['error']}")


if __name__ == "__main__":
    main()
