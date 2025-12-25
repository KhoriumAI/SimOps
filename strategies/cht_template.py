"""
CHT Template Strategy - Protocol B: Conjugate Heat Transfer
===========================================================

Automated CHT setup ensuring conformal meshing between Solid and Fluid domains.
Implements the "Correct Approach" using shared topology (BooleanFragments).

Algorithm:
1. Enclosure Generation: Create fluid domain around solid (3L upstream, 15L downstream).
2. Boolean Fragment: Imprint solid onto fluid to ensure shared nodes at interface.
3. Region Identification:
   - Solid: User-defined material (e.g., Aluminum)
   - Fluid: Air/Water
   - Interface: Strictly coupled (shared nodes)
4. Meshing: Conformal mesh generation.
5. Export: Mesh file and detailed boundary information.

Usage:
    strategy = CHTTemplateStrategy()
    result = strategy.run(input_file, output_dir, flow_axis='x')
"""

import gmsh
import numpy as np
import logging
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class CHTConfig:
    """Configuration for CHT analysis"""
    # Flow physics
    flow_velocity: float = 10.0  # m/s
    flow_axis: str = 'x'         # 'x', 'y', or 'z'
    flow_direction: int = 1      # 1 or -1
    
    # Domain sizing (multipliers of characteristic length L)
    upstream_mult: float = 3.0
    downstream_mult: float = 15.0
    side_mult: float = 5.0      # Far-field distance
    
    # Materials
    solid_material: str = "Aluminum"
    fluid_material: str = "Air"
    
    # Mesh settings
    element_order: int = 1       # Linear elements often preferred for CFD
    mesh_size_factor: float = 1.0
    boundary_layers: int = 3     # Number of inflation layers in fluid

@dataclass
class RegionInfo:
    """Info about a discovered volume"""
    tag: int
    type: str  # 'solid', 'fluid'
    centroid: Tuple[float, float, float]
    volume: float

class CHTTemplateStrategy:
    """
    Conjugate Heat Transfer Template
    
    Automates the creation of a multi-region (Solid + Fluid) simulation
    with guaranteed shared topology at the interface.
    """
    
    def __init__(self, config: Optional[CHTConfig] = None):
        self.config = config or CHTConfig()
        
        # State
        self.solid_orig_bbox = None
        self.solid_orig_centroid = None
        self.char_length = 0.0
        
        # Tags
        self.solid_volume_tag: Optional[int] = None
        self.fluid_volume_tag: Optional[int] = None
        
        # Boundary tags
        self.inlet_tags: List[int] = []
        self.outlet_tags: List[int] = []
        self.farfield_tags: List[int] = []
        self.interface_tags: List[int] = []
        
        # Results
        self.node_counts = {}
        self.element_counts = {}

    def run(self, input_file: str, output_dir: Path, 
            flow_axis: Optional[str] = None, 
            velocity: Optional[float] = None) -> Dict[str, Any]:
        """
        Run the CHT pipeline.
        
        Args:
            input_file: Path to solid CAD file (STEP/IGES)
            output_dir: Output directory
            flow_axis: Override flow axis ('x', 'y', 'z')
            velocity: Override inlet velocity
            
        Returns:
            Dictionary with results and file paths.
        """
        if flow_axis: self.config.flow_axis = flow_axis.lower()
        if velocity: self.config.flow_velocity = velocity
            
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        job_name = Path(input_file).stem
        
        logger.info(f"[CHT] Starting CHT analysis for: {job_name}")
        
        try:
            # 1. Initialize
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("cht_model")
            
            # 2. Import Solid
            logger.info("[CHT] Importing solid geometry...")
            gmsh.model.occ.importShapes(input_file)
            gmsh.model.occ.synchronize()
            
            # Get original solid stats for tracking
            self._analyze_input_solid()
            
            # 3. Create Fluid Enclosure and Boolean Fragment
            self._create_domain_and_topology()
            
            # 4. Identify Regions and Boundaries
            self._identify_entities()
            
            # 5. Define Physical Groups
            self._create_physical_groups()
            
            # 6. Generate Mesh
            mesh_file = output_dir / f"{job_name}_cht.msh"
            self._generate_mesh(str(mesh_file))
            
            # 7. Write Region Info / Solver setup
            info_file = output_dir / f"{job_name}_regions.json"
            self._write_region_info(str(info_file))
            
            # 8. Export VTK
            vtk_file = output_dir / f"{job_name}_cht.vtk"
            gmsh.write(str(vtk_file))
            
            gmsh.finalize()
            
            return {
                'success': True,
                'mesh_file': str(mesh_file),
                'vtk_file': str(vtk_file),
                'info_file': str(info_file),
                'regions': {
                    'solid_vol': self.solid_volume_tag,
                    'fluid_vol': self.fluid_volume_tag,
                    'interface_faces': len(self.interface_tags)
                }
            }
            
        except Exception as e:
            logger.error(f"[CHT] Failed: {e}", exc_info=True)
            try:
                gmsh.finalize()
            except:
                pass
            return {
                'success': False,
                'error': str(e)
            }

    def _analyze_input_solid(self):
        """Analyze the imported solid to determine bounding box and logic."""
        # Assume all imported 3D entities are the "Solid"
        solids = gmsh.model.getEntities(3)
        if not solids:
            raise RuntimeError("No solid volume found in input file")
            
        # Compute combined bounding box
        min_pt = np.array([float('inf')] * 3)
        max_pt = np.array([float('-inf')] * 3)
        
        total_vol = 0
        w_center = np.zeros(3)
        
        for dim, tag in solids:
            bb = gmsh.model.getBoundingBox(dim, tag)
            min_pt = np.minimum(min_pt, [bb[0], bb[1], bb[2]])
            max_pt = np.maximum(max_pt, [bb[3], bb[4], bb[5]])
            
            # Center of mass for tracking
            props = gmsh.model.occ.getMass(dim, tag)
            # Occ.getMass returns simply mass/vol for volume. 
            # Center of gravity is separate.
            cog = gmsh.model.occ.getCenterOfMass(dim, tag)
            vol = props # For solids, getMass is volume if density=1
            
            total_vol += vol
            w_center += np.array(cog) * vol
            
        self.solid_orig_bbox = (min_pt, max_pt)
        self.solid_orig_centroid = w_center / total_vol if total_vol > 0 else np.zeros(3)
        
        # Characteristic length (Diagonal of bbox)
        diag = np.linalg.norm(max_pt - min_pt)
        self.char_length = diag
        
        logger.info(f"[CHT] Solid BBox: {min_pt} to {max_pt}")
        logger.info(f"[CHT] Characteristic Length (L): {self.char_length:.4f}")

    def _create_domain_and_topology(self):
        """
        Create the fluid enclosure and perform BooleanFragments.
        This is the core "Shared Topology" operation.
        """
        min_b, max_b = self.solid_orig_bbox
        L = self.char_length
        c = self.config
        
        # Determine Enclosure Bounds based on Flow Direction
        # Default box centered on solid, except upstream/downstream
        
        # Map axes to indices
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        idx = axis_map.get(c.flow_axis, 0)
        
        enc_min = [min_b[0], min_b[1], min_b[2]]
        enc_max = [max_b[0], max_b[1], max_b[2]]
        
        # Apply padding
        for i in range(3):
            size_dim = max_b[i] - min_b[i]
            # If this is the flow axis
            if i == idx:
                if c.flow_direction > 0:
                    # Flow +X: Upstream is -X, Downstream is +X
                    enc_min[i] -= c.upstream_mult * L
                    enc_max[i] += c.downstream_mult * L
                else:
                    # Flow -X: Upstream is +X, Downstream is -X
                    enc_min[i] -= c.downstream_mult * L
                    enc_max[i] += c.upstream_mult * L
            else:
                # Side boundaries
                center = (max_b[i] + min_b[i]) / 2
                half_width = size_dim / 2 + (c.side_mult * L)
                enc_min[i] = center - half_width
                enc_max[i] = center + half_width
                
        logger.info(f"[CHT] Enclosure Bounds: {enc_min} to {enc_max}")
        
        # Create Box
        box_tag = gmsh.model.occ.addBox(
            enc_min[0], enc_min[1], enc_min[2],
            enc_max[0] - enc_min[0],
            enc_max[1] - enc_min[1],
            enc_max[2] - enc_min[2]
        )
        
        # Get existing solids
        input_solids = gmsh.model.getEntities(3)
        solid_tags_list = [t for d, t in input_solids] # These are the tags of the object
        
        # Boolean Fragment
        # object = [(3, tag)...]
        # tool = [(3, box)]
        # Fragment(object, tool) splits everything.
        
        object_dimtags = [(3, t) for t in solid_tags_list]
        tool_dimtags = [(3, box_tag)]
        
        logger.info("[CHT] Executing BooleanFragment (Imprint)...")
        # Output: [(dim, tag)], map
        out_dimtags, out_map = gmsh.model.occ.fragment(object_dimtags, tool_dimtags)
        gmsh.model.occ.synchronize()
        
        # out_dimtags contains all resulting volumes.
        # We need to distinguish Fluid from Solid.
        # Strategy: The resulting volume that contains the original centroid (or is the original solid) is Solid.
        # The new volume (box - solid) is Fluid.
        
        # More robust: check if volume is inside the original bbox or check com.
        pass # Actual logic in identify_entities

    def _identify_entities(self):
        """Identify which volumes are Solid vs Fluid and categorize faces."""
        
        all_vols = gmsh.model.getEntities(3)
        
        # Find Solid Volume
        # Logic: The volume closest to the original centroid is likely the solid.
        # Or: Check which volume is inside the original Solid bounding box.
        
        best_dist = float('inf')
        solid_cand = None
        fluid_cand = None
        
        # We expect at least 2 volumes (Solid, Fluid)
        # If the box completely engulfed the solid (which it should), we get:
        # 1. The solid itself (maybe split if complex, but usually preserved)
        # 2. The fluid (Box minus Solid)
        
        # Wait, if solid was multiple parts, we might have multiple solid vols.
        # For simplicity, assume one solid part or unioned.
        
        # Let's filter by volume location
        for dim, tag in all_vols:
            cog = gmsh.model.occ.getCenterOfMass(dim, tag)
            dist = np.linalg.norm(np.array(cog) - self.solid_orig_centroid)
            
            # A simple heuristic: The fluid volume is HUGE compared to solid.
            # And its COM will be far from solid COM usually.
            
            df_threshold = (self.char_length * 0.5)
            if dist < df_threshold: # Heuristic
                # Likely solid
                self.solid_volume_tag = tag
            else:
                self.fluid_volume_tag = tag
        
        # Fallback if identification fails (e.g. symm)
        if self.solid_volume_tag is None:
            # Sort by volume? Fluid is usually much larger.
            # Sort by Mass (Volume)
            sorted_vols = []
            for dim, tag in all_vols:
                mass = gmsh.model.occ.getMass(dim, tag)
                sorted_vols.append((tag, mass))
            
            sorted_vols.sort(key=lambda x: x[1])
            # Largest is Fluid
            self.fluid_volume_tag = sorted_vols[-1][0]
            # Rest are Solid
            self.solid_volume_tag = sorted_vols[0][0] # Taking the smaller one
            
        logger.info(f"[CHT] Identified Solid Tag: {self.solid_volume_tag}")
        logger.info(f"[CHT] Identified Fluid Tag: {self.fluid_volume_tag}")
        
        # Now identify Faces
        # Get boundaries of Fluid
        fluid_surfs = gmsh.model.getBoundary([(3, self.fluid_volume_tag)], combined=False, oriented=False, recursive=False)
        solid_surfs = gmsh.model.getBoundary([(3, self.solid_volume_tag)], combined=False, oriented=False, recursive=False)
        
        fluid_surf_tags = set(abs(t) for d, t in fluid_surfs)
        solid_surf_tags = set(abs(t) for d, t in solid_surfs)
        
        # INTERFACE: Intersection of Fluid and Solid boundaries
        self.interface_tags = list(fluid_surf_tags.intersection(solid_surf_tags))
        
        # FLUID OUTER BOUNDARIES (Inlet, Outlet, Walls)
        # These are faces in Fluid that are NOT in Solid
        outer_tags = fluid_surf_tags - solid_surf_tags
        
        # Classify Outer Faces based on Normal / Position
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        idx = axis_map.get(self.config.flow_axis, 0)
        
        for tag in outer_tags:
            # Get geometric info
            cog = gmsh.model.occ.getCenterOfMass(2, tag)
            # Normal check is better for inlet/outlet
            # Sample normal at center
            # ...
            # Simple heuristic based on BBox of the Enclosure
            # We know the enclosure bounds.
            
            # Check if face is at the min limit of flow axis -> Inlet
            # (Assuming +FlowDir)
            
            # We can use the COG of the face to check against global bounds
            if self._is_approx(cog[idx], self.solid_orig_bbox[0][idx] - self.char_length * self.config.upstream_mult, tol=1e-3) \
               or self._is_min_bound(tag, idx):
                 # This is likely the upstream face
                 if self.config.flow_direction > 0:
                     self.inlet_tags.append(tag)
                 else:
                     self.outlet_tags.append(tag)
            
            elif self._is_max_bound(tag, idx):
                 if self.config.flow_direction > 0:
                     self.outlet_tags.append(tag)
                 else:
                     self.inlet_tags.append(tag)
            else:
                self.farfield_tags.append(tag)
                
        logger.info(f"[CHT] Interface Faces: {len(self.interface_tags)}")
        logger.info(f"[CHT] Inlet Faces: {len(self.inlet_tags)}")
        logger.info(f"[CHT] Outlet Faces: {len(self.outlet_tags)}")
        
    def _is_approx(self, a, b, tol=1.0):
        return abs(a - b) < tol

    def _is_min_bound(self, tag, axis_idx):
        # Check if face is at global min of the model for that axis
        bb = gmsh.model.getBoundingBox(2, tag)
        model_bb = gmsh.model.getBoundingBox(-1, -1)
        return abs(bb[0 + axis_idx] - model_bb[0 + axis_idx]) < 1e-4

    def _is_max_bound(self, tag, axis_idx):
        bb = gmsh.model.getBoundingBox(2, tag)
        model_bb = gmsh.model.getBoundingBox(-1, -1)
        return abs(bb[3 + axis_idx] - model_bb[3 + axis_idx]) < 1e-4

    def _create_physical_groups(self):
        """Create standard physical groups for solver."""
        
        # Volumes
        p_fluid = gmsh.model.addPhysicalGroup(3, [self.fluid_volume_tag])
        gmsh.model.setPhysicalName(3, p_fluid, "Fluid")
        
        p_solid = gmsh.model.addPhysicalGroup(3, [self.solid_volume_tag])
        gmsh.model.setPhysicalName(3, p_solid, f"Solid_{self.config.solid_material}")
        
        # Faces
        if self.inlet_tags:
            p_inlet = gmsh.model.addPhysicalGroup(2, self.inlet_tags)
            gmsh.model.setPhysicalName(2, p_inlet, "Inlet")
            
        if self.outlet_tags:
            p_outlet = gmsh.model.addPhysicalGroup(2, self.outlet_tags)
            gmsh.model.setPhysicalName(2, p_outlet, "Outlet")
            
        if self.farfield_tags:
            p_wall = gmsh.model.addPhysicalGroup(2, self.farfield_tags)
            gmsh.model.setPhysicalName(2, p_wall, "FarField_Walls")
            
        if self.interface_tags:
            p_int = gmsh.model.addPhysicalGroup(2, self.interface_tags)
            gmsh.model.setPhysicalName(2, p_int, "Interface_Solid_Fluid")

    def _generate_mesh(self, output_file: str):
        """Configure and generate mesh."""
        
        # Mesh Sizing
        # L is char length.
        
        # Solid: Finer, for conduction
        # Fluid: Finer near interface, coarser far away
        
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), self.char_length * 0.1 * self.config.mesh_size_factor)
        
        # Boundary Layer would go here (BoundaryLayer Field)
        
        gmsh.option.setNumber("Mesh.Algorithm3D", 10) # HXT
        gmsh.option.setNumber("Mesh.ElementOrder", self.config.element_order)
        
        logger.info("[CHT] Generating mesh...")
        gmsh.model.mesh.generate(3)
        
        gmsh.write(output_file)
        logger.info(f"[CHT] Mesh saved to {output_file}")

    def _write_region_info(self, output_file: str):
        """Write JSON description of regions for solver setup."""
        data = {
            "analysis_type": "CHT",
            "materials": {
                "solid": self.config.solid_material,
                "fluid": self.config.fluid_material
            },
            "regions": [
                {"name": "Fluid", "type": "fluid", "volume_tags": [self.fluid_volume_tag]},
                {"name": "Solid", "type": "solid", "volume_tags": [self.solid_volume_tag]}
            ],
            "boundaries": [
                {"name": "Inlet", "type": "inlet", "tags": self.inlet_tags, "velocity": self.config.flow_velocity},
                {"name": "Outlet", "type": "outlet", "tags": self.outlet_tags, "pressure": 0.0},
                {"name": "FarField", "type": "wall", "tags": self.farfield_tags, "condition": "slip"},
                {"name": "Interface", "type": "coupled_wall", "tags": self.interface_tags}
            ]
        }
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

def main():
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python cht_template.py <input.step> [output_dir] [axis] [velocity]")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("cht_output")
    
    strategy = CHTTemplateStrategy()
    strategy.run(input_file, output_dir)
    print("Done.")

if __name__ == "__main__":
    main()
