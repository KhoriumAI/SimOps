"""
CFD Meshing Strategy with Boundary Layer Support
=================================================

Generates CFD-ready meshes with boundary layers (inflation layers/prism cells)
for thermal and airflow analysis.

Key Features:
- Gmsh BoundaryLayer field for prism/inflation layer generation
- Auto-applies to all wall surfaces
- Linear elements only (Order 1)
- Optimized for thermal/CFD simulation
- Robust fallback to Distance+Threshold fields if BL generation fails

Author: SimOps Team
Date: 2024
"""

import gmsh
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

from core.logging.sim_logger import SimLogger
logger = SimLogger("CFDStrategy")

@dataclass
class CFDMeshConfig:
    """Configuration for CFD mesh generation"""
    
    # Boundary layer settings
    num_layers: int = 5                 # Number of inflation layers
    first_layer_height: float = None    # Auto-calculated if None
    growth_rate: float = 1.2            # Layer growth ratio
    create_prisms: bool = True          # True for prisms, False for tets
    
    # Global mesh settings
    mesh_size_factor: float = 1.0       # Multiplier for auto mesh sizing
    min_mesh_size: float = None         # Minimum element size (auto if None)
    max_mesh_size: float = None         # Maximum element size (auto if None)
    
    # Optimization
    smoothing_steps: int = 10           # Mesh smoothing iterations
    optimize_netgen: bool = True        # Use Netgen optimizer
    
    # Virtual Wind Tunnel (for external CFD/aerodynamics)
    virtual_wind_tunnel: bool = False   # Enable automatic domain enclosure
    
    # Output
    mesh_format: str = "msh"            # Output format: msh, vtk, stl


class CFDMeshStrategy:
    """
    CFD-focused meshing with boundary layers for thermal/airflow analysis.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.geometry_info: Dict = {}
        self.is_stl = False
        
    def _log(self, message: str):
        if self.verbose:
            logger.info(message)
            
    def generate_cfd_mesh(self, 
                          cad_file: str, 
                          output_file: str,
                          config: Optional[CFDMeshConfig] = None) -> Tuple[bool, Dict]:
        config = config or CFDMeshConfig()
        
        self._log("=" * 70)
        self._log("CFD MESH STRATEGY - Boundary Layer Generation")
        self._log("=" * 70)
        self._log(f"Input:  {cad_file}")
        self._log(f"Output: {output_file}")
        self._log(f"Layers: {config.num_layers}, Growth: {config.growth_rate}")
        self._log("")
        
        try:
            # Initialize Gmsh
            if not gmsh.isInitialized():
                gmsh.initialize()
            gmsh.clear()
            gmsh.option.setNumber("General.Terminal", 1 if self.verbose else 0)
            
            # Step 1: Load CAD
            self._log("[1/5] Loading CAD geometry...")
            self._load_cad(cad_file)
            
            # Step 2: Analyze geometry
            self._log("[2/5] Analyzing geometry...")
            self._analyze_geometry(config)

            # Step 1.5: Virtual Wind Tunnel (External Aero)
            # Check conditions for wind tunnel generation
            try:
                from core.geometry.wind_tunnel import VirtualWindTunnel
                has_wind_tunnel_class = True
            except ImportError:
                has_wind_tunnel_class = False
                
            self._log(f"[Debug] Wind Tunnel Check: enabled={config.virtual_wind_tunnel}, have_class={has_wind_tunnel_class}, is_stl={self.is_stl}")
            
            wind_tunnel_metadata = None
            if config.virtual_wind_tunnel and has_wind_tunnel_class and not self.is_stl:
                self._log("[1.5/5] Generating Virtual Wind Tunnel Domain...")
                try:
                    # Get object volumes before wind tunnel creation
                    volumes = gmsh.model.getEntities(dim=3)
                    object_volumes = [tag for dim, tag in volumes]
                    
                    # Create wind tunnel (this will tag boundaries)
                    wt = VirtualWindTunnel(verbose=self.verbose)
                    wind_tunnel_metadata = wt.create_domain(object_volumes)
                    
                    self._log(f"  [OK] Fluid domain created")
                    self._log(f"  L_char for Reynolds: {wind_tunnel_metadata.get('L_char', 'N/A')}")
                    
                    # Wind tunnel handles tagging, so skip auto_tag_geometry
                    self._log("[2.5/5] Skipping Golden Template Tags (Wind Tunnel handles boundaries)")
                except Exception as e:
                    self._log(f"  [Warning] Wind Tunnel generation failed: {e}")
                    self._log(f"  [Warning] Falling back to standard tagging")
                    # Fall through to auto-tag
                    self._log("[2.5/5] Applying Golden Template Tags...")
                    self._auto_tag_geometry()
            else:
                self._log("[1.5/5] Virtual Wind Tunnel SKIPPED (disabled or conditions not met)")
                # Step 2.5: Auto-Tag for Golden Template (Day 1)
                self._log("[2.5/5] Applying Golden Template Tags...")
                self._auto_tag_geometry()
            
            # Step 3: Set up boundary layers
            self._log("[3/5] Configuring boundary layers...")
            bl_success = self._setup_boundary_layers(config)
            if not bl_success:
                self._log("[!] BoundaryLayer field failed or N/A. Falling back to Distance+Threshold.")
                self._setup_fallback_fields(config)
            
            # Step 4: Configure mesh parameters
            self._log("[4/5] Setting mesh parameters...")
            self._configure_mesh_params(config)
            
            # Step 5: Generate mesh
            self._log("[5/5] Generating 3D mesh...")
            gmsh.model.mesh.generate(3)
            
            # Optimize
            if config.optimize_netgen:
                self._log("Optimizing mesh (Netgen)...")
                gmsh.model.mesh.optimize("Netgen")
            
            if config.smoothing_steps > 0:
                self._log(f"Smoothing mesh ({config.smoothing_steps} iterations)...")
                gmsh.option.setNumber("Mesh.Smoothing", config.smoothing_steps)
                
            # Get statistics
            stats = self._get_mesh_stats()
            self._log("")
            self._log("=" * 70)
            self._log("MESH STATISTICS")
            self._log("=" * 70)
            self._log(f"  Nodes:       {stats['num_nodes']:,}")
            self._log(f"  Tetrahedra:  {stats['num_tets']:,}")
            self._log(f"  Prisms:      {stats['num_prisms']:,}")
            self._log(f"  Hexahedra:   {stats['num_hexes']:,}")
            self._log(f"  Physical Groups: {stats['physical_groups']}")
            self._log("")
            
            # Write output
            self._log(f"Writing mesh to: {output_file}")
            
            # [Critical] Force MSH 2.2 for OpenFOAM compatibility
            # gmshToFoam often results in "bad stream" or "unknown element" with MSH 4.1
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
            
            gmsh.write(output_file)
            
            # Also write VTK
            vtk_file = str(Path(output_file).with_suffix('.vtk'))
            gmsh.write(vtk_file)
            stats['vtk_file'] = vtk_file
            
            # Save wind tunnel metadata for Reynolds calculation
            if wind_tunnel_metadata:
                stats['wind_tunnel'] = wind_tunnel_metadata
                # Write to JSON for solver to read
                import json
                meta_file = Path(output_file).parent / "mesh_metadata.json"
                with open(meta_file, 'w') as f:
                    json.dump({
                        'wind_tunnel': {
                            'L_char': wind_tunnel_metadata.get('L_char', 1.0),
                            'diagonal': wind_tunnel_metadata.get('diagonal', 1.0),
                        },
                        'mesh_stats': {
                            'num_nodes': stats['num_nodes'],
                            'num_tets': stats['num_tets'],
                            'num_prisms': stats['num_prisms'],
                        }
                    }, f, indent=2)
                self._log(f"  Wrote mesh metadata to: {meta_file}")
            
            self._log("[OK] CFD mesh generation complete!")
            self._log("=" * 70)
            
            return True, stats
            
        except Exception as e:
            self._log(f"[ERROR] Mesh generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False, {"error": str(e)}
            
        finally:
            if gmsh.isInitialized():
                gmsh.finalize()
                
    def _load_cad(self, cad_file: str):
        path = Path(cad_file)
        if not path.exists():
            raise FileNotFoundError(f"CAD file not found: {cad_file}")
            
        ext = path.suffix.lower()
        if ext == '.stl':
            self.is_stl = True
            self._log("  [!] STL input detected. Using robust fallback mode.")
            gmsh.merge(str(path))
            
            # Create surface loop and volume for STL
            # Only do this if strictly necessary - gmsh usually handles STL meshing automatically 
            # if we request 3D generation.
            
        elif ext in ['.step', '.stp', '.iges', '.igs', '.brep']:
            gmsh.model.occ.importShapes(str(path))
            gmsh.model.occ.synchronize()
        else:
            gmsh.merge(str(path))
            
        # Ensure we have entities
        gmsh.model.occ.synchronize()
        
    def _analyze_geometry(self, config: CFDMeshConfig):
        # Compute bounding box
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
        diagonal = np.sqrt((xmax-xmin)**2 + (ymax-ymin)**2 + (zmax-zmin)**2)
        
        # Get dimensions
        dx, dy, dz = xmax-xmin, ymax-ymin, zmax-zmin
        dims = [d for d in [dx, dy, dz] if d > 1e-9]
        min_dim = min(dims) if dims else diagonal * 0.1
        
        self.geometry_info = {
            'bbox': (xmin, ymin, zmin, xmax, ymax, zmax),
            'diagonal': diagonal,
            'min_dim': min_dim
        }
        
        self._log(f"  Diagonal: {diagonal:.2f}")
        self._log(f"  Min feature size (approx): {min_dim:.2f}")
        
        # Auto-calculate mesh sizes
        # Modified for coarser sanity checks (target 5-20k tets instead of 500k)
        if config.min_mesh_size is None:
            config.min_mesh_size = diagonal / 30.0  # Was / 100.0
        if config.max_mesh_size is None:
            config.max_mesh_size = diagonal / 10.0  # Was / 20.0
            
        # ADAPTIVE FIRST LAYER HEIGHT
        if config.first_layer_height is None and config.num_layers > 0:
            # Base it on the smallest dimension to avoid collapsing thin parts
            # But keep it reasonable for boundary layers (y+ approx)
            
            # Start with 1/250 diameter standard (coarser for testing)
            base_h = diagonal / 250.0
            
            # Clamp based on min dimension (ensure at least 5 layers fit in 1/10th the min thickness)
            # This is critical to prevent HXT 3D mesh failures due to self-intersecting BLs
            max_safe_h = (min_dim / 10.0) / config.num_layers
            
            config.first_layer_height = min(base_h, max_safe_h)
            
        elif config.first_layer_height is None:
            # No boundary layers - set a reasonable default
            config.first_layer_height = diagonal / 100.0
            
        self._log(f"  Auto mesh size: {config.min_mesh_size:.4f} to {config.max_mesh_size:.4f}")
        self._log(f"  Adaptive first layer: {config.first_layer_height:.6f}")

        # Tag Volume
        volumes = gmsh.model.getEntities(dim=3)
        if volumes:
            vol_tags = [tag for dim, tag in volumes]
            pv = gmsh.model.addPhysicalGroup(3, vol_tags)
            gmsh.model.setPhysicalName(3, pv, "Material_Aluminum")
            self._log(f"  [Tag] Assigned 'Material_Aluminum' to {len(vol_tags)} volumes")

    def _apply_tagging_rules(self):
        """Apply custom tagging rules provided in config"""
        gmsh.model.removePhysicalGroups()
        
        if not self.tagging_rules:
            return
            
        self._log(f"[Tagging] Processing {len(self.tagging_rules)} custom rules...")
        
        surfaces = gmsh.model.getEntities(dim=2)
        remaining_surfaces = set(tag for dim, tag in surfaces)
        used_surfaces = set()
        
        # BBox for reference
        xmin, ymin, zmin, xmax, ymax, zmax = self.geometry_info['bbox']
        diag = self.geometry_info['diagonal']
        
        # Helper to check selector
        def match_selector(s_tag, selector):
            # Handle Pydantic or dict
            s_type = getattr(selector, 'type', selector.get('type') if isinstance(selector, dict) else '')
            tol_val = getattr(selector, 'tolerance', selector.get('tolerance', 0.001) if isinstance(selector, dict) else 0.001)
            
            # Absolute tolerance based on diagonal if relative logic needed, but schema implies absolute?
            # Schema says "tolerance: float = 0.001". Likely absolute or relative to unit?
            # Let's assume absolute but scale by diagonal if very small? 
            # safe logic: tol = tol_val * diag
            tol = tol_val * diag
            
            s_bb = gmsh.model.getBoundingBox(2, s_tag)
            s_zmin, s_zmax = s_bb[2], s_bb[5]
            
            if s_type == 'z_min':
                # Surface min Z is close to global min Z AND it is flat (max Z is also close)
                # If s_zmax is far, it's a side wall touching the bottom.
                return (abs(s_zmin - zmin) < tol) and (abs(s_zmax - zmin) < tol)
                
            elif s_type == 'z_max':
                return (abs(s_zmax - zmax) < tol) and (abs(s_zmin - zmax) < tol)
                
            return False

        for rule in self.tagging_rules:
            tag_name = getattr(rule, 'tag_name', rule.get('tag_name') if isinstance(rule, dict) else '')
            selector = getattr(rule, 'selector', rule.get('selector') if isinstance(rule, dict) else {})
            s_type = getattr(selector, 'type', selector.get('type') if isinstance(selector, dict) else '')
            
            matched_tags = []
            
            if s_type == 'all_remaining':
                matched_tags = list(remaining_surfaces)
            else:
                for tag in list(remaining_surfaces): # Check available surfaces
                    if match_selector(tag, selector):
                        matched_tags.append(tag)
            
            if matched_tags:
                p = gmsh.model.addPhysicalGroup(2, matched_tags)
                gmsh.model.setPhysicalName(2, p, tag_name)
                self._log(f"  [Tag] Rule '{tag_name}': Assigned {len(matched_tags)} surfaces")
                
                # Mark as used (remove from remaining)
                for t in matched_tags:
                    if t in remaining_surfaces:
                        remaining_surfaces.remove(t)
                        used_surfaces.add(t)
            else:
                self._log(f"  [Tag] Rule '{tag_name}': No matching surfaces found")
                
        # Always tag volume
        volumes = gmsh.model.getEntities(dim=3)
        if volumes:
            vol_tags = [tag for dim, tag in volumes]
            pv = gmsh.model.addPhysicalGroup(3, vol_tags)
            gmsh.model.setPhysicalName(3, pv, "Fluid_Domain") # Default to Fluid for CFD
            self._log(f"  [Tag] Assigned 'Fluid_Domain' to {len(vol_tags)} volumes")

    def _auto_tag_geometry(self):
        """
        Implements 'Golden Template' Auto-Tagging.
        Or uses Custom Rules if provided.
        """
        if hasattr(self, 'tagging_rules') and self.tagging_rules:
            self._apply_tagging_rules()
            return

        # Clear existing physical groups to avoid duplicates
        gmsh.model.removePhysicalGroups()
        
        # Get BBox Z-min
        z_min = self.geometry_info['bbox'][2]
        z_tolerance = 1e-3 * self.geometry_info['diagonal'] # 0.1% tolerance
        
        surfaces = gmsh.model.getEntities(dim=2)
        heat_source_surfaces = []
        wall_surfaces = []
        
        for dim, tag in surfaces:
            # Check bounding box of the surface
            s_xmin, s_ymin, s_zmin, s_xmax, s_ymax, s_zmax = gmsh.model.getBoundingBox(dim, tag)
            
            # If the surface is essentially flat on the Z-min plane
            # Strategy: Check if the surface's center or min points align with global Z-min
            
            # A strict check: absolute z-min of surface must be close to global z-min
            if abs(s_zmin - z_min) < z_tolerance:
                 # It touches the bottom. Is it flat? 
                 # If s_zmax is also close to z_min, it's a flat bottom face.
                 if abs(s_zmax - z_min) < z_tolerance:
                     heat_source_surfaces.append(tag)
                 else:
                     # It's a side wall that touches bottom.
                     wall_surfaces.append(tag)
            else:
                 wall_surfaces.append(tag)
                 
        # Create Physical Groups
        if heat_source_surfaces:
            p1 = gmsh.model.addPhysicalGroup(2, heat_source_surfaces)
            gmsh.model.setPhysicalName(2, p1, "BC_HeatSource")
            self._log(f"  [Tag] Assigned 'BC_HeatSource' to {len(heat_source_surfaces)} surfaces (Z={z_min:.2f})")
        else:
            self._log("  [Warning] No HeatSource surfaces found at Z-min!")
            
        if wall_surfaces:
            p2 = gmsh.model.addPhysicalGroup(2, wall_surfaces)
            gmsh.model.setPhysicalName(2, p2, "BC_Wall_Adiabatic")
            self._log(f"  [Tag] Assigned 'BC_Wall_Adiabatic' to {len(wall_surfaces)} surfaces")
            
        # Tag Volume
        volumes = gmsh.model.getEntities(dim=3)
        if volumes:
            vol_tags = [tag for dim, tag in volumes]
            pv = gmsh.model.addPhysicalGroup(3, vol_tags)
            gmsh.model.setPhysicalName(3, pv, "Material_Aluminum")
            self._log(f"  [Tag] Assigned 'Material_Aluminum' to {len(vol_tags)} volumes")
        
    def _setup_boundary_layers(self, config: CFDMeshConfig) -> bool:
        if self.is_stl:
            return False # Gmsh BoundaryLayer field needs curves/surfaces from CAD kernel usually
            
        try:
            surfaces = gmsh.model.getEntities(dim=2)
            if not surfaces:
                return False
                
            surf_tags = [tag for dim, tag in surfaces]
            
             # Create BoundaryLayer field
            field_tag = gmsh.model.mesh.field.add("BoundaryLayer")
            
            # Robust Option Setting (GMSH API varies)
            try:
                gmsh.model.mesh.field.setNumbers(field_tag, "SurfacesList", surf_tags)
            except Exception:
                try:
                    gmsh.model.mesh.field.setNumbers(field_tag, "Surfaces", surf_tags)
                except Exception as e:
                    self._log(f"  [Warning] Could not set surfaces for BL: {e}")
                    
            gmsh.model.mesh.field.setNumber(field_tag, "Size", config.first_layer_height)
            gmsh.model.mesh.field.setNumber(field_tag, "Ratio", config.growth_rate)
            
            # Calculate total BL thickness from parameters
            # h + h*r + h*r^2 ... = h * (r^n - 1) / (r - 1)
            if abs(config.growth_rate - 1.0) < 1e-4:
                total_thickness = config.first_layer_height * config.num_layers
            else:
                total_thickness = config.first_layer_height * (config.growth_rate ** config.num_layers - 1) / (config.growth_rate - 1)
                
            gmsh.model.mesh.field.setNumber(field_tag, "Thickness", total_thickness)
            gmsh.model.mesh.field.setNumber(field_tag, "NbLayers", config.num_layers)
            
            if config.create_prisms:
                try:
                    gmsh.model.mesh.field.setNumber(field_tag, "Quads", 0)
                except:
                    pass
                    
            gmsh.model.mesh.field.setAsBoundaryLayer(field_tag)
            self._log(f"  [OK] Gmsh BoundaryLayer field configured (Target thickness: {total_thickness:.3f})")
            return True
            
        except Exception as e:
            self._log(f"  Warning: BoundaryLayer setup failed: {e}")
            return False

    def _setup_fallback_fields(self, config: CFDMeshConfig):
        """Fallback to Distance + Threshold fields for refinement without guaranteed prisms"""
        surfaces = gmsh.model.getEntities(dim=2)
        if not surfaces:
             # Try to get surfaces from 3D...
             return
             
        surf_tags = [tag for dim, tag in surfaces]
        
        # Calculate thickness
        if abs(config.growth_rate - 1.0) < 1e-4:
            total_thickness = config.first_layer_height * config.num_layers
        else:
            total_thickness = config.first_layer_height * (config.growth_rate ** config.num_layers - 1) / (config.growth_rate - 1)

        # 1. Distance field
        dist_field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist_field, "SurfacesList", surf_tags)
        
        # 2. Threshold field
        thresh_field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(thresh_field, "InField", dist_field)
        
        # Size inside the boundary layer zone
        gmsh.model.mesh.field.setNumber(thresh_field, "SizeMin", config.first_layer_height)
        gmsh.model.mesh.field.setNumber(thresh_field, "SizeMax", config.max_mesh_size)
        
        # DistMin = where size starts increasing (0)
        # DistMax = where size reaches MaxSize (total thickness * 2 usually for smooth transition)
        gmsh.model.mesh.field.setNumber(thresh_field, "DistMin", 0)
        gmsh.model.mesh.field.setNumber(thresh_field, "DistMax", total_thickness * 2.0)
        
        gmsh.model.mesh.field.setAsBackgroundMesh(thresh_field)
        self._log(f"  [Fallback] Configured Distance+Threshold sizing (Transition dist: {total_thickness*2.0:.3f})")

    def _configure_mesh_params(self, config: CFDMeshConfig):
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.MeshSizeMin", config.min_mesh_size * config.mesh_size_factor)
        gmsh.option.setNumber("Mesh.MeshSizeMax", config.max_mesh_size * config.mesh_size_factor)
        
        # Enable parallel meshing with HXT for speed
        import os
        # Cap threads to 8 to prevent system instability on high-core machines
        num_threads = min(8, max(1, os.cpu_count() or 4))
        gmsh.option.setNumber("General.NumThreads", num_threads)
        gmsh.option.setNumber("Mesh.MaxNumThreads1D", num_threads)
        gmsh.option.setNumber("Mesh.MaxNumThreads2D", num_threads)
        gmsh.option.setNumber("Mesh.MaxNumThreads3D", num_threads)
        
        # HXT 3D algorithm is parallel and robust
        gmsh.option.setNumber("Mesh.Algorithm", 6)      # Frontal-Delaunay 2D
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)   # HXT (Parallel, Robust)
        
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeThreshold", 0.3)

    def _get_mesh_stats(self) -> Dict:
        stats = {
            'num_nodes': 0,
            'num_tets': 0,
            'num_prisms': 0,
            'num_hexes': 0,
            'num_pyramids': 0,
            'physical_groups': {}
        }
        
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        stats['num_nodes'] = len(node_tags)
        
        elem_types, elem_tags, _ = gmsh.model.mesh.getElements(dim=3)
        for i, etype in enumerate(elem_types):
            count = len(elem_tags[i])
            if etype == 4: stats['num_tets'] += count
            elif etype == 5: stats['num_hexes'] += count
            elif etype == 6: stats['num_prisms'] += count
            elif etype == 7: stats['num_pyramids'] += count
            
        # Get Physical Groups
        phys_groups = gmsh.model.getPhysicalGroups()
        for dim, tag in phys_groups:
            name = gmsh.model.getPhysicalName(dim, tag)
            stats['physical_groups'][name] = tag
            
        return stats

def generate_cfd_mesh(cad_file: str, 
                      output_file: str,
                      num_layers: int = 5,
                      growth_rate: float = 1.2,
                      verbose: bool = True) -> Tuple[bool, Dict]:
    config = CFDMeshConfig(num_layers=num_layers, growth_rate=growth_rate)
    strategy = CFDMeshStrategy(verbose=verbose)
    return strategy.generate_cfd_mesh(cad_file, output_file, config)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python cfd_strategy.py <cad_file> [output]")
        sys.exit(1)
    
    cad = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    if not out: out = str(Path(cad).with_suffix('')) + "_cfd.msh"
    
    success, stats = generate_cfd_mesh(cad, out)
    if not success: sys.exit(1)
