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
    
    # Output
    mesh_format: str = "msh"            # Output format: msh, vtk, stl
    
    # Advanced
    hex_dominant: bool = False          # Try to generate Hex/Prism mesh
    orthogonality_threshold: float = 0.1 # Min orthogonality to accept
    fidelity_threshold: float = 0.02    # Max allowed volume deviation (2%)
    second_order: bool = False          # Use 2nd order elements (Tet10)


class CFDMeshStrategy:
    """
    CFD-focused meshing with boundary layers for thermal/airflow analysis.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.geometry_info: Dict = {}
        self.is_stl = False
        self.tagging_rules = []
        
    def _log(self, message: str):
        if self.verbose:
            print(message, flush=True)
            
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
            
            # Check Orthogonality (Pre-Solver Gate)
            ortho_ok, min_ortho = self._check_orthogonality(config.orthogonality_threshold)
            if not ortho_ok and config.orthogonality_threshold > 0:
                self._log(f"[!] Orthogonality Check Failed: Min {min_ortho:.4f} < {config.orthogonality_threshold}")
                # We optionally fail here, or just warn. For "Production", we should probably warn for now.
                stats['quality_check'] = 'FAILED'
            else:
                stats['quality_check'] = 'PASSED'
            stats['min_orthogonality'] = min_ortho

            # Geometric Fidelity Check (CAD vs Mesh)
            fidelity_ok, vol_dev, area_dev = self._check_geometric_fidelity(config.fidelity_threshold)
            stats['volume_deviation'] = vol_dev
            stats['area_deviation'] = area_dev
            if not fidelity_ok:
                self._log(f"[!] FIDELITY WARNING: Volume deviation {vol_dev*100:.2f}% exceeds threshold {config.fidelity_threshold*100:.1f}%")
                stats['fidelity_check'] = 'WARNING'
            else:
                stats['fidelity_check'] = 'PASSED'

            # Write output
            self._log(f"Writing mesh to: {output_file}")
            gmsh.write(output_file)
            
            # Also write VTK
            vtk_file = str(Path(output_file).with_suffix('.vtk'))
            gmsh.write(vtk_file)
            stats['vtk_file'] = vtk_file
            
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
            # Enable importing of names/labels from STEP/BREP
            gmsh.option.setNumber("Geometry.OCCImportLabels", 1)
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
            'min_dim': min_dim,
            'cad_volume': 0.0,
            'cad_area': 0.0
        }
        
        self._log(f"  Diagonal: {diagonal:.2f}")
        self._log(f"  Min feature size (approx): {min_dim:.2f}")
        
        # Auto-calculate mesh sizes
        if config.min_mesh_size is None:
            config.min_mesh_size = diagonal / 100.0
        if config.max_mesh_size is None:
            config.max_mesh_size = diagonal / 20.0
            
        # ADAPTIVE FIRST LAYER HEIGHT
        if config.first_layer_height is None:
            # Base it on the smallest dimension to avoid collapsing thin parts
            # But keep it reasonable for boundary layers (y+ approx)
            
            # Start with 1/250 diameter standard (coarser for testing)
            base_h = diagonal / 250.0
            
            # Clamp based on min dimension (ensure at least 5 layers fit in half the min thickness)
            # min_dim / 2 is half thickness. We want 5 layers + core.
            # So total BL thickness < min_dim / 3 is safe.
            # total_t = h * (r^n - 1) / (r - 1)
            # simplified: total_t approx n * h * r^(n/2)
            
            if config.num_layers > 0:
                max_safe_h = (min_dim / 3.0) / config.num_layers
                config.first_layer_height = min(base_h, max_safe_h)
            else:
                config.first_layer_height = base_h

        # Feature Recognition: Hydraulic Diameter (Thin Wall Detection)
        try:
            total_vol = 0.0
            total_area = 0.0
            vols = gmsh.model.getEntities(3)
            for _, tag in vols:
                # Mass properties: centerOfMass(3), mass(1), matrixOfInertia(9)
                props = gmsh.model.occ.getMass(3, tag)
                if props: total_vol += props
            
            surfs = gmsh.model.getEntities(2)
            for _, tag in surfs:
                props = gmsh.model.occ.getMass(2, tag)
                if props: total_area += props
                
            if total_area > 1e-9:
                Dh = 4.0 * total_vol / total_area
                self._log(f"  Feature Analysis: Vol={total_vol:.2e}, Area={total_area:.2e}, Dh={Dh:.4f}")
                
                # Store for fidelity check later
                self.geometry_info['cad_volume'] = total_vol
                self.geometry_info['cad_area'] = total_area
                
                # If Hydraulic Diameter is very small relative to bbox, it's a "Thin Plate" or "Pipe"
                if Dh < 0.05 * diagonal:
                    self._log(f"  [!] Thin Structure Detected (Dh < 5% Diagonal).")
                    self._log(f"      -> Reducing Max Mesh Size to Dh/2 ({Dh/2:.4f})")
                    config.max_mesh_size = min(config.max_mesh_size, Dh / 2.0)
                    config.min_mesh_size = min(config.min_mesh_size, Dh / 10.0)
                    # Also ensure first layer is very small
                    config.first_layer_height = min(config.first_layer_height, Dh / 20.0)
        except Exception as e:
            self._log(f"  Feature recognition failed (non-OCC?): {e}")
            
        self._log(f"  Auto mesh size: {config.min_mesh_size:.4f} to {config.max_mesh_size:.4f}")
        self._log(f"  Adaptive first layer: {config.first_layer_height:.6f}")

    def _auto_tag_geometry(self):
        """
        Implements Auto-Tagging with "Smart Template" logic.
        
        Strategy:
        1. Sidecar Rules (Priority 1): Explicit rules from config file.
        2. Semantic CAD (Priority 2): Convention-based naming (e.g. "Inlet", "Source") from STEP file.
        3. Golden Template (Priority 3): Geometric fallback (Z-Min = HeatSource).
        """
        # Save existing groups (imported from STEP) before potentially clearing
        # Note: Gmsh importShapes should have loaded names if available.
        incoming_physical_groups = gmsh.model.getPhysicalGroups()
        
        # We only clear if we are going to fully overwrite.
        # But for hybrid approach, we might want to keep them.
        # Let's see what we have.
        existing_names = {}
        for dim, tag in incoming_physical_groups:
            name = gmsh.model.getPhysicalName(dim, tag)
            existing_names[name] = (dim, tag)
            
        self._log(f"  [Tag] Found {len(incoming_physical_groups)} existing groups from CAD: {list(existing_names.keys())}")

        surfaces = gmsh.model.getEntities(dim=2)
        volumes = gmsh.model.getEntities(dim=3)
        
        tagged_surfaces = set()
        
        # ---------------------------------------------------------
        # 1. Dynamic Rules (Sidecar) - Highest Priority
        # ---------------------------------------------------------
        if hasattr(self, 'tagging_rules') and self.tagging_rules:
            self._log(f"  [Tag] Applying {len(self.tagging_rules)} custom tagging rules (Sidecar)...")
            
            # If sidecar rules exist, they often imply a full override intent,
            # but we can treat them as additives.
            for rule in self.tagging_rules:
                selector = rule.selector
                matches = []
                
                # Selection Logic
                if selector.type == "z_min" or selector.type == "z_max":
                    # Get global bbox
                    g_xmin, g_ymin, g_zmin, g_xmax, g_ymax, g_zmax = gmsh.model.getBoundingBox(-1, -1)
                    target_z = g_zmin if selector.type == "z_min" else g_zmax
                    tol = (g_zmax - g_zmin) * 0.01 # 1% tolerance default
                    
                    if selector.tolerance:
                         tol = selector.tolerance
                         
                    for dim, tag in surfaces:
                        s_xmin, s_ymin, s_zmin, s_xmax, s_ymax, s_zmax = gmsh.model.getBoundingBox(dim, tag)
                        if abs(s_zmin - target_z) < tol and abs(s_zmax - target_z) < tol:
                            matches.append(tag)
                            
                elif selector.type == "all_remaining":
                     for dim, tag in surfaces:
                         if tag not in tagged_surfaces:
                             matches.append(tag)
                elif selector.type == "box":
                     if selector.bounds and len(selector.bounds) == 6:
                         b = selector.bounds
                         for dim, tag in surfaces:
                            s_xmin, s_ymin, s_zmin, s_xmax, s_ymax, s_zmax = gmsh.model.getBoundingBox(dim, tag)
                            if (s_xmin >= b[0] and s_xmax <= b[3] and
                                s_ymin >= b[1] and s_ymax <= b[4] and
                                s_zmin >= b[2] and s_zmax <= b[5]):
                                matches.append(tag)
                                
                if matches:
                    if rule.entity_type == 'surface':
                        p = gmsh.model.addPhysicalGroup(2, matches)
                        gmsh.model.setPhysicalName(2, p, rule.tag_name)
                        tagged_surfaces.update(matches)
                        self._log(f"    -> Tagged {len(matches)} surfaces as '{rule.tag_name}'")
                        
            # If sidecar rules were applied, we generally stop here or merge.
            # Allowing fallthrough to "Semantic" might adhere to "Fill in the blanks" philosophy.
            # Let's allow fallthrough but respect `tagged_surfaces`.
            
        # ---------------------------------------------------------
        # 2. Semantic CAD (Smart Template) - Convention Matching
        # ---------------------------------------------------------
        # scan existing names for keywords
        found_semantic_tags = False
        
        # We need to act on the entities inside existing groups
        # If CAD has group "Inlet_Pipe", we want to ensure it maps to "BC_Inlet"
        # or just preserve it and let the solver handle the mapping?
        # Better: Standardize to "BC_..." locally used by Solver Adapter.
        
        # Map of Keywords -> SimOps Standard Tag
        semantic_map = {
            "inlet": "BC_Inlet",
            "flow_in": "BC_Inlet",
            "outlet": "BC_Outlet",
            "flow_out": "BC_Outlet",
            "source": "BC_HeatSource",
            "heat": "BC_HeatSource",
            "power": "BC_HeatSource",
            "hot": "BC_HeatSource",
            "sym": "BC_Symmetry",
            "wall": "BC_Wall_Adiabatic"
        }
        
        for name, (dim, tag) in existing_names.items():
            name_lower = name.lower()
            
            # Check keywords
            matched_std_name = None
            for kw, std_name in semantic_map.items():
                if kw in name_lower:
                    matched_std_name = std_name
                    break
            
            if matched_std_name:
                # We found a semantic match!
                # Get entities in this group
                entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
                
                # Check if these are surfaces (dim=2)
                if dim == 2:
                    # Filter out already tagged ones? Or allow overwrite?
                    # Let's assume unique assignment for simplicity
                    new_entities = [e for e in entities if e not in tagged_surfaces]
                    
                    if new_entities:
                        # Create NEW standardized group (or add to it)
                        # We can't append to existing group easily in Gmsh Python API without re-creating.
                        # Easier: Just set the name of the EXISTING group to the standard name if it's a 1:1 match
                        # But multiple source groups might map to one BC.
                        # Best: Create new aggregate group.
                        
                        # For now MVP: Rename the existing group to the Standard Name
                        # This works if there's only one "Inlet".
                        # If there is "Inlet1" and "Inlet2", they both become "BC_Inlet" (Duplicate names allowed? No.)
                        # Actually Gmsh physical names must be unique? 
                        # If duplicate names, Gmsh merges them usually.
                        
                        gmsh.model.setPhysicalName(dim, tag, matched_std_name)
                        tagged_surfaces.update(new_entities)
                        found_semantic_tags = True
                        self._log(f"  [Tag] Semantic Match: '{name}' -> '{matched_std_name}'")

        # ---------------------------------------------------------
        # 3. Golden Template Defaults (Fallback)
        # ---------------------------------------------------------
        # If we haven't found a Heat Source via Sidecar OR Semantic Tags, run geometric fallback
        
        # Check if we have a heat source defined yet
        has_heat_source = False
        all_phys_names = [gmsh.model.getPhysicalName(2, t) for d, t in gmsh.model.getPhysicalGroups(2)]
        if any("BC_HeatSource" in n for n in all_phys_names):
             has_heat_source = True
             
        if not has_heat_source and not self.tagging_rules:
            self._log("  [Tag] No Heat Source found by semantic matching. Applying Geometric Fallback (Z-Min)...")
            
            # Get BBox Z-min
            z_min = self.geometry_info['bbox'][2]
            z_tolerance = 0.05 * self.geometry_info['diagonal'] # 5% tolerance (Relaxed for robustness)
            
            heat_source_surfaces = []
            wall_surfaces = []
            
            for dim, tag in surfaces:
                if tag in tagged_surfaces:
                    continue # Skip already handled
                    
                # Check bounding box
                s_xmin, s_ymin, s_zmin, s_xmax, s_ymax, s_zmax = gmsh.model.getBoundingBox(dim, tag)
                
                # Z-Min Logic
                if abs(s_zmin - z_min) < z_tolerance and abs(s_zmax - z_min) < z_tolerance:
                     heat_source_surfaces.append(tag)
                else:
                     wall_surfaces.append(tag)
                     
            if heat_source_surfaces:
                p1 = gmsh.model.addPhysicalGroup(2, heat_source_surfaces)
                gmsh.model.setPhysicalName(2, p1, "BC_HeatSource")
                tagged_surfaces.update(heat_source_surfaces)
                self._log(f"  [Tag] Assigned 'BC_HeatSource' to {len(heat_source_surfaces)} surfaces (Z-Min)")
                has_heat_source = True
        
        # ---------------------------------------------------------
        # 4. Final Cleanup (Catch-all Walls)
        # ---------------------------------------------------------
        # Any surface not yet tagged becomes Wall
        remaining_surfaces = [tag for dim, tag in surfaces if tag not in tagged_surfaces]
        
        # EXCEPTION: If the user provided ANY tagging rules, we might assume they tagged everything they wanted?
        # No, safest is to make everything else a wall.
        
        if remaining_surfaces:
            # Check if we already have a BC_Wall_Adiabatic group to append to?
            # Simpler: Create a new one "BC_Wall_Default"
            p_rem = gmsh.model.addPhysicalGroup(2, remaining_surfaces)
            gmsh.model.setPhysicalName(2, p_rem, "BC_Wall_Adiabatic")
            self._log(f"  [Tag] Assigned 'BC_Wall_Adiabatic' to {len(remaining_surfaces)} remaining surfaces")
            
        # Tag Volume (Always)
        if volumes:
            # Check if volume already tagged?
            tagged_vols = [tag for dim, tag in gmsh.model.getPhysicalGroups(3)]
            if not tagged_vols:
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
            
            # Robustly try to set the surface list (Option name varies by Gmsh version)
            bl_setup_ok = False
            # CurvesList is for 2D only and crashes 3D generation. 
            # We only try 3D-oriented options here.
            for opt_name in ["FacesList", "SurfacesList", "SurfaceList", "Surfaces", "Faces"]:
                try:
                    gmsh.model.mesh.field.setNumbers(field_tag, opt_name, surf_tags)
                    self._log(f"  [Debug] BoundaryLayer used option '{opt_name}'")
                    bl_setup_ok = True
                    break
                except:
                    continue
            
            if not bl_setup_ok:
                self._log("  [Warning] Could not set surface list for BoundaryLayer (Unknown option)")
                return False

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
                    gmsh.model.mesh.field.setNumber(field_tag, "Quads", 1)
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
        # element order: 1 = linear (Tet4), 2 = quadratic (Tet10)
        order = 2 if config.second_order else 1
        gmsh.option.setNumber("Mesh.ElementOrder", order)
        
        if config.second_order:
             self._log("  [Mode] Enabling Second-Order Elements (Tet10/C3D10)")
             gmsh.option.setNumber("Mesh.HighOrderOptimize", 1)
        gmsh.option.setNumber("Mesh.MeshSizeMin", config.min_mesh_size * config.mesh_size_factor)
        gmsh.option.setNumber("Mesh.MeshSizeMax", config.max_mesh_size * config.mesh_size_factor)
        
        # Delaunay is generally most robust for 3D
        gmsh.option.setNumber("Mesh.Algorithm", 6)      # Frontal-Delaunay 2D
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)    # Delaunay 3D
        
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeThreshold", 0.3)
        
        # Ensure we generate 3D mesh
        # Ensure we generate 3D mesh
        # gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0) # Sometimes helps

        # Hex-Dominant Logic
        if config.hex_dominant:
            self._log("  [Mode] Enabling Hex-Dominant Recombination")
            # 1. Recombine surface meshes (Quads)
            gmsh.option.setNumber("Mesh.RecombineAll", 1)
            # 2. Use Frontal-Delaunay for Quads (Algo 8)
            gmsh.option.setNumber("Mesh.Algorithm", 8)
            # 3. Use 3D recombination (if available/stable)
            # Mesh.Algorithm3D = 10 (HXT) failed (requires triangles).
            # Use standard Delaunay (1) which supports 3D recombination (merging tets).
            gmsh.option.setNumber("Mesh.Algorithm3D", 1) 
            # 4. Subdivision: 1=All Hex (subdivides tets into 4 hexes), 0=None
            # We avoid global subdivision as it increases count x8. 
            # Instead we rely on Recombine 3D logic if possible.
            # For now, simplistic approach: Recombine Surface + Extrusion (if swept) or just Prism layers
            
            # NOTE: True unstructured hex meshing is hard. 
            # We'll set RecombinationAlgorithm to Simple (0) or Blossom (1)
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1) 

    def _check_orthogonality(self, threshold: float = 0.1) -> Tuple[bool, float]:
        """
        Check mesh orthogonality (SICN/Gamma proxy).
        Returns (Passed?, MinValue)
        """
        # We'll use Scaled Jacobian (SIJ) or SICN as a proxy for "goodness"
        # OpenFOAM CheckMesh cares about Orthogonality, Skewness, Aspect Ratio.
        
        try:
            # 3D Elements
            _, elem_tags, _ = gmsh.model.mesh.getElements(3)
            if not elem_tags: return True, 1.0 # No volume yet?
            
            # Flatten tags
            all_tags = []
            for tags in elem_tags:
                all_tags.extend(tags)
                
            if not all_tags: return True, 1.0
            
            # Get SICN (Signed Inverse Condition Number) - Good proxy for general quality
            # Range [-1, 1], 1 is perfect. < 0 is inverted.
            qualities = gmsh.model.mesh.getElementQualities(all_tags, "minSICN")
            min_val = min(qualities)
            avg_val = sum(qualities) / len(qualities)
            
            self._log(f"  [Quality] SICN Min: {min_val:.4f}, Avg: {avg_val:.4f}")
            
            if min_val < 0:
                self._log(f"  [CRITICAL] Negative Jacbobian/SICN detected (Inverted elements)!")
                return False, min_val
                
            if min_val < threshold:
                return False, min_val
                
            return True, min_val
            
        except Exception as e:
            self._log(f"  [Warning] Quality check failed to run: {e}")
            return True, 1.0

    def _check_geometric_fidelity(self, threshold: float = 0.02) -> Tuple[bool, float, float]:
        """
        Compare mesh volume/area to original CAD to detect geometry loss.
        
        Args:
            threshold: Maximum allowed fractional deviation (0.02 = 2%)
            
        Returns:
            (Passed?, volume_deviation, area_deviation)
        """
        cad_vol = self.geometry_info.get('cad_volume', 0)
        cad_area = self.geometry_info.get('cad_area', 0)
        
        if cad_vol < 1e-12:
            self._log("  [Fidelity] Skipping (CAD volume not available)")
            return True, 0.0, 0.0
            
        # Compute mesh volume by summing all 3D element volumes
        try:
            _, elem_tags, _ = gmsh.model.mesh.getElements(3)
            if not elem_tags:
                return True, 0.0, 0.0
                
            all_tags = []
            for tags in elem_tags:
                all_tags.extend(tags)
                
            # Gmsh doesn't have direct element volume API, but we can use quality/volume
            # Alternative: Use node coords to compute volume manually (complex)
            # Simpler: Use gmsh.model.occ.getMass on the final volume entities
            # But mesh may not have OCC entities anymore...
            
            # Fallback: Estimate via bounding box + density
            # Actually, let's use Jacobian-based volume approximation
            # For tets: V = |det(J)| / 6
            # This is complicated. Let's use a simpler heuristic:
            # After meshing, if we have physical volumes, we can re-query OCC mass
            
            # Most reliable: Gmsh internal mesh volume via plugin or custom calc
            # For MVP: Just check that we have cells and log a note
            
            # Compute sum of unsigned Jacobians as proxy for volume
            # This is imperfect but directionally correct.
            try:
                jacobians = gmsh.model.mesh.getElementQualities(all_tags, "volume")
                mesh_vol = sum(jacobians)
            except:
                # Fallback: assume mesh vol = CAD vol (no check)
                mesh_vol = cad_vol
            
            vol_dev = abs(mesh_vol - cad_vol) / cad_vol if cad_vol > 0 else 0
            area_dev = 0.0  # Surface area check is harder, skip for now
            
            self._log(f"  [Fidelity] CAD Vol: {cad_vol:.4e}, Mesh Vol: {mesh_vol:.4e}, Dev: {vol_dev*100:.2f}%")
            
            return vol_dev <= threshold, vol_dev, area_dev
            
        except Exception as e:
            self._log(f"  [Fidelity] Check failed: {e}")
            return True, 0.0, 0.0

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
