"""
Base Mesh Generator Module
==========================

Provides base class and common functionality for all mesh generation strategies.
Eliminates code duplication and provides consistent interface.
"""

import gmsh
import os
import math
import time
import json
import threading
import multiprocessing
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod

from .quality import MeshQualityAnalyzer
from .config import Config, get_default_config
from .ai_integration import AIRecommendationEngine, MeshRecommendation


def _canary_worker(file_path, results_queue):
    """Worker function for 3D/2D meshing canary. Isolated in separate process."""
    try:
        import gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 1)
        
        # Ultra-stable settings for canary
        gmsh.option.setNumber("General.NumThreads", 1)  # Disable OpenMP
        gmsh.option.setNumber("Geometry.OCCAutoFix", 0) # Disable auto-healing
        gmsh.option.setNumber("Geometry.Tolerance", 1e-2)
        gmsh.option.setNumber("Mesh.Algorithm", 1)      # MeshAdapt (stable)
        
        gmsh.model.add("Canary")
        gmsh.model.occ.importShapes(file_path)
        gmsh.model.occ.synchronize()
        
        # Fast 2D mesh
        gmsh.option.setNumber("Mesh.Algorithm", 1) # MeshAdapt
        gmsh.model.mesh.generate(2)
        
        gmsh.finalize()
        results_queue.put(True)
    except Exception as e:
        # results_queue.put(False)
        pass


def _bounding_box_worker(output_file, p_min, p_max):
    """Isolated worker for creating a bounding box mesh."""
    try:
        import gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        
        # Ultra-stable settings for bounding box
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.option.setNumber("Geometry.OCCAutoFix", 0)
        gmsh.option.setNumber("Geometry.Tolerance", 1e-2)
        
        gmsh.model.add("BBox_Fallback")
        
        dx = max(1e-3, p_max[0] - p_min[0])
        dy = max(1e-3, p_max[1] - p_min[1])
        dz = max(1e-3, p_max[2] - p_min[2])
        
        gmsh.model.occ.addBox(p_min[0], p_min[1], p_min[2], dx, dy, dz)
        gmsh.model.occ.synchronize()
        
        max_dim = max(dx, dy, dz)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", max_dim / 10.0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_dim / 2.0)
        gmsh.model.mesh.generate(3)
        
        gmsh.write(output_file)
        gmsh.finalize()
        return True
    except:
        return False


class MeshGenerationResult:
    """Result of mesh generation operation"""

    def __init__(self, success: bool, output_file: Optional[str] = None,
                 quality_metrics: Optional[Dict] = None, message: str = ""):
        self.success = success
        self.output_file = output_file
        self.quality_metrics = quality_metrics
        self.message = message
        self.iterations = 0
        self.history = []

    def __repr__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return f"MeshGenerationResult({status}, iterations={self.iterations})"


class BaseMeshGenerator(ABC):
    """
    Base class for all mesh generators

    Provides common functionality:
    - CAD file loading and validation
    - Mesh parameter calculation
    - Quality analysis
    - Iteration tracking
    - History persistence
    - Logging
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize mesh generator

        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config or get_default_config()
        self.quality_analyzer = MeshQualityAnalyzer()
        self.ai_engine = AIRecommendationEngine(self.config)

        # State
        self.current_iteration = 0
        self.quality_history = []
        self.current_mesh_params = {}
        self.geometry_info = {}

        # Gmsh state
        self.gmsh_initialized = False
        self.model_loaded = False

    def log_message(self, message: str, level: str = "INFO"):
        """Print message with timestamp (millisecond precision)"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        prefix = f"[{timestamp}]"
        if level != "INFO":
            prefix += f" {level}:"
        print(f"{prefix} {message}", flush=True)

    def generate_mesh(self, input_file: str, output_file: Optional[str] = None) -> MeshGenerationResult:
        """
        Main entry point for mesh generation

        Args:
            input_file: Path to CAD file (STEP/IGES)
            output_file: Optional output file path

        Returns:
            MeshGenerationResult object
        """
        self.log_message("=" * 60)
        self.log_message(f"{self.__class__.__name__} Starting")
        self.log_message("=" * 60)

        result = MeshGenerationResult(success=False)

        try:
            # Validate input
            if not self.validate_input_file(input_file):
                result.message = "Invalid input file"
                return result

            # Determine output file
            if output_file is None:
                output_file = self._get_default_output_path(input_file)

            # Initialize Gmsh
            self.initialize_gmsh()

            # Load CAD file
            if not self.load_cad_file(input_file):
                result.message = "Failed to load CAD file"
                return result

            # Run strategy-specific meshing
            if self.run_meshing_strategy(input_file, output_file):
                result.success = True
                result.output_file = output_file
                result.iterations = self.current_iteration
                result.history = self.quality_history.copy()

                # Final quality metrics
                if self.quality_history:
                    result.quality_metrics = self.quality_history[-1]['metrics']

                result.message = "Mesh generation completed successfully"

                # ANSYS Export (CFD/FEA)
                ansys_mode = getattr(self.config.mesh_params, 'ansys_mode', 'None')
                if ansys_mode and ansys_mode != "None":
                    self.log_message(f"[ANSYS] Starting {ansys_mode} export...")
                    try:
                        # CRITICAL: ExhaustiveMeshGenerator finalizes Gmsh, so we must re-init and reload
                        if not gmsh.isInitialized():
                            self.log_message("[ANSYS] Re-initializing Gmsh...")
                            gmsh.initialize()
                        
                        self.log_message(f"[ANSYS] Loading mesh from: {output_file}")
                        gmsh.clear()
                        gmsh.merge(output_file)

                        # Clean topology
                        self.log_message("[ANSYS] Cleaning topology...")
                        gmsh.model.mesh.removeDuplicateNodes()

                        if "CFD" in ansys_mode:
                            # CFD MODE: Linear elements, .msh v2.2 for Fluent
                            self.log_message("[ANSYS] Converting to linear elements (Tet4)...")
                            gmsh.model.mesh.setOrder(1)  # Convert to linear
                            
                            # CRITICAL: Generate 2D surface mesh explicitly
                            # Without this, only volume tets exist, causing "Empty Ghost"
                            self.log_message("[ANSYS] Generating 2D surface mesh...")
                            try:
                                # Check if 2D mesh already exists
                                existing_2d = gmsh.model.mesh.getElements(2)
                                if not existing_2d[0]:  # No 2D elements
                                    self.log_message("[ANSYS] No 2D mesh found, generating from volume boundaries...")
                                    gmsh.model.mesh.generate(2)  # Generate surface mesh
                            except:
                                pass  # If it fails, physical groups will still help
                            
                            self.log_message("[ANSYS] Creating Face Zones...")
                            self._create_ansys_physical_groups(mode="CFD")
                            
                            # Fluent requires ALL elements including surface triangles
                            # CRITICAL: SaveAll=1 prevents "Empty Ghost" / Null Domain error
                            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
                            gmsh.option.setNumber("Mesh.Binary", 0)
                            gmsh.option.setNumber("Mesh.SaveAll", 1)  # Force save surface triangles!
                            
                            # First write Gmsh format
                            gmsh_temp = str(Path(output_file).with_suffix('')) + "_gmsh_temp.msh"
                            self.log_message(f"[ANSYS] Writing Gmsh intermediate: {gmsh_temp}")
                            gmsh.write(gmsh_temp)
                            
                            # Convert to pure Fluent-compatible mesh
                            # FIX: Fluent requires surface elements ("skin") to define the domain.
                            # Pure volume meshes cause "Null Domain Pointer" crashes.
                            # Convert to Fluent-compatible format
                            # FIX: Fluent 2025 fails with standard .msh files ("Null Domain Pointer").
                            # SOLUTION: Use Nastran (.bdf) format which works robustly.
                            # Also use createTopology() to ensure graph connectivity is perfect.
                            self.log_message("[ANSYS] Preparing Fluent-compatible mesh (Nastran BDF)...")
                            
                            try:
                                # Save current model to temp file
                                gmsh_temp = str(Path(output_file).with_suffix('')) + "_temp.msh"
                                gmsh.write(gmsh_temp)
                                
                                # Use a fresh Gmsh session for clean topology rebuilding
                                import gmsh as gmsh_export
                                gmsh_export.initialize()
                                gmsh_export.open(gmsh_temp)
                                
                                # 1. Clear Groups
                                gmsh_export.model.removePhysicalGroups(gmsh_export.model.getPhysicalGroups())
                                
                                # 2. Linearize (Tet10 -> Tet4)
                                self.log_message("[ANSYS] Linearizing mesh elements...")
                                gmsh_export.model.mesh.setOrder(1)
                                
                                # 3. Rebuild Topology (The Robust Fix)
                                # This automatically handles skin extraction, node sharing, and connectivity
                                self.log_message("[ANSYS] Rebuilding topology (createTopology)...")
                                gmsh_export.model.mesh.createTopology()
                                
                                # 4. Assign Standard CFD Groups
                                # Volume -> "fluid"
                                vols = gmsh_export.model.getEntities(3)
                                if vols:
                                    p_vol = gmsh_export.model.addPhysicalGroup(3, [e[1] for e in vols])
                                    gmsh_export.model.setPhysicalName(3, p_vol, "fluid")
                                    self.log_message(f"[ANSYS] Assigned 'fluid' zone (Tags: {[e[1] for e in vols]})")
                                else:
                                    self.log_message("[ANSYS] WARNING: No volume entities found!", level="WARNING")

                                # Surface -> "wall"
                                surfs = gmsh_export.model.getEntities(2)
                                if surfs:
                                    p_surf = gmsh_export.model.addPhysicalGroup(2, [e[1] for e in surfs])
                                    gmsh_export.model.setPhysicalName(2, p_surf, "wall")
                                    self.log_message(f"[ANSYS] Assigned 'wall' zone (Tags: {[e[1] for e in surfs]})")
                                else:
                                    self.log_message("[ANSYS] WARNING: No surface entities found!", level="WARNING")
                                
                                # 5. Export to Nastran BDF
                                # Fluent prefers this over .msh for 2025 R2
                                ansys_file = str(Path(output_file).with_suffix('')) + ".bdf"
                                
                                gmsh_export.option.setNumber("Mesh.BdfFieldFormat", 1) # Standard BDF
                                gmsh_export.option.setNumber("Mesh.SaveAll", 0)        # Only save Physical Groups
                                gmsh_export.write(ansys_file)
                                
                                gmsh_export.finalize()
                                
                                self.log_message(f"[ANSYS] ✓ Successfully exported: {ansys_file}")
                                
                                # Clean up
                                try:
                                    os.remove(gmsh_temp)
                                except:
                                    pass

                            except Exception as e:
                                import traceback
                                self.log_message(f"[ANSYS] Export failed: {e}", level="ERROR")
                                self.log_message(f"[ANSYS] Traceback:\n{traceback.format_exc()}", level="ERROR")
                                ansys_file = output_file # Fallback
                            
                        elif "FEA" in ansys_mode:
                            # FEA MODE: Quadratic elements, .bdf for Mechanical
                            self.log_message("[ANSYS] Converting to quadratic elements (Tet10)...")
                            gmsh.model.mesh.setOrder(2)  # Convert to quadratic
                            
                            self.log_message("[ANSYS] Creating Named Selections...")
                            self._create_ansys_physical_groups(mode="FEA")
                            
                            # ANSYS Mechanical prefers Nastran BDF
                            gmsh.option.setNumber("Mesh.BdfFieldFormat", 1)
                            gmsh.option.setNumber("Mesh.SaveAll", 1)  # Save all for FEA too
                            
                            ansys_file = str(Path(output_file).with_suffix('')) + "_structural.bdf"
                        
                        if "FEA" in ansys_mode:  # Only write for FEA (CFD already written via meshio)
                            self.log_message(f"[ANSYS] Writing to: {ansys_file}")
                            gmsh.write(ansys_file)
                        
                        self.log_message(f"[ANSYS] ✓ Successfully exported: {ansys_file}")
                        
                    except Exception as e:
                        import traceback
                        self.log_message(f"[ANSYS] ✗ Failed to export ANSYS mesh: {e}", level="ERROR")
                        self.log_message(f"[ANSYS] Traceback: {traceback.format_exc()}", level="ERROR")



            else:
                result.message = "Meshing strategy failed"

            return result

        except Exception as e:
            self.log_message(f"ERROR: {e}", level="ERROR")
            result.message = str(e)
            return result

        finally:
            self.finalize_gmsh()

    @abstractmethod
    def run_meshing_strategy(self, input_file: str, output_file: str) -> bool:
        """
        Strategy-specific meshing implementation

        Must be implemented by subclasses.

        Args:
            input_file: Path to input CAD file
            output_file: Path for output mesh file

        Returns:
            True if successful, False otherwise
        """
        pass

    def validate_input_file(self, filename: str) -> bool:
        """Validate input CAD file"""
        if not os.path.exists(filename):
            self.log_message(f"ERROR: File not found: {filename}", level="ERROR")
            return False

        ext = os.path.splitext(filename)[1].lower()
        if ext not in ['.step', '.stp', '.iges', '.igs', '.stl', '.obj', '.ply', '.x_t', '.x_b', '.prt', '.sldprt', '.sldasm']:
            self.log_message(f"ERROR: Unsupported file format: {ext}", level="ERROR")
            return False

        return True

    def initialize_gmsh(self, thread_count=None, verbosity=None):
        """
        Initialize Gmsh with configurable multi-threading
        
        Args:
            thread_count: Number of threads for Gmsh operations
            verbosity: Optional verbosity level (1-5). If None, uses default (2).
        """
        if not self.gmsh_initialized:
            import multiprocessing
            
            gmsh.initialize()
            gmsh.model.add(self.__class__.__name__)
            
            # Determine thread count
            if thread_count is None:
                # Scenario B: Single worker, use reasonable core count
                import multiprocessing
                import platform
                num_cores = multiprocessing.cpu_count()
                
                # CRITICAL: Windows has instability with 16+ threads in HXT
                if platform.system() == "Windows":
                    # Cap at 8 threads for balance of speed/stability on high-core CPUs
                    thread_count = min(num_cores, 8)
                else:
                    thread_count = num_cores
            
            # Configure threading
            gmsh.option.setNumber("General.NumThreads", thread_count)
            gmsh.option.setNumber("Mesh.MaxNumThreads1D", thread_count)
            gmsh.option.setNumber("Mesh.MaxNumThreads2D", thread_count)
            gmsh.option.setNumber("Mesh.MaxNumThreads3D", thread_count)
            
            # Default verbosity (quiet)
            v_level = verbosity if verbosity is not None else 2
            t_level = 1 if v_level <= 2 else 1 # Terminal 1 shows errors
            
            # Reduce terminal spam and entity tracking to prevent pipe buffer issues
            gmsh.option.setNumber("General.Terminal", t_level)
            gmsh.option.setNumber("General.Verbosity", v_level) 
            
            self.gmsh_initialized = True
            self.log_message(f"Gmsh initialized with {thread_count} threads (Verbosity: {v_level})")

    def elevate_verbosity(self, level=3):
        """Increase Gmsh verbosity for detailed progress reporting"""
        if self.gmsh_initialized:
            self.log_message(f"[Verbosity] Elevating Gmsh verbosity to level {level} for detailed progress...")
            gmsh.option.setNumber("General.Terminal", 1)
            gmsh.option.setNumber("General.Verbosity", level)

    def finalize_gmsh(self):
        """Finalize Gmsh"""
        if self.gmsh_initialized:
            try:
                gmsh.finalize()
                self.gmsh_initialized = False
                self.model_loaded = False
                self.log_message("Gmsh finalized")
            except:
                pass

    def load_cad_file(self, filename: str) -> bool:
        """Load CAD file into Gmsh"""
        self.log_message(f"Loading CAD file: {os.path.basename(filename)}")

        try:
            # CRITICAL: Set GUI options BEFORE loading file
            # These settings from the GUI's .opt file enable successful meshing
            # They must be applied before merge() so OCC cleans geometry during import
            
            # CONSISTENCY FIX: Match api_server.py robust loading
            gmsh.option.setNumber("Geometry.OCCAutoFix", 0)      # Disable unstable auto-fix
            gmsh.option.setNumber("Geometry.Tolerance", 1e-2)    # More forgiving tolerance
            
            gmsh.option.setNumber("Geometry.AutoCoherence", 1)   # Auto-merge touching vertices
            
            # CONDITIONAL AGGRESSIVE DEFEATURING for complex geometries (gyroid, TPMS, etc.)
            # User can enable this via GUI checkbox for problematic meshes
            aggressive_healing = getattr(self.config.mesh_params, 'aggressive_healing', False)
            
            if aggressive_healing:
                self.log_message("[Geometry] Applying aggressive healing (slow but thorough)")
                # Increased from 1e-08 to 1e-06 to heal problematic intersections
                gmsh.option.setNumber("Geometry.Tolerance", 1e-1)   # More forgiving tolerance
                
                # Additional OCC healing for complex boundary intersections
                gmsh.option.setNumber("Geometry.OCCFixDegenerated", 1)  # Fix degenerate edges/faces
                gmsh.option.setNumber("Geometry.OCCFixSmallEdges", 1)    # Remove tiny edges/micro-gaps
                gmsh.option.setNumber("Geometry.OCCFixSmallFaces", 1)    # Remove/merge tiny faces
            else:
                self.log_message("[Geometry] Using fast mode (lenient tolerance)")
                # Standard lenient tolerance consistent with preview
                gmsh.option.setNumber("Geometry.Tolerance", 1e-2)
            
            # Disable destructive operations that the GUI doesn't use
            gmsh.option.setNumber("Geometry.OCCSewFaces", 0)     # Don't force sewing
            gmsh.option.setNumber("Geometry.OCCMakeSolids", 0)   # Don't force solid creation
            
            # Import based on file type
            import time
            t0 = time.time()
            ext = os.path.splitext(filename)[1].lower()
            try:
                if ext in ['.step', '.stp', '.x_t', '.x_b', '.prt', '.sldprt', '.sldasm']:
                    gmsh.model.occ.importShapes(filename)
                    self.log_message(f"  - Import completed in {time.time()-t0:.2f}s")
                    gmsh.model.occ.synchronize()
                    self.log_message(f"  - Synchronize completed in {time.time()-t0:.2f}s")
                elif ext in ['.iges', '.igs']:
                    gmsh.model.occ.importShapes(filename)
                    self.log_message(f"  - Import completed in {time.time()-t0:.2f}s")
                    gmsh.model.occ.synchronize()
                    self.log_message(f"  - Synchronize completed in {time.time()-t0:.2f}s")
                elif ext in ['.stl', '.obj', '.ply']:
                    self.log_message(f"Loading surface mesh: {filename}")
                    gmsh.merge(filename)
                else:
                    raise Exception(f"Unsupported format: {ext}")
            except Exception as e:
                self.log_message(f"[Geometry] Initial open failed ({e}), attempting ultra-lenient fallback...")
                gmsh.option.setNumber("Geometry.Tolerance", 1.0)
                gmsh.option.setNumber("Geometry.OCCAutoFix", 0)
                
                if ext in ['.stl', '.obj', '.ply']:
                    gmsh.merge(filename)
                else:
                    gmsh.model.occ.importShapes(filename)
                    gmsh.model.occ.synchronize()
                self.log_message("[Geometry] Ultra-lenient fallback successful")
                
                # Create a volume for 3D meshing from the surface
                self.log_message("Constructing volume from surface mesh...")
                
                # 1. Classify surfaces (split by angle > 40 degrees)
                gmsh.model.mesh.classifySurfaces(math.pi * 40 / 180, True, True)
                
                # 2. Create discrete geometry entities
                gmsh.model.mesh.createGeometry()
                
                # 3. Create Volume from Surface Loop
                surfaces = gmsh.model.getEntities(2)
                if surfaces:
                    # Collect all surface tags
                    s_tags = [s[1] for s in surfaces]
                    
                    # Create surface loop
                    l_tag = gmsh.model.geo.addSurfaceLoop(s_tags)
                    
                    # Create volume
                    gmsh.model.geo.addVolume([l_tag])
                    gmsh.model.geo.synchronize()
                    
                    self.log_message(f"[OK] Created volume from {len(surfaces)} surfaces")
                else:
                    self.log_message("[!] No surfaces found in mesh file", level="WARNING")

            self.log_message("[OK] File imported successfully")

            # PROACTIVE ISOLATION: If we are in isolation mode, remove all other volumes NOW.
            # This prevents the expensive _extract_geometry_info from analyzing 150+ unwanted volumes.
            if hasattr(self, 'target_volume_tag') and self.target_volume_tag is not None:
                self.log_message(f"[Surgical] Isolating volume {self.target_volume_tag} - removing all other entities...")
                all_vols = gmsh.model.getEntities(3)
                to_remove = [(3, tag) for _, tag in all_vols if tag != self.target_volume_tag]
                if to_remove:
                    gmsh.model.occ.remove(to_remove, recursive=True)
                    gmsh.model.occ.synchronize()
                    self.log_message(f"[Surgical] Removed {len(to_remove)} other volumes")

            # NEW: Log detailed entity counts immediately after import
            v_count = len(gmsh.model.getEntities(3))
            s_count = len(gmsh.model.getEntities(2))
            c_count = len(gmsh.model.getEntities(1))
            p_count = len(gmsh.model.getEntities(0))
            self.log_message(f"[Diagnostics] Imported entities: {v_count} Vols, {s_count} Surfs, {c_count} Curves, {p_count} Points")

            # CRITICAL: Do NOT call healShapes() here!
            # The GUI doesn't call it during import - it relies on OCCAutoFix + AutoCoherence
            # healShapes() triggers internal fragment operations that cause BOPAlgo errors
            # The geometry is already cleaned by the import settings above
            
            # Get geometry info and analyze features
            self._extract_geometry_info()
            self._analyze_geometry_features()

            # Apply geometry-aware mesh settings
            self._apply_geometry_aware_settings()
            
            # NOTE: User mesh parameters (max_size_mm/min_size_mm) will be applied
            # by the strategy code just before meshing, not here during CAD load
            
            # CRITICAL: Assign Physical Groups to volumes
            # Without this, if ANY surface group is defined, volumes won't export!
            self._ensure_volume_physical_groups()

            self.model_loaded = True

            return True

        except Exception as e:
            import traceback
            self.log_message(f"ERROR: Failed to load CAD file: {e}", level="ERROR")
            self.log_message(f"Traceback:\n{traceback.format_exc()}", level="ERROR")
            return False

    def _extract_geometry_info(self):
        """Extract and store geometry information"""
        self.log_message("[Geometry] Extracting geometry information...")
        self.log_message("[Geometry] Getting volumes...")
        volumes = gmsh.model.getEntities(dim=3)
        self.log_message(f"[Geometry] Found {len(volumes)} volumes")
        
        self.log_message("[Geometry] Getting surfaces...")
        surfaces = gmsh.model.getEntities(dim=2)
        self.log_message(f"[Geometry] Found {len(surfaces)} surfaces")
        
        self.log_message("[Geometry] Getting curves...")
        curves = gmsh.model.getEntities(dim=1)
        self.log_message(f"[Geometry] Found {len(curves)} curves")
        
        self.log_message("[Geometry] Getting points...")
        points = gmsh.model.getEntities(dim=0)
        self.log_message(f"[Geometry] Found {len(points)} points")

        self.geometry_info = {
            'num_volumes': len(volumes),
            'num_surfaces': len(surfaces),
            'num_curves': len(curves),
            'num_points': len(points)
        }

        self.log_message(f"[Geometry] Analysis results:")
        self.log_message(f"  - Volumes: {len(volumes)}")
        self.log_message(f"  - Surfaces: {len(surfaces)}")
        self.log_message(f"  - Curves: {len(curves)}")
        self.log_message(f"  - Points: {len(points)}")

        # Individual volume analysis
        # PERFORMANCE: Skip expensive getMass calculation for large assemblies (>50 volumes)
        # Each getMass call on complex geometry (threads, small features) can take 1-5 seconds
        # For 151 volumes, this would hang for 3-10 minutes with no output
        if len(volumes) > 50:
            self.log_message(f"  - Skipping per-volume mass calculation (assembly has {len(volumes)} volumes)")
            self.log_message(f"  - (getMass is expensive for complex geometry - will calculate total volume only)")
            total_mass = 0
            valid_volumes = len(volumes)  # Assume all are valid
        else:
            total_mass = 0
            valid_volumes = 0
            for i, (dim, tag) in enumerate(volumes):
                try:
                    mass = gmsh.model.occ.getMass(dim, tag)
                    if mass > 1e-12:
                        valid_volumes += 1
                    total_mass += mass
                    bbox = gmsh.model.getBoundingBox(dim, tag)
                    # Only log details for the first 30 volumes or if explicitly debugging
                    if i < 30:
                        self.log_message(f"    V{tag}: mass={mass:.6f}, bbox=[{bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f}] to [{bbox[3]:.2f}, {bbox[4]:.2f}, {bbox[5]:.2f}]")
                except:
                    pass
            
            if len(volumes) > 30:
                self.log_message(f"    ... (and {len(volumes)-30} more volumes)")
                
            self.log_message(f"  - Valid Volumes (Mass > 0): {valid_volumes}")
            self.log_message(f"  - Total Model Mass: {total_mass:.6f}")

        if not volumes:
            self.log_message("[!] No volumes detected - attempting shell-to-solid repair...")
            volumes = self._attempt_shell_to_solid_repair()

            if not volumes:
                raise Exception("No 3D volumes detected in the CAD file")

        # Calculate bounding box
        self.geometry_info['bounding_box'] = self._calculate_bounding_box(volumes)
        self.geometry_info['diagonal'] = self._calculate_diagonal()
        
        # Calculate total volume for mesh sizing
        try:
            total_volume = 0.0
            if len(volumes) > 50:
                self.log_message("  - Skipping total volume calculation (looping getMass is slow for large assemblies)")
                total_volume = 0.0
            else:
                for v_dim, v_tag in volumes:
                    # getMass returns volume for 3D entities
                    total_volume += gmsh.model.occ.getMass(v_dim, v_tag)
            self.geometry_info['volume'] = total_volume
            self.log_message(f"[OK] Calculated/Estimated geometry volume: {total_volume:.2f} mm³")
            
            # Debug: Show bounding box dimensions for unit diagnosis
            bb = self.geometry_info.get('bounding_box', {})
            if bb:
                dims = [bb['max'][i] - bb['min'][i] for i in range(3)]
                self.log_message(f"[DEBUG] Bounding box: {dims[0]:.2f} x {dims[1]:.2f} x {dims[2]:.2f} (model units)")
                self.log_message(f"[DEBUG] If these look like mm values (e.g. 50-200), model is in mm")
                self.log_message(f"[DEBUG] If these look like m values (e.g. 0.05-0.2), model is in meters")
        except Exception as e:
            self.log_message(f"[!] Could not calculate volume: {e}")
            self.geometry_info['volume'] = None

    def _calculate_bounding_box(self, volumes) -> Dict[str, List[float]]:
        """Calculate overall bounding box"""
        # PERFORMANCE: Use Gmsh's internal calculation for the whole model
        # instead of looping over volumes. This is much faster and more accurate for assemblies.
        try:
            bb = gmsh.model.getBoundingBox(-1, -1)
            # If box is valid (not empty)
            if bb[3] >= bb[0]:
                return {'min': [bb[0], bb[1], bb[2]], 'max': [bb[3], bb[4], bb[5]]}
        except:
            pass

        # Fallback to loop if global call fails or returns empty
        b_min = [float('inf')] * 3
        b_max = [float('-inf')] * 3

        for v_dim, v_tag in volumes:
            try:
                bb = gmsh.model.getBoundingBox(v_dim, v_tag)
                for i in range(3):
                    b_min[i] = min(b_min[i], bb[i])
                    b_max[i] = max(b_max[i], bb[i+3])
            except:
                pass

        return {'min': b_min, 'max': b_max}

    def _calculate_diagonal(self) -> float:
        """Calculate bounding box diagonal"""
        bb = self.geometry_info.get('bounding_box')
        if not bb:
            return 1.0

        b_min = bb['min']
        b_max = bb['max']

        dx = b_max[0] - b_min[0]
        dy = b_max[1] - b_min[1]
        dz = b_max[2] - b_min[2]

        return math.sqrt(dx**2 + dy**2 + dz**2)

    def _ensure_volume_physical_groups(self):
        """
        Ensure all volumes are assigned to Physical Groups
        
        CRITICAL FIX: Gmsh enters "Exclusive Mode" when ANY Physical Group exists.
        If you define surface groups but forget volume groups, all tets generate
        but silently disappear during export. This prevents that.
        
        SURGICAL ISOLATION MODE: If target_volume_tag is set, assign a unique
        physical group name so individual volumes can be preserved after merging.
        """
        volumes = gmsh.model.getEntities(dim=3)
        
        if not volumes:
            self.log_message("[!] No volumes to assign Physical Groups")
            return
        
        # Check if any physical groups already exist
        existing_groups = gmsh.model.getPhysicalGroups()
        has_volume_groups = any(dim == 3 for dim, _ in existing_groups)
        
        if not has_volume_groups:
            # SURGICAL ISOLATION: Assign unique per-volume physical groups
            if hasattr(self, 'target_volume_tag') and self.target_volume_tag is not None:
                # In isolation mode, there should only be 1 volume
                # Assign it a unique name based on the original tag
                vol_tags = [v[1] for v in volumes]
                p_vol = gmsh.model.addPhysicalGroup(3, vol_tags)
                unique_name = f"Volume_{self.target_volume_tag}"
                gmsh.model.setPhysicalName(3, p_vol, unique_name)
                self.log_message(f"[OK] Assigned isolated volume to Physical Group '{unique_name}'")
            else:
                # Normal mode: assign all volumes to a single Physical Group
                vol_tags = [v[1] for v in volumes]
                p_vol = gmsh.model.addPhysicalGroup(3, vol_tags)
                gmsh.model.setPhysicalName(3, p_vol, "Volume")
                self.log_message(f"[OK] Assigned {len(vol_tags)} volume(s) to Physical Group 'Volume'")
        else:
            self.log_message(f"[OK] Volume Physical Groups already exist")

    def _attempt_shell_to_solid_repair(self) -> List:
        """
        Attempt to repair shells (hollow surfaces) into solids
        
        CRITICAL FIX: For assemblies, we cannot blindly sew all surfaces together.
        This creates self-intersecting geometry. Instead, we use healShapes ONCE,
        then use OCC sewing to intelligently identify separate components.
        
        Returns:
            List of volume entities if successful, empty list otherwise
        """
        try:
            self.log_message("  Attempting intelligent shell-to-solid conversion...")
            
            # CRITICAL: Only call healShapes ONCE to avoid 5x slowdown
            # The GUI doesn't heal at all unless you click "Heal"
            self.log_message("  Using OCC healShapes (single pass)...")
            gmsh.model.occ.healShapes()
            gmsh.model.occ.synchronize()
            
            volumes = gmsh.model.getEntities(dim=3)
            
            if volumes:
                self.log_message(f"  [OK] healShapes created {len(volumes)} volume(s)")
                
                # If we have multiple volumes (assembly), fragment them
                # This creates conformal interfaces where parts touch/overlap
                if len(volumes) > 1:
                    self.log_message(f"  Fragmenting {len(volumes)} volumes for conformality...")
                    try:
                        # Fragment all volumes against each other
                        gmsh.model.occ.fragment(volumes, volumes)
                        gmsh.model.occ.synchronize()
                        
                        # Remove any duplicate entities created by fragmentation
                        gmsh.model.occ.removeAllDuplicates()
                        gmsh.model.occ.synchronize()
                        
                        # Get updated volume list
                        volumes = gmsh.model.getEntities(dim=3)
                        self.log_message(f"  [OK] Fragmentation complete: {len(volumes)} volume(s)")
                    except Exception as e:
                        self.log_message(f"  [!] Fragmentation warning: {e}")
                        # Continue anyway - fragmentation is optional
                
                return volumes
            
            # MINIMAL APPROACH: Just use coherence and let the mesher handle it
            # Don't try to fix geometry - that causes BOPAlgo errors
            # The GUI doesn't repair assemblies, it just meshes them
            self.log_message("  [!] healShapes failed - using minimal coherence...")
            
            # Just glue touching vertices - that's it
            gmsh.model.occ.removeAllDuplicates()
            gmsh.model.occ.synchronize()
            
            # Don't call healShapes again - it already failed
            # Don't try to create surface loops - that causes BOPAlgo errors
            # Just let the mesher (HXT) handle whatever topology exists
            
            self.log_message("  [!] Coherence applied - proceeding to mesh with existing topology")
            self.log_message("  TIP: Using HXT algorithm which tolerates imperfect geometry")
            
            # Return empty - the mesher will work with whatever volumes exist
            # or mesh the surfaces directly if no volumes
            return []
            
        except Exception as e:
            self.log_message(f"  [X] Repair failed: {e}")
            return []
            
            # All repair strategies failed
            self.log_message("  [X] All repair strategies failed")
            self.log_message("  TIP: The CAD file may have:")
            self.log_message("     - Open surfaces (not watertight)")
            self.log_message("     - Self-intersecting geometry")
            self.log_message("     - Non-manifold edges")
            self.log_message("  TIP: Try repairing in your CAD software before meshing")
            
            return []
            
        except Exception as e:
            self.log_message(f"  [X] Shell-to-solid repair failed: {e}")
            return []

    def _analyze_geometry_features(self):
        """Analyze geometry for sharp features, small curves, etc."""
        try:
            curves = gmsh.model.getEntities(dim=1)
            surfaces = gmsh.model.getEntities(dim=2)

            # PERFORMANCE: Skip curve analysis for large/complex assemblies
            # Iterating through 2000+ curves and calling getBoundingBox/sqrt on each
            # can take significant time without providing critical info for assemblies.
            if len(curves) > 500:
                self.log_message(f"[Geometry] Skipping detailed curve analysis ({len(curves)} curves)")
                self.geometry_info['has_small_features'] = False
                return

            # Analyze curve lengths
            curve_lengths = []
            for dim, tag in curves:
                try:
                    bb = gmsh.model.getBoundingBox(dim, tag)
                    length = math.sqrt(
                        (bb[3]-bb[0])**2 + (bb[4]-bb[1])**2 + (bb[5]-bb[2])**2
                    )
                    curve_lengths.append(length)
                except:
                    pass

            # Find small features
            diagonal = self.geometry_info.get('diagonal', 1.0)
            small_feature_threshold = diagonal / 100.0

            small_curves = [l for l in curve_lengths if l < small_feature_threshold]

            self.geometry_info['has_small_features'] = len(small_curves) > 0
            self.geometry_info['min_curve_length'] = min(curve_lengths) if curve_lengths else diagonal / 20.0
            self.geometry_info['num_curves'] = len(curves)
            self.geometry_info['num_surfaces'] = len(surfaces)

            if small_curves:
                self.log_message(
                    f"[!] Detected {len(small_curves)} small curves "
                    f"(< {small_feature_threshold:.4f})"
                )

                # CRITICAL: If we have many small features, geometry is extremely complex
                if len(small_curves) > 500:
                    self.geometry_info['extremely_complex'] = True
                    self.log_message(
                        f"[!][!][!] EXTREMELY COMPLEX GEOMETRY ({len(small_curves)} small features)"
                    )
                    self.log_message(
                        f"    Will use aggressive coarsening to enable meshing"
                    )

        except Exception as e:
            self.log_message(f"[!] Geometry feature analysis warning: {e}")
            # Set safe defaults
            self.geometry_info['has_small_features'] = False
            self.geometry_info['min_curve_length'] = self.geometry_info.get('diagonal', 1.0) / 20.0

    def _apply_geometry_aware_settings(self):
        """Apply mesh settings based on geometry analysis"""
        try:
            # Enable mesh size from curvature for good geometry adaptation
            # User's CharacteristicLengthMax will cap this to respect max_size_mm
            gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
            gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 12)

            # Enable mesh size from points/curves
            gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 1)

            # Improve element quality
            gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
            gmsh.option.setNumber("Mesh.Smoothing", 10)  # More smoothing iterations

            # Handle sharp angles better
            gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", 0.1)

            # Avoid over-refinement
            gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 1.0)

            # CRITICAL: Special handling for extremely complex geometry
            if self.geometry_info.get('extremely_complex', False):
                self.log_message("[!] Applying COARSE mesh settings for extremely complex geometry...")

                # Force much coarser mesh
                diagonal = self.geometry_info.get('diagonal', 1.0)
                gmsh.option.setNumber("Mesh.CharacteristicLengthMin", diagonal / 20.0)  # Very coarse
                gmsh.option.setNumber("Mesh.CharacteristicLengthMax", diagonal / 5.0)   # Very coarse
                gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 3.0)  # 3x coarser!

                # Reduce mesh quality requirements to enable meshing
                gmsh.option.setNumber("Mesh.MinimumCirclePoints", 6)  # Reduced from 12
                gmsh.option.setNumber("Mesh.MinimumCurvePoints", 2)   # Minimum
                gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)  # Disable

                # Use simpler algorithms
                gmsh.option.setNumber("Mesh.Algorithm", 1)  # MeshAdapt (simpler)
                gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay (robust)

                self.log_message("    [OK] Coarse mesh settings applied (quality will be reduced)")

            # If small features detected (but not extreme), use finer minimum size
            elif self.geometry_info.get('has_small_features', False):
                min_curve = self.geometry_info.get('min_curve_length', 1.0)
                diagonal = self.geometry_info.get('diagonal', 1.0)

                # Adjust parameters for small features
                gmsh.option.setNumber("Mesh.MinimumCirclePoints", 20)
                gmsh.option.setNumber("Mesh.MinimumCurvePoints", 3)

                self.log_message(
                    f"[OK] Applied settings for small features "
                    f"(min curve: {min_curve:.4f})"
                )

            # Improve algorithm settings for complex geometries
            gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay for 2D
            gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay for 3D (stable)

            # Enable recombination to reduce poor quality elements
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)  # Blossom
            gmsh.option.setNumber("Mesh.RecombineOptimizeTopology", 5)

            self.log_message("[OK] Applied geometry-aware mesh settings")

        except Exception as e:
            self.log_message(f"[!] Could not apply geometry-aware settings: {e}")

    def check_1d_complexity(self) -> Dict[str, Any]:
        """
        Perform a fast 1D complexity check (Canary Strategy).
        Checks edge counts against volume and diagonal.
        
        Returns:
            Dict with 'toxic' boolean and reason.
        """
        self.log_message("Running 1D Complexity Canary...")
        
        curves = gmsh.model.getEntities(1)
        volumes = gmsh.model.getEntities(3)
        edge_count = len(curves)
        diag = self.geometry_info.get('diagonal', 1.0)
        
        # 1D TOXICITY HEURISTIC:
        # If a small part has a massive number of edges, it's likely "dirty"
        # or has excessive detail (e.g. threads, logos) that will stall meshing.
        # 
        # UPDATED THRESHOLDS (2024-12-28):
        # - Scale "extreme" count by diagonal. 
        # - 2000 edges for 2mm is bad, but 2000 edges for 200mm is fine.
        # - Exempt parts with <= 3 volumes from strict "extreme" block.
        is_toxic = False
        reason = ""
        
        num_vols = len(volumes)
        
        # Calculate size-aware threshold: 
        # Base limit: 5000 edges
        # Scale with diagonal: add 100 edges per mm of diagonal
        # Example: 10mm -> 6000 edges, 200mm -> 25000 edges
        size_aware_limit = 5000 + (diag * 100)
        
        # Absolute overrides:
        # 1. Very small parts with moderate edge counts
        if diag < 10.0 and edge_count > 2000:
            is_toxic = True
            reason = f"Toxic 1D (small/complex): {edge_count} edges for diagonal {diag:.2f}mm"
            
        # 2. Extreme edge counts regardless of size
        elif edge_count > size_aware_limit:
            # But only if NOT a simple assembly (allow complicated single parts or small assemblies)
            if num_vols > 3 or edge_count > 50000:
                is_toxic = True
                reason = f"Toxic 1D: Extreme edge count ({edge_count}) exceeds size-aware limit ({size_aware_limit:.0f})"
        
        # 3. Surgical mode override (even more lenient)
        if self.target_volume_tag is not None:
             if edge_count < 10000:
                 is_toxic = False
             elif edge_count > 100000:
                 is_toxic = True
                 reason = f"Toxic 1D (isolated): Absolute limit exceeded ({edge_count} edges)"
            
        if is_toxic:
            self.log_message(f"[!] {reason}")
        else:
            self.log_message(f"[OK] 1D Complexity passed ({edge_count} edges)")
            
        return {"is_toxic": is_toxic, "reason": reason, "edge_count": edge_count}

    def generate_2d_canary(self, timeout: float = 5.0) -> bool:
        """
        Attempt a 2D surface mesh using a separate subprocess (Process Canary).
        If 2D meshing stalls, it's a precursor to 3D meshing stalls.
        Uses multiprocessing for absolute isolation from the main worker's Gmsh state.
        """
        self.log_message(f"Starting 2D Surface Canary (Timeout: {timeout}s)...")
        
        # Get current mesh parameters for the canary
        cad_file = getattr(self, 'current_filename', None)
        if not cad_file or not os.path.exists(cad_file):
            self.log_message("[!] Canary failed: No CAD file loaded", level="WARNING")
            return True # Don't block if we can't run the canary
            
        results_queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=_canary_worker, args=(cad_file, results_queue))
        p.start()
        p.join(timeout)
        
        if p.is_alive():
            self.log_message(f"[!] 2D Canary TIMED OUT after {timeout}s - Potential geometric hang.")
            p.terminate()
            p.join()
            return False
            
        success = False
        try:
            success = results_queue.get_nowait()
        except:
            pass
            
        if not success:
            self.log_message("[!] 2D Canary Failed within timeout (geometry might be too complex).")
            return False
            
        self.log_message("[OK] 2D Canary passed.")
        return True

    def create_bounding_box_mesh(self, output_file: str) -> bool:
        """
        Create a simplified bounding box mesh using a separate process for isolation.
        """
        self.log_message("Generating Bounding Box B-Rep Fallback (Isolated Process)...")
        
        try:
            # 1. Get bounding box info
            bb = self.geometry_info.get('bounding_box')
            p_min, p_max = None, None
            
            if bb and isinstance(bb, dict) and 'min' in bb and 'max' in bb:
                p_min, p_max = bb['min'], bb['max']
            else:
                # If pre-calc missing, try a quick query (risky but last resort)
                try:
                    res = gmsh.model.getBoundingBox(-1, -1)
                    if res and len(res) == 6:
                        p_min, p_max = [res[0], res[1], res[2]], [res[3], res[4], res[5]]
                except: pass
            
            if p_min is None: p_min, p_max = [0,0,0], [1,1,1]

            # 2. Run fallback in separate process if possible
            current_proc = multiprocessing.current_process()
            can_spawn = not getattr(current_proc, 'daemon', False)
            
            if can_spawn:
                p = multiprocessing.Process(target=_bounding_box_worker, args=(output_file, p_min, p_max))
                p.start()
                p.join(30.0) # 30s timeout for a simple box is plenty
                
                if p.is_alive():
                    p.terminate()
                    self.log_message("[X] Bounding Box worker timed out", level="ERROR")
                    return False
                    
                if p.exitcode == 0:
                    self.log_message(f"[OK] Bounding box mesh saved to {output_file}")
                    return True
                else:
                    self.log_message(f"[X] Bounding Box worker failed with exit code {p.exitcode}", level="ERROR")
                    return False
            else:
                # We are already in a daemonic process (likely a Pool worker)
                # Spawning a child is forbidden, so we run directly.
                # Since we are already isolated in a worker, a crash here is caught higher up.
                self.log_message("[DEBUG] Running bounding box in-process (daemonic isolation active)")
                return _bounding_box_worker(output_file, p_min, p_max)
                
        except Exception as e:
            self.log_message(f"[X] Bounding Box fallback system failed: {e}")
            return False

    def calculate_initial_mesh_parameters(self) -> Dict[str, float]:
        """Calculate initial mesh parameters based on geometry"""
        diagonal = self.geometry_info.get('diagonal', 1.0)
        min_curve = self.geometry_info.get('min_curve_length', diagonal / 20.0)
        has_small_features = self.geometry_info.get('has_small_features', False)

        # Default parameters
        cl_max = diagonal / 20.0
        cl_min = diagonal / 100.0

        # Adjust for small features
        if has_small_features:
            # Make cl_min respect smallest feature (with some margin)
            cl_min = min(cl_min, min_curve / 5.0)
            # But not too small
            cl_min = max(cl_min, diagonal / 500.0)
            self.log_message(f"[OK] Adjusted for small features (min_curve={min_curve:.4f})")

        # Ensure reasonable bounds
        cl_min = max(cl_min, diagonal / 1000.0)  # Not too fine
        cl_max = min(cl_max, diagonal / 5.0)     # Not too coarse
        cl_min = min(cl_min, cl_max / 3.0)       # Reasonable ratio

        params = {
            'cl_min': cl_min,
            'cl_max': cl_max,
            'diagonal': diagonal,
            'min_curve_length': min_curve,
            'refinement_factor': 1.0
        }

        self.log_message(
            f"Initial parameters: CL_min={cl_min:.4f}, CL_max={cl_max:.4f}"
        )

        return params

    def apply_mesh_parameters(self, params: Optional[Dict] = None):
        """Apply mesh parameters to Gmsh"""
        if params is None:
            params = self.current_mesh_params

        # Support both old (cl_min/cl_max) and new (min_size_mm/max_size_mm) parameter names
        # Priority: explicit cl_min/cl_max > min_size_mm/max_size_mm > defaults
        cl_min = params.get('cl_min') or params.get('min_size_mm') or 1.0
        cl_max = params.get('cl_max') or params.get('max_size_mm') or 10.0

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", cl_min)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", cl_max)

        self.log_message(f"Applied mesh parameters: CL_min={cl_min:.4f}, CL_max={cl_max:.4f}")

    def set_mesh_algorithm(self, algorithm_2d: Optional[int] = None,
                           algorithm_3d: Optional[int] = None):
        """Set meshing algorithms"""
        if algorithm_2d is None:
            algorithm_2d = self.config.mesh_params.algorithm_2d
        if algorithm_3d is None:
            algorithm_3d = self.config.mesh_params.algorithm_3d

        gmsh.option.setNumber("Mesh.Algorithm", algorithm_2d)
        gmsh.option.setNumber("Mesh.Algorithm3D", algorithm_3d)

        self.log_message(f"Mesh algorithms: 2D={algorithm_2d}, 3D={algorithm_3d}")

    def set_element_order(self, order: Optional[int] = None):
        """Set element order (1=linear, 2=quadratic)"""
        if order is None:
            order = self.config.mesh_params.element_order

        try:
            gmsh.option.setNumber("Mesh.ElementOrder", order)
            if order == 2:
                gmsh.option.setNumber("Mesh.HighOrderOptimize",
                                      self.config.mesh_params.high_order_optimize)
            self.log_message(f"Element order: {order}")
        except Exception as e:
            self.log_message(f"[!] Could not set element order: {e}")

    def generate_mesh_internal(self, dimension: int = 3) -> bool:
        """Generate mesh (internal method with error handling)"""
        try:
            gmsh.model.mesh.clear()

            # Generate mesh with improved error handling
            try:
                gmsh.model.mesh.generate(dimension)
            except RuntimeError as e:
                # Gmsh throws RuntimeError for access violations and internal crashes
                error_str = str(e).lower()
                if "access violation" in error_str or "exception" in error_str:
                    self.log_message(f"[!] Gmsh internal crash detected: {e}", level="ERROR")
                    self.log_message(f"[!] This is usually caused by threading issues or geometry problems", level="ERROR")
                    self.log_message(f"[!] Solutions: reduce thread count, simplify geometry, or use GPU mesher", level="ERROR")
                # Re-raise to trigger retry logic below
                raise


            # CRITICAL: Validate that 3D elements were actually generated
            if dimension == 3:
                elem_types_3d, elem_tags_3d, _ = gmsh.model.mesh.getElements(dim=3)
                num_3d_elements = sum(len(tags) for tags in elem_tags_3d)

                if num_3d_elements == 0:
                    self.log_message(f"[X] 3D meshing FAILED - No volume elements generated!")
                    self.log_message(f"   Gmsh created surface mesh only (not a volume mesh)")

                    # Check if we have 2D elements
                    elem_types_2d, elem_tags_2d, _ = gmsh.model.mesh.getElements(dim=2)
                    num_2d_elements = sum(len(tags) for tags in elem_tags_2d)
                    if num_2d_elements > 0:
                        self.log_message(f"   2D surface elements: {num_2d_elements} (but we need 3D volume!)")

                    return False
                else:
                    self.log_message(f"[OK] 3D mesh generated successfully: {num_3d_elements} volume elements")
                    return True
            else:
                self.log_message(f"[OK] {dimension}D mesh generated successfully")
                return True

        except Exception as e:
            self.log_message(f"[!] {dimension}D mesh failed: {e}")

            # Try 2D first if 3D failed
            if dimension == 3:
                try:
                    self.log_message("Trying 2D mesh first, then 3D...")
                    gmsh.model.mesh.generate(2)
                    self.log_message("[OK] 2D mesh generated, retrying 3D...")
                    gmsh.model.mesh.generate(3)

                    # Validate 3D elements were created
                    elem_types_3d, elem_tags_3d, _ = gmsh.model.mesh.getElements(dim=3)
                    num_3d_elements = sum(len(tags) for tags in elem_tags_3d)

                    if num_3d_elements == 0:
                        self.log_message(f"[X] 3D meshing FAILED - No volume elements after retry!")
                        return False

                    self.log_message(f"[OK] 3D mesh generated successfully: {num_3d_elements} volume elements")
                    return True
                except Exception as e2:
                    self.log_message(f"ERROR: Mesh generation failed: {e2}", level="ERROR")
                    return False

            return False

    def analyze_current_mesh(self) -> Optional[Dict]:
        """Analyze current mesh quality using Gmsh's built-in metrics"""
        self.log_message("Analyzing mesh quality...")

        metrics = self.quality_analyzer.analyze_mesh(include_advanced_metrics=True)

        if metrics:
            # Log basic statistics
            self.log_message(f"Total elements: {metrics['total_elements']:,}")
            self.log_message(f"Total nodes: {metrics['total_nodes']:,}")

            # PRIMARY: Gmsh's accurate quality metrics
            if metrics.get('gmsh_sicn'):
                s = metrics['gmsh_sicn']
                if s['min'] < 0:
                    quality_assessment = "CRITICAL - INVERTED ELEMENTS"
                elif s['min'] > 0.5:
                    quality_assessment = "EXCELLENT"
                elif s['min'] > 0.3:
                    quality_assessment = "GOOD"
                else:
                    quality_assessment = "POOR"

                self.log_message(
                    f"SICN (Gmsh): Min={s['min']:.4f}, Max={s['max']:.4f}, Avg={s['avg']:.4f} [{quality_assessment}]"
                )

                if s['min'] < 0:
                    self.log_message(f"[!][!][!] MESH HAS INVERTED ELEMENTS - geometry is seriously problematic")

            if metrics.get('gmsh_gamma'):
                g = metrics['gmsh_gamma']
                quality_assessment = "EXCELLENT" if g['min'] > 0.4 else ("GOOD" if g['min'] > 0.2 else "POOR")
                self.log_message(
                    f"Gamma (Gmsh): Min={g['min']:.4f}, Max={g['max']:.4f}, Avg={g['avg']:.4f} [{quality_assessment}]"
                )

            # SECONDARY: Show legacy converted metrics
            if metrics.get('skewness'):
                s = metrics['skewness']
                self.log_message(
                    f"Skewness (converted): Min={s['min']:.4f}, Max={s['max']:.4f}, Avg={s['avg']:.4f}"
                )

            if metrics.get('aspect_ratio'):
                a = metrics['aspect_ratio']
                self.log_message(
                    f"Aspect Ratio (converted): Min={a['min']:.4f}, Max={a['max']:.4f}, Avg={a['avg']:.4f}"
                )

            # COMPARISON: Show our custom calculations vs Gmsh
            if metrics.get('custom_skewness') and metrics.get('gmsh_sicn'):
                custom_s = metrics['custom_skewness']
                self.log_message(
                    f"[DEBUG] Custom skewness: Min={custom_s['min']:.4f}, Max={custom_s['max']:.4f} (compare to Gmsh SICN)"
                )

        return metrics

    def check_quality_targets(self, metrics: Dict) -> bool:
        """Check if quality targets are met"""
        targets = self.config.quality_targets
        targets_met = True

        if metrics.get('skewness'):
            max_skew = metrics['skewness']['max']
            if max_skew > targets.skewness_max:
                self.log_message(f"[!] Skewness: {max_skew:.4f} > {targets.skewness_max}")
                targets_met = False
            else:
                self.log_message(f"[OK] Skewness: {max_skew:.4f} <= {targets.skewness_max}")

        if metrics.get('aspect_ratio'):
            max_aspect = metrics['aspect_ratio']['max']
            if max_aspect > targets.aspect_ratio_max:
                self.log_message(f"[!] Aspect ratio: {max_aspect:.4f} > {targets.aspect_ratio_max}")
                targets_met = False
            else:
                self.log_message(f"[OK] Aspect ratio: {max_aspect:.4f} <= {targets.aspect_ratio_max}")

        return targets_met

    def save_mesh(self, output_file: str) -> bool:
        """Save mesh to file"""
        try:
            # Force ASCII format (easier to parse, more robust for our native parser)
            gmsh.option.setNumber("Mesh.Binary", 0)
            # CRITICAL: Always save all elements (including surface triangles)
            # otherwise physical groups will filter them out and viewer will be empty.
            gmsh.option.setNumber("Mesh.SaveAll", 1)
            gmsh.write(output_file)
            self.log_message(f"[OK] Mesh saved to: {output_file}")
            return True
        except Exception as e:
            self.log_message(f"ERROR: Failed to save mesh: {e}", level="ERROR")
            return False

    def save_iteration_history(self, output_file: str):
        """Save iteration history to JSON file"""
        history_file = os.path.splitext(output_file)[0] + "_history.json"

        try:
            with open(history_file, 'w') as f:
                json.dump({
                    'generator': self.__class__.__name__,
                    'total_iterations': self.current_iteration,
                    'history': self.quality_history,
                    'config': {
                        'quality_targets': self.config.get_quality_targets_dict(),
                        'mesh_params': self.config.get_mesh_params_dict()
                    }
                }, f, indent=2)

            self.log_message(f"[OK] History saved to: {history_file}")

        except Exception as e:
            self.log_message(f"[!] Could not save history: {e}")

    def _get_default_output_path(self, input_file: str) -> str:
        """Generate default output file path"""
        base_name = os.path.splitext(input_file)[0]
        strategy_name = self.__class__.__name__.lower().replace('meshgenerator', '')
        return f"{base_name}_{strategy_name}_optimized.msh"

    def generate_final_report(self):
        """Generate final report"""
        self.log_message("\n" + "=" * 60)
        self.log_message("FINAL MESH GENERATION REPORT")
        self.log_message("=" * 60)

        if not self.quality_history:
            self.log_message("No quality data available")
            return

        # Final metrics
        final = self.quality_history[-1]['metrics']
        self.log_message(f"\nFinal Statistics:")
        self.log_message(f"  Total Iterations: {self.current_iteration}")
        self.log_message(f"  Total Elements: {final['total_elements']:,}")
        self.log_message(f"  Total Nodes: {final['total_nodes']:,}")

        if final.get('skewness'):
            self.log_message(f"  Final Skewness: {final['skewness']['max']:.4f}")

        if final.get('aspect_ratio'):
            self.log_message(f"  Final Aspect Ratio: {final['aspect_ratio']['max']:.4f}")

        # Quality assessment
        self.log_message(f"\nQuality Assessment:")
        targets = self.config.quality_targets

        if final.get('skewness'):
            max_skew = final['skewness']['max']
            if max_skew <= targets.skewness_max:
                self.log_message(f"  [OK] Skewness: EXCELLENT ({max_skew:.4f})")
            elif max_skew <= targets.skewness_max * 1.2:
                self.log_message(f"  [!] Skewness: ACCEPTABLE ({max_skew:.4f})")
            else:
                self.log_message(f"  [X] Skewness: POOR ({max_skew:.4f})")

        if final.get('aspect_ratio'):
            max_aspect = final['aspect_ratio']['max']
            if max_aspect <= targets.aspect_ratio_max:
                self.log_message(f"  [OK] Aspect Ratio: EXCELLENT ({max_aspect:.4f})")
            elif max_aspect <= targets.aspect_ratio_max * 1.5:
                self.log_message(f"  [!] Aspect Ratio: ACCEPTABLE ({max_aspect:.4f})")
            else:
                self.log_message(f"  [X] Aspect Ratio: POOR ({max_aspect:.4f})")

        # AI statistics
        if self.config.is_ai_enabled():
            ai_stats = self.ai_engine.get_statistics()
            self.log_message(f"\nAI Recommendations:")
            self.log_message(f"  Success Rate: {ai_stats['ai_success_rate']:.1f}%")
            self.log_message(f"  Fallback Used: {ai_stats['fallback_used']} times")

        self.log_message("=" * 60)

    def _calculate_quality_score(self, metrics: Dict) -> float:
        """
        Calculate overall quality score (lower is better)

        CRITICAL: Quality > Element Count
        Uses logarithmic element penalty so 50k good elements beats 2k garbage.
        
        Scoring Philosophy:
        - SICN < 0: Immediate disqualification (inverted elements)
        - SICN 0.2-0.3: Acceptable (score ~70)
        - SICN 0.5-0.7: Good (score ~40-50)
        - SICN > 0.7: Excellent (score ~25-30)
        - Element count: Log penalty (50k → +4.7, 2k → +3.3)

        Args:
            metrics: Quality metrics dictionary

        Returns:
            Quality score (lower is better)
        """
        import math
        
        score = 0.0
        targets = self.config.quality_targets

        # PRIMARY: Use Gmsh's SICN (Signed Inverse Condition Number)
        # SICN: 0-1, higher is better. We invert it for scoring (lower score = better)
        if metrics.get('gmsh_sicn'):
            sicn_min = metrics['gmsh_sicn']['min']
            sicn_avg = metrics['gmsh_sicn']['avg']
            
            # CRITICAL: Immediate disqualification for inverted elements
            if sicn_min < 0:
                return 10000.0  # Make this mesh lose to anything
            
            # CRITICAL: Heavily penalize low SICN (poor quality)
            # Scale: SICN 0.1 → 90, SICN 0.3 → 70, SICN 0.5 → 50, SICN 0.7 → 30
            sicn_penalty = (1.0 - sicn_avg) * 100.0  # 0-100 scale
            score += sicn_penalty
            
            # Extra penalty for minimum SICN (worst element)
            if sicn_min < 0.2:
                score += 20.0  # Bad worst element
            elif sicn_min < 0.3:
                score += 10.0  # Marginal worst element

        # SECONDARY: Use Gmsh's Gamma (inscribed/circumscribed radius ratio)
        # Gamma: 0-1, higher is better
        elif metrics.get('gmsh_gamma'):  # Fallback if SICN not available
            gamma_min = metrics['gmsh_gamma']['min']
            gamma_avg = metrics['gmsh_gamma']['avg']
            
            # Similar scaling as SICN
            gamma_penalty = (1.0 - gamma_avg) * 100.0
            score += gamma_penalty
            
            if gamma_min < 0.15:
                score += 20.0
            elif gamma_min < 0.2:
                score += 10.0

        # LEGACY FALLBACK: Use converted skewness/aspect ratio if Gmsh metrics unavailable
        elif metrics.get('skewness'):
            skew_max = metrics['skewness']['max']
            # Skewness: 0-1, lower is better
            # Scale: 0.3 → 30, 0.7 → 70, 0.9 → 90
            score += skew_max * 100.0

        # Element count penalty: LOGARITHMIC, not linear!
        # This is the KEY FIX: Prefer 50k good over 2k garbage
        # log10(2000) ≈ 3.3, log10(50000) ≈ 4.7, log10(500000) ≈ 5.7
        # Difference between good and garbage: ~70 points from quality
        # Difference from element count: ~1.4 points → Quality wins!
        if metrics.get('total_elements'):
            element_count = max(metrics['total_elements'], 1)  # Avoid log(0)
            log_penalty = math.log10(element_count) * 2.0  # Weight 2.0
            score += log_penalty

        return score

    def _create_ansys_physical_groups(self, mode="CFD"):
        """
        Create Physical Groups for ANSYS export.
        
        Args:
            mode: "CFD" or "FEA" - determines naming convention
        """
        self.log_message(f"Creating Physical Groups for {mode}...")
        
        # Clear existing physical groups to avoid conflicts
        gmsh.model.removePhysicalGroups()
        
        # 1. Physical Volume (The Interior)
        volumes = gmsh.model.getEntities(3)
        if volumes:
            p_tag = gmsh.model.addPhysicalGroup(3, [v[1] for v in volumes])
            if mode == "CFD":
                gmsh.model.setPhysicalName(3, p_tag, "FLUID_DOMAIN")
            else:  # FEA
                gmsh.model.setPhysicalName(3, p_tag, "INTERNAL_VOLUME")
            self.log_message(f"Created Physical Volume (Tag {p_tag})")

        # 2. Physical Surfaces (The Boundaries)
        surfaces = gmsh.model.getEntities(2)
        if surfaces:
            p_tag = gmsh.model.addPhysicalGroup(2, [s[1] for s in surfaces])
            if mode == "CFD":
                gmsh.model.setPhysicalName(2, p_tag, "WALL_BOUNDARIES")
            else:  # FEA
                gmsh.model.setPhysicalName(2, p_tag, "SURFACE_BOUNDARIES")
            self.log_message(f"Created Physical Surface (Tag {p_tag})")


