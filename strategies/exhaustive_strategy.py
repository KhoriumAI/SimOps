"""
Exhaustive Mesh Generation Strategy
====================================

This strategy tries EVERYTHING possible before giving up:
- Multiple algorithms (Delaunay, HXT, MMG3D, Frontal, etc.)
- Different element types (tet, hex, prism, hybrid)
- Boundary layer meshing
- Curvature-based refinement
- Anisotropic meshing
- Multiple size field strategies
- Recombination attempts
- Various optimization levels

Use this for difficult geometries where standard approaches fail.

DEFAULT FOCUS: FEA (Finite Element Analysis / Structural Analysis)
- Quadratic elements (order 2) for better accuracy
- Quality metrics optimized for structural simulations
- Boundary layers optional (more critical for CFD)

For CFD applications, consider adjusting:
- Boundary layer thickness and growth rates
- Element sizing near walls
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess
import concurrent.futures
import glob

# Project Structure Constants
PROJECT_ROOT = str(Path(__file__).parent.parent)

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.mesh_generator import BaseMeshGenerator
from core.config import Config
from core.config import Config
from core.geometry_cleanup import GeometryCleanup
from core.winding_resurface import resurface_coacd_output
import gmsh
import json
from datetime import datetime

# Check for supplementary meshers
try:
    from strategies.tetgen_strategy import TetGenMeshGenerator, TETGEN_AVAILABLE
except ImportError:
    TETGEN_AVAILABLE = False

try:
    from strategies.pymesh_strategy import PyMeshMeshGenerator, PYMESH_AVAILABLE
except ImportError:
    PYMESH_AVAILABLE = False


import multiprocessing
import time
import concurrent.futures
import os
from typing import Dict, List, Optional, Tuple

def run_strategy_wrapper(args):
    """
    Worker function to run a single strategy in a separate process.
    Must be top-level for pickling.
    """
    # CRITICAL: Import here so worker processes have access
    # (multiprocessing pickles the function, losing the module imports)
    import sys
    import os
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from strategies.exhaustive_strategy import ExhaustiveMeshGenerator
    from core.config import Config
    import gmsh
    
    strategy_name, input_file, config, stop_event, geometry_info = args
    
    # Check if we should even start
    if stop_event.is_set():
        return (strategy_name, False, {}, "Cancelled before start")
    
    # Create a fresh generator instance
    try:
        # Initialize generator with config and stop_event
        generator = ExhaustiveMeshGenerator(config, stop_event)
        
        # Populate geometry info from orchestrator (in case loading fails)
        if geometry_info:
            generator.geometry_info = geometry_info
        
        # Initialize Gmsh for this process
        generator.initialize_gmsh(thread_count=1)
        
        # [DEBUG] Enable Gmsh output in workers for diagnostics
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("General.Verbosity", 3)
        
        try:
            # Load the file
            load_success = generator.load_cad_file(input_file)
            
            # For bounding box, we can proceed even if load failed (using orchestrator bbox)
            if not load_success and strategy_name != "bounding_box_fallback":
                return (strategy_name, False, {}, "Failed to load CAD file")

            # Find the strategy method
            method_name = f"_try_{strategy_name}"
            method = getattr(generator, method_name, None)
            if not method:
                return (strategy_name, False, {}, f"Unknown strategy method: {method_name}")
            
            # Run the strategy
            success, metrics = method()
            
            # Check stop event again before saving
            if stop_event.is_set() and not success:
                return (strategy_name, False, {}, "Cancelled during execution")
            
            # If successful, save the mesh to a temp file to pass back
            temp_output = f"temp_mesh_{strategy_name}_{os.getpid()}.msh"
            if success:
                generator.save_mesh(temp_output)
                return (strategy_name, True, metrics, temp_output)
            else:
                return (strategy_name, False, metrics, "Strategy returned failure")
                
        finally:
            generator.finalize_gmsh()
            
    except Exception as e:
        return (strategy_name, False, {}, str(e))

class ExhaustiveMeshGenerator(BaseMeshGenerator):
    """
    Exhaustive mesh generator that tries every possible approach
    
    Implements ANSYS-like features:
    - Boundary layers (inflation/growth layers)
    - Curvature adaptation
    - Anisotropic sizing
    - Multiple element types (tet, hex, prism, hybrid)
    - Aggressive optimization
    - Body sizing
    - Edge sizing for sharp features
    """

    def __init__(self, config: Optional[Config] = None, stop_event=None, target_volume_tag: Optional[int] = None):
        super().__init__(config)
        self.strategies_attempted = []
        self.all_attempts = []  # Store ALL attempts, even failed ones
        self.geometry_cleanup = GeometryCleanup()
        self.stop_event = stop_event
        self.target_volume_tag = target_volume_tag

    def check_stop(self) -> bool:
        """Check if we should stop early"""
        if self.stop_event and self.stop_event.is_set():
            self.log_message("[!] Stop signal received, aborting...")
            return True
        return False

    def run_meshing_strategy(self, input_file: str, output_file: str) -> bool:
        """
        Run automatic meshing - optimizes parameters
        """
        self.log_message("\n" + "=" * 60)
        self.log_message("AUTOMATIC MESH GENERATION")
        self.log_message("Optimizing mesh generation parameters...")
        self.log_message("=" * 60)

        # Analyze geometry for defeaturing (this happens in main process once)
        self.log_message("\nAnalyzing geometry...")
        geom_stats = self.geometry_cleanup.analyze_geometry()

        # Define all strategies to try
        # Use custom order from config if provided
        default_strategy_names = [
            "ultra_stable_tet", # High priority for complex geometry
            "tet_delaunay_optimized",
            "tet_frontal_optimized",
            "tet_hxt_optimized",
            "tet_mmg3d_optimized",
            "tet_with_boundary_layers",
            "anisotropic_curvature",
            "hybrid_prism_tet",
            "hybrid_hex_tet",
            "recombined_to_hex",
            "transfinite_structured",
            "very_coarse_tet",
            "adaptive_coarse_to_fine",
            "linear_tet_delaunay",
            "linear_tet_frontal",
            "subdivide_and_mesh",
            "automatic_gmsh_default",
            "resurfacing_reconstruction",
            "bounding_box_fallback" # Absolute last resort
        ]
        
        # Use custom strategy order from config if provided
        using_custom_order = False
        if hasattr(self.config.mesh_params, 'strategy_order') and self.config.mesh_params.strategy_order:
            strategy_names = list(self.config.mesh_params.strategy_order)  # Make a copy to avoid mutation
            using_custom_order = True
            self.log_message(f"Using custom strategy order: {strategy_names}")
        else:
            strategy_names = list(default_strategy_names)  # Make a copy to avoid mutation
        
        # Get score threshold from config
        score_threshold = getattr(self.config.mesh_params, 'score_threshold', 50.0)

        # Only add supplementary meshers if NOT using custom strategy order
        # (User's custom selection should be respected exactly)
        if not using_custom_order:
            if TETGEN_AVAILABLE:
                strategy_names.append("tetgen")
            if PYMESH_AVAILABLE:
                strategy_names.append("pymesh")
        
        # ANSYS FLUENT COMPATIBILITY: Filter out hex/poly strategies
        # Fluent export only supports tetrahedral meshes
        ansys_mode = getattr(self.config.mesh_params, 'ansys_mode', 'None')
        if ansys_mode and ansys_mode != "None":
            # List of strategies that produce non-tetrahedral elements
            hex_poly_strategies = [
                "hybrid_hex_tet",
                "recombined_to_hex",
                "transfinite_structured",
                "subdivide_and_mesh",  # Produces hex elements
                "resurfacing_reconstruction",  # Might produce non-tet
            ]
            
            original_count = len(strategy_names)
            strategy_names = [s for s in strategy_names if s not in hex_poly_strategies]
            filtered_count = original_count - len(strategy_names)
            
            if filtered_count > 0:
                self.log_message(f"[ANSYS] Filtered out {filtered_count} hex/poly strategies (Fluent requires tets)")
                self.log_message(f"[ANSYS] Remaining strategies: {len(strategy_names)}")

        # Determine number of workers
        # Use up to 65% of available cores for parallel meshing
        # Cap at 6 workers to prevent Windows pool shutdown hangs
        # (8 workers caused deadlocks during graceful termination)
        total_cores = multiprocessing.cpu_count()
        max_workers = max(1, min(int(total_cores * 0.65), 6))
        
        # Allow override via environment variable for advanced tuning
        if 'MESH_MAX_WORKERS' in os.environ:
            try:
                override = int(os.environ['MESH_MAX_WORKERS'])
                if 1 <= override <= total_cores:
                    max_workers = override
                    self.log_message(f"Using MESH_MAX_WORKERS override: {max_workers}")
            except ValueError:
                pass
        
        num_workers = max_workers
        
        self.log_message(f"Starting optimization pool with {num_workers} workers (65% of {total_cores} cores, capped at 6)")
        self.log_message(f"Estimated RAM usage: ~{num_workers * 500}MB during meshing")
        
        # --- EARLY CANARY CHECKS ---
        # Run canaries once BEFORE spawning workers to avoid redundant checks
        self.log_message("\n[CANARY] Running pre-flight geometry checks...")
        
        # Define assembly threshold (assembly-aware canary)
        num_vols = self.geometry_info.get('num_volumes', 0)
        diag = self.geometry_info.get('diagonal', 1.0)
        
        # DEFINITION: 
        # is_assembly: True if it's a true multi-part assembly (> 3 volumes)
        # is_large_part: True if > 20mm, which deserves a meshing attempt even if complex
        is_assembly = num_vols > 3
        is_large_part = diag > 20.0
        
        complexity = self.check_1d_complexity()
        if complexity['is_toxic']:
            # ORCHESTRATOR-LEVEL OVERRIDE:
            # We only fail-fast (bbox) if it's a small complex part or a massive assembly.
            # If it's a large part (> 20mm) or has few volumes (<= 3), we try to mesh it anyway
            # UNLESS it's truly pathological (> 100k edges).
            
            should_fail_fast = True
            
            if is_large_part or num_vols <= 3:
                # Lenient for large/simple-assembly parts
                if complexity['edge_count'] < 100000:
                    should_fail_fast = False
                    self.log_message(f"[CANARY] Overriding 1D toxicity for large/simple part (Diag: {diag:.2f}mm, Vols: {num_vols}). Attempting meshing...")
            
            if should_fail_fast:
                self.log_message(f"[!] Geometry failed early 1D canary: {complexity['reason']}")
                self.log_message("[!] Skipping optimization - creating bounding box fallback...")
                return self.create_bounding_box_mesh(output_file)
        
        # 2D Surface Canary - ONLY for single parts (<= 3 volumes)
        # For assemblies, meshing ALL surfaces at once is slow and redundant
        if num_vols <= 3:
            # For complex but large parts, increase 2D timeout
            canary_timeout = 10.0 if complexity['edge_count'] > 5000 else 5.0
            if not self.generate_2d_canary(timeout=canary_timeout):
                self.log_message("[!] Geometry failed 2D canary (surface mesh timeout after 5.0s)")
                self.log_message("[!] Skipping optimization - creating bounding box fallback...")
                return self.create_bounding_box_mesh(output_file)
            self.log_message("[OK] Canary checks passed - geometry is meshable")
        else:
            self.log_message("[CANARY] Assembly detected - skipping global 2D surface canary (isolated parts will be checked individually)")
            self.log_message("[OK] Orchestrator-level canary passed")
        
        # --- SURGICAL ISOLATION TRIGGER ---
        num_vols = self.geometry_info.get('num_volumes', 0)
        use_surgical = num_vols > 3
        if use_surgical:
            self.log_message(f"\n[SURGICAL] Assembly detected ({num_vols} volumes). Initializing Surgical Amputation Loop...")
            self.log_message(f"[SURGICAL] Bypassing {num_workers}-worker optimization (surgical uses 6 dedicated workers)")
            return self._run_surgical_isolation_loop(input_file, output_file)

        best_mesh_path = None
        best_score = float('inf')
        best_strategy = None
        
        # OPTIMIZATION: Sequential mode is now DISCOURAGED even for single parts
        # because C++ level crashes (SIGSEGV/SIGABRT) in Gmsh/OCC kill the main process.
        # We enforce process isolation for ALL strategies to ensure fallback triggers.
        use_sequential = False 
        
        if use_sequential:
            # (Dead code path kept for potential future local-only debugging toggle)
            pass
        else:
            # PARALLEL MODE: Use multiprocessing Pool for multiple strategies
            # CRITICAL: Use Manager to create a Windows-safe picklable Event proxy
            with multiprocessing.Manager() as manager:
                stop_event = manager.Event()
                worker_args = [(name, input_file, self.config, stop_event, self.geometry_info) for name in strategy_names]
                
                pool = None
                try:
                    pool = multiprocessing.Pool(processes=num_workers)
                    
                    # Use imap_unordered to process results as they arrive
                    # Wrap iteration in try-except to catch worker process crashes (SIGABRT/SIGSEGV)
                    results_iter = pool.imap_unordered(run_strategy_wrapper, worker_args)
                    
                    while True:
                        try:
                            # next(results_iter) will raise StopIteration when done
                            res = next(results_iter)
                            if not res: continue
                            strategy_name, success, metrics, result_data = res
                        except StopIteration:
                            break
                        except Exception as pool_err:
                            self.log_message(f"[!] Critical worker crash detected (e.g. SIGABRT/SIGSEGV): {pool_err}")
                            continue 
                            
                        # If we already found a winner, we are just draining the pool
                        if stop_event.is_set():
                            continue
                            
                        self.log_message(f"Strategy finished: {strategy_name} (Success={success})")
                        
                        attempt_data = {
                            'strategy': strategy_name,
                            'success': success,
                            'metrics': metrics
                        }
                        
                        if success:
                            temp_mesh_file = result_data
                            score = self._calculate_quality_score(metrics)
                            attempt_data['score'] = score
                            self.log_message(f"  Quality Score: {score:.2f}")
                            
                            if self.check_quality_targets(metrics):
                                self.log_message(f"\n[!!!] WINNER FOUND: {strategy_name} meets quality targets!")
                                self.log_message("Signaling other strategies to stop (Graceful Shutdown)...")
                                
                                # Signal stop - this safely signals all workers via the Manager proxy
                                stop_event.set()
                                
                                best_mesh_path = temp_mesh_file
                                best_score = score
                                best_strategy = strategy_name
                                self.all_attempts.append(attempt_data)
                                break # Exit loop and proceed to cleanup
                            
                            # Check if score is below threshold (user-settable early stop)
                            if score < score_threshold:
                                self.log_message(f"\n[!!!] WINNER FOUND: {strategy_name} score {score:.2f} < threshold {score_threshold}!")
                                self.log_message("Signaling other strategies to stop (Score Threshold)...")
                                
                                stop_event.set()
                                
                                best_mesh_path = temp_mesh_file
                                best_score = score
                                best_strategy = strategy_name
                                self.all_attempts.append(attempt_data)
                                break
                            
                            # Keep as candidate
                            if score < best_score:
                                best_score = score
                                best_mesh_path = temp_mesh_file
                                best_strategy = strategy_name
                                self.log_message(f"  New best candidate (score {score:.2f})")
                                
                        else:
                            attempt_data['error'] = result_data
                            attempt_data['score'] = float('inf')
                        
                        self.all_attempts.append(attempt_data)
                    
                    # Graceful shutdown strategy:
                    # 1. We already signaled stop_event.set() if we broke the loop.
                    # 2. Close the pool to prevent new work from starting
                    # 3. Workers will check stop_event and exit gracefully
                    pool.close()
                    pool.join()
                    
                    
                except Exception as e:
                    self.log_message(f"[!] Execution error: {e}")
                    if pool:
                        pool.terminate() # Fallback to hard kill on error
                        pool.join()
                
        # --- LAST DITCH SAFETY FALLBACK ---
        # If no winners and no candidates after the optimization, 
        # try the bounding box fallback ONE LAST TIME in the main process.
        if not best_mesh_path:
            self.log_message("\n[SAFETY] No successful strategy found. Attempting main-process last-ditch fallback...")
            last_ditch_file = f"last_ditch_bbox_{os.getpid()}.msh"
            if self.create_bounding_box_mesh(last_ditch_file):
                best_mesh_path = last_ditch_file
                best_score = 999.0
                best_strategy = "last_ditch_bounding_box"
                self.all_attempts.append({
                    'strategy': best_strategy,
                    'success': True,
                    'score': best_score,
                    'metrics': {
                        'total_elements': 12,
                        'total_nodes': 8,
                        'gmsh_sicn': {'min': 1.0, 'max': 1.0, 'avg': 1.0},
                        'strategy': 'last_ditch_bounding_box'
                    }
                })
                
        # Process the winner
        if best_mesh_path and os.path.exists(best_mesh_path):
            self.log_message(f"\n{'='*60}")
            self.log_message(f"BEST STRATEGY: {best_strategy} (score: {best_score:.2f})")
            self.log_message(f"{'='*60}")

            # Finalize the mesh
            gmsh.initialize(['-terminal', '1'])
            try:
                gmsh.model.add("Finalizer")
                gmsh.merge(best_mesh_path)
                
                # Create Physical Groups
                volumes = gmsh.model.getEntities(3)
                if volumes and not gmsh.model.getPhysicalGroups(3):
                    p_tag = gmsh.model.addPhysicalGroup(3, [v[1] for v in volumes])
                    gmsh.model.setPhysicalName(3, p_tag, "Volume")
                    
                surfaces = gmsh.model.getEntities(2)
                if surfaces and not gmsh.model.getPhysicalGroups(2):
                    p_tag = gmsh.model.addPhysicalGroup(2, [s[1] for s in surfaces])
                    gmsh.model.setPhysicalName(2, p_tag, "Surface")
                    
                gmsh.option.setNumber("Mesh.SaveAll", 1)
                gmsh.write(output_file)
                
            finally:
                gmsh.finalize()
                
            # Cleanup
            try:
                os.remove(best_mesh_path)
            except:
                pass
            
            # Finalize the result metadata
            # Finalize the result metadata
            self.quality_history = self.all_attempts.copy()

            self._generate_exhaustive_report(output_file)
            self._save_detailed_report_to_file(output_file, success=True)
            return True
            
        else:
            self.log_message("\n[X][X][X] ALL STRATEGIES FAILED [X][X][X]")
            self._generate_failure_report()
            self._save_detailed_report_to_file(output_file, success=False)
            return False

    # Strategy implementations

    def _apply_painted_refinement(self):
        """
        Apply mesh refinement based on painted regions using Gmsh fields.
        Uses MathEval fields to avoid modifying geometry.
        """
        if not hasattr(self.config, 'painted_regions') or not self.config.painted_regions:
            return

        self.log_message(f"Applying paintbrush refinement to {len(self.config.painted_regions)} regions...")
        
        try:
            diagonal = self.geometry_info.get('diagonal', 1.0)
            base_size = diagonal / 20.0 # Approximate base size
            
            fields = []
            
            for i, region in enumerate(self.config.painted_regions):
                center = region.get('center')
                radius = region.get('radius')
                
                if not center or not radius:
                    self.log_message(f"[!] Region {i+1}: Missing center or radius, skipping")
                    continue
                
                # CRITICAL: Validate coordinates to prevent Gmsh parse errors
                try:
                    xc, yc, zc = center
                    
                    # DEBUG: Log the actual values
                    self.log_message(f"[DEBUG] Region {i+1}: center=({xc}, {yc}, {zc}), radius={radius}")
                    
                    # Check for NaN, Infinity, or non-numeric values
                    if not all(isinstance(v, (int, float)) and not (v != v or abs(v) == float('inf')) for v in [xc, yc, zc, radius]):
                        self.log_message(f"[!] Region {i+1}: Invalid coordinates (NaN/Inf), skipping")
                        continue
                    
                    # Ensure radius is positive and reasonable
                    if radius <= 0 or radius > diagonal * 10:
                        self.log_message(f"[!] Region {i+1}: Invalid radius {radius} (diagonal={diagonal}), skipping")
                        continue
                    
                    # Check if coordinates are very large (could cause issues)
                    if any(abs(v) > 1e6 for v in [xc, yc, zc]):
                        self.log_message(f"[!] Region {i+1}: Coordinates too large, skipping")
                        continue
                        
                except (ValueError, TypeError) as e:
                    self.log_message(f"[!] Region {i+1}: Coordinate error: {e}, skipping")
                    continue
                    
                # 1. Create distance field using MathEval
                # F = Sqrt((x-xc)^2 + (y-yc)^2 + (z-zc)^2)
                dist_field = gmsh.model.mesh.field.add("MathEval")
                
                # CRITICAL: Wrap negative coordinates in parentheses to avoid double-minus issue
                # Example: x--14.5 becomes x-(-14.5) which Gmsh can parse correctly
                xc_str = f"({xc:.6f})" if xc < 0 else f"{xc:.6f}"
                yc_str = f"({yc:.6f})" if yc < 0 else f"{yc:.6f}"
                zc_str = f"({zc:.6f})" if zc < 0 else f"{zc:.6f}"
                
                expr = f"Sqrt((x-{xc_str})*(x-{xc_str}) + (y-{yc_str})*(y-{yc_str}) + (z-{zc_str})*(z-{zc_str}))"
                
                self.log_message(f"[DEBUG] Region {i+1}: Creating field with expr: {expr[:80]}...")
                
                try:
                    gmsh.model.mesh.field.setString(dist_field, "F", expr)
                    self.log_message(f"[DEBUG] Region {i+1}: Field created successfully")
                except Exception as e:
                    self.log_message(f"[!] Region {i+1}: Failed to set MathEval expression: {e}")
                    self.log_message(f"[!] Expression was: {expr}")
                    continue
                
                # 2. Create Threshold field
                thresh_field = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(thresh_field, "IField", dist_field)
                
                # Refinement parameters
                # SizeMin: Size inside the painted region (very fine)
                # SizeMax: Size outside (default)
                # DistMin: Radius where SizeMin applies
                # DistMax: Radius where transition to SizeMax ends
                
                # Refine factor: how much finer than base size?
                # Let's say 5x finer for now, or use a fixed small value
                target_min = base_size / 5.0
                
                gmsh.model.mesh.field.setNumber(thresh_field, "SizeMin", target_min)
                gmsh.model.mesh.field.setNumber(thresh_field, "SizeMax", base_size)
                gmsh.model.mesh.field.setNumber(thresh_field, "DistMin", radius * 0.8)
                gmsh.model.mesh.field.setNumber(thresh_field, "DistMax", radius * 1.5)
                
                fields.append(thresh_field)
                
            if fields:
                # 3. Combine all fields using Min
                min_field = gmsh.model.mesh.field.add("Min")
                gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", fields)
                
                # Set as background mesh
                gmsh.model.mesh.field.setAsBackgroundMesh(min_field)
                self.log_message(f"[OK] Applied refinement fields for {len(fields)} regions")
            else:
                self.log_message(f"[!] No valid refinement fields created")
                
        except Exception as e:
            self.log_message(f"[!] Failed to apply painted refinement: {e}")
            import traceback
            self.log_message(f"[!] Traceback: {traceback.format_exc()}")

    def _try_resurfacing_reconstruction(self) -> Tuple[bool, Optional[Dict]]:
        """
        Winding Number Resurfacing Strategy
        
        Uses CoACD decomposition followed by Barnes-Hut winding number field
        and Marching Cubes to reconstruct a watertight surface, then meshes it.
        Ideal for dirty geometry with gaps, overlaps, or self-intersections.
        """
        self.log_message("Resurfacing Reconstruction (Winding Number)...")
        
        try:
            # 1. Run resurfacing pipeline
            # This creates a new watertight STL file
            resurfaced_stl = str(Path(self.temp_dir) / "resurfaced_watertight.stl")
            
            # We need an input mesh. If we have a CAD file, we might need to convert it first.
            # But resurface_coacd_output handles conversion if needed? No, it expects a mesh path.
            # For now, we'll try to use the input file directly if it's a mesh, or convert if it's STEP.
            
            input_file = self.current_mesh_params.get('input_file') # Wait, where is input_file stored?
            # It's passed to run_meshing_strategy but not stored in self.
            # However, run_strategy_wrapper passes it to the worker.
            # But inside the worker, we are in _try_... methods which don't take args.
            
            # We need to access the input file.
            # Actually, we can just use the loaded model in Gmsh to export an STL for resurfacing.
            temp_input_stl = str(Path(self.temp_dir) / "temp_input_for_resurface.stl")
            gmsh.write(temp_input_stl)
            
            success = resurface_coacd_output(
                input_mesh_path=temp_input_stl,
                output_mesh_path=resurfaced_stl,
                resolution=64, # Medium resolution for speed
                threshold=0.5
            )
            
            if not success:
                self.log_message("Resurfacing failed to generate watertight mesh")
                return False, None
                
            # 2. Mesh the resurfaced STL
            # Clear current model and load the new STL
            gmsh.clear()
            gmsh.merge(resurfaced_stl)
            
            # 3. Classify surfaces to recover sharp features
            gmsh.model.mesh.classifySurfaces(40 * 3.14159 / 180, True, True)
            gmsh.model.mesh.createGeometry()
            
            # 4. Generate Volume
            s = gmsh.model.getEntities(2)
            l = gmsh.model.geo.addSurfaceLoop([e[1] for e in s])
            gmsh.model.geo.addVolume([l])
            gmsh.model.geo.synchronize()
            
            # 5. Mesh 3D
            gmsh.option.setNumber("Mesh.Algorithm3D", 10) # HXT
            gmsh.model.mesh.generate(3)
            
            # 6. Optimize
            gmsh.model.mesh.optimize("Netgen")
            
            return True, self.analyze_current_mesh()
            
        except Exception as e:
            self.log_message(f"Resurfacing strategy failed: {e}")
            return False, None

    def _try_bounding_box_fallback(self) -> Tuple[bool, Optional[Dict]]:
        """
        Absolute last resort: Create a bounding box mesh.
        """
        self.log_message("\n[Strategy] Starting Bounding Box Fallback...")
        temp_file = f"bbox_temp_{os.getpid()}.msh"
        if self.create_bounding_box_mesh(temp_file):
            # CRITICAL: Merge the mesh back into the current process's Gmsh state
            # so that save_mesh() in the orchestrator/wrapper works.
            try:
                gmsh.merge(temp_file)
                # Cleanup the temp file
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                
                # Analyze the box mesh to get proper metrics (count, sicn, etc.)
                metrics = self.analyze_current_mesh()
                return True, metrics
            except Exception as e:
                self.log_message(f"Failed to analyze bounding box: {e}")
                return True, {"num_elements": 12, "strategy": "bounding_box"}
        return False, {}

    def _try_ultra_stable_tet(self) -> Tuple[bool, Optional[Dict]]:
        """
        ULTRA-STABLE fallback using MeshAdapt and single-threading.
        Used for the most complex/toxic geometries that crash HXT.
        """
        self.log_message("\n[Strategy] Starting Ultra-Stable Tet Meshing (MeshAdapt + Single-Thread)")
        try:
            # 1. Enforce absolute stability
            gmsh.option.setNumber("General.NumThreads", 1)  # Disable OpenMP
            gmsh.option.setNumber("Geometry.OCCAutoFix", 0) # Disable unstable auto-healing
            gmsh.option.setNumber("Geometry.Tolerance", 1e-2)
            
            # 2. Algorithm 1 (MeshAdapt) is the most robust for difficult CAD
            gmsh.option.setNumber("Mesh.Algorithm", 1) 
            gmsh.option.setNumber("Mesh.Algorithm3D", 1) # Standard Delaunay 3D (HXT can crash)
            
            # 3. Lenient meshing options
            gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", 0.9)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1) # Reduce complexity
            
            # 4. Clear and Generate
            gmsh.model.mesh.clear()
            gmsh.model.mesh.generate(3)
            
            return True, self.analyze_current_mesh()
        except Exception as e:
            self.log_message(f"[!] Ultra-Stable strategy failed: {e}")
            return False, {}

    def _try_tet_delaunay_optimized(self) -> Tuple[bool, Optional[Dict]]:
        """
        Standard strategy (now using HXT for speed/robustness)
        Renamed from 'Delaunay' but keeps the name for compatibility
        """
        self.log_message("Tetrahedral HXT (Fast-Path) with standard optimization...")

        # Build mesh parameters from config (max_size_mm/min_size_mm from GUI)
        mesh_params = {
            'max_size_mm': self.config.mesh_params.max_size_mm,
            'min_size_mm': self.config.mesh_params.min_size_mm
        }
        self.apply_mesh_parameters(mesh_params)

        gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay 2D
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT (Parallel, Robust)
        gmsh.option.setNumber("Mesh.ElementOrder", self.config.mesh_params.element_order)  # User-selected

        # Apply paintbrush refinement if available
        self._apply_painted_refinement()

        # Standard optimization only (Disable Netgen)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 0) # Disable slow Netgen
        gmsh.option.setNumber("Mesh.Smoothing", 10)

        return self._generate_and_analyze()

    def _try_tet_frontal_optimized(self) -> Tuple[bool, Optional[Dict]]:
        """Frontal algorithm with optimization"""
        self.log_message("Tetrahedral Frontal-Delaunay 3D...")

        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # Frontal-Delaunay 3D
        gmsh.option.setNumber("Mesh.ElementOrder", self.config.mesh_params.element_order)

        # Apply paintbrush refinement if available
        self._apply_painted_refinement()

        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.Smoothing", 15)

        return self._generate_and_analyze()

    def _try_tet_hxt_optimized(self) -> Tuple[bool, Optional[Dict]]:
        """HXT algorithm (good for quality)"""
        self.log_message("Tetrahedral HXT (quality-focused)...")

        # PRODUCTION SETTINGS: Prevent 4-million element explosion
        diagonal = self.geometry_info.get('diagonal', 100.0)
        gmsh.option.setNumber("Mesh.MeshSizeMin", diagonal / 200.0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", diagonal / 10.0)

        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT (Correct ID is 10)
        gmsh.option.setNumber("Mesh.ElementOrder", self.config.mesh_params.element_order)

        # Apply paintbrush refinement if available
        self._apply_painted_refinement()

        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 0) # Disable slow Netgen
        gmsh.option.setNumber("Mesh.Smoothing", 15)

        return self._generate_and_analyze()

    def _try_tet_mmg3d_optimized(self) -> Tuple[bool, Optional[Dict]]:
        """MMG3D algorithm (good for aspect ratio)"""
        self.log_message("Tetrahedral MMG3D (aspect ratio optimization)...")

        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Algorithm3D", 7)  # MMG3D
        gmsh.option.setNumber("Mesh.ElementOrder", self.config.mesh_params.element_order)

        # Apply paintbrush refinement if available
        self._apply_painted_refinement()

        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.Smoothing", 15)

        return self._generate_and_analyze()

    def _try_tet_with_boundary_layers(self) -> Tuple[bool, Optional[Dict]]:
        """Tetrahedral with boundary layer meshing (inflation layers)"""
        self.log_message("Tetrahedral with boundary layers (ANSYS-style inflation)...")

        try:
            # Set up boundary layer
            surfaces = gmsh.model.getEntities(dim=2)

            # Create boundary layer field
            field_tag = gmsh.model.mesh.field.add("BoundaryLayer")

            # Get all surface tags
            surf_tags = [tag for dim, tag in surfaces]

            gmsh.model.mesh.field.setNumbers(field_tag, "FacesList", surf_tags)
            gmsh.model.mesh.field.setNumber(field_tag, "Size",
                                            self.geometry_info.get('diagonal', 1.0) / 200.0)
            gmsh.model.mesh.field.setNumber(field_tag, "Ratio", 1.3)  # Growth rate
            gmsh.model.mesh.field.setNumber(field_tag, "Thickness",
                                            self.geometry_info.get('diagonal', 1.0) / 50.0)
            gmsh.model.mesh.field.setNumber(field_tag, "NumLayers", 3)  # 3 layers

            gmsh.model.mesh.field.setAsBoundaryLayer(field_tag)

            self.log_message("[OK] Boundary layers configured (3 layers, 1.3 growth rate)")

            # Standard tet mesh for core
            gmsh.option.setNumber("Mesh.Algorithm", 6)
            gmsh.option.setNumber("Mesh.Algorithm3D", 1)
            gmsh.option.setNumber("Mesh.ElementOrder", self.config.mesh_params.element_order)
            gmsh.option.setNumber("Mesh.Smoothing", 15)

            return self._generate_and_analyze()

        except Exception as e:
            self.log_message(f"[!] Boundary layer setup failed: {e}")
            return False, None

    def _try_anisotropic_curvature(self) -> Tuple[bool, Optional[Dict]]:
        """Anisotropic meshing based on curvature (ANSYS-like curvature adaptation)"""
        self.log_message("Anisotropic meshing with curvature adaptation...")

        try:
            # Create curvature-based size field
            field_tag = gmsh.model.mesh.field.add("Curvature")
            diagonal = self.geometry_info.get('diagonal', 1.0)

            gmsh.model.mesh.field.setNumber(field_tag, "NNodesByEdge", 100)  # High resolution
            gmsh.model.mesh.field.setNumber(field_tag, "SizeMin", diagonal / 500.0)
            gmsh.model.mesh.field.setNumber(field_tag, "SizeMax", diagonal / 20.0)

            gmsh.model.mesh.field.setAsBackgroundMesh(field_tag)

            self.log_message("[OK] Curvature-based size field active")

            # Enable anisotropic meshing
            gmsh.option.setNumber("Mesh.AnisoMax", 100)  # Allow high anisotropy
            gmsh.option.setNumber("Mesh.Algorithm", 6)
            gmsh.option.setNumber("Mesh.Algorithm3D", 1)
            gmsh.option.setNumber("Mesh.ElementOrder", self.config.mesh_params.element_order)

            return self._generate_and_analyze()

        except Exception as e:
            self.log_message(f"[!] Anisotropic meshing failed: {e}")
            return False, None

    def _try_hybrid_prism_tet(self) -> Tuple[bool, Optional[Dict]]:
        """Hybrid mesh: Prisms at boundaries, tets in core"""
        self.log_message("Hybrid mesh: Prisms + Tetrahedra...")

        try:
            # Create boundary layer with prisms
            surfaces = gmsh.model.getEntities(dim=2)
            surf_tags = [tag for dim, tag in surfaces]

            field_tag = gmsh.model.mesh.field.add("BoundaryLayer")
            gmsh.model.mesh.field.setNumbers(field_tag, "FacesList", surf_tags)
            gmsh.model.mesh.field.setNumber(field_tag, "Size",
                                            self.geometry_info.get('diagonal', 1.0) / 150.0)
            gmsh.model.mesh.field.setNumber(field_tag, "Ratio", 1.2)
            gmsh.model.mesh.field.setNumber(field_tag, "Thickness",
                                            self.geometry_info.get('diagonal', 1.0) / 40.0)
            gmsh.model.mesh.field.setNumber(field_tag, "NumLayers", 2)
            gmsh.model.mesh.field.setNumber(field_tag, "CreatePrisms", 1)  # Prisms!

            gmsh.model.mesh.field.setAsBoundaryLayer(field_tag)

            self.log_message("[OK] Prism boundary layers configured")

            gmsh.option.setNumber("Mesh.Algorithm", 6)
            gmsh.option.setNumber("Mesh.Algorithm3D", 1)
            gmsh.option.setNumber("Mesh.ElementOrder", self.config.mesh_params.element_order)

            return self._generate_and_analyze()

        except Exception as e:
            self.log_message(f"[!] Hybrid prism mesh failed: {e}")
            return False, None

    def _try_hybrid_hex_tet(self) -> Tuple[bool, Optional[Dict]]:
        """Hybrid: Recombine to hexahedra where possible, tets elsewhere"""
        self.log_message("Hybrid hex-tet mesh (recombination)...")

        try:
            gmsh.option.setNumber("Mesh.Algorithm", 8)  # Delaunay for quads
            gmsh.option.setNumber("Mesh.RecombineAll", 1)  # Try to recombine
            gmsh.option.setNumber("Mesh.Algorithm3D", 1)
            gmsh.option.setNumber("Mesh.Recombine3DAll", 1)  # 3D recombination
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3)  # Simple full-quad
            gmsh.option.setNumber("Mesh.ElementOrder", self.config.mesh_params.element_order)

            self.log_message("[OK] Recombination to hex enabled")

            return self._generate_and_analyze()

        except Exception as e:
            self.log_message(f"[!] Hex-tet hybrid failed: {e}")
            return False, None

    def _try_recombined_to_hex(self) -> Tuple[bool, Optional[Dict]]:
        """Pure hexahedral (if geometry allows)"""
        self.log_message("Attempting pure hexahedral mesh...")

        try:
            gmsh.option.setNumber("Mesh.Algorithm", 8)
            gmsh.option.setNumber("Mesh.RecombineAll", 1)
            gmsh.option.setNumber("Mesh.Algorithm3D", 1)
            gmsh.option.setNumber("Mesh.Recombine3DAll", 1)
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)  # Blossom
            gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)  # All quads
            gmsh.option.setNumber("Mesh.ElementOrder", self.config.mesh_params.element_order)

            return self._generate_and_analyze()

        except Exception as e:
            self.log_message(f"[!] Pure hex mesh failed: {e}")
            return False, None

    def _try_transfinite_structured(self) -> Tuple[bool, Optional[Dict]]:
        """Structured transfinite mesh (if geometry is simple enough)"""
        self.log_message("Attempting structured transfinite mesh...")

        try:
            # Try to apply transfinite to all curves and surfaces
            curves = gmsh.model.getEntities(dim=1)
            surfaces = gmsh.model.getEntities(dim=2)
            volumes = gmsh.model.getEntities(dim=3)

            # Set transfinite on curves
            failed_curves = 0
            for dim, tag in curves:
                try:
                    gmsh.model.mesh.setTransfiniteCurve(tag, 10)
                except Exception as e:
                    failed_curves += 1

            # Set transfinite on surfaces (with hole detection)
            failed_surfaces = 0
            surfaces_with_holes = 0
            for dim, tag in surfaces:
                try:
                    gmsh.model.mesh.setTransfiniteSurface(tag)
                    gmsh.model.mesh.setRecombine(dim, tag)
                except Exception as e:
                    error_msg = str(e).lower()
                    if "hole" in error_msg:
                        surfaces_with_holes += 1
                        self.log_message(f"  [!] Surface {tag} has holes, skipping transfinite")
                    else:
                        failed_surfaces += 1

            # Set transfinite on volumes
            failed_volumes = 0
            for dim, tag in volumes:
                try:
                    gmsh.model.mesh.setTransfiniteVolume(tag)
                except Exception as e:
                    failed_volumes += 1

            # If too many failures, this approach won't work
            if surfaces_with_holes > 0:
                self.log_message(f"[!] Transfinite failed: {surfaces_with_holes} surface(s) have holes")
                self.log_message("  Note: Transfinite meshing requires simple surfaces without holes")
                self.log_message("  Skipping this strategy for this geometry")
                return False, None

            if failed_surfaces > len(surfaces) / 2:
                self.log_message(f"[!] Too many surfaces failed transfinite ({failed_surfaces}/{len(surfaces)})")
                return False, None

            gmsh.option.setNumber("Mesh.ElementOrder", self.config.mesh_params.element_order)

            self.log_message("[OK] Transfinite constraints applied")

            return self._generate_and_analyze()

        except Exception as e:
            error_msg = str(e)
            if "hole" in error_msg.lower():
                self.log_message(f"[!] Transfinite mesh failed: Geometry has surfaces with holes")
                self.log_message("  Transfinite meshing requires simple, hole-free surfaces")
            else:
                self.log_message(f"[!] Transfinite mesh failed: {e}")
            return False, None

    def _try_very_coarse_tet(self) -> Tuple[bool, Optional[Dict]]:
        """Very coarse tetrahedral mesh"""
        self.log_message("Very coarse tetrahedral mesh (low element count)...")

        diagonal = self.geometry_info.get('diagonal', 1.0)

        # Very coarse
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", diagonal / 10.0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", diagonal / 3.0)

        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)  # Linear for coarse
        gmsh.option.setNumber("Mesh.Smoothing", 10)

        return self._generate_and_analyze()

    def _try_adaptive_coarse_to_fine(self) -> Tuple[bool, Optional[Dict]]:
        """Adaptive refinement from coarse base"""
        self.log_message("Adaptive coarse-to-fine refinement...")

        try:
            diagonal = self.geometry_info.get('diagonal', 1.0)
            volume = self.geometry_info.get('volume', None)
            
            # Calculate heuristic size based on target elements
            target_elements = getattr(self.config.mesh_params, 'target_elements', 8000)
            
            heuristic_max = None
            if target_elements and target_elements > 0:
                target_elements = max(target_elements, 100) # prevent tiny targets
                if volume and volume > 0:
                    # Volume-based sizing (Robust)
                    # Size ~ 2.0 * (Volume / Target)^(1/3)
                    curr_vol_per_elem = volume / float(target_elements)
                    heuristic_max = 2.0 * (curr_vol_per_elem ** (1.0/3.0))
                    self.log_message(f"[DEBUG] Volume-based sizing: Target {target_elements} -> Size {heuristic_max:.4f} mm")
                
                if not heuristic_max:
                    # Fallback to Diagonal scaling if volume missing
                    scaling_factor = 10.0 * (target_elements / 38000.0) ** (1.0/3.0)
                    heuristic_max = diagonal / scaling_factor
                    self.log_message(f"[DEBUG] Diagonal-based sizing: Target {target_elements} -> Size {heuristic_max:.4f} mm")
            else:
                self.log_message("[DEBUG] No target_elements specified - skipping volume-based heuristic")


            heuristic_min = heuristic_max / 10.0

            # Determine final sizing: max_size and min_size are now independent
            user_max_size = getattr(self.config.mesh_params, 'max_size_mm', None)
            user_min_size = getattr(self.config.mesh_params, 'min_size_mm', None)
            
            self.log_message(f"[DEBUG] user_max_size from config: {user_max_size}")
            self.log_message(f"[DEBUG] user_min_size from config: {user_min_size}")
            
            if user_max_size and user_max_size > 0:
                size_max = float(user_max_size)
                self.log_message(f"[DEBUG] Using user max_size: {size_max} mm")
            else:
                # No user constraint - use volume/target-based heuristic
                size_max = heuristic_max if heuristic_max else (diagonal / 10.0)
                self.log_message(f"[DEBUG] Using heuristic max_size: {size_max} mm")
            
            # Min size is now independent of max size
            if user_min_size and user_min_size > 0:
                size_min = float(user_min_size)
                self.log_message(f"[DEBUG] Using user min_size: {size_min} mm")
            else:
                # Default: 1mm floor to prevent hang on tiny features
                size_min = 1.0
                self.log_message(f"[DEBUG] Using default min_size: {size_min} mm")
            
            # Sanity check: min should be < max
            if size_min >= size_max:
                self.log_message(f"[WARNING] min_size ({size_min}) >= max_size ({size_max}), clamping min to max/2")
                size_min = size_max / 2.0

            # Create size field: fine near geometry, coarse inside
            field_tag = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(field_tag, "FacesList",
                                             [tag for dim, tag in gmsh.model.getEntities(dim=2)])

            threshold_field = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(threshold_field, "IField", field_tag)
            gmsh.model.mesh.field.setNumber(threshold_field, "SizeMin", size_min)
            gmsh.model.mesh.field.setNumber(threshold_field, "SizeMax", size_max)
            gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", size_max * 0.2)
            gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", size_max)

            gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)

            gmsh.option.setNumber("Mesh.Algorithm", 6)
            gmsh.option.setNumber("Mesh.Algorithm3D", 1)
            gmsh.option.setNumber("Mesh.ElementOrder", self.config.mesh_params.element_order)

            return self._generate_and_analyze()

        except Exception as e:
            self.log_message(f"[!] Adaptive refinement failed: {e}")
            return False, None

    def _try_linear_tet_delaunay(self) -> Tuple[bool, Optional[Dict]]:
        """Linear tetrahedral (simpler, more robust)"""
        self.log_message("Linear tetrahedral mesh (fallback)...")

        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)  # Linear
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.Smoothing", 20)

        return self._generate_and_analyze()

    def _try_linear_tet_frontal(self) -> Tuple[bool, Optional[Dict]]:
        """Linear frontal (different algorithm)"""
        self.log_message("Linear frontal mesh...")

        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.Smoothing", 15)

        return self._generate_and_analyze()

    def _try_subdivide_and_mesh(self) -> Tuple[bool, Optional[Dict]]:
        """Subdivide geometry and mesh pieces"""
        self.log_message("Attempting geometry subdivision...")

        try:
            # This is a simplified version - real implementation would be more sophisticated
            gmsh.option.setNumber("Mesh.Algorithm", 6)
            gmsh.option.setNumber("Mesh.Algorithm3D", 1)
            gmsh.option.setNumber("Mesh.ElementOrder", 1)
            gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 2.0)  # Coarser

            return self._generate_and_analyze()

        except Exception as e:
            self.log_message(f"[!] Subdivision failed: {e}")
            return False, None

    def _try_automatic_gmsh_default(self) -> Tuple[bool, Optional[Dict]]:
        """Let Gmsh use all defaults (last resort)"""
        self.log_message("Trying Gmsh automatic defaults (last resort)...")

        # Reset to absolute defaults
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)

        return self._generate_and_analyze()

    def _try_tetgen(self) -> Tuple[bool, Optional[Dict]]:
        """Try TetGen as supplementary mesher (requires tetgen package)"""
        self.log_message("Trying TetGen supplementary mesher...")

        if not TETGEN_AVAILABLE:
            self.log_message("[!] TetGen not available")
            return False, None

        try:
            # TetGen operates on surface mesh, so generate 2D first
            gmsh.model.mesh.clear()
            gmsh.model.mesh.generate(2)

            # Use TetGen strategy
            tetgen_gen = TetGenMeshGenerator(self.config)

            # TetGen will extract surface and mesh volume
            temp_output = "temp_tetgen_mesh.msh"

            if tetgen_gen.run_meshing_strategy("", temp_output):
                # Analyze the TetGen mesh
                metrics = self.analyze_current_mesh()
                if metrics:
                    return True, metrics

            return False, None

        except Exception as e:
            self.log_message(f"[!] TetGen failed: {e}")
            return False, None

    def _try_pymesh(self) -> Tuple[bool, Optional[Dict]]:
        """Try PyMesh for mesh repair and generation (requires pymesh2/pymeshfix package)"""
        self.log_message("Trying PyMesh supplementary mesher...")

        if not PYMESH_AVAILABLE:
            self.log_message("[!] PyMesh not available")
            return False, None

        try:
            # PyMesh operates on surface mesh, so generate 2D first
            gmsh.model.mesh.clear()
            gmsh.model.mesh.generate(2)

            # Use PyMesh strategy
            pymesh_gen = PyMeshMeshGenerator(self.config)

            # PyMesh will repair surface and mesh volume
            temp_output = "temp_pymesh_mesh.msh"

            if pymesh_gen.run_meshing_strategy("", temp_output):
                # Analyze the PyMesh mesh
                metrics = self.analyze_current_mesh()
                if metrics:
                    return True, metrics

            return False, None

        except Exception as e:
            self.log_message(f"[!] PyMesh failed: {e}")
            return False, None

    def _generate_and_analyze(self) -> Tuple[bool, Optional[Dict]]:
        """Generate mesh and analyze quality"""
        try:
            # Check if we should stop before starting expensive mesh generation
            if self.check_stop():
                return False, None
            
            # Generate mesh
            if not self.generate_mesh_internal(dimension=3):
                return False, None

            # Check if we should defer quality analysis for faster display
            defer_quality = getattr(self.config, 'defer_quality', False)
            
            if defer_quality:
                # Return minimal metrics without expensive quality analysis
                self.log_message("[Deferred] Skipping quality analysis for faster display...")
                
                # Get basic element count only
                try:
                    elem_types, elem_tags, _ = gmsh.model.mesh.getElements(dim=3)
                    total_elements = sum(len(tags) for tags in elem_tags)
                    node_tags, _, _ = gmsh.model.mesh.getNodes()
                    total_nodes = len(node_tags)
                    
                    metrics = {
                        'total_elements': total_elements,
                        'total_nodes': total_nodes,
                        'deferred': True,
                        # Placeholder values - actual quality will be calculated in background
                        'gmsh_sicn': {'min': 0, 'max': 0, 'avg': 0},
                        'gmsh_gamma': {'min': 0, 'max': 0, 'avg': 0},
                        'skewness': {'min': 0, 'max': 0, 'avg': 0},
                        'aspect_ratio': {'min': 0, 'max': 0, 'avg': 0}
                    }
                    self.log_message(f"[Deferred] Generated mesh: {total_elements:,} elements, {total_nodes:,} nodes")
                    return True, metrics
                except Exception as e:
                    self.log_message(f"[!] Failed to get basic metrics: {e}")
                    return False, None
            else:
                # Full quality analysis
                metrics = self.analyze_current_mesh()
                if not metrics:
                    return False, None

                # Check if we have valid elements
                if metrics.get('total_elements', 0) == 0:
                    self.log_message("[!] No elements generated")
                    return False, None

                # Check for degenerate elements (skewness = 1.0)
                if metrics.get('skewness') and metrics['skewness']['max'] >= 0.99:
                    self.log_message("[!] Degenerate elements detected (skewness ~= 1.0)")
                    # Still return the metrics for comparison, but mark as problematic
                    return True, metrics  # Allow comparison even if poor

                return True, metrics

        except Exception as e:
            self.log_message(f"[!] Generation/analysis failed: {e}")
            return False, None

    def _generate_exhaustive_report(self, output_file: str):
        """Generate comprehensive report of all attempts"""
        self.log_message("\n" + "=" * 60)
        self.log_message("EXHAUSTIVE MESHING REPORT")
        self.log_message("=" * 60)

        self.log_message(f"\nTotal strategies attempted: {len(self.all_attempts)}")

        # Count successes
        successes = [a for a in self.all_attempts if a.get('success', False)]
        self.log_message(f"Successful: {len(successes)}/{len(self.all_attempts)}")

        # Show all attempts
        self.log_message("\nAll Attempts:")
        for i, attempt in enumerate(self.all_attempts, 1):
            strategy = attempt.get('strategy', 'unknown')
            success = "[OK]" if attempt.get('success') else "[X]"
            score = attempt.get('score', float('inf'))

            if score < float('inf'):
                self.log_message(f"  {i}. {success} {strategy}: score={score:.2f}")
            else:
                error = attempt.get('error', 'failed')
                self.log_message(f"  {i}. {success} {strategy}: {error}")

        # Best result
        best = min(self.all_attempts, key=lambda x: x.get('score', float('inf')))
        if best.get('score', float('inf')) < float('inf'):
            self.log_message(f"\n[*] BEST RESULT: {best['strategy']}")
            self.log_message(f"   Score: {best['score']:.2f}")

            metrics = best.get('metrics', {})
            if metrics.get('skewness'):
                s = metrics['skewness']
                self.log_message(f"   Skewness: Max={s['max']:.4f}, Avg={s['avg']:.4f}, Min={s['min']:.4f}")
            if metrics.get('aspect_ratio'):
                a = metrics['aspect_ratio']
                self.log_message(f"   Aspect Ratio: Max={a['max']:.2f}, Avg={a['avg']:.2f}")

        self.log_message(f"\nFinal mesh saved to: {output_file}")
        self.log_message("=" * 60)

    def _run_surgical_isolation_loop(self, input_file: str, output_file: str) -> bool:
        """
        Run the Surgical Amputation loop: Mesh each volume in isolation.
        Uses the proven subprocess pattern from surgical_orchestrator.py.
        """
        self.log_message("\n" + "=" * 60)
        self.log_message("EXECUTING SURGICAL ISOLATION LOOP")
        self.log_message("Targeting each volume individually for robust assembly meshing.")
        self.log_message("=" * 60)
        
        temp_dir = os.path.join(os.path.dirname(output_file), "temp_surgical")
        
        # CRITICAL FIX: Force clear cache to prevent stale inverted meshes
        # The surgical loop relies on file existence for caching.
        # If we had a bad run (inverted elements), the bad MSH files persist.
        if os.path.exists(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
                self.log_message("[CACHE] Cleared stale surgical cache to ensure fresh generation.")
            except Exception as e:
                self.log_message(f"[CACHE] Warning: Failed to clear cache: {e}")
                
        os.makedirs(temp_dir, exist_ok=True)
        
        volumes = gmsh.model.getEntities(3)
        vol_tags = [v[1] for v in volumes]
        self.log_message(f"Volumes to process: {len(vol_tags)}")
        
        worker_script = os.path.join(PROJECT_ROOT, "core", "isolation_worker_script.py")
        
        # Proven pattern: Direct subprocess spawning with ThreadPoolExecutor for I/O coordination
        def spawn_worker(tag):
            worker_msh = os.path.join(temp_dir, f"vol_{tag}.msh")
            
            # Skip if already done
            if os.path.exists(worker_msh):
                return tag, True, "cached"
            
            # Retrieve current strategy from config or default to HXT (fastest)
            current_strategy = "tet_hxt_optimized"
            try:
                if self.config and self.config.mesh_params and self.config.mesh_params.mesh_strategy:
                    current_strategy = self.config.mesh_params.mesh_strategy
            except: pass

            cmd = [
                sys.executable,
                worker_script,
                "--input", input_file,
                "--output", worker_msh,
                "--tag", str(tag),
                "--strategy", str(current_strategy)
            ]
            
            try:
                # 300s per volume (5 minutes max)
                # Use subprocess.run but log progress on completion (keeps logs cleaner)
                result = subprocess.run(
                    cmd,
                    timeout=300,
                    capture_output=True,
                    text=True,
                    cwd=PROJECT_ROOT
                )
                
                if result.returncode == 0 and os.path.exists(worker_msh):
                    return tag, True, "meshed"
                else:
                    # Log failure reason (last 500 chars of output)
                    output_tail = (result.stdout + result.stderr)[-500:] if (result.stdout or result.stderr) else ""
                    return tag, False, f"exit_{result.returncode}: {output_tail}"
                    
            except subprocess.TimeoutExpired:
                return tag, False, "timeout"
            except Exception as e:
                return tag, False, f"exception: {str(e)}"
        
        # Run workers in parallel batches
        max_workers = min(len(vol_tags), 6)  # Windows stability cap
        self.log_message(f"Starting {max_workers} parallel workers...")
        
        completed = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(spawn_worker, tag): tag for tag in vol_tags}
            
            for future in concurrent.futures.as_completed(futures):
                tag = futures[future]
                try:
                    vol_tag, success, reason = future.result()
                    completed.append((vol_tag, success))
                    
                    status = "[OK]" if success else "[FAIL]"
                    self.log_message(f"  {status} Volume {vol_tag}: {reason}")
                    
                except Exception as exc:
                    self.log_message(f"  [ERROR] Volume {tag} raised: {exc}")
                    completed.append((tag, False))
        
        # Summary
        success_count = sum(1 for _, s in completed if s)
        self.log_message(f"\nIsolation Complete: {success_count}/{len(vol_tags)} successful")
        
        # Final Merge - PRESERVE INDIVIDUAL PHYSICAL GROUPS
        # CRITICAL: When merging .msh files, gmsh.merge() adds MESH ELEMENTS, not geometric volumes.
        # Each individual .msh file already has physical groups assigned ("Volume_N").
        # We need to merge them while preserving their individual identities.
        self.log_message("Merging all isolated meshes...")
        
        # Safely ensure Gmsh is initialized
        if not gmsh.isInitialized():
            gmsh.initialize()
            
        gmsh.model.add("Surgical_Assembly")
        
        msh_files = sorted(glob.glob(os.path.join(temp_dir, "vol_*.msh")))
        self.log_message(f"Found {len(msh_files)} mesh files to merge")
        
        # Merge all mesh files - this preserves their physical groups
        for msh_file in msh_files:
            try:
                gmsh.merge(msh_file)
                # CRITICAL: Renumber after each merge to prevent tag collisions.
                # If we don't do this, multiple parts will share the same element tags (1, 2, 3...),
                # which causes quality mapping to only save data for the last part merged.
                gmsh.model.mesh.renumberNodes()
                gmsh.model.mesh.renumberElements()
            except Exception as e:
                self.log_message(f"  [!] Failed to merge {os.path.basename(msh_file)}: {e}")
        
        # CRITICAL: For assemblies, we must remove duplicate nodes at the interfaces
        # between volumes. Otherwise, the mesh is just a collection of disjoint parts,
        # which causes errors during global quality analysis and rendering.
        self.log_message("Cleaning assembly topology (removing duplicate nodes)...")
        gmsh.model.mesh.removeDuplicateNodes()
        
        # Final renumbering to ensure a perfectly contiguous, unique tag space
        gmsh.model.mesh.renumberNodes()
        gmsh.model.mesh.renumberElements()
        
        # Ensure model is synchronized for analysis
        # gmsh.model.mesh.synchronize() # Not strictly necessary for mesh-only models but good practice
        
        # Verify physical groups were preserved
        phys_groups = gmsh.model.getPhysicalGroups(3)
        if phys_groups:
            self.log_message(f"[OK] Merged mesh contains {len(phys_groups)} physical volume groups")
        else:
            self.log_message("[WARNING] No physical groups found after merge! Mesh may not export correctly.")
        
        # Analyze current assembly (Surgical)
        self.log_message("Analyzing final assembly quality...")
        # Reduce complexity for large assemblies to fix 15s+ slowness
        try:
            metrics = self.quality_analyzer.analyze_mesh(include_advanced_metrics=False)
            
            if metrics:
                self.log_message(f"[SURGICAL] Final assembly metrics: {metrics['total_elements']} elements, {metrics['total_nodes']} nodes")
                # Store metrics for worker retrieval
                self.quality_history.append({'iteration': 99, 'params': {'surgical': True}, 'metrics': metrics})
                
                # Check for inverted elements
                if metrics.get('gmsh_sicn') and metrics['gmsh_sicn']['min'] < 0:
                    self.log_message("[!] WARNING: Merged assembly contains inverted elements. Attempting auto-repair...")
                    
                    # Select all volume elements (dim=3)
                    # We can't easily valid-check individual elements via Python API rapidly,
                    # but reverse() on 3D entities often fixes orientation issues from merging.
                    # A more robust way is to re-orient based on volume.
                    
                    try:
                        # Get all 3D entities
                        vol_tags = [t[1] for t in gmsh.model.getEntities(3)]
                        if vol_tags:
                            # Re-orient elements in all volumes
                            gmsh.model.mesh.reverse([(3, t) for t in vol_tags])
                            self.log_message(f"  > Applied orientation fix to {len(vol_tags)} volumes")
                            
                            # Re-analyze to check if it worked
                            metrics = self.quality_analyzer.analyze_mesh(include_advanced_metrics=False)
                            if metrics.get('gmsh_sicn') and metrics['gmsh_sicn']['min'] >= 0:
                                self.log_message("  > [SUCCESS] Auto-repair fixed inverted elements!")
                                # Update stored metrics
                                self.quality_history[-1]['metrics'] = metrics
                            else:
                                self.log_message("  > [FAILED] Auto-repair could not fix all inverted elements. CAD geometry may be self-intersecting.")
                    except Exception as repair_err:
                         self.log_message(f"  > [ERROR] Repair failed: {repair_err}")
        except Exception as e:
            self.log_message(f"[!] Analysis failed: {e}")

        # Write the merged mesh
        # For assemblies, SaveAll=0 avoids saving redundant, overlapping surface meshes
        # that cause "double counting" artifacts and z-fighting in the viewer.
        # Our native parser extracts the boundary from the volume tets anyway.
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.write(output_file)
        gmsh.finalize()
        
        self.log_message(f"[OK] Final mesh saved: {output_file}")
        return True

    def _generate_failure_report(self):
        """Generate report when everything failed"""
        self.log_message("\n" + "=" * 60)
        self.log_message("COMPLETE FAILURE REPORT")
        self.log_message("=" * 60)

        self.log_message(f"\nAll {len(self.all_attempts)} strategies failed!")
        self.log_message("\nAttempted strategies:")
        for attempt in self.all_attempts:
            strategy = attempt.get('strategy', 'unknown')
            error = attempt.get('error', 'unknown failure')
            self.log_message(f"  [X] {strategy}: {error}")

        self.log_message("\nPossible causes:")
        self.log_message("  1. Geometry has serious issues (self-intersections, gaps, etc.)")
        self.log_message("  2. Very small features that can't be meshed")
        self.log_message("  3. Invalid CAD file")
        self.log_message("  4. Geometry too complex for automatic meshing")

        self.log_message("\nRecommendations:")
        self.log_message("  1. Check geometry in CAD software")
        self.log_message("  2. Simplify geometry (remove small features)")
        self.log_message("  3. Try manual meshing in Gmsh GUI: gmsh your_file.step")
        self.log_message("  4. Check for geometry errors in CAD export")
        self.log_message("=" * 60)

    def _save_detailed_report_to_file(self, output_file: str, success: bool):
        """
        Save comprehensive report to text file for crash/failure analysis

        Creates a report with:
        - Human-readable summary at the top
        - All technical details below for debugging
        """
        # Create report filename
        report_file = output_file.replace('.msh', '_exhaustive_report.txt')

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            with open(report_file, 'w') as f:
                # ===== HUMAN-READABLE SUMMARY =====
                f.write("=" * 70 + "\n")
                f.write("EXHAUSTIVE MESH GENERATION REPORT - SUMMARY\n")
                f.write("=" * 70 + "\n\n")
                f.write(f"Report Generated: {timestamp}\n")
                f.write(f"Output File: {output_file}\n")
                f.write(f"Overall Status: {'SUCCESS' if success else 'FAILED'}\n\n")

                # Strategy summary
                f.write(f"Total Strategies Attempted: {len(self.all_attempts)}\n")
                successes = [a for a in self.all_attempts if a.get('success', False)]
                f.write(f"Successful Strategies: {len(successes)}/{len(self.all_attempts)}\n\n")

                # Best result (if any)
                if successes:
                    best = min(self.all_attempts, key=lambda x: x.get('score', float('inf')))
                    if best.get('score', float('inf')) < float('inf'):
                        f.write("[*] BEST RESULT:\n")
                        f.write(f"  Strategy: {best['strategy']}\n")
                        f.write(f"  Quality Score: {best['score']:.2f} (lower is better)\n")

                        metrics = best.get('metrics', {})

                        # Show Gmsh native metrics (primary - most accurate)
                        if metrics.get('gmsh_sicn'):
                            s = metrics['gmsh_sicn']
                            f.write(f"  SICN (Gmsh): Min={s['min']:.4f}, Max={s['max']:.4f}, Avg={s['avg']:.4f}\n")
                        if metrics.get('gmsh_gamma'):
                            g = metrics['gmsh_gamma']
                            f.write(f"  Gamma (Gmsh): Min={g['min']:.4f}, Max={g['max']:.4f}, Avg={g['avg']:.4f}\n")

                        # Show converted metrics (secondary - for reference)
                        if metrics.get('skewness'):
                            s = metrics['skewness']
                            f.write(f"  Skewness (converted): Max={s['max']:.4f}, Avg={s['avg']:.4f}, Min={s['min']:.4f}\n")
                        if metrics.get('aspect_ratio'):
                            a = metrics['aspect_ratio']
                            f.write(f"  Aspect Ratio (converted): Max={a['max']:.2f}, Avg={a['avg']:.2f}, Min={a['min']:.2f}\n")

                        # Show element counts
                        if metrics.get('total_elements'):
                            f.write(f"  Total Elements: {metrics['total_elements']:,}\n")
                        if metrics.get('total_nodes'):
                            f.write(f"  Total Nodes: {metrics['total_nodes']:,}\n")
                        f.write("\n")

                # Quality assessment
                if success and successes:
                    best = min(self.all_attempts, key=lambda x: x.get('score', float('inf')))
                    metrics = best.get('metrics', {})
                    f.write("QUALITY ASSESSMENT:\n")

                    # Assess based on Gmsh SICN (most accurate)
                    if metrics.get('gmsh_sicn'):
                        sicn_min = metrics['gmsh_sicn']['min']
                        if sicn_min < 0:
                            f.write(f"  [X] SICN: CRITICAL - INVERTED ELEMENTS ({sicn_min:.4f} < 0)\n")
                        elif sicn_min > 0.5:
                            f.write(f"  [OK] SICN: EXCELLENT ({sicn_min:.4f} > 0.5)\n")
                        elif sicn_min > 0.3:
                            f.write(f"  [!] SICN: GOOD ({sicn_min:.4f} > 0.3)\n")
                        else:
                            f.write(f"  [X] SICN: POOR ({sicn_min:.4f} <= 0.3)\n")

                    # Assess based on Gmsh Gamma
                    if metrics.get('gmsh_gamma'):
                        gamma_min = metrics['gmsh_gamma']['min']
                        if gamma_min > 0.4:
                            f.write(f"  [OK] Gamma: EXCELLENT ({gamma_min:.4f} > 0.4)\n")
                        elif gamma_min > 0.2:
                            f.write(f"  [!] Gamma: GOOD ({gamma_min:.4f} > 0.2)\n")
                        else:
                            f.write(f"  [X] Gamma: POOR ({gamma_min:.4f} <= 0.2)\n")

                    # Also show converted metrics assessments for reference
                    if metrics.get('skewness'):
                        max_skew = metrics['skewness']['max']
                        if max_skew <= 0.7:
                            f.write(f"  [OK] Skewness (converted): EXCELLENT ({max_skew:.4f} <= 0.7)\n")
                        elif max_skew <= 0.85:
                            f.write(f"  [!] Skewness (converted): ACCEPTABLE ({max_skew:.4f} <= 0.85)\n")
                        else:
                            f.write(f"  [X] Skewness (converted): POOR ({max_skew:.4f} > 0.85)\n")

                    if metrics.get('aspect_ratio'):
                        max_aspect = metrics['aspect_ratio']['max']
                        if max_aspect <= 5.0:
                            f.write(f"  [OK] Aspect Ratio (converted): EXCELLENT ({max_aspect:.2f} <= 5.0)\n")
                        elif max_aspect <= 10.0:
                            f.write(f"  [!] Aspect Ratio (converted): ACCEPTABLE ({max_aspect:.2f} <= 10.0)\n")
                        else:
                            f.write(f"  [X] Aspect Ratio (converted): POOR ({max_aspect:.2f} > 10.0)\n")
                    f.write("\n")

                # ===== TECHNICAL DETAILS =====
                f.write("\n" + "=" * 70 + "\n")
                f.write("TECHNICAL DETAILS (For Crash/Failure Analysis)\n")
                f.write("=" * 70 + "\n\n")

                # Geometry information
                f.write("GEOMETRY INFORMATION:\n")
                for key, value in self.geometry_info.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")

                # Detailed attempt log
                f.write("ALL STRATEGY ATTEMPTS:\n")
                f.write("-" * 70 + "\n")

                for i, attempt in enumerate(self.all_attempts, 1):
                    strategy = attempt.get('strategy', 'unknown')
                    success_flag = attempt.get('success', False)
                    score = attempt.get('score', float('inf'))

                    f.write(f"\nAttempt {i}: {strategy}\n")
                    f.write(f"  Status: {'SUCCESS' if success_flag else 'FAILED'}\n")

                    if success_flag:
                        f.write(f"  Quality Score: {score:.2f}\n")
                        metrics = attempt.get('metrics', {})
                        if metrics:
                            f.write(f"  Metrics:\n")
                            # Gmsh native metrics (primary)
                            if metrics.get('gmsh_sicn'):
                                s = metrics['gmsh_sicn']
                                f.write(f"    SICN (Gmsh): min={s['min']:.4f}, max={s['max']:.4f}, avg={s['avg']:.4f}\n")
                            if metrics.get('gmsh_gamma'):
                                g = metrics['gmsh_gamma']
                                f.write(f"    Gamma (Gmsh): min={g['min']:.4f}, max={g['max']:.4f}, avg={g['avg']:.4f}\n")
                            # Converted metrics (secondary)
                            if metrics.get('skewness'):
                                s = metrics['skewness']
                                f.write(f"    Skewness (converted): min={s['min']:.4f}, max={s['max']:.4f}, avg={s['avg']:.4f}\n")
                            if metrics.get('aspect_ratio'):
                                a = metrics['aspect_ratio']
                                f.write(f"    Aspect Ratio (converted): min={a['min']:.4f}, max={a['max']:.4f}, avg={a['avg']:.4f}\n")
                            if metrics.get('total_elements'):
                                f.write(f"    Total Elements: {metrics['total_elements']:,}\n")
                            if metrics.get('total_nodes'):
                                f.write(f"    Total Nodes: {metrics['total_nodes']:,}\n")
                    else:
                        error = attempt.get('error', 'unknown')
                        f.write(f"  Error: {error}\n")

                    f.write("-" * 70 + "\n")

                # Configuration used
                f.write("\nCONFIGURATION:\n")
                f.write(f"  Quality Targets:\n")
                f.write(f"    Max Skewness: {self.config.quality_targets.skewness_max}\n")
                f.write(f"    Max Aspect Ratio: {self.config.quality_targets.aspect_ratio_max}\n")
                f.write(f"    Min Angle: {self.config.quality_targets.min_angle_min}\n")
                f.write(f"  Mesh Parameters:\n")
                f.write(f"    Element Order: {self.config.mesh_params.element_order}\n")
                f.write(f"    Max Iterations: {self.config.mesh_params.max_iterations}\n")

                # Recommendations if failed
                if not success:
                    f.write("\nRECOMMENDATIONS:\n")
                    f.write("  1. Check geometry for self-intersections or gaps in CAD software\n")
                    f.write("  2. Simplify geometry (remove small features, fillets < 1mm)\n")
                    f.write("  3. Try opening in Gmsh GUI: gmsh your_file.step\n")
                    f.write("  4. Export CAD with higher tolerance or different format (IGES vs STEP)\n")
                    f.write("  5. Consider defeaturing or geometry repair tools\n")

                f.write("\n" + "=" * 70 + "\n")
                f.write("END OF REPORT\n")
                f.write("=" * 70 + "\n")

            self.log_message(f"\n[OK] Detailed report saved to: {report_file}")

        except Exception as e:
            self.log_message(f"[!] Could not save detailed report: {e}")


def main():
    """Command-line interface"""
    if len(sys.argv) > 1:
        cad_file = sys.argv[1]
    else:
        cad_file = input("Enter CAD file path: ").strip()

    try:
        generator = ExhaustiveMeshGenerator()
        result = generator.generate_mesh(cad_file)

        if result.success:
            print(f"\n[OK] Exhaustive meshing completed!")
            print(f"Output file: {result.output_file}")
        else:
            print(f"\n[X] All meshing strategies failed")
            print("See detailed report above for diagnostics")
            sys.exit(1)

    except Exception as e:
        print(f"\n[X] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
