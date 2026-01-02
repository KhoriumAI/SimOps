"""
Assembly Mesh Generation Strategy
===================================

Optimized pipeline for multi-volume assemblies (>3 volumes).
Uses fail-fast approach: try meshing → immediately box failures (no repair).

Pipeline:
1. Split STEP into individual STLs
2. Fast-pass volume meshing with surface classification
3. Box failures immediately (no 30s timeout repair)
4. Merge all meshes into single model with physical groups
"""

import gmsh
import os
import math
import struct
import time
import glob
import multiprocessing
from typing import Dict, List, Optional, Tuple, Set
from core.mesh_generator import BaseMeshGenerator, MeshGenerationResult
from core.config import Config

# [CRITICAL FIX] Allow this script to spawn workers even if it is a worker itself
# This prevents "AssertionError: daemonic processes are not allowed to have children"
try:
    multiprocessing.current_process().daemon = False
except Exception:
    pass


def _parallel_split_worker(worker_id: int, total_workers: int, step_file: str, 
                           temp_stl_dir: str, triage_mode: bool,
                           cad_density_threshold: float,
                           max_diag_for_triage: float,
                           min_size: float = 1.0, 
                           max_size: float = 10.0) -> Tuple[List[str], Set[int]]:
    """Worker function for parallel assembly splitting"""
    import gmsh
    import os
    import math
    import numpy as np
    
    # --- QUALITY CONSTANTS (The "Sanitized" Gate) ---
    # RAISED THRESHOLD: 0.00015 crashed it. We set safety deck at 0.001.
    FATAL_GAMMA_FLOOR = 0.001
    POOR_QUALITY_THRESHOLD = 0.05
    AVG_QUALITY_THRESHOLD = 0.40
    # ---------------------------------------------
    
    stl_files = []
    toxic_volumes = set()
    
    try:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.Binary", 1)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.option.setNumber("Mesh.MeshOnlyVisible", 1)
        gmsh.option.setNumber("General.NumThreads", 1)  # 1 Process = 1 Core
        
        # High-quality surface mesh settings (Clamped)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
        gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 20)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_size)
        
        # V6 Immortal Settings
        # Perturb nodes by 1e-8 to avoid symmetric singularities (Mirrored Part Fix)
        gmsh.option.setNumber("Mesh.RandomFactor", 1e-8)

        
        gmsh.open(step_file)
        
        volumes = gmsh.model.getEntities(dim=3)
        # Hide everything initially
        gmsh.model.setVisibility(gmsh.model.getEntities(), 0, recursive=True)
        
        # Modulo Load Balancing
        my_volumes = [v for i, v in enumerate(volumes) if i % total_workers == worker_id]
        
        for dim, tag in my_volumes:
            try:
                # 1. TRIAGE
                bbox = gmsh.model.getBoundingBox(dim, tag)
                dx, dy, dz = bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2]
                diagonal = math.sqrt(dx*dx + dy*dy + dz*dz)
                
                # Get boundaries (Surfaces)
                surfs = gmsh.model.getBoundary([(dim, tag)], recursive=False)
                
                if triage_mode:
                    curves = gmsh.model.getBoundary([(dim, tag)], recursive=True)
                    num_features = len(surfs) + len(curves)
                    cad_density = num_features / (diagonal + 1e-9)
                    
                    if cad_density > cad_density_threshold and diagonal < max_diag_for_triage:
                        toxic_volumes.add(tag)
                        continue

                gmsh.model.setVisibility([(dim, tag)], 1, recursive=True)
                print(f"[Worker {worker_id}] Vol {tag}: Generating surface mesh...", flush=True)
                gmsh.model.mesh.generate(2)
                
                # --- CRITICAL FIX: CLEANUP REMOVED ---
                # We do NOT run removeDuplicateNodes() because it causes hangs on sliver geometry.
                # We rely on the Quality Gate to reject bad meshes instead.
                
                # Check for success and quality
                _, elem_tags, _ = gmsh.model.mesh.getElements(2, -1) # All 2D elements in isolate model
                if not elem_tags or len(elem_tags[0]) == 0:
                    toxic_volumes.add(tag)
                    gmsh.model.setVisibility([(dim, tag)], 0, recursive=True)
                    gmsh.model.mesh.clear()
                    continue
                
                # --- GATE 2: GAMMA SAFETY CHECK ---
                quality_ok = True
                # Check GAMMA (The metric causing your specific crash)
                # FIX: 'minGamma' rejected by Gmsh, using 'gamma' (lowercase)
                quals = gmsh.model.mesh.getElementQualities(elem_tags[0].tolist(), "gamma")
                
                if len(quals) > 0:
                    min_q = np.min(quals)
                    avg_q = np.mean(quals)
                    
                    # CONDITION A: FATAL (Crash Risk)
                    if min_q < FATAL_GAMMA_FLOOR:
                        # Log it even in non-verbose mode if it's a fatal failure
                        print(f"[Worker {worker_id}] Vol {tag} REJECT: Fatal Gamma ({min_q:.1e})", flush=True)
                        quality_ok = False
                    
                    # CONDITION B: POOR COMPOSITE (Messy Mesh)
                    elif (min_q < POOR_QUALITY_THRESHOLD) and (avg_q < AVG_QUALITY_THRESHOLD):
                        print(f"[Worker {worker_id}] Vol {tag} REJECT: Poor Quality (Min:{min_q:.3f}, Avg:{avg_q:.3f})", flush=True)
                        quality_ok = False
                else:
                    quality_ok = False # No qualities found
                
                if not quality_ok:
                    # RULE: Diplomatic Immunity
                    # If part is large, we CANNOT box it (simulation integrity).
                    # We accept the bad mesh (distortion) over a box (short-circuit).
                    if diagonal >= max_diag_for_triage:
                        print(f"[Worker {worker_id}] Vol {tag} IMMUNITY GRANTED (Size {diagonal:.1f}mm). Keeping distorted mesh.", flush=True)
                        # We used to call removeDuplicateNodes() here, but it caused hangs.
                        # V6 Strategy: Simply proceed with the mesh we have.
                    else:
                        toxic_volumes.add(tag)
                        gmsh.model.setVisibility([(dim, tag)], 0, recursive=True)
                        gmsh.model.mesh.clear()
                        continue
                
                # 3. EXPORT
                surface_tags = [s_tag for (s_dim, s_tag) in surfs if s_dim == 2]
                if not surface_tags:
                    toxic_volumes.add(tag)
                else:
                    gmsh.model.removePhysicalGroups()
                    gmsh.model.addPhysicalGroup(2, surface_tags, 1)
                    
                    stl_path = os.path.join(temp_stl_dir, f"vol_{tag}.stl")
                    gmsh.write(stl_path)
                    
                    if os.path.exists(stl_path) and os.path.getsize(stl_path) > 100:
                        stl_files.append(stl_path)
                        # Mark large parts for extended meshing timeout
                        if diagonal >= max_diag_for_triage:
                            with open(stl_path + ".immune", "w") as f:
                                f.write("1")
                    else:
                        toxic_volumes.add(tag)
                        if os.path.exists(stl_path): os.remove(stl_path)

                # 4. CLEANUP
                gmsh.model.mesh.clear()
                gmsh.model.setVisibility([(dim, tag)], 0, recursive=True)
                
            except Exception:
                toxic_volumes.add(tag)

        gmsh.finalize()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[Worker {worker_id}] CRITICAL FAILURE: {e}", flush=True)
        
    return stl_files, toxic_volumes


def get_stl_bounds(stl_path: str) -> Optional[Tuple[float, ...]]:
    """Extract bounding box from binary STL without gmsh"""
    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = float('-inf')
    
    try:
        with open(stl_path, 'rb') as f:
            header = f.read(80)
            count_bytes = f.read(4)
            if len(count_bytes) < 4:
                return None
            count = struct.unpack('<I', count_bytes)[0]
            
            for _ in range(count):
                data = f.read(50)
                if len(data) < 50:
                    break
                floats = struct.unpack('<12fH', data)
                for i in range(3, 12, 3):
                    x, y, z = floats[i], floats[i+1], floats[i+2]
                    min_x, max_x = min(min_x, x), max(max_x, x)
                    min_y, max_y = min(min_y, y), max(max_y, y)
                    min_z, max_z = min(min_z, z), max(max_z, z)
        
        if min_x == float('inf'):
            return None
        return (min_x, min_y, min_z, max_x, max_y, max_z)
    except Exception:
        return None


def generate_bounding_box_mesh(stl_path: str, output_path: str) -> Tuple[bool, str]:
    """Create simple box mesh as fallback for failed parts"""
    bounds = get_stl_bounds(stl_path)
    if not bounds:
        return False, "Could not parse STL"
        
    min_x, min_y, min_z, max_x, max_y, max_z = bounds
    dx, dy, dz = max_x - min_x, max_y - min_y, max_z - min_z
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("BoundingBox")
        gmsh.model.occ.addBox(min_x, min_y, min_z, dx, dy, dz)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(3)
        gmsh.write(output_path)
        return True, "Success"
    except Exception as e:
        return False, str(e)
    finally:
        gmsh.finalize()


def _volume_mesh_process_safe(stl_path, output_path, min_size=1.0, max_size=10.0):
    """
    Isolated process: Creates a proper solid volume from STL skin,
    then meshes with Delaunay (Algo 1) to prevent HXT crashes.
    """
    import gmsh, sys
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    
    # --- PROVEN ROBUST SETTINGS ---
    # 1=Delaunay (Tank), 10=HXT (Glass/Fast)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1) 
    
    # CRITICAL: Disable Optimizers (These often trigger HXT internally)
    gmsh.option.setNumber("Mesh.Optimize", 0) 
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)
    gmsh.option.setNumber("Mesh.HighOrderOptimize", 0)
    
    # --- SIZE CLAMPING ---
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_size)
    
    try:
        # 1. Load STL as discrete surface
        gmsh.merge(stl_path)
        
        # --- THE FIX: Create a Volume Entity ---
        # This tells Gmsh that the STL is a watertight solid, not a floating sheet
        surfaces = gmsh.model.getEntities(2)
        if surfaces:
            surface_tags = [s[1] for s in surfaces]
            # Create Surface Loop (watertight bag)
            s_loop = gmsh.model.geo.addSurfaceLoop(surface_tags)
            # Define Volume (solid matter inside the bag)
            vol = gmsh.model.geo.addVolume([s_loop])
            # Synchronize to make it real
            gmsh.model.geo.synchronize()
            
            # Generate 3D Mesh (now it knows to fill the volume)
            gmsh.model.mesh.generate(3)
            
            # Assign Physical Volume (Crucial for Solver Export)
            gmsh.model.addPhysicalGroup(3, [vol], 1, name="SolidVolume")
            
            gmsh.write(output_path)
        else:
            # No surfaces found - fail
            sys.exit(1)
            
    except Exception:
        sys.exit(1) # Signal failure to parent
    finally:
        gmsh.finalize()

def _robust_mesh_task(stl_path, output_path, timeout=30, min_size=1.0, max_size=10.0):
    """
    Supervisor function running in pool.
    Spawns a child process for meshing to contain crashes/hangs.
    """
    import multiprocessing
    
    # 1. Skip if exists
    if os.path.exists(output_path):
        return output_path

    # 2. Check for "PRE-BOX" marker from Splitter
    if os.path.exists(stl_path + ".marker"):
        generate_bounding_box_mesh(stl_path, output_path)
        return output_path

    # 3. Launch Isolated Process
    p = multiprocessing.Process(target=_volume_mesh_process_safe, args=(stl_path, output_path, min_size, max_size))
    p.start()
    p.join(timeout)
    
    # 4. Handle Hangs (Timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        # Fallback to Box
        generate_bounding_box_mesh(stl_path, output_path)
        return output_path
    
    # 5. Handle Crashes (Exit Code != 0)
    if p.exitcode != 0:
        generate_bounding_box_mesh(stl_path, output_path)
        return output_path
        
    return output_path


class AssemblyMeshGenerator(BaseMeshGenerator):
    """
    Optimized mesh generator for multi-volume assemblies.
    
    Strategy:
    - Auto-detects assemblies (>3 volumes)
    - Splits into individual STLs
    - Fast-pass meshing with immediate boxing on failure
    - Merges back into single model
    """
    
    def __init__(self, config: Optional[Config] = None):
        super().__init__(config)
        self.temp_stl_dir = None
        self.temp_mesh_dir = None
        self.failed_parts = []
        self.boxed_parts = []
        self.triage_mode = True  # Default to active triage
        
    def log_message(self, message: str, level: str = "INFO"):
        """Override to respect MESH_VERBOSE environment variable"""
        is_verbose = os.environ.get("MESH_VERBOSE", "1") == "1"
        if is_verbose or level in ["ERROR", "WARNING", "OK"]:
            super().log_message(message, level)
        
    def run_meshing_strategy(self, input_file: str, output_file: str) -> bool:
        """
        Execute assembly meshing pipeline
        
        Returns:
            True if successful, False otherwise
        """
        self.log_message("Starting Assembly Meshing Pipeline")
        pipeline_start = time.time()
        
        try:
            # Stage 1: Split into STLs (with early toxicity detection)
            split_start = time.time()
            stl_files = self._split_assembly_to_stls(input_file)
            split_time = time.time() - split_start
            self.log_message(f"[TIMER] Split Phase: {split_time:.2f}s")
            
            # Clean up gmsh after split (before starting fast-pass)
            self.finalize_gmsh()
            
            # Report split results
            early_boxed = len(getattr(self, 'toxic_volumes', set()))
            self.log_message(f"Split into {len(stl_files)} healthy STLs, {early_boxed} toxic parts skipped")
            
            if not stl_files and early_boxed == 0:
                self.log_message("No components to process", "ERROR")
                return False
            
            # Stage 2: Fast-pass meshing (only healthy STLs)
            mesh_start = time.time()
            mesh_files = self._fast_pass_meshing(stl_files) if stl_files else []
            mesh_time = time.time() - mesh_start
            self.log_message(f"[TIMER] Mesh Phase: {mesh_time:.2f}s")
            
            # Stage 3: Box failures (including early-detected toxic volumes)
            total_to_box = len(self.failed_parts) + early_boxed
            if total_to_box > 0:
                self.log_message(f"Boxing {total_to_box} failed/toxic parts ({len(self.failed_parts)} meshing failures, {early_boxed} pre-detected)")
                mesh_files.extend(self._box_failed_parts())
            
            # Stage 4: Merge into final model
            merge_start = time.time()
            success = self._merge_assembly(mesh_files, output_file)
            merge_time = time.time() - merge_start
            self.log_message(f"[TIMER] Merge Phase: {merge_time:.2f}s")
            
            # Report statistics
            total_time = time.time() - pipeline_start
            self.log_message(f"Assembly Complete: {len(mesh_files)} parts total")
            self.log_message(f"  Meshed: {len(mesh_files) - len(self.boxed_parts)}")
            self.log_message(f"  Boxed: {len(self.boxed_parts)} (early: {early_boxed}, post-fail: {len(self.failed_parts)})")
            self.log_message(f"[TIMER] Total Assembly Wall Time: {total_time:.2f}s")
            self.log_message("=" * 50)
            
            return success
            
        except Exception as e:
            self.log_message(f"Assembly pipeline failed: {e}", "ERROR")
            return False
        finally:
            self._cleanup_temp_dirs()
    
    def _split_assembly_to_stls(self, step_file: str) -> List[str]:
        """Split STEP assembly into individual STL files with parallel hyper-fast triage"""
        import tempfile
        import time
        self.temp_stl_dir = tempfile.mkdtemp(prefix="assembly_stls_")
        
        # Determine number of workers (Safe for 32GB RAM)
        NUM_WORKERS = 3
        
        self.log_message(f"Splitting assembly into STLs using {NUM_WORKERS} parallel workers...")
        
        # Triage Thresholds
        CAD_DENSITY_THRESHOLD = 12.0
        MAX_DIAGONAL_FOR_TRIAGE = 50.0
        
        # Prepare parallel args
        args = []
        for i in range(NUM_WORKERS):
            args.append((
                i, 
                NUM_WORKERS, 
                step_file, 
                self.temp_stl_dir, 
                self.triage_mode, 
                CAD_DENSITY_THRESHOLD, 
                MAX_DIAGONAL_FOR_TRIAGE,
                self.config.mesh_params.min_size_mm or 1.0,
                self.config.mesh_params.max_size_mm or 10.0
            ))
        
        start_time = time.time()
        
        # Run in Parallel
        try:
            # Note: Using multiprocessing.Pool requires the worker to be at module level
            with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
                results = pool.starmap(_parallel_split_worker, args)
        except Exception as e:
            self.log_message(f"Parallel split failed: {e}. Falling back to sequential.", "WARNING")
            # Sequential fallback (simplified reuse of logic)
            results = [_parallel_split_worker(0, 1, step_file, self.temp_stl_dir, self.triage_mode, CAD_DENSITY_THRESHOLD, MAX_DIAGONAL_FOR_TRIAGE, self.config.mesh_params.min_size_mm or 1.0, self.config.mesh_params.max_size_mm or 10.0)]
        
        # Combine results
        stl_files = []
        toxic_volumes = set()
        for worker_stls, worker_toxic in results:
            stl_files.extend(worker_stls)
            toxic_volumes.update(worker_toxic)
            
        elapsed = time.time() - start_time
        self.log_message(f"Split complete in {elapsed:.2f}s: {len(stl_files)} healthy, {len(toxic_volumes)} triaged/toxic (will box)")
        
        self.toxic_volumes = toxic_volumes
        return stl_files
    
    def _fast_pass_meshing(self, stl_files: List[str]) -> List[str]:
        """
        Robust Parallel Volume Meshing
        Replaced fragile sequential loop with Process-Isolated Batch Mesher.
        """
        import tempfile
        self.temp_mesh_dir = tempfile.mkdtemp(prefix="assembly_meshes_")
        
        # Prepare arguments for parallel pool
        tasks = []
        for stl in stl_files:
            basename = os.path.basename(stl).replace(".stl", ".msh")
            out_path = os.path.join(self.temp_mesh_dir, basename)
            # Dynamic timeout: 90s for large/immune parts, 30s for small ones
            timeout = 90 if os.path.exists(stl + ".immune") else 30
            min_s = self.config.mesh_params.min_size_mm or 1.0
            max_s = self.config.mesh_params.max_size_mm or 10.0
            tasks.append((stl, out_path, timeout, min_s, max_s))

        self.log_message(f"Starting Robust Volume Meshing ({len(tasks)} parts)...")
        start_t = time.time()
        
        # Execute in parallel
        # Note: Workers are 100% isolated processes, so they don't share memory/crash the GUI
        try:
            with multiprocessing.Pool(processes=4) as pool:
                mesh_files = pool.starmap(_robust_mesh_task, tasks)
        except Exception as e:
            self.log_message(f"Parallel meshing error: {e}", "ERROR")
            return []

        elapsed = time.time() - start_t
        self.log_message(f"Volume meshing complete in {elapsed:.2f}s")
        
        # Validate results
        valid_meshes = [m for m in mesh_files if os.path.exists(m)]
        return valid_meshes
    
    def _box_failed_parts(self) -> List[str]:
        """Box failed parts immediately (no repair attempt)"""
        boxed_meshes = []
        
        for stl_path in self.failed_parts:
            basename = os.path.basename(stl_path).replace(".stl", ".msh")
            output_path = os.path.join(self.temp_mesh_dir, basename)
            
            success, msg = generate_bounding_box_mesh(stl_path, output_path)
            if success:
                boxed_meshes.append(output_path)
                self.boxed_parts.append(basename)
            else:
                self.log_message(f"Boxing failed for {basename}: {msg}", "ERROR")
        
        return boxed_meshes
    
    def _merge_assembly(self, mesh_files: List[str], output_file: str) -> bool:
        """
        Merge all component meshes into single model with ID-shifted nodes.
        
        Each parallel worker generates meshes starting at Node #1.
        Naive gmsh.merge() causes ID collisions ("Teleporter" spikes).
        This method explicitly shifts Node/Element IDs to guarantee uniqueness.
        """
        if not mesh_files:
            return False
            
        try:
            self.log_message(f"Assembling {len(mesh_files)} components with ID-shifting...")
            
            # Master Arrays to hold the combined data
            all_nodes = []      # List of dicts with node data
            all_elements = []   # List of dicts with element data
            boxed_physical_tags = []  # Track which Physical Groups are boxed
            
            node_offset = 0
            element_offset = 0
            vol_tag_offset = 0
            
            # Load each mesh file and extract with shifted IDs
            self.initialize_gmsh()
            
            for i, msh_path in enumerate(mesh_files):
                try:
                    # Check if this is a boxed part
                    basename = os.path.basename(msh_path)
                    is_boxed_part = basename in self.boxed_parts
                    
                    # Clear and Load
                    gmsh.clear()
                    gmsh.merge(msh_path)
                    
                    # Get Volume Entities (Dimension 3)
                    vol_tags = gmsh.model.getEntities(3)
                    
                    for dim, tag in vol_tags:
                        # New unique tag for the master model
                        new_tag = tag + vol_tag_offset
                        
                        # Track if this volume came from a boxed part
                        if is_boxed_part:
                            boxed_physical_tags.append(new_tag)
                        
                        # --- NODES ---
                        node_tags, coords, parametric = gmsh.model.mesh.getNodes(dim, tag, includeBoundary=True)
                        
                        # Shift Node IDs
                        shifted_node_tags = [int(t + node_offset) for t in node_tags]
                        
                        all_nodes.append({
                            "dim": dim,
                            "tag": new_tag,
                            "node_tags": shifted_node_tags,
                            "coords": list(coords),
                            "parametric": list(parametric) if len(parametric) > 0 else []
                        })
                        
                        # --- ELEMENTS ---
                        elem_types, elem_tags_list, elem_node_tags_list = gmsh.model.mesh.getElements(dim, tag)
                        
                        for e_type, e_tags, e_nodes in zip(elem_types, elem_tags_list, elem_node_tags_list):
                            shifted_elem_tags = [int(t + element_offset) for t in e_tags]
                            shifted_elem_nodes = [int(t + node_offset) for t in e_nodes]
                            
                            all_elements.append({
                                "dim": dim,
                                "tag": new_tag,
                                "type": int(e_type),
                                "elem_tags": shifted_elem_tags,
                                "node_tags": shifted_elem_nodes
                            })
                    
                    # Update offsets
                    max_n = gmsh.model.mesh.getMaxNodeTag()
                    max_e = gmsh.model.mesh.getMaxElementTag()
                    node_offset += max_n
                    element_offset += max_e
                    vol_tag_offset += len(vol_tags)
                    
                except Exception as e:
                    self.log_message(f"Warning: Failed to load {os.path.basename(msh_path)}: {e}", "WARNING")
                    continue
            
            # --- RECONSTRUCTION ---
            self.log_message(f"Rebuilding Master Model with {len(all_nodes)} node sets...")
            gmsh.clear()
            gmsh.model.add("MasterAssembly")
            
            # 1. Create Discrete Entities (Volumes)
            created_entities = set()
            for item in all_nodes:
                if item["tag"] not in created_entities:
                    gmsh.model.addDiscreteEntity(item["dim"], item["tag"])
                    created_entities.add(item["tag"])
            
            # 2. Add Nodes
            for item in all_nodes:
                gmsh.model.mesh.addNodes(item["dim"], item["tag"], item["node_tags"], item["coords"], item["parametric"])
            
            # 3. Add Elements
            for item in all_elements:
                gmsh.model.mesh.addElements(item["dim"], item["tag"], [item["type"]], [item["elem_tags"]], [item["node_tags"]])
            
            # 4. Create Physical Groups (One per volume)
            for tag in created_entities:
                gmsh.model.addPhysicalGroup(3, [tag], tag, name=f"Part_{tag}")
            
            # 5. Save - Force MSH 2.2 for maximum viewer compatibility
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
            gmsh.option.setNumber("Mesh.Binary", 1)
            gmsh.write(output_file)
            
            # 6. Export boxed parts metadata for viewer debugging
            import json
            boxed_json_path = output_file.replace(".msh", "_boxed.json")
            with open(boxed_json_path, "w") as f:
                json.dump({"boxed_physical_tags": boxed_physical_tags}, f)
            self.log_message(f"Boxed parts metadata saved: {len(boxed_physical_tags)} boxed volumes")
            
            self.finalize_gmsh()
            self.log_message(f"Assembly complete: {len(created_entities)} parts merged")
            return True
            
        except Exception as e:
            self.log_message(f"Merge failed: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            try: gmsh.finalize()
            except: pass
            return False
    
    def _cleanup_temp_dirs(self):
        """Clean up temporary directories with retry for Windows locking"""
        import shutil
        import time
        
        for temp_dir in [self.temp_stl_dir, self.temp_mesh_dir]:
            if temp_dir and os.path.exists(temp_dir):
                # Try up to 3 times to account for lazy file release on Windows
                for attempt in range(3):
                    try:
                        shutil.rmtree(temp_dir)
                        break
                    except Exception as e:
                        if attempt < 2:
                            time.sleep(0.5) # Wait for file handles to clear
                        else:
                            self.log_message(f"Cleanup warning: {e}", "WARNING")
