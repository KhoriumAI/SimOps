import os
import time
import subprocess
import logging
import numpy as np
import gmsh
from pathlib import Path
from typing import Dict, Any, List, Optional
from .base import ISolver
from ..materials import get_material, MaterialProperties

from core.logging.sim_logger import SimLogger

logger = SimLogger("CalculiXAdapter")

class CalculiXAdapter(ISolver):
    """
    Adapter for running thermal analysis using CalculiX (ccx).
    Generates .inp files from Gmsh mesh and parsing .frd results.
    Supports: Steady State, Transient, Convection (*FILM), and Tet4/Tet10/Tri3/Tri6.
    """
    
    def __init__(self, ccx_binary: str = "ccx"):
        self.ccx_binary = ccx_binary
        
        # Check if we are on Windows and check default install location
        if os.name == 'nt' and self.ccx_binary == "ccx":
             default_paths = [
                 Path(r"C:\calculix\calculix_2.22_4win\ccx.exe"),
                 Path(r"C:\Users\markm\Downloads\Simops\calculix_native\CalculiX-2.23.0-win-x64\bin\ccx.exe")
             ]
             for dp in default_paths:
                 if dp.exists():
                     self.ccx_binary = str(dp)
                     logger.info(f"[CalculiX] Found binary at {self.ccx_binary}")
                     break
        else:
             logger.info(f"[CalculiX] Using binary: {self.ccx_binary}")
        
    def run(self, mesh_file: Path, output_dir: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run CalculiX thermal analysis.
        """
        job_name = mesh_file.stem.replace(".msh", "")
        inp_file = output_dir / f"{job_name}.inp"
        
        logger.log_stage("Meshing (Conversion)")
        logger.info(f"[CalculiX] Converting mesh {mesh_file.name} to INP...")
        
        # Scale Factor (mm->m default=1.0 unless configured)
        scale_factor = config.get("unit_scaling", 1.0)
        
        # 1. Convert Mesh and Generate INP
        logger.info(f"DEBUG_CONFIG: {config}")
        stats = self._generate_inp(mesh_file, inp_file, config, scale=scale_factor)
        
        # 2. Run CalculiX
        logger.log_stage("Solving (CalculiX)")
        logger.info(f"[CalculiX] executing {self.ccx_binary} {job_name}...")
        
        # Binary validation
        import shutil
        actual_binary = shutil.which(self.ccx_binary) or self.ccx_binary
        if not os.path.exists(actual_binary) and not shutil.which(self.ccx_binary):
            err_msg = f"CalculiX binary not found: {self.ccx_binary}. Check your installation."
            logger.log_error("BINARY_NOT_FOUND", err_msg)
            raise FileNotFoundError(err_msg)

        log_file = output_dir / f"{job_name}.log"
        stop_file = output_dir / "STOP_SIM"
        
        start_time = time.time()
        
        try:
            # ccx expects jobname without extension
            cwd = str(output_dir)
            
            # Run
            cmd = [self.ccx_binary, job_name]
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "4"
            
            interrupted = False
            process = subprocess.Popen(
                cmd, 
                cwd=cwd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                encoding='utf-8',
                errors='replace'
            )
            
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                
                if line:
                    print(f"[CCX] {line}", end='')
                    
                # Graceful Stop Check
                if stop_file.exists():
                    logger.warning(f"[CalculiX] Stop signal detected! Terminating {job_name}...")
                    process.terminate()
                    interrupted = True
                    # Wait a bit for it to flush
                    try:
                        process.wait(timeout=5)
                    except:
                        process.kill()
                    stop_file.unlink() # Cleanup signal file
                    break
            
            if process.returncode != 0 and not interrupted:
                err_msg = f"CalculiX execution failed with code {process.returncode}"
                logger.log_error("SOLVER_CRASH", err_msg)
                raise RuntimeError(err_msg)
                
            if interrupted:
                logger.info("[CalculiX] Proceeding to report generation from partial results...")
            else:
                logger.info("CalculiX execution complete.")
                
        except Exception as e:
            logger.error(f"CalculiX run error: {e}")
            raise
            
        solve_time = time.time() - start_time
        logger.log_metric("solve_time", solve_time, "s")
        
        # 3. Parse Results
        frd_file = output_dir / f"{job_name}.frd"
        if not frd_file.exists():
            raise FileNotFoundError(f"CalculiX output file not found: {frd_file}")
            
        results = self._parse_frd(frd_file, stats['node_map'], stats['elements'], config)
        results['solve_time'] = solve_time
        results['mesh_stats'] = stats
        results['success'] = True  # Mark as successful for worker
        
        return results

    def _generate_inp(self, mesh_file: Path, inp_file: Path, config: Dict[str, Any], scale: float = 1.0) -> Dict:
        """
        Reads Gmsh file and writes CalculiX input deck.
        Returns dict with node_map (internal ID -> coords) for parsing later.
        """
        gmsh.initialize()
        gmsh.open(str(mesh_file))
        
        try:
            # Nodes
            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            node_coords = node_coords.reshape(-1, 3) * scale
            
            # Volume Elements (Tet4 = C3D4, Tet10 = C3D10)
            elem_types, elem_tags, elem_nodes = gmsh.model.mesh.getElements(dim=3)
            
            c3d4_elems = []
            c3d10_elems = []
            
            for i, etype in enumerate(elem_types):
                if etype == 4: # Tet4
                    nodes = elem_nodes[i].reshape(-1, 4).astype(int)
                    tags = elem_tags[i].astype(int)
                    chunk = np.column_stack((tags, nodes))
                    c3d4_elems.append(chunk)
                elif etype == 11: # Tet10
                    # Gmsh 11: 4 corners, 6 edges
                    # CalculiX C3D10: Permutation verified by skills/verify_tet10_mapping.py
                    # [0, 1, 2, 3, 4, 5, 6, 7, 9, 8] (Swap89) passes CalculiX validation
                    # This matches the structural adapter mapping.
                    raw_nodes = elem_nodes[i].reshape(-1, 10).astype(int)
                    tags = elem_tags[i].astype(int)
                    permuted_nodes = raw_nodes[:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]]
                    chunk = np.column_stack((tags, permuted_nodes))
                    c3d10_elems.append(chunk)

            if not c3d4_elems and not c3d10_elems:
                raise ValueError("No Tetrahedral elements found in mesh")
            
            # Skin Elements (Tri3=2, Tri6=9) for Convection (*FILM)
            surf_types, surf_tags, surf_nodes = gmsh.model.mesh.getElements(dim=2)
            tri3_list = []
            tri6_list = []
            
            for i, etype in enumerate(surf_types):
                 if etype == 2: # Tri3
                     nodes = surf_nodes[i].reshape(-1, 3).astype(int)
                     tags = surf_tags[i].astype(int)
                     tri3_list.append(np.column_stack((tags, nodes)))
                 elif etype == 9: # Tri6
                     nodes = surf_nodes[i].reshape(-1, 6).astype(int)
                     tags = surf_tags[i].astype(int)
                     tri6_list.append(np.column_stack((tags, nodes)))
            
            # Ensure Unique Connectivity (Crucial for CalculiX shell normal estimation)
            # Ensure Unique Connectivity (Crucial for CalculiX shell normal estimation)
            # ANTIGRAVITY FIX: Bypassed sorting logic to preserve winding order.
            # Gmsh getElements returns correct topology; sorting destroys it.
            if tri3_list:
                tri3_elems = np.vstack(tri3_list)
            else:
                tri3_elems = []
                
            if tri6_list:
                tri6_elems = np.vstack(tri6_list)
            else:
                tri6_elems = []


            
            # Combine all for parsing/viz return
            if c3d4_elems:
                all_elems = np.vstack(c3d4_elems)
            elif c3d10_elems:
                all_elems = np.vstack(c3d10_elems)
            else:
                all_elems = np.vstack(c3d4_elems + c3d10_elems)
            
            # Heat Source Calculation (Needs node_coords)
            heat_load = config.get("heat_load_watts", 0.0)
            bf_val = 0.0
            if heat_load > 0:
                bbox_vol = (np.max(node_coords[:,0]) - np.min(node_coords[:,0])) * \
                           (np.max(node_coords[:,1]) - np.min(node_coords[:,1])) * \
                           (np.max(node_coords[:,2]) - np.min(node_coords[:,2]))
                if bbox_vol > 0:
                    # BF in mW/mm³ (mm-tonne-s with power in mW)
                    bf_val = (heat_load * 1000.0) / bbox_vol
                    logger.info(f"   [Physics] BBox Volume: {bbox_vol:.2f} mm³")
                    logger.info(f"   [Physics] BF Density: {bf_val:.6g} mW/mm³")
                else:
                    logger.error("[Physics] Invalid bbox_vol = 0! Check mesh geometry.") 
            
            # Identify Boundary Nodes from Physical Groups (Semantic Detection)
            # Look for BC_HeatSource physical group from mesh
            hot_tags = np.array([], dtype=int)
            cold_tags = np.array([], dtype=int)
            heat_source_found = False
            
            physical_groups = gmsh.model.getPhysicalGroups(dim=2)  # Surface groups
            for dim, ptag in physical_groups:
                name = gmsh.model.getPhysicalName(dim, ptag)
                if 'heatsource' in name.lower() or 'heat_source' in name.lower() or 'bc_heatsource' in name.lower():
                    # Get entities in this physical group
                    entities = gmsh.model.getEntitiesForPhysicalGroup(dim, ptag)
                    for entity in entities:
                        node_tags_entity, _, _ = gmsh.model.mesh.getNodes(dim, entity)
                        hot_tags = np.concatenate([hot_tags, node_tags_entity.astype(int)])
                    heat_source_found = True
                    logger.info(f"[CalculiX] Using BC_HeatSource physical group for hot boundary ({len(hot_tags)} nodes)")
            
            # Fallback to geometric heuristic if no semantic tags found
            if not heat_source_found:
                # Find best axis for gradient (avoid flat axes)
                ranges = [
                    np.max(node_coords[:, 0]) - np.min(node_coords[:, 0]),
                    np.max(node_coords[:, 1]) - np.min(node_coords[:, 1]),
                    np.max(node_coords[:, 2]) - np.min(node_coords[:, 2])
                ]
                
                # Default to Z, but if it's too flat, pick the largest dimension
                axis_idx = 2
                if ranges[2] < 1e-4 or (ranges[2] < 0.01 * max(ranges[0], ranges[1])):
                    axis_idx = np.argmax(ranges)
                    logger.info(f"[CalculiX] Mesh flat in Z ({ranges[2]:.4f}). Auto-selecting axis {axis_idx} (range={ranges[axis_idx]:.4f})")

                coords = node_coords[:, axis_idx]
                c_min, c_max = np.min(coords), np.max(coords)
                c_rng = c_max - c_min if c_max > c_min else 1.0
                tol = config.get("bc_tolerance", 0.01)
                
                # NEW: heat_source_at_min option (applies to the selected axis)
                if config.get("heat_source_at_z_min", True):
                    hot_mask = coords < (c_min + tol * c_rng)
                    cold_mask = coords > (c_max - tol * c_rng)
                else:
                    hot_mask = coords > (c_max - tol * c_rng)
                    cold_mask = coords < (c_min + tol * c_rng)
                
                hot_tags = node_tags[hot_mask].astype(int)
                cold_tags = node_tags[cold_mask].astype(int)
                
                # Safety: Ensure we don't select the entire mesh if it's truly degenerate
                if len(hot_tags) >= len(node_tags) * 0.9:
                    # Just grab top/bottom few nodes
                    sort_idx = np.argsort(coords)
                    hot_tags = node_tags[sort_idx[:max(1, len(node_tags)//100)]].astype(int)
                    cold_tags = node_tags[sort_idx[-max(1, len(node_tags)//100):]].astype(int)
                    logger.warning("[CalculiX] BC selection too broad. Capping to top/bottom 1% of nodes.")

                logger.info(f"[CalculiX] Using geo-fallback (axis {axis_idx}) for boundaries: {len(hot_tags)} hot, {len(cold_tags)} cold nodes.")

            
            # Write INP
            with open(inp_file, 'w') as f:
                f.write("*HEADING\n")
                f.write(f"SimOps Thermal Analysis: {mesh_file.name}\n")
                
                # Nodes
                f.write("*NODE\n")
                # Unit Scaling: User requested MM units. Defaulting to 1.0 (No scaling).
                scale = config.get("unit_scaling", 1.0) 
                logger.info(f"[CalculiX] Using mesh scale {scale} (1.0 = mm)")

                
                for tag, (x,y,z) in zip(node_tags, node_coords):
                    f.write(f"{int(tag)}, {x*scale:.6f}, {y*scale:.6f}, {z*scale:.6f}\n")
                    
                # Volume Elements
                if c3d4_elems:
                    f.write("*ELEMENT, TYPE=C3D4, ELSET=E_Vol\n")
                    for row in np.vstack(c3d4_elems):
                        f.write(f"{row[0]}, {row[1]}, {row[2]}, {row[3]}, {row[4]}\n")
                        
                if c3d10_elems:
                    f.write("*ELEMENT, TYPE=C3D10, ELSET=E_Vol\n")
                    for row in np.vstack(c3d10_elems):
                         # ID, 1-10
                         f.write(f"{row[0]}")
                         for n in row[1:]:
                             f.write(f", {n}")
                         f.write("\n")

                # Skin Elements (S3/S6) - Re-enabled for convection
                if len(tri3_elems) > 0:
                    f.write("*ELEMENT, TYPE=S3, ELSET=E_Skin\n")
                    for row in np.vstack(tri3_elems):
                        f.write(f"{row[0]}, {row[1]}, {row[2]}, {row[3]}\n")

                if len(tri6_elems) > 0:
                     f.write("*ELEMENT, TYPE=S6, ELSET=E_Skin\n")

                     for row in np.vstack(tri6_elems):
                         f.write(f"{row[0]}")
                         for n in row[1:]:
                             f.write(f", {n}")
                         f.write("\n")
                    
                # Materials & Sections
                # Consistent Units: Watts, mm, kg, Kelvin, Seconds
                # Conductivity: W/(mm K) = W/(m K) * 1e-3
                # Density: kg/mm^3 = kg/m^3 * 1e-9
                # Specific Heat: J/(kg K) [unchanged]
                
                f.write("*MATERIAL, NAME=Mat1\n")
                
                # Fetch material properties
                mat_name = config.get("material_name")
                material: Optional[MaterialProperties] = None
                
                if mat_name:
                    try:
                        material = get_material(mat_name)
                        logger.info(f"[CalculiX] Using material '{mat_name}' from DB")
                    except KeyError:
                        logger.warning(f"[CalculiX] Material '{mat_name}' not found in DB. Falling back to manual properties.")
                
                # Default to Aluminum 6061-T6 if no material specified and no manual override provided
                # This replaces the hardcoded k=150 (Steel-ish?) with k=167 (Aluminum)
                if not material:
                    # Check if user provided manual overrides
                    if "thermal_conductivity" not in config and "density" not in config:
                         logger.info("[CalculiX] No material specified. Defaulting to Aluminum 6061-T6.")
                         material = get_material("Aluminum_6061_T6")

                # Extraction helpers
                def get_k(m): return m.conductivity if m else 150.0 
                def get_rho(m): return m.density if m else 2700.0
                def get_cp(m): return m.specific_heat if m else 900.0

                # Priority: Config Override > Material DB > Hardcoded Fallback (shouldn't happen with default above)
                k_si = config.get("thermal_conductivity", get_k(material))
                rho_si = config.get("density", get_rho(material))
                cp_si = config.get("specific_heat", get_cp(material))

                # ===================================================================
                # UNIT SCALING: mm-tonne-s system
                # ===================================================================
                # Thermal Conductivity: W/mK -> mW/mmK (N/sK)
                if isinstance(k_si, list):
                    k = [[val * 0.001, t] for val, t in k_si]
                else:
                    k = k_si * 0.001
                
                # Density: kg/m³ -> tonne/mm³
                if isinstance(rho_si, list):
                    rho = [[val * 1e-12, t] for val, t in rho_si]
                else:
                    rho = rho_si * 1e-12
                
                # Specific Heat: J/kgK -> mm²/s²K
                if isinstance(cp_si, list):
                    cp = [[val * 1e6, t] for val, t in cp_si]
                else:
                    cp = cp_si * 1e6
                
                def write_param(card_name, value):
                    f.write(f"{card_name}\n")
                    if isinstance(value, list):
                        for pair in value:
                            # Expecting [Value, Temp]
                            f.write(f"{pair[0]:g}, {pair[1]:g}\n")
                    else:
                        f.write(f"{value:g}\n")
                        
                # Write material properties to INP file
                write_param("*CONDUCTIVITY", k)
                write_param("*DENSITY", rho)
                write_param("*SPECIFIC HEAT", cp)
                
                # Log final scaled values for verification
                logger.info(f"   [Material] k={k if not isinstance(k, list) else k[0][0]:.6g} mW/mmK")
                logger.info(f"   [Material] rho={rho if not isinstance(rho, list) else rho[0][0]:.6g} tonne/mm^3")
                logger.info(f"   [Material] cp={cp if not isinstance(cp, list) else cp[0][0]:.6g} mm^2/s^2K")
                
                f.write("*SOLID SECTION, ELSET=E_Vol, MATERIAL=Mat1\n")
                
                if len(tri3_elems) > 0 or len(tri6_elems) > 0:
                     # Shell Section for Skin
                     # Thickness 1.0mm (standard for mm units) helps avoiding zero-normal issues
                     f.write("*SHELL SECTION, ELSET=E_Skin, MATERIAL=Mat1\n")
                     f.write("1.0\n")


                
                # Create Node Set for All Nodes (for Initial Conditions)
                f.write("*NSET, NSET=N_All\n")
                tag_strs = [str(t) for t in node_tags]
                for k in range(0, len(tag_strs), 8):
                    f.write(", ".join(tag_strs[k:k+8]) + "\n")

                # Intelligent Mapping: Check aliases
                # ALL inputs expected in CELSIUS. Defaults are in Celsius too.
                amb_in = config.get("ambient_temperature")
                if amb_in is None: 
                    amb_in = config.get("ambient_temp_c")
                if amb_in is None:
                    amb_in = config.get("ambient_temp_celsius", 20.0)  # Default 20°C in CELSIUS

                # Initial Conditions (Required for Transient)
                init_temp_in = config.get("initial_temperature", amb_in)
                init_temp = init_temp_in + 273.15
                logger.info(f"   [Physics] Init Temp: {init_temp_in}C -> {init_temp}K")
                
                f.write("*INITIAL CONDITIONS, TYPE=TEMPERATURE\n")
                f.write(f"N_All, {init_temp}\n")

                # Step
                f.write("*STEP\n")
                
                if config.get("transient", True):  # Default to transient
                     # Transient Analysis
                     dt = config.get("time_step", 2.0)
                     t_end = config.get("duration", 60.0) # Increased default duration
                     f.write("*HEAT TRANSFER, DIRECT\n")
                     f.write(f"{dt}, {t_end}\n")
                else:
                     # Steady State
                     f.write("*HEAT TRANSFER, STEADY STATE\n")

                # Heat Source (Volumetric) moved after nodes are loaded

                if bf_val > 0:
                     f.write("*DFLUX\n")
                     f.write(f"E_Vol, BF, {bf_val:g}\n")
                     logger.info(f"   [Physics] Applied Volumetric Heat: {heat_load}W -> BF={bf_val:g}")
                f.write("*BOUNDARY\n")
                # ANTIGRAVITY FIX: ALL temperature inputs are in CELSIUS.
                # Check multiple key aliases for heat source temperature
                t_hot_in = config.get("heat_source_temperature")
                if t_hot_in is None:
                    t_hot_in = config.get("source_temp_c")
                if t_hot_in is None:
                    t_hot_in = config.get("source_temp_celsius", 100.0)  # Default 100°C in CELSIUS
                
                t_cold_in = amb_in
                
                # Heuristic: User inputs are Celsius (e.g. 20, 100). Solver needs Kelvin.
                # Threshold check < 200 is risky if user inputs 150K (cryogenic).
                # But given context, inputs are C. Use explicit conversion log.
                
                t_hot_K = t_hot_in + 273.15
                t_cold_K = t_cold_in + 273.15
                
                logger.info(f"   [Physics] Converted Temp: Hot {t_hot_in}C -> {t_hot_K}K, Cold {t_cold_in}C -> {t_cold_K}K")
                
                t_hot = t_hot_K
                t_cold = t_cold_K
                
                # Hot BC (Z-Max) - Always applied if T_hot defined (conceptually)
                # We could make this optional too, but usually we need a source.
                if config.get("fix_hot_boundary", True):
                    for tag in hot_tags:
                        f.write(f"{tag}, 11, 11, {t_hot}\n")
                        
                # Cold BC - Default to True for a clear gradient unless explicitly disabled
                if config.get("fix_cold_boundary", True): 
                    for tag in cold_tags:
                        f.write(f"{tag}, 11, 11, {t_cold}\n")
                    logger.info(f"   [Physics] Fixed Cold Boundary at {t_cold_in}C ({len(cold_tags)} nodes)")
                    
                # Convection (*SFILM on *SURFACE)
                # The old *FILM on E_Skin shell elements fails in CCX 2.17.
                # Instead, we use *SURFACE to define boundary faces on the volume elements,
                # then apply *SFILM (surface film) to that surface.
                # ===================================================================
                # UNIT SCALING: h in mm-tonne-s system
                # h [W/m²K] -> h [mW/mm²K] = h_si * 1e-3
                # Benchmark: 25.0 W/m²K -> 0.025 mW/mm²K
                # ===================================================================
                h_si = config.get("convection_coeff", 25.0) 
                h = h_si * 1e-3  # FIXED: was 1e-6 (wrong by 1000×)
                
                T_inf_in = amb_in # Use resolved ambient
                T_inf = T_inf_in + 273.15
                logger.info(f"   [Physics] Convection h: {h_si} W/m²K -> {h:.6g} mW/mm²K")
                logger.info(f"   [Physics] Convection T_inf: {T_inf_in}C -> {T_inf}K")
                
                if h_si > 0 and (len(tri3_elems) > 0 or len(tri6_elems) > 0):
                    # Define a surface from the skin element faces
                    # For surface loads on volume elements, we need to identify the faces.
                    # Since we extracted boundary triangles, they correspond to volume faces.
                    # We use TYPE=ELEMENT to define a surface from element sets.
                    f.write("*SURFACE, NAME=S_Convection, TYPE=ELEMENT\n")
                    f.write("E_Skin, S1\n")  # S1 is the primary face for shell/surface elements
                    
                    # Apply surface film (convection) to the defined surface
                    f.write(f"*SFILM\n")
                    # FIX: Remove spaces for safer parsing
                    f.write(f"S_Convection,F,{T_inf},{h}\n")

                # =========================================================
                # NEW: Surface Flux (*DFLUX)
                # =========================================================
                q_flux_si = config.get("surface_flux_wm2")
                if q_flux_si is not None:
                     # Identify where to apply flux. Default: Hot Boundary Nodes -> Elements?
                     # *DFLUX applies to element faces.
                     # We need element faces corresponding to the hot zone.
                     # This is tricky without explicitly finding faces.
                     # Re-use the E_Skin strategy? 
                     # Create a new ELSET E_FluxSelect containing skin elements near Hot Zone.
                     
                     # Filter tri_elems by Z location or heuristic
                     if len(tri3_elems) + len(tri6_elems) > 0:
                         flux_elems = []
                         
                         skin_nodes_flat = []
                         skin_tags_flat = []
                         if len(tri3_elems) > 0:
                             skin_nodes_flat.append(tri3_elems[:, 1:]) 
                             skin_tags_flat.append(tri3_elems[:, 0])
                         if len(tri6_elems) > 0:
                             skin_nodes_flat.append(tri6_elems[:, 1:]) 
                             skin_tags_flat.append(tri6_elems[:, 0])
                             
                         # Check geometric bounds of skin elements
                         for stags, snodes in zip(skin_tags_flat, skin_nodes_flat):
                             for i, eid in enumerate(stags):
                                 # Get nodes for this element
                                 enodes = snodes[i]
                                 # Check if these nodes are in hot_tags
                                 if np.any(np.isin(enodes, hot_tags)):
                                     flux_elems.append(eid)
                                     
                         if flux_elems:
                             f.write(f"*ELSET, ELSET=E_FluxSelect\n")
                             for eid in flux_elems:
                                 f.write(f"{int(eid)},\n")
                             
                             f.write("*SURFACE, NAME=S_Flux, TYPE=ELEMENT\n")
                             f.write("E_FluxSelect, S1\n")
                             
                             # ===================================================================
                             # UNIT SCALING: Surface flux in mm-tonne-s system
                             # q [W/m²] -> q [mW/mm²] = q_si * 1e-3
                             # ===================================================================
                             q_flux = q_flux_si * 1e-3  # FIXED: was 1e-6
                             f.write(f"*DFLUX\n")
                             # Load type S means flux per unit area
                             f.write(f"S_Flux, S, {q_flux}\n")
                             logger.info(f"[CalculiX] Applied Surface Flux {q_flux_si} W/m^2 to {len(flux_elems)} elements")
                         else:
                             logger.warning("[CalculiX] Surface Flux requested but no elements found in Hot Zone!")

                # =========================================================
                # NEW: Volumetric Heat Source (*DFLUX with BF)
                # =========================================================
                # Q in W/m^3 -> W/mm^3 (1e-9)
                vol_heat_si = config.get("volumetric_heat_wm3")
                if vol_heat_si is not None:
                    vol_heat = vol_heat_si * 1e-9
                    # CalculiX uses *DFLUX with BF (Body Flux) for volumetric heat
                    f.write(f"*DFLUX\n")
                    f.write(f"E_Vol, BF, {vol_heat}\n")
                    logger.info(f"[CalculiX] Applied Volumetric Flux (BF) {vol_heat_si} W/m^3 to E_Vol")



                # Output
                # *NODE PRINT generates .dat (ASCII tabular) which is parseable 
                # *NODE FILE generates .frd (ASCII results) for visualization/parsing
                # Always set FREQUENCY=1 to ensure output is written
                f.write("*NODE FILE, NSET=N_All, FREQUENCY=1\n")
                f.write("NT\n")
                f.write("*NODE PRINT, NSET=N_All, FREQUENCY=1\n")
                f.write("NT\n")
                f.write("*END STEP\n")
            
            # ANTIGRAVITY FIX: Remap element node tags to 0-based indices
            # The all_elems array contains [Tag, Node1, Node2...] where Nodes are 1-based Tags.
            # We need them to be indices into the node_coords array.
            
            # Create a lookup table: Node Tag -> Index
            tag_to_index = {tag: idx for idx, tag in enumerate(node_tags)}
            
            # Remap elements
            # all_elems[:, 1:] contains the node tags
            remapped_elems = all_elems[:, 1:].copy()
            
            # Vectorized mapping is hard with dict, use loop or np.searchsorted if sorted
            # Assuming node_tags returned by Gmsh are sorted (usually strict increasing)
            # Use searchsorted for speed
            
            # Verify if tags are sorted
            if np.all(np.diff(node_tags) >= 0):
                # Sorted
                remapped_elems = np.searchsorted(node_tags, remapped_elems)
            else:
                # Fallback to slow map or argsort
                sorter = np.argsort(node_tags)
                remapped_elems = sorter[np.searchsorted(node_tags, remapped_elems, sorter=sorter)]
                
            return {
                'node_map': dict(zip(node_tags, node_coords)),
                'elements': remapped_elems, # Now 0-based indices
                'num_nodes': len(node_tags),
                'num_elements': len(remapped_elems)
            }
        finally:
            gmsh.finalize()
            
    def _parse_frd(self, frd_file: Path, node_map: Dict, elements: np.ndarray, config: Dict[str, Any] = {}) -> Dict:
        """
        Parses .frd file for NT (Nodal Temperature) values.
        Supports Transient (Multiple Steps).
        Returns dict compatible with SimulationResult, plus 'time_series'.
        """
        
        with open(frd_file, 'r') as f:
            lines = f.readlines()
            
        # Data storage
        all_steps = [] # List of {time: float, temperatures: Dict[nid, float]}
        
        current_time = 0.0
        current_temps = {} # nid -> val
        reading_nt = False
        
        # Check if we found ANY steps
        found_data = False
        
        for line in lines:
            # 1. Check for Step Header (Time)
            # Format: "  100CL  101 1.000000000       11008 ..."
            stripped = line.strip()
            if stripped.startswith("100CL"):
                parts = stripped.split()
                if len(parts) >= 3:
                    try:
                        # If we have collected data for previous step, store it
                        if current_temps:
                             all_steps.append({'time': current_time, 'data': current_temps})
                             
                        # Start new step
                        current_time = float(parts[2])
                        current_temps = {} # Reset buffer
                    except ValueError:
                        pass
            
            # For steady-state (no 100CL), check if we're starting to read data
            # and haven't initialized time yet
            if stripped.startswith("-4") and current_time == 0.0 and not current_temps:
                # This is likely a steady-state analysis, set time to 1.0
                current_time = 1.0
                

            # 2. Check for Temperature Block Start
            # " -4  NT..." or " -4  NDTEMP..."
            if stripped.startswith("-4") and ("NT" in stripped or "NDTEMP" in stripped or "P" in stripped):
                reading_nt = True
                found_data = True
                continue
                
            # 3. Check for Block End
            if stripped.startswith("-3"):
                if reading_nt:

                    reading_nt = False
                    # End of NT block for this step.
                    # We usually append to all_steps here? 
                    # But 100CL logic handles "Flush on new step".
                    # What about the LAST step?
                    # We need to flush after loop.
                continue
                
            # 4. Parse Data lines
            if reading_nt:
                # FRD format is fixed-width: " -1" marker + node_id (10 chars) + value
                # Example: " -1         13.29949E+002" = node 1, temp 329.949K
                # Use regex to handle this format
                try:
                    # CalculiX FRD Format (Scientific Notation):
                    # Columns 1-3:   " -1" (record type marker)
                    # Columns 4-13:  Node ID (10 chars, right-justified)
                    # Columns 14-25: Temperature value (12 chars, scientific notation)
                    # Example: " -1       1143.70102E+002"
                    #           ^^^  ^^^^^^^^^^  ^^^^^^^^^^^^
                    #          0-3     3-13        13-25
                    if line.startswith(" -1"):
                        nid = int(line[3:13].strip())
                        val = float(line[13:25].strip())
                        current_temps[nid] = val
                except (ValueError, IndexError):
                    pass

        # Flush final step check
        if current_temps:
             all_steps.append({'time': current_time, 'data': current_temps})
             
        logger.info(f"[CalculiX] Parsed {len(all_steps)} steps from FRD.")
        print(f"[CalculiX] Parsed {len(all_steps)} steps from FRD.")
        
        if not all_steps:
            # DEBUG: Print header of FRD file to see what's wrong
            try:
                print(f"FRD Header (first 20 lines):")
                for line in lines[:20]:
                    print(line.strip())
            except: pass
            logger.warning("Could not parse temperatures from FRD.")
            # Fallback to defaults?
            all_steps.append({'time': 0.0, 'data': {}})

        # Process Time Series Stats
        # And determine FINAL result (Last Step)
        
        time_series_stats = []
        final_temps = {}
        
        sorted_nids = sorted(node_map.keys())
        
        for step in all_steps:
             lbl = step['time']
             data = step['data']
             
             # Stats
             vals = list(data.values())
             if vals:
                 t_min = min(vals)
                 t_max = max(vals)
                 t_mean = sum(vals)/len(vals)
             else:
                 t_min = 300.0; t_max = 300.0; t_mean = 300.0
                 
             time_series_stats.append({
                 'time': lbl,
                 'min': t_min,
                 'max': t_max,
                 'mean': t_mean
             })
             
             final_temps = data # Update latest
             
        # Build full Time Series (Array Data for Visualization)
        time_series = []
        for step in all_steps:
             lbl = step['time']
             data = step['data']
             # Fill array aligned with sorted_nids
             step_T = []
             for nid in sorted_nids:
                 step_T.append(data.get(nid, 300.0)) # Fallback to 300K if missing
             time_series.append({
                 'time': lbl, 
                 'temperature': np.array(step_T)
             })
             
        # Convert Final Result to Array
        T_array = []
        parsed_coords = []
        
        for nid in sorted_nids:
            T_array.append(final_temps.get(nid, 300.0))
            parsed_coords.append(node_map[nid])
            
        T_array = np.array(T_array)
        parsed_coords = np.array(parsed_coords)
        
        t_min = float(np.min(T_array)) if len(T_array)>0 else 0.0
        t_max = float(np.max(T_array)) if len(T_array)>0 else 0.0

        # =====================================================================
        # CONVERGENCE DETECTION
        # Check if dT (mean temperature change between steps) has flatlined
        # =====================================================================
        converged = False
        convergence_step = None
        dT_threshold = 0.1  # K - if mean temp changes less than this, consider converged
        
        dT_history = []
        if len(time_series_stats) >= 3:
            # Calculate dT between consecutive steps
            for i in range(1, len(time_series_stats)):
                dT = abs(time_series_stats[i]['mean'] - time_series_stats[i-1]['mean'])
                dT_history.append(dT)
            
            # Check for flatline (last 3 steps all below threshold)
            if len(dT_history) >= 3:
                last_3_dT = dT_history[-3:]
                if all(dt < dT_threshold for dt in last_3_dT):
                    converged = True
                    convergence_step = time_series_stats[-3]['time']
                    logger.info(f"  [Convergence] Detected at t={convergence_step:.1f}s (dT < {dT_threshold}K for 3 steps)")
        elif len(time_series_stats) >= 2:
            # Still calc history even if not enough for 3-step flatline check
            for i in range(1, len(time_series_stats)):
                dT = abs(time_series_stats[i]['mean'] - time_series_stats[i-1]['mean'])
                dT_history.append(dT)
        
        if not converged and len(time_series_stats) > 0:
            logger.info(f"  [Convergence] Not reached in {len(time_series_stats)} steps. Final dT={dT_history[-1]:.3f}K" if len(time_series_stats) > 1 else "  [Convergence] Only 1 step, cannot assess")

        # =====================================================================
        # HEAT FLUX CALCULATION (Watts convected away)
        # Q = h * A * (T_surface - T_ambient)
        # Estimate surface area from mesh and use mean surface temperature
        # =====================================================================
        heat_flux_watts = None
        try:
            # Get surface nodes (approximate: nodes on outer boundary)
            # Use radial distance from Z-axis to identify surface
            r = np.sqrt(parsed_coords[:, 0]**2 + parsed_coords[:, 1]**2)
            r_max = np.max(r)
            surface_mask = r > (r_max * 0.95)  # Nodes within 5% of max radius
            
            if np.any(surface_mask):
                T_surface_mean = np.mean(T_array[surface_mask])
                T_amb_val = config.get("ambient_temperature")
                if T_amb_val is None: T_amb_val = config.get("ambient_temp_c")
                if T_amb_val is None: T_amb_val = 20.0
                
                T_ambient = T_amb_val + 273.15  # Convert C to K
                h = config.get("convection_coeff", 25.0)
                
                # Estimate surface area from bounding box (rough lateral surface)
                z_range = np.max(parsed_coords[:, 2]) - np.min(parsed_coords[:, 2])
                # Estimate radius from X/Y range
                r_est = (np.max(parsed_coords[:, 0]) - np.min(parsed_coords[:, 0])) / 2.0
                A_lateral = (2 * np.pi * r_est * z_range) / 1e6  # mm2 -> m2
                
                heat_flux_watts = h * A_lateral * (T_surface_mean - T_ambient)
                logger.log_metric("heat_flux_watts", heat_flux_watts, "W")
                logger.info(f"  [Heat Flux] Q = {heat_flux_watts:.1f}W (h={h} W/m^2K, A={A_lateral:.4f}m^2, dT={T_surface_mean - T_ambient:.1f}K)")
        except Exception as e:
            logger.warning(f"  [Heat Flux] Calculation failed: {e}")

        logger.info(f"   [CalculiX] Solver finished. Temp range: {t_min:.1f}K - {t_max:.1f}K ({t_min-273.15:.1f}C - {t_max-273.15:.1f}C)")
        
        return {
            'temperature': T_array,
            'node_coords': parsed_coords,
            'min_temp': t_min,
            'max_temp': t_max,
            'elements': elements,
            'time_series_stats': time_series_stats,
            'heat_flux_watts': heat_flux_watts if 'heat_flux_watts' in locals() else 0.0,
            'converged': converged,
            'convergence_step': convergence_step,
            'time_series': time_series,
            'convergence_threshold': dT_threshold
        }
