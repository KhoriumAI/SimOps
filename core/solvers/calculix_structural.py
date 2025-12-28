import os
import time
import subprocess
import logging
import numpy as np
import gmsh
from pathlib import Path
from typing import Dict, Any, List, Optional
from .base import ISolver

from core.logging.sim_logger import SimLogger

logger = SimLogger("CalculiXStructural")

class CalculiXStructuralAdapter(ISolver):
    """
    Adapter for running structural analysis using CalculiX (ccx).
    Supports: Static Linear Analysis, Gravity Loads, Displacement/Stress output.
    """
    
    def __init__(self, ccx_binary: str = "ccx"):
        self.ccx_binary = ccx_binary if ccx_binary else "ccx"
        
        # Check if we are on Windows and check default install location
        if os.name == 'nt' and self.ccx_binary == "ccx":
             default_path = Path(r"C:\calculix\calculix_2.22_4win\ccx.exe")
             if default_path.exists():
                 self.ccx_binary = str(default_path)
                 logger.info(f"[CalculiX-Struct] Found binary at {self.ccx_binary}")

    def run(self, mesh_file: Path, output_dir: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run CalculiX structural analysis.
        """
        job_name = mesh_file.stem.replace(".msh", "")
        # Distinguish structural jobs/files if needed, but usually dispatch handles name
        inp_file = output_dir / f"{job_name}.inp"
        
        logger.log_stage("Meshing (Conversion)")
        logger.info(f"[CalculiX-Struct] Converting mesh {mesh_file.name} to INP...")
        
        scale_factor = config.get("unit_scaling", 1.0)
        
        # 1. Generate INP
        stats = self._generate_inp(mesh_file, inp_file, config, scale=scale_factor)
        
        # 2. Run CalculiX
        logger.log_stage("Solving (CalculiX)")
        logger.info(f"[CalculiX-Struct] executing {self.ccx_binary} {job_name}...")
        
        # Binary validation
        import shutil
        actual_binary = shutil.which(self.ccx_binary) or self.ccx_binary
        if not os.path.exists(actual_binary) and not shutil.which(self.ccx_binary):
            err_msg = f"CalculiX binary not found: {self.ccx_binary}. Check your installation."
            logger.log_error("BINARY_NOT_FOUND", err_msg)
            raise FileNotFoundError(err_msg)

        start_time = time.time()
        
        try:
            cwd = str(output_dir)
            cmd = [self.ccx_binary, job_name]
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "4"
            
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
            
            if process.returncode != 0:
                err_msg = f"CalculiX execution failed with code {process.returncode}"
                logger.log_error("SOLVER_CRASH", err_msg)
                raise RuntimeError(err_msg)
                
            logger.info("CalculiX execution complete.")
            
        except Exception as e:
            logger.log_error("RUN_ERROR", str(e))
            raise
            
        solve_time = time.time() - start_time
        logger.log_metric("solve_time", solve_time, "s")
        
        # 3. Parse Results
        frd_file = output_dir / f"{job_name}.frd"
        if not frd_file.exists():
            raise FileNotFoundError(f"CalculiX output file not found: {frd_file}")
            
        results = self._parse_frd(frd_file, stats['node_map'], stats['elements'])
        results['solve_time'] = solve_time
        results['mesh_stats'] = stats
        results['num_elements'] = len(stats.get('elements', []))
        results['num_nodes'] = len(stats.get('node_map', {}))
        results['success'] = True
        
        # Sanity Checks
        disp_mag = results.get('displacement_magnitude', np.array([]))
        von_mises = results.get('von_mises', np.array([]))
        g_load = abs(config.get('gravity_load_g', 0.0))
        
        if len(disp_mag) > 0 and (np.max(disp_mag) > 1000.0 or np.isnan(np.max(disp_mag))):
            logger.warning(f"[CalculiX-Struct] SUSPICIOUS: Max Displacement = {np.max(disp_mag):.2f} mm")
            results['warnings'] = results.get('warnings', []) + ['Large displacement detected']
            
        if len(von_mises) > 0 and np.max(von_mises) < 1e-9 and g_load > 0.1:
            logger.warning("[CalculiX-Struct] SUSPICIOUS: Zero stress under load.")
            results['warnings'] = results.get('warnings', []) + ['Zero stress detected']
        
        return results

    def _generate_inp(self, mesh_file: Path, inp_file: Path, config: Dict[str, Any], scale: float = 1.0) -> Dict:
        """
        Generates CalculiX input deck for STATIC STRUCTURAL analysis.
        """
        gmsh.initialize()
        gmsh.open(str(mesh_file))
        
        try:
            # Nodes
            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            node_coords = node_coords.reshape(-1, 3) * scale
            
            # ANTIGRAVITY FIX: Sort by tag to ensure alignment with _parse_frd (which uses sorted keys)
            p = np.argsort(node_tags)
            node_tags = node_tags[p]
            node_coords = node_coords[p]
            
            # Volume Elements
            elem_types, elem_tags, elem_nodes = gmsh.model.mesh.getElements(dim=3)
            
            all_vol_elems = [] # For INP writing
            
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
                    # CalculiX C3D10: Validated Permutation [2,0,1,3, 6,4,5, 9,7,8]
                    # This maps Gmsh node logical order to CalculiX to ensure positive volume 
                    # and uniform stress (verified via tests/physics/verify_tet10_order.py).
                    
                    # UPDATE 2025-12-26: Previous permutation caused distortions. Identity failed due to
                    # mismatch in Edge 5 (1-3) and Edge 6 (2-3) ordering between GMSH and CCX.
                    # Validated Fix: Swap indices 8 and 9 (Edges 1-3 and 2-3).
                    
                    # Old (Rotated):
                    # perm_indices = [2, 0, 1, 3, 6, 4, 5, 9, 7, 8]
                    
                    # New (Swap89):
                    perm_indices = [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]
                    
                    raw_nodes = elem_nodes[i].reshape(-1, 10).astype(int)
                    permuted_nodes = raw_nodes[:, perm_indices]
                    tags = elem_tags[i].astype(int)
                    chunk = np.column_stack((tags, permuted_nodes))
                    c3d10_elems.append(chunk)

            if not c3d4_elems and not c3d10_elems:
                raise ValueError("No Tetrahedral elements found")
                
            # Combine for parsing return
            if c3d4_elems:
                all_elems_ret = np.vstack(c3d4_elems)
            elif c3d10_elems:
                all_elems_ret = np.vstack(c3d10_elems)
            else:
                all_elems_ret = np.vstack(c3d4_elems + c3d10_elems)

            # Identification of Fixed Boundary
            clamp_dir = config.get("clamping_direction", "X_MAX").upper()
            fixed_tol = config.get("fixed_boundary_tolerance", 0.02)
            
            coords_map = {'X': 0, 'Y': 1, 'Z': 2}
            axis_char = clamp_dir[0]
            axis_idx = coords_map.get(axis_char, 2)
            
            v = node_coords[:, axis_idx]
            v_min, v_max = np.min(v), np.max(v)
            v_rng = v_max - v_min
            
            if "MAX" in clamp_dir:
                fixed_mask = v > (v_max - fixed_tol * v_rng)
            else:
                fixed_mask = v < (v_min + fixed_tol * v_rng)
                
            fixed_tags = node_tags[fixed_mask].astype(int)
            
            if len(fixed_tags) == 0:
                logger.warning(f"[CalculiX-Struct] No nodes found for fixed boundary ({clamp_dir})! Simulation will likely fail.")
                # Fallback: Fix at least one node
                fallback_idx = np.argmax(v) if "MAX" in clamp_dir else np.argmin(v)
                fixed_tags = np.array([node_tags[fallback_idx].astype(int)])
                logger.info(f"[CalculiX-Struct] Fallback: Fixed node {fixed_tags[0]} at {clamp_dir}")

            logger.info(f"[CalculiX-Struct] Found {len(fixed_tags)} fixed nodes at {clamp_dir}")

            # Write INP
            with open(inp_file, 'w') as f:
                f.write("*HEADING\n")
                f.write(f"SimOps Structural Analysis: {mesh_file.name}\n")
                
                # Nodes
                f.write("*NODE\n")
                for tag, (x,y,z_c) in zip(node_tags, node_coords):
                    f.write(f"{int(tag)}, {x:.6f}, {y:.6f}, {z_c:.6f}\n")
                    
                # Elements
                if c3d4_elems:
                    f.write("*ELEMENT, TYPE=C3D4, ELSET=E_Vol\n")
                    for row in np.vstack(c3d4_elems):
                        f.write(f"{row[0]}, {row[1]}, {row[2]}, {row[3]}, {row[4]}\n")
                if c3d10_elems:
                    f.write("*ELEMENT, TYPE=C3D10, ELSET=E_Vol\n")
                    for row in np.vstack(c3d10_elems):
                         f.write(f"{row[0]}")
                         for n in row[1:]: f.write(f", {n}")
                         f.write("\n")
                         
                # Material (Elastic)
                # Units: MPa (N/mm^2) for E, mm for length, Tonne/mm^3 for Density?
                # Careful with consistency. 
                # Thermal adapter used: W/mmK, kg/mm^3.
                # Here we stick to SI consistent with mm:
                # Length: mm
                # Force: N
                # Stress: MPa (N/mm^2)
                # Density: Tonne/mm^3 (1000 kg/m^3 = 1e-9 tonne/mm^3)
                # Gravity: mm/s^2 (9.81 m/s^2 = 9810 mm/s^2)
                
                # Material Properties Database
                MATERIAL_PROPERTIES = {
                    "Al6061-T6": {
                        "youngs_modulus": 68900.0,  # MPa  
                        "poissons_ratio": 0.33,
                        "density": 2700.0,  # kg/m³
                    },
                    "Steel": {
                        "youngs_modulus": 210000.0,  # MPa
                        "poissons_ratio": 0.3,
                        "density": 7850.0,  # kg/m³
                    }
                }
                
                # Get material from config
                material_name = config.get("material", "Steel")
                mat_props = MATERIAL_PROPERTIES.get(material_name, MATERIAL_PROPERTIES["Steel"])
                
                logger.info(f"[Material] Using: {material_name}, E={mat_props['youngs_modulus']} MPa")
                
                # Allow explicit overrides
                E = config.get("youngs_modulus", mat_props["youngs_modulus"])
                nu = config.get("poissons_ratio", mat_props["poissons_ratio"])
                rho = config.get("density", mat_props["density"])
                
                # Convert Density to consistent units (tonne/mm^3 = kg/m^3 * 1e-12)
                rho_scaled = rho * 1e-12
                
                f.write("*MATERIAL, NAME=Mat1\n")
                f.write("*ELASTIC\n")
                f.write(f"{E}, {nu}\n")
                f.write("*DENSITY\n")
                f.write(f"{rho_scaled:.5E}, 0.0\n")
                
                f.write("*SOLID SECTION, ELSET=E_Vol, MATERIAL=Mat1\n")
                
                # Fixed Boundary
                if len(fixed_tags) > 0:
                     f.write("*NSET, NSET=N_Fixed\n")
                     for i, tag in enumerate(fixed_tags):
                          if i>0 and i%10==0: f.write("\n")
                          if i%10!=0: f.write(", ")
                          f.write(str(tag))
                     f.write("\n")
                     
                     f.write("*BOUNDARY\n")
                     f.write("N_Fixed, 1, 3, 0.0\n") 
                     logger.info(f"[CalculiX-Struct] Applied Fixed Boundary on {len(fixed_tags)} nodes via {clamp_dir}")
                else:
                     logger.warning("[CalculiX-Struct] NO FIXED NODES FOUND. Part may be unstable.")
                     f.write("*NSET, NSET=N_Fixed\n")
                     f.write("1\n") # Dummy fallback
                     f.write("*BOUNDARY\n")
                     f.write("N_Fixed, 1, 3, 0.0\n")
                
                # --- Diagnostic Block: Mass & Load Check ---
                try:
                    total_vol = 0
                    # Tet4
                    for chunk in c3d4_elems:
                        for row in chunk:
                            # row[0] is tag, row[1:5] are node tags
                            indices = [np.searchsorted(node_tags, tag) for tag in row[1:5]]
                            nodes = [node_coords[i] for i in indices]
                            a, b, c, d = nodes[0], nodes[1], nodes[2], nodes[3]
                            vol = abs(np.dot(a-d, np.cross(b-d, c-d))) / 6.0
                            total_vol += vol
                    # Tet10 (Approx as Tet4 of corners)
                    for chunk in c3d10_elems:
                        for row in chunk:
                            # row[0] is tag, row[1:5] are corner node tags
                            indices = [np.searchsorted(node_tags, tag) for tag in row[1:5]]
                            nodes = [node_coords[i] for i in indices]
                            a, b, c, d = nodes[0], nodes[1], nodes[2], nodes[3]
                            vol = abs(np.dot(a-d, np.cross(b-d, c-d))) / 6.0
                            total_vol += vol
                    
                    total_mass_kg = total_vol * rho * 1e-9 # rho in kg/m^3 -> kg/mm^3
                    total_force_n_exp = total_mass_kg * 9.81 * abs(g_load)
                    logger.info(f"[Diagnostics] Est. Volume: {total_vol:.2f} mm^3")
                    logger.info(f"[Diagnostics] Est. Mass: {total_mass_kg:.6f} kg")
                    logger.info(f"[Diagnostics] Expected Gravity Load (1G): {total_force_n_exp:.4f} N")
                except Exception as diag_e:
                    logger.warning(f"[Diagnostics] Failed to estimate load: {diag_e}")

                
                # Step
                f.write("*STEP\n")
                f.write("*STATIC\n")
                
                # Loads
                g_load = config.get("gravity_load_g", 0.0)
                tip_load = config.get("tip_load", None)
                
                if tip_load:
                    # Apply Tip Load (Concentrated Force)
                    # Find Z-max nodes (assuming beam along Z)
                    tip_mask = z > (z_max - fixed_tol * z_rng)
                    tip_tags = node_tags[tip_mask].astype(int)
                    num_tip = len(tip_tags)
                    
                    if num_tip > 0:
                        # Calculate Centroid for Reference Node
                        centroid = np.mean(node_coords[tip_mask], axis=0)
                        ref_node_id = 999999
                        
                        # Write Ref Node
                        f.write(f"*NODE, NSET=N_Ref\n")
                        f.write(f"{ref_node_id}, {centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f}\n")
                        
                        f.write("*NSET, NSET=N_Tip\n")
                        for i, tag in enumerate(tip_tags):
                             if i>0 and i%10==0: f.write("\n")
                             if i%10!=0: f.write(", ")
                             f.write(str(tag))
                        f.write("\n")
                        
                        # ANTIGRAVITY FIX: Use Rigid Body to distribute load and avoid singularities
                        # User Request: "apply stresses to areas... otherwise it will go to infinity"
                        f.write("*RIGID BODY, NSET=N_Tip, REF NODE=999999\n")
                        
                        f.write("*CLOAD\n")
                        # Apply Total Load to Reference Node
                        if abs(tip_load[0]) > 1e-9: f.write(f"{ref_node_id}, 1, {tip_load[0]:.5E}\n")
                        if abs(tip_load[1]) > 1e-9: f.write(f"{ref_node_id}, 2, {tip_load[1]:.5E}\n")
                        if abs(tip_load[2]) > 1e-9: f.write(f"{ref_node_id}, 3, {tip_load[2]:.5E}\n")
                        
                        logger.info(f"[CalculiX-Struct] Applied Rigid Body Load: {tip_load}N on RefNode {ref_node_id} (linked to {num_tip} nodes)")
                    else:
                        logger.warning("[CalculiX-Struct] Tip Load requested but no tip nodes found. Check coordinates.")
                        
                elif abs(g_load) > 1e-6:
                    # Gravity in -Z (usually)
                    # Convert g_load (g) to acceleration (mm/s^2)
                    # 1g = 9810 mm/s^2
                    grav_acc = 9810.0 * abs(g_load)
                    sign = -1.0 if g_load > 0 else 1.0
                    f.write(f"*DLOAD\n")
                    f.write(f"E_Vol, GRAV, {grav_acc:.5E}, 0.0, 0.0, {sign}\n")
                    logger.info(f"[CalculiX-Struct] Applied Gravity: {g_load}g ({grav_acc} mm/s^2)")
                
                # Output
                # Use *NODE FILE for .frd output (binary/ascii)
                f.write("*NODE FILE\n")
                f.write("U, S, E, RF\n") # Displacement, Stress, Strain, Reaction Forces
                f.write("*END STEP\n")
                
            # ANTIGRAVITY FIX: Remap element node tags to 0-based indices for visualization
            # The all_elems_ret array contains [Tag, Node1, Node2...] where Nodes are 1-based Tags.
            # We need them to be indices into the node_coords array.
            
            remapping_elems = all_elems_ret[:, 1:].copy()
            
            # Efficient Remap using searchsorted
            # node_tags is verified sorted above.
            remapped_elems = remapping_elems.copy() # Initialization for safety
            if len(node_tags) > 0:
                 remapped_elems = np.searchsorted(node_tags, remapping_elems)
            

            
            return {
                'node_map': dict(zip(node_tags, node_coords)),
                'elements': remapped_elems
            }

        finally:
            gmsh.finalize()

    def _parse_frd(self, frd_file: Path, node_map: Dict, elements: np.ndarray) -> Dict:
        """
        Parses FRD for Displacement (U) and Stress (S).
        Computes Von Mises.
        """
        with open(frd_file, 'r') as f:
            lines = f.readlines()
            
        disp_map = {} # nid -> [ux, uy, uz]
        stress_map = {} # nid -> [sxx, syy, szz, sxy, syz, szx]
        strain_map = {} # nid -> [exx, eyy, ezz, exy, eyz, ezx]
        rf_map = {} # nid -> [fx, fy, fz]
        
        mode = None
        
        for line in lines:
            line_stripped = line.strip()
            if line_stripped.startswith("-4"):
                if "DISP" in line_stripped: mode = "DISP"
                elif "STRESS" in line_stripped: mode = "STRESS"
                elif "STRAIN" in line_stripped: mode = "STRAIN"
                elif "FORC" in line_stripped: mode = "FORC"
                else: mode = None
                continue
            
            if line_stripped.startswith("-3"):
                mode = None
                continue
                
            
            if mode and line_stripped.startswith("-1"):
                # Fixed width parsing (CalculiX FRD format: 1X, I2, I10, 6E12.5)
                # Chars 0-3: " -1"
                # Chars 3-13: NID (10 chars)
                # Chars 13-25: Val1 (12 chars)
                # ...
                try:
                    nid = int(line[3:13])
                    # ANTIGRAVITY FIX: Robust Parser for Windows CalculiX Binary
                    # 1. Slice line[13:] to completely exclude Node ID column (prevent ID-Value merge)
                    # 2. Use Regex to handle overlapping columns with 3-digit exponents (e.g. 1.0E+004-1.0E-003)
                    import re
                    vals_str = re.findall(r'[-+]?\d*\.\d+[Ee][-+]+\d{3}', line[13:])
                    
                    vals = []
                    try:
                        vals = [float(v) for v in vals_str]
                    except: pass
                    
                    if mode == "DISP":
                        # ux, uy, uz
                        if len(vals) >= 3:
                            disp_map[nid] = vals[:3]
                        
                    elif mode == "STRESS":
                        # sxx, syy, szz, sxy, syz, szx
                        if len(vals) >= 6:
                            stress_map[nid] = vals[:6]
                        
                    elif mode == "STRAIN":
                        # exx, eyy, ezz, exy, eyz, ezx
                        if len(vals) >= 6:
                            strain_map[nid] = vals[:6]
                        
                    elif mode == "FORC":
                        # fx, fy, fz
                        if len(vals) >= 3:
                            rf_map[nid] = vals[:3]

                except ValueError as e:
                    continue  # Skip malformed lines
                        
        # Reconstruct Arrays aligned with Node Map
        sorted_nids = sorted(node_map.keys())
        tag_to_idx = {tag: i for i, tag in enumerate(sorted_nids)}
        
        U_list = []
        S_list = []
        Coords_list = []
        
        for nid in sorted_nids:
            Coords_list.append(node_map[nid])
            U_list.append(disp_map.get(nid, [0.0, 0.0, 0.0]))
            S_list.append(stress_map.get(nid, [0.0]*6))
            
        U = np.array(U_list)
        S = np.array(S_list)
        Coords = np.array(Coords_list)
        
        # Convert elements: They are already 0-based indices from _generate_inp fix.
        # So we just ensure they are integer array.
        elements = np.array(elements, dtype=int)
        
        # Compute Von Mises
        # VM = sqrt(0.5 * [(sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2 + 6(sxy^2 + syz^2 + szx^2)])
        sxx, syy, szz = S[:,0], S[:,1], S[:,2]
        sxy, syz, szx = S[:,3], S[:,4], S[:,5]
        
        vm_sq = 0.5 * ((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2 + 6*(sxy**2 + syz**2 + szx**2))
        von_mises = np.sqrt(vm_sq)
        
        disp_mag = np.linalg.norm(U, axis=1)
        
        # FINAL PHYSICS CHECK: Print to stdout (Worker catches this)
        print(f"DEBUG: Max Disp = {np.max(disp_mag):.6e} mm")
        print(f"DEBUG: Max Stress = {np.max(von_mises):.6e} MPa")
        
        # Max Strain (Von Mises equivalent strain? or just Max Principal?)
        # Let's just do Max Component for now or VM Strain implies same formula as Stress
        E_list = []
        RF_list = []
        for nid in sorted_nids:
            E_list.append(strain_map.get(nid, [0.0]*6))
            RF_list.append(rf_map.get(nid, [0.0, 0.0, 0.0]))
            
        E = np.array(E_list)
        RF = np.array(RF_list)
        
        # Max Von Mises Strain
        exx, eyy, ezz = E[:,0], E[:,1], E[:,2]
        exy, eyz, ezx = E[:,3], E[:,4], E[:,5]
        # Engineering shear strain usually reported? CCX outputs Tensor Strain.
        # Tensor Strain VM:
        # eff = 2/3 * sqrt( 0.5 * [(exx-eyy)^2 + ... + 6*(exy^2...)] ) ?
        # CCX "TOSTRAIN" is effective strain.
        # We'll just take Max Abs Component for simplicity if VM is complex to verify without formula lookup
        # Actually, let's just stick to max(abs(exx)) or similar for overview.
        max_strain = np.max(np.abs(E)) if len(E)>0 else 0.0
        
        # Total Reaction Force (Sum of RF)
        # RF is usually non-zero only at fixed nodes.
        total_rf = np.sum(RF, axis=0) # [Fx, Fy, Fz] total
        logger.info(f"[CalculiX-Struct] Total Reaction Force: {total_rf} N")
        
        # Debug Logging for Parse Size

        
        return {
            'node_coords': Coords,
            'displacement': U,
            'displacement_magnitude': disp_mag,
            'stress': S,
            'debug_info': f"Nodes={len(Coords)}, Elems={len(elements)}",
            'strain': E,
            'reaction_forces': RF,
            'von_mises': von_mises,
            'elements': elements,
            # For compat
            'temperature': von_mises, 
            'max_disp': np.max(disp_mag) if len(disp_mag)>0 else 0.0,
            'max_stress': np.max(von_mises) if len(von_mises)>0 else 0.0,
            'max_strain': max_strain,
            'total_reaction_force': total_rf.tolist(),
            'reaction_force_z': total_rf[2]
        }
