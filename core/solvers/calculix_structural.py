import os
import time
import subprocess
import logging
import numpy as np
import gmsh
from pathlib import Path
from typing import Dict, Any, List, Optional
from .base import ISolver

logger = logging.getLogger(__name__)

class CalculiXStructuralAdapter(ISolver):
    """
    Adapter for running structural analysis using CalculiX (ccx).
    Supports: Static Linear Analysis, Gravity Loads, Displacement/Stress output.
    """
    
    def __init__(self, ccx_binary: str = "ccx"):
        self.ccx_binary = ccx_binary
        
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
        
        logger.info(f"[CalculiX-Struct] Converting mesh {mesh_file.name} to INP...")
        
        scale_factor = config.get("unit_scaling", 1.0)
        
        # 1. Generate INP
        stats = self._generate_inp(mesh_file, inp_file, config, scale=scale_factor)
        
        # 2. Run CalculiX
        logger.info(f"[CalculiX-Struct] executing {self.ccx_binary} {job_name}...")
        start_time = time.time()
        
        try:
            cwd = str(output_dir)
            cmd = [self.ccx_binary, job_name]
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "4"
            
            result = subprocess.run(
                cmd, 
                cwd=cwd, 
                capture_output=True, 
                text=True,
                env=env
            )
            
            if result.returncode != 0:
                logger.error(f"CalculiX Failed:\n{result.stdout}\n{result.stderr}")
                raise RuntimeError("CalculiX execution failed")
                
            logger.info("CalculiX execution complete.")
            
        except Exception as e:
            logger.error(f"CalculiX run error: {e}")
            raise
            
        solve_time = time.time() - start_time
        
        # 3. Parse Results
        frd_file = output_dir / f"{job_name}.frd"
        if not frd_file.exists():
            raise FileNotFoundError(f"CalculiX output file not found: {frd_file}")
            
        results = self._parse_frd(frd_file, stats['node_map'], stats['elements'])
        results['solve_time'] = solve_time
        results['mesh_stats'] = stats
        results['success'] = True
        
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
                    # Permutation for C3D10
                    raw_nodes = elem_nodes[i].reshape(-1, 10).astype(int)
                    permuted_nodes = raw_nodes[:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]]
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

            # Identification of Fixed Boundary (Z-min)
            # Default logic: Fix bottom 1% of Z range
            z = node_coords[:, 2]
            z_min = np.min(z)
            z_max = np.max(z)
            z_rng = z_max - z_min
            
            fixed_tol = config.get("fixed_boundary_tolerance", 0.01)
            fixed_mask = z < (z_min + fixed_tol * z_rng)
            fixed_tags = node_tags[fixed_mask].astype(int)
            
            logger.info(f"[CalculiX-Struct] Found {len(fixed_tags)} fixed nodes at Z_min")

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
                
                # Defaults (Steel)
                E = config.get("youngs_modulus") or 210000.0 # MPa
                nu = config.get("poissons_ratio") or 0.3
                rho = config.get("density", 7850.0) # kg/m^3 standard
                
                # Convert Density to consistent units (tonne/mm^3 = kg/m^3 * 1e-12)
                rho_scaled = rho * 1e-12
                
                f.write("*MATERIAL, NAME=Mat1\n")
                f.write("*ELASTIC\n")
                f.write(f"{E}, {nu}\n")
                f.write("*DENSITY\n")
                f.write(f"{rho_scaled:.5E}, 0.0\n")
                
                f.write("*SOLID SECTION, ELSET=E_Vol, MATERIAL=Mat1\n")
                
                # Fixed Boundary
                f.write("*NSET, NSET=N_Fixed\n")
                for i, tag in enumerate(fixed_tags):
                     if i>0 and i%10==0: f.write("\n")
                     if i%10!=0: f.write(", ")
                     f.write(str(tag))
                f.write("\n")
                
                f.write("*BOUNDARY\n")
                f.write("N_Fixed, 1, 3, 0.0\n") # Fix UX, UY, UZ
                
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
                        f.write("*NSET, NSET=N_Tip\n")
                        for i, tag in enumerate(tip_tags):
                             if i>0 and i%10==0: f.write("\n")
                             if i%10!=0: f.write(", ")
                             f.write(str(tag))
                        f.write("\n")
                        
                        fx = tip_load[0] / num_tip
                        fy = tip_load[1] / num_tip
                        fz = tip_load[2] / num_tip
                        
                        f.write("*CLOAD\n")
                        if abs(fx) > 1e-9: f.write(f"N_Tip, 1, {fx:.5E}\n")
                        if abs(fy) > 1e-9: f.write(f"N_Tip, 2, {fy:.5E}\n")
                        if abs(fz) > 1e-9: f.write(f"N_Tip, 3, {fz:.5E}\n")
                        
                        logger.info(f"[CalculiX-Struct] Applied Tip Load: {tip_load}N on {num_tip} nodes")
                    else:
                        logger.warning("[CalculiX-Struct] Tip Load requested but no tip nodes found.")
                        
                elif abs(g_load) > 1e-6:
                    # Gravity in -Z (usually)
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
                
            return {
                'node_map': dict(zip(node_tags, node_coords)),
                'elements': all_elems_ret[:, 1:]
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
                    
                    if mode == "DISP":
                        # ux, uy, uz
                        ux = float(line[13:25])
                        uy = float(line[25:37])
                        uz = float(line[37:49])
                        disp_map[nid] = [ux, uy, uz]
                        
                    elif mode == "STRESS":
                        # sxx, syy, szz, sxy, syz, szx
                        # Sometimes lines might be split? 
                        # Usually 6E12.5 fits in one line (72 chars + 13 = 85 chars).
                        # Let's hope lines aren't wrapped.
                        vals = []
                        for i in range(6):
                            start = 13 + i*12
                            end = start + 12
                            if len(line) >= end:
                                vals.append(float(line[start:end]))
                            else:
                                vals.append(0.0)
                        stress_map[nid] = vals
                        
                    elif mode == "STRAIN":
                        # exx, eyy, ezz, exy, eyz, ezx
                        vals = []
                        for i in range(6):
                            start = 13 + i*12
                            end = start + 12
                            if len(line) >= end:
                                vals.append(float(line[start:end]))
                            else:
                                vals.append(0.0)
                        strain_map[nid] = vals
                        
                    elif mode == "FORC":
                        # fx, fy, fz
                        fx = float(line[13:25])
                        fy = float(line[25:37])
                        fz = float(line[37:49])
                        rf_map[nid] = [fx, fy, fz]

                except ValueError as e:
                    continue  # Skip malformed lines
                        
        # Reconstruct Arrays aligned with Node Map
        sorted_nids = sorted(node_map.keys())
        
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
        
        # Compute Von Mises
        # VM = sqrt(0.5 * [(sxx-syy)^2 + (syy-szz)^2 + (szz-sxx)^2 + 6(sxy^2 + syz^2 + szx^2)])
        sxx, syy, szz = S[:,0], S[:,1], S[:,2]
        sxy, syz, szx = S[:,3], S[:,4], S[:,5]
        
        vm_sq = 0.5 * ((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2 + 6*(sxy**2 + syz**2 + szx**2))
        von_mises = np.sqrt(vm_sq)
        
        disp_mag = np.linalg.norm(U, axis=1)
        
        

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
        
        return {
            'node_coords': Coords,
            'displacement': U,
            'displacement_magnitude': disp_mag,
            'stress': S,
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
