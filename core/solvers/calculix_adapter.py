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
             default_path = Path(r"C:\calculix\calculix_2.22_4win\ccx.exe")
             logger.info(f"[Debug] Checking default path: {default_path} (Exists: {default_path.exists()})")
             if default_path.exists():
                 self.ccx_binary = str(default_path)
                 logger.info(f"[CalculiX] Found binary at {self.ccx_binary}")
        else:
             logger.info(f"[Debug] Skipping default path check. os.name={os.name}, binary={self.ccx_binary}")
        
    def run(self, mesh_file: Path, output_dir: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run CalculiX thermal analysis.
        """
        job_name = mesh_file.stem.replace(".msh", "")
        inp_file = output_dir / f"{job_name}.inp"
        
        logger.info(f"[CalculiX] Converting mesh {mesh_file.name} to INP...")
        
        # Scale Factor (mm->m default=1.0 unless configured)
        scale_factor = config.get("unit_scaling", 1.0)
        
        # 1. Convert Mesh and Generate INP
        stats = self._generate_inp(mesh_file, inp_file, config, scale=scale_factor)
        
        # 2. Run CalculiX
        logger.info(f"[CalculiX] executing {self.ccx_binary} {job_name}...")
        start_time = time.time()
        
        try:
            # ccx expects jobname without extension
            cwd = str(output_dir)
            
            # Check if ccx exists
            try:
                subprocess.run([self.ccx_binary, "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except FileNotFoundError:
                raise RuntimeError(f"CalculiX binary '{self.ccx_binary}' not found in PATH")
                
            # Run
            cmd = [self.ccx_binary, job_name]
            # CCX uses OpenMP, set threads
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
                    # Gmsh 11: 4 corners, 6 edges (0-1, 1-2, 2-0, 0-3, 1-3, 2-3)
                    # CalculiX C3D10: Permutation required (Swap last two nodes usually)
                    # Permutation: [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]
                    raw_nodes = elem_nodes[i].reshape(-1, 10).astype(int)
                    permuted_nodes = raw_nodes[:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]]
                    
                    tags = elem_tags[i].astype(int)
                    chunk = np.column_stack((tags, permuted_nodes))
                    c3d10_elems.append(chunk)

            if not c3d4_elems and not c3d10_elems:
                raise ValueError("No Tetrahedral elements found in mesh")
            
            # Skin Elements (Tri3=2, Tri6=9) for Convection (*FILM)
            surf_types, surf_tags, surf_nodes = gmsh.model.mesh.getElements(dim=2)
            tri3_elems = []
            tri6_elems = []
            
            for i, etype in enumerate(surf_types):
                 if etype == 2: # Tri3
                     nodes = surf_nodes[i].reshape(-1, 3).astype(int)
                     tags = surf_tags[i].astype(int)
                     chunk = np.column_stack((tags, nodes))
                     tri3_elems.append(chunk)
                 elif etype == 9: # Tri6
                     nodes = surf_nodes[i].reshape(-1, 6).astype(int)
                     tags = surf_tags[i].astype(int)
                     chunk = np.column_stack((tags, nodes))
                     tri6_elems.append(chunk)
            
            # Combine all for parsing/viz return
            if c3d4_elems:
                all_elems = np.vstack(c3d4_elems)
            elif c3d10_elems:
                all_elems = np.vstack(c3d10_elems)
            else:
                all_elems = np.vstack(c3d4_elems + c3d10_elems)
            
            # Identify Boundary Nodes (Legacy/Heuristic Support for Fixed Temp)
            z = node_coords[:, 2]
            z_min, z_max = np.min(z), np.max(z)
            z_rng = z_max - z_min
            
            tol = config.get("bc_tolerance", 0.01)
            hot_mask = z > (z_max - tol * z_rng)
            cold_mask = z < (z_min + tol * z_rng)
            
            hot_tags = node_tags[hot_mask].astype(int)
            cold_tags = node_tags[cold_mask].astype(int)
            
            # Write INP
            with open(inp_file, 'w') as f:
                f.write("*HEADING\n")
                f.write(f"SimOps Thermal Analysis: {mesh_file.name}\n")
                
                # Nodes
                f.write("*NODE\n")
                for tag, (x,y,z) in zip(node_tags, node_coords):
                    f.write(f"{int(tag)}, {x:.6f}, {y:.6f}, {z:.6f}\n")
                    
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

                # Skin Elements (S3/S6)
                if tri3_elems:
                    f.write("*ELEMENT, TYPE=S3, ELSET=E_Skin\n")
                    for row in np.vstack(tri3_elems):
                        f.write(f"{row[0]}, {row[1]}, {row[2]}, {row[3]}\n")

                if tri6_elems:
                     f.write("*ELEMENT, TYPE=S6, ELSET=E_Skin\n")
                     for row in np.vstack(tri6_elems):
                         f.write(f"{row[0]}")
                         for n in row[1:]:
                             f.write(f", {n}")
                         f.write("\n")
                    
                # Materials & Sections
                k = config.get("thermal_conductivity", 150.0)
                rho = config.get("density", 2700.0)
                cp = config.get("specific_heat", 900.0)
                
                f.write("*MATERIAL, NAME=Mat1\n")
                f.write("*CONDUCTIVITY\n")
                f.write(f"{k}\n")
                f.write("*DENSITY\n")
                f.write(f"{rho}\n")
                f.write("*SPECIFIC HEAT\n")
                f.write(f"{cp}\n")
                
                f.write("*SOLID SECTION, ELSET=E_Vol, MATERIAL=Mat1\n")
                
                if tri3_elems or tri6_elems:
                     # Dummy Shell Section for Skin (Thickness=0.001) to support FILM
                     f.write("*SHELL SECTION, ELSET=E_Skin, MATERIAL=Mat1\n")
                     f.write("0.001\n")
                
                # Step
                f.write("*STEP\n")
                
                if config.get("transient", False):
                     # Transient Analysis
                     dt = config.get("time_step", 1.0)
                     t_end = config.get("duration", 10.0)
                     f.write("*HEAT TRANSFER, DIRECT\n")
                     f.write(f"{dt}, {t_end}\n")
                else:
                     # Steady State
                     f.write("*HEAT TRANSFER, STEADY STATE\n")
                
                # Boundary Conditions (Conduction)
                f.write("*BOUNDARY\n")
                # Legacy Support: Hot Top / Cold Bottom
                t_hot = config.get("heat_source_temperature", 800.0)
                t_cold = config.get("ambient_temperature", 300.0)
                
                for tag in hot_tags:
                    f.write(f"{tag}, 11, 11, {t_hot}\n")
                for tag in cold_tags:
                    f.write(f"{tag}, 11, 11, {t_cold}\n")
                    
                # Convection (*FILM)
                # Apply to E_Skin faces
                h = config.get("convection_coeff", 0.0)
                T_inf = config.get("ambient_temperature", 293.0) 
                
                if h > 0 and (tri3_elems or tri6_elems):
                     f.write("*FILM\n")
                     # Apply to Face 1 (Top) of Shells
                     f.write(f"E_Skin, F1, {T_inf}, {h}\n")

                # Output
                f.write("*NODE FILE\n")
                f.write("NT\n")
                f.write("*END STEP\n")
                
            return {
                'node_map': dict(zip(node_tags, node_coords)),
                'elements': all_elems[:, 1:] # Strip ID, keep nodes
            }
        finally:
            gmsh.finalize()
            
    def _parse_frd(self, frd_file: Path, node_map: Dict, elements: np.ndarray) -> Dict:
        """
        Parses .frd file for NT (Nodal Temperature) values.
        Returns dict compatible with SimulationResult.
        """
        temperatures = {}
        
        reading_temps = False
        
        with open(frd_file, 'r') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            # Check for Temperature block
            # -4 is Nodal Results
            # NT, NDTEMP, P
            if line.startswith(" -4") and ("NT" in line or "NDTEMP" in line or "P" in line):
                reading_temps = True
                continue
            
            if reading_temps:
                if line.startswith(" -3"): # End of block
                    break
                
                parts = line.split()
                if len(parts) < 2: continue
                
                try:
                    # -1 123 3.000E+02
                    if parts[0] == '-1':
                        nid = int(parts[1])
                        val = float(parts[2])
                        temperatures[nid] = val
                except:
                    pass
                    
        if not temperatures:
            logger.warning("Could not parse temperatures from FRD.")
            
        # Convert to arrays matching node_map order
        sorted_nids = sorted(node_map.keys())
        T_array = []
        parsed_coords = []
        
        for nid in sorted_nids:
            T_array.append(temperatures.get(nid, 300.0)) # Default if missing
            parsed_coords.append(node_map[nid])
            
        T_array = np.array(T_array)
        parsed_coords = np.array(parsed_coords)
        
        # Calculate stats
        t_min = float(np.min(T_array)) if len(T_array)>0 else 0.0
        t_max = float(np.max(T_array)) if len(T_array)>0 else 0.0
        
        return {
            'temperature': T_array,
            'node_coords': parsed_coords,
            'min_temp': t_min,
            'max_temp': t_max,
            'elements': elements 
        }
