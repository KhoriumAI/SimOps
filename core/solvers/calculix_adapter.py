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
        
        # 1. Convert Mesh and Generate INP
        stats = self._generate_inp(mesh_file, inp_file, config)
        
        # 2. Run CalculiX
        logger.info(f"[CalculiX] executing {self.ccx_binary} {job_name}...")
        start_time = time.time()
        
        try:
            # ccx expects jobname without extension
            # it reads jobname.inp and writes jobname.frd
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

    def _generate_inp(self, mesh_file: Path, inp_file: Path, config: Dict[str, Any]) -> Dict:
        """
        Reads Gmsh file and writes CalculiX input deck.
        Returns dict with node_map (internal ID -> coords) for parsing later.
        """
        gmsh.initialize()
        gmsh.open(str(mesh_file))
        
        try:
            # Nodes
            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            node_coords = node_coords.reshape(-1, 3)
            
            # CalculiX requires 1-based indexing sequence usually, but handles arbitrary tags if specified.
            # We will just write *NODE using Gmsh tags directly.
            
            # Elements (Tet4 = C3D4, Tet10 = C3D10)
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
            
            # Combine all for parsing/viz return (Warning: mixed types might break simple array return)
            # For result parsing we just need nodes usually? 
            # Actually run() returns 'elements' for VTK visualization.
            # VTK visualization of Tet10 usually falls back to Tet4 or requires special handling.
            # Simops currently expects Linear Tets for simple VTK writer.
            # We will return ALL elements, but VTK writer might need update if we pass 10 nodes.
            # For now, let's stack them.
            if c3d4_elems:
                all_elems = np.vstack(c3d4_elems)
            elif c3d10_elems:
                all_elems = np.vstack(c3d10_elems)
            else:
                all_elems = np.vstack(c3d4_elems + c3d10_elems)
            
            # Identify Boundary Nodes
            # Heuristic for MVP: Top 10% Z = Hot, Bottom 10% Z = Cold
            z = node_coords[:, 2]
            z_min, z_max = np.min(z), np.max(z)
            z_rng = z_max - z_min
            
            hot_mask = z > (z_max - 0.01 * z_rng)
            cold_mask = z < (z_min + 0.01 * z_rng)
            
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
                    
                # Elements
                if c3d4_elems:
                    f.write("*ELEMENT, TYPE=C3D4, ELSET=Eall\n")
                    for row in np.vstack(c3d4_elems):
                        f.write(f"{row[0]}, {row[1]}, {row[2]}, {row[3]}, {row[4]}\n")
                        
                if c3d10_elems:
                    f.write("*ELEMENT, TYPE=C3D10, ELSET=Eall\n")
                    for row in np.vstack(c3d10_elems):
                         # ID, 1-10
                         f.write(f"{row[0]}")
                         for n in row[1:]:
                             f.write(f", {n}")
                         f.write("\n")

                    
                # Material (Aluminum)
                k = config.get("thermal_conductivity", 150.0)
                f.write("*MATERIAL, NAME=ALUMINUM\n")
                f.write("*CONDUCTIVITY\n")
                f.write(f"{k}\n")   # Configurable K
                f.write("*SOLID SECTION, ELSET=Eall, MATERIAL=ALUMINUM\n")
                
                # Step
                f.write("*STEP\n")
                f.write("*HEAT TRANSFER, STEADY STATE\n")
                
                # BCs
                # Get temps from config or defaults
                t_hot = config.get("heat_source_temperature", 800.0)
                t_cold = config.get("ambient_temperature", 300.0)
                
                f.write("*BOUNDARY\n")
                # Temp usually DOF 11 in CalculiX heat transfer
                for node in hot_tags:
                    f.write(f"{node}, 11, 11, {t_hot}\n")
                for node in cold_tags:
                    f.write(f"{node}, 11, 11, {t_cold}\n")
                    
                # Output
                f.write("*NODE FILE\n")
                f.write("NT\n") # Nodal Temperature
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
        
        # Current logic: Minimal parsing to get the scalar field
        # FRD format is block based.
        # We look for '100CL' (Nodal values) block or similar.
        # Simplified parser for MVP.
        
        reading_temps = False
        
        with open(frd_file, 'r') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            # Check for Temperature block
            # -4 is Nodal Results
            # NT = Nodal Temperature, NDTEMP = Nodal Temperature (alt), P = Pressure (sometimes used for temperature in some contexts)
            if line.startswith(" -4") and ("NT" in line or "NDTEMP" in line or "P" in line):
                reading_temps = True
                continue
            
            if reading_temps:
                if line.startswith(" -3"): # End of block
                    break
                
                # Format: NodeID, Value
                # -1 node_id value (sometimes formatting varies)
                parts = line.split()
                if len(parts) < 2: continue
                
                try:
                    # FRD formatting is fixed width usually, but split works often
                    # -1 123 3.000E+02
                    if parts[0] == '-1':
                        nid = int(parts[1])
                        val = float(parts[2])
                        temperatures[nid] = val
                except:
                    pass
                    
        if not temperatures:
            # Fallback for binary or different format?
            # CCX default is usually ASCII.
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
        
        return {
            'temperature': T_array,
            'node_coords': parsed_coords,
            'min_temp': float(np.min(T_array)) if len(T_array)>0 else 0.0,
            'max_temp': float(np.max(T_array)) if len(T_array)>0 else 0.0,
            # Pass elements? If needed for VTK. We have to re-read or cache them.
            # For now, let result parsing handle that if needed, or re-read mesh.
            # The worker expects 'elements' for VTK export.
            'elements': elements 
        }
