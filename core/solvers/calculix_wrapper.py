"""
CalculiX Solver Wrapper
=======================

Wraps the CalculiX (ccx) solver for thermal analysis.
Converts Gmsh .msh meshes to Abaqus .inp format, generates input decks,
runs the solver, and parses the results.
"""

import os
import shutil
import subprocess
import tempfile
import time
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import gmsh
import re

class CalculiXWrapper:
    """
    Wrapper for CalculiX (ccx) solver.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        
    def _log(self, msg: str):
        if self.verbose:
            print(msg, flush=True)
            
    def is_available(self) -> bool:
        """Check if ccx is installed"""
        try:
            # Try 'ccx' (Linux/WSL) or 'ccx.exe' (Windows with PATH)
            cmd = ['ccx', '--version']
            # ccx doesn't always have --version, so just try running it without args
            # usually it prints help and exits with 0 or 1
            result = subprocess.run(['ccx'], capture_output=True, text=True, timeout=5)
            # If we get here, the binary exists
            return True
        except FileNotFoundError:
            return False
        except Exception:
            return False
            
    def solve_thermal(self, 
                     mesh_file: str, 
                     output_dir: str,
                     k_thermal: float | List[List[float]],
                     t_ambient: float,
                     t_source: float) -> Dict:
        """
        Run steady-state thermal analysis.
        
        Args:
            mesh_file: Path to Gmsh .msh file
            output_dir: Directory for results
            k_thermal: Thermal conductivity
            t_ambient: Ambient temperature (K)
            t_source: Heat source temperature (K)
            
        Returns:
            Dict containing results
        """
        self._log(f"[CalculiX] Starting thermal analysis...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create unique work directory to avoid collisions
        work_dir = Path(tempfile.mkdtemp(prefix="ccx_run_"))
        job_name = "job"
        
        try:
            # 1. Convert Mesh and Extract Geometrical Sets
            self._log("[CalculiX] Converting mesh to Abaqus format...")
            inp_mesh_file = work_dir / f"{job_name}.msh" # Using .msh extension for consistency in internal naming, actual content is Abaqus
            
            # We use Gmsh to convert .msh to .inp (Abaqus)
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.open(mesh_file)
            
            # Extract bounds for BCs
            bbox = gmsh.model.getBoundingBox(-1, -1)
            z_min, z_max = bbox[2], bbox[5]
            z_range = z_max - z_min
            
            # Get all nodes to identify those for BCs
            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            node_coords = node_coords.reshape(-1, 3)
            
            # Identify sets manually
            hot_nodes = []
            cold_nodes = []
            
            for tag, (x, y, z) in zip(node_tags, node_coords):
                if z > z_max - 0.1 * z_range:
                    hot_nodes.append(tag)
                elif z < z_min + 0.1 * z_range:
                    cold_nodes.append(tag)
                    
            # Export mesh to Abaqus format
            abaqus_file = work_dir / "mesh.inp"
            gmsh.write(str(abaqus_file))
            gmsh.finalize()
            
            # 2. Prepare Input Deck
            self._log("[CalculiX] Generating input deck...")
            
            # Read the generated mesh file
            mesh_content = abaqus_file.read_text()
            
            # Create Node Sets for BCs
            nset_hot = "*NSET, NSET=Nhot\n" + self._format_list(hot_nodes)
            nset_cold = "*NSET, NSET=Ncold\n" + self._format_list(cold_nodes)
            
            # Create a set for all nodes (for output)
            nset_all = "*NSET, NSET=Nall\n" + self._format_list(list(node_tags))

            # Patch mesh content to ensure ELSET=Volume exists
            # This is critical because Gmsh might give physical group names or just Eall
            mesh_lines = mesh_content.splitlines()
            new_mesh_lines = []
            for line in mesh_lines:
                if line.lstrip().upper().startswith("*ELEMENT"):
                    # Force add ELSET=Volume if not present
                    if "ELSET" not in line.upper():
                        line = line.strip() + ", ELSET=Volume"
                    else:
                        # normalize elset name
                        line = re.sub(r'ELSET=[\w\d_]+', 'ELSET=Volume', line, flags=re.IGNORECASE)
                        if "ELSET=Volume" not in line: 
                             line += ", ELSET=Volume"
                new_mesh_lines.append(line)
            
            (work_dir / "mesh.inp").write_text("\n".join(new_mesh_lines))
            
            # Input deck with *NODE PRINT (for .dat output which is easier to parse)
            
            # Handle Material Properties (Scalar or Tabular)
            mat_section = "*MATERIAL, NAME=STEEL\n*CONDUCTIVITY\n"
            
            if isinstance(k_thermal, list):
                # Tabular: [ [k1, T1], [k2, T2] ]
                for row in k_thermal:
                    mat_section += f"{row[0]}, {row[1]}\n"
            else:
                # Scalar
                mat_section += f"{k_thermal}\n"

            deck_content = f"""*HEADING
SimOps Thermal Analysis
*INCLUDE, INPUT=mesh.inp
{nset_hot}
{nset_cold}
{nset_all}
{mat_section}
*SOLID SECTION, ELSET=Volume, MATERIAL=STEEL
*STEP
*HEAT TRANSFER, STEADY STATE
*BOUNDARY
Nhot, 11, 11, {t_source}
Ncold, 11, 11, {t_ambient}
*NODE PRINT, NSET=Nall
NT
*END STEP
"""
            (work_dir / f"{job_name}.inp").write_text(deck_content)
            
            # 3. Run CalculiX
            self._log(f"[CalculiX] Running solver (ccx {job_name})...")
            start_time = time.time()
            
            process = subprocess.run(
                ['ccx', job_name],
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                env={**os.environ, 'OMP_NUM_THREADS': '1'}
            )
            
            runtime = time.time() - start_time
            
            dat_file = work_dir / f"{job_name}.dat"
            
            # CCX might return non-zero but still produce results (warnings etc)
            if not dat_file.exists():
                 self._log(f"[!] CalculiX failed.")
                 self._log(f"Stdout: {process.stdout}")
                 self._log(f"Stderr: {process.stderr}")
                 raise RuntimeError("CalculiX solver failed (no output file)")
                 
            self._log(f"[CalculiX] Solver finished in {runtime:.2f}s")
            
            # 4. Parse Results (.dat file)
            temps = self._parse_dat_temperatures(dat_file)
            
            # Reconstruct full temperature array matching original Gmsh node ordering
            temperature_array = np.zeros(len(node_tags))
            tag_to_index = {tag: i for i, tag in enumerate(node_tags)}
            
            for node_id, temp in temps.items():
                if node_id in tag_to_index:
                    temperature_array[tag_to_index[node_id]] = temp
            
            results = {
                'temperature': temperature_array,
                'min_temp': float(np.min(temperature_array)) if len(temperature_array) > 0 else 0.0,
                'max_temp': float(np.max(temperature_array)) if len(temperature_array) > 0 else 0.0,
                'solve_time': runtime,
            }
            
            return results

        except Exception as e:
            self._log(f"[CalculiX] Error: {e}")
            import traceback
            traceback.print_exc()
            raise e
        finally:
             if not self.verbose: 
                try:
                    shutil.rmtree(work_dir)
                except:
                    pass

    def _format_list(self, items: List[int], chunk_size: int = 10) -> str:
        """Format list of integers for Abaqus input (comma separated, max 16 per line)"""
        lines = []
        for i in range(0, len(items), chunk_size):
            chunk = items[i:i+chunk_size]
            lines.append(", ".join(map(str, chunk)))
        return "\n".join(lines)
    
    def _parse_dat_temperatures(self, dat_file: Path) -> Dict[int, float]:
        """
        Parse .dat file for nodal temperatures.
        Expects *NODE PRINT, NSET=Nall
        """
        temps = {}
        with open(dat_file, 'r') as f:
            lines = f.readlines()
            
        reading = False
        for line in lines:
            line = line.strip()
            if "NODE" in line and "NT11" in line:
                reading = True
                continue
            
            if reading:
                # Stop on empty line or next header or page break
                if not line or "MAXIMUM" in line or "MINIMUM" in line or "NODE" in line:
                    if "NODE" in line: # New block starting?
                         # Usually just one block if steady state.
                         pass
                    else:
                        reading = False
                        break # Done reading current block
                
                # Parse: NodeID  Value
                parts = line.split()
                try:
                    # Ignore lines that don't start with a number
                    if not parts[0].isdigit():
                        continue
                        
                    node_id = int(parts[0])
                    val = float(parts[1])
                    temps[node_id] = val
                except:
                    pass
                    
        return temps
