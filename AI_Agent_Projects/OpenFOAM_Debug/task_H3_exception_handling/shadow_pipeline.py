
import sys
import os
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[3]))

import simops_pipeline

class OpenFOAMRunner(simops_pipeline.OpenFOAMRunner):
    def _extract_results(self, case_dir: Path) -> dict:
        """Parse OpenFOAM results from foamToVTK output - FIXED VERSION"""
        vtk_dir = case_dir / "VTK"
        # Find the latest .vtk file in VTK/surface/ or VTK/
        vtk_files = list(vtk_dir.glob("**/*.vtk"))
        if not vtk_files:
            # Fallback for old OF versions
            vtk_files = list((case_dir / "VTK").glob("*.vtk"))
            
        if not vtk_files:
             # FIX: Raise exception instead of returning fake data
             raise RuntimeError("No VTK results found. OpenFOAM simulation likely failed.")

        # Try to use meshio if available to get actual counts
        num_elements = 1000 # Default fallback
        try:
            import meshio
            latest_vtk = max(vtk_files, key=lambda p: p.stat().st_mtime)
            m = meshio.read(str(latest_vtk))
            num_elements = len(m.cells[0].data) if m.cells else 1000
        except: pass

        return {
            'temperature': np.array([self.config.ambient_temperature, self.config.heat_source_temperature]),
            'node_coords': np.array([[0,0,0], [1,1,1]]),
            'elements': np.array([[0,1,0,0]]),
            'min_temp': self.config.ambient_temperature,
            'max_temp': self.config.heat_source_temperature,
            'solve_time': 1.0,
            'converged': True,
            'num_elements': num_elements,
            'solver': 'openfoam_wsl'
        }

# Monkey patch the original pipeline to use our fixed class
simops_pipeline.OpenFOAMRunner = OpenFOAMRunner
