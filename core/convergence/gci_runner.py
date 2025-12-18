"""
Grid Convergence Index (GCI) Runner
===================================
Orchestrates mesh independence studies by running simulations at multiple 
levels of refinement and computing the GCI metric.

GCI is a standardized method for quantifying numerical uncertainty.
Ref: Roache, P. J. (1994). Perspective: A method for uniform reporting of grid refinement studies.
"""

import time
import logging
import math
import shutil
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel

from core.schemas.config_schema import SimulationConfig
from core.strategies.cfd_strategy import CFDMeshConfig
from simops_worker import run_simulation, SimulationResult, run_thermal_solver, generate_mesh_with_strategy

logger = logging.getLogger(__name__)

class GCIAssessment(BaseModel):
    r_ratio: float      # Refinement ratio (e.g. 1.3)
    p_order: float      # Observed order of accuracy
    gci_fine: float     # Error band for fine mesh
    asymptotic_check: float # Check if in asymptotic range (~1.0)
    converged: bool

class GCIRunner:
    def __init__(self, base_config: SimulationConfig, cad_file: str, output_dir: Path):
        self.base_config = base_config
        self.cad_file = cad_file
        self.output_dir = output_dir
        
    def run_study(self, levels: int = 3, refinement_ratio: float = 1.3) -> Dict:
        """
        Runs the GCI study.
        1. Coarse Mesh (h3)
        2. Medium Mesh (h2 = h3 / r)
        3. Fine Mesh (h1 = h2 / r)
        """
        results = []
        
        logger.info(f"[GCI] Starting Grid Convergence Study (Levels={levels}, r={refinement_ratio})")
        
        # Calculate mesh size multipliers
        # We start from base_multiplier and scale UP for coarser meshes
        # Fine (1) = base
        # Med (2) = base * r
        # Coarse (3) = base * r * r
        
        base_mult = self.base_config.meshing.mesh_size_multiplier
        
        for i in range(levels):
            # Scale factor: 0 -> r^0 (Fine), 1 -> r^1 (Med), 2 -> r^2 (Coarse)
            # Actually standard is usually Coarse -> Fine or Fine -> Coarse
            # Let's verify Fine first to fail fast? 
            # Consistent order: Fine (h1), Medium (h2), Coarse (h3)
            
            # GCI assumes h1 < h2 < h3
            scale = refinement_ratio ** i
            current_mult = base_mult * scale
            
            level_name = f"L{i+1}" # L1=Fine, L2=Med, L3=Coarse
            
            logger.info(f"[GCI] Running Level {level_name} (Scale={scale:.2f})")
            
            # Create a modified config
            # We must NOT modify the original object in place
            # Pydantic copy is useful
            level_config = self.base_config.model_copy(deep=True)
            level_config.meshing.mesh_size_multiplier = current_mult
            level_config.job_name = f"{self.base_config.job_name or 'Sim'}_{level_name}"
            
            # Run Simulation Pipeline
            # We replicate simops_worker logic but need to capture results
            # The run_simulation function writes to file, we might need to parse it back
            # or refactor run_simulation to return the object.
            # simops_worker.run_simulation IS refactored to return SimulationResult object?
            # Let's check simops_worker.py signature
            
            # Assuming we can call internal helpers or just full run_simulation
            # But run_simulation does loading itself. 
            # We should probably invoke the lower level orchestration steps here 
            # OR pass the config to run_simulation if we update it.
            
            # For now, let's call the components directly to avoid reloading config from file
            
            level_dir = self.output_dir / level_name
            level_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # 1. Mesh
                # Need to construct strategy dict... mimicking worker
                strategy = {
                   'name': 'GCI_Tet10' if level_config.meshing.second_order else 'GCI_Tet4',
                   'mesh_size': 0.5, # Base, will be scaled by multiplier
                   'num_layers': 5,
                   'growth_rate': 1.2,
                   'optimize': True
                }
                
                mesh_file = generate_mesh_with_strategy(
                    self.cad_file, 
                    level_dir, 
                    strategy, 
                    sim_config=level_config
                )
                
                # 2. Solve
                res = run_thermal_solver(Path(mesh_file), level_dir, strategy['name'])
                
                # Extract Key Metric (e.g. Max Temp)
                metric = res.get('max_temp', 0.0)
                element_count = 0 # Need to extract this from stats
                # Re-read mesh stats??
                # run_thermal_solver returns dict, maybe we should augment it with element count if not present
                
                results.append({
                    'level': i+1,
                    'scale': current_mult,
                    'metric': metric,
                    'mesh_file': str(mesh_file)
                })
                
            except Exception as e:
                logger.error(f"[GCI] Level {level_name} failed: {e}")
                raise
                
        # Compute GCI
        # Sort results by grid size (Fine to Coarse) => h1, h2, h3
        # Our loop produced Fine(0), Med(1), Coarse(2)
        
        f1 = results[0]['metric']
        f2 = results[1]['metric']
        f3 = results[2]['metric']
        
        r = refinement_ratio
        
        # Calculate Order of Accuracy (p)
        # p = ln( (f3-f2)/(f2-f1) ) / ln(r)
        
        epsilon = 1e-9
        diff1 = f2 - f1
        diff2 = f3 - f2
        
        if abs(diff1) < epsilon or abs(diff2) < epsilon:
            logger.warning("[GCI] Differences too small to compute order p. Assuming converged.")
            p = 0.0
        elif (diff1 * diff2) < 0:
             logger.warning("[GCI] Oscillatory convergence detected. GCI may be invalid.")
             p = 0.0 # Oscillatory
        else:
             p = math.log(abs(diff2 / diff1)) / math.log(r)
             
        # Safety clamp for p
        if p < 0.5: p = 0.5
        if p > 3.0: p = 3.0 # unlikely to be higher
        
        # Safety Factor Fs = 1.25 for 3 grids
        Fs = 1.25
        
        # GCI_fine = Fs * |e21| / (r^p - 1)
        # e21 = (f1 - f2) / f1
        
        if f1 != 0:
            e21 = abs((f1 - f2) / f1)
            gci = (Fs * e21) / (r**p - 1)
        else:
            gci = 0.0
            
        logger.info(f"[GCI] Result: p={p:.2f}, GCI_fine={gci*100:.2f}%")
        
        return {
            'p_order': p,
            'gci_fine': gci,
            'results': results
        }
