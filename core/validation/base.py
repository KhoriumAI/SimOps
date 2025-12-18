import numpy as np
import gmsh
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class BenchmarkCase(ABC):
    """
    Abstract base class for analytical verification benchmarks.
    """
    
    def __init__(self, name: str, output_dir: Path):
        self.name = name
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    @abstractmethod
    def generate_geometry(self) -> Path:
        """
        Generate geometry file (STEP or GEO).
        Returns path to generated file.
        """
        pass
        
    @abstractmethod
    def get_config_overrides(self) -> Dict:
        """
        Return configuration overrides for the simulation 
        (e.g., specific BCs, conductivity).
        """
        pass
        
    @abstractmethod
    def get_analytical_solution(self, x: float, y: float, z: float) -> float:
        """
        Return the exact expected temperature at (x,y,z).
        """
        pass
        
    def verify(self, result: Dict, tolerance: float = 0.02) -> Dict:
        """
        Compare simulation result to analytical solution.
        
        Args:
            result: Dictionary from thermal solver containing 'node_coords' and 'temperature'
            tolerance: Maximum allowed relative error (0.02 = 2%)
            
        Returns:
            Dict with verification stats (RMS error, Max error, Status)
        """
        coords = result['node_coords']
        temps_sim = result['temperature']
        
        if len(coords) != len(temps_sim):
            raise ValueError("Size mismatch between coordinates and temperatures")
            
        temps_exact = []
        errors = []
        
        for i, (x, y, z) in enumerate(coords):
            t_exact = self.get_analytical_solution(x, y, z)
            t_sim = temps_sim[i]
            
            error = abs(t_sim - t_exact)
            
            # Avoid division by zero in relative error
            if abs(t_exact) > 1e-9:
                rel_error = error / abs(t_exact)
            else:
                rel_error = error
                
            temps_exact.append(t_exact)
            errors.append(rel_error)
            
        errors = np.array(errors)
        max_error = np.max(errors)
        rms_error = np.sqrt(np.mean(errors**2))
        
        passed = max_error <= tolerance
        
        status = "PASS" if passed else "FAIL"
        
        logger.info(f"[{self.name}] Verification Result: {status}")
        logger.info(f"  Max Relative Error: {max_error*100:.4f}% (Threshold: {tolerance*100}%)")
        logger.info(f"  RMS Error:          {rms_error*100:.4f}%")
        
        return {
            'case': self.name,
            'status': status,
            'max_relative_error': max_error,
            'rms_error': rms_error,
            'passed': passed
        }
