from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any

class ISolver(ABC):
    """Abstract interface for Physics Solvers"""
    
    @abstractmethod
    def run(self, mesh_file: Path, output_dir: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the solver on the given mesh.
        
        Args:
            mesh_file: Path to the mesh file (.msh)
            output_dir: Directory for solver artifacts/logs
            config: Simulation configuration (materials, BCs)
            
        Returns:
            Dict containing results metadata (e.g., min/max temp, file paths)
        """
        pass
