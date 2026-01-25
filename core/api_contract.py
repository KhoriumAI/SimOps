"""
API Contract: Frontend-Backend Communication Schema
====================================================

Defines the contract between GUI (workers.py) and subprocess (mesh_worker_subprocess.py).
This module contains ONLY data structures - no logic, no imports of heavy libraries.

Usage:
    from core.api_contract import MeshJobRequest, MeshJobResponse
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import json


class MeshStrategy(Enum):
    """Available meshing strategies."""
    EXHAUSTIVE = "Exhaustive (Parallel Race)"
    GPU_DELAUNAY = "Tetrahedral (GPU Delaunay)"
    HEX_DOMINANT = "Hex Dominant"
    HEX_OPENFOAM = "Hex Dominant Testing"
    POLYHEDRAL = "Polyhedral (Dual)"
    DELAUNAY = "Tetrahedral (Delaunay)"


class JobStatus(Enum):
    """Mesh job status codes."""
    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"
    IN_PROGRESS = "in_progress"


@dataclass
class MeshJobRequest:
    """
    Contract for mesh generation requests.
    Sent from GUI to subprocess via JSON config file.
    """
    cad_file: str
    mesh_strategy: str = "Exhaustive (Parallel Race)"
    target_elements: int = 50000
    max_size_mm: Optional[float] = None
    min_size_mm: Optional[float] = None
    element_order: int = 1  # 1=Tet4, 2=Tet10
    fast_mode: bool = False
    defer_quality: bool = False
    ansys_mode: str = "None"
    score_threshold: float = 25.0
    strategy_order: Optional[List[str]] = None
    painted_regions: Optional[Dict] = None
    worker_count: Optional[int] = None
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'MeshJobRequest':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class QualityMetrics:
    """Mesh quality statistics."""
    sicn_min: float = 0.0
    sicn_avg: float = 0.0
    sicn_max: float = 1.0
    sicn_10_percentile: float = 0.0
    gamma_min: float = 0.0
    gamma_avg: float = 0.0
    gamma_max: float = 1.0


@dataclass
class MeshJobResponse:
    """
    Contract for mesh generation responses.
    Returned from subprocess to GUI as final JSON line.
    """
    success: bool
    output_file: Optional[str] = None
    strategy: str = ""
    message: str = ""
    total_elements: int = 0
    total_nodes: int = 0
    quality_metrics: Optional[Dict[str, float]] = None
    per_element_quality: Optional[Dict[int, float]] = None
    per_element_gamma: Optional[Dict[int, float]] = None
    per_element_skewness: Optional[Dict[int, float]] = None
    per_element_aspect_ratio: Optional[Dict[int, float]] = None
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    deferred: bool = False
    visualization_mode: Optional[str] = None  # "polyhedral" for special handling
    
    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d)
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'MeshJobResponse':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def failure(cls, error: str) -> 'MeshJobResponse':
        """Factory for failure responses."""
        return cls(success=False, error=error)
