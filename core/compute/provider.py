"""
Compute Orchestration for SimOps
=================================
Unified abstraction for Local Docker and Cloud Modal execution.
Merged from Forge TASK_02_COMPUTE_PROVIDER.
"""

from enum import Enum
from typing import Dict, Any
from abc import ABC, abstractmethod
from core.schemas import SimConfig

class JobStatus(str, Enum):
    """Job execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class ComputeProvider(ABC):
    """Abstract base class for compute providers."""
    
    @abstractmethod
    def submit_job(self, config: SimConfig) -> str:
        """Submit a simulation job. Returns job_id."""
        pass
    
    @abstractmethod
    def get_status(self, job_id: str) -> JobStatus:
        """Get current job status."""
        pass
    
    @abstractmethod
    def get_results(self, job_id: str) -> Dict[str, Any]:
        """Retrieve results for completed job."""
        pass
