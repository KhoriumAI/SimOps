"""
Meshing Gateway: Single Entry Point for All Meshing Strategies
================================================================

Defines abstract interface that all meshing operations go through.
This module contains ONLY interface definitions - implementations are separate.

Usage:
    from core.meshing_gateway import IMeshingGateway, get_gateway
    
    gateway = get_gateway()
    result = gateway.generate_mesh(request)
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Callable
from .api_contract import MeshJobRequest, MeshJobResponse, MeshStrategy


class IMeshingGateway(ABC):
    """
    Abstract interface for mesh generation.
    
    All mesh generation requests go through this single gateway.
    Enables:
    - Consistent logging at boundary
    - Easy mocking for frontend development
    - Strategy swapping without changing callers
    """
    
    @abstractmethod
    def generate_mesh(
        self, 
        request: MeshJobRequest,
        progress_callback: Optional[Callable[[str, int], None]] = None
    ) -> MeshJobResponse:
        """
        Generate mesh using the strategy specified in request.
        
        Args:
            request: Validated mesh generation request
            progress_callback: Optional (message, percentage) callback
            
        Returns:
            MeshJobResponse with success/failure and mesh data
        """
        pass
    
    @abstractmethod
    def get_available_strategies(self) -> List[MeshStrategy]:
        """Return list of available meshing strategies."""
        pass
    
    @abstractmethod
    def cancel(self) -> bool:
        """
        Cancel running mesh generation.
        
        Returns:
            True if cancellation was requested, False if nothing running
        """
        pass


class MockMeshingGateway(IMeshingGateway):
    """
    Mock gateway for frontend testing without heavy backend dependencies.
    Returns dummy results instantly.
    """
    
    def generate_mesh(
        self, 
        request: MeshJobRequest,
        progress_callback: Optional[Callable[[str, int], None]] = None
    ) -> MeshJobResponse:
        if progress_callback:
            progress_callback("Mocking mesh generation...", 50)
            progress_callback("Done (mock)", 100)
        
        return MeshJobResponse(
            success=True,
            output_file="/mock/output.msh",
            strategy="mock",
            message="Mock mesh generated (no actual computation)",
            total_elements=1000,
            total_nodes=500,
            quality_metrics={
                "sicn_min": 0.3,
                "sicn_avg": 0.7,
                "sicn_max": 1.0
            }
        )
    
    def get_available_strategies(self) -> List[MeshStrategy]:
        return list(MeshStrategy)
    
    def cancel(self) -> bool:
        return True


# Gateway registry for dependency injection
_gateway_instance: Optional[IMeshingGateway] = None


def get_gateway() -> IMeshingGateway:
    """
    Get the current meshing gateway instance.
    
    Returns MockMeshingGateway if no real gateway registered.
    """
    global _gateway_instance
    if _gateway_instance is None:
        _gateway_instance = MockMeshingGateway()
    return _gateway_instance


def register_gateway(gateway: IMeshingGateway):
    """Register a real gateway implementation."""
    global _gateway_instance
    _gateway_instance = gateway
