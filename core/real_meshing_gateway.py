"""
Real Meshing Gateway: Production Implementation
================================================

Wraps existing meshing functions behind the IMeshingGateway interface.
This file contains the IMPLEMENTATION - the interface is in meshing_gateway.py.
"""

from typing import List, Optional, Callable
from .api_contract import MeshJobRequest, MeshJobResponse, MeshStrategy
from .meshing_gateway import IMeshingGateway, register_gateway


class RealMeshingGateway(IMeshingGateway):
    """
    Production gateway that routes to actual meshing strategies.
    
    Wraps the existing generate_* functions from mesh_worker_subprocess.py
    without changing their internal logic.
    """
    
    def __init__(self):
        self._cancel_requested = False
        self._available_strategies = [
            MeshStrategy.EXHAUSTIVE,
            MeshStrategy.DELAUNAY,
        ]
        # Check for optional strategies
        try:
            from core.gpu_mesher import GPU_AVAILABLE
            if GPU_AVAILABLE:
                self._available_strategies.append(MeshStrategy.GPU_DELAUNAY)
        except ImportError:
            pass
    
    def generate_mesh(
        self, 
        request: MeshJobRequest,
        progress_callback: Optional[Callable[[str, int], None]] = None
    ) -> MeshJobResponse:
        """
        Route to appropriate meshing function based on strategy.
        
        AIRLOCK: All meshing goes through here. If this fails,
        check: (1) was request valid? (2) which strategy was selected?
        """
        self._cancel_requested = False
        
        # Log entry point for debugging
        print(f"[GATEWAY] Received request: strategy={request.mesh_strategy}", flush=True)
        
        try:
            # Convert request to quality_params dict (backward compatible)
            from dataclasses import asdict
            quality_params = asdict(request)
            quality_params.pop('cad_file', None)  # Remove, passed separately
            
            # Route based on strategy
            strategy = request.mesh_strategy
            
            if 'GPU Delaunay' in strategy:
                from apps.cli.mesh_worker_subprocess import generate_gpu_delaunay_mesh
                result = generate_gpu_delaunay_mesh(
                    request.cad_file, None, quality_params
                )
            elif 'Hex Dominant Testing' in strategy or 'Hex OpenFOAM' in strategy:
                from apps.cli.mesh_worker_subprocess import generate_openfoam_hex_wrapper
                result = generate_openfoam_hex_wrapper(
                    request.cad_file, None, quality_params
                )
            elif 'Hex Dominant' in strategy:
                from apps.cli.mesh_worker_subprocess import generate_hex_dominant_mesh
                result = generate_hex_dominant_mesh(
                    request.cad_file, None, quality_params
                )
            elif 'Polyhedral' in strategy:
                from apps.cli.mesh_worker_subprocess import generate_polyhedral_mesh
                result = generate_polyhedral_mesh(
                    request.cad_file, None, quality_params
                )
            else:
                # Default: exhaustive strategy
                from apps.cli.mesh_worker_subprocess import generate_mesh
                result = generate_mesh(
                    request.cad_file, None, quality_params
                )
            
            # Log exit point
            print(f"[GATEWAY] Completed: success={result.get('success')}", flush=True)
            
            # Wrap result in contract type
            return MeshJobResponse.from_dict(result)
            
        except Exception as e:
            print(f"[GATEWAY] Exception: {e}", flush=True)
            return MeshJobResponse.failure(str(e))
    
    def get_available_strategies(self) -> List[MeshStrategy]:
        return self._available_strategies
    
    def cancel(self) -> bool:
        self._cancel_requested = True
        return True


def init_real_gateway():
    """
    Initialize and register the real gateway.
    Call this at subprocess startup.
    """
    gateway = RealMeshingGateway()
    register_gateway(gateway)
    return gateway
