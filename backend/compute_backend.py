"""
Compute Backend Abstraction Layer

Provides a pluggable interface for dispatching preview/mesh generation
to different compute providers (local, SSH tunnel, HTTP remote, Modal.com).

Usage:
    from compute_backend import get_preferred_backend
    
    backend = get_preferred_backend()
    result = backend.generate_preview("/path/to/model.step")
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, List
import time
import os
import subprocess
import sys
import json
import tempfile
from pathlib import Path


class ComputeBackend(ABC):
    """Abstract compute backend for preview/mesh generation"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is currently reachable"""
        pass
    
    @abstractmethod
    def generate_preview(self, cad_file_path: str, timeout: int = 120) -> Dict:
        """
        Generate preview mesh from a CAD file.
        
        Args:
            cad_file_path: Path to the input CAD file (STEP, etc.)
            timeout: Maximum time in seconds to wait for result
            
        Returns:
            Dict with keys:
                - vertices: List of floats [x1,y1,z1, x2,y2,z2, ...]
                - numVertices: Number of vertices
                - numTriangles: Number of triangles
                - isPreview: True
                - status: 'success' or 'error'
                - error: Error message (if status is 'error')
        """
        pass
    
    def benchmark(self, cad_file_path: str, iterations: int = 3) -> Dict:
        """
        Run benchmark on this backend.
        
        Args:
            cad_file_path: Path to test CAD file
            iterations: Number of iterations to average
            
        Returns:
            Dict with timing statistics
        """
        times = []
        errors = []
        
        for i in range(iterations):
            start = time.time()
            try:
                result = self.generate_preview(cad_file_path)
                elapsed = time.time() - start
                
                if "error" in result:
                    errors.append(result["error"])
                else:
                    times.append(elapsed)
                    
            except Exception as e:
                errors.append(str(e))
        
        if not times:
            return {
                "backend": self.name,
                "available": self.is_available(),
                "error": errors[0] if errors else "All iterations failed",
                "errors": errors
            }
        
        return {
            "backend": self.name,
            "available": True,
            "avg_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "times": times,
            "iterations": len(times),
            "failed_iterations": len(errors)
        }


class LocalGMSHBackend(ComputeBackend):
    """
    Local GMSH compute backend.
    Runs GMSH directly on the current machine.
    """
    
    @property
    def name(self) -> str:
        return "local_gmsh"
    
    def is_available(self) -> bool:
        """Check if GMSH is available"""
        try:
            import gmsh
            return True
        except ImportError:
            return False
    
    def generate_preview(self, cad_file_path: str, timeout: int = 120) -> Dict:
        """Generate preview using local GMSH"""
        try:
            import gmsh
        except ImportError:
            return {"error": "GMSH not installed", "status": "error"}
        
        try:
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)  # Suppress output
            gmsh.option.setNumber("General.Verbosity", 0)
            
            # Disable optimization for speed
            gmsh.option.setNumber("Mesh.Optimize", 0)
            gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)
            gmsh.option.setNumber("Mesh.Algorithm", 1)  # MeshAdapt
            gmsh.option.setNumber("Mesh.MaxRetries", 1)
            
            # Robust loading settings
            gmsh.option.setNumber("Geometry.OCCAutoFix", 0)
            gmsh.option.setNumber("Geometry.Tolerance", 1e-2)
            gmsh.option.setNumber("General.NumThreads", 1)  # Single thread for stability
            
            # Load CAD file
            gmsh.open(cad_file_path)
            gmsh.model.occ.synchronize()
            
            # Calculate mesh sizing based on bounding box
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
            diag = ((xmax-xmin)**2 + (ymax-ymin)**2 + (zmax-zmin)**2)**0.5
            gmsh.option.setNumber("Mesh.MeshSizeMin", diag / 100.0)
            gmsh.option.setNumber("Mesh.MeshSizeMax", diag / 20.0)
            
            # Generate surface mesh
            gmsh.model.mesh.generate(2)
            
            # Extract mesh data
            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            nodes = {int(tag): [node_coords[3*i], node_coords[3*i+1], node_coords[3*i+2]] 
                     for i, tag in enumerate(node_tags)}
            
            vertices = []
            elem_types, _, node_tags_list = gmsh.model.mesh.getElements(2)
            
            for etype, enodes in zip(elem_types, node_tags_list):
                if etype == 2:  # 3-node triangle
                    enodes_list = enodes.astype(int).tolist()
                    for i in range(0, len(enodes_list), 3):
                        n1, n2, n3 = enodes_list[i], enodes_list[i+1], enodes_list[i+2]
                        if n1 in nodes and n2 in nodes and n3 in nodes:
                            vertices.extend(nodes[n1] + nodes[n2] + nodes[n3])
            
            return {
                "vertices": vertices,
                "numVertices": len(vertices) // 3,
                "numTriangles": len(vertices) // 9,
                "isPreview": True,
                "status": "success"
            }
            
        except Exception as e:
            return {"error": str(e), "status": "error"}
        finally:
            try:
                gmsh.finalize()
            except:
                pass


class HTTPRemoteBackend(ComputeBackend):
    """
    HTTP-based remote compute backend.
    Sends CAD files to a remote HTTP endpoint for processing.
    Used for SSH tunnel to Threadripper and future Modal.com integration.
    """
    
    def __init__(self, endpoint_url: str = "http://localhost:8080", endpoint_path: str = "/mesh"):
        self.endpoint_url = endpoint_url.rstrip('/')
        self.endpoint_path = endpoint_path
        self._name = f"http_remote ({endpoint_url})"
    
    @property
    def name(self) -> str:
        return self._name
    
    def is_available(self) -> bool:
        """Check if remote endpoint is reachable"""
        try:
            import requests
            response = requests.get(
                f"{self.endpoint_url}{self.endpoint_path}",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
    
    def generate_preview(self, cad_file_path: str, timeout: int = 120) -> Dict:
        """Send CAD file to remote endpoint for preview generation"""
        try:
            import requests
        except ImportError:
            return {"error": "requests library not installed", "status": "error"}
        
        try:
            with open(cad_file_path, 'rb') as f:
                response = requests.post(
                    f"{self.endpoint_url}{self.endpoint_path}",
                    files={'file': (Path(cad_file_path).name, f, 'application/octet-stream')},
                    timeout=timeout
                )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": f"Remote returned status {response.status_code}: {response.text[:200]}",
                    "status": "error"
                }
                
        except Exception as e:
            return {"error": str(e), "status": "error"}


class SSHTunnelBackend(HTTPRemoteBackend):
    """
    SSH Tunnel backend - specialized HTTPRemoteBackend for Threadripper.
    Assumes SSH tunnel is already established mapping localhost:8080 -> Threadripper:8080
    """
    
    def __init__(self, local_port: int = 8080):
        super().__init__(
            endpoint_url=f"http://localhost:{local_port}",
            endpoint_path="/mesh"
        )
        self._name = f"ssh_tunnel (localhost:{local_port})"


# =============================================================================
# Backend Factory
# =============================================================================

def get_available_backends() -> List[ComputeBackend]:
    """Get list of all configured backends"""
    backends = []
    
    # SSH tunnel (Threadripper)
    ssh_port = int(os.environ.get('SSH_TUNNEL_PORT', '8080'))
    backends.append(SSHTunnelBackend(local_port=ssh_port))
    
    # Local GMSH
    backends.append(LocalGMSHBackend())
    
    # Custom remote endpoint
    remote_url = os.environ.get('REMOTE_COMPUTE_URL')
    if remote_url:
        backends.append(HTTPRemoteBackend(endpoint_url=remote_url))
    
    return backends


def get_preferred_backend(strategy: str = None) -> ComputeBackend:
    """
    Get the preferred compute backend based on configuration.
    
    Args:
        strategy: Override strategy. Options:
            - 'auto': Try SSH tunnel first, fallback to local (default)
            - 'local': Use local GMSH only
            - 'ssh_tunnel': Use SSH tunnel only (fail if unavailable)
            - 'remote_http': Use custom remote URL
            
    Returns:
        ComputeBackend instance
    """
    if strategy is None:
        strategy = os.environ.get('COMPUTE_BACKEND', 'auto')
    
    if strategy == 'local':
        return LocalGMSHBackend()
    
    elif strategy == 'ssh_tunnel':
        ssh_port = int(os.environ.get('SSH_TUNNEL_PORT', '8080'))
        return SSHTunnelBackend(local_port=ssh_port)
    
    elif strategy == 'remote_http':
        remote_url = os.environ.get('REMOTE_COMPUTE_URL', 'http://localhost:8080')
        return HTTPRemoteBackend(endpoint_url=remote_url)
    
    elif strategy == 'auto':
        # Try SSH tunnel first
        ssh_backend = SSHTunnelBackend()
        if ssh_backend.is_available():
            return ssh_backend
        
        # Fallback to local
        return LocalGMSHBackend()
    
    else:
        raise ValueError(f"Unknown compute backend strategy: {strategy}")


class FallbackBackend(ComputeBackend):
    """
    Fallback backend that tries multiple backends in order.
    Used for 'auto' strategy with actual fallback execution.
    """
    
    def __init__(self, backends: List[ComputeBackend]):
        self.backends = backends
    
    @property
    def name(self) -> str:
        return f"fallback ({', '.join(b.name for b in self.backends)})"
    
    def is_available(self) -> bool:
        return any(b.is_available() for b in self.backends)
    
    def generate_preview(self, cad_file_path: str, timeout: int = 120) -> Dict:
        """Try each backend in order until one succeeds"""
        errors = []
        
        for backend in self.backends:
            if not backend.is_available():
                errors.append(f"{backend.name}: not available")
                continue
            
            result = backend.generate_preview(cad_file_path, timeout)
            
            if "error" not in result:
                result["_used_backend"] = backend.name
                return result
            
            errors.append(f"{backend.name}: {result.get('error', 'unknown error')}")
        
        return {
            "error": "All backends failed",
            "details": errors,
            "status": "error"
        }
