"""
Modal Client for Flask Backend
Provides async job spawning and result retrieval for Modal GPU workers.

Usage:
    from modal_client import modal_client
    
    call = modal_client.spawn_mesh_job(bucket, key, quality_params)
    result = modal_client.get_job_result(call.object_id)
"""
import modal
import os
import time
from typing import Dict, Any, Optional


class ModalClient:
    """
    Client for interacting with Modal serverless GPU workers.
    Provides methods to trigger meshing and preview jobs.
    
    Updated for Modal 1.3+ API (uses from_name instead of lookup).
    """
    
    def __init__(self):
        self._mesh_fn = None
        self._preview_fn = None
        self._app_name = None
        
    def _get_app_name(self):
        """Get app name from Flask config or environment"""
        if self._app_name is None:
            try:
                from flask import current_app
                self._app_name = current_app.config.get('MODAL_APP_NAME', 'khorium-production')
            except (RuntimeError, ImportError, ModuleNotFoundError):
                # Outside Flask context or Flask not installed, use environment or default
                self._app_name = os.environ.get('MODAL_APP_NAME', 'khorium-production')
        return self._app_name
            
    def _get_mesh_fn(self):
        """Get the meshing function reference (lazy initialization)"""
        if self._mesh_fn is None:
            app_name = self._get_app_name()
            fn_name = os.environ.get('MODAL_MESH_FUNCTION', 'generate_mesh')
            print(f"[ModalClient] Getting mesh function: {app_name}/{fn_name}")
            # Modal 1.3+ API: use from_name instead of lookup
            self._mesh_fn = modal.Function.from_name(app_name, fn_name)
        return self._mesh_fn

    def _get_preview_fn(self):
        """Get the preview function reference (lazy initialization)"""
        if self._preview_fn is None:
            app_name = self._get_app_name()
            fn_name = os.environ.get('MODAL_PREVIEW_FUNCTION', 'generate_preview_mesh')
            print(f"[ModalClient] Getting preview function: {app_name}/{fn_name}")
            # Modal 1.3+ API: use from_name instead of lookup
            self._preview_fn = modal.Function.from_name(app_name, fn_name)
        return self._preview_fn

    def spawn_mesh_job(self, bucket: str, key: str, quality_params: Optional[Dict] = None):
        """
        Spawn an asynchronous mesh generation job on Modal.
        Returns a Modal FunctionCall object which contains the object_id.
        """
        try:
            fn = self._get_mesh_fn()
            # .spawn() returns a modal.functions.FunctionCall (async)
            call = fn.spawn(bucket, key, quality_params)
            print(f"[ModalClient] Spawned mesh job: {call.object_id}")
            return call
        except Exception as e:
            print(f"[ModalClient ERROR] Failed to spawn mesh job: {e}")
            raise

    def spawn_preview_job(self, bucket: str, key: str):
        """
        Spawn an asynchronous preview generation job on Modal.
        Returns a Modal FunctionCall object.
        """
        try:
            fn = self._get_preview_fn()
            call = fn.spawn(bucket, key)
            print(f"[ModalClient] Spawned preview job: {call.object_id}")
            return call
        except Exception as e:
            print(f"[ModalClient ERROR] Failed to spawn preview job: {e}")
            raise

    def get_job_result(self, call_id: str, timeout: int = 600):
        """
        Wait for a Modal job to complete and return the result.
        This blocks for the duration of the timeout or until completion.
        """
        try:
            # Reconstruct the call object from id
            call = modal.functions.FunctionCall.from_id(call_id)
            # .get() blocks until result is ready
            return call.get(timeout=timeout)
        except TimeoutError:
            # Re-raise timeout so caller can handle polling
            raise
        except Exception as e:
            print(f"[ModalClient ERROR] Failed to get job result for {call_id}: {e}")
            return {"success": False, "message": f"Modal job failed or timed out: {e}"}
    
    def remote_mesh(self, bucket: str, key: str, quality_params: Optional[Dict] = None, timeout: int = 600):
        """
        Synchronous convenience method: spawn and wait for mesh result.
        """
        try:
            fn = self._get_mesh_fn()
            # .remote() blocks until complete
            return fn.remote(bucket, key, quality_params)
        except Exception as e:
            print(f"[ModalClient ERROR] Remote mesh failed: {e}")
            return {"success": False, "message": f"Modal remote call failed: {e}"}


# Global instance
modal_client = ModalClient()
