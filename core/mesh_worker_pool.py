"""
Pre-Warmed Mesh Worker Pool
===========================

Implements a background worker pool that pre-loads all heavy modules
(gmsh, numpy, cupy) so mesh generation starts instantly when requested.

Usage:
    # In GUI initialization:
    from core.mesh_worker_pool import MeshWorkerPool
    pool = MeshWorkerPool()
    pool.start_warmup()  # Start pre-loading in background
    
    # When user clicks "Generate Mesh":
    result = pool.generate_mesh(cad_file, config)  # Returns instantly if warmed
"""

import subprocess
import sys
import json
import time
import threading
import queue
import os
from pathlib import Path
from typing import Dict, Optional, Callable, Any
import tempfile


class MeshWorkerPool:
    """
    Pre-warmed subprocess pool for instant mesh generation.
    
    The pool maintains a warm subprocess that has already loaded all
    heavy imports (gmsh, numpy, cupy), eliminating the 3-5s startup delay.
    """
    
    def __init__(self, num_workers: int = 1):
        """
        Initialize the worker pool.
        
        Args:
            num_workers: Number of warm workers to maintain (default 1)
        """
        self.num_workers = num_workers
        self.warm_workers = queue.Queue()
        self.is_warming = False
        self.warmup_thread = None
        self._lock = threading.Lock()
        
        # Get worker script path
        self.worker_script = Path(__file__).parent.parent / "apps" / "cli" / "mesh_worker_daemon.py"
        
    def start_warmup(self, callback: Optional[Callable[[str], None]] = None):
        """
        Start warming up workers in the background.
        
        This should be called during GUI initialization so workers
        are ready when the user clicks "Generate Mesh".
        
        Args:
            callback: Optional function to call with status updates
        """
        if self.is_warming:
            return
            
        self.is_warming = True
        self.warmup_thread = threading.Thread(
            target=self._warmup_workers,
            args=(callback,),
            daemon=True
        )
        self.warmup_thread.start()
        
    def _warmup_workers(self, callback: Optional[Callable[[str], None]] = None):
        """Background thread that pre-warms workers."""
        for i in range(self.num_workers):
            try:
                if callback:
                    callback(f"Pre-loading mesh engine ({i+1}/{self.num_workers})...")
                
                worker = WarmWorker(self.worker_script)
                worker.start()
                
                # Wait for worker to signal it's ready
                if worker.wait_for_ready(timeout=30):
                    self.warm_workers.put(worker)
                    if callback:
                        callback(f"Mesh engine ready ({i+1}/{self.num_workers})")
                else:
                    if callback:
                        callback(f"Worker {i+1} failed to warm up")
                    worker.stop()
                    
            except Exception as e:
                if callback:
                    callback(f"Warmup error: {e}")
                    
        self.is_warming = False
        
    def get_warm_worker(self, timeout: float = 0.1) -> Optional['WarmWorker']:
        """
        Get a pre-warmed worker if available.
        
        Args:
            timeout: How long to wait for a worker
            
        Returns:
            WarmWorker if available, None otherwise
        """
        try:
            return self.warm_workers.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def generate_mesh(
        self,
        cad_file: str,
        config: Dict[str, Any],
        output_callback: Optional[Callable[[str], None]] = None,
        use_warm_worker: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a mesh, using a warm worker if available.
        
        Args:
            cad_file: Path to CAD file
            config: Mesh configuration dict
            output_callback: Function to receive stdout lines
            use_warm_worker: Try to use a pre-warmed worker first
            
        Returns:
            Result dict with mesh info
        """
        # Try to get a warm worker
        worker = None
        if use_warm_worker:
            worker = self.get_warm_worker(timeout=0.1)
            
        if worker:
            # Use warm worker - should be near-instant
            if output_callback:
                output_callback("[WARM] Using pre-loaded mesh engine")
            result = worker.generate_mesh(cad_file, config, output_callback)
            
            # Return worker to pool after use (or replace with new one)
            self._return_or_replace_worker(worker)
            return result
        else:
            # Fall back to cold start
            if output_callback:
                output_callback("[COLD] Starting mesh engine (pre-load next time)...")
            return self._cold_generate(cad_file, config, output_callback)
            
    def _cold_generate(
        self,
        cad_file: str,
        config: Dict[str, Any],
        output_callback: Optional[Callable[[str], None]] = None
    ) -> Dict[str, Any]:
        """Cold-start mesh generation (traditional subprocess)."""
        from apps.cli import mesh_worker_subprocess
        
        # Write config to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            config_file = f.name
            
        try:
            worker_script = Path(__file__).parent.parent / "apps" / "cli" / "mesh_worker_subprocess.py"
            
            cmd = [
                sys.executable,
                str(worker_script),
                cad_file,
                "--config-file", config_file
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            output_lines = []
            for line in process.stdout:
                line = line.strip()
                output_lines.append(line)
                if output_callback:
                    output_callback(line)
                    
            process.wait()
            
            return {
                'success': process.returncode == 0,
                'output': output_lines
            }
            
        finally:
            os.unlink(config_file)
            
    def _return_or_replace_worker(self, worker: 'WarmWorker'):
        """Return a worker to the pool or spawn a replacement."""
        if worker.is_healthy():
            self.warm_workers.put(worker)
        else:
            worker.stop()
            # Start a new warmup in background
            threading.Thread(
                target=self._warmup_single_worker,
                daemon=True
            ).start()
            
    def _warmup_single_worker(self):
        """Warm up a single replacement worker."""
        try:
            worker = WarmWorker(self.worker_script)
            worker.start()
            if worker.wait_for_ready(timeout=30):
                self.warm_workers.put(worker)
        except Exception:
            pass
            
    def shutdown(self):
        """Stop all workers and cleanup."""
        while not self.warm_workers.empty():
            try:
                worker = self.warm_workers.get_nowait()
                worker.stop()
            except queue.Empty:
                break


class WarmWorker:
    """
    A pre-warmed subprocess that has loaded all heavy modules.
    
    The worker runs a daemon script that imports gmsh, numpy, cupy
    on startup and then waits for commands via stdin.
    """
    
    def __init__(self, worker_script: Path):
        self.worker_script = worker_script
        self.process = None
        self.is_ready = threading.Event()
        self.reader_thread = None
        self.output_queue = queue.Queue()
        
    def start(self):
        """Start the daemon process."""
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path(__file__).parent.parent)
        
        self.process = subprocess.Popen(
            [sys.executable, str(self.worker_script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env
        )
        
        # Start output reader thread
        self.reader_thread = threading.Thread(
            target=self._read_output,
            daemon=True
        )
        self.reader_thread.start()
        
    def _read_output(self):
        """Background thread to read process output."""
        for line in self.process.stdout:
            line = line.strip()
            self.output_queue.put(line)
            
            # Check for ready signal
            if line == "[READY]":
                self.is_ready.set()
                
    def wait_for_ready(self, timeout: float = 30) -> bool:
        """Wait for the worker to signal it's ready."""
        return self.is_ready.wait(timeout=timeout)
        
    def is_healthy(self) -> bool:
        """Check if the worker process is still alive."""
        return self.process is not None and self.process.poll() is None
        
    def generate_mesh(
        self,
        cad_file: str,
        config: Dict[str, Any],
        output_callback: Optional[Callable[[str], None]] = None
    ) -> Dict[str, Any]:
        """
        Send a mesh generation command to the warm worker.
        
        Args:
            cad_file: Path to CAD file
            config: Mesh configuration
            output_callback: Function to receive output lines
            
        Returns:
            Result dict
        """
        if not self.is_healthy():
            return {'success': False, 'error': 'Worker not healthy'}
            
        # Send command as JSON
        command = {
            'action': 'generate',
            'cad_file': cad_file,
            'config': config
        }
        
        try:
            self.process.stdin.write(json.dumps(command) + '\n')
            self.process.stdin.flush()
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
        # Read output until we get a result
        result = {'success': False, 'output': []}
        
        while True:
            try:
                line = self.output_queue.get(timeout=300)  # 5 min timeout
            except queue.Empty:
                result['error'] = 'Timeout waiting for mesh result'
                break
                
            if output_callback:
                output_callback(line)
                
            result['output'].append(line)
            
            # Check for completion markers
            if line.startswith('[RESULT]'):
                try:
                    result_json = json.loads(line[8:])
                    result.update(result_json)
                except json.JSONDecodeError:
                    pass
                break
            elif line.startswith('[ERROR]'):
                result['error'] = line[7:]
                break
                
        return result
        
    def stop(self):
        """Stop the daemon process."""
        if self.process:
            try:
                self.process.stdin.write('{"action": "shutdown"}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()


# Global pool instance for easy access
_global_pool = None


def get_pool() -> MeshWorkerPool:
    """Get or create the global mesh worker pool."""
    global _global_pool
    if _global_pool is None:
        _global_pool = MeshWorkerPool()
    return _global_pool


def warmup_mesh_engine(callback: Optional[Callable[[str], None]] = None):
    """
    Convenience function to start warming up the mesh engine.
    
    Call this during GUI initialization:
        warmup_mesh_engine(lambda msg: status_bar.showMessage(msg))
    """
    pool = get_pool()
    pool.start_warmup(callback)
