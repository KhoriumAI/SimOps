#!/usr/bin/env python
"""
SimOps Hot Folder Watcher
=========================

Monitors an input directory for new STEP/IGES files and queues them
for simulation via Redis Queue (RQ).

This is the "seamless" interface - engineers just save files to a folder.

Usage:
    python watcher.py [--input /path/to/input] [--poll 5]
"""

import os
import sys
import time
import logging
import platform
from pathlib import Path
from datetime import datetime
from typing import Set, Optional

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent
from redis import Redis
from rq import Queue

# Try to import platform-specific locking
if platform.system() == 'Windows':
    import msvcrt
else:
    import fcntl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [WATCHER] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/logs/watcher.log') if os.path.exists('/logs') else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)


# Supported CAD and Mesh file extensions
SUPPORTED_EXTENSIONS = {'.step', '.stp', '.iges', '.igs', '.brep', '.msh', '.vtk', '.vtu', '.unv', '.inp'}


class PathHandler:
    """Cross-platform path handling for hot folder"""
    
    @staticmethod
    def normalize(path_str: str) -> Path:
        """Normalize path string to Path object, handling UNC and cleaning."""
        if not path_str:
            return Path('.')
            
        clean_path = os.path.normpath(path_str)
        return Path(clean_path)

    @staticmethod
    def is_network_path(path: Path) -> bool:
        """Check if path is likely a network resource."""
        s = str(path)
        return s.startswith(r'\\') or s.startswith('//')


class SimOpsEventHandler(FileSystemEventHandler):
    """
    Handles file system events for the hot folder.
    
    When a new CAD file appears, it:
    1. Waits for the file to finish writing (Atomic Locking)
    2. Validates it's a supported format
    3. Queues it for simulation
    """
    
    def __init__(self, queue: Queue, output_dir: Path):
        super().__init__()
        self.queue = queue
        self.output_dir = output_dir
        self.processing: Set[str] = set()  # Track files being processed
        self.processed: Set[str] = set()   # Track completed files
        
    def on_created(self, event: FileCreatedEvent):
        """Handle new file creation"""
        if event.is_directory:
            return
            
        file_path = PathHandler.normalize(event.src_path)
        self._process_file(file_path)
        
    def on_modified(self, event):
        """Handle file modification (catches copy completion)"""
        if event.is_directory:
            return
            
        file_path = PathHandler.normalize(event.src_path)
        
        if str(file_path) not in self.processing and str(file_path) not in self.processed:
            self._process_file(file_path)

    def on_deleted(self, event):
        """Handle file deletion - allow re-processing if added again"""
        if event.is_directory:
            return
            
        file_path = PathHandler.normalize(event.src_path)
        str_path = str(file_path)
        
        if str_path in self.processed:
            # logger.info(f"♻️  File removed, clearing cache: {file_path.name}")
            self.processed.discard(str_path)
            
        if str_path in self.processing:
            self.processing.discard(str_path)
    
    def _process_file(self, file_path: Path):
        """Process a new CAD file"""
        # Check extension
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return
            
        # Skip temp files
        if file_path.name.startswith('.') or file_path.name.startswith('~'):
            return

        # Skip already used files
        if file_path.name.startswith('USED_'):
            return
            
        str_path = str(file_path)
        
        # Already processing?
        if str_path in self.processing:
            return
            
        # Already done?
        if str_path in self.processed:
            return
            
        # -------------------------------------------------------------
        # TRACK 2: Sidecar Logic
        # If this is a .JSON file, ignore it. We only queue the geometry.
        # The worker will look for the JSON itself.
        # -------------------------------------------------------------
        if file_path.suffix.lower() == '.json':
            logger.info(f"ℹ️ Config file detected: {file_path.name} (Waiting for geometry...)")
            return
            
        # Wait for file to finish writing
        # logger.info(f"⏳ Detecting file stability: {file_path.name}")
        # if not self._wait_for_file_stable(file_path):
        #    logger.warning(f"File not stable or locked, skipping: {file_path.name}")
        #    return
             
        # Mark as processing
        self.processing.add(str_path)
        
        logger.info(f"📥 New CAD file detected: {file_path.name}")
        
        # -------------------------------------------------------------
        # 3. Flexible Sidecar Detection
        # -------------------------------------------------------------
        # Look for a config file.
        # Priority 1: Exact Match (file.step -> file.json)
        # Priority 2: Temporal Match (Any *.json modified within 15s)
        
        config_path = None
        
        # 1. Exact Match
        exact_sidecar = file_path.with_suffix('.json')
        if exact_sidecar.exists():
            logger.info(f"   + Config found (Exact): {exact_sidecar.name}")
            config_path = exact_sidecar
        else:
            # 2. Temporal Match
            # Scan directory for OTHER json files
            try:
                cad_mtime = file_path.stat().st_mtime
                candidates = []
                for f in file_path.parent.glob('*.json'):
                    if f.name.startswith('USED_'): continue # Skip used
                    if f == exact_sidecar: continue 
                    
                    try:
                        f_mtime = f.stat().st_mtime
                        delta = abs(f_mtime - cad_mtime)
                        if delta < 15.0: # 15 second window
                            candidates.append((delta, f))
                    except: pass
                
                if candidates:
                    # Sort by closest time
                    candidates.sort(key=lambda x: x[0])
                    best_match = candidates[0][1]
                    logger.info(f"   + Config found (Temporal +/- {candidates[0][0]:.1f}s): {best_match.name}")
                    config_path = best_match
            except Exception as e:
                logger.warning(f"Error during temporal scan: {e}")

        # -------------------------------------------------------------
        # 4. Renaming Strategy ("USED_")
        # -------------------------------------------------------------
        # We rename the files BEFORE queuing so the worker works on the stable "USED_" path
        # and checking the folder prevents re-loops.
        
        final_cad_path = file_path
        final_config_path = config_path

        try:
            # Rename CAD
            used_cad_name = f"USED_{file_path.name}"
            used_cad_path = file_path.with_name(used_cad_name)
            
            # Simple rename
            os.rename(file_path, used_cad_path)
            final_cad_path = used_cad_path
            logger.info(f"   -> Renamed to {used_cad_path.name}")
            
            # Rename Config (if found)
            if config_path:
                used_config_name = f"USED_{config_path.name}"
                used_config_path = config_path.with_name(used_config_name)
                os.rename(config_path, used_config_path)
                final_config_path = used_config_path
                logger.info(f"   -> Renamed config to {used_config_path.name}")
                
        except OSError as e:
            logger.error(f"Failed to rename input files: {e}")
            logger.warning("   -> Proceeding with ORIGINAL filenames (Read-only input detected)")
            # Fallback to original paths so simulation still runs
            final_cad_path = file_path
            final_config_path = config_path

        # Queue the job with (potentially new, potentially original) paths

        
        # Queue the job
        try:
            job = self.queue.enqueue(
                'simops_worker.run_simulation',
                str(final_cad_path),
                str(self.output_dir),
                config_path=str(final_config_path) if final_config_path else None,
                job_timeout=1800,  # 30 min timeout
                result_ttl=86400,  # Keep result for 24h
                failure_ttl=86400,
                meta={
                    'filename': file_path.name, # Keep original name for display
                    'queued_at': datetime.now().isoformat(),
                }
            )
            logger.info(f"✅ Queued job {job.id} for: {file_path.name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to queue {file_path.name}: {e}")
            self.processing.discard(str_path)
            
    def _wait_for_file_stable(self, file_path: Path, timeout: int = 300) -> bool:
        """
        Wait for file to stop changing and be free of locks.
        
        Args:
            file_path: Path to file
            timeout: Max seconds to wait
            
        Returns:
            True if file is stable and ready, False if timeout
        """
        if not file_path.exists():
            return False
            
        prev_size = -1
        stable_count = 0
        start = time.time()
        last_log = start
        
        while time.time() - start < timeout:
            # 1. Check Size Stability
            try:
                curr_size = file_path.stat().st_size
            except OSError:
                time.sleep(1)
                continue
                
            # Log progress for large files every 10s
            if time.time() - last_log > 10:
                logger.info(f"   Waiting for {file_path.name}... Size: {curr_size/1024/1024:.2f} MB")
                last_log = time.time()

            if curr_size == prev_size and curr_size > 0:
                # 2. Check File Locks (Atomic Check)
                if self._is_file_locked(file_path):
                    stable_count = 0 # Reset if locked
                else:
                    stable_count += 1
                    if stable_count >= 3:  # Stable for ~3 seconds
                        return True
            else:
                stable_count = 0
                
            prev_size = curr_size
            
            # Exponential backoff-ish
            sleep_time = 1.0
            time.sleep(sleep_time)
            
        return False

    def _is_file_locked(self, filepath: Path) -> bool:
        """Checks if a file is locked by another process."""
        if not filepath.exists():
            return True # Treat missing as locked/unavailable

        if platform.system() == 'Windows':
            try:
                # Try to open for exclusive writing. 
                # If it fails, it's open by someone else (like a copy process).
                fd = os.open(str(filepath), os.O_RDWR | os.O_BINARY)
                # Attempt a lock (optional, opening exclusively is often enough on Windows)
                try:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                finally:
                    os.close(fd)
                return False
            except OSError:
                return True
        else:
            # Linux/Unix
            # Bind mounts (WSL2/Docker) often don't support flock, causing false positives.
            # Relying on size stability check is sufficient for hot folders.
            return False


def scan_existing_files(input_dir: Path, handler: SimOpsEventHandler):
    """Process any existing files in the input directory"""
    logger.info(f"Scanning for existing files in: {input_dir}")
    
    count = 0
    if not input_dir.exists():
        logger.warning(f"Input directory does not exist: {input_dir}")
        return

    try:
        for file_path in input_dir.iterdir():
            try:
                if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    if not file_path.name.startswith('.'):
                        handler._process_file(file_path)
                        count += 1
            except PermissionError:
                logger.warning(f"Permission denied accessing: {file_path}")
                continue
    except Exception as e:
        logger.error(f"Error scanning input dir: {e}")
                
    if count > 0:
        logger.info(f"Queued {count} existing files")


def run_watcher():
    """Main watcher loop"""
    # Configuration from environment
    redis_host = os.environ.get('REDIS_HOST', 'localhost')
    redis_port = int(os.environ.get('REDIS_PORT', 6379))
    
    # Path handling
    input_str = os.environ.get('INPUT_DIR', './input')
    output_str = os.environ.get('OUTPUT_DIR', './output')
    
    input_dir = PathHandler.normalize(input_str)
    output_dir = PathHandler.normalize(output_str)
    
    logger.info("=" * 60)
    logger.info("   SIMOPS HOT FOLDER WATCHER (ENHANCED)")
    logger.info("=" * 60)
    logger.info(f"   Redis:  {redis_host}:{redis_port}")
    logger.info(f"   Input:  {input_dir}")
    logger.info(f"   Output: {output_dir}")
    logger.info("=" * 60)
    
    # Create directories if legal (might fail on network shares if no permissions)
    try:
        if not PathHandler.is_network_path(input_dir):
            input_dir.mkdir(parents=True, exist_ok=True)
        if not PathHandler.is_network_path(output_dir):
            output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.warning(f"Could not create directories (might be network paths): {e}")

    # Connect to Redis
    logger.info("Connecting to Redis...")
    redis_conn = Redis(host=redis_host, port=redis_port)
    
    # Ping to verify connection
    try:
        redis_conn.ping()
        logger.info("✅ Redis connection successful")
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
        sys.exit(1)
        
    # Create job queue
    queue = Queue('simops', connection=redis_conn)
    
    # Set up file watcher
    # Use PollingObserver for better Docker/WSL2 bind mount support
    from watchdog.observers.polling import PollingObserver as Observer
    
    handler = SimOpsEventHandler(queue, output_dir)
    observer = Observer(timeout=float(os.environ.get('POLL_INTERVAL', 2.0)))
    
    try:
        observer.schedule(handler, str(input_dir), recursive=False)
        observer.start()
        logger.info(f"👀 Watching for new files in: {input_dir}")
        logger.info("   Drop STEP/IGES files here for automatic simulation")
    except OSError as e:
        logger.error(f"Failed to watch directory {input_dir}: {e}")
        logger.error("Is the path correct and accessible?")
        sys.exit(1)
    
    # Scan existing files first
    scan_existing_files(input_dir, handler)
    
    logger.info("")
    
    try:
        while True:
            # Check for Emergency Stop
            check_emergency_stop(input_dir, queue)
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down watcher...")
        observer.stop()
        
    observer.join()
    logger.info("Watcher stopped")


def check_emergency_stop(input_dir: Path, queue: Queue):
    """
    Checks for presence of 'STOP' or 'STOP_ALL' file.
    If found:
      1. Purge Redis Queue
      2. Kill Worker Containers
      3. Delete Stop File
    """
    stop_file = input_dir / "STOP"
    stop_all = input_dir / "STOP_ALL"
    
    target_file = None
    if stop_file.exists(): target_file = stop_file
    elif stop_all.exists(): target_file = stop_all
    
    if target_file:
        logger.warning(f"🛑 EMERGENCY STOP DETECTED: {target_file.name}")
        
        # 1. Purge Queue
        try:
            count = queue.empty()
            logger.info(f"   -> Purged {count} pending jobs from queue")
        except Exception as e:
            logger.error(f"   -> Failed to purge queue: {e}")
            
        # 2. Kill Docker Workers
        # We assume workers are named 'simops-worker-N'
        import subprocess
        try:
            # Find containers
            cmd = "docker ps -q --filter name=simops-worker"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            container_ids = result.stdout.strip().split()
            
            if container_ids:
                logger.info(f"   -> Found {len(container_ids)} active workers. Stopping...")
                kill_cmd = f"docker kill {' '.join(container_ids)}"
                subprocess.run(kill_cmd, shell=True, check=True)
                logger.info("   -> ✅ Workers stopped.")
            else:
                logger.info("   -> No active workers found.")
                
        except Exception as e:
            logger.error(f"   -> Failed to stop duplicate workers: {e}")
            
        # 3. Clean up
        try:
            target_file.unlink()
            logger.info(f"   -> Removed stop signal file.")
        except Exception as e:
            logger.error(f"   -> Failed to remove stop file: {e}")
            
        logger.info("========================================")
        logger.info("   SYSTEM HALTED BY USER")
        logger.info("========================================")


if __name__ == '__main__':
    run_watcher()
