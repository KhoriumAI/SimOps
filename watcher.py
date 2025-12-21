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


# Supported CAD file extensions
SUPPORTED_EXTENSIONS = {'.step', '.stp', '.iges', '.igs', '.brep'}


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
        
        # Check for sidecar presence (just for logging)
        sidecar = file_path.with_suffix('.json')
        if sidecar.exists():
             logger.info(f"   + Sidecar config found: {sidecar.name}")
        
        # Queue the job
        try:
            job = self.queue.enqueue(
                'simops_worker.run_simulation',
                str(file_path),
                str(self.output_dir),
                job_timeout=1800,  # 30 min timeout
                result_ttl=86400,  # Keep result for 24h
                failure_ttl=86400,
                meta={
                    'filename': file_path.name,
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
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down watcher...")
        observer.stop()
        
    observer.join()
    logger.info("Watcher stopped")


if __name__ == '__main__':
    run_watcher()
