# core/automation/dispatcher.py
import time
import os
import json
import subprocess
import threading
import glob
import logging
from datetime import datetime
from pathlib import Path

try:
    from core.automation.registry import COMMANDS
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from core.automation.registry import COMMANDS

# CONFIG
BASE_DIR = Path(__file__).parent.parent.parent.absolute()
JOBS_DIR = BASE_DIR / "jobs_queue"
LOG_DIR = BASE_DIR / "jobs_log"
DISPATCHER_LOG_DIR = BASE_DIR / "logs"
POLL_RATE = 1.0 # Seconds

# Ensure required directories exist before logging
for d in [JOBS_DIR, LOG_DIR, DISPATCHER_LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(DISPATCHER_LOG_DIR / "dispatcher.log")
    ]
)
logger = logging.getLogger("Dispatcher")

def setup():
    """Ensure required directories exist."""
    logger.info("SIMOPS DISPATCHER ACTIVE")
    logger.info(f"   - Monitoring: {JOBS_DIR}")
    logger.info(f"   - Whitelisted Commands: {list(COMMANDS.keys())}")

def run_job(job_file: Path):
    """Executes a single job in a separate thread."""
    # Handle .processing extension or .json
    job_base_name = job_file.stem
    job_id = f"{job_base_name}.json" # Keep logic consistent with ID usually being filename
    
    try:
        # Load the ticket
        with open(job_file, 'r') as f:
            data = json.load(f)
        
        # 1. VALIDATE INTENT
        intent = data.get("intent")
        args = data.get("args", {})
        
        if intent not in COMMANDS:
            logger.error(f"[{job_id}] DENIED: Unknown intent '{intent}'")
            return

        # Inject base_dir into args for path-robust commands
        if not args: args = {}
        import sys
        args["base_dir"] = str(BASE_DIR).replace("\\", "/")
        args["python_exe"] = sys.executable.replace("\\", "/")

        # 2. CONSTRUCT COMMAND
        raw_cmd = COMMANDS[intent]
        try:
            final_cmd = raw_cmd.format(**args)
        except KeyError as e:
            logger.error(f"[{job_id}] ERROR: Missing argument {e}")
            return

        logger.info(f"[{job_id}] STARTING: {intent} -> {final_cmd}")
        
        # 3. EXECUTE
        start_time = time.time()
        # Use shell=True for Windows command execution
        result = subprocess.run(
            final_cmd, 
            shell=True, 
            capture_output=True, 
            text=True
        )
        duration = time.time() - start_time
        
        # 4. LOG RESULT
        status = "SUCCESS" if result.returncode == 0 else "FAILED"
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "job": job_id,
            "intent": intent,
            "command": final_cmd,
            "status": status,
            "duration": f"{duration:.2f}s",
            "stdout": result.stdout,
            "stderr": result.stderr
        }
        
        log_path = LOG_DIR / f"{job_base_name}.result.json"
        with open(log_path, 'w') as f:
            json.dump(log_entry, f, indent=2)
            
        if status == "SUCCESS":
            logger.info(f"[OK] [{job_id}] COMPLETED ({duration:.2f}s)")
        else:
            logger.warning(f"[FAIL] [{job_id}] FAILED ({duration:.2f}s). Check logs: {log_path}")
            # Print a snippet of stderr for immediate debugging if it failed
            if result.stderr:
                logger.error(f"[{job_id}] STDERR: {result.stderr[:500]}")

    except Exception as e:
        logger.error(f"[ERROR] [{job_id}] CRITICAL FAIL: {e}")
    finally:
        # Cleanup: Remove the ticket so we don't re-run it
        try:
            if job_file.exists():
                os.remove(job_file)
        except Exception as e:
            logger.error(f"Failed to cleanup job file {job_file}: {e}")

# Max concurrent jobs to prevent resource exhaustion
MAX_CONCURRENT_JOBS = 2
job_semaphore = threading.Semaphore(MAX_CONCURRENT_JOBS)

def run_job_wrapper(job_file: Path):
    """Wrapper to handle semaphore acquisition"""
    with job_semaphore:
        run_job(job_file)

def main_loop():
    setup()
    try:
        while True:
            # Find all .json files in queue
            jobs = list(JOBS_DIR.glob("*.json"))
            
            for job in jobs:
                # Check if we can acquire semaphore immediately (non-blocking would be complex in loop)
                # simpler: Just spawn the thread, but the thread waits on semaphore.
                # BUT if we allow infinite threads waiting, that's bad too? 
                # Actually, threads themselves are cheap, it's the subprocess that's heavy.
                # However, 100 threads waiting is fine.
                
                # ISSUE: If we spawn 100 threads, they will all try to delete the file later.
                # We need to move the file to "processing" state so main loop doesn't pick it up again.
                
                # Rename file to .processing to "claim" it
                processing_file = job.with_suffix(".processing")
                try:
                    job.rename(processing_file)
                except OSError:
                    # Could not rename, maybe another thread took it?
                    continue
                
                t = threading.Thread(target=run_job_wrapper, args=(processing_file,))
                t.daemon = True 
                t.start()
                
                # Tiny sleep to avoid file system race conditions
                time.sleep(0.1)
                
                # Tiny sleep to avoid file system race conditions
                time.sleep(0.1)
                
            time.sleep(POLL_RATE)
    except KeyboardInterrupt:
        logger.info("Dispatcher shutting down...")

if __name__ == "__main__":
    main_loop()
