import time
import os
import json
import subprocess
import threading
import glob
from datetime import datetime
import sys

# Add current directory to path so we can import registry
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import importlib
import registry

# CONFIG
JOBS_DIR = "jobs_queue"
LOG_DIR = "jobs_log"
POLL_RATE = 1.0 # Seconds

def get_commands():
    """Reloads the registry and returns the COMMANDS dict."""
    importlib.reload(registry)
    return registry.COMMANDS

def setup():
    for d in [JOBS_DIR, LOG_DIR]:
        os.makedirs(d, exist_ok=True)
    commands = get_commands()
    print(f"DISPATCHER ACTIVE.")
    print(f"   - Watching: ./{JOBS_DIR}")
    print(f"   - Allowed Commands: {list(commands.keys())}")
    sys.stdout.flush()

def run_job(job_file):
    """Executes a single job in a separate thread."""
    job_id = os.path.basename(job_file)
    
    try:
        # Retry reading in case of file system race
        data = None
        for _ in range(5):
            try:
                with open(job_file, 'r') as f:
                    data = json.load(f)
                break
            except (json.JSONDecodeError, PermissionError):
                time.sleep(0.1)
        
        if data is None:
            print(f"[X] [{job_id}] FAIL: Could not read job file.")
            return

        # 1. VALIDATE INTENT
        intent = data.get("intent")
        args = data.get("args", {})
        
        commands = get_commands()
        if intent not in commands:
            print(f"[X] [{job_id}] DENIED: Unknown intent '{intent}'")
            sys.stdout.flush()
            return

        # 2. CONSTRUCT COMMAND
        raw_cmd = commands[intent]
        try:
            # Add project_root if needed and not provided
            if "{project_root}" in raw_cmd and "project_root" not in args:
                args["project_root"] = os.path.dirname(os.path.abspath(__file__))
            
            final_cmd = raw_cmd.format(**args)
        except KeyError as e:
            print(f"[X] [{job_id}] ERROR: Missing argument {e}")
            sys.stdout.flush()
            return

        print(f"[*] [{job_id}] STARTING: {intent}")
        print(f"   Cmd: {final_cmd}")
        sys.stdout.flush()
        
        # 3. EXECUTE
        start_time = time.time()
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
            "stderr": result.stderr,
            "returncode": result.returncode
        }
        
        log_path = os.path.join(LOG_DIR, job_id.replace(".json", ".log"))
        with open(log_path, 'w') as f:
            json.dump(log_entry, f, indent=2)
            
        marker = "[OK]" if status == "SUCCESS" else "[!]"
        print(f"{marker} [{job_id}] {status} ({duration:.2f}s) -> Logs saved to {log_path}")
        sys.stdout.flush()

    except Exception as e:
        print(f"[ERROR] [{job_id}] CRITICAL FAIL: {e}")
        sys.stdout.flush()
    finally:
        # Cleanup: Remove the ticket so we don't re-run it
        if os.path.exists(job_file):
            try:
                os.remove(job_file)
            except Exception as e:
                print(f"[!] [{job_id}] Failed to remove job file: {e}")
                sys.stdout.flush()

def main_loop():
    setup()
    while True:
        try:
            # Find all .json files in queue
            jobs = glob.glob(os.path.join(JOBS_DIR, "*.json"))
            
            for job in jobs:
                # To prevent double-pickup, rename it first
                processing_file = job + ".processing"
                try:
                    os.rename(job, processing_file)
                except OSError:
                    continue # Already being processed or similar
                
                # Create a localized thread for this job
                t = threading.Thread(target=run_job, args=(processing_file,))
                t.daemon = True # Don't block exit
                t.start()
                
                time.sleep(0.1)
                
            time.sleep(POLL_RATE)
        except KeyboardInterrupt:
            print("\nShutting down dispatcher...")
            break
        except Exception as e:
            print(f"Dispatcher loop error: {e}")
            time.sleep(POLL_RATE)

if __name__ == "__main__":
    main_loop()
