#!/usr/bin/env python3
"""
Simple test to verify subprocess and TemporaryFile work
"""
import sys
import subprocess
import tempfile
import time

def test_simple():
    print("Testing simple subprocess...")
    
    # Command that just prints and exits
    cmd = [sys.executable, "-c", "import sys; import time; print('Hello from subprocess', flush=True); time.sleep(1); print('Done', flush=True)"]
    
    # Use NamedTemporaryFile so we can open a separate reader
    with tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8', delete=False) as log_file:
        log_path = log_file.name
        
    print(f"Log file: {log_path}")
    
    try:
        with open(log_path, 'w') as w_file:
            process = subprocess.Popen(
                cmd,
                stdout=w_file,
                stderr=subprocess.STDOUT,
                text=True
            )
            print(f"Started PID: {process.pid}")
            
            with open(log_path, 'r') as r_file:
                while process.poll() is None:
                    line = r_file.readline()
                    if line:
                        print(f"[SUB] {line.strip()}")
                    else:
                        time.sleep(0.1)
                        
                # Final read
                for line in r_file:
                    print(f"[SUB] {line.strip()}")
                    
        print("Finished")
    finally:
        if os.path.exists(log_path):
            os.unlink(log_path)

if __name__ == "__main__":
    test_simple()
