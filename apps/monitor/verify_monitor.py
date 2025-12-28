import time
import sys
import pandas as pd
from log_watcher import LogWatcher

# Add current dir to path
import os
sys.path.append(os.path.dirname(__file__))

def verify():
    # 1. Initialize Watcher
    log_dir = os.path.join(os.path.dirname(__file__), 'sim_logs')
    watcher = LogWatcher(log_dir)
    
    print("Waiting for logs to populate...")
    time.sleep(5) # Wait for generator
    
    # 2. Poll
    watcher.poll()
    
    df = watcher.get_dataframe()
    print(f"\nFound {len(df)} jobs.")
    
    if len(df) == 0:
        print("FAIL: No jobs found!")
        return
        
    # 3. Verify Columns
    print(f"Columns: {df.columns.tolist()}")
    
    # 4. Check internal data for Metrics and Stage
    sample_job = df.iloc[0]
    raw = sample_job['_raw_data']
    
    print(f"\nSample Job: {sample_job['Job ID']}")
    print(f"Stage: {sample_job['Stage']}")
    print(f"Metrics: {raw.get('Metrics', {})}")
    
    # Assertions
    if 'Metrics' not in raw:
        print("FAIL: 'Metrics' dict missing from internal state")
    elif not raw['Metrics']:
        print("FAIL: Metrics dict is empty (Parsing failed?)")
    else:
        print("PASS: Metrics captured successfully")
        
    if sample_job['Stage'] not in ['Initializing', 'Meshing', 'Solving', 'Post-Processing']:
         print(f"FAIL: Unexpected Stage '{sample_job['Stage']}'")
    else:
         print(f"PASS: Stage '{sample_job['Stage']}' is valid")
         
    # Check if we have Reynolds
    # Note: Reynolds is computed in 'solver' phase, which might take 10s+ in generator
    # We might need to wait longer
    max_wait = 20
    for i in range(max_wait):
        watcher.poll()
        df = watcher.get_dataframe()
        raw = df.iloc[0]['_raw_data']
        if 'Reynolds' in raw or 'reynolds_number' in raw.get('Metrics', {}):
             print(f"PASS: Reynolds number found after {i}s")
             break
        time.sleep(1)
        
    print("\nVerification Complete.")

if __name__ == "__main__":
    verify()
