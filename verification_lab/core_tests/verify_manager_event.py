"""
Test script to verify Windows Manager-based Event pickling works correctly.
This is a minimal reproduction to confirm the fix before running the full meshing pipeline.
"""

import multiprocessing
import time
import os

def worker_function(args):
    """Simulate a worker that checks stop_event"""
    worker_id, stop_event = args
    
    print(f"Worker {worker_id} started (PID: {os.getpid()})")
    
    # Check if we should even start
    if stop_event.is_set():
        print(f"Worker {worker_id}: Cancelled before start")
        return (worker_id, False, "Cancelled before start")
    
    # Simulate some work
    for i in range(10):
        # Check stop event during work
        if stop_event.is_set():
            print(f"Worker {worker_id}: Received stop signal at iteration {i}")
            return (worker_id, False, f"Stopped at iteration {i}")
        
        time.sleep(0.5)
        print(f"Worker {worker_id}: Working... (iteration {i+1}/10)")
    
    print(f"Worker {worker_id}: Completed successfully")
    return (worker_id, True, "Completed")

def test_manager_event():
    """Test Manager-based Event with multiprocessing Pool"""
    print("=" * 60)
    print("Testing Windows Manager-based Event Pickling")
    print("=" * 60)
    
    num_workers = 2
    
    # CRITICAL: Use Manager to create Windows-safe picklable Event
    with multiprocessing.Manager() as manager:
        print("\n[OK] Manager created successfully")
        
        # Create Event via Manager (Windows-safe proxy)
        stop_event = manager.Event()
        print("[OK] Manager.Event() created successfully")
        
        # Prepare worker arguments (this will test pickling)
        worker_args = [(i, stop_event) for i in range(4)]
        print(f"[OK] Worker args prepared: {len(worker_args)} workers")
        
        print(f"\nStarting Pool with {num_workers} workers...")
        with multiprocessing.Pool(processes=num_workers) as pool:
            print("[OK] Pool created successfully")
            
            # Use imap_unordered to process results as they arrive
            results_iter = pool.imap_unordered(worker_function, worker_args)
            print("[OK] Workers started (Event was pickled successfully!)")
            
            # Monitor results
            print("\nMonitoring results...")
            result_count = 0
            for worker_id, success, message in results_iter:
                result_count += 1
                print(f"\n[Result {result_count}] Worker {worker_id}: {message}")
                
                # After first worker completes, signal others to stop
                if result_count == 1:
                    print("\n[!!!] First worker completed! Signaling others to stop...")
                    stop_event.set()
            
            print("\n[OK] All results collected")
            pool.close()
            pool.join()
            print("[OK] Pool closed and joined")
    
    print("\n" + "=" * 60)
    print("TEST PASSED: Manager-based Event works on Windows!")
    print("=" * 60)

if __name__ == "__main__":
    # Required for Windows
    multiprocessing.freeze_support()
    
    try:
        test_manager_event()
        print("\n✅ SUCCESS: No pickle errors, graceful shutdown works!")
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
