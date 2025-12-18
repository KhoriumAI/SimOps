import argparse
import time
import sys
from pathlib import Path

def main():
    """
    Dummy worker that simulates a workload for CI/CD pipeline testing.
    """
    parser = argparse.ArgumentParser(description="Dummy SimOps Worker")
    parser.add_argument("input_file", help="Path to input file")
    parser.add_argument("--output", "-o", help="Output directory", default="output")
    args = parser.parse_args()

    print(f"[DummyWorker] Initializing job for input: {args.input_file}")
    print("[DummyWorker] Simulating physics (but not really)...")
    
    # Sleep to simulate work (10s as requested)
    time.sleep(10)
    
    # Convert paths
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result_file = output_dir / "results.txt"
    
    print(f"[DummyWorker] Writing results to {result_file}...")
    with open(result_file, "w") as f:
        f.write(f"Simulation Result for: {args.input_file}\n")
        f.write(f"Timestamp: {time.ctime()}\n")
        f.write("Status: SUCCESS\n")
        f.write("Computed Value: 42\n")
        
    print("[DummyWorker] Job Complete!")

if __name__ == "__main__":
    main()
