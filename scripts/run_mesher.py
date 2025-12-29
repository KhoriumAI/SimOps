import subprocess
import sys
import os
import argparse
from tqdm import tqdm
import time

def run_mesher_with_progress(input_file, output_file):
    script_path = os.path.join(os.path.dirname(__file__), "robust_mesher.py")
    
    cmd = [sys.executable, script_path, "--input", input_file, "--output", output_file]
    
    print(f"Running robust mesher on {input_file}...")
    
    # Start Gmsh as a subprocess
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True,
        bufsize=1
    )
    
    # Heuristic progress bar
    # It's hard to know exact number of surfaces/volumes before loading, 
    # but we can show activity.
    pbar = tqdm(total=100, desc="Initializing", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}] {postfix}')
    
    current_stage = "Init"
    
    try:
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            
            if line:
                line = line.strip()
                # print(f"LOG: {line}") # Debug
                
                if "Fragmenting" in line:
                    current_stage = "Fragmenting"
                    pbar.set_description("Fragmenting Volumes")
                    pbar.update(5)
                elif "Starting 2D Mesh" in line:
                    current_stage = "Mesh 2D"
                    pbar.set_description("Meshing Surfaces (2D)")
                    pbar.update(10)
                elif "Starting 3D Mesh" in line:
                    current_stage = "Mesh 3D"
                    pbar.set_description("Meshing Volumes (3D)")
                    pbar.update(30) # Jump to 45%
                elif "Meshing surface" in line:
                    pbar.update(0.1) # Small increment
                elif "Meshing volume" in line:
                    pbar.update(0.5)
                elif "quality" in line.lower():
                    pbar.set_postfix_str(line[-40:])
                elif "Done!" in line:
                    pbar.set_description("Completed")
                    pbar.update(100 - pbar.n)
                elif "error" in line.lower() or "warning" in line.lower() or "failed" in line.lower() or "exception" in line.lower():
                    print(f"\nLOG: {line}")
                # else:
                #    print(f"LOG: {line}") # Debug output for everything else if needed
                    
    except KeyboardInterrupt:
        process.kill()
        print("\nInterrupted.")
        sys.exit(1)
        
    pbar.close()
    
    if process.returncode == 0:
        print(f"\nSuccess! Mesh saved to {output_file}")
    else:
        print(f"\nFailed with return code {process.returncode}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Mesher with Progress")
    parser.add_argument("--input", required=True, help="Input CAD file")
    parser.add_argument("--output", default="robust_mesh.msh", help="Output mesh file")
    args = parser.parse_args()
    
    run_mesher_with_progress(args.input, args.output)
