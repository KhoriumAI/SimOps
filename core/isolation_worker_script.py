import gmsh
import sys
import os
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from core.config import get_default_config
from strategies.exhaustive_strategy import ExhaustiveMeshGenerator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tag", type=int, required=True)
    args = parser.parse_args()

    print(f"[Worker V{args.tag}] Starting...", flush=True)

    # Load default config
    config = get_default_config()
    
    # Initialize Generator in Isolation Mode
    gen = ExhaustiveMeshGenerator(config, target_volume_tag=args.tag)
    
    # Use 1 thread per worker for maximum stability
    gen.initialize_gmsh(thread_count=1)
    
    print(f"[Worker V{args.tag}] Loading CAD file...", flush=True)
    try:
        if not gen.load_cad_file(args.input):
            print(f"[Worker V{args.tag}] CAD load failed!", flush=True)
            sys.exit(1)
            
        # Canary 1: Complexity
        comp = gen.check_1d_complexity()
        if comp['is_toxic']:
            gen.create_bounding_box_mesh(args.output)
            sys.exit(0)
            
        # Canary 2: 2D Surface
        if not gen.generate_2d_canary(timeout=5.0):
            gen.create_bounding_box_mesh(args.output)
            sys.exit(0)
            
        # Try 3D
        success, _ = gen._try_tet_hxt_optimized()
        if success:
            gen.save_mesh(args.output)
            sys.exit(0)
        else:
            gen.create_bounding_box_mesh(args.output)
            sys.exit(0)
            
    except Exception as e:
        print(f"Worker Error: {e}")
        sys.exit(1)
    finally:
        try: gmsh.finalize()
        except: pass

if __name__ == "__main__":
    main()
