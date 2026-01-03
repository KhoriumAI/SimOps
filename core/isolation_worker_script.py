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
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("--max-size", type=float, default=10.0)
    parser.add_argument("--min-size", type=float, default=1.0)
    parser.add_argument("--strategy", type=str, default="tet_hxt_optimized")
    args = parser.parse_args()

    print(f"[Worker V{args.tag}] Starting...", flush=True)

    # Create config and apply parameters
    config = get_default_config()
    config.mesh_params.element_order = args.order
    config.mesh_params.max_size_mm = args.max_size
    config.mesh_params.min_size_mm = args.min_size
    
    # Initialize Generator in Isolation Mode
    gen = ExhaustiveMeshGenerator(config, target_volume_tag=args.tag)
    
    # Use 1 thread per worker for maximum stability
    gen.initialize_gmsh(thread_count=1)
    
    # Set Gmsh verbosity to 3 for detailed sub-logs if user wants them
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("General.Verbosity", 3)
    
    # CRITICAL FIX: Ensure unique Node/Element IDs to prevent collisions during Assembly Merge
    # By offsetting each worker's tags by 1,000,000 * volume_id, we guarantee no overlap.
    # This solves the "Inverted Elements" issue caused by merging conflicting IDs.
    tag_offset = args.tag * 1000000
    try:
        gmsh.option.setNumber("Mesh.FirstNodeTag", tag_offset)
        gmsh.option.setNumber("Mesh.FirstElementTag", tag_offset)
        print(f"[Worker V{args.tag}] Applied Tag Offset: {tag_offset}", flush=True)
    except Exception as e:
        print(f"[Worker V{args.tag}] WARNING: Failed to set Tag Offset: {e}. Merge collisions may occur.", flush=True)

    print(f"[Worker V{args.tag}] Loading CAD file...", flush=True)
    try:
        if not gen.load_cad_file(args.input):
            print(f"[Worker V{args.tag}] ERROR: CAD load failed", flush=True)
            sys.exit(1)
            
        # Canary 1: Complexity
        comp = gen.check_1d_complexity()
        print(f"[Worker V{args.tag}] 1D Complexity: {comp['edge_count']} edges", flush=True)
        if comp['is_toxic']:
            print(f"[Worker V{args.tag}] TOXIC detected, boxing...", flush=True)
            gen.create_bounding_box_mesh(args.output)
            sys.exit(0)
            
        # Canary 2: 2D Surface
        print(f"[Worker V{args.tag}] Running 2D Canary...", flush=True)
        if not gen.generate_2d_canary(timeout=10.0):
            print(f"[Worker V{args.tag}] Canary timeout, boxing...", flush=True)
            gen.create_bounding_box_mesh(args.output)
            sys.exit(0)
            
        # Try 3D
        print(f"[Worker V{args.tag}] Generating 3D mesh (Order {args.order}, MinSize {args.min_size}, MaxSize {args.max_size}, Strategy {args.strategy})...", flush=True)
        
        # Dynamic Strategy Selection (User requested restoration of HXT + flexibility)
        success = False
        
        if "hxt" in args.strategy.lower():
            # Use HXT (Fast, Parallel) - User's preferred default for speed
            try:
                success, _ = gen._try_tet_hxt_optimized()
            except Exception as e:
                print(f"[Worker V{args.tag}] HXT failed: {e}. Falling back to Delaunay.", flush=True)
                success = False
        
        # Fallback or explicit Delaunay
        if not success:
             try:
               print(f"[Worker V{args.tag}] Using Standard Delaunay (Robust)...", flush=True)
               gmsh.option.setNumber("Mesh.Algorithm3D", 1) # 1: Delaunay
               gmsh.option.setNumber("Mesh.Optimize", 1)    # Basic optimization
               gmsh.model.mesh.generate(3)
               success = True
             except Exception as e:
               print(f"[Worker V{args.tag}] 3D Generation failed: {e}", flush=True)
               success = False

        if success:
            # CRITICAL: Enforce unique Physical Group for assembly stitching
            # We must use the original tag 'args.tag' so that when merged, 
            # this volume stays distinct from others.
            gmsh.model.removePhysicalGroups()
            final_vols = gmsh.model.getEntities(3)
            if final_vols:
                # Assuming single volume in isolation, but robustly handle list
                vol_tags = [v[1] for v in final_vols]
                # Use args.tag as the Physical Tag ID
                p_tag = args.tag 
                gmsh.model.addPhysicalGroup(3, vol_tags, p_tag)
                gmsh.model.setPhysicalName(3, p_tag, f"Volume_{args.tag}")
                print(f"[Worker V{args.tag}] Assigned Physical Volume {p_tag} (Name: Volume_{args.tag})", flush=True)
            
            # CRITICAL: Set save_all=False to only save the Physical Volume (tets).
            # If set to True, we save duplicate surface meshes which cause self-intersections when merged.
            gen.save_mesh(args.output, save_all=False)
            print(f"[Worker V{args.tag}] SUCCESS", flush=True)
            sys.exit(0)
        else:
            print(f"[Worker V{args.tag}] 3D failed, boxing...", flush=True)
            gen.create_bounding_box_mesh(args.output)
            sys.exit(0)
            
    except Exception as e:
        print(f"[Worker V{args.tag}] EXCEPTION: {e}", flush=True)
        sys.exit(1)
    finally:
        try: gmsh.finalize()
        except: pass

if __name__ == "__main__":
    main()
