import gmsh
import sys
import multiprocessing
import argparse
import os

def mesh_robust(input_file, output_file):
    # Initialize
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1) # Print progress to terminal
    
    print(f"Loading {input_file} with Aggressive Healing...")

    # 1. LOAD & HEAL
    # Load the STEP file. Using OCC kernel is mandatory for complex steps.
    try:
        # Aggressive healing to fix "1D mesh not closed" errors
        gmsh.option.setNumber("Geometry.OCCFixSmallEdges", 1)
        gmsh.option.setNumber("Geometry.OCCFixSmallFaces", 1)
        gmsh.option.setNumber("Geometry.OCCSewFaces", 1)
        
        gmsh.model.occ.importShapes(input_file)
        gmsh.model.occ.synchronize()
    except Exception as e:
        print(f"Error loading file: {e}")
        gmsh.finalize()
        sys.exit(1)

    # 2. THE SECRET SAUCE: FRAGMENTATION
    # This fixes the "hanging" issue. It cuts overlapping volumes so they share faces.
    ent_volumes = gmsh.model.getEntities(3)
    if len(ent_volumes) > 0:
        print(f"Fragmenting {len(ent_volumes)} volumes to ensure conformity...")
        try:
            # Fragment all volumes against each other
            gmsh.model.occ.fragment(ent_volumes, ent_volumes)
            gmsh.model.occ.synchronize()
        except Exception as e:
            print(f"Fragmentation warning: {e}") 
    else:
        print("No volumes found to fragment.")

    # 3. PLAN B: THE "FORCE FEED"
    # Prevent infinite loops on sliver surfaces significantly
    print("Applying 'Force Feed' settings (Plan B)...")
    
    # Stop trying to optimize invalid elements after 3 attempts (Default is infinite)
    gmsh.option.setNumber("Mesh.MaxRetries", 3)
    
    # Allow "Ugly" Elements - don't hang trying to fix them
    gmsh.option.setNumber("Mesh.Optimize", 0) 
    gmsh.option.setNumber("Mesh.HighOrderOptimize", 0)
    
    # Clamp Sizing to avoid microscopic elements on slivers
    # Setting min size prevents the mesher from going into the abyss
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.1) 
    gmsh.option.setNumber("Mesh.MeshSizeMax", 10.0)

    # Force Single Threading to avoid memory access violations
    gmsh.option.setNumber("General.NumThreads", 1) 
    
    # Algorithm 2D: 6 = Frontal-Delaunay (Robust)
    gmsh.option.setNumber("Mesh.Algorithm", 6) 
    
    # Algorithm 3D: 1 = Delaunay (The standard, robust mesher)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1) 
    
    # Disable Netgen
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)

    # 4. MESH
    print("Starting 2D Mesh...")
    gmsh.model.mesh.generate(2) 

    print("Starting 3D Mesh...")
    gmsh.model.mesh.generate(3)

    # 5. SAVE
    print(f"Saving to {output_file}...")
    gmsh.write(output_file)
    gmsh.finalize()
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robust Mesher")
    parser.add_argument("--input", required=True, help="Input CAD file")
    parser.add_argument("--output", required=True, help="Output mesh file")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
        
    mesh_robust(args.input, args.output)
