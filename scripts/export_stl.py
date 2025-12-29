import gmsh
import sys
import os

def export_stl(input_file, output_file):
    print(f"Loading {input_file}...")
    gmsh.initialize()
    
    # 1. LOAD
    try:
        # Enable Aggressive Healing to fix "1D mesh" errors
        gmsh.option.setNumber("Geometry.OCCFixSmallEdges", 1)
        gmsh.option.setNumber("Geometry.OCCFixSmallFaces", 1)
        gmsh.option.setNumber("Geometry.OCCSewFaces", 1)
        
        gmsh.model.occ.importShapes(input_file)
        gmsh.model.occ.synchronize()
    except Exception as e:
        print(f"Error loading file: {e}")
        gmsh.finalize()
        sys.exit(1)

    # 2. SKIP FRAGMENTATION FOR STL EXPORT
    # fTetWild doesn't need conformal interfaces; fragmentation creates slivers.
    # We just need a watertight surface mesh.
    print("Skipping fragmentation (fTetWild will handle non-conformal boundaries)...")

    # 3. MESH SETTINGS FOR SPEED & ROBUSTNESS
    print("Generating surface mesh (STL) with Force Feed settings...")
    
    # KILL THE OPTIMIZER to prevent hanging on slivers
    gmsh.option.setNumber("Mesh.Optimize", 0)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)
    gmsh.option.setNumber("Mesh.HighOrderOptimize", 0)
    gmsh.option.setNumber("Mesh.MaxRetries", 0) # Don't retry invalid elements
    
    gmsh.option.setNumber("Mesh.Algorithm", 6) # Frontal Delaunay for 2D
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0) # Disable curvature sizing (too many tiny triangles)
    
    # CRITICAL: Set a min size to bridge small gaps
    # User suggested 0.5 - 5.0
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5) 
    gmsh.option.setNumber("Mesh.MeshSizeMax", 5.0)

    # 4. GENERATE 2D MESH
    # Wrap in try/except to allow partial mesh output
    try:
        gmsh.model.mesh.generate(2) # Generate Surface Mesh ONLY
        print("2D mesh generation completed successfully.")
    except Exception as e:
        print(f"Warning: 2D mesh generation had issues: {e}")
        print("Saving partial mesh anyway...")
    
    # 5. WRITE STL
    print(f"Writing to {output_file}...")
    gmsh.write(output_file)
    gmsh.finalize()
    print("Done!")

if __name__ == "__main__":
    # Hardcoded defaults as per user instruction, or args
    input_CAD = "cad_files/00-CM240411_BIANCA-C00_HEATER_BOAR_0704A1.STEP"
    output_STL = "dirty_assembly.stl"
    
    # Check if files exist
    if not os.path.exists(input_CAD):
        # try absolute path
        input_CAD = os.path.join(os.getcwd(), input_CAD)
        if not os.path.exists(input_CAD):
            print(f"Error: Input file {input_CAD} not found")
            sys.exit(1)
            
    export_stl(input_CAD, output_STL)
