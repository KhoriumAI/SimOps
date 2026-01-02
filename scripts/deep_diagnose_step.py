import gmsh
import os
import sys

def test_load(filepath, use_healing=True):
    print(f"\n--- Testing Load (Healing={use_healing}) ---")
    gmsh.initialize()
    if not use_healing:
        gmsh.option.setNumber("Geometry.OCCFixSmallEdges", 0)
        gmsh.option.setNumber("Geometry.OCCFixSmallFaces", 0)
        gmsh.option.setNumber("Geometry.OCCFixDegenerated", 0)
        gmsh.option.setNumber("Geometry.OCCAutoFix", 0)
    
    gmsh.model.add("TestModel")
    
    try:
        # Try both methods
        print(f"Attempting gmsh.open()...")
        gmsh.open(filepath)
        print("Synchronizing...")
        gmsh.model.occ.synchronize()
        
        entities = gmsh.model.getEntities()
        print(f"Entities found: {len(entities)}")
        for dim in range(4):
            count = len([e for e in entities if e[0] == dim])
            print(f"  Dim {dim}: {count}")
            
        bbox = gmsh.model.getBoundingBox(-1, -1)
        print(f"BBox: {bbox}")
        
    except Exception as e:
        print(f"Error during load: {e}")
    
    gmsh.finalize()

def test_import_shapes(filepath):
    print(f"\n--- Testing gmsh.model.occ.importShapes() ---")
    gmsh.initialize()
    gmsh.model.add("ImportModel")
    try:
        # Disable healing
        gmsh.option.setNumber("Geometry.OCCFixSmallEdges", 0)
        gmsh.option.setNumber("Geometry.OCCFixSmallFaces", 0)
        gmsh.option.setNumber("Geometry.OCCFixDegenerated", 0)
        gmsh.option.setNumber("Geometry.OCCAutoFix", 0)
        
        print(f"Importing shapes...")
        out = gmsh.model.occ.importShapes(filepath)
        print(f"Imported {len(out)} top-level shapes.")
        
        print("Synchronizing...")
        gmsh.model.occ.synchronize()
        
        entities = gmsh.model.getEntities()
        print(f"Entities found: {len(entities)}")
        for dim in range(4):
            count = len([e for e in entities if e[0] == dim])
            print(f"  Dim {dim}: {count}")
            
    except Exception as e:
        print(f"Error during importShapes: {e}")
    gmsh.finalize()

if __name__ == "__main__":
    path = r"C:/Users/markm/Downloads/MeshPackageLean/cad_files/00010009_d97409455fa543b3a224250f_step_000 (1).step"
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)
        
    test_load(path, use_healing=True)
    test_load(path, use_healing=False)
    test_import_shapes(path)
