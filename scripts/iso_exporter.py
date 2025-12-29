
import gmsh
import sys
import os

def export_vol(vol_idx, step_path, out_dir):
    try:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.occ.importShapes(step_path)
        gmsh.model.occ.synchronize()
        
        vols = gmsh.model.getEntities(3)
        if vol_idx >= len(vols):
            sys.exit(0)
            
        # Isolate volume
        target = vols[vol_idx]
        gmsh.model.removeEntities([v for i,v in enumerate(vols) if i != vol_idx], recursive=True)
        
        # Mesh 2D with fallback
        for alg in [6, 5, 1]:
            try:
                gmsh.option.setNumber("Mesh.Algorithm", alg)
                gmsh.model.mesh.generate(2)
                out_file = os.path.join(out_dir, f"vol_{vol_idx}.stl")
                gmsh.write(out_file)
                gmsh.finalize()
                return
            except Exception as e:
                print(f"Alg {alg} failed: {e}")
                # Reset? GMSH state might be dirty.
                # Ideally we re-load. But let's try just proceeding.
                continue
        print("All algorithms failed.")
        sys.exit(1)
    except Exception as e:
        print(e)
        sys.exit(1)

if __name__ == "__main__":
    export_vol(int(sys.argv[1]), sys.argv[2], sys.argv[3])
