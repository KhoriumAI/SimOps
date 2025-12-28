import gmsh
import sys

def check_cad(file_path):
    gmsh.initialize()
    gmsh.model.occ.importShapes(file_path)
    gmsh.model.occ.synchronize()
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
    print(f"Bounding Box: [{xmin}, {ymin}, {zmin}] to [{xmax}, {ymax}, {zmax}]")
    print(f"Dimensions: DX={xmax-xmin}, DY={ymax-ymin}, DZ={zmax-zmin}")
    gmsh.finalize()

if __name__ == "__main__":
    check_cad(sys.argv[1])
