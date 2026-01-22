import gmsh
import sys
import json
import os

def run_fast_mesh(input_path, output_path):
    """
    Executes GMSH with validiation checks disabled for maximum throughput.
    """
    print(f"[CLI] Processing: {input_path}")
    
    gmsh.initialize()
    # Terminal=1 enables real-time progress logging in the subprocess
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("General.Verbosity", 3)
    
    try:
        gmsh.open(input_path)
        gmsh.model.occ.synchronize()

        # DISABLING OPTIMIZATION
        # Forces GMSH to output the first valid mesh it finds
        # This prevents infinite loops on dirty geometry
        gmsh.option.setNumber("Mesh.CheckAllElements", 0)       
        gmsh.option.setNumber("Mesh.Optimize", 0)               
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)         
        gmsh.option.setNumber("Mesh.Algorithm", 1) # MeshAdapt
        gmsh.option.setNumber("Mesh.MaxRetries", 1)
        
        # SATURATE CORES
        # Threadripper can handle independent surface meshing in parallel
        gmsh.option.setNumber("General.NumThreads", os.cpu_count() or 32)
        
        # COARSE SIZING
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
        diag = ((xmax-xmin)**2 + (ymax-ymin)**2 + (zmax-zmin)**2)**0.5
        gmsh.option.setNumber("Mesh.MeshSizeMin", diag / 100.0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", diag / 20.0)

        gmsh.model.mesh.generate(2)
        
        # EXTRACT DATA
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        nodes = {int(tag): [node_coords[3*i], node_coords[3*i+1], node_coords[3*i+2]] 
                 for i, tag in enumerate(node_tags)}
        
        vertices = []
        elem_types, _, node_tags_list = gmsh.model.mesh.getElements(2)
        
        for etype, enodes in zip(elem_types, node_tags_list):
            # Type 2 is a 3-node triangle
            if etype == 2:
                enodes_list = enodes.astype(int).tolist()
                for i in range(0, len(enodes_list), 3):
                    n1, n2, n3 = enodes_list[i], enodes_list[i+1], enodes_list[i+2]
                    if n1 in nodes and n2 in nodes and n3 in nodes:
                        vertices.extend(nodes[n1] + nodes[n2] + nodes[n3])

        result = {
            "vertices": vertices,
            "numVertices": len(vertices) // 3,
            "numTriangles": len(vertices) // 9,
            "isPreview": True,
            "status": "success"
        }
        
        with open(output_path, 'w') as f:
            json.dump(result, f)
            
        print(f"[CLI] Success. Saved to {output_path}")
        return True

    except Exception as e:
        print(f"[CLI ERROR] {e}")
        return False
    finally:
        gmsh.finalize()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python mesh_fast_cli.py <input.step> <output.json>")
        sys.exit(1)
        
    success = run_fast_mesh(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)
