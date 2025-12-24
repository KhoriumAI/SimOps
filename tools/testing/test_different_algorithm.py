#!/usr/bin/env python3
"""Test with different algorithms"""
import gmsh

for alg_num, alg_name in [(1, "MeshAdapt"), (5, "Delaunay"), (6, "Frontal-Delaunay"), (8, "Delaunay-Quad")]:
    print(f"\n{'='*60}")
    print(f"Testing Algorithm {alg_num}: {alg_name}")
    print('='*60)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.open("CAD_files/Cylinder.step")

    # Set mesh sizes
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.01)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)

    # Set algorithm
    gmsh.option.setNumber("Mesh.Algorithm", alg_num)

    try:
        import time
        start = time.time()
        gmsh.model.mesh.generate(2)
        elapsed = time.time() - start

        nodes = gmsh.model.mesh.getNodes()
        elem_2d = gmsh.model.mesh.getElements(2)

        print(f"[OK] SUCCESS in {elapsed:.2f}s")
        print(f"  Nodes: {len(nodes[0])}")
        print(f"  2D elements: {sum(len(tags) for tags in elem_2d[1])}")

    except KeyboardInterrupt:
        print(f"[X] TIMEOUT/KILLED")
    except Exception as e:
        print(f"[X] ERROR: {e}")
    finally:
        gmsh.finalize()
