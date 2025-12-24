#!/usr/bin/env python3
"""Inspect cylinder surfaces to see what gmsh reports"""
import gmsh

gmsh.initialize()
gmsh.open("CAD_files/Cylinder.step")

surfaces = gmsh.model.getEntities(dim=2)
print(f"Found {len(surfaces)} surfaces:")
print()

for dim, tag in surfaces:
    surf_type = gmsh.model.getType(dim, tag)
    bounds = gmsh.model.getBoundingBox(dim, tag)

    # Get parametric bounds
    try:
        param_dim = gmsh.model.getParametrizationBounds(dim, tag)
        print(f"Surface {tag}:")
        print(f"  Type: {surf_type}")
        print(f"  Bbox: ({bounds[0]:.3f}, {bounds[1]:.3f}, {bounds[2]:.3f}) to ({bounds[3]:.3f}, {bounds[4]:.3f}, {bounds[5]:.3f})")
        print(f"  Param bounds: u=[{param_dim[0][0]:.3f}, {param_dim[0][1]:.3f}], v=[{param_dim[1][0]:.3f}, {param_dim[1][1]:.3f}]")

        # Try to get curvature at center
        u_mid = (param_dim[0][0] + param_dim[0][1]) / 2
        v_mid = (param_dim[1][0] + param_dim[1][1]) / 2

        try:
            curvs = gmsh.model.getCurvature(dim, tag, [u_mid, v_mid])
            print(f"  Curvature at center: k1={curvs[0]:.6f}, k2={curvs[1]:.6f}")
        except Exception as e:
            print(f"  Curvature: Failed - {e}")

    except Exception as e:
        print(f"Surface {tag}: Type={surf_type}, Error: {e}")

    print()

gmsh.finalize()
