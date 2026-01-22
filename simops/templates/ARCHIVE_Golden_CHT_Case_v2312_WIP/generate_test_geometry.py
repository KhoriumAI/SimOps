#!/usr/bin/env python3
"""
Golden CHT Case - Test Geometry Generator
==========================================

Generates simple STL geometry files for validation testing:
- heatsink.stl: 30x30x5mm aluminum plate
- chip.stl: 10x10x2mm silicon die on top of heatsink

Run this script before running the OpenFOAM case to generate the test geometry.
"""

import os
from pathlib import Path


def write_stl_box(filename: str, x_min: float, y_min: float, z_min: float,
                   x_max: float, y_max: float, z_max: float, name: str = "box"):
    """Write a simple box STL file."""
    
    # Define the 8 vertices
    v = [
        (x_min, y_min, z_min),  # 0
        (x_max, y_min, z_min),  # 1
        (x_max, y_max, z_min),  # 2
        (x_min, y_max, z_min),  # 3
        (x_min, y_min, z_max),  # 4
        (x_max, y_min, z_max),  # 5
        (x_max, y_max, z_max),  # 6
        (x_min, y_max, z_max),  # 7
    ]
    
    # Define 12 triangular facets (2 per face, 6 faces)
    facets = [
        # Bottom (z_min) - normal (0, 0, -1)
        ((0, 0, -1), v[0], v[2], v[1]),
        ((0, 0, -1), v[0], v[3], v[2]),
        # Top (z_max) - normal (0, 0, 1)
        ((0, 0, 1), v[4], v[5], v[6]),
        ((0, 0, 1), v[4], v[6], v[7]),
        # Front (y_min) - normal (0, -1, 0)
        ((0, -1, 0), v[0], v[1], v[5]),
        ((0, -1, 0), v[0], v[5], v[4]),
        # Back (y_max) - normal (0, 1, 0)
        ((0, 1, 0), v[2], v[3], v[7]),
        ((0, 1, 0), v[2], v[7], v[6]),
        # Left (x_min) - normal (-1, 0, 0)
        ((-1, 0, 0), v[0], v[4], v[7]),
        ((-1, 0, 0), v[0], v[7], v[3]),
        # Right (x_max) - normal (1, 0, 0)
        ((1, 0, 0), v[1], v[2], v[6]),
        ((1, 0, 0), v[1], v[6], v[5]),
    ]
    
    with open(filename, 'w') as f:
        f.write(f"solid {name}\n")
        for normal, v1, v2, v3 in facets:
            f.write(f"  facet normal {normal[0]} {normal[1]} {normal[2]}\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {v1[0]} {v1[1]} {v1[2]}\n")
            f.write(f"      vertex {v2[0]} {v2[1]} {v2[2]}\n")
            f.write(f"      vertex {v3[0]} {v3[1]} {v3[2]}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write(f"endsolid {name}\n")
    
    print(f"  Created: {filename}")


def main():
    """Generate test geometry files."""
    script_dir = Path(__file__).parent
    tri_surface_dir = script_dir / "constant" / "triSurface"
    tri_surface_dir.mkdir(parents=True, exist_ok=True)
    
    print("Golden CHT Case - Generating Test Geometry")
    print("=" * 50)
    
    # Heatsink: 30x30x5mm plate centered at origin, bottom at z=0
    # Units in mm (OpenFOAM will convert via convertToMeters)
    heatsink_file = tri_surface_dir / "heatsink.stl"
    write_stl_box(
        str(heatsink_file),
        x_min=-15, y_min=-15, z_min=0,
        x_max=15,  y_max=15,  z_max=5,
        name="heatsink"
    )
    
    # Chip: 10x10x2mm die centered on top of heatsink
    chip_file = tri_surface_dir / "chip.stl"
    write_stl_box(
        str(chip_file),
        x_min=-5,  y_min=-5,  z_min=5,
        x_max=5,   y_max=5,   z_max=7,
        name="chip"
    )
    
    print()
    print("Geometry files created in:", tri_surface_dir)
    print()
    print("Next steps:")
    print("  1. Run: blockMesh")
    print("  2. Run: snappyHexMesh -overwrite")
    print("  3. Run: splitMeshRegions -cellZones -overwrite")
    print("  4. Run: chtMultiRegionFoam")


if __name__ == "__main__":
    main()
