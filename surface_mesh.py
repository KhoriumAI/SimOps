#!/usr/bin/env python3
"""
Surface-only mesh generation - fast and simple for visualization
"""

import gmsh
import sys
import json
from pathlib import Path
from surface_quality import extract_surface_quality

def generate_surface_mesh(cad_file: str, max_size: float = 5.0, output_file: str = None):
    """
    Generate a 2D surface mesh only (no volumetric elements)
    Perfect for visualization, much faster than 3D meshing
    
    Args:
        cad_file: Path to CAD file (.step, .stp)
        max_size: Maximum element size in mm
        output_file: Optional output path (defaults to same name with _surface.msh)
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    
    try:
        cad_path = Path(cad_file)
        if not cad_path.exists():
            print(f"[X] File not found: {cad_file}")
            return False
        
        print(f"\n{'='*70}")
        print(f"SURFACE MESH GENERATION")
        print(f"{'='*70}")
        print(f"File: {cad_path.name}")
        print(f"Max element size: {max_size}mm")
        print()
        
        # Load CAD
        print("Loading CAD geometry...")
        gmsh.open(str(cad_file))
        
        # Get surface entities
        surfaces = gmsh.model.getEntities(dim=2)
        print(f"Found {len(surfaces)} surfaces")
        
        # Set mesh size
        print(f"Setting mesh size: {max_size}mm")
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", max_size * 0.1)
        
        # Algorithm settings for robustness
        gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay for 2D
        gmsh.option.setNumber("Mesh.RecombineAll", 0)  # Keep triangles
        gmsh.option.setNumber("Mesh.ElementOrder", 1)  # Linear elements for speed
        
        # Generate 2D mesh ONLY (no 3D)
        print("\nGenerating surface mesh...")
        gmsh.model.mesh.generate(2)  # 2 = surface mesh only
        
        # Count elements
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        element_types, element_tags, _ = gmsh.model.mesh.getElements(dim=2)
        
        num_nodes = len(node_tags)
        num_triangles = sum(len(tags) for tags in element_tags)
        
        print(f"\n[OK] Surface mesh generated!")
        print(f"  Nodes: {num_nodes:,}")
        print(f"  Triangles: {num_triangles:,}")
        
        # Determine output file
        if output_file is None:
            output_file = str(cad_path.parent / (cad_path.stem + "_surface.msh"))
        
        # Save mesh
        gmsh.write(output_file)
        print(f"\n[OK] Saved: {output_file}")

        gmsh.finalize()

        # Calculate surface quality metrics
        print("\nCalculating surface quality metrics...")
        quality_data = extract_surface_quality(output_file)

        if quality_data and quality_data.get('statistics'):
            stats = quality_data['statistics']
            print(f"\n📊 Quality Analysis:")
            print(f"  Quality score (0-1): {stats['avg_quality']:.3f} avg, {stats['min_quality']:.3f} min")
            print(f"  Skewness (0-1): {stats['avg_skewness']:.3f} avg, {stats['max_skewness']:.3f} max")
            print(f"  Aspect ratio: {stats['avg_aspect_ratio']:.2f} avg, {stats['max_aspect_ratio']:.2f} max")
            print(f"  Worst 10%: {stats['worst_10_count']} triangles highlighted in red")

            # Save quality data alongside mesh file
            quality_file = str(Path(output_file).with_suffix('.quality.json'))
            with open(quality_file, 'w') as f:
                json.dump(quality_data, f, indent=2)
            print(f"\n[OK] Quality data saved: {quality_file}")

        return True
        
    except Exception as e:
        print(f"\n[X] Error: {e}")
        import traceback
        traceback.print_exc()
        gmsh.finalize()
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python surface_mesh.py <cad_file> [max_size_mm]")
        print("\nExample: python surface_mesh.py teapot.step 5.0")
        sys.exit(1)
    
    cad_file = sys.argv[1]
    max_size = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    
    success = generate_surface_mesh(cad_file, max_size)
    sys.exit(0 if success else 1)
