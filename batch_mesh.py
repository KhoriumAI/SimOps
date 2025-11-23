#!/usr/bin/env python3
"""
Batch mesh generation script - no GUI required
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from mesh_worker_subprocess import generate_mesh

def mesh_file(cad_file: str, target_elements: int = 5000, max_size_mm: float = 10.0):
    """Mesh a single CAD file with specified parameters"""
    
    file_path = Path(cad_file)
    if not file_path.exists():
        print(f"[X] File not found: {cad_file}")
        return None
    
    print(f"\n{'='*70}")
    print(f"Meshing: {file_path.name}")
    print(f"{'='*70}")
    
    # Configure quality parameters
    quality_params = {
        'quality_preset': 'Medium',
        'target_elements': target_elements,
        'max_size_mm': max_size_mm,
        'curvature_adaptive': True
    }
    
    print(f"Parameters:")
    print(f"  Target elements: {target_elements:,}")
    print(f"  Max element size: {max_size_mm} mm")
    print(f"  Curvature adaptive: ON")
    print()
    
    # Generate mesh
    result = generate_mesh(cad_file, quality_params=quality_params)
    
    if result.get('success'):
        print(f"\n[OK] SUCCESS!")
        print(f"  Output: {result['output_file']}")
        print(f"  Elements: {result.get('total_elements', 0):,}")
        print(f"  Nodes: {result.get('total_nodes', 0):,}")
        
        # Show quality metrics
        metrics = result.get('quality_metrics', {})
        if 'geometric_accuracy' in metrics:
            print(f"  Shape Accuracy: {metrics['geometric_accuracy']:.3f}")
        if 'sicn_avg' in metrics:
            print(f"  SICN (avg): {metrics['sicn_avg']:.3f}")
        
        return result
    else:
        print(f"\n[X] FAILED: {result.get('error')}")
        return None

if __name__ == "__main__":
    print("="*70)
    print("BATCH MESH GENERATION")
    print("="*70)
    
    # Define files to mesh
    mesh_dir = Path("/Users/animeneko/Downloads/Mesh Animation")
    
    files_to_mesh = [
        {
            'file': mesh_dir / "teapot.step",
            'target_elements': 8000,
            'max_size_mm': 5.0
        },
        {
            'file': mesh_dir / "M2 16T 32pd Spur.STEP",
            'target_elements': 5000,
            'max_size_mm': 2.0
        },
        {
            'file': mesh_dir / "Stanford_Bunny.step",
            'target_elements': 10000,
            'max_size_mm': 3.0
        },
        {
            'file': mesh_dir / "Tree3.step",
            'target_elements': 6000,
            'max_size_mm': 8.0
        }
    ]
    
    results = []
    for i, config in enumerate(files_to_mesh, 1):
        print(f"\n[{i}/{len(files_to_mesh)}]")
        result = mesh_file(
            str(config['file']),
            target_elements=config['target_elements'],
            max_size_mm=config['max_size_mm']
        )
        results.append({
            'file': config['file'].name,
            'success': result is not None,
            'result': result
        })
    
    # Summary
    print("\n" + "="*70)
    print("BATCH SUMMARY")
    print("="*70)
    
    successful = sum(1 for r in results if r['success'])
    print(f"Completed: {successful}/{len(results)}")
    print()
    
    for r in results:
        status = "[OK]" if r['success'] else "[X]"
        print(f"{status} {r['file']}")
        if r['success'] and r['result']:
            print(f"    {r['result'].get('total_elements', 0):,} elements")
    
    print("\n" + "="*70)
