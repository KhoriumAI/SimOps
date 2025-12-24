
import sys
from pathlib import Path

# Add root to python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.geometry_analyzer import analyze_cad_geometry

def test_analysis():
    print("Testing Geometry Analyzer...")
    
    test_files = [
        "cad_files/Cube.step",
        "cad_files/Loft.step",
        "cad_files/Cylinder.step"
    ]
    
    for relative_path in test_files:
        path = Path(__file__).parent.parent / relative_path
        print(f"\nAnalyzing: {path.name}")
        
        if not path.exists():
            print(f"  [Skipped] File not found: {path}")
            continue
            
        try:
            analysis = analyze_cad_geometry(str(path), verbose=False)
            print(analysis)
        except Exception as e:
            print(f"  [Failed] {e}")

if __name__ == "__main__":
    test_analysis()
