"""
Test Quality Scoring Logic
===========================

Verifies that the new logarithmic scoring prioritizes quality over element count.
"""

def test_quality_scoring():
    """Test that adaptive_coarse_to_fine beats very_coarse_tet"""
    import sys
    sys.path.insert(0, 'c:\\Users\\Owner\\Downloads\\MeshPackageLean')
    
    from core.mesh_generator import BaseMeshGenerator
    from core.config import get_default_config
    
    # Create a dummy generator to access scoring method
    class DummyGenerator(BaseMeshGenerator):
        def run_meshing_strategy(self, input_file, output_file):
            return True
    
    gen = DummyGenerator(get_default_config())
    
    # Metrics from actual test (user's data)
    adaptive_metrics = {
        'total_elements': 59059,
        'gmsh_sicn': {'min': 0.50, 'avg': 0.75, 'max': 0.95}  # Good quality
    }
    
    very_coarse_metrics = {
        'total_elements': 2105,
        'gmsh_sicn': {'min': 0.20, 'avg': 0.37, 'max': 0.80}  # Bad quality
    }
    
    # Calculate scores (lower is better)
    adaptive_score = gen._calculate_quality_score(adaptive_metrics)
    coarse_score = gen._calculate_quality_score(very_coarse_metrics)
    
    print("="*60)
    print("QUALITY SCORING TEST")
    print("="*60)
    print(f"\nadaptive_coarse_to_fine:")
    print(f"  Elements: 59,059")
    print(f"  SICN avg: 0.75 (Good)")
    print(f"  SICN min: 0.50")
    print(f"  Score: {adaptive_score:.2f}")
    
    print(f"\nvery_coarse_tet:")
    print(f"  Elements: 2,105")
    print(f"  SICN avg: 0.37 (Bad)")
    print(f"  SICN min: 0.20")
    print(f"  Score: {coarse_score:.2f}")
    
    print(f"\n{'='*60}")
    
    if adaptive_score < coarse_score:
        print("PASS: adaptive_coarse_to_fine WINS (lower score)")
        print(f"   Score difference: {coarse_score - adaptive_score:.2f} points")
        print("   Quality is correctly prioritized over element count!")
    else:
        print("FAIL: very_coarse_tet WINS (lower score)")
        print(f"   Score difference: {adaptive_score - coarse_score:.2f} points")
        print("   Element count still dominating - fix needed!")
    
    print(f"{'='*60}\n")
    
    # Show the math breakdown
    import math
    print("Score Breakdown:")
    print(f"\nadaptive_coarse_to_fine:")
    print(f"  Quality penalty: (1.0 - 0.75) * 100 = {(1.0 - 0.75) * 100:.1f}")
    print(f"  Element penalty: log10(59059) * 2.0 = {math.log10(59059) * 2.0:.1f}")
    print(f"  Total: {adaptive_score:.2f}")
    
    print(f"\nvery_coarse_tet:")
    print(f"  Quality penalty: (1.0 - 0.37) * 100 = {(1.0 - 0.37) * 100:.1f}")
    print(f"  Min SICN penalty: +20 (SICN min < 0.2)")
    print(f"  Element penalty: log10(2105) * 2.0 = {math.log10(2105) * 2.0:.1f}")
    print(f"  Total: {coarse_score:.2f}")
    
    return adaptive_score < coarse_score

if __name__ == "__main__":
    success = test_quality_scoring()
    sys.exit(0 if success else 1)
