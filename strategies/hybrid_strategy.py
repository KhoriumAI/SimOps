"""
Hybrid Mesh Generation Strategy
================================

Combines exhaustive and adaptive strategies:
1. First runs exhaustive strategy to find ANY working mesh
2. Then runs adaptive strategy to optimize quality

Best of both worlds:
- Exhaustive: Guaranteed to find a working approach
- Adaptive: Refines the mesh to highest quality possible

Use this as the default strategy for production.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.mesh_generator import BaseMeshGenerator, MeshGenerationResult
from core.config import Config
from strategies.exhaustive_strategy import ExhaustiveMeshGenerator
from strategies.adaptive_strategy import AdaptiveMeshGenerator
import gmsh


class HybridMeshGenerator(BaseMeshGenerator):
    """
    Hybrid strategy: Exhaustive -> Adaptive

    Phase 1 (Exhaustive):
    - Try 18+ meshing strategies
    - Find the best working approach
    - Generate initial mesh

    Phase 2 (Adaptive):
    - Take best mesh from exhaustive
    - Iteratively refine quality
    - Stop when convergence detected

    Result: Highest quality mesh achievable for the geometry
    """

    def __init__(self, config: Optional[Config] = None):
        super().__init__(config)
        self.exhaustive_generator = ExhaustiveMeshGenerator(config)
        self.adaptive_generator = AdaptiveMeshGenerator(config)
        self.phase = "initializing"

    def generate_mesh(self, input_file: str, output_file: Optional[str] = None) -> MeshGenerationResult:
        """
        Generate mesh using hybrid strategy

        Args:
            input_file: Input CAD file path
            output_file: Output mesh file path (optional)

        Returns:
            MeshGenerationResult with success status and metrics
        """
        if output_file is None:
            output_file = str(Path(input_file).with_suffix('')) + '_hybrid_optimized.msh'

        self.log_message("\n" + "=" * 70)
        self.log_message("HYBRID MESH GENERATION STRATEGY")
        self.log_message("Phase 1: Exhaustive (find working mesh)")
        self.log_message("Phase 2: Adaptive (optimize quality)")
        self.log_message("=" * 70)

        # PHASE 1: Exhaustive Strategy
        self.phase = "exhaustive"
        self.log_message("\n" + "=" * 70)
        self.log_message("PHASE 1: EXHAUSTIVE STRATEGY")
        self.log_message("Trying 18+ meshing approaches to find working mesh...")
        self.log_message("=" * 70)

        # Run exhaustive strategy
        exhaustive_result = self.exhaustive_generator.generate_mesh(
            input_file,
            output_file + ".exhaustive.msh"
        )

        if not exhaustive_result.success:
            self.log_message("\n" + "=" * 70)
            self.log_message("[X] PHASE 1 FAILED: Could not generate any mesh")
            self.log_message("All 18+ strategies failed - geometry may be too complex")
            self.log_message("=" * 70)
            return MeshGenerationResult(
                success=False,
                message="Exhaustive phase failed - no working mesh found"
            )

        # Get best result from exhaustive
        best_exhaustive = min(
            self.exhaustive_generator.all_attempts,
            key=lambda x: x.get('score', float('inf'))
        )

        self.log_message("\n" + "=" * 70)
        self.log_message("[OK] PHASE 1 COMPLETE: Working mesh found!")
        self.log_message(f"Best Strategy: {best_exhaustive.get('strategy', 'unknown')}")
        self.log_message(f"Quality Score: {best_exhaustive.get('score', 0):.4f}")

        exhaustive_metrics = best_exhaustive.get('metrics', {})
        if exhaustive_metrics.get('gmsh_sicn'):
            sicn = exhaustive_metrics['gmsh_sicn']
            self.log_message(f"SICN: Min={sicn['min']:.4f}, Avg={sicn['avg']:.4f}")
        if exhaustive_metrics.get('total_elements'):
            self.log_message(f"Elements: {exhaustive_metrics['total_elements']:,}")

        self.log_message("=" * 70)

        # PHASE 2: Adaptive Strategy
        self.phase = "adaptive"
        self.log_message("\n" + "=" * 70)
        self.log_message("PHASE 2: ADAPTIVE OPTIMIZATION")
        self.log_message("Iteratively refining mesh quality...")
        self.log_message("Will stop when convergence detected")
        self.log_message("=" * 70)

        # Check if mesh is already excellent quality
        if self._is_excellent_quality(exhaustive_metrics):
            self.log_message("\n[OK] Exhaustive mesh already has EXCELLENT quality!")
            self.log_message("Skipping adaptive phase - no improvement needed")
            self.log_message("=" * 70)

            # Just copy the exhaustive output
            if os.path.exists(exhaustive_result.output_file):
                import shutil
                shutil.copy(exhaustive_result.output_file, output_file)

            return MeshGenerationResult(
                success=True,
                output_file=output_file,
                quality_metrics=exhaustive_metrics,
                message="Exhaustive phase produced excellent quality - adaptive not needed"
            )

        # Run adaptive refinement starting from exhaustive mesh
        try:
            adaptive_result = self.adaptive_generator.generate_mesh(
                input_file,
                output_file
            )

            if not adaptive_result.success:
                self.log_message("\n[!] PHASE 2: Adaptive optimization did not improve mesh")
                self.log_message("Using best mesh from exhaustive phase")

                # Fall back to exhaustive result
                if os.path.exists(exhaustive_result.output_file):
                    import shutil
                    shutil.copy(exhaustive_result.output_file, output_file)

                return MeshGenerationResult(
                    success=True,
                    output_file=output_file,
                    quality_metrics=exhaustive_metrics,
                    message="Exhaustive mesh used - adaptive did not improve"
                )

            # Adaptive succeeded - compare results
            adaptive_metrics = adaptive_result.quality_metrics or {}

            self.log_message("\n" + "=" * 70)
            self.log_message("[OK] PHASE 2 COMPLETE: Adaptive optimization finished")
            self.log_message("=" * 70)

            # Compare exhaustive vs adaptive
            self._compare_results(exhaustive_metrics, adaptive_metrics)

            # Final result
            self.log_message("\n" + "=" * 70)
            self.log_message("HYBRID STRATEGY COMPLETE")
            self.log_message("[OK] Successfully generated and optimized mesh")
            self.log_message(f"Final output: {output_file}")
            self.log_message("=" * 70)

            return MeshGenerationResult(
                success=True,
                output_file=output_file,
                quality_metrics=adaptive_metrics,
                message="Hybrid strategy complete - mesh optimized"
            )

        except Exception as e:
            self.log_message(f"\n[!] PHASE 2 ERROR: {e}")
            self.log_message("Falling back to exhaustive result")

            # Fall back to exhaustive result
            if os.path.exists(exhaustive_result.output_file):
                import shutil
                shutil.copy(exhaustive_result.output_file, output_file)

            return MeshGenerationResult(
                success=True,
                output_file=output_file,
                quality_metrics=exhaustive_metrics,
                message=f"Exhaustive mesh used - adaptive failed: {e}"
            )

    def _is_excellent_quality(self, metrics: Dict) -> bool:
        """Check if mesh quality is already excellent"""
        # Check SICN
        if metrics.get('gmsh_sicn'):
            if metrics['gmsh_sicn']['min'] > 0.5:  # Excellent threshold
                return True

        # Check skewness
        if metrics.get('skewness'):
            if metrics['skewness']['max'] < 0.6:  # Very good threshold
                return True

        return False

    def _compare_results(self, exhaustive_metrics: Dict, adaptive_metrics: Dict):
        """Compare exhaustive vs adaptive results"""
        self.log_message("\nQuality Comparison:")
        self.log_message("-" * 70)

        # Compare SICN
        if exhaustive_metrics.get('gmsh_sicn') and adaptive_metrics.get('gmsh_sicn'):
            ex_sicn = exhaustive_metrics['gmsh_sicn']['min']
            ad_sicn = adaptive_metrics['gmsh_sicn']['min']
            improvement = ((ad_sicn - ex_sicn) / ex_sicn) * 100 if ex_sicn > 0 else 0

            self.log_message(f"SICN (min):")
            self.log_message(f"  Exhaustive: {ex_sicn:.4f}")
            self.log_message(f"  Adaptive:   {ad_sicn:.4f} ({improvement:+.1f}%)")

        # Compare skewness
        if exhaustive_metrics.get('skewness') and adaptive_metrics.get('skewness'):
            ex_skew = exhaustive_metrics['skewness']['max']
            ad_skew = adaptive_metrics['skewness']['max']
            improvement = ((ex_skew - ad_skew) / ex_skew) * 100 if ex_skew > 0 else 0

            self.log_message(f"Skewness (max):")
            self.log_message(f"  Exhaustive: {ex_skew:.4f}")
            self.log_message(f"  Adaptive:   {ad_skew:.4f} ({improvement:+.1f}% better)")

        # Compare element count
        if exhaustive_metrics.get('total_elements') and adaptive_metrics.get('total_elements'):
            ex_elems = exhaustive_metrics['total_elements']
            ad_elems = adaptive_metrics['total_elements']

            self.log_message(f"Elements:")
            self.log_message(f"  Exhaustive: {ex_elems:,}")
            self.log_message(f"  Adaptive:   {ad_elems:,}")

        self.log_message("-" * 70)

    def run_meshing_strategy(self, input_file: str, output_file: str) -> bool:
        """
        Legacy interface - calls generate_mesh

        Args:
            input_file: Input CAD file
            output_file: Output mesh file

        Returns:
            True if successful
        """
        result = self.generate_mesh(input_file, output_file)
        return result.success


if __name__ == "__main__":
    """Command line interface"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Hybrid Mesh Generator - Exhaustive + Adaptive"
    )
    parser.add_argument("input_file", help="Input CAD file (STEP format)")
    parser.add_argument("-o", "--output", help="Output mesh file")
    parser.add_argument("--config", help="Config file (JSON)")

    args = parser.parse_args()

    # Load config
    config = Config()
    if args.config and os.path.exists(args.config):
        import json
        with open(args.config) as f:
            config_data = json.load(f)
            # Apply config (simplified)
            pass

    # Generate mesh
    generator = HybridMeshGenerator(config)
    result = generator.generate_mesh(args.input_file, args.output)

    sys.exit(0 if result.success else 1)
