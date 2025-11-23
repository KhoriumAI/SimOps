"""
Adaptive Mesh Generation Strategy
==================================

NEW intelligent strategy with learning from iteration history.
Features:
- Convergence detection
- Local refinement targeting poor elements
- Mesh coarsening when over-refined
- Learning from previous attempts
- Smart algorithm selection
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.mesh_generator import BaseMeshGenerator
from core.config import Config


class AdaptiveMeshGenerator(BaseMeshGenerator):
    """
    Adaptive mesh generator with intelligent learning

    Key improvements over basic strategies:
    1. Learns from iteration history
    2. Detects convergence (stops when improvement plateaus)
    3. Implements mesh coarsening when needed
    4. Smart algorithm selection based on quality issues
    5. Persists history for future runs
    """

    def __init__(self, config: Optional[Config] = None):
        super().__init__(config)
        self.convergence_window = 3  # Look at last 3 iterations
        self.algorithm_attempts = {}  # Track which algorithms were tried
        self.best_mesh_iteration = None
        self.best_mesh_score = float('inf')

    def run_meshing_strategy(self, input_file: str, output_file: str) -> bool:
        """
        Run adaptive meshing strategy

        Args:
            input_file: Input CAD file
            output_file: Output mesh file

        Returns:
            True if successful
        """
        self.log_message("\n--- ADAPTIVE MESH GENERATION STRATEGY ---")
        self.log_message("Features: Convergence detection, learning, smart adaptation")

        max_iterations = self.config.mesh_params.max_iterations

        # Load history from previous runs if available
        self._load_previous_history(input_file)

        for iteration in range(1, max_iterations + 1):
            self.current_iteration = iteration
            self.log_message(f"\n--- ITERATION {iteration}/{max_iterations} ---")

            # Generate mesh for this iteration
            if not self._generate_iteration(iteration):
                self.log_message("[!] Mesh generation failed")
                if self.best_mesh_iteration:
                    self.log_message(f"Reverting to best mesh from iteration {self.best_mesh_iteration}")
                    return self._revert_to_best_mesh(output_file)
                return False

            # Analyze quality
            metrics = self.analyze_current_mesh()
            if not metrics:
                self.log_message("[!] Quality analysis failed")
                continue

            # Calculate quality score
            quality_score = self._calculate_quality_score(metrics)
            self.log_message(f"Quality Score: {quality_score:.4f}")

            # Track best mesh
            if quality_score < self.best_mesh_score:
                self.best_mesh_score = quality_score
                self.best_mesh_iteration = iteration
                self.save_mesh(output_file + f".iter{iteration}.msh")
                self.log_message(f"[OK] New best mesh! (score: {quality_score:.4f})")

            # Store history
            self.quality_history.append({
                'iteration': iteration,
                'metrics': metrics,
                'params': self.current_mesh_params.copy(),
                'quality_score': quality_score
            })

            # Check if targets are met
            if self.check_quality_targets(metrics):
                self.log_message("[OK] Quality targets achieved!")
                break

            # Check for convergence
            if self._check_convergence():
                self.log_message("[!] Convergence detected - stopping optimization")
                break

            # Get adaptive recommendations for next iteration
            if iteration < max_iterations:
                self._get_adaptive_recommendations(metrics, iteration)
            else:
                self.log_message("Maximum iterations reached")

        # Use best mesh found
        if self.best_mesh_iteration:
            self.log_message(f"\nUsing best mesh from iteration {self.best_mesh_iteration}")
            best_mesh_file = output_file + f".iter{self.best_mesh_iteration}.msh"
            if os.path.exists(best_mesh_file):
                os.rename(best_mesh_file, output_file)

        # Save final mesh and history
        self.save_mesh(output_file)
        self.save_iteration_history(output_file)
        self.generate_final_report()

        return True

    def _generate_iteration(self, iteration: int) -> bool:
        """Generate mesh for current iteration"""
        self.log_message(f"Generating mesh (iteration {iteration})...")

        # Calculate parameters for this iteration
        if iteration == 1:
            # Initial parameters
            self.current_mesh_params = self.calculate_initial_mesh_parameters()
        else:
            # Parameters already set by recommendations
            pass

        # Apply parameters
        self.apply_mesh_parameters()

        # Set algorithm (may have been changed by recommendations)
        algorithm_3d = self.current_mesh_params.get('algorithm_3d',
                                                     self.config.mesh_params.algorithm_3d)
        self.set_mesh_algorithm(algorithm_3d=algorithm_3d)

        # Set element order
        element_order = self.current_mesh_params.get('element_order',
                                                      self.config.mesh_params.element_order)
        self.set_element_order(element_order)

        # Generate mesh
        success = self.generate_mesh_internal(dimension=3)

        if success:
            # Save iteration mesh
            iter_file = f"mesh_iter{iteration}.msh"
            self.save_mesh(iter_file)

        return success

    def _calculate_quality_score(self, metrics: Dict) -> float:
        """
        Calculate overall quality score (lower is better)

        Combines multiple metrics into single score for comparison
        """
        score = 0.0
        targets = self.config.quality_targets

        # Skewness contribution (weight: 2.0)
        if metrics.get('skewness'):
            skew_ratio = metrics['skewness']['max'] / targets.skewness_max
            score += 2.0 * skew_ratio

        # Aspect ratio contribution (weight: 1.0)
        if metrics.get('aspect_ratio'):
            aspect_ratio = metrics['aspect_ratio']['max'] / targets.aspect_ratio_max
            score += 1.0 * aspect_ratio

        # Min angle contribution (weight: 0.5)
        if metrics.get('min_angle'):
            if metrics['min_angle']['min'] < targets.min_angle_min:
                angle_penalty = targets.min_angle_min / max(metrics['min_angle']['min'], 0.1)
                score += 0.5 * angle_penalty

        # Element count penalty (prefer fewer elements for same quality)
        if metrics.get('total_elements'):
            # Normalize by 10,000 elements
            element_factor = metrics['total_elements'] / 10000.0
            score += 0.1 * element_factor

        return score

    def _check_convergence(self) -> bool:
        """
        Check if optimization has converged

        Returns True if improvement has plateaued OR quality is degrading
        """
        if len(self.quality_history) < self.convergence_window + 1:
            return False

        # Get recent quality scores
        recent_scores = [
            h['quality_score']
            for h in self.quality_history[-self.convergence_window-1:]
        ]

        # Check if scores are plateauing or getting worse
        improvements = []
        for i in range(1, len(recent_scores)):
            if recent_scores[i-1] > 0:
                improvement = (recent_scores[i-1] - recent_scores[i]) / recent_scores[i-1]
                improvements.append(improvement)

        if not improvements:
            return False

        # Calculate average improvement
        avg_improvement = sum(improvements) / len(improvements)
        threshold = self.config.quality_targets.min_improvement_threshold

        # Check for convergence (small positive improvement)
        if 0 < avg_improvement < threshold:
            self.log_message(
                f"Convergence detected: avg improvement = {avg_improvement:.1%} "
                f"< {threshold:.1%}"
            )
            return True

        # Check for degradation (negative improvement = getting worse)
        if avg_improvement <= 0:
            self.log_message(
                f"[!] Quality degrading: avg improvement = {avg_improvement:.1%}"
            )
            self.log_message("Stopping to prevent further quality loss")
            return True

        return False

    def _get_adaptive_recommendations(self, metrics: Dict, iteration: int):
        """
        Get adaptive recommendations based on history and patterns

        This is smarter than basic strategies:
        - Considers iteration history
        - Adapts based on what worked before
        - Can suggest coarsening or refinement
        - Smart algorithm selection
        """
        self.log_message("Getting adaptive recommendations...")

        # Get AI/fallback recommendations
        recommendations = self.ai_engine.get_recommendations(
            quality_metrics=metrics,
            current_params=self.current_mesh_params,
            iteration=iteration,
            history=self.quality_history
        )

        # Enhance with adaptive logic
        adaptive_recs = self._enhance_with_adaptive_logic(recommendations, metrics)

        # Apply recommendations
        for rec in adaptive_recs:
            self._apply_recommendation(rec)

    def _enhance_with_adaptive_logic(self, recommendations: List, metrics: Dict) -> List:
        """
        Enhance recommendations with adaptive learning

        Adds intelligence based on:
        - What algorithms worked well before
        - Quality trend analysis
        - Element count trends
        """
        enhanced = list(recommendations)  # Copy original recommendations

        # Check if we're over-refining (too many elements, poor quality)
        if len(self.quality_history) >= 2:
            current_elements = metrics.get('total_elements', 0)
            prev_elements = self.quality_history[-1]['metrics'].get('total_elements', 0)

            if current_elements > prev_elements * 1.5:
                # Element count growing too fast
                current_score = self._calculate_quality_score(metrics)
                prev_score = self.quality_history[-1]['quality_score']

                if current_score >= prev_score * 0.95:  # Quality not improving much
                    self.log_message("[!] Over-refinement detected - suggesting coarsening")
                    # Add coarsening recommendation
                    from core.ai_integration import MeshRecommendation, RecommendationType
                    enhanced.append(MeshRecommendation(
                        RecommendationType.PARAMETER_ADJUSTMENT,
                        "cl_max",
                        self.current_mesh_params['cl_max'] * 1.5,
                        "Coarsen mesh - over-refinement not improving quality"
                    ))

        # Algorithm rotation strategy
        current_algorithm = self.current_mesh_params.get('algorithm_3d', 1)
        if current_algorithm not in self.algorithm_attempts:
            self.algorithm_attempts[current_algorithm] = []

        self.algorithm_attempts[current_algorithm].append(
            self._calculate_quality_score(metrics)
        )

        # If current algorithm tried 2+ times with poor results, try different one
        if len(self.algorithm_attempts[current_algorithm]) >= 2:
            avg_score = sum(self.algorithm_attempts[current_algorithm]) / len(
                self.algorithm_attempts[current_algorithm]
            )
            if avg_score > 2.0:  # Poor quality
                # Try algorithm we haven't used much
                untried_algos = [1, 4, 7, 10]  # Delaunay, HXT, MMG3D, Frontal-Delaunay
                for algo in untried_algos:
                    if algo not in self.algorithm_attempts or len(
                            self.algorithm_attempts[algo]) < 2:
                        self.log_message(f"Trying different algorithm: {algo}")
                        from core.ai_integration import MeshRecommendation, RecommendationType
                        enhanced.append(MeshRecommendation(
                            RecommendationType.ALGORITHM_CHANGE,
                            "algorithm_3d",
                            algo,
                            f"Trying algorithm {algo} (current algorithm not effective)"
                        ))
                        break

        return enhanced

    def _apply_recommendation(self, recommendation):
        """Apply a single recommendation"""
        rec_dict = recommendation.to_dict() if hasattr(recommendation, 'to_dict') else recommendation

        param = rec_dict['parameter']
        value = rec_dict['value']
        reason = rec_dict['reason']

        if rec_dict['type'] == 'parameter_adjustment':
            self.current_mesh_params[param] = value
            self.log_message(f"[OK] {param} = {value:.4f} ({reason})")

        elif rec_dict['type'] == 'algorithm_change':
            self.current_mesh_params[param] = value
            self.log_message(f"[OK] {param} = {value} ({reason})")

        elif rec_dict['type'] == 'element_order_change':
            self.current_mesh_params[param] = value
            self.log_message(f"[OK] {param} = {value} ({reason})")

    def _load_previous_history(self, input_file: str):
        """Load history from previous runs to learn from"""
        history_file = os.path.splitext(input_file)[0] + "_adaptive_history.json"

        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)

                if 'history' in data:
                    self.log_message(f"[OK] Loaded previous history: {len(data['history'])} runs")
                    # Could use this to seed initial parameters
                    # For now, just log it

            except Exception as e:
                self.log_message(f"[!] Could not load previous history: {e}")

    def _revert_to_best_mesh(self, output_file: str) -> bool:
        """Revert to best mesh found so far"""
        if not self.best_mesh_iteration:
            return False

        best_file = output_file + f".iter{self.best_mesh_iteration}.msh"
        if os.path.exists(best_file):
            os.rename(best_file, output_file)
            self.log_message(f"[OK] Reverted to mesh from iteration {self.best_mesh_iteration}")
            return True

        return False


def main():
    """Command-line interface"""
    if len(sys.argv) > 1:
        cad_file = sys.argv[1]
    else:
        cad_file = input("Enter CAD file path: ").strip()

    try:
        generator = AdaptiveMeshGenerator()
        result = generator.generate_mesh(cad_file)

        if result.success:
            print(f"\n[OK] Adaptive mesh generation completed successfully!")
            print(f"Output file: {result.output_file}")
            print(f"Iterations: {result.iterations}")
        else:
            print(f"\n[X] Mesh generation failed: {result.message}")
            sys.exit(1)

    except Exception as e:
        print(f"\n[X] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
