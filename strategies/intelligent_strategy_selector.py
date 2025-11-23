"""
Intelligent Strategy Selection for Exhaustive Meshing
======================================================

Learns from failures to skip similar strategies and predict success.

Key Ideas:
1. **Failure Pattern Detection**: If 2-3 related strategies fail similarly, skip the rest
2. **Geometry Classification**: Categorize geometry and use historical success rates
3. **Error Analysis**: Different error types -> different strategy recommendations
4. **Adaptive Ordering**: Reorder strategies based on geometry features

Example:
    If "tet_delaunay" and "tet_frontal" both fail with:
    - "No mesh generated" -> Skip all tet strategies, try hybrid
    - "Poor quality" -> Continue with tets but add optimization
    - "Out of memory" -> Skip fine meshes, use coarse variations
"""

from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from enum import Enum
import re


class FailureType(Enum):
    """Types of meshing failures"""
    NO_MESH_GENERATED = "no_mesh"
    POOR_QUALITY = "poor_quality"
    OUT_OF_MEMORY = "out_of_memory"
    TIMEOUT = "timeout"
    GEOMETRY_ERROR = "geometry_error"
    ALGORITHM_ERROR = "algorithm_error"
    INVERTED_ELEMENTS = "inverted_elements"
    UNKNOWN = "unknown"


class StrategyFamily(Enum):
    """Groups of related strategies"""
    TET_DELAUNAY = "tet_delaunay"
    TET_FRONTAL = "tet_frontal"
    TET_HXT = "tet_hxt"
    TET_MMG3D = "tet_mmg3d"
    HYBRID = "hybrid"
    STRUCTURED = "structured"
    BOUNDARY_LAYER = "boundary_layer"
    ANISOTROPIC = "anisotropic"
    COARSE = "coarse"
    LINEAR = "linear"
    SUPPLEMENTARY = "supplementary"


@dataclass
class StrategyAttempt:
    """Record of a strategy attempt"""
    name: str
    family: StrategyFamily
    success: bool
    failure_type: Optional[FailureType]
    score: float
    metrics: Optional[Dict]
    error_message: Optional[str]


class GeometryProfile:
    """Characterizes geometry to predict best strategies"""

    def __init__(self, geometry_info: Dict):
        """
        Extract geometry features for strategy selection

        Args:
            geometry_info: From BaseMeshGenerator._extract_geometry_info()
        """
        self.volumes = geometry_info.get('volumes', 0)
        self.surfaces = geometry_info.get('surfaces', 0)
        self.curves = geometry_info.get('curves', 0)
        self.diagonal = geometry_info.get('diagonal', 1.0)
        self.volume = geometry_info.get('volume', 0.0)

        # Derived features
        self.complexity = self._compute_complexity()
        self.is_simple = self.complexity < 100
        self.is_complex = self.complexity > 500
        self.is_very_complex = self.complexity > 2000

    def _compute_complexity(self) -> float:
        """
        Estimate geometry complexity

        Returns:
            Complexity score (0-10000+)
        """
        # Weighted sum of entities
        return self.surfaces * 2 + self.curves * 0.5 + self.volumes * 10

    def predict_good_families(self) -> List[StrategyFamily]:
        """
        Predict which strategy families likely to succeed

        Returns:
            Ordered list of strategy families (best first)
        """
        families = []

        if self.is_simple:
            # Simple geometry: Standard algorithms work well
            families = [
                StrategyFamily.TET_DELAUNAY,
                StrategyFamily.TET_FRONTAL,
                StrategyFamily.TET_HXT,
                StrategyFamily.STRUCTURED,
            ]
        elif self.is_complex:
            # Complex geometry: Need robust algorithms
            families = [
                StrategyFamily.TET_MMG3D,
                StrategyFamily.TET_HXT,
                StrategyFamily.ANISOTROPIC,
                StrategyFamily.TET_FRONTAL,
                StrategyFamily.BOUNDARY_LAYER,
            ]
        elif self.is_very_complex:
            # Very complex: Use advanced techniques
            families = [
                StrategyFamily.TET_MMG3D,
                StrategyFamily.ANISOTROPIC,
                StrategyFamily.COARSE,
                StrategyFamily.LINEAR,
                StrategyFamily.SUPPLEMENTARY,
            ]
        else:
            # Moderate complexity: Try standard first, then robust
            families = [
                StrategyFamily.TET_DELAUNAY,
                StrategyFamily.TET_MMG3D,
                StrategyFamily.TET_HXT,
                StrategyFamily.TET_FRONTAL,
                StrategyFamily.ANISOTROPIC,
            ]

        return families


class IntelligentStrategySelector:
    """
    Intelligently selects strategies to try based on failure patterns
    """

    def __init__(self, geometry_profile: GeometryProfile):
        """
        Initialize selector

        Args:
            geometry_profile: Geometry characteristics
        """
        self.geometry = geometry_profile
        self.attempts: List[StrategyAttempt] = []
        self.skipped_families: Set[StrategyFamily] = set()

        # Strategy family mapping
        self.strategy_to_family = {
            "tet_delaunay_optimized": StrategyFamily.TET_DELAUNAY,
            "tet_frontal_optimized": StrategyFamily.TET_FRONTAL,
            "tet_hxt_optimized": StrategyFamily.TET_HXT,
            "tet_mmg3d_optimized": StrategyFamily.TET_MMG3D,
            "tet_with_boundary_layers": StrategyFamily.BOUNDARY_LAYER,
            "anisotropic_curvature": StrategyFamily.ANISOTROPIC,
            "hybrid_prism_tet": StrategyFamily.HYBRID,
            "hybrid_hex_tet": StrategyFamily.HYBRID,
            "recombined_to_hex": StrategyFamily.STRUCTURED,
            "transfinite_structured": StrategyFamily.STRUCTURED,
            "very_coarse_tet": StrategyFamily.COARSE,
            "adaptive_coarse_to_fine": StrategyFamily.COARSE,
            "linear_tet_delaunay": StrategyFamily.LINEAR,
            "linear_tet_frontal": StrategyFamily.LINEAR,
            "subdivide_and_mesh": StrategyFamily.COARSE,
            "automatic_gmsh_default": StrategyFamily.LINEAR,
            "tetgen_fallback": StrategyFamily.SUPPLEMENTARY,
            "pymesh_repair_fallback": StrategyFamily.SUPPLEMENTARY,
        }

    def classify_failure(self, error_message: str, metrics: Optional[Dict]) -> FailureType:
        """
        Classify failure type from error message and metrics

        Args:
            error_message: Error message from exception
            metrics: Quality metrics (if mesh was generated)

        Returns:
            FailureType classification
        """
        if not error_message:
            error_message = ""

        error_lower = error_message.lower()

        # Check for specific error patterns
        if "no mesh" in error_lower or "no elements" in error_lower:
            return FailureType.NO_MESH_GENERATED

        if "memory" in error_lower or "malloc" in error_lower:
            return FailureType.OUT_OF_MEMORY

        if "timeout" in error_lower or "timed out" in error_lower:
            return FailureType.TIMEOUT

        if "geometry" in error_lower or "topology" in error_lower:
            return FailureType.GEOMETRY_ERROR

        if "algorithm" in error_lower or "convergence" in error_lower:
            return FailureType.ALGORITHM_ERROR

        # Check metrics for quality issues
        if metrics:
            sicn_min = metrics.get('sicn', {}).get('min', 1.0)
            if sicn_min < 0:
                return FailureType.INVERTED_ELEMENTS
            elif sicn_min < 0.3:
                return FailureType.POOR_QUALITY

        return FailureType.UNKNOWN

    def record_attempt(self,
                       name: str,
                       success: bool,
                       metrics: Optional[Dict] = None,
                       score: float = float('inf'),
                       error_message: str = "") -> None:
        """
        Record a strategy attempt

        Args:
            name: Strategy name
            success: Whether it succeeded
            metrics: Quality metrics
            score: Quality score (lower is better)
            error_message: Error message if failed
        """
        family = self.strategy_to_family.get(name, StrategyFamily.LINEAR)

        failure_type = None if success else self.classify_failure(error_message, metrics)

        attempt = StrategyAttempt(
            name=name,
            family=family,
            success=success,
            failure_type=failure_type,
            score=score,
            metrics=metrics,
            error_message=error_message
        )

        self.attempts.append(attempt)

        # Update skipped families based on failure patterns
        self._update_skip_list()

    def _update_skip_list(self) -> None:
        """
        Analyze failure patterns and update skip list

        Rules:
        1. If 2+ strategies from same family fail with NO_MESH -> skip family
        2. If 3+ strategies fail with GEOMETRY_ERROR -> skip complex strategies
        3. If 2+ fail with OUT_OF_MEMORY -> skip fine meshes, only use coarse
        4. If all tets fail -> try supplementary meshers
        """
        if len(self.attempts) < 2:
            return

        # Count failures by family
        family_failures = {}
        for attempt in self.attempts:
            if not attempt.success and attempt.failure_type:
                family = attempt.family
                failure = attempt.failure_type

                if family not in family_failures:
                    family_failures[family] = {}

                if failure not in family_failures[family]:
                    family_failures[family][failure] = 0

                family_failures[family][failure] += 1

        # Rule 1: Skip family if 2+ NO_MESH failures
        for family, failures in family_failures.items():
            no_mesh_count = failures.get(FailureType.NO_MESH_GENERATED, 0)
            if no_mesh_count >= 2:
                self.skipped_families.add(family)

        # Rule 2: Skip structured if geometry errors
        geometry_errors = sum(
            1 for a in self.attempts
            if not a.success and a.failure_type == FailureType.GEOMETRY_ERROR
        )
        if geometry_errors >= 3:
            self.skipped_families.add(StrategyFamily.STRUCTURED)
            self.skipped_families.add(StrategyFamily.HYBRID)

        # Rule 3: Skip fine if memory errors
        memory_errors = sum(
            1 for a in self.attempts
            if not a.success and a.failure_type == FailureType.OUT_OF_MEMORY
        )
        if memory_errors >= 2:
            # Don't skip families, but caller should filter by coarseness
            pass

        # Rule 4: If all basic tets failed, prioritize supplementary
        tet_families = [
            StrategyFamily.TET_DELAUNAY,
            StrategyFamily.TET_FRONTAL,
            StrategyFamily.TET_HXT
        ]
        tet_attempts = [a for a in self.attempts if a.family in tet_families]
        if len(tet_attempts) >= 3 and all(not a.success for a in tet_attempts):
            # Don't skip, but caller should prioritize supplementary
            pass

    def should_skip_strategy(self, strategy_name: str) -> Tuple[bool, str]:
        """
        Determine if a strategy should be skipped

        Args:
            strategy_name: Name of strategy to check

        Returns:
            (should_skip, reason)
        """
        family = self.strategy_to_family.get(strategy_name, StrategyFamily.LINEAR)

        if family in self.skipped_families:
            return True, f"Family {family.value} has multiple failures"

        # Check for repeated similar failures
        similar_attempts = [a for a in self.attempts if a.family == family]
        if len(similar_attempts) >= 3:
            failures = [a for a in similar_attempts if not a.success]
            if len(failures) >= 3:
                return True, f"3+ failures from family {family.value}"

        return False, ""

    def get_recommended_order(self, available_strategies: List[str]) -> List[str]:
        """
        Reorder strategies based on geometry and failure patterns

        Args:
            available_strategies: All strategies that could be tried

        Returns:
            Reordered list (best strategies first)
        """
        # Get predicted good families for this geometry
        good_families = self.geometry.predict_good_families()

        # Score each strategy
        scored_strategies = []
        for strategy in available_strategies:
            family = self.strategy_to_family.get(strategy, StrategyFamily.LINEAR)

            # Base score from family prediction
            if family in good_families:
                base_score = good_families.index(family)
            else:
                base_score = len(good_families)

            # Penalty if family is skipped
            if family in self.skipped_families:
                base_score += 1000  # Push to end

            # Penalty for repeated failures in this family
            family_attempts = [a for a in self.attempts if a.family == family]
            failure_count = sum(1 for a in family_attempts if not a.success)
            base_score += failure_count * 2

            scored_strategies.append((strategy, base_score))

        # Sort by score (lower is better)
        scored_strategies.sort(key=lambda x: x[1])

        return [s[0] for s in scored_strategies]

    def get_summary(self) -> str:
        """
        Get summary of attempts and skips

        Returns:
            Human-readable summary
        """
        lines = []
        lines.append("="*60)
        lines.append("INTELLIGENT STRATEGY SELECTOR SUMMARY")
        lines.append("="*60)

        lines.append(f"\nGeometry Profile:")
        lines.append(f"  Complexity: {self.geometry.complexity:.0f}")
        lines.append(f"  Classification: {'Simple' if self.geometry.is_simple else 'Complex' if self.geometry.is_complex else 'Very Complex' if self.geometry.is_very_complex else 'Moderate'}")

        lines.append(f"\nAttempts: {len(self.attempts)}")
        successes = sum(1 for a in self.attempts if a.success)
        lines.append(f"  Successes: {successes}")
        lines.append(f"  Failures: {len(self.attempts) - successes}")

        if self.skipped_families:
            lines.append(f"\nSkipped Families: {len(self.skipped_families)}")
            for family in self.skipped_families:
                lines.append(f"  - {family.value}")

        # Failure breakdown
        failure_counts = {}
        for attempt in self.attempts:
            if not attempt.success and attempt.failure_type:
                ft = attempt.failure_type
                failure_counts[ft] = failure_counts.get(ft, 0) + 1

        if failure_counts:
            lines.append(f"\nFailure Types:")
            for ft, count in failure_counts.items():
                lines.append(f"  - {ft.value}: {count}")

        lines.append("="*60)

        return "\n".join(lines)


def test_selector():
    """Test the intelligent selector"""
    # Mock geometry
    geometry_info = {
        'volumes': 1,
        'surfaces': 150,
        'curves': 300,
        'diagonal': 100.0,
        'volume': 1000.0
    }

    profile = GeometryProfile(geometry_info)
    selector = IntelligentStrategySelector(profile)

    # Simulate some failures
    selector.record_attempt("tet_delaunay_optimized", False, error_message="No mesh generated")
    selector.record_attempt("tet_frontal_optimized", False, error_message="No mesh generated")
    selector.record_attempt("tet_hxt_optimized", True, score=0.5)

    # Check if similar strategies should be skipped
    should_skip, reason = selector.should_skip_strategy("linear_tet_delaunay")
    print(f"Skip linear_tet_delaunay? {should_skip} - {reason}")

    # Get recommended order
    all_strategies = [
        "tet_delaunay_optimized",
        "tet_mmg3d_optimized",
        "anisotropic_curvature",
        "linear_tet_frontal",
    ]

    ordered = selector.get_recommended_order(all_strategies)
    print(f"\nRecommended order: {ordered}")

    print(f"\n{selector.get_summary()}")


if __name__ == "__main__":
    test_selector()
