"""
Mesh Optimization Configuration Module
=======================================

Centralized configuration for all mesh optimization features:
- Quality constraints and thresholds
- Iterative refinement settings
- Adaptive sizing parameters
- Material properties for ANSYS export
- GPU optimization settings

Usage:
    from core.optimization_config import OptimizationConfig, QualityPreset

    # Use preset
    config = OptimizationConfig.from_preset(QualityPreset.HIGH)

    # Or customize
    config = OptimizationConfig(
        sicn_threshold=0.4,
        max_refinement_iterations=5
    )
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional


class QualityPreset(Enum):
    """Predefined quality presets"""
    FAST = "fast"           # Fast meshing, acceptable quality
    MEDIUM = "medium"       # Balanced speed/quality
    HIGH = "high"           # Slow meshing, excellent quality
    PRODUCTION = "production"  # Maximum quality for final analysis


@dataclass
class QualityConstraints:
    """Global mesh quality constraints"""

    # SICN quality thresholds
    sicn_target: float = 0.3        # Target minimum SICN
    sicn_excellent: float = 0.5     # Excellent quality threshold
    sicn_acceptable: float = 0.2    # Minimum acceptable quality

    # Aspect ratio limits
    max_aspect_ratio: float = 1e6   # Maximum element aspect ratio
    target_aspect_ratio: float = 5.0  # Target aspect ratio for optimization

    # Size gradation
    size_gradation_factor: float = 1.3  # Max size change between neighbors (1.3 = 30%)
    min_size_factor: float = 0.1     # Min size as fraction of max size
    max_size_factor: float = 2.0     # Max size as fraction of target size

    # Angle tolerances
    facet_overlap_tolerance: float = 0.1  # Angle tolerance for facet overlap

    # Optimization thresholds
    optimize_threshold: float = 0.3   # Only optimize elements below this SICN


@dataclass
class IterativeRefinementConfig:
    """Configuration for iterative quality-based refinement"""

    enabled: bool = True
    max_iterations: int = 3           # Maximum refinement passes
    worst_element_percent: float = 0.1  # Refine worst 10% of elements

    # Refinement factors based on quality severity
    refinement_factor_inverted: float = 4.0   # For SICN < 0 (inverted)
    refinement_factor_poor: float = 3.0       # For SICN < 0.1
    refinement_factor_bad: float = 2.5        # For SICN < 0.2
    refinement_factor_acceptable: float = 2.0  # For SICN < 0.3

    # Clustering parameters
    cluster_distance_mm: float = 50.0  # Spatial clustering distance
    max_clusters: int = 20             # Limit to avoid field explosion


@dataclass
class AdaptiveSizingConfig:
    """Configuration for intelligent adaptive sizing"""

    enabled: bool = True

    # Feature detection thresholds
    complex_geometry_threshold: int = 50    # Small curves to trigger adaptive sizing
    broken_geometry_threshold: int = 500    # Extremely complex geometry

    # Sizing parameters
    min_layers_across_feature: int = 3      # Minimum element layers across small features
    gradient_factor: float = 3.0            # Size transition over 3x feature size
    max_gradient: float = 1.5               # Max size increase between elements

    # Field limits (prevent explosion)
    max_curve_fields: int = 100
    max_surface_fields: int = 50
    max_curvature_fields: int = 20


@dataclass
class GPUOptimizationConfig:
    """Configuration for GPU-accelerated mesh optimization"""

    enabled: bool = False           # Disabled by default (requires CuPy)
    max_iterations: int = 10        # Vertex relocation iterations
    step_size: float = 0.1          # Optimization step size
    quality_threshold: float = 0.3  # Stop if quality exceeds this


@dataclass
class AdvancedFeaturesConfig:
    """
    Configuration for advanced geometry handling features

    Implements production-grade techniques for challenging geometries:
    - Virtual topology for sliver faces (fillets/lofts)
    - Metric-driven refinement for small features
    - Boundary layer inflation for thin channels
    """

    # Feature 1: Virtual Topology
    enable_virtual_topology: bool = True
    sliver_aspect_threshold: float = 20.0      # Aspect ratio to flag as sliver

    # Feature 2: Metric-Driven Refinement
    enable_adaptive_refinement: bool = True
    small_feature_threshold: float = 1.0       # mm - features smaller than this get refined
    gradient_factor: float = 3.0               # Distance to grade size (in multiples of feature size)
    min_elements_across_feature: int = 5       # Minimum elements to resolve a small feature

    # Feature 3: Boundary Layers
    enable_boundary_layers: bool = False       # Disabled by default (advanced feature)
    thin_channel_threshold: float = 5.0        # mm - gaps smaller than this get boundary layers
    boundary_layer_first_height: Optional[float] = None  # Auto-calculate if None
    boundary_layer_growth_rate: float = 1.2    # Layer thickness growth (1.2 = 20%)
    boundary_layer_count: int = 5              # Number of boundary layers

    # General advanced settings
    auto_detect_features: bool = True          # Automatically detect problematic features
    verbose_analysis: bool = True              # Print detailed geometry analysis


@dataclass
class MaterialProperties:
    """Material properties for ANSYS export"""

    name: str = "Structural Steel"
    youngs_modulus: float = 2.0e11       # Pa
    poisson_ratio: float = 0.3
    density: float = 7850                # kg/m³
    thermal_expansion: Optional[float] = 1.2e-5  # 1/K
    thermal_conductivity: Optional[float] = 60.5  # W/(m·K)

    @classmethod
    def steel(cls):
        return cls()

    @classmethod
    def aluminum(cls):
        return cls(
            name="Aluminum Alloy",
            youngs_modulus=7.1e10,
            poisson_ratio=0.33,
            density=2770,
            thermal_expansion=2.4e-5,
            thermal_conductivity=170
        )

    @classmethod
    def titanium(cls):
        return cls(
            name="Titanium Alloy (Ti-6Al-4V)",
            youngs_modulus=1.14e11,
            poisson_ratio=0.34,
            density=4430,
            thermal_expansion=8.6e-6,
            thermal_conductivity=6.7
        )

    @classmethod
    def copper(cls):
        return cls(
            name="Copper",
            youngs_modulus=1.3e11,
            poisson_ratio=0.34,
            density=8960,
            thermal_expansion=1.65e-5,
            thermal_conductivity=401
        )

    @classmethod
    def inconel(cls):
        return cls(
            name="Inconel 718 (Superalloy)",
            youngs_modulus=2.0e11,
            poisson_ratio=0.29,
            density=8190,
            thermal_expansion=1.3e-5,
            thermal_conductivity=11.4
        )


@dataclass
class OptimizationConfig:
    """
    Master configuration for all mesh optimization features

    Combines all optimization settings into a single configuration object.
    Can be created from presets or customized per-parameter.
    """

    quality: QualityConstraints = field(default_factory=QualityConstraints)
    iterative_refinement: IterativeRefinementConfig = field(default_factory=IterativeRefinementConfig)
    adaptive_sizing: AdaptiveSizingConfig = field(default_factory=AdaptiveSizingConfig)
    gpu_optimization: GPUOptimizationConfig = field(default_factory=GPUOptimizationConfig)
    advanced_features: AdvancedFeaturesConfig = field(default_factory=AdvancedFeaturesConfig)
    material: MaterialProperties = field(default_factory=MaterialProperties.steel)

    # General settings
    verbose: bool = True
    parallel_enabled: bool = True
    num_workers: int = 4

    @classmethod
    def from_preset(cls, preset: QualityPreset) -> 'OptimizationConfig':
        """
        Create configuration from a quality preset

        Args:
            preset: QualityPreset enum value

        Returns:
            OptimizationConfig with preset values
        """
        config = cls()

        if preset == QualityPreset.FAST:
            # Fast meshing: minimal optimization
            config.quality.sicn_target = 0.2
            config.iterative_refinement.enabled = False
            config.adaptive_sizing.enabled = False
            config.gpu_optimization.enabled = False

        elif preset == QualityPreset.MEDIUM:
            # Balanced: use defaults
            pass  # Defaults are already "medium" quality

        elif preset == QualityPreset.HIGH:
            # High quality: aggressive optimization
            config.quality.sicn_target = 0.4
            config.quality.max_aspect_ratio = 1e5  # Stricter aspect ratio
            config.iterative_refinement.max_iterations = 5
            config.adaptive_sizing.min_layers_across_feature = 4

        elif preset == QualityPreset.PRODUCTION:
            # Maximum quality: everything enabled
            config.quality.sicn_target = 0.5
            config.quality.max_aspect_ratio = 1e4
            config.quality.size_gradation_factor = 1.2  # Smoother transitions
            config.iterative_refinement.max_iterations = 7
            config.iterative_refinement.refinement_factor_poor = 4.0
            config.adaptive_sizing.min_layers_across_feature = 5
            config.adaptive_sizing.gradient_factor = 2.5  # Smoother size transitions
            config.gpu_optimization.enabled = True  # Enable GPU if available

            # Enable all advanced features for production
            config.advanced_features.enable_virtual_topology = True
            config.advanced_features.enable_adaptive_refinement = True
            config.advanced_features.enable_boundary_layers = True  # For CFD/flow analysis
            config.advanced_features.small_feature_threshold = 0.5  # Stricter (0.5mm)
            config.advanced_features.min_elements_across_feature = 7  # More elements

        return config

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary (for JSON export)"""
        return {
            'quality': {
                'sicn_target': self.quality.sicn_target,
                'max_aspect_ratio': self.quality.max_aspect_ratio,
                'size_gradation_factor': self.quality.size_gradation_factor,
            },
            'iterative_refinement': {
                'enabled': self.iterative_refinement.enabled,
                'max_iterations': self.iterative_refinement.max_iterations,
            },
            'adaptive_sizing': {
                'enabled': self.adaptive_sizing.enabled,
                'min_layers_across_feature': self.adaptive_sizing.min_layers_across_feature,
            },
            'gpu_optimization': {
                'enabled': self.gpu_optimization.enabled,
            },
            'material': {
                'name': self.material.name,
                'youngs_modulus': self.material.youngs_modulus,
                'density': self.material.density,
            },
            'general': {
                'verbose': self.verbose,
                'parallel_enabled': self.parallel_enabled,
            }
        }

    def print_summary(self):
        """Print human-readable configuration summary"""
        print("=" * 70)
        print("MESH OPTIMIZATION CONFIGURATION")
        print("=" * 70)

        print("\nQuality Constraints:")
        print(f"  Target SICN: >= {self.quality.sicn_target}")
        print(f"  Max aspect ratio: {self.quality.max_aspect_ratio:.0e}")
        print(f"  Size gradation: {self.quality.size_gradation_factor}x max")

        print("\nIterative Refinement:")
        print(f"  Enabled: {self.iterative_refinement.enabled}")
        if self.iterative_refinement.enabled:
            print(f"  Max iterations: {self.iterative_refinement.max_iterations}")
            print(f"  Worst element %: {self.iterative_refinement.worst_element_percent * 100:.0f}%")

        print("\nAdaptive Sizing:")
        print(f"  Enabled: {self.adaptive_sizing.enabled}")
        if self.adaptive_sizing.enabled:
            print(f"  Min layers across features: {self.adaptive_sizing.min_layers_across_feature}")
            print(f"  Gradient factor: {self.adaptive_sizing.gradient_factor}x")

        print("\nGPU Optimization:")
        print(f"  Enabled: {self.gpu_optimization.enabled}")
        if self.gpu_optimization.enabled:
            print(f"  Max iterations: {self.gpu_optimization.max_iterations}")

        print("\nAdvanced Features:")
        print(f"  Virtual topology (sliver faces): {self.advanced_features.enable_virtual_topology}")
        if self.advanced_features.enable_virtual_topology:
            print(f"    Sliver aspect threshold: {self.advanced_features.sliver_aspect_threshold}")
        print(f"  Metric-driven refinement: {self.advanced_features.enable_adaptive_refinement}")
        if self.advanced_features.enable_adaptive_refinement:
            print(f"    Small feature threshold: {self.advanced_features.small_feature_threshold} mm")
            print(f"    Min elements across feature: {self.advanced_features.min_elements_across_feature}")
        print(f"  Boundary layer inflation: {self.advanced_features.enable_boundary_layers}")
        if self.advanced_features.enable_boundary_layers:
            print(f"    Thin channel threshold: {self.advanced_features.thin_channel_threshold} mm")
            print(f"    Growth rate: {self.advanced_features.boundary_layer_growth_rate}")

        print("\nMaterial Properties:")
        print(f"  Material: {self.material.name}")
        print(f"  Young's Modulus: {self.material.youngs_modulus:.2e} Pa")
        print(f"  Density: {self.material.density:.1f} kg/m³")

        print("=" * 70)


# Example usage
if __name__ == "__main__":
    # Test presets
    print("FAST Preset:")
    fast_config = OptimizationConfig.from_preset(QualityPreset.FAST)
    fast_config.print_summary()

    print("\n" * 2)

    print("PRODUCTION Preset:")
    prod_config = OptimizationConfig.from_preset(QualityPreset.PRODUCTION)
    prod_config.print_summary()

    # Test material properties
    print("\n" * 2)
    print("Material Library:")
    for material_name, material_cls in [
        ("Steel", MaterialProperties.steel),
        ("Aluminum", MaterialProperties.aluminum),
        ("Titanium", MaterialProperties.titanium),
        ("Inconel", MaterialProperties.inconel)
    ]:
        mat = material_cls()
        print(f"\n{mat.name}:")
        print(f"  E = {mat.youngs_modulus:.2e} Pa ({mat.youngs_modulus/1e9:.0f} GPa)")
        print(f"  ρ = {mat.density:.0f} kg/m³")
