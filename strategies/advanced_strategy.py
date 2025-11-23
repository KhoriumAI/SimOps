"""
Advanced Meshing Strategy
=========================

Production-grade meshing strategy integrating:
1. Virtual topology for sliver faces
2. Metric-driven adaptive refinement for small features
3. Boundary layer inflation for thin channels

This strategy applies all advanced techniques automatically based on
geometry analysis and configuration settings.
"""

import gmsh
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Tuple

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.mesh_generator import BaseMeshGenerator, MeshGenerationResult
from core.optimization_config import OptimizationConfig, QualityPreset
from core.advanced_geometry import (
    apply_advanced_meshing,
    AdvancedGeometryAnalyzer,
    VirtualTopologyManager,
    AdaptiveRefinementEngine,
    BoundaryLayerGenerator
)


class AdvancedMeshingStrategy(BaseMeshGenerator):
    """
    Advanced meshing strategy with production-grade features

    Workflow:
    1. Load CAD geometry
    2. Analyze for problematic features
    3. Apply virtual topology for sliver faces
    4. Create adaptive size fields for small features
    5. Generate boundary layers for thin channels
    6. Generate high-quality mesh
    7. Post-process and export

    Usage:
        from strategies.advanced_strategy import AdvancedMeshingStrategy
        from core.optimization_config import OptimizationConfig, QualityPreset

        config = OptimizationConfig.from_preset(QualityPreset.PRODUCTION)
        strategy = AdvancedMeshingStrategy(config=config)

        result = strategy.generate_mesh(
            "input.step",
            "output.msh",
            global_mesh_size=5.0
        )
    """

    def __init__(self, config: Optional[OptimizationConfig] = None, use_gpu: bool = True):
        """
        Initialize advanced meshing strategy

        Args:
            config: Optimization configuration (uses defaults if None)
            use_gpu: Enable GPU acceleration for quality analysis
        """
        super().__init__(config, use_gpu)

        self.log_message("=" * 70)
        self.log_message("ADVANCED MESHING STRATEGY")
        self.log_message("=" * 70)
        self.log_message("Features:")
        self.log_message(f"  * Virtual topology: {self.config.advanced_features.enable_virtual_topology}")
        self.log_message(f"  * Adaptive refinement: {self.config.advanced_features.enable_adaptive_refinement}")
        self.log_message(f"  * Boundary layers: {self.config.advanced_features.enable_boundary_layers}")
        self.log_message("=" * 70)

        # Advanced geometry analysis results
        self.geometry_analysis: Optional[Dict] = None
        self.advanced_meshing_results: Optional[Dict] = None

    def _strategy_name(self) -> str:
        """Return strategy name for logging"""
        return "Advanced Production Meshing"

    def generate_mesh(self,
                     input_file: str,
                     output_file: Optional[str] = None,
                     global_mesh_size: float = 5.0,
                     **kwargs) -> MeshGenerationResult:
        """
        Generate mesh using advanced techniques

        Args:
            input_file: Path to CAD file (STEP/STP)
            output_file: Output mesh file path
            global_mesh_size: Global element size (mm)
            **kwargs: Additional parameters

        Returns:
            MeshGenerationResult with quality metrics
        """
        start_time = time.time()

        self.log_message(f"\n{'='*70}")
        self.log_message(f"Starting {self._strategy_name()}")
        self.log_message(f"Input: {input_file}")
        self.log_message(f"Global mesh size: {global_mesh_size} mm")
        self.log_message(f"{'='*70}\n")

        # Print configuration summary
        if self.config.verbose:
            self.config.print_summary()

        try:
            # Step 1: Initialize Gmsh
            self.log_message("[Step 1/7] Initializing Gmsh...")
            gmsh.initialize()
            gmsh.model.add("advanced_mesh")
            gmsh.option.setNumber("General.Terminal", 1)
            # FIX: Force single-threaded mode to prevent threading conflicts
            gmsh.option.setNumber("General.NumThreads", 1)
            self.gmsh_initialized = True

            # Step 2: Load CAD geometry
            self.log_message("[Step 2/7] Loading CAD geometry...")
            gmsh.model.occ.importShapes(input_file)
            gmsh.model.occ.synchronize()

            # FIX: Heal geometry to prevent infinite loop bugs in edge recovery
            try:
                gmsh.model.occ.healShapes()
                gmsh.model.occ.synchronize()
                self.log_message("  [OK] Geometry healed")
            except Exception as e:
                self.log_message(f"  Warning: Geometry healing failed: {e}")

            self.model_loaded = True

            entities = gmsh.model.getEntities(dim=3)
            if not entities:
                raise ValueError("No 3D volumes found in CAD file")

            self.log_message(f"  [OK] Loaded {len(entities)} volume(s)")

            # Step 3: Advanced geometry analysis
            if self.config.advanced_features.auto_detect_features:
                self.log_message("[Step 3/7] Analyzing geometry for advanced features...")

                self.advanced_meshing_results = apply_advanced_meshing(
                    sliver_threshold=self.config.advanced_features.sliver_aspect_threshold,
                    feature_threshold=self.config.advanced_features.small_feature_threshold,
                    channel_threshold=self.config.advanced_features.thin_channel_threshold,
                    global_mesh_size=global_mesh_size,
                    enable_virtual_topology=self.config.advanced_features.enable_virtual_topology,
                    enable_adaptive_refinement=self.config.advanced_features.enable_adaptive_refinement,
                    enable_boundary_layers=self.config.advanced_features.enable_boundary_layers
                )

                self.geometry_analysis = self.advanced_meshing_results['analysis']
            else:
                self.log_message("[Step 3/7] Skipping advanced analysis (auto_detect_features=False)")

            # Step 4: Set global mesh size
            self.log_message("[Step 4/7] Configuring mesh parameters...")
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", global_mesh_size * 0.1)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", global_mesh_size * 2.0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 10)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)

            # Quality-driven algorithm selection
            gmsh.option.setNumber("Mesh.Algorithm", 6)        # Frontal-Delaunay for 2D
            gmsh.option.setNumber("Mesh.Algorithm3D", 1)      # Delaunay for 3D (stable)
            gmsh.option.setNumber("Mesh.Optimize", 1)         # Enable optimization
            gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)   # Netgen optimization

            # Size gradation control
            gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 1.0)
            gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)

            self.log_message("  [OK] Mesh parameters configured")

            # Step 5: Generate mesh
            self.log_message("[Step 5/7] Generating mesh...")

            self.log_message("  * Generating 1D mesh (curves)...")
            gmsh.model.mesh.generate(1)

            self.log_message("  * Generating 2D mesh (surfaces)...")
            gmsh.model.mesh.generate(2)

            self.log_message("  * Generating 3D mesh (volumes)...")
            gmsh.model.mesh.generate(3)

            self.log_message("  [OK] Mesh generation complete")

            # Step 6: Optimize mesh quality
            self.log_message("[Step 6/7] Optimizing mesh quality...")
            gmsh.model.mesh.optimize("Netgen")
            self.log_message("  [OK] Optimization complete")

            # Step 7: Analyze quality
            self.log_message("[Step 7/7] Analyzing mesh quality...")
            quality_metrics = self._analyze_mesh_quality()

            # Generate output filename if not provided
            if output_file is None:
                input_path = Path(input_file)
                output_file = str(input_path.parent / f"{input_path.stem}_advanced.msh")

            # Write mesh
            gmsh.write(output_file)
            self.log_message(f"  [OK] Mesh saved to: {output_file}")

            elapsed = time.time() - start_time
            self.log_message(f"\n{'='*70}")
            self.log_message(f"[OK] Mesh generation successful!")
            self.log_message(f"  Total time: {elapsed:.1f} seconds")
            self.log_message(f"  Output: {output_file}")
            self.log_message(f"{'='*70}\n")

            # Create result object
            result = MeshGenerationResult(
                success=True,
                output_file=output_file,
                quality_metrics=quality_metrics,
                message=f"Successfully generated mesh with advanced features in {elapsed:.1f}s"
            )

            # Add advanced analysis results to quality metrics
            if self.geometry_analysis:
                result.quality_metrics['geometry_analysis'] = self.geometry_analysis
            if self.advanced_meshing_results:
                result.quality_metrics['advanced_features'] = {
                    'virtual_faces_created': self.advanced_meshing_results['virtual_faces_created'],
                    'size_fields_created': self.advanced_meshing_results['size_fields_created'],
                    'boundary_layers_created': self.advanced_meshing_results['boundary_layers_created']
                }

            return result

        except Exception as e:
            self.log_message(f"\n[X] Error during mesh generation: {str(e)}", level="ERROR")
            import traceback
            traceback.print_exc()

            return MeshGenerationResult(
                success=False,
                message=f"Mesh generation failed: {str(e)}"
            )

        finally:
            if self.gmsh_initialized:
                gmsh.finalize()
                self.gmsh_initialized = False

    def _analyze_mesh_quality(self) -> Dict:
        """
        Analyze mesh quality using GPU-accelerated analyzer

        Returns:
            Dictionary with quality metrics
        """
        try:
            # Get mesh data from Gmsh
            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            node_coords = node_coords.reshape(-1, 3)

            # Get tetrahedra
            elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=3)

            if len(elem_types) == 0:
                self.log_message("  [!] No 3D elements found", level="WARNING")
                return {'error': 'No 3D elements'}

            # Find tetrahedral elements (type 4)
            tet_idx = None
            for i, elem_type in enumerate(elem_types):
                if elem_type == 4:  # Tetrahedron
                    tet_idx = i
                    break

            if tet_idx is None:
                self.log_message("  [!] No tetrahedral elements found", level="WARNING")
                return {'error': 'No tetrahedral elements'}

            tet_connectivity = elem_node_tags[tet_idx].reshape(-1, 4) - 1  # Convert to 0-based

            # Analyze quality
            metrics = self.quality_analyzer.analyze_mesh(node_coords, tet_connectivity)

            # Print quality summary
            self.log_message(f"\n  Quality Summary:")
            self.log_message(f"    Elements: {metrics.get('num_elements', 0)}")
            self.log_message(f"    Nodes: {metrics.get('num_nodes', 0)}")
            self.log_message(f"    SICN min: {metrics.get('sicn_min', 0):.3f}")
            self.log_message(f"    SICN mean: {metrics.get('sicn_mean', 0):.3f}")
            self.log_message(f"    Aspect ratio mean: {metrics.get('aspect_ratio_mean', 0):.2f}")

            # Quality rating
            sicn_min = metrics.get('sicn_min', 0)
            if sicn_min >= 0.4:
                rating = "Excellent"
            elif sicn_min >= 0.3:
                rating = "Good"
            elif sicn_min >= 0.2:
                rating = "Fair"
            elif sicn_min >= 0.1:
                rating = "Poor"
            else:
                rating = "Critical"

            self.log_message(f"    Overall rating: {rating}")

            # Add advanced feature impact if available
            if self.advanced_meshing_results:
                virt_faces = self.advanced_meshing_results['virtual_faces_created']
                size_fields = self.advanced_meshing_results['size_fields_created']
                bl_regions = self.advanced_meshing_results['boundary_layers_created']

                if virt_faces > 0 or size_fields > 0 or bl_regions > 0:
                    self.log_message(f"\n  Advanced Features Applied:")
                    if virt_faces > 0:
                        self.log_message(f"    * {virt_faces} virtual composite faces")
                    if size_fields > 0:
                        self.log_message(f"    * {size_fields} adaptive size fields")
                    if bl_regions > 0:
                        self.log_message(f"    * {bl_regions} boundary layer regions")

            return metrics

        except Exception as e:
            self.log_message(f"  [!] Quality analysis failed: {e}", level="WARNING")
            return {'error': str(e)}


# Example usage and testing
if __name__ == "__main__":
    print("""
Advanced Meshing Strategy
=========================

Production-grade meshing with automatic feature detection and handling.

Features:
1. Virtual Topology - Eliminates sliver elements from fillets/lofts
2. Adaptive Refinement - Resolves small features with localized sizing
3. Boundary Layers - Structured meshing for thin channels

Usage Example:
--------------
from strategies.advanced_strategy import AdvancedMeshingStrategy
from core.optimization_config import OptimizationConfig, QualityPreset

# Use production preset (all features enabled)
config = OptimizationConfig.from_preset(QualityPreset.PRODUCTION)

# Create strategy
strategy = AdvancedMeshingStrategy(config=config)

# Generate mesh
result = strategy.generate_mesh(
    input_file="model.step",
    output_file="output.msh",
    global_mesh_size=5.0
)

if result.success:
    print(f"Success! Quality: SICN min = {result.quality_metrics['sicn_min']:.3f}")
else:
    print(f"Failed: {result.message}")

Configuration Options:
---------------------
# Enable/disable individual features
config.advanced_features.enable_virtual_topology = True
config.advanced_features.enable_adaptive_refinement = True
config.advanced_features.enable_boundary_layers = False  # CFD only

# Adjust thresholds
config.advanced_features.sliver_aspect_threshold = 20.0  # Aspect ratio
config.advanced_features.small_feature_threshold = 1.0   # mm
config.advanced_features.thin_channel_threshold = 5.0    # mm

For integration with GUI, see gui_final.py.
For batch processing, see batch_mesh.py.
""")
