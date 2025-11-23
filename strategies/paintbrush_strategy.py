"""
Paintbrush Meshing Strategy
============================

Meshing strategy that applies paintbrush-based local refinement.

This strategy extends the base mesh generator to support user-painted
refinement regions. It creates gmsh size fields around painted surfaces
and combines them with any existing refinement strategies.

Features:
- Applies paintbrush refinement fields before meshing
- Compatible with curvature-adaptive and intelligent sizing
- Falls back to standard meshing if no regions painted
- Provides detailed logging of refinement setup

Usage:
    from strategies.paintbrush_strategy import PaintbrushStrategy
    from core.paintbrush_geometry import PaintbrushSelector

    # Setup painted regions
    selector = PaintbrushSelector()
    selector.load_cad_geometry()
    # ... user paints regions ...

    # Generate mesh
    strategy = PaintbrushStrategy(painted_regions=selector.get_painted_regions())
    strategy.generate_mesh(input_file, output_file, params)
"""

import sys
from pathlib import Path
from typing import Optional, Dict, List
import gmsh

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.mesh_generator import BaseMeshGenerator
from core.paintbrush_field_generator import PaintbrushFieldGenerator
from core.curvature_adaptive import CurvatureAdaptiveMesher


class PaintbrushStrategy(BaseMeshGenerator):
    """
    Meshing strategy with paintbrush-based local refinement.

    Combines user-painted refinement regions with automatic curvature-based
    refinement for high-quality meshes.
    """

    def __init__(self,
                 painted_regions: Optional[List] = None,  # List[PaintedRegion]
                 enable_curvature_adaptive: bool = True,
                 gradient_factor: float = 10.0):
        """
        Initialize paintbrush strategy.

        Args:
            painted_regions: List of PaintedRegion objects (can be None)
            enable_curvature_adaptive: Whether to also apply curvature refinement
            gradient_factor: Controls size transition smoothness (higher = smoother)
        """
        super().__init__()
        self.painted_regions = painted_regions or []
        self.enable_curvature_adaptive = enable_curvature_adaptive
        self.gradient_factor = gradient_factor
        self.field_generator = PaintbrushFieldGenerator(gradient_factor=gradient_factor)

    def run_meshing_strategy(self, input_file: str, output_file: str) -> bool:
        """
        Execute paintbrush meshing strategy.

        Args:
            input_file: Path to CAD file (.step, .iges)
            output_file: Path for output mesh file (.msh)

        Returns:
            True if successful, False otherwise
        """
        try:
            self.log_message("Starting Paintbrush Meshing Strategy")
            self.log_message("=" * 60)

            # Load CAD file
            self.log_message(f"Loading CAD file: {Path(input_file).name}")
            if not self.load_cad_file(input_file):
                self.log_message("ERROR: Failed to load CAD file")
                return False

            # Extract geometry information
            self.log_message("Analyzing geometry...")
            geom_info = self._extract_geometry_info()

            if not geom_info:
                self.log_message("ERROR: No geometry found")
                return False

            self.log_message(f"  Volumes: {len(geom_info.get('volumes', []))}")
            self.log_message(f"  Surfaces: {len(geom_info.get('surfaces', []))}")
            self.log_message(f"  Diagonal: {geom_info.get('diagonal', 0):.2f} mm")

            # Calculate base mesh size
            base_size = self._calculate_base_size(geom_info)
            self.log_message(f"  Base mesh size: {base_size:.3f} mm")

            # Setup refinement fields
            self.log_message("\nSetting up refinement fields...")
            fields_to_combine = []

            # 1. Curvature-adaptive field (if enabled)
            if self.enable_curvature_adaptive:
                self.log_message("  Creating curvature-adaptive field...")
                try:
                    curvature_mesher = CurvatureAdaptiveMesher()
                    field_tag = curvature_mesher.create_curvature_adaptive_field(
                        base_size=base_size,
                        curvature_threshold=0.1
                    )
                    if field_tag is not None:
                        fields_to_combine.append(field_tag)
                        self.log_message(f"    Added curvature field {field_tag}")
                except Exception as e:
                    self.log_message(f"    Warning: Could not create curvature field: {e}")

            # 2. Paintbrush refinement fields
            if self.painted_regions:
                self.log_message(f"  Creating paintbrush fields ({len(self.painted_regions)} regions)...")

                # Show statistics
                total_surfaces = len(geom_info.get('surfaces', []))
                painted_surface_tags = set()
                for region in self.painted_regions:
                    painted_surface_tags.update(region.surface_tags)

                self.log_message(f"    Total surfaces painted: {len(painted_surface_tags)}/{total_surfaces}")

                for i, region in enumerate(self.painted_regions):
                    refinement_pct = (region.refinement_level - 1) * 100
                    self.log_message(f"    Region {i+1}: {len(region.surface_tags)} surfaces, "
                                   f"{region.refinement_level:.1f}x refinement (+{refinement_pct:.0f}%)")

                # Create paintbrush field
                paintbrush_field = self.field_generator.create_paintbrush_fields(
                    painted_regions=self.painted_regions,
                    base_size=base_size,
                    existing_fields=fields_to_combine
                )

                # This field already combines with existing fields
                final_field = paintbrush_field

            elif fields_to_combine:
                # Only curvature field, no paintbrush
                self.log_message("  No painted regions - using curvature field only")
                if len(fields_to_combine) == 1:
                    final_field = fields_to_combine[0]
                else:
                    min_field = gmsh.model.mesh.field.add("Min")
                    gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", fields_to_combine)
                    final_field = min_field

            else:
                # No fields - use global sizing
                self.log_message("  No painted regions - using global sizing")
                final_field = None

            # Apply size field
            if final_field is not None:
                self.log_message(f"  Setting field {final_field} as background mesh")
                gmsh.model.mesh.field.setAsBackgroundMesh(final_field)
            else:
                # Set global characteristic length
                self.log_message(f"  Setting global mesh size: {base_size:.3f} mm")
                gmsh.option.setNumber("Mesh.CharacteristicLengthMin", base_size * 0.5)
                gmsh.option.setNumber("Mesh.CharacteristicLengthMax", base_size * 2.0)

            # Generate mesh
            self.log_message("\nGenerating 3D mesh...")
            self.log_message("  This may take several minutes depending on refinement level...")

            success = self.generate_mesh_internal(
                mesh_algorithm=self.config.get_default_param('mesh_algorithm', 1),
                optimize=self.config.get_default_param('optimize_mesh', True)
            )

            if not success:
                self.log_message("ERROR: Mesh generation failed")
                return False

            # Save mesh
            self.log_message(f"\nSaving mesh to: {Path(output_file).name}")
            self.save_mesh(output_file)

            # Analyze quality
            self.log_message("\nAnalyzing mesh quality...")
            metrics = self.analyze_current_mesh()

            if metrics:
                self.log_message("Mesh Quality Metrics:")
                self.log_message(f"  Total elements: {metrics.get('total_elements', 0):,}")
                self.log_message(f"  Total nodes: {metrics.get('total_nodes', 0):,}")

                sicn = metrics.get('gmsh_sicn', {})
                if sicn:
                    self.log_message(f"  SICN: min={sicn.get('min', 0):.4f}, "
                                   f"avg={sicn.get('avg', 0):.4f}, "
                                   f"max={sicn.get('max', 0):.4f}")

                gamma = metrics.get('gmsh_gamma', {})
                if gamma:
                    self.log_message(f"  Gamma: min={gamma.get('min', 0):.4f}, "
                                   f"avg={gamma.get('avg', 0):.4f}")

                # Check if paintbrush improved quality
                if self.painted_regions:
                    self.log_message("\nPaintbrush Refinement Applied:")
                    self.log_message(f"  {len(painted_surface_tags)} surfaces refined")
                    self.log_message(f"  Max refinement: {max(r.refinement_level for r in self.painted_regions):.1f}x")

            self.log_message("\n" + "=" * 60)
            self.log_message("Paintbrush meshing complete!")

            return True

        except Exception as e:
            self.log_message(f"ERROR in paintbrush strategy: {e}")
            import traceback
            self.log_message(traceback.format_exc())
            return False

    def _calculate_base_size(self, geom_info: Dict) -> float:
        """
        Calculate appropriate base mesh size from geometry.

        Args:
            geom_info: Geometry information dictionary

        Returns:
            Base mesh size in mm
        """
        diagonal = geom_info.get('diagonal', 100.0)

        # Check if target element count is specified
        target_elements = self.config.get_default_param('target_elements', None)

        if target_elements:
            # Estimate base size from target element count
            # Rough formula: num_elements ~= (diagonal / element_size)^3
            estimated_size = diagonal / (target_elements ** (1/3))
            self.log_message(f"  Calculated from target elements ({target_elements:,})")
        else:
            # Use quality preset
            quality_preset = self.config.get_default_param('quality_preset', 'Medium')
            quality_map = {
                'Coarse': diagonal / 10,
                'Medium': diagonal / 20,
                'Fine': diagonal / 40,
                'Very Fine': diagonal / 80
            }
            estimated_size = quality_map.get(quality_preset, diagonal / 20)
            self.log_message(f"  Quality preset: {quality_preset}")

        # Apply safety limits
        estimated_size = max(0.1, min(estimated_size, diagonal / 5))

        return estimated_size

    def log_message(self, message: str):
        """Log message to console"""
        print(message)


if __name__ == "__main__":
    print("Paintbrush Meshing Strategy - Test Mode")
    print("\nUsage:")
    print("  from strategies.paintbrush_strategy import PaintbrushStrategy")
    print("  from core.paintbrush_geometry import PaintbrushSelector, PaintedRegion")
    print()
    print("  # Create painted regions")
    print("  selector = PaintbrushSelector()")
    print("  selector.load_cad_geometry()")
    print("  region = selector.add_painted_region([1, 2, 3], brush_radius=5.0, refinement_level=5.0)")
    print()
    print("  # Generate mesh")
    print("  strategy = PaintbrushStrategy(painted_regions=selector.get_painted_regions())")
    print("  strategy.generate_mesh('input.step', 'output.msh', {})")
