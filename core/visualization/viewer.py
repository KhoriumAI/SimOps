"""
SimOps Interactive 3D Viewer
=============================

Interactive 3D results viewer using PyVista with WebGL backend.
Supports cross-section capabilities, color mapping, and HTML export.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import pyvista as pv

from .colormaps import get_colormap, DEFAULT_COLORMAP
from .export_webgl import export_to_webgl

logger = logging.getLogger(__name__)


class SimOpsViewer:
    """
    Interactive 3D viewer for simulation results.
    
    Features:
    - Load VTK/VTU files
    - Add cross-sections with clipping planes
    - Export to interactive WebGL HTML
    - Multiple color maps
    - Camera controls (orbit, zoom, reset)
    """
    
    def __init__(self, window_size: Tuple[int, int] = (1024, 768)):
        """
        Initialize viewer.
        
        Args:
            window_size: Window size as (width, height)
        """
        self.window_size = window_size
        self.mesh: Optional[pv.DataSet] = None
        self.scalar_field: Optional[str] = None
        self.colormap: str = DEFAULT_COLORMAP
        self.plotter: Optional[pv.Plotter] = None
        self.cross_sections: List[dict] = []
        
    def load_results(self, vtu_file: str, scalar_field: Optional[str] = None) -> bool:
        """
        Load VTK/VTU simulation results.
        
        Args:
            vtu_file: Path to VTU/VTK file
            scalar_field: Name of scalar field to visualize (auto-detect if None)
            
        Returns:
            True if loading succeeded, False otherwise
        """
        try:
            vtu_path = Path(vtu_file)
            if not vtu_path.exists():
                logger.error(f"File not found: {vtu_file}")
                return False
            
            logger.info(f"Loading results from: {vtu_file}")
            
            # Read mesh
            self.mesh = pv.read(str(vtu_path))
            
            # Auto-detect scalar field if not specified
            if scalar_field is None:
                # Prefer Temperature field, otherwise use first available
                if 'Temperature' in self.mesh.array_names:
                    self.scalar_field = 'Temperature'
                elif 'Temperature_C' in self.mesh.array_names:
                    self.scalar_field = 'Temperature_C'
                elif 'T' in self.mesh.array_names:
                    self.scalar_field = 'T'
                elif len(self.mesh.array_names) > 0:
                    self.scalar_field = self.mesh.array_names[0]
                else:
                    logger.error("No scalar fields found in mesh")
                    return False
            else:
                self.scalar_field = scalar_field
            
            logger.info(f"Loaded mesh with {self.mesh.n_cells} cells")
            logger.info(f"Scalar field: {self.scalar_field}")
            logger.info(f"Available fields: {self.mesh.array_names}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load results: {e}")
            return False
    
    def add_cross_section(
        self,
        origin: Tuple[float, float, float] = (0, 0, 0),
        normal: Tuple[float, float, float] = (1, 0, 0)
    ):
        """
        Add clipping plane for internal view.
        
        Args:
            origin: Origin point of clipping plane (x, y, z)
            normal: Normal vector of clipping plane (x, y, z)
        """
        self.cross_sections.append({
            'origin': origin,
            'normal': normal
        })
        logger.info(f"Added cross-section: origin={origin}, normal={normal}")
    
    def set_colormap(self, colormap_name: str):
        """
        Set color map for visualization.
        
        Args:
            colormap_name: Name of color map (e.g., 'viridis', 'plasma', 'inferno')
        """
        self.colormap = get_colormap(colormap_name)
        logger.info(f"Color map set to: {self.colormap}")
    
    def export_webgl(self, output_html: str, title: str = "SimOps 3D Viewer") -> bool:
        """
        Export interactive HTML with WebGL rendering.
        
        Args:
            output_html: Path to output HTML file
            title: HTML page title
            
        Returns:
            True if export succeeded, False otherwise
        """
        if self.mesh is None:
            logger.error("No mesh loaded. Call load_results() first.")
            return False
        
        try:
            # Create plotter for export
            plotter = pv.Plotter(
                notebook=False,
                off_screen=True,
                window_size=self.window_size
            )
            
            # Add mesh with or without cross-sections
            if len(self.cross_sections) > 0:
                # Apply all cross-sections
                clipped_mesh = self.mesh
                for cs in self.cross_sections:
                    clipped_mesh = clipped_mesh.clip(
                        normal=cs['normal'],
                        origin=cs['origin']
                    )
                
                plotter.add_mesh(
                    clipped_mesh,
                    scalars=self.scalar_field,
                    cmap=self.colormap,
                    show_edges=False,
                    scalar_bar_args={'title': self.scalar_field}
                )
            else:
                # No cross-sections, show full mesh
                plotter.add_mesh(
                    self.mesh,
                    scalars=self.scalar_field,
                    cmap=self.colormap,
                    show_edges=False,
                    scalar_bar_args={'title': self.scalar_field}
                )
            
            # Set view and add axes
            plotter.view_isometric()
            plotter.add_axes()
            
            # Export to HTML
            success = export_to_webgl(plotter, output_html, title=title)
            plotter.close()
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to export WebGL: {e}")
            return False
    
    def show(self, interactive: bool = True):
        """
        Show interactive viewer window.
        
        Args:
            interactive: If True, show interactive window. If False, just render.
        """
        if self.mesh is None:
            logger.error("No mesh loaded. Call load_results() first.")
            return
        
        try:
            # Create interactive plotter
            self.plotter = pv.Plotter(
                notebook=False,
                window_size=self.window_size
            )
            
            # Add mesh with cross-section plane widget if requested
            if len(self.cross_sections) > 0:
                # Use interactive clip plane widget
                self.plotter.add_mesh_clip_plane(
                    self.mesh,
                    scalars=self.scalar_field,
                    cmap=self.colormap,
                    show_edges=False,
                    crinkle=False,  # Smooth cut
                    normal=self.cross_sections[0]['normal'],
                    origin=self.cross_sections[0]['origin'],
                    scalar_bar_args={'title': self.scalar_field}
                )
            else:
                self.plotter.add_mesh(
                    self.mesh,
                    scalars=self.scalar_field,
                    cmap=self.colormap,
                    show_edges=False,
                    scalar_bar_args={'title': self.scalar_field}
                )
            
            # Add controls
            self.plotter.view_isometric()
            self.plotter.add_axes()
            
            # Show controls help
            logger.info("Controls:")
            logger.info(" - Left click + drag: Rotate")
            logger.info(" - Right click + drag: Pan")
            logger.info(" - Scroll: Zoom")
            logger.info(" - 'r': Reset camera")
            if len(self.cross_sections) > 0:
                logger.info(" - Drag arrow/plane widget: Adjust cross-section")
            
            # Show window
            if interactive:
                self.plotter.show()
            
        except Exception as e:
            logger.error(f"Failed to show viewer: {e}")
    
    def close(self):
        """Close the viewer."""
        if self.plotter is not None:
            self.plotter.close()
            self.plotter = None


def create_mock_heatsink() -> pv.DataSet:
    """
    Create a mock heatsink mesh with realistic temperature gradients.
    
    Returns:
        PyVista mesh with Temperature field
    """
    # Geometry - base + fins
    base = pv.Cube(center=(0, 0, 0), x_length=0.1, y_length=0.05, z_length=0.01)
    
    # Add fins
    fins = []
    for x_offset in [-0.03, 0.0, 0.03]:
        fin = pv.Cube(center=(x_offset, 0, 0.015), x_length=0.01, y_length=0.05, z_length=0.02)
        fins.append(fin)
    
    # Combine geometry
    heatsink = base
    for fin in fins:
        heatsink += fin
    
    # Generate realistic temperature field
    centers = heatsink.cell_centers().points
    x, y, z = centers[:, 0], centers[:, 1], centers[:, 2]
    
    # Gaussian heat source at center bottom
    r2 = x**2 + y**2 + (z + 0.005)**2
    sigma_sq = 0.002
    
    T_ambient = 27.0
    T_max = 85.0
    dT = T_max - T_ambient
    
    temps = T_ambient + dT * np.exp(-r2 / sigma_sq)
    
    # Add convection cooling (cooler at top)
    z_factor = (z + 0.005) / 0.03
    temps -= 10 * z_factor * (temps - T_ambient) / dT
    
    heatsink['Temperature'] = temps
    
    return heatsink
