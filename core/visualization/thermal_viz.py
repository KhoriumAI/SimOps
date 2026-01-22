"""
Thermal Visualization Module (Matplotlib Fallback)
==================================================

Generates temperature contour plots from simulation results (VTK/VMesh).
Uses Matplotlib to avoid VTK DLL blocking issues.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import meshio
import numpy as np
import logging
from pathlib import Path
from typing import List, Optional, Union

logger = logging.getLogger(__name__)

class ThermalVisualizer:
    """Generate thermal visualizations using Matplotlib"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_snapshots(
        self,
        mesh_file: Union[str, Path],
        job_name: str,
        temperature_field: str = 'temperature',
        angles: List[str] = None,
        colormap: str = 'inferno'
    ) -> List[str]:
        """
        Generate multi-angle snapshots of temperature field.

        Args:
            mesh_file: Path to mesh file with temperature data
            job_name: Job name for file naming
            temperature_field: Name of temperature field in mesh data
            angles: List of view angles to generate
            colormap: Matplotlib colormap name (default: 'inferno')

        Returns:
            List of paths to generated PNG images
        """
        if angles is None:
            angles = ['iso', 'top', 'side']
            
        try:
            logger.info(f"Loading mesh {mesh_file} for viz...")
            mesh = meshio.read(str(mesh_file))
            
            # Extract Points
            points = mesh.points
            x = points[:, 0]
            y = points[:, 1]
            z = points[:, 2]
            
            # Extract Temperature
            temps = None
            if temperature_field in mesh.point_data:
                temps = mesh.point_data[temperature_field]
            else:
                # Fallback search
                for k in mesh.point_data.keys():
                    if 'temp' in k.lower() or k == 'T':
                        temps = mesh.point_data[k]
                        break
            
            if temps is None:
                logger.warning("No temperature data found for viz")
                return []
            
            # Subsample for performance if too many points
            if len(points) > 50000:
                idx = np.random.choice(len(points), 50000, replace=False)
                x, y, z = x[idx], y[idx], z[idx]
                temps = temps[idx]
            
            image_paths = []
            
            for angle in angles:
                output_path = self.output_dir / f"{job_name}_{angle}.png"
                
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                # Scatter plot with colormap
                img = ax.scatter(x, y, z, c=temps, cmap=colormap, s=1, alpha=0.8)
                
                # Colorbar
                cbar = plt.colorbar(img, ax=ax, shrink=0.6)
                cbar.set_label('Temperature (C)')
                
                # Titles and Labels
                ax.set_title(f"{job_name} - {angle.capitalize()}")
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                
                # Set View Angle
                if angle == 'iso':
                    ax.view_init(elev=30, azim=45)
                elif angle == 'top':
                    ax.view_init(elev=90, azim=0)
                elif angle == 'side':
                    ax.view_init(elev=0, azim=90)
                elif angle == 'front':
                    ax.view_init(elev=0, azim=0)
                
                # Hide axes for cleaner look (optional, but keep for scale for now)
                # ax.set_axis_off()
                
                plt.tight_layout()
                plt.savefig(str(output_path), dpi=100)
                plt.close(fig)
                
                image_paths.append(str(output_path))
                
            return image_paths
            
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            import traceback
            traceback.print_exc()
            return []
