"""
Matplotlib-based 3D Thermal Visualization (PyVista-free fallback)
==================================================================

Uses matplotlib's 3D capabilities for environments where PyVista is unavailable
or incompatible (e.g., Python 3.14+).
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import List
import logging

logger = logging.getLogger(__name__)


def read_vtk_legacy(vtk_file):
    """Read legacy VTK file and extract points and temperature"""
    with open(vtk_file, 'r') as f:
        lines = f.readlines()

    # Find POINTS section
    points = []
    elements = []
    temps = []

    i = 0
    while i < len(lines):
        if lines[i].startswith('POINTS'):
            num_points = int(lines[i].split()[1])
            i += 1
            while len(points) < num_points and i < len(lines):
                coords = lines[i].strip().split()
                for j in range(0, len(coords), 3):
                    if len(points) < num_points and j+2 < len(coords):
                        points.append([float(coords[j]), float(coords[j+1]), float(coords[j+2])])
                i += 1

        elif lines[i].startswith('CELLS'):
            parts = lines[i].split()
            num_cells = int(parts[1])
            i += 1
            while len(elements) < num_cells and i < len(lines):
                cell = [int(x) for x in lines[i].strip().split()]
                if len(cell) == 5 and cell[0] == 4:  # Tetrahedral
                    elements.append(cell[1:])  # Skip the count
                i += 1

        elif lines[i].startswith('POINT_DATA'):
            num_data = int(lines[i].split()[1])
            # Skip SCALARS and LOOKUP_TABLE lines
            i += 3
            while len(temps) < num_data and i < len(lines):
                try:
                    temp = float(lines[i].strip())
                    temps.append(temp)
                except ValueError:
                    pass
                i += 1
            break
        else:
            i += 1

    return np.array(points), np.array(elements), np.array(temps)


def generate_matplotlib_3d_view(
    vtk_path: str,
    output_path: str,
    view_name: str = 'isometric',
    colormap: str = 'coolwarm',
    window_size: tuple = (1920, 1080),
    dpi: int = 150
):
    """
    Generate 3D thermal visualization using matplotlib

    Args:
        vtk_path: Path to VTK file
        output_path: Output PNG path
        view_name: Camera view ('isometric', 'top', 'front')
        colormap: Colormap name
        window_size: Figure size in pixels
        dpi: DPI for output
    """
    logger.info(f"Generating {view_name} view using matplotlib 3D...")

    # Read VTK file
    points, elements, temps = read_vtk_legacy(vtk_path)

    if len(points) == 0 or len(temps) == 0:
        logger.error("Failed to read VTK file")
        return None

    logger.info(f"Loaded {len(points)} points, {len(temps)} temperature values")
    logger.info(f"Temperature range: {temps.min():.2f}K - {temps.max():.2f}K")

    # Check if temperature data is valid
    if temps.min() == temps.max():
        logger.warning("All temperatures are the same - visualization may appear uniform")

    # Convert to Celsius
    temps_c = temps - 273.15
    logger.info(f"Temperature range (C): {temps_c.min():.2f}°C - {temps_c.max():.2f}°C")

    # Create figure
    fig = plt.figure(figsize=(window_size[0]/dpi, window_size[1]/dpi), facecolor='white')
    ax = fig.add_subplot(111, projection='3d', facecolor='white')

    # Extract coordinates
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Plot surface mesh using triangulation
    # For better visualization, plot only the surface nodes (those on the boundary)
    # We can identify these as nodes that appear in fewer elements

    # Determine marker size based on point density
    num_points = len(points)
    if num_points < 500:
        marker_size = 50
    elif num_points < 2000:
        marker_size = 20
    elif num_points < 10000:
        marker_size = 10
    else:
        marker_size = 5

    # Create scatter plot with temperature coloring
    # Force explicit vmin/vmax to ensure color mapping works
    temp_range = temps_c.max() - temps_c.min()
    if temp_range < 1e-6:
        # If temperatures are nearly uniform, add small variation for visibility
        vmin, vmax = temps_c.min() - 1, temps_c.max() + 1
    else:
        vmin, vmax = temps_c.min(), temps_c.max()

    scatter = ax.scatter(x, y, z, c=temps_c, cmap=colormap,
                        s=marker_size, alpha=0.8, edgecolors='none',
                        vmin=vmin, vmax=vmax)

    logger.info(f"Created scatter plot with colormap range: {vmin:.2f}°C - {vmax:.2f}°C")

    # Add surface wireframe for better depth perception
    if len(elements) > 0:
        # Plot subset of tetrahedra edges for wireframe effect (not all to avoid clutter)
        sample_indices = np.random.choice(len(elements), min(200, len(elements)), replace=False)
        for elem_idx in sample_indices:
            elem = elements[elem_idx]
            # Plot edges of tetrahedron
            edges = [
                (elem[0], elem[1]), (elem[0], elem[2]), (elem[0], elem[3]),
                (elem[1], elem[2]), (elem[1], elem[3]), (elem[2], elem[3])
            ]
            for edge in edges:
                ax.plot3D([x[edge[0]], x[edge[1]]],
                         [y[edge[0]], y[edge[1]]],
                         [z[edge[0]], z[edge[1]]],
                         'k-', linewidth=0.1, alpha=0.1)

    # Set view angle
    if view_name == 'isometric':
        ax.view_init(elev=30, azim=45)
    elif view_name == 'top':
        ax.view_init(elev=90, azim=0)
    elif view_name == 'front':
        ax.view_init(elev=0, azim=0)

    # Labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Thermal Analysis - {view_name.capitalize()} View', fontsize=14, pad=20)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Temperature (°C)', fontsize=12)

    # Set aspect ratio
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    z_range = z.max() - z.min()
    max_range = max(x_range, y_range, z_range)

    mid_x = (x.max() + x.min()) / 2
    mid_y = (y.max() + y.min()) / 2
    mid_z = (z.max() + z.min()) / 2

    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"Saved {view_name} view to {output_path}")
    return output_path


def generate_thermal_views_matplotlib(
    vtu_path: str,
    output_dir: Path,
    job_name: str,
    views: List[str] = None,
    colormap: str = 'coolwarm'
) -> List[str]:
    """
    Generate thermal visualizations using matplotlib (PyVista-free)

    Args:
        vtu_path: Path to VTK file
        output_dir: Output directory
        job_name: Job name for file naming
        views: List of views to generate
        colormap: Colormap name

    Returns:
        List of generated image paths
    """
    if views is None:
        views = ['isometric', 'top', 'front']

    image_paths = []

    for view_name in views:
        if view_name == 'section':
            logger.info("Section view not supported in matplotlib fallback, skipping")
            continue

        output_path = output_dir / f"{job_name}_thermal_{view_name}.png"

        try:
            result = generate_matplotlib_3d_view(
                str(vtu_path),
                str(output_path),
                view_name=view_name,
                colormap=colormap
            )
            if result:
                image_paths.append(str(output_path))
        except Exception as e:
            logger.error(f"Failed to generate {view_name} view: {e}")
            continue

    return image_paths
