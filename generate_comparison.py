"""
Generate side-by-side comparison of old vs new visualization
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import griddata

# Read VTK file
vtk_file = Path(r"c:\Users\markm\Downloads\Simops\simops_output\fddbfe8a\thermal_result.vtk")

with open(vtk_file, 'r') as f:
    lines = f.readlines()

# Extract data
points = []
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
    elif lines[i].startswith('POINT_DATA'):
        num_data = int(lines[i].split()[1])
        i += 3
        while len(temps) < num_data and i < len(lines):
            try:
                temps.append(float(lines[i].strip()))
            except ValueError:
                pass
            i += 1
        break
    else:
        i += 1

node_coords = np.array(points)
temperature = np.array(temps)

print(f"Loaded {len(node_coords)} nodes")
print(f"Temperature range: {temperature.min():.1f}K - {temperature.max():.1f}K")

# Create comparison figure
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Thermal Visualization Comparison: Old vs New', fontsize=16, fontweight='bold')

# OLD METHOD (Row 1): Scatter plots
x, y, z = node_coords[:, 0], node_coords[:, 1], node_coords[:, 2]
T_min, T_max = temperature.min(), temperature.max()

# Old XY
axes[0, 0].scatter(x, y, c=temperature, cmap='coolwarm', s=5, vmin=T_min, vmax=T_max, alpha=0.7)
axes[0, 0].set_title('OLD: XY (Top) - Scatter', fontsize=12)
axes[0, 0].set_xlabel('X')
axes[0, 0].set_ylabel('Y')
axes[0, 0].set_aspect('equal')
axes[0, 0].grid(True, alpha=0.3)

# Old XZ
axes[0, 1].scatter(x, z, c=temperature, cmap='coolwarm', s=5, vmin=T_min, vmax=T_max, alpha=0.7)
axes[0, 1].set_title('OLD: XZ (Front) - Scatter', fontsize=12)
axes[0, 1].set_xlabel('X')
axes[0, 1].set_ylabel('Z')
axes[0, 1].set_aspect('equal')
axes[0, 1].grid(True, alpha=0.3)

# Old YZ
axes[0, 2].scatter(y, z, c=temperature, cmap='coolwarm', s=5, vmin=T_min, vmax=T_max, alpha=0.7)
axes[0, 2].set_title('OLD: YZ (Side) - Scatter', fontsize=12)
axes[0, 2].set_xlabel('Y')
axes[0, 2].set_ylabel('Z')
axes[0, 2].set_aspect('equal')
axes[0, 2].grid(True, alpha=0.3)

# NEW METHOD (Row 2): Interpolated contours
views = [
    (x, y, 'X', 'Y', 0),
    (x, z, 'X', 'Z', 1),
    (y, z, 'Y', 'Z', 2)
]

for idx, (x_view, y_view, xlabel, ylabel, col) in enumerate(views):
    # Create grid
    x_range = x_view.max() - x_view.min()
    y_range = y_view.max() - y_view.min()
    grid_res = 100

    xi = np.linspace(x_view.min() - 0.05*x_range, x_view.max() + 0.05*x_range, grid_res)
    yi = np.linspace(y_view.min() - 0.05*y_range, y_view.max() + 0.05*y_range, grid_res)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    # Interpolate
    zi = griddata((x_view, y_view), temperature, (xi_grid, yi_grid), method='linear', fill_value=T_min)

    # Plot contour
    contour = axes[1, col].contourf(xi_grid, yi_grid, zi, levels=20, cmap='coolwarm', vmin=T_min, vmax=T_max)
    axes[1, col].contour(xi_grid, yi_grid, zi, levels=10, colors='black', linewidths=0.3, alpha=0.3)

    axes[1, col].set_title(f'NEW: {xlabel}{ylabel} - Interpolated Contour', fontsize=12)
    axes[1, col].set_xlabel(xlabel)
    axes[1, col].set_ylabel(ylabel)
    axes[1, col].set_aspect('equal')
    axes[1, col].grid(True, alpha=0.3)

# Add single colorbar for all plots
fig.colorbar(contour, ax=axes, label='Temperature (K)', shrink=0.6, pad=0.02)

# Add annotation
fig.text(0.5, 0.02,
         'OLD: Random-looking scatter due to 3D-to-2D projection\n' +
         'NEW: Clear thermal gradients via interpolation',
         ha='center', fontsize=11, style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout(rect=[0, 0.05, 1, 0.98])
output_file = vtk_file.parent / 'visualization_comparison.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
plt.close()

print(f"\nComparison saved to: {output_file}")
print("\nKey differences:")
print("  OLD: Scatter plots show random red/blue distribution")
print("  NEW: Contours show clear heat flow from hot to cold")
