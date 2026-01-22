import meshio
import sys

vtu_path = sys.argv[1]
m = meshio.read(vtu_path)
print(f"File: {vtu_path}")
print(f"Point data fields: {list(m.point_data.keys())}")
print(f"Cell data fields: {list(m.cell_data.keys())}")
if 'T' in m.point_data:
    T = m.point_data['T']
    print(f"Temperature T found: min={T.min()}, max={T.max()}, shape={T.shape}")
elif 'temperature' in m.point_data:
    T = m.point_data['temperature']
    print(f"Temperature field found: min={T.min()}, max={T.max()}, shape={T.shape}")
else:
    print("No temperature field found in point data.")

print(f"Number of points: {len(m.points)}")
print(f"Number of cells: {sum(len(c.data) for c in m.cells)}")
