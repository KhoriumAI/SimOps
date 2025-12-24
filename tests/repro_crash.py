
import vtk
import numpy as np

# --- MOCK CLASS & FUNCTIONS ---
class MockViewer:
    def __init__(self):
        self.current_mesh_nodes = {}
        self.current_tetrahedra = []
        self.current_hexahedra = []
        self.current_cross_section_data = None
        self.cross_section_element_mode = 'auto'

    def _iter_volume_elements(self):
        for tet in self.current_tetrahedra:
            yield tet

    def _signed_distance_to_plane(self, point, plane_origin, plane_normal):
        # Explicit float conversion
        diff = np.array(point) - np.array(plane_origin)
        return np.dot(diff, plane_normal)

    def _get_volume_elements_intersecting_plane(self, plane_origin, plane_normal):
        intersecting = []
        for element in self._iter_volume_elements():
            node_ids = element['nodes']
            vertices = [self.current_mesh_nodes[nid] for nid in node_ids]
            distances = [self._signed_distance_to_plane(v, plane_origin, plane_normal)
                         for v in vertices]
            has_positive = any(d > 1e-10 for d in distances)
            has_negative = any(d < -1e-10 for d in distances)
            if has_positive and has_negative:
                intersecting.append(element)
        return intersecting

    def generate_layered_cross_section(self, plane_origin, plane_normal):
        intersecting_elements = self._get_volume_elements_intersecting_plane(plane_origin, plane_normal)
        
        if not intersecting_elements:
            return vtk.vtkPolyData()
        
        tet_faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
        
        all_points = []
        all_triangles = []
        point_offset = 0
        
        
        # Track tet IDs for quality coloring (MOCK)
        face_to_element_id = []
        
        for element in intersecting_elements:
            node_ids = element['nodes']
            vertices = [self.current_mesh_nodes[nid] for nid in node_ids]
            
            local_offset = point_offset
            all_points.extend(vertices)
            point_offset += len(vertices)
            
            if element['type'] == 'tetrahedron':
                for face_indices in tet_faces:
                    tri_indices = [local_offset + i for i in face_indices]
                    all_triangles.append(tri_indices)
                    face_to_element_id.append(element['id'])
        
        points = vtk.vtkPoints()
        for pt in all_points:
            points.InsertNextPoint(pt)
        
        triangles = vtk.vtkCellArray()
        for tri in all_triangles:
            triangle = vtk.vtkTriangle()
            for i, idx in enumerate(tri):
                triangle.GetPointIds().SetId(i, idx)
            triangles.InsertNextCell(triangle)
        
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetPolys(triangles)

        # --- COLOR LOGIC MOCK ---
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("Colors")
        
        print(f"Generating colors for {len(face_to_element_id)} faces...")
        for _ in face_to_element_id:
             # Randomly assign color logic
             colors.InsertNextTuple3(255, 0, 0)
        
        poly_data.GetCellData().SetScalars(colors)
        
        return poly_data

# --- SETUP MOCK DATA ---
viewer = MockViewer()

# Create a grid of tetrahedra
nodes = {}
tets = []
node_id = 0

for x in range(5):
    for y in range(5):
        for z in range(5):
            # Create a cube of nodes and split into tets
            ids = []
            for dx in [0, 1]:
                for dy in [0, 1]:
                    for dz in [0, 1]:
                        px, py, pz = x+dx, y+dy, z+dz
                        nodes[node_id] = [float(px), float(py), float(pz)]
                        ids.append(node_id)
                        node_id += 1
            
            # Simple tet decomposition (not perfect, just for stress test)
            tets.append({'id': len(tets), 'type': 'tetrahedron', 'nodes': [ids[0], ids[1], ids[2], ids[5]]})
            tets.append({'id': len(tets), 'type': 'tetrahedron', 'nodes': [ids[0], ids[2], ids[3], ids[7]]})

viewer.current_mesh_nodes = nodes
viewer.current_tetrahedra = tets

print(f"Created mocked mesh with {len(nodes)} nodes and {len(tets)} tets")

# --- EXECUTE CRASH SCENARIO ---
origin = [2.5, 2.5, 2.5]
normal = [1, 0, 0]

print("Generating cross section...")
poly = viewer.generate_layered_cross_section(origin, normal)
print(f"Generated polydata: {poly.GetNumberOfPoints()} points, {poly.GetNumberOfCells()} cells")

# --- RENDER TEST ---
renderer = vtk.vtkRenderer()
renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)
renderWindow.SetOffScreenRendering(1) # Headless mode

actor = vtk.vtkActor()
renderer.AddActor(actor)

# 1. Simulate the CRASH scenario: GC of local object
def dangerous_setup():
    safe_poly = vtk.vtkPolyData()
    safe_poly.DeepCopy(poly)
    
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(safe_poly)
    actor.SetMapper(mapper)
    print("Mapper set up. Exiting function scope (safe_poly will be GC'd)...")
    return mapper # We return mapper, but safe_poly is lost

# 2. Simulate the FIX scenario
def safe_setup():
    viewer.current_cross_section_data = vtk.vtkPolyData()
    viewer.current_cross_section_data.DeepCopy(poly)
    
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(viewer.current_cross_section_data)
    actor.SetMapper(mapper)
    print("Mapper set up with persisted data.")
    return mapper

# Toggle this manually to verification
USE_SAFE_MODE = False

if not USE_SAFE_MODE:
    print("Running in DANGEROUS/CRASH mode...")
    crash_mapper = dangerous_setup()
else:
    print("Running in SAFE mode...")
    safe_mapper = safe_setup()

print("Attempting to render...")
try:
    renderWindow.Render()
    print("Render complete! Success.")
except Exception as e:
    print(f"Caught Python Exception: {e}") 
