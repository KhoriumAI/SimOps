"""
Polyhedral Visualizer
=====================

Handles loading and visualization of polyhedral meshes (dual graphs).
"""

import vtk
import json
import numpy as np
from pathlib import Path

class PolyhedralVisualizer:
    def __init__(self, viewer):
        self.viewer = viewer

    def load_file(self, json_path: str):
        """Load polyhedral mesh from JSON file using proper VTK_POLYHEDRON cells"""
        debug_log = Path("poly_debug.txt")
        with open(debug_log, 'w') as f:
            f.write(f"=== POLYHEDRAL LOAD DEBUG (Refactored) ===\n")
            f.write(f"JSON: {json_path}\n")
        
        try:
            # 1. Load JSON
            if not Path(json_path).exists():
                print(f"[POLY-ERROR] File not found: {json_path}")
                return "FAILED"
            
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            nodes = data.get('nodes', {})
            elements = data.get('elements', [])
            
            with open(debug_log, 'a') as f:
                f.write(f"Nodes: {len(nodes)}, Elements: {len(elements)}\n")
            
            print(f"[POLY-VIZ] Loaded {len(nodes)} nodes, {len(elements)} elements")
            
            if not nodes or not elements:
                print("[POLY-VIZ ERROR] No nodes or elements in JSON")
                return "FAILED"
            
            self.viewer.clear_view()
            
            # 1. Create VTK Points
            points = vtk.vtkPoints()
            node_map = {}
            sorted_node_ids = sorted(nodes.keys(), key=lambda x: int(x))
            
            for i, node_id in enumerate(sorted_node_ids):
                coords = nodes[node_id]
                points.InsertNextPoint(coords)
                node_map[str(node_id)] = i
                node_map[int(node_id)] = i
                
            # 2. Create Unstructured Grid
            ugrid = vtk.vtkUnstructuredGrid()
            ugrid.SetPoints(points)
            
            # 3. Insert Cells
            count = 0
            skipped = 0
            
            for elem_idx, elem in enumerate(elements):
                try:
                    if elem['type'] != 'polyhedron':
                        continue
                        
                    faces = elem['faces']
                    if len(faces) < 3:
                        skipped += 1
                        continue
                    
                    face_stream = [len(faces)]
                    
                    for face_nodes in faces:
                        face_stream.append(len(face_nodes))
                        try:
                            face_indices = [node_map[nid] for nid in face_nodes]
                            face_stream.extend(face_indices)
                        except KeyError:
                            skipped += 1
                            continue
                    
                    id_list = vtk.vtkIdList()
                    for val in face_stream:
                        id_list.InsertNextId(val)
                    
                    ugrid.InsertNextCell(vtk.VTK_POLYHEDRON, id_list)
                    count += 1
                        
                except Exception as e:
                    print(f"[POLY-VIZ ERROR] Failed to insert element {elem_idx}: {e}")
                    skipped += 1
                    continue
            
            if count == 0:
                print("[POLY-VIZ ERROR] No cells were created!")
                return "FAILED"
            
            # Extract surface
            geometry_filter = vtk.vtkGeometryFilter()
            geometry_filter.SetInputData(ugrid)
            geometry_filter.Update()
            polydata = geometry_filter.GetOutput()
            
            if polydata.GetNumberOfCells() == 0:
                print("[POLY-VIZ ERROR] Surface extraction produced 0 cells!")
                return "FAILED"
            
            # Mapper/Actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(polydata)
            
            self.viewer.current_actor = vtk.vtkActor()
            self.viewer.current_actor.SetMapper(mapper)
            self.viewer.current_actor.GetProperty().SetEdgeVisibility(1)
            self.viewer.current_actor.GetProperty().SetColor(0.7, 0.8, 0.9)
            self.viewer.current_actor.GetProperty().SetOpacity(1.0)
            
            self.viewer.renderer.AddActor(self.viewer.current_actor)
            self.viewer.current_volumetric_grid = ugrid
            self.viewer.current_poly_data = None
            
            # Reset camera
            bounds = ugrid.GetBounds()
            center = [(bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, (bounds[4]+bounds[5])/2]
            size = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
            
            camera = self.viewer.renderer.GetActiveCamera()
            camera.SetFocalPoint(center)
            camera.SetPosition(center[0]+size*1.5, center[1]+size*1.5, center[2]+size*1.5)
            camera.SetViewUp(0, 0, 1)
            self.viewer.renderer.ResetCamera(bounds)
            
            self.viewer.vtk_widget.GetRenderWindow().Render()
            self.viewer.info_label.setText(f"Polyhedral Mesh: {ugrid.GetNumberOfCells()} cells")
            
            return "SUCCESS"
            
        except Exception as e:
            print(f"[POLY-VIZ ERROR] Exception: {e}")
            import traceback
            traceback.print_exc()
            return "FAILED"
