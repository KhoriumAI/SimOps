"""
Cross Section Handler
=====================

Handles cross-section visualization, plane intersection, and clipping.
"""

import vtk
import numpy as np

class CrossSectionHandler:
    def __init__(self, viewer):
        self.viewer = viewer
        self.clipping_enabled = False
        self.clip_plane = None
        self.clip_axis = 'x'
        self.clip_offset = 0.0
        self.cross_section_actor = None
        self.cross_section_mode = "layered"
        self.cross_section_element_mode = "auto"

    def set_clipping(self, enabled: bool, axis: str = 'x', offset: float = 0.0):
        self.clipping_enabled = enabled
        self.clip_axis = axis.lower()
        self.clip_offset = offset
        
        if not self.viewer.current_poly_data:
            return
        
        if enabled:
            self._apply_clipping()
        else:
            self._remove_clipping()
    
    def set_cross_section_mode(self, mode: str):
        if mode not in ["perfect", "layered"]:
            print(f"[WARNING] Invalid cross-section mode '{mode}', using 'perfect'")
            mode = "perfect"
        
        self.cross_section_mode = mode
        print(f"[DEBUG] Cross-section mode set to: {mode}")
        
        if self.clipping_enabled and self.viewer.current_poly_data:
            self._apply_clipping()
            
    def set_cross_section_element_mode(self, mode: str):
        mode = mode.lower()
        if mode not in ("auto", "tetrahedra", "hexahedra"):
            mode = "auto"
        self.cross_section_element_mode = mode
        print(f"[DEBUG] Cross-section element mode set to: {mode}")
        if self.clipping_enabled and self.viewer.current_poly_data:
            self._apply_clipping()

    def _remove_clipping(self):
        if self.viewer.current_actor:
            self.viewer.current_actor.GetMapper().RemoveAllClippingPlanes()
        
        if self.cross_section_actor:
            self.viewer.renderer.RemoveActor(self.cross_section_actor)
            self.cross_section_actor = None
        
        self.viewer.vtk_widget.GetRenderWindow().Render()

    def _apply_clipping(self):
        if not self.viewer.current_poly_data:
            return

        bounds = self.viewer.current_poly_data.GetBounds()
        center = [
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2
        ]
        
        size = [
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4]
        ]
        
        plane = vtk.vtkPlane()
        origin = list(center)
        normal = [0, 0, 0]
        
        if self.clip_axis == 'x':
            normal = [-1, 0, 0]
            origin[0] += (self.clip_offset / 100.0) * size[0]
        elif self.clip_axis == 'y':
            normal = [0, -1, 0]
            origin[1] += (self.clip_offset / 100.0) * size[1]
        elif self.clip_axis == 'z':
            normal = [0, 0, -1]
            origin[2] += (self.clip_offset / 100.0) * size[2]
            
        plane.SetOrigin(origin)
        plane.SetNormal(normal)
        self.clip_plane = plane
        
        # Apply clipping to main actor
        if self.viewer.current_actor:
            mapper = self.viewer.current_actor.GetMapper()
            mapper.RemoveAllClippingPlanes()
            mapper.AddClippingPlane(plane)
            
        # Generate cross-section
        if self.cross_section_actor:
            self.viewer.renderer.RemoveActor(self.cross_section_actor)
            self.cross_section_actor = None
            
        if self.cross_section_mode == "layered":
            cross_section_poly = self._generate_layered_cross_section(origin, normal)
        else:
            cross_section_poly = self._generate_cross_section_mesh(origin, normal)
            
        if cross_section_poly and cross_section_poly.GetNumberOfPoints() > 0:
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(cross_section_poly)
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            
            # Style
            if self.cross_section_mode == "layered":
                actor.GetProperty().SetColor(0.8, 0.8, 0.8)
                actor.GetProperty().SetOpacity(1.0)
                actor.GetProperty().SetEdgeVisibility(True)
                actor.GetProperty().SetLineWidth(1.5)
            else:
                actor.GetProperty().SetColor(1.0, 0.2, 0.2)
                actor.GetProperty().SetLineWidth(2)
                actor.GetProperty().SetLighting(False)
            
            self.viewer.renderer.AddActor(actor)
            self.cross_section_actor = actor
            
        self.viewer.vtk_widget.GetRenderWindow().Render()

    def _generate_cross_section_mesh(self, plane_origin, plane_normal):
        # If we have a volumetric grid (Polyhedra), use vtkCutter
        if self.viewer.current_volumetric_grid and self.viewer.current_volumetric_grid.GetNumberOfCells() > 0:
            cell_type = self.viewer.current_volumetric_grid.GetCellType(0)
            if cell_type == vtk.VTK_POLYHEDRON:
                plane = vtk.vtkPlane()
                plane.SetOrigin(plane_origin)
                plane.SetNormal(plane_normal)
                
                cutter = vtk.vtkCutter()
                cutter.SetInputData(self.viewer.current_volumetric_grid)
                cutter.SetCutFunction(plane)
                cutter.Update()
                return cutter.GetOutput()
        
        # Fallback to manual intersection
        intersecting_elements = self._get_volume_elements_intersecting_plane(plane_origin, plane_normal)
        
        if not intersecting_elements:
            return vtk.vtkPolyData()
            
        all_points = []
        all_triangles = []
        
        for element in intersecting_elements:
            polygon_points = self._intersect_element_with_plane(element, plane_origin, plane_normal)
            if len(polygon_points) < 3:
                continue
                
            point_offset = len(all_points)
            num_points = len(polygon_points)
            all_points.extend(polygon_points)
            
            if num_points == 3:
                all_triangles.append([point_offset, point_offset + 1, point_offset + 2])
            elif num_points == 4:
                all_triangles.append([point_offset, point_offset + 1, point_offset + 2])
                all_triangles.append([point_offset, point_offset + 2, point_offset + 3])
            else:
                for i in range(1, num_points - 1):
                    all_triangles.append([point_offset, point_offset + i, point_offset + i + 1])
                    
        points = vtk.vtkPoints()
        for p in all_points:
            points.InsertNextPoint(p)
            
        triangles = vtk.vtkCellArray()
        for t in all_triangles:
            triangles.InsertNextCell(3)
            triangles.InsertCellPoint(t[0])
            triangles.InsertCellPoint(t[1])
            triangles.InsertCellPoint(t[2])
            
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetPolys(triangles)
        
        return poly_data

    def _generate_layered_cross_section(self, plane_origin, plane_normal):
        # If we have a volumetric grid (Polyhedra), use vtkExtractGeometry
        if self.viewer.current_volumetric_grid and self.viewer.current_volumetric_grid.GetNumberOfCells() > 0:
            cell_type = self.viewer.current_volumetric_grid.GetCellType(0)
            if cell_type == vtk.VTK_POLYHEDRON:
                # Create an implicit function for the clip
                plane = vtk.vtkPlane()
                plane.SetOrigin(plane_origin)
                plane.SetNormal([-n for n in plane_normal]) # Invert normal to keep the "back" side
                
                extract = vtk.vtkExtractGeometry()
                extract.SetInputData(self.viewer.current_volumetric_grid)
                extract.SetImplicitFunction(plane)
                extract.ExtractInsideOn()
                extract.ExtractBoundaryCellsOn()
                extract.Update()
                
                # Extract surface of the result
                surface = vtk.vtkDataSetSurfaceFilter()
                surface.SetInputData(extract.GetOutput())
                surface.Update()
                return surface.GetOutput()

        # Manual implementation for Tets/Hexes
        intersecting_elements = self._get_volume_elements_intersecting_plane(plane_origin, plane_normal)
        
        if not intersecting_elements:
            return vtk.vtkPolyData()
            
        ugrid = vtk.vtkUnstructuredGrid()
        points = vtk.vtkPoints()
        
        # We need to map global node IDs to local point IDs
        node_id_to_local = {}
        
        for element in intersecting_elements:
            node_ids = element['nodes']
            local_ids = []
            
            for nid in node_ids:
                if nid not in node_id_to_local:
                    coords = self.viewer.current_mesh_nodes[nid]
                    pid = points.InsertNextPoint(coords)
                    node_id_to_local[nid] = pid
                local_ids.append(node_id_to_local[nid])
            
            if element['type'] == 'tetrahedron':
                ugrid.InsertNextCell(vtk.VTK_TETRA, 4, local_ids)
            elif element['type'] == 'hexahedron':
                ugrid.InsertNextCell(vtk.VTK_HEXAHEDRON, 8, local_ids)
                
        ugrid.SetPoints(points)
        
        # Extract outer boundary of the intersecting elements
        surface_filter = vtk.vtkDataSetSurfaceFilter()
        surface_filter.SetInputData(ugrid)
        surface_filter.Update()
        boundary_poly = surface_filter.GetOutput()
        
        # Generate CAP faces by slicing each element with the plane
        # This creates the solid "capped" appearance for hex meshes
        cap_points = vtk.vtkPoints()
        cap_polys = vtk.vtkCellArray()
        
        for element in intersecting_elements:
            polygon_pts = self._intersect_element_with_plane(element, plane_origin, plane_normal)
            if len(polygon_pts) >= 3:
                # Add points and create polygon cell
                pt_ids = []
                for pt in polygon_pts:
                    pid = cap_points.InsertNextPoint(pt)
                    pt_ids.append(pid)
                
                cap_polys.InsertNextCell(len(pt_ids))
                for pid in pt_ids:
                    cap_polys.InsertCellPoint(pid)
        
        cap_poly = vtk.vtkPolyData()
        cap_poly.SetPoints(cap_points)
        cap_poly.SetPolys(cap_polys)
        
        # Merge boundary surface + cap faces for complete solid cross-section
        append_filter = vtk.vtkAppendPolyData()
        append_filter.AddInputData(boundary_poly)
        append_filter.AddInputData(cap_poly)
        append_filter.Update()
        
        return append_filter.GetOutput()


    def _iter_volume_elements(self):
        if not self.viewer.current_mesh_nodes:
            return
        
        mode = self.cross_section_element_mode
        tets = self.viewer.current_tetrahedra or []
        hexes = self.viewer.current_hexahedra or []
        
        if mode == "auto":
            if hexes and (len(hexes) >= len(tets)):
                mode = "hexahedra"
            elif tets:
                mode = "tetrahedra"
            elif hexes:
                mode = "hexahedra"
            else:
                mode = "tetrahedra"
        
        if mode == "tetrahedra" and tets:
            for tet in tets: yield tet
        elif mode == "hexahedra" and hexes:
            for hexa in hexes: yield hexa
        else:
            for tet in tets: yield tet
            for hexa in hexes: yield hexa

    def _get_volume_elements_intersecting_plane(self, plane_origin, plane_normal):
        intersecting = []
        for element in self._iter_volume_elements() or []:
            node_ids = element['nodes']
            vertices = [self.viewer.current_mesh_nodes[nid] for nid in node_ids]
            distances = [self._signed_distance_to_plane(v, plane_origin, plane_normal)
                         for v in vertices]
            has_positive = any(d > 1e-10 for d in distances)
            has_negative = any(d < -1e-10 for d in distances)
            if has_positive and has_negative:
                intersecting.append(element)
        return intersecting

    def _signed_distance_to_plane(self, point, plane_origin, plane_normal):
        diff = np.array(point) - np.array(plane_origin)
        return np.dot(diff, plane_normal)

    def _intersect_edge_with_plane(self, v1, v2, plane_origin, plane_normal):
        d1 = self._signed_distance_to_plane(v1, plane_origin, plane_normal)
        d2 = self._signed_distance_to_plane(v2, plane_origin, plane_normal)
        
        if d1 * d2 > 0:
            return None
        
        t = d1 / (d1 - d2)
        v1_arr = np.array(v1)
        v2_arr = np.array(v2)
        intersection = v1_arr + t * (v2_arr - v1_arr)
        return intersection.tolist()

    def _intersect_element_with_plane(self, element, plane_origin, plane_normal):
        node_ids = element['nodes']
        vertices = [self.viewer.current_mesh_nodes[nid] for nid in node_ids]
        
        if element['type'] == 'tetrahedron':
            edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        elif element['type'] == 'hexahedron':
            edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        else:
            return []
        
        intersection_points = []
        for i, j in edges:
            point = self._intersect_edge_with_plane(vertices[i], vertices[j], plane_origin, plane_normal)
            if point is not None:
                intersection_points.append(point)
        
        # Deduplicate
        EPSILON = 1e-9
        unique_points = []
        for pt in intersection_points:
            is_duplicate = False
            for existing_pt in unique_points:
                dist_sq = sum((pt[k] - existing_pt[k])**2 for k in range(3))
                if dist_sq < EPSILON * EPSILON:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_points.append(pt)
        
        if len(unique_points) < 3:
            return unique_points
            
        # Sort points
        centroid = np.mean(unique_points, axis=0)
        normal = np.array(plane_normal)
        normal = normal / np.linalg.norm(normal)
        
        if abs(normal[0]) < 0.9:
            arbitrary = np.array([1, 0, 0])
        else:
            arbitrary = np.array([0, 1, 0])
            
        u = np.cross(normal, arbitrary)
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        
        def get_angle(pt):
            relative = np.array(pt) - centroid
            x = np.dot(relative, u)
            y = np.dot(relative, v)
            return np.arctan2(y, x)
            
        unique_points.sort(key=get_angle)
        return unique_points
