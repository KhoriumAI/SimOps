"""
Mesh Loader
===========

Handles loading of mesh files (.msh) and CAD files (.step).
"""

import vtk
import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Tuple

class MeshLoader:
    def __init__(self, viewer):
        self.viewer = viewer

    def load_step_file(self, filepath: str):
        self.viewer.clear_view()
        self.viewer.quality_label.setVisible(False)
        self.viewer.info_label.setText(f"Loading: {Path(filepath).name}")

        try:
            import pyvista as pv
            
            # WINDOWS FIX 1: File Locking
            tmp = tempfile.NamedTemporaryFile(suffix='.stl', delete=False)
            tmp_stl = tmp.name
            tmp.close() 

            # Tessellate CAD for preview display
            gmsh_script = f"""
import gmsh
import json
import sys

try:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("General.Verbosity", 2)

    gmsh.open(r"{filepath}")

    bbox = gmsh.model.getBoundingBox(-1, -1)
    bbox_dims = [bbox[3]-bbox[0], bbox[4]-bbox[1], bbox[5]-bbox[2]]

    volumes_3d = gmsh.model.getEntities(dim=3)
    total_volume_raw = 0.0
    for vol_dim, vol_tag in volumes_3d:
        total_volume_raw += gmsh.model.occ.getMass(vol_dim, vol_tag)

    bbox_volume_raw = bbox_dims[0] * bbox_dims[1] * bbox_dims[2]

    unit_name = "m"
    unit_scale = 1.0
    if total_volume_raw > 10000:
        unit_scale = 0.001
        unit_name = "mm"
    elif max(bbox_dims) > 1000:
        unit_scale = 0.001
        unit_name = "mm"

    total_volume = total_volume_raw * (unit_scale ** 3)
    bbox_diag = (bbox_dims[0]**2 + bbox_dims[1]**2 + bbox_dims[2]**2)**0.5 * unit_scale

    geom_info = {{
        "volume": total_volume, 
        "bbox_diagonal": bbox_diag, 
        "units_detected": unit_name
    }}
    print("GEOM_INFO:" + json.dumps(geom_info))

    gmsh.model.mesh.generate(2)
    gmsh.write(r"{tmp_stl}")
    gmsh.finalize()
    print("SUCCESS_MARKER")

except Exception as e:
    print("GMSH_ERROR:" + str(e))
    sys.exit(1)
"""

            # WINDOWS FIX 2: Environment Variables
            current_env = os.environ.copy()

            result = subprocess.run(
                [sys.executable, "-c", gmsh_script],
                capture_output=True,
                text=True,
                timeout=45,
                env=current_env
            )

            if result.returncode != 0 or "SUCCESS_MARKER" not in result.stdout:
                print(f"--- GMSH SUBPROCESS FAILED ---")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                raise Exception(f"CAD conversion subprocess failed.")

            geom_info = None
            for line in result.stdout.split('\n'):
                if line.startswith("GEOM_INFO:"):
                    geom_info = json.loads(line[10:])
                    break

            if not os.path.exists(tmp_stl) or os.path.getsize(tmp_stl) < 100:
                raise Exception("STL file was not created or is empty.")

            mesh = pv.read(tmp_stl)
            os.unlink(tmp_stl)

            if mesh.n_points == 0:
                raise Exception("Empty mesh - no geometry in CAD file")

            poly_data = mesh.cast_to_unstructured_grid().extract_surface()
            self.viewer.current_poly_data = poly_data

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(poly_data)

            self.viewer.current_actor = vtk.vtkActor()
            self.viewer.current_actor.SetMapper(mapper)
            
            self.viewer.current_actor.GetProperty().SetColor(0.3, 0.5, 0.8)
            self.viewer.current_actor.GetProperty().SetInterpolationToPhong()
            self.viewer.current_actor.GetProperty().EdgeVisibilityOff()
            self.viewer.current_actor.GetProperty().SetAmbient(0.3)
            self.viewer.current_actor.GetProperty().SetDiffuse(0.7)
            self.viewer.current_actor.GetProperty().SetSpecular(0.2)

            self.viewer.renderer.AddActor(self.viewer.current_actor)
            self.viewer.renderer.ResetCamera()
            self.viewer.vtk_widget.GetRenderWindow().Render()

            volume_text = ""
            if geom_info and 'volume' in geom_info:
                v = geom_info['volume']
                if v > 0.001: 
                    volume_text = f"<br>Volume: {v:.4f} m³"
                else: 
                    volume_text = f"<br>Volume: {v*1e9:.0f} mm³"

            self.viewer.info_label.setText(
                f"<b>CAD Preview</b><br>"
                f"{Path(filepath).name}<br>"
                f"<span style='color: #6c757d;'>{poly_data.GetNumberOfPoints():,} nodes{volume_text}</span>"
            )

            return geom_info

        except Exception as e:
            print(f"Load Error: {e}")
            import traceback
            traceback.print_exc()
            self.viewer.info_label.setText(f"CAD Loaded<br><small>(Preview Unavailable)</small><br>Click 'Generate Mesh'")
            return None

    def load_mesh_file(self, filepath: str, result: dict = None):
        print(f"\n{'='*70}")
        print(f"[MESH_LOADER DEBUG] load_mesh_file called: {filepath}")
        print(f"[MESH_LOADER DEBUG] result keys: {list(result.keys()) if result else 'None'}")
        print(f"[MESH_LOADER DEBUG] Viewer: {self.viewer}")
        print(f"[MESH_LOADER DEBUG] Renderer: {self.viewer.renderer}")
        
        self.viewer.clear_view()
        self.viewer.info_label.setText("Loading mesh...")

        try:
            nodes, elements = self._parse_msh_file(filepath)
            print(f"[DEBUG] Parsed {len(nodes)} nodes, {len(elements)} elements")

            # Try to load surface quality data if not provided
            if not (result and result.get('per_element_quality')):
                quality_file = Path(filepath).with_suffix('.quality.json')
                if quality_file.exists():
                    try:
                        with open(quality_file, 'r') as f:
                            surface_quality = json.load(f)

                        if not result: result = {}
                        result['per_element_quality'] = surface_quality.get('per_element_quality', {})
                        self.viewer.current_quality_data = surface_quality
                        
                        result['quality_metrics'] = {
                            'sicn_10_percentile': surface_quality.get('quality_threshold_10', 0.3),
                            'sicn_min': surface_quality.get('statistics', {}).get('min_quality', 0.0),
                            'sicn_avg': surface_quality.get('statistics', {}).get('avg_quality', 0.5),
                            'sicn_max': surface_quality.get('statistics', {}).get('max_quality', 1.0)
                        }
                    except Exception as e:
                        print(f"[DEBUG] Could not load quality data: {e}")
            else:
                self.viewer.current_quality_data = result

            points = vtk.vtkPoints()
            cells = vtk.vtkCellArray()

            node_map = {}
            for idx, (node_id, coords) in enumerate(nodes.items()):
                points.InsertNextPoint(coords)
                node_map[node_id] = idx

            tet_count = sum(1 for e in elements if e['type'] == 'tetrahedron')
            hex_count = sum(1 for e in elements if e['type'] == 'hexahedron')
            tri_count = sum(1 for e in elements if e['type'] == 'triangle')
            quad_count = sum(1 for e in elements if e['type'] == 'quadrilateral')

            # Visualize SURFACE TRIANGLES/QUADS
            for element in elements:
                if element['type'] == 'triangle':
                    tri = vtk.vtkTriangle()
                    for i, node_id in enumerate(element['nodes']):
                        tri.GetPointIds().SetId(i, node_map[node_id])
                    cells.InsertNextCell(tri)
                elif element['type'] == 'quadrilateral':
                    quad = vtk.vtkQuad()
                    for i, node_id in enumerate(element['nodes']):
                        quad.GetPointIds().SetId(i, node_map[node_id])
                    cells.InsertNextCell(quad)

            poly_data = vtk.vtkPolyData()
            poly_data.SetPoints(points)
            poly_data.SetPolys(cells)

            self.viewer.current_poly_data = poly_data
            self.viewer.current_mesh_nodes = nodes
            self.viewer.current_mesh_elements = elements
            self.viewer.current_node_map = node_map
            
            self.viewer.current_tetrahedra = [e for e in elements if e['type'] == 'tetrahedron']
            self.viewer.current_hexahedra = [e for e in elements if e['type'] == 'hexahedron']
            
            # Build volumetric grid for 3D clipping
            if self.viewer.current_tetrahedra or self.viewer.current_hexahedra:
                self.viewer.current_volumetric_grid = vtk.vtkUnstructuredGrid()
                self.viewer.current_volumetric_grid.SetPoints(points)
                
                for tet in self.viewer.current_tetrahedra:
                    vtk_tet = vtk.vtkTetra()
                    for i, nid in enumerate(tet['nodes']):
                        vtk_tet.GetPointIds().SetId(i, node_map[nid])
                    self.viewer.current_volumetric_grid.InsertNextCell(vtk_tet.GetCellType(), vtk_tet.GetPointIds())
                
                for hex_elem in self.viewer.current_hexahedra:
                    vtk_hex = vtk.vtkHexahedron()
                    for i, nid in enumerate(hex_elem['nodes']):
                        vtk_hex.GetPointIds().SetId(i, node_map[nid])
                    self.viewer.current_volumetric_grid.InsertNextCell(vtk_hex.GetCellType(), vtk_hex.GetPointIds())
            else:
                self.viewer.current_volumetric_grid = None
            
            # Apply quality colors if available
            if result and result.get('quality_metrics') and result.get('per_element_quality'):
                try:
                    per_elem_quality = result['per_element_quality']
                    surface_elements = [e for e in elements if e['type'] in ('triangle', 'quadrilateral')]
                    
                    all_qualities = [q for q in per_elem_quality.values() if q is not None]
                    if all_qualities:
                        global_min, global_max = min(all_qualities), max(all_qualities)
                    else:
                        global_min, global_max = 0.0, 1.0
                    
                    colors = vtk.vtkUnsignedCharArray()
                    colors.SetNumberOfComponents(3)
                    colors.SetName("Colors")

                    def hsl_to_rgb(h, s, l):
                        def hue_to_rgb(p, q, t):
                            if t < 0: t += 1
                            if t > 1: t -= 1
                            if t < 1/6: return p + (q - p) * 6 * t
                            if t < 1/2: return q
                            if t < 2/3: return p + (q - p) * (2/3 - t) * 6
                            return p
                        if s == 0: r = g = b = l
                        else:
                            q = l * (1 + s) if l < 0.5 else l + s - l * s
                            p = 2 * l - q
                            r = hue_to_rgb(p, q, h + 1/3)
                            g = hue_to_rgb(p, q, h)
                            b = hue_to_rgb(p, q, h - 1/3)
                        return int(r * 255), int(g * 255), int(b * 255)

                    for element in surface_elements:
                        elem_id = element['id']
                        quality = per_elem_quality.get(elem_id, per_elem_quality.get(str(elem_id), None))

                        if quality is None:
                            colors.InsertNextTuple3(150, 150, 150)
                        else:
                            quality_range = global_max - global_min
                            if quality_range > 0.0001:
                                normalized = (quality - global_min) / quality_range
                            else:
                                normalized = 1.0
                            
                            normalized = max(0.0, min(1.0, normalized))
                            hue = normalized * 0.33
                            r, g, b = hsl_to_rgb(hue, 1.0, 0.5)
                            colors.InsertNextTuple3(r, g, b)

                    poly_data.GetCellData().SetScalars(colors)
                except Exception as e:
                    print(f"[DEBUG ERROR] Could not apply quality colors: {e}")

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(poly_data)

            if result and result.get('per_element_quality') and poly_data.GetCellData().GetScalars():
                mapper.SetScalarModeToUseCellData()
                mapper.ScalarVisibilityOn()
                mapper.SetColorModeToDirectScalars()
            else:
                mapper.ScalarVisibilityOff()

            self.viewer.current_actor = vtk.vtkActor()
            self.viewer.current_actor.SetMapper(mapper)
            print(f"[MESH_LOADER DEBUG] Created actor: {self.viewer.current_actor}")
            print(f"[MESH_LOADER DEBUG] Mapper: {mapper}")

            if not (result and result.get('per_element_quality')):
                self.viewer.current_actor.GetProperty().SetColor(0.2, 0.7, 0.4)

            self.viewer.current_actor.GetProperty().SetOpacity(1.0)
            self.viewer.current_actor.GetProperty().SetInterpolationToFlat()
            
            if result and result.get('per_element_quality'):
                self.viewer.current_actor.GetProperty().SetAmbient(0.8)
                self.viewer.current_actor.GetProperty().SetDiffuse(0.5)
                self.viewer.current_actor.GetProperty().SetSpecular(0.0)
            else:
                self.viewer.current_actor.GetProperty().SetAmbient(0.4)
                self.viewer.current_actor.GetProperty().SetDiffuse(0.7)
                self.viewer.current_actor.GetProperty().SetSpecular(0.2)
                self.viewer.current_actor.GetProperty().SetSpecularPower(15)

            self.viewer.current_actor.GetProperty().EdgeVisibilityOn()
            self.viewer.current_actor.GetProperty().SetEdgeColor(0.0, 0.0, 0.0)
            self.viewer.current_actor.GetProperty().SetLineWidth(1.0)

            print(f"[MESH_LOADER DEBUG] Adding actor to renderer...")
            self.viewer.renderer.AddActor(self.viewer.current_actor)
            print(f"[MESH_LOADER DEBUG] Actors in renderer: {self.viewer.renderer.GetActors().GetNumberOfItems()}")
            self.viewer.renderer.ResetCamera()
            print(f"[MESH_LOADER DEBUG] Camera reset, calling Render()...")
            self.viewer.vtk_widget.GetRenderWindow().Render()
            print(f"[MESH_LOADER DEBUG] Render() complete")

            # Update Info Label
            if result:
                display_nodes = result.get('total_nodes', len(nodes))
                display_elements = result.get('total_elements', len(elements))
            else:
                display_nodes = len(nodes)
                display_elements = len(elements)

            info_lines = [
                f"<b>Mesh Generated</b><br>",
                f"{Path(filepath).name}<br>",
                f"<span style='color: #6c757d;'>",
                f"Nodes: {display_nodes:,} * Elements: {display_elements:,}<br>",
                f"Tetrahedra: {tet_count:,} * Triangles: {tri_count:,}"
            ]

            if result and result.get('quality_metrics'):
                metrics = result['quality_metrics']
                info_lines.append("<br><b>Quality Metrics (avg):</b><br>")
                if 'sicn_avg' in metrics:
                    sicn = metrics['sicn_avg']
                    sicn_color = "#198754" if sicn >= 0.7 else "#ffc107" if sicn >= 0.5 else "#dc3545"
                    info_lines.append(f"<span style='color: {sicn_color};'>SICN: {sicn:.3f}</span> ")
                if 'gamma_avg' in metrics:
                    gamma = metrics['gamma_avg']
                    gamma_color = "#198754" if gamma >= 0.6 else "#ffc107" if gamma >= 0.4 else "#dc3545"
                    info_lines.append(f"<span style='color: {gamma_color};'>γ: {gamma:.3f}</span><br>")

            self.viewer.info_label.setText("".join(info_lines))
            self.viewer.info_label.adjustSize()
            
            return "SUCCESS"

        except Exception as e:
            error_msg = f"Error loading mesh: {str(e)}"
            print(f"[DEBUG ERROR] {error_msg}")
            import traceback
            traceback.print_exc()
            self.viewer.info_label.setText(error_msg)
            return f"ERROR: {e}"

    def _parse_msh_file(self, filepath: str):
        nodes = {}
        elements = []

        with open(filepath, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if line == "$Nodes":
                i += 2
                while lines[i].strip() != "$EndNodes":
                    parts = lines[i].strip().split()
                    if len(parts) == 4:
                        num_nodes = int(parts[3])
                        i += 1
                        node_tags = []
                        for _ in range(num_nodes):
                            node_tags.append(int(lines[i].strip()))
                            i += 1
                        for tag in node_tags:
                            coords = lines[i].strip().split()
                            nodes[tag] = [float(coords[0]), float(coords[1]), float(coords[2])]
                            i += 1
                    else:
                        i += 1

            elif line == "$Elements":
                i += 2
                while lines[i].strip() != "$EndElements":
                    parts = lines[i].strip().split()
                    if len(parts) == 4:
                        element_type = int(parts[2])
                        num_elements = int(parts[3])
                        i += 1

                        if element_type == 4: # Tet4
                            for _ in range(num_elements):
                                data = lines[i].strip().split()
                                if len(data) >= 5:
                                    elements.append({
                                        "id": int(data[0]),
                                        "type": "tetrahedron",
                                        "nodes": [int(data[1]), int(data[2]), int(data[3]), int(data[4])]
                                    })
                                i += 1
                        elif element_type == 5: # Hex8
                            for _ in range(num_elements):
                                data = lines[i].strip().split()
                                if len(data) >= 9:
                                    elements.append({
                                        "id": int(data[0]),
                                        "type": "hexahedron",
                                        "nodes": [int(data[j]) for j in range(1, 9)]
                                    })
                                i += 1
                        elif element_type == 2: # Tri3
                            for _ in range(num_elements):
                                data = lines[i].strip().split()
                                if len(data) >= 4:
                                    elements.append({
                                        "id": int(data[0]),
                                        "type": "triangle",
                                        "nodes": [int(data[1]), int(data[2]), int(data[3])]
                                    })
                                i += 1
                        elif element_type == 3: # Quad4
                            for _ in range(num_elements):
                                data = lines[i].strip().split()
                                if len(data) >= 5:
                                    elements.append({
                                        "id": int(data[0]),
                                        "type": "quadrilateral",
                                        "nodes": [int(data[1]), int(data[2]), int(data[3]), int(data[4])]
                                    })
                                i += 1
                        else:
                            for _ in range(num_elements):
                                i += 1
                    else:
                        i += 1
            else:
                i += 1

        return nodes, elements
