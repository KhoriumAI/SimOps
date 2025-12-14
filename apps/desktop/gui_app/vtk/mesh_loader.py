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

            print(f"[CAD PREVIEW] Loading STL: {tmp_stl}")
            mesh = pv.read(tmp_stl)
            os.unlink(tmp_stl)

            print(f"[CAD PREVIEW] Mesh stats: {mesh.n_points} points, {mesh.n_cells} cells")
            
            if mesh.n_points == 0:
                raise Exception("Empty mesh - no geometry in CAD file")

            poly_data = mesh.cast_to_unstructured_grid().extract_surface()
            print(f"[CAD PREVIEW] Surface extracted: {poly_data.GetNumberOfPoints()} points, {poly_data.GetNumberOfCells()} cells")
            
            # Apply smooth normals for better CAD visualization
            try:
                import vtk
                normals_gen = vtk.vtkPolyDataNormals()
                normals_gen.SetInputData(poly_data)
                normals_gen.ComputePointNormalsOn()
                normals_gen.ComputeCellNormalsOff()
                normals_gen.SplittingOn()
                normals_gen.SetFeatureAngle(60.0)
                normals_gen.Update()
                smooth_poly_data = normals_gen.GetOutput()
                print(f"[CAD PREVIEW] Smooth normals generated")
                self.viewer.current_poly_data = smooth_poly_data
            except Exception as e:
                print(f"[CAD PREVIEW] Normals failed: {e}, using raw mesh")
                smooth_poly_data = poly_data
                self.viewer.current_poly_data = poly_data

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(smooth_poly_data)

            self.viewer.current_actor = vtk.vtkActor()
            self.viewer.current_actor.SetMapper(mapper)
            
            self.viewer.current_actor.GetProperty().SetColor(0.3, 0.5, 0.8)
            self.viewer.current_actor.GetProperty().SetInterpolationToPhong()
            self.viewer.current_actor.GetProperty().EdgeVisibilityOff()
            self.viewer.current_actor.GetProperty().BackfaceCullingOff()  # Fix for complex manifolds
            self.viewer.current_actor.GetProperty().ShadingOn()
            self.viewer.current_actor.GetProperty().SetAmbient(0.3)
            self.viewer.current_actor.GetProperty().SetDiffuse(0.7)
            self.viewer.current_actor.GetProperty().SetSpecular(0.5)
            self.viewer.current_actor.GetProperty().SetSpecularPower(40.0)

            print(f"[CAD PREVIEW] Adding actor to renderer")
            self.viewer.renderer.AddActor(self.viewer.current_actor)
            self.viewer.renderer.ResetCamera()
            self.viewer.vtk_widget.GetRenderWindow().Render()
            print(f"[CAD PREVIEW] Render complete")

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
            # Parse mesh file
            # If it's a Fluent .msh, _parse_msh_file (Gmsh) will return empty or error.
            # We add a fallback to check for a sibling .vtk file.
            
            nodes, elements = {}, []
            try:
                nodes, elements = self._parse_msh_file(filepath)
            except Exception:
                pass # Fallback to VTK check
                
            if not nodes or not elements:
                print(f"[MESH_LOADER DEBUG] Parsing returned empty/none. Nodes: {len(nodes) if nodes else 0}, Elements: {len(elements) if elements else 0}")
                # Check for sibling .vtk file (e.g. model.vtk or model_fluent.vtk)
                # If filepath is model.msh, check model.vtk or model.vtu
                vtk_path = Path(filepath).with_suffix('.vtk')
                if not vtk_path.exists():
                     vtk_path = Path(filepath).with_suffix('.vtu')
                     
                if not vtk_path.exists():
                     print(f"[MESH_LOADER DEBUG] Fallback VTK/VTU not found at {vtk_path}")
                     pass 
                else:
                    print(f"[MESH_LOADER] Loading VTK fallback: {vtk_path}")
                    import pyvista as pv
                    try:
                        mesh = pv.read(str(vtk_path))
                        print(f"[MESH_LOADER DEBUG] PyVista loaded mesh: {type(mesh)}, Points: {mesh.n_points}, Cells: {mesh.n_cells}")
                        
                        # Convert PyVista mesh to VTK Generic
                        # For rendering, we can just use the polydata directly
                        if mesh.n_points > 0:
                            poly_data = mesh.cast_to_unstructured_grid().extract_surface()
                            self.viewer.current_poly_data = poly_data
                            
                            mapper = vtk.vtkPolyDataMapper()
                            mapper.SetInputData(poly_data)
                            
                            self.viewer.current_actor = vtk.vtkActor()
                            self.viewer.current_actor.SetMapper(mapper)
                            
                            # Set default style
                            self.viewer.current_actor.GetProperty().SetColor(0.2, 0.7, 0.4)
                            self.viewer.current_actor.GetProperty().EdgeVisibilityOn()
                            
                            self.viewer.renderer.AddActor(self.viewer.current_actor)
                            self.viewer.renderer.ResetCamera()
                            self.viewer.vtk_widget.GetRenderWindow().Render()
                            
                            self.viewer.info_label.setText(
                                f"<b>Mesh Loaded (VTK)</b><br>"
                                f"{vtk_path.name}<br>"
                                f"Nodes: {mesh.n_points:,} * Cells: {mesh.n_cells:,}"
                            )
                            return "SUCCESS"
                        else:
                            print("[MESH_LOADER DEBUG] PyVista mesh has 0 points!")
                    except Exception as e:
                        print(f"[MESH_LOADER DEBUG] Failed to load PyVista fallback: {e}")
            
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
                f"Hexes: {hex_count:,} * Tets: {tet_count:,} * Tris: {tri_count:,}"
            ]

            if result and result.get('quality_metrics'):
                metrics = result['quality_metrics']
                info_lines.append("<br><b>Quality Metrics (avg):</b><br>")
                
                if 'jacobian_avg' in metrics:
                    jac = metrics['jacobian_avg']
                    jac_color = "#198754" if jac >= 0.7 else "#ffc107" if jac >= 0.3 else "#dc3545"
                    info_lines.append(f"<span style='color: {jac_color};'>Jacobian: {jac:.3f}</span><br>")
                elif 'sicn_avg' in metrics:
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
            lines = [l.strip() for l in f.readlines()]
            
        # Detect version
        version = 4.1 # Default assumption if no header (though standard requires it)
        i = 0
        while i < len(lines):
            line = lines[i]
            if line == "$MeshFormat":
                i += 1
                if i < len(lines):
                    parts = lines[i].split()
                    if len(parts) >= 1:
                        try:
                            version = float(parts[0])
                        except:
                            pass
                break
            i += 1
            
        print(f"[MESH_LOADER DEBUG] Detected Gmsh version: {version}")

        i = 0
        while i < len(lines):
            line = lines[i]

            if line == "$Nodes":
                i += 1
                if i >= len(lines): break
                
                # Gmsh 2.2: 
                # num_nodes
                # id x y z
                # ...
                
                # Gmsh 4.1:
                # num_blocks num_nodes ...
                # block_header
                # ...
                
                if version < 3.0:
                    # Gmsh 2.x parsing
                    try:
                        num_nodes = int(lines[i])
                        i += 1
                        for _ in range(num_nodes):
                            if i >= len(lines): break
                            parts = lines[i].split()
                            if len(parts) >= 4:
                                nid = int(parts[0])
                                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                                nodes[nid] = [x, y, z]
                            i += 1
                        # Skip $EndNodes
                        if i < len(lines) and lines[i] == "$EndNodes":
                            i += 1
                        continue # loop back
                    except ValueError:
                        print("[MESH_LOADER ERROR] Failed parsing Gmsh 2.2 Nodes")
                        i += 1
                        continue
                else:
                    # Gmsh 4.x parsing (Existing logic was 4.x based)
                    # existing logic skipped the count line: i += 2 (current i is at Header)
                    # Let's adjust to be robust using the detected logic
                    
                    # Original code skipped 2 lines from $Nodes.
                    # line[i] is now at num_blocks... (after $Nodes)
                    i += 1 # Skip num_blocks line
                    
                    while i < len(lines) and lines[i] != "$EndNodes":
                        parts = lines[i].split()
                        if len(parts) == 4: # Block header
                            try:
                                num_nodes_in_block = int(parts[3])
                                i += 1
                                node_tags = []
                                for _ in range(num_nodes_in_block):
                                    node_tags.append(int(lines[i]))
                                    i += 1
                                for tag in node_tags:
                                    coords = lines[i].split()
                                    nodes[tag] = [float(coords[0]), float(coords[1]), float(coords[2])]
                                    i += 1
                            except ValueError:
                                i += 1
                        else:
                            i += 1
                    
                    if i < len(lines) and lines[i] == "$EndNodes":
                        i += 1
                    continue

            elif line == "$Elements":
                i += 1
                if i >= len(lines): break
                
                if version < 3.0:
                    # Gmsh 2.2 Elements
                    # num_elements
                    # id type num_tags tag1 ... node1 ...
                    try:
                        num_elements = int(lines[i])
                        i += 1
                        for _ in range(num_elements):
                            if i >= len(lines): break
                            parts = lines[i].split()
                            # id, type, num_tags, ...
                            if len(parts) >= 3:
                                eid = int(parts[0])
                                etype = int(parts[1])
                                num_tags = int(parts[2])
                                # nodes start after tags
                                # parts index: 0=id, 1=type, 2=num_tags, 3..3+num_tags-1=tags, 3+num_tags...=nodes
                                node_start_idx = 3 + num_tags
                                node_ids = [int(n) for n in parts[node_start_idx:]]
                                
                                elem_data = None
                                if etype == 2: # Triangle 3-node
                                    elem_data = {"id": eid, "type": "triangle", "nodes": node_ids}
                                elif etype == 3: # Quad 4-node
                                    elem_data = {"id": eid, "type": "quadrilateral", "nodes": node_ids}
                                elif etype == 4: # Tet 4-node
                                    elem_data = {"id": eid, "type": "tetrahedron", "nodes": node_ids}
                                elif etype == 5: # Hex 8-node
                                    elem_data = {"id": eid, "type": "hexahedron", "nodes": node_ids}
                                
                                if elem_data:
                                    elements.append(elem_data)
                            i += 1
                        if i < len(lines) and lines[i] == "$EndElements":
                             i += 1
                        continue
                    except ValueError:
                         print("[MESH_LOADER ERROR] Failed parsing Gmsh 2.2 Elements")
                         i += 1
                         continue

                else:
                    # Gmsh 4.x Elements
                    # Original logic was:
                    # i += 2 (skip header and counts?)
                    # while != $EndElements
                    #   parts[2] is type, parts[3] is num
                    
                    # line[i] is num_blocks...
                    i += 1 
                    
                    while i < len(lines) and lines[i] != "$EndElements":
                        parts = lines[i].split()
                        if len(parts) == 4:
                            element_type = int(parts[2])
                            num_elements_in_block = int(parts[3])
                            i += 1
                            
                            type_str = None
                            expected_nodes = 0
                            if element_type == 2: 
                                type_str = "triangle"
                                expected_nodes = 3
                            elif element_type == 3: 
                                type_str = "quadrilateral"
                                expected_nodes = 4
                            elif element_type == 4: 
                                type_str = "tetrahedron"
                                expected_nodes = 4
                            elif element_type == 5: 
                                type_str = "hexahedron"
                                expected_nodes = 8
                                
                            if type_str:
                                for _ in range(num_elements_in_block):
                                    data = lines[i].split()
                                    # Gmsh 4.1: tag is first? No, ID is first.
                                    # Actually 4.1 format in blocks usually just lists node tags?
                                    # Format: elementTag nodeTag1 ... nodeTagN
                                    if len(data) >= 1 + expected_nodes:
                                        eid = int(data[0])
                                        enodes = [int(x) for x in data[1:1+expected_nodes]]
                                        elements.append({
                                            "id": eid,
                                            "type": type_str,
                                            "nodes": enodes
                                        })
                                    i += 1
                            else:
                                i += num_elements_in_block
                        else:
                            i += 1
                    
                    if i < len(lines) and lines[i] == "$EndElements":
                        i += 1
                    continue

            else:
                i += 1

        return nodes, elements
