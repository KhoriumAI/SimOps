"""
Component Visualizer
====================

Handles visualization of multi-component meshes (e.g., CoACD results).
"""

import vtk
import numpy as np
from pathlib import Path
from typing import Dict

class ComponentVisualizer:
    def __init__(self, viewer):
        self.viewer = viewer

    def load_visualization(self, result: Dict):
        """Load and display CoACD components with PyVista"""
        debug_log = Path("poly_debug.txt")
        with open(debug_log, 'w') as f:
            f.write("=== COMPONENT VIZ DEBUG (Refactored) ===\n")
            f.write(f"Function called\n")
            f.write(f"Result keys: {list(result.keys())}\n")
        
        print("="*80)
        print("[COMPONENT-VIZ] *** FUNCTION CALLED *** IN COMPONENT_VIZ.PY")
        print("="*80)
        
        try:
            import pyvista as pv
            
            component_files = result.get('component_files', [])
            if not component_files:
                print("[ERROR] No component files in result")
                self.viewer.info_label.setText("Error: No components found")
                return "FAILED"
            
            print(f"[COMPONENT-VIZ] Loading {len(component_files)} components...")
            self.viewer.clear_view()
            
            # Load and merge all components
            merged_mesh = None
            for i, comp_file in enumerate(component_files):
                if not Path(comp_file).exists():
                    print(f"[WARNING] Component file not found: {comp_file}")
                    continue
                
                # Load with PyVista
                comp_mesh = pv.read(comp_file)
                
                # Ensure Component_ID scalar exists
                if "Component_ID" not in comp_mesh.array_names:
                    comp_mesh["Component_ID"] = np.full(comp_mesh.n_cells, i, dtype=int)
                
                # Merge into single mesh
                if merged_mesh is None:
                    merged_mesh = comp_mesh
                else:
                    merged_mesh = merged_mesh.merge(comp_mesh)
                
                print(f"[COMPONENT-VIZ] Loaded component {i}: {comp_mesh.n_cells} cells")
            
            if merged_mesh is None:
                print("[ERROR] No components loaded")
                self.viewer.info_label.setText("Error: Failed to load components")
                return "FAILED"
            
            # Always extract surface for display, but check if we have volume data
            ugrid = merged_mesh.cast_to_unstructured_grid()
            
            # Filter to keep only volume cells if they exist (Robust check)
            cell_types = ugrid.celltypes
            # Check for 3D cell types: Tet(10), Voxel(11), Hex(12), Wedge(13), Pyramid(14)
            vol_mask = np.isin(cell_types, [vtk.VTK_HEXAHEDRON, vtk.VTK_TETRA, vtk.VTK_WEDGE, vtk.VTK_PYRAMID, 
                                          vtk.VTK_HEXAHEDRON, 11, 12, 10, 13, 14])
            
            if np.any(vol_mask):
                print(f"[COMPONENT-VIZ] Found {np.sum(vol_mask)} volume cells! Filtering...")
                ugrid = ugrid.extract_cells(vol_mask)
                has_volume_cells = True
            else:
                print("[COMPONENT-VIZ] No volume cells found in successful meshes")
                has_volume_cells = False
            
            polydata = ugrid.extract_surface()
            
            print(f"[COMPONENT-VIZ] Displaying surface with {polydata.GetNumberOfCells()} faces (from {ugrid.n_cells} volume cells)" if has_volume_cells else f"[COMPONENT-VIZ] Displaying surface mesh")
            
            # Create mapper with categorical colors
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(polydata)
            mapper.SetScalarModeToUseCellData()
            mapper.SetScalarRange(0, result.get('num_components', 10))
            mapper.ScalarVisibilityOn()
            
            # Create categorical color lookup table
            lut = vtk.vtkLookupTable()
            num_colors = max(10, result.get('num_components', 10))
            lut.SetNumberOfTableValues(num_colors)
            lut.Build()
            
            # Use distinct colors (tab20 colormap approximation)
            colors = [
                (0.12, 0.47, 0.71), (1.0, 0.50, 0.05), (0.17, 0.63, 0.17),
                (0.84, 0.15, 0.16), (0.58, 0.40, 0.74), (0.55, 0.34, 0.29),
                (0.89, 0.47, 0.76), (0.50, 0.50, 0.50), (0.74, 0.74, 0.13),
                (0.09, 0.75, 0.81), (0.68, 0.78, 0.91), (1.0, 0.73, 0.47),
                (0.60, 0.87, 0.54), (1.0, 0.60, 0.60), (0.79, 0.70, 0.84),
                (0.78, 0.70, 0.65), (0.98, 0.75, 0.83), (0.78, 0.78, 0.78),
                (0.86, 0.86, 0.55), (0.62, 0.85, 0.88)
            ]
            for i in range(num_colors):
                color = colors[i % len(colors)]
                lut.SetTableValue(i, color[0], color[1], color[2], 1.0)
            
            mapper.SetLookupTable(lut)
            
            # Create actor
            self.viewer.current_actor = vtk.vtkActor()
            self.viewer.current_actor.SetMapper(mapper)
            
            # Set initial opacity
            self.viewer.current_actor.GetProperty().SetOpacity(1.0)
            
            # Increase ambient lighting to reduce dark shadows
            self.viewer.current_actor.GetProperty().SetAmbient(0.6)  # High ambient for visibility
            self.viewer.current_actor.GetProperty().SetDiffuse(0.6)
            self.viewer.current_actor.GetProperty().SetSpecular(0.2)
            
            # Enable edge visibility to show hex mesh structure
            if has_volume_cells:
                self.viewer.current_actor.GetProperty().SetEdgeVisibility(True)
                self.viewer.current_actor.GetProperty().SetEdgeColor(0.2, 0.2, 0.2)  # Dark gray edges
                self.viewer.current_actor.GetProperty().SetLineWidth(1.0)
            
            # Add to renderer
            self.viewer.renderer.AddActor(self.viewer.current_actor)
            
            # Create scalar bar for component IDs
            scalar_bar = vtk.vtkScalarBarActor()
            scalar_bar.SetLookupTable(lut)
            scalar_bar.SetTitle("Component ID")
            scalar_bar.SetNumberOfLabels(min(num_colors, 10))
            scalar_bar.SetPosition(0.85, 0.1)
            scalar_bar.SetWidth(0.12)
            scalar_bar.SetHeight(0.8)
            self.viewer.renderer.AddActor2D(scalar_bar)
            
            # Update info label
            self.viewer.info_label.setText(f"Components: {len(component_files)}")
            
            # Reset camera
            self.viewer.renderer.ResetCamera()
            self.viewer.vtk_widget.GetRenderWindow().Render()
            
            return "SUCCESS"
            
        except Exception as e:
            print(f"[COMPONENT-VIZ ERROR] {e}")
            import traceback
            traceback.print_exc()
            return "FAILED"
