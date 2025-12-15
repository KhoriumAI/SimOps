
import sys
from pathlib import Path
import pyvista as pv

class MockViewer:
    def __init__(self):
        self.info_label = self
        self.renderer = self
        self.vtk_widget = self
    def setText(self, text): print(f"[GUI LABEL] {text}")
    def clear_view(self): print("[GUI] View cleared")
    def AddActor(self, actor): print("[GUI] Actor added")
    def ResetCamera(self): print("[GUI] Camera reset")
    def Render(self): print("[GUI] Render called")
    def GetRenderWindow(self): return self

class MeshLoader:
    def __init__(self, viewer):
        self.viewer = viewer

    def load_mesh(self, filepath):
        print(f"Testing load: {filepath}")
        
        # 1. Parse .msh
        nodes = {}
        elements = []
        try:
            print("Parsing .msh...")
            # Minimal parser simulation
            with open(filepath, 'r') as f:
                if "$MeshFormat" in f.read(100):
                    print("  -> Gmsh header found")
                else:
                    print("  -> No Gmsh header (Fluent format?)")
        except Exception as e:
            print(f"Parse error: {e}")

        # 2. Check Fallback
        if not nodes or not elements:
            print("Nodes/Elements empty. Checking fallback...")
            vtk_path = Path(filepath).with_suffix('.vtk')
            if not vtk_path.exists():
                print(f"  -> .vtk not found: {vtk_path}")
                vtk_path = Path(filepath).with_suffix('.vtu')
            
            if vtk_path.exists():
                print(f"  -> Found fallback: {vtk_path}")
                try:
                    mesh = pv.read(str(vtk_path))
                    print(f"  -> PV Read Success: {mesh.n_points} points, {mesh.n_cells} cells")
                    if mesh.n_points > 0:
                        print("  -> SUCCCESS: Mesh has points.")
                        return "SUCCESS"
                    else:
                        print("  -> FAIL: Mesh has 0 points.")
                except Exception as e:
                    print(f"  -> PV Read Failed: {e}")
            else:
                print("  -> No fallback file found")
        
        return "FALLTHROUGH"

# Run it
loader = MeshLoader(MockViewer())
mesh_path = "C:/Users/Owner/Downloads/MeshPackageLean/apps/cli/generated_meshes/model_openfoam_hex.msh"
result = loader.load_mesh(mesh_path)
print(f"Result: {result}")
