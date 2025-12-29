import sys
import os
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    # Set fake display for VTK if needed (though we only test logic here)
    # os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    from apps.desktop.gui_app.vtk_viewer import VTK3DViewer
    import vtk
    from PyQt5.QtWidgets import QApplication
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def test_quality_calculation():
    # Create a dummy QApplication for VTK viewer
    app = QApplication(sys.argv)
    
    # Instantiate viewer (it might fail if VTK/Qt integration is not perfect in this environment)
    try:
        # We need to mock the constructor to avoid UI initialization if possible
        # but let's try direct instantiation first
        viewer = VTK3DViewer()
    except Exception as e:
        print(f"Could not instantiate VTK3DViewer (expected in headless): {e}")
        print("Testing method in isolation...")
        # If it fails, we can just test the method by injecting it into a dummy object
        class DummyViewer:
            hsl_to_rgb = staticmethod(lambda h, s, l: (0,0,0)) # placeholder
        
        from apps.desktop.gui_app.vtk_viewer import VTK3DViewer
        viewer = DummyViewer()
        viewer._calculate_mesh_quality = VTK3DViewer._calculate_mesh_quality.__get__(viewer)

    # Test meshes
    test_meshes = [
        Path("apps/desktop/test_cube.msh"),
        Path("cad_files/Cylinder.clean.stl")
    ]
    
    for test_mesh in test_meshes:
        if not test_mesh.exists():
            print(f"Test mesh {test_mesh} not found, skipping...")
            continue

        print(f"\n{'-'*40}")
        print(f"Testing quality calculation for: {test_mesh}")
        
        # Test _calculate_mesh_quality
        result = viewer._calculate_mesh_quality(str(test_mesh))
        
        if result:
            print("Quality calculation SUCCESS")
            print(f"Metrics: {result.get('quality_metrics')}")
            
            # Verify keys
            expected_keys = ['per_element_quality', 'quality_metrics']
            for key in expected_keys:
                if key in result:
                    print(f"  Found {key}")
                else:
                    print(f"  MISSING {key}")
            
            # Verify metrics keys
            metric_keys = ['sicn_min', 'sicn_avg', 'sicn_max', 'sicn_10_percentile']
            for key in metric_keys:
                if key in result['quality_metrics']:
                    print(f"    Found metric: {key} = {result['quality_metrics'][key]:.4f}")
                else:
                    print(f"    MISSING metric: {key}")
        else:
            print("Quality calculation FAILED")

if __name__ == "__main__":
    test_quality_calculation()
