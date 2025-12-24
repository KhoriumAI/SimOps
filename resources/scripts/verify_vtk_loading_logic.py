
import vtk
import os
import sys

def create_dummy_vtu(filename):
    print(f"Creating dummy VTU: {filename}")
    
    # Create points
    points = vtk.vtkPoints()
    points.InsertNextPoint(0, 0, 0)
    points.InsertNextPoint(1, 0, 0)
    points.InsertNextPoint(0, 1, 0)
    points.InsertNextPoint(0, 0, 1)
    
    # Create a tet
    tet = vtk.vtkTetra()
    tet.GetPointIds().SetId(0, 0)
    tet.GetPointIds().SetId(1, 1)
    tet.GetPointIds().SetId(2, 2)
    tet.GetPointIds().SetId(3, 3)
    
    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(points)
    grid.InsertNextCell(tet.GetCellType(), tet.GetPointIds())
    
    # Write
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(grid)
    writer.Write()
    print("Write complete.")

def verify_load_vtu(filename):
    print(f"Verifying load of: {filename}")
    
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    
    ugrid = reader.GetOutput()
    print(f"Loaded: {ugrid.GetNumberOfPoints()} points, {ugrid.GetNumberOfCells()} cells")
    
    if ugrid.GetNumberOfCells() != 1:
        print("FAIL: Expected 1 cell")
        return False
        
    # Test geometry filter (surface extraction)
    print("Testing surface extraction...")
    geo = vtk.vtkGeometryFilter()
    geo.SetInputData(ugrid)
    geo.Update()
    poly = geo.GetOutput()
    print(f"Surface: {poly.GetNumberOfCells()} cells (Expected 4 faces for a tet)")
    
    if poly.GetNumberOfCells() != 4:
        print("FAIL: Expected 4 surface faces")
        return False
        
    print("SUCCESS: VTU loading logic is valid.")
    return True

if __name__ == "__main__":
    vtu_file = "test_verification.vtu"
    try:
        create_dummy_vtu(vtu_file)
        if verify_load_vtu(vtu_file):
            print("VERIFICATION PASSED")
        else:
            print("VERIFICATION FAILED")
    except Exception as e:
        print(f"EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(vtu_file):
            os.remove(vtu_file)
