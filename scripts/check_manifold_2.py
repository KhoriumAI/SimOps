import os
import sys
import vtk
import gmsh
import math
import tempfile

def check_step_is_manifold(step_file_path, mesh_size_max=None, mesh_size_min=None):
    """
    Check if a STEP file contains manifold geometry.
    
    Args:
        step_file_path: Path to the STEP file to check
        mesh_size_max: Optional max mesh size (mm). If None, calculated as diag/20
        mesh_size_min: Optional min mesh size (mm). If None, calculated as diag/100
    
    Returns:
        bool: True if geometry is manifold, False otherwise
    """
    # Print only our own status messages
    print(f"Checking: {step_file_path}")
    
    if not os.path.exists(step_file_path):
        print("RESULT: FAIL (File not found)")
        return False

    # --- 1. SILENCE VTK LOGGING ---
    
    # A. Silence the new VTK 9.x Logger (This stops the console spam)
    # This was present in your original check_manifold.py lines 29-32
    try:
        vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)
    except AttributeError:
        pass # Handle older VTK versions gracefully

    # B. Capture Legacy VTK Errors (So we can read them without printing)
    log_capture = vtk.vtkStringOutputWindow()
    vtk.vtkOutputWindow.SetInstance(log_capture)

    tmp_stl = None
    
    try:
        # --- PHASE 2: GMSH CONVERSION ---
        # (Exact settings from vtk_viewer.py)
        
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0) # Silence Gmsh
        gmsh.open(step_file_path)

        # BBox & Sizing - use provided values or calculate from bbox diagonal
        bbox = gmsh.model.getBoundingBox(-1, -1)
        dims = [bbox[3]-bbox[0], bbox[4]-bbox[1], bbox[5]-bbox[2]]
        diag = (dims[0]**2 + dims[1]**2 + dims[2]**2)**0.5

        # Use user-provided sizes or fall back to bbox-based defaults
        actual_min_size = mesh_size_min if mesh_size_min is not None else diag / 100.0
        actual_max_size = mesh_size_max if mesh_size_max is not None else diag / 20.0
        
        gmsh.option.setNumber("Mesh.MeshSizeMin", actual_min_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", actual_max_size)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)


        gmsh.model.mesh.generate(2)
        
        fd, tmp_stl = tempfile.mkstemp(suffix=".stl")
        os.close(fd)
        gmsh.write(tmp_stl)
        gmsh.finalize()

        # --- PHASE 3: VTK PIPELINE ---
        
        # A. Read STL
        reader = vtk.vtkSTLReader()
        reader.SetFileName(tmp_stl)
        reader.Update()
        
        # B. Clean PolyData (Simulate PyVista topology)
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(reader.GetOutput())
        cleaner.Update()
        poly_data = cleaner.GetOutput()
        
        cell_count = poly_data.GetNumberOfCells()
        processing_poly_data = poly_data

        # C. Conditional Subdivision (<10k cells limit)
        if cell_count < 10000:
            print(f" -> Small mesh ({cell_count} cells). Running Subdivision check...")
            subdivision = vtk.vtkLinearSubdivisionFilter()
            subdivision.SetInputData(poly_data)
            subdivision.SetNumberOfSubdivisions(1)
            # This Update() will quietly log errors to 'log_capture' if broken
            subdivision.Update()
            processing_poly_data = subdivision.GetOutput()
        else:
            print(f" -> Large mesh ({cell_count} cells). Skipping Subdivision.")

        # D. Normals
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(processing_poly_data)
        normals.ComputePointNormalsOn()
        normals.ComputeCellNormalsOff()
        normals.SplittingOn()
        normals.SetFeatureAngle(60.0)
        normals.Update()

        # --- PHASE 4: VERIFY LOGS ---
        errors = log_capture.GetOutput()
        
        # Check specifically for the subdivision failure
        if "Subdivision failed" in errors or "non-manifold" in errors:
            print("RESULT: FAIL (Non-manifold geometry detected)")
            return False
            
        # Strict Mode: Fail on any other VTK error
        if "Error" in errors or "ERR" in errors:
            # Clean up the error message for display (take first line only)
            first_error = errors.strip().split('\n')[0]
            print(f"RESULT: FAIL (VTK Error: {first_error})")
            return False

        print("RESULT: PASS")
        return True

    except Exception as e:
        print(f"RESULT: FAIL (Crash: {e})")
        return False
        
    finally:
        if tmp_stl and os.path.exists(tmp_stl):
            try:
                os.remove(tmp_stl)
            except:
                pass
        try:
            if gmsh.isInitialized(): gmsh.finalize()
        except:
            pass

if __name__ == "__main__":
    # CRASH_STEP_FILE = "./orchestrator/batch_data/queue/s79160335.step"
    CRASH_STEP_FILE = "./orchestrator/batch_data/queue/s79160295.step"
    check_step_is_manifold(CRASH_STEP_FILE)