import os
import tempfile
import logging

# Lazy imports to prevent worker startup lag
try:
    import vtk
    import gmsh
except ImportError:
    pass

class TopologyInspector:
    """
    Wraps VTK manifold checking logic.
    Refactored from check_step_is_manifold.py
    """
    
    # Configuration matches your original script
    MESH_MIN_DIVISOR = 100.0
    SUBDIVISION_LEVEL = 1
    FEATURE_ANGLE = 60.0
    SKIP_CHECK_THRESHOLD = 10000

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def scan(self, step_path: str) -> dict:
        """
        Runs the manifold check pipeline.
        Returns: {'is_manifold': bool, 'details': str}
        """
        if not step_path or not isinstance(step_path, str) or not os.path.exists(step_path):
                return {'is_manifold': False, 'details': 'File not found or invalid path'}
        # 1. Silence VTK Logs
        try:
            vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)
        except:
            pass
            
        log_capture = vtk.vtkStringOutputWindow()
        vtk.vtkOutputWindow.SetInstance(log_capture)

        tmp_stl = None
        try:
            # 2. GMSH Conversion (STEP -> STL)
            # Use a fresh instance if possible, or careful state management
            if not gmsh.isInitialized():
                gmsh.initialize()
            
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("Inspector_Temp")
            gmsh.open(step_path)
            
            # Smart sizing
            bbox = gmsh.model.getBoundingBox(-1, -1)
            dims = [bbox[3]-bbox[0], bbox[4]-bbox[1], bbox[5]-bbox[2]]
            diag = (dims[0]**2 + dims[1]**2 + dims[2]**2)**0.5 or 1.0
            
            gmsh.option.setNumber("Mesh.MeshSizeMin", diag / self.MESH_MIN_DIVISOR)
            gmsh.option.setNumber("Mesh.MeshSizeMax", diag / 20.0)
            
            gmsh.model.mesh.generate(2)
            
            fd, tmp_stl = tempfile.mkstemp(suffix=".stl")
            os.close(fd)
            gmsh.write(tmp_stl)
            
            # CRITICAL FIX: Do volume/mass check BEFORE removing model
            # This was the bug - we were checking volumes on an empty model
            vols = gmsh.model.getEntities(3)
            total_vol = 0.0
            if vols:
                for dim, tag in vols:
                    try:
                        total_vol += gmsh.model.occ.getMass(dim, tag)
                    except:
                        pass
            
            # Detailed logging for diagnostics
            self.logger.info(f"[Guardian] Geometry scan: {len(vols)} volumes, total_mass={total_vol:.6f}")
            
            # Now we can safely remove the model
            gmsh.model.remove() 

            # 3. VTK Pipeline
            reader = vtk.vtkSTLReader()
            reader.SetFileName(tmp_stl)
            reader.Update()

            cleaner = vtk.vtkCleanPolyData()
            cleaner.SetInputData(reader.GetOutput())
            cleaner.Update()
            poly_data = cleaner.GetOutput()
            
            cell_count = poly_data.GetNumberOfCells()
            self.logger.info(f"[Guardian] VTK mesh: {cell_count} cells")
            
            # Conditional Subdivision (The "Stress Test")
            if cell_count < self.SKIP_CHECK_THRESHOLD:
                sub = vtk.vtkLinearSubdivisionFilter()
                sub.SetInputData(poly_data)
                sub.SetNumberOfSubdivisions(self.SUBDIVISION_LEVEL)
                sub.Update() # This triggers errors for bad geometry

            # Normal Generation (catches inverted faces)
            normals = vtk.vtkPolyDataNormals()
            normals.SetInputData(poly_data)
            normals.SetFeatureAngle(self.FEATURE_ANGLE)
            normals.Update()

            # 4. Log Analysis
            errors = log_capture.GetOutput()
            if "Subdivision failed" in errors or "non-manifold" in errors:
                return {'is_manifold': False, 'details': 'Non-manifold geometry detected'}
            
            if "Error" in errors or "ERR" in errors:
                return {'is_manifold': False, 'details': f'VTK Pipeline Error: {errors[:50]}...'}

            # 5. Volume Check (use pre-computed values from Gmsh)
            if not vols:
                return {'is_manifold': False, 'details': 'No 3D volumes detected'}
            
            if total_vol <= 1e-9:
                return {'is_manifold': False, 'details': 'Zero volume detected (Open Shell/Collapsed)'}

            return {'is_manifold': True, 'details': 'Passed strict topology check'}

        except Exception as e:
            return {'is_manifold': False, 'details': str(e)}

        finally:
            if tmp_stl and os.path.exists(tmp_stl):
                try: os.remove(tmp_stl)
                except: pass