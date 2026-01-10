import os
import tempfile
import logging
import gmsh

# Lazy imports to prevent worker startup lag
# VTK will be imported inside methods only when needed


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

        tmp_stl = None
        try:
            # 1. GMSH Conversion (STEP -> STL) - Common for both Trimesh and VTK
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
            
            # Volume/Mass check using Gmsh
            vols = gmsh.model.getEntities(3)
            total_vol = 0.0
            if vols:
                for dim, tag in vols:
                    try:
                        total_vol += gmsh.model.occ.getMass(dim, tag)
                    except:
                        pass
            
            self.logger.info(f"[Guardian] Geometry scan: {len(vols)} volumes, total_mass={total_vol:.6f}")
            gmsh.model.remove() 

            # Volume Check
            if not vols:
                return {'is_manifold': False, 'details': 'No 3D volumes detected'}
            if total_vol <= 1e-9:
                return {'is_manifold': False, 'details': 'Zero volume detected (Open Shell/Collapsed)'}

            # 2. Optimized Inspection with Trimesh (Fast, no VTK scan)
            try:
                import trimesh
                mesh = trimesh.load(tmp_stl, force='mesh', process=False)
                
                if not mesh.is_watertight:
                    return {'is_manifold': False, 'details': 'Non-watertight geometry (holes/gaps detected)'}
                
                # Basic winding check (optional, but good)
                try: 
                    if not mesh.is_winding_consistent:
                        # Warning only, as it might be fixable
                        self.logger.warning("[Guardian] Inconsistent winding detected")
                except: pass
                
                return {'is_manifold': True, 'details': 'Passed strict topology check (Trimesh)'}
                
            except ImportError:
                # Fallback to VTK if Trimesh is missing
                return self._scan_with_vtk(tmp_stl, total_vol)

        except Exception as e:
            return {'is_manifold': False, 'details': str(e)}

        finally:
            if tmp_stl and os.path.exists(tmp_stl):
                try: os.remove(tmp_stl)
                except: pass

    def _scan_with_vtk(self, stl_path: str, total_vol: float) -> dict:
        """Legacy VTK-based scanning (slower startup)"""
        try:
            import vtk
            vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)
            
            log_capture = vtk.vtkStringOutputWindow()
            vtk.vtkOutputWindow.SetInstance(log_capture)
            
            reader = vtk.vtkSTLReader()
            reader.SetFileName(stl_path)
            reader.Update()

            cleaner = vtk.vtkCleanPolyData()
            cleaner.SetInputData(reader.GetOutput())
            cleaner.Update()
            poly_data = cleaner.GetOutput()
            
            cell_count = poly_data.GetNumberOfCells()
            
            # Conditional Subdivision
            if cell_count < self.SKIP_CHECK_THRESHOLD:
                sub = vtk.vtkLinearSubdivisionFilter()
                sub.SetInputData(poly_data)
                sub.SetNumberOfSubdivisions(self.SUBDIVISION_LEVEL)
                sub.Update()

            errors = log_capture.GetOutput()
            if "Subdivision failed" in errors or "non-manifold" in errors:
                return {'is_manifold': False, 'details': 'Non-manifold geometry detected (VTK)'}
            
            return {'is_manifold': True, 'details': 'Passed strict topology check (VTK)'}
            
        except ImportError:
            return {'is_manifold': False, 'details': 'Inspection failed: Neither Trimesh nor VTK available'}
        except Exception as e:
            return {'is_manifold': False, 'details': f'VTK error: {e}'}