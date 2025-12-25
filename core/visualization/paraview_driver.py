"""
ParaView Integration Driver
===========================

Provides high-quality 3D streamline visualization using ParaView's pvbatch.
Falls back gracefully when ParaView is not available.
"""

import os
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class ParaViewDriver:
    """
    Driver for ParaView's pvbatch command for high-quality CFD visualization.
    
    Tier 1 in the visualization pipeline - produces publication-quality
    3D streamline renders with proper lighting and color mapping.
    """
    
    def __init__(self):
        self.pvbatch_path = self._find_pvbatch()
    
    def _find_pvbatch(self) -> Optional[str]:
        """Locate pvbatch executable on the system."""
        # Common installation paths
        search_paths = [
            # Windows paths
            r"C:\Program Files\ParaView 5.12.0\bin\pvbatch.exe",
            r"C:\Program Files\ParaView 5.11.0\bin\pvbatch.exe",
            r"C:\Program Files\ParaView 5.10.0\bin\pvbatch.exe",
            r"C:\Program Files\ParaView\bin\pvbatch.exe",
            # Linux paths (including Docker)
            "/usr/bin/pvbatch",
            "/opt/paraview/bin/pvbatch",
            "/usr/local/bin/pvbatch",
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                logger.info(f"Found ParaView at: {path}")
                return path
        
        # Try PATH
        try:
            result = subprocess.run(
                ["which", "pvbatch"] if os.name != 'nt' else ["where", "pvbatch"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                path = result.stdout.strip().split('\n')[0]
                logger.info(f"Found ParaView in PATH: {path}")
                return path
        except Exception:
            pass
        
        logger.warning("ParaView (pvbatch) not found on system")
        return None
    
    def is_available(self) -> bool:
        """Check if ParaView is available."""
        return self.pvbatch_path is not None
    
    def generate_streamlines(
        self, 
        vtk_file: str, 
        output_png: str, 
        seed_center: List[float], 
        seed_radius: float
    ) -> bool:
        """
        Generate streamline visualization using ParaView.
        
        Args:
            vtk_file: Path to VTK/VTU file with velocity field
            output_png: Output PNG path
            seed_center: [x, y, z] center for streamline seeds
            seed_radius: Radius for point cloud seed distribution
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            logger.warning("ParaView not available, skipping 3D render.")
            return False
        
        # 1. Generate Python Script
        script_content = self._create_streamlines_script(
            vtk_file, output_png, seed_center, seed_radius
        )
        
        # 2. Write to temp file
        script_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.py', delete=False, encoding='utf-8'
            ) as tf:
                tf.write(script_content)
                script_path = tf.name
            
            # 3. Run pvbatch
            cmd = [self.pvbatch_path, script_path]
            logger.info(f"Running ParaView: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=120
            )
            
            if result.returncode != 0:
                logger.error(f"ParaView execution failed: {result.stderr}")
                logger.error(f"ParaView stdout: {result.stdout}")
                return False
            
            if not Path(output_png).exists():
                logger.error("ParaView finished but output PNG not found.")
                return False
            
            logger.info(f"ParaView successfully generated: {output_png}")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("ParaView timed out after 120 seconds")
            return False
        except Exception as e:
            logger.error(f"ParaView error: {e}")
            return False
        finally:
            # Cleanup script
            if script_path and os.path.exists(script_path):
                try:
                    os.unlink(script_path)
                except Exception:
                    pass
    
    def _create_streamlines_script(
        self, 
        vtk_file: str, 
        output_png: str,
        center: List[float], 
        radius: float
    ) -> str:
        """Generate ParaView Python script for streamline visualization."""
        # Convert Windows paths to forward slashes for ParaView
        safe_vtk = str(Path(vtk_file).absolute()).replace('\\', '/')
        safe_out = str(Path(output_png).absolute()).replace('\\', '/')
        
        return f'''# Auto-generated ParaView Script for Streamline Visualization
# Generated by SimOps ParaViewDriver

from paraview.simple import *

# Disable automatic GUI interactions in batch mode
paraview.simple._DisableFirstRenderCameraReset()

try:
    # 1. Load Data
    print("Loading VTK file: {safe_vtk}")
    reader = OpenDataFile('{safe_vtk}')
    if reader is None:
        raise RuntimeError("Failed to open VTK file")
    UpdatePipeline()
    
    # Check for velocity field
    data_info = reader.GetDataInformation()
    point_data = data_info.GetPointDataInformation()
    
    has_velocity = False
    for i in range(point_data.GetNumberOfArrays()):
        if point_data.GetArrayInformation(i).GetName() == 'U':
            has_velocity = True
            break
    
    if not has_velocity:
        # Try cell data
        cell_data = data_info.GetCellDataInformation()
        for i in range(cell_data.GetNumberOfArrays()):
            if cell_data.GetArrayInformation(i).GetName() == 'U':
                # Need to convert to point data
                print("Converting cell data to point data...")
                cellToPoint = CellDatatoPointData(Input=reader)
                reader = cellToPoint
                UpdatePipeline()
                has_velocity = True
                break
    
    if not has_velocity:
        print("WARNING: No velocity field 'U' found in data")
    
    # 2. Stream Tracer
    print("Creating streamlines...")
    streamTracer = StreamTracer(Input=reader)
    streamTracer.Vectors = ['POINTS', 'U']
    streamTracer.MaximumStreamlineLength = 100.0
    streamTracer.IntegrationDirection = 'BOTH'
    
    # Setup Point Source for seeds
    streamTracer.SeedType = 'Point Cloud'
    streamTracer.SeedType.Center = {center}
    streamTracer.SeedType.Radius = {radius}
    streamTracer.SeedType.NumberOfPoints = 100
    
    UpdatePipeline()
    
    # 3. Tube Filter (make lines visible)
    print("Adding tube filter...")
    tube = Tube(Input=streamTracer)
    tube.Radius = {radius * 0.01}  # 1% of seed radius
    
    UpdatePipeline()
    
    # 4. Rendering Setup
    print("Setting up render view...")
    renderView = CreateView('RenderView')
    renderView.ViewSize = [1920, 1080]
    renderView.Background = [1.0, 1.0, 1.0]  # White background
    renderView.OrientationAxesVisibility = 1
    
    # Display Tube with velocity coloring
    tubeDisplay = Show(tube, renderView)
    tubeDisplay.Representation = 'Surface'
    
    # Color by velocity magnitude
    ColorBy(tubeDisplay, ('POINTS', 'U', 'Magnitude'))
    tubeDisplay.SetScalarBarVisibility(renderView, True)
    
    # Apply Turbo colormap
    uLUT = GetColorTransferFunction('U')
    uLUT.ApplyPreset('Turbo', True)
    
    # Camera positioning
    ResetCamera()
    renderView.ResetCamera()
    
    # Isometric-ish view
    camera = renderView.GetActiveCamera()
    camera.Elevation(30)
    camera.Azimuth(45)
    renderView.ResetCamera()
    
    # 5. Save Screenshot
    print("Saving screenshot to: {safe_out}")
    SaveScreenshot('{safe_out}', renderView, ImageResolution=[1920, 1080])
    
    print("ParaView script completed successfully")
    
except Exception as e:
    print(f"ParaView script error: {{e}}")
    import traceback
    traceback.print_exc()
    raise
'''
    
    def generate_multi_view(
        self,
        vtk_file: str,
        output_dir: str,
        job_name: str,
        seed_center: List[float],
        seed_radius: float
    ) -> List[str]:
        """
        Generate 4-view streamline visualization (iso, front, side, top).
        
        Returns list of generated PNG paths.
        """
        if not self.is_available():
            return []
        
        views = [
            ('iso', 45, 30),
            ('front', 0, 0),
            ('side', 90, 0),
            ('top', 0, 90),
        ]
        
        generated = []
        for view_name, azimuth, elevation in views:
            output_png = str(Path(output_dir) / f"{job_name}_streamlines_{view_name}.png")
            
            if self.generate_streamlines(vtk_file, output_png, seed_center, seed_radius):
                generated.append(output_png)
            else:
                logger.warning(f"Failed to generate {view_name} view")
        
        return generated
