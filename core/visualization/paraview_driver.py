"""
ParaView Driver
===============

Manages interaction with ParaView's pvbatch executable to generate
high-quality 3D visualizations.
"""

import os
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import Optional, Dict, List
from core.config import get_default_config

logger = logging.getLogger(__name__)

class ParaViewDriver:
    """
    Drives pvbatch to generate visualizations.
    """
    
    def __init__(self):
        self.config = get_default_config()
        self.pvbatch_path = self.config.paths.get_pvbatch()
        
    def is_available(self) -> bool:
        """Check if ParaView is available."""
        return self.pvbatch_path is not None
        
    def generate_streamlines(self, vtk_file: str, output_png: str, 
                           seed_center: List[float], seed_radius: float) -> bool:
        """
        Generate a streamlines visualization.
        
        Args:
            vtk_file: Path to input VTK/VTU file
            output_png: Path where png should be saved
            seed_center: [x,y,z] center for streamline seed source
            seed_radius: Radius for streamline seed source
            
        Returns:
            True if successful
        """
        if not self.is_available():
            logger.warning("ParaView not available, skipping 3D render.")
            return False
            
        # 1. Generate Python Script
        script_content = self._create_streamlines_script(
            vtk_file, output_png, seed_center, seed_radius
        )
        
        # 2. Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tf:
            tf.write(script_content)
            script_path = tf.name
            
        try:
            # 3. Run pvbatch
            # We must quote the executable path as it may contain spaces
            cmd = [self.pvbatch_path, script_path]
            
            logger.info(f"Running ParaView: {cmd}")
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120
            )
            
            if result.returncode != 0:
                logger.error(f"ParaView execution failed: {result.stderr}")
                return False
                
            if not Path(output_png).exists():
                logger.error("ParaView finished but output PNG not found.")
                return False
                
            return True
            
        except Exception as e:
            logger.exception(f"Error running ParaView: {e}")
            return False
            
        finally:
            # Cleanup script
            if os.path.exists(script_path):
                os.unlink(script_path)
                
    def _create_streamlines_script(self, vtk_file: str, output_png: str,
                                 center: List[float], radius: float) -> str:
        """Creates the Python script for pvbatch."""
        # Note: We escape backslashes for Windows paths in the generated script
        safe_vtk = str(Path(vtk_file).absolute()).replace('\\', '/')
        safe_out = str(Path(output_png).absolute()).replace('\\', '/')
        
        return f"""
# Auto-generated ParaView Script
from paraview.simple import *

# 1. Load Data
reader = OpenDataFile('{safe_vtk}')
UpdatePipeline()

# 2. Stream Tracer
streamTracer = StreamTracer(Input=reader)
streamTracer.Vectors = ['POINTS', 'U']
streamTracer.MaximumStreamlineLength = 100.0

# Setup Point Source for seeds
streamTracer.SeedType = 'Point Cloud'
streamTracer.SeedType.Center = {center}
streamTracer.SeedType.Radius = {radius}
streamTracer.SeedType.NumberOfPoints = 100

# 3. Tube Filter (make lines thick)
tube = Tube(Input=streamTracer)
tube.Radius = 0.005 # configurable?

# 4. Rendering
renderView = CreateView('RenderView')
renderView.ViewSize = [1920, 1080]
renderView.Background = [1, 1, 1] # White background
renderView.OrientationAxesVisibility = 1

# Display Tube
tubeDisplay = Show(tube, renderView)
tubeDisplay.ColorArrayName = ['POINTS', 'U']
tubeDisplay.SetScalarBarVisibility(renderView, True)

# Color Map
uLUT = GetColorTransferFunction('U')
uLUT.ApplyPreset('Turbo', True)

# Camera (Auto-fit)
ResetCamera()
# Optional: Adjust camera angle if needed?
# camera = GetActiveCamera()
# camera.Elevation(30)

# 5. Save
SaveScreenshot('{safe_out}', renderView)
"""
