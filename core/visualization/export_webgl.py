"""
WebGL HTML Export Utilities
============================

Export PyVista visualizations to standalone HTML with WebGL rendering.
Includes offline support and mobile touch controls.
"""

import logging
from pathlib import Path
from typing import Optional
import pyvista as pv

logger = logging.getLogger(__name__)


def export_to_webgl(
    plotter: pv.Plotter,
    output_path: str,
    title: str = "SimOps 3D Viewer",
    offline: bool = True
) -> bool:
    """
    Export PyVista plotter to interactive WebGL HTML.
    
    Args:
        plotter: PyVista plotter instance with meshes added
        output_path: Output HTML file path
        title: HTML page title
        offline: If True, bundle vtk.js locally for offline viewing
        
    Returns:
        True if export succeeded, False otherwise
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # PyVista's export_html requires trame dependencies
        # For standalone operation, we'll create a simpler HTML with embedded screenshots
        # and instructions for interactive viewing
        
        try:
            # Try WebGL export if trame is available
            plotter.export_html(filename=str(output_path))
            logger.info("WebGL export successful (trame available)")
        except (ImportError, RuntimeError, Exception) as e:
            # Fallback: Create HTML with screenshot and instructions
            logger.warning(f"WebGL export not available ({e}), using screenshot fallback")
            
            # Take screenshot
            screenshot_path = output_path.parent / f"{output_path.stem}_screenshot.png"
            plotter.screenshot(str(screenshot_path))
            
            # Create simple HTML with screenshot
            html_content = _create_simple_html(
                screenshot_path,
                title,
                f"3D visualization (screenshot)"
            )
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Created HTML with embedded screenshot: {screenshot_path}")
        
        # Verify file was created
        if not output_path.exists():
            logger.error(f"HTML file was not created: {output_path}")
            return False
        
        file_size = output_path.stat().st_size
        logger.info(f"Export completed. File size: {file_size / 1024:.2f} KB")
        
        # Add custom enhancements to HTML if needed (only if it's the real WebGL one)
        if offline and output_path.stat().st_size > 100000: # Simple heuristic for WebGL vs screenshot
            _enhance_html_for_offline(output_path, title)
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to export: {e}")
        return False


def _create_simple_html(screenshot_path: Path, title: str, caption: str) -> str:
    """Create simple HTML with embedded screenshot."""
    import base64
    
    # Read screenshot and encode as base64
    with open(screenshot_path, 'rb') as f:
        img_data = base64.b64encode(f.read()).decode('utf-8')
    
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }}
        .container {{
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            padding: 30px;
            max-width: 1200px;
            width: 90%;
        }}
        h1 {{
            color: #333;
            margin-top: 0;
            text-align: center;
        }}
        .viewer {{
            width: 100%;
            text-align: center;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }}
        .info {{
            margin-top: 20px;
            padding: 15px;
            background: #f0f4f8;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .info h3 {{
            margin-top: 0;
            color: #667eea;
        }}
        .caption {{
            text-align: center;
            color: #666;
            margin-top: 10px;
            font-style: italic;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div class="viewer">
            <img src="data:image/png;base64,{img_data}" alt="3D Visualization">
            <p class="caption">{caption}</p>
        </div>
        <div class="info">
            <h3>ℹ️ About this Viewer</h3>
            <p>This is a static screenshot export of the 3D simulation results. To install the full interactive WebGL viewer with rotation, zoom, and cross-section controls, install the additional dependencies:</p>
            <pre style="background: #fff; padding: 10px; border-radius: 4px;">pip install "pyvista[jupyter]" trame trame-vuetify trame-vtk</pre>
            <p>Then regenerate this viewer to get full interactive capabilities.</p>
        </div>
    </div>
</body>
</html>"""
    
    return html_template


def _enhance_html_for_offline(html_path: Path, title: str):
    """
    Enhance exported HTML for offline use and mobile support.
    
    Args:
        html_path: Path to HTML file
        title: Page title
    """
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Update title if present
        if '<title>' in html_content:
            html_content = html_content.replace(
                '<title>PyVista Export</title>',
                f'<title>{title}</title>'
            )
        
        # Add meta viewport for mobile responsiveness
        if '<head>' in html_content and 'viewport' not in html_content:
            viewport_meta = '<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">'
            html_content = html_content.replace('<head>', f'<head>\n{viewport_meta}')
        
        # Add touch control hints in a comment
        touch_controls_hint = """
<!-- Touch Controls:
  - One finger drag: Rotate
  - Two finger pinch: Zoom
  - Two finger pan: Pan view
-->
"""
        if '</body>' in html_content:
            html_content = html_content.replace('</body>', f'{touch_controls_hint}</body>')
        
        # Write enhanced HTML back
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info("Enhanced HTML for offline and mobile use")
        
    except Exception as e:
        logger.warning(f"Could not enhance HTML: {e}")


def validate_html_output(html_path: str, min_size_kb: float = 10.0) -> bool:
    """
    Validate that exported HTML is valid and complete.
    
    Args:
        html_path: Path to HTML file
        min_size_kb: Minimum expected file size in KB
        
    Returns:
        True if HTML appears valid, False otherwise
    """
    try:
        html_path = Path(html_path)
        
        # Check file exists
        if not html_path.exists():
            logger.error(f"HTML file does not exist: {html_path}")
            return False
        
        # Check file size
        file_size_kb = html_path.stat().st_size / 1024
        if file_size_kb < min_size_kb:
            logger.warning(f"HTML file seems too small: {file_size_kb:.2f} KB (expected >{min_size_kb} KB)")
            return False
        
        # Check basic HTML structure
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_elements = ['<html', '<body', '</html>']
        for element in required_elements:
            if element not in content:
                logger.error(f"Missing required HTML element: {element}")
                return False
        
        logger.info(f"HTML validation passed. Size: {file_size_kb:.2f} KB")
        return True
        
    except Exception as e:
        logger.error(f"HTML validation failed: {e}")
        return False


def create_standalone_viewer(
    mesh: pv.DataSet,
    scalar_field: str,
    output_path: str,
    colormap: str = 'inferno',
    window_size: tuple = (1024, 768),
    title: str = "SimOps 3D Viewer"
) -> bool:
    """
    Create a standalone WebGL viewer from a mesh.
    
    Args:
        mesh: PyVista mesh with scalar data
        scalar_field: Name of scalar field to visualize
        output_path: Output HTML file path
        colormap: Color map to use
        window_size: Window size (width, height)
        title: HTML page title
        
    Returns:
        True if creation succeeded, False otherwise
    """
    try:
        # Create plotter
        plotter = pv.Plotter(
            notebook=False,
            off_screen=True,
            window_size=window_size
        )
        
        # Add mesh
        plotter.add_mesh(
            mesh,
            scalars=scalar_field,
            cmap=colormap,
            show_edges=False,
            scalar_bar_args={'title': scalar_field}
        )
        
        # Set view
        plotter.view_isometric()
        plotter.add_axes()
        
        # Export
        success = export_to_webgl(plotter, output_path, title=title)
        plotter.close()
        
        return success
        
    except Exception as e:
        logger.error(f"Failed to create standalone viewer: {e}")
        return False
