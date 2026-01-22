import base64
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def generate_html_report(job_id, results, output_dir):
    """
    Generates a standalone HTML report with embedded Base64 images.
    Uses the same data extraction logic as ThermalPDFReportGenerator.
    """
    output_path = Path(output_dir) / "report.html"
    output_dir = Path(output_dir)
    
    # helper to embed image
    def get_image_base64(img_path):
        if not img_path:
            return ""
        img_path = Path(img_path)
        if not img_path.exists():
            # Try relative to output_dir if absolute path doesn't exist
            if not img_path.is_absolute():
                img_path = output_dir / img_path
            if not img_path.exists():
                logger.warning(f"Image not found: {img_path}")
                return ""
        try:
            with open(img_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode('utf-8')
                ext = img_path.suffix[1:] if img_path.suffix else 'png'
                return f"data:image/{ext};base64,{encoded}"
        except Exception as e:
            logger.error(f"Failed to encode image {img_path}: {e}")
            return ""

    # Extract data using the same logic as ThermalPDFReportGenerator
    # The pipeline returns metadata with keys like max_temperature_K, min_temperature_K, solve_time_s
    # Try multiple possible keys to be robust
    max_temp_k = results.get('max_temperature_K') or results.get('max_temp')
    min_temp_k = results.get('min_temperature_K') or results.get('min_temp')
    
    # If we got Celsius values, convert to Kelvin
    max_temp_c = results.get('max_temperature_C')
    min_temp_c = results.get('min_temperature_C')
    
    # Convert from Celsius if we have C values but not K values
    if max_temp_c is not None and max_temp_k is None:
        max_temp_k = max_temp_c + 273.15
    if min_temp_c is not None and min_temp_k is None:
        min_temp_k = min_temp_c + 273.15
    
    # If we have a value that's clearly in Celsius (< 100), convert it
    if max_temp_k is not None and max_temp_k < 100:
        max_temp_k = max_temp_k + 273.15
    if min_temp_k is not None and min_temp_k < 100:
        min_temp_k = min_temp_k + 273.15
    
    # Use actual values or default to reasonable values (not 0.0 which looks like an error)
    max_temp = max_temp_k if max_temp_k is not None and max_temp_k > 0 else (results.get('ambient_temperature', 293.15) + 10)
    min_temp = min_temp_k if min_temp_k is not None and min_temp_k > 0 else results.get('ambient_temperature', 293.15)
    
    # Log what we found for debugging
    logger.info(f"HTML Report - Extracted temperatures: max={max_temp:.2f}K, min={min_temp:.2f}K")
    logger.info(f"HTML Report - Results keys: {list(results.keys())}")
    
    solve_time = results.get('solve_time_s') or results.get('solve_time') or results.get('total_time_s') or 0.0
    converged = results.get('converged', True)
    success = results.get('success', converged)
    
    # Extract additional metrics for display
    num_elements = results.get('num_elements') or results.get('n_elements') or results.get('mesh_elements') or 0
    num_nodes = results.get('num_nodes', 0)
    iterations_run = results.get('iterations_run') or results.get('iterations')
    material = results.get('material', 'Unknown')
    ambient_temp = results.get('ambient_temperature', 293.15)
    heat_power = results.get('heat_source_power', 0)
    solver_name = results.get('solver', 'builtin')
    
    # Get image paths - try multiple possible locations
    png_file = results.get('png_file')
    if png_file:
        # If it's a relative path, make it relative to output_dir
        png_path = Path(png_file)
        if not png_path.is_absolute():
            png_file = output_dir / png_path
        else:
            png_file = png_path
    else:
        # Try multiple fallback locations
        possible_names = ["temperature_map.png", "thermal_result.png", "temperature_distribution.png"]
        png_file = None
        for name in possible_names:
            candidate = output_dir / name
            if candidate.exists():
                png_file = candidate
                logger.info(f"HTML Report - Found PNG at: {png_file}")
                break
        
        if not png_file:
            # Last resort: look for any PNG in output_dir
            png_files = list(output_dir.glob("*.png"))
            if png_files:
                png_file = png_files[0]
                logger.info(f"HTML Report - Using first PNG found: {png_file}")
            else:
                logger.warning(f"HTML Report - No PNG file found in {output_dir}")
                png_file = output_dir / "temperature_map.png"  # Placeholder path
    
    # Generate multi-angle visualizations if VTK file is available
    vtk_file = results.get('vtk_file')
    visualization_images = []
    
    # First, try to use the PNG file that was already generated by the pipeline
    if png_file and png_file.exists():
        img_base64 = get_image_base64(png_file)
        if img_base64:
            visualization_images.append(('Temperature Distribution', img_base64))
            logger.info(f"HTML Report - Added PNG visualization: {png_file}")
    else:
        logger.warning(f"HTML Report - PNG file not found: {png_file}")
    
    # Try to generate multi-angle views from VTK if available
    if vtk_file:
        vtk_path = Path(vtk_file) if Path(vtk_file).is_absolute() else output_dir / vtk_file
        if vtk_path.exists():
            logger.info(f"HTML Report - Attempting to generate views from VTK: {vtk_path}")
            job_name = Path(results.get('input_file', job_id)).stem if results.get('input_file') else job_id
            
            # Try PyVista-based multi-angle visualization first (better quality)
            try:
                from core.reporting.thermal_multi_angle_viz import generate_thermal_views
                view_images = generate_thermal_views(
                    vtu_path=str(vtk_path),
                    output_dir=output_dir,
                    job_name=job_name,
                    views=['isometric', 'top', 'front'],
                    colormap='coolwarm'
                )
                for view_path in view_images:
                    if Path(view_path).exists():
                        view_name = Path(view_path).stem.replace(f"{job_name}_thermal_", "").replace("_", " ").title()
                        img_b64 = get_image_base64(view_path)
                        if img_b64:
                            visualization_images.append((view_name, img_b64))
                            logger.info(f"HTML Report - Added multi-angle view: {view_name}")
            except ImportError as e:
                logger.info(f"PyVista visualization not available: {e}")
            except Exception as e:
                logger.warning(f"Could not generate multi-angle views with PyVista: {e}")
                import traceback
                logger.debug(traceback.format_exc())
            
            # Fallback to thermal visualizer if PyVista failed
            if len(visualization_images) <= 1:  # Only have the PNG, need more views
                try:
                    from core.visualization.thermal_viz import ThermalVisualizer
                    viz = ThermalVisualizer(output_dir)
                    viz_images = viz.generate_snapshots(
                        mesh_file=str(vtk_path),
                        job_name=job_name,
                        angles=['iso', 'top', 'side']
                    )
                    for viz_path in viz_images:
                        if Path(viz_path).exists():
                            view_name = Path(viz_path).stem.replace(f"{job_name}_", "").title()
                            img_b64 = get_image_base64(viz_path)
                            if img_b64:
                                visualization_images.append((view_name, img_b64))
                                logger.info(f"HTML Report - Added fallback view: {view_name}")
                except Exception as e2:
                    logger.warning(f"Fallback visualization also failed: {e2}")
                    import traceback
                    logger.debug(traceback.format_exc())
        else:
            logger.warning(f"HTML Report - VTK file not found: {vtk_path}")
    
    # If no visualizations were generated, show placeholder
    if not visualization_images:
        logger.warning("HTML Report - No visualization images found for HTML report")
        visualization_images = [('No visualization available', '')]
    else:
        logger.info(f"HTML Report - Generated {len(visualization_images)} visualization(s)")

    def generate_visualization_html(images):
        """Generate HTML for visualization images"""
        if not images or (len(images) == 1 and not images[0][1]):
            return '<p style="color: #999; text-align: center;">No visualization data available</p>'
        
        html_parts = []
        for title, img_base64 in images:
            if img_base64:
                html_parts.append(f'''
                    <div style="margin-bottom: 30px;">
                        <h3 style="font-size: 1.1em; margin-bottom: 10px; color: #555;">{title}</h3>
                        <img src="{img_base64}" alt="{title}" style="max-width: 100%; height: auto; border-radius: 8px; border: 1px solid #ddd;">
                    </div>
                ''')
        return '\n'.join(html_parts)

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SimOps Thermal Report - {job_id}</title>
    <style>
        body {{
            font-family: 'Inter', -apple-system, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 40px 20px;
            background-color: #f9f9f9;
        }}
        .report-card {{
            background: white;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }}
        h1 {{
            margin-top: 0;
            color: #1a1a1a;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }}
        .meta {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 30px;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .metric-item {{
            background: #f0f4f8;
            padding: 15px;
            border-radius: 8px;
        }}
        .metric-label {{
            font-size: 0.8em;
            text-transform: uppercase;
            color: #555;
            font-weight: bold;
        }}
        .metric-value {{
            font-size: 1.4em;
            font-weight: 700;
            color: #0056b3;
        }}
        .visualization {{
            margin-top: 40px;
            text-align: center;
        }}
        .visualization img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            border: 1px solid #ddd;
        }}
        .footer {{
            margin-top: 60px;
            font-size: 0.8em;
            color: #999;
            text-align: center;
        }}
        @media print {{
            body {{ background: white; padding: 0; }}
            .report-card {{ box-shadow: none; padding: 0; }}
        }}
    </style>
</head>
<body>
    <div class="report-card">
        <h1>SimOps Thermal Analysis Report</h1>
        <div class="meta">
            <div><strong>Job ID:</strong> {job_id}</div>
            <div><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
            <div><strong>Material:</strong> {material}</div>
            <div><strong>Solver:</strong> {solver_name}</div>
            {f'<div><strong>Heat Source Power:</strong> {heat_power:.1f} W</div>' if heat_power > 0 else ''}
            {f'<div><strong>Iterations:</strong> {iterations_run}</div>' if iterations_run else ''}
        </div>

        <div class="metrics">
            <div class="metric-item">
                <div class="metric-label">Max Temperature</div>
                <div class="metric-value">{max_temp:.2f} K ({max_temp - 273.15:.2f} °C)</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Min Temperature</div>
                <div class="metric-value">{min_temp:.2f} K ({min_temp - 273.15:.2f} °C)</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Temperature Range</div>
                <div class="metric-value">{max_temp - min_temp:.2f} K</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Solve Time</div>
                <div class="metric-value">{solve_time:.2f} s</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Elements</div>
                <div class="metric-value">{num_elements:,}</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Status</div>
                <div class="metric-value" style="color: {'#28a745' if success else '#dc3545'}">
                    {'CONVERGED' if success else 'NOT CONVERGED'}
                </div>
            </div>
        </div>

        <div class="visualization">
            <h2>Temperature Distribution MAP</h2>
            {generate_visualization_html(visualization_images)}
        </div>

        <div class="footer">
            Generated by SimOps Engineer Engine. Standalone Report.
        </div>
    </div>
</body>
</html>
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    return str(output_path)
