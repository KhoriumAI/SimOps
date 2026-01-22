import os
import sys
import shutil
import uuid
import subprocess
from pathlib import Path
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Handle PyInstaller _MEIPASS for bundled resources
if hasattr(sys, '_MEIPASS'):
    MEI_ROOT = Path(sys._MEIPASS)
    # 1. Update sys.path to include bundled root
    sys.path.insert(0, str(MEI_ROOT))
else:
    MEI_ROOT = Path(__file__).parent.parent
    sys.path.insert(0, str(MEI_ROOT))

# 1. Try parent directory (Dev mode: simops-backend/.. -> root)
# sys.path.insert(0, str(Path(__file__).parent.parent))

# 2. Try current directory (Bundled mode: everything in resources root)
sys.path.insert(0, str(Path(__file__).parent))

# Import SimOps Pipeline
try:
    from simops_pipeline import run_simops_pipeline, SimOpsConfig
except ImportError as e:
    import traceback
    print("Warning: Could not import simops_pipeline.py.")
    print(f"Current Directory: {os.getcwd()}")
    print(f"Sys Path: {sys.path}")
    if hasattr(sys, '_MEIPASS'):
        print(f"MEIPASS Content: {os.listdir(sys._MEIPASS)}")
        # Check if simops_pipeline exists
        print(f"simops_pipeline exists: {os.path.exists(os.path.join(sys._MEIPASS, 'simops_pipeline.py'))}")
    
    traceback.print_exc()
    run_simops_pipeline = None

# Import HTML Reporter
try:
    from core.reporting.html_reporter import generate_html_report
except ImportError:
    print("Warning: Could not import generate_html_report")
    generate_html_report = None

# Import AI Generator (Task 01)
try:
    # Add root to sys.path to find _forge if not already there
    ROOT_DIR = Path(__file__).parent.parent
    if str(ROOT_DIR) not in sys.path:
        sys.path.append(str(ROOT_DIR))
        
    from _forge.task_01_ai_schema.ai_service import AIGenerator as AISetupGenerator
except ImportError as e:
    print(f"Warning: Could not import AIGenerator: {e}")
    AISetupGenerator = None

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}) # Multi-origin permissive for triage


@app.before_request
def log_request_info():
    pass # Disable verbose logging for production, or keep minimal

# Use user home directory for storage to ensure write permissions
# (Program Files is read-only)
BASE_DIR = Path.home() / '.simops'
UPLOAD_FOLDER = BASE_DIR / 'uploads'
OUTPUT_FOLDER = BASE_DIR / 'simops_output'

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "SimOps Backend"})

@app.route('/api/diagnostics', methods=['GET'])
def diagnostics():
    """Diagnostic endpoint to check bundled dependencies"""
    diag = {
        "gmsh_available": False,
        "gmsh_version": None,
        "openfoam_available": False,
        "openfoam_check": None,
        "upload_folder": str(UPLOAD_FOLDER),
        "upload_folder_exists": UPLOAD_FOLDER.exists(),
        "upload_folder_writable": os.access(str(UPLOAD_FOLDER), os.W_OK),
        "meipass": getattr(sys, '_MEIPASS', None),
    }

    # Check GMSH
    try:
        import gmsh
        diag["gmsh_available"] = True
        # Try to get version
        try:
            gmsh.initialize()
            diag["gmsh_version"] = gmsh.option.getString("General.Version")
            gmsh.finalize()
        except Exception as e:
            diag["gmsh_init_error"] = str(e)
    except ImportError as e:
        diag["gmsh_import_error"] = str(e)

    # Check OpenFOAM availability
    try:
        from tools.thermal_job_runner import OpenFOAMRunner
        runner = OpenFOAMRunner(dry_run=False)
        diag["openfoam_available"] = runner.openfoam_available
        if runner.openfoam_available:
            diag["openfoam_check"] = "OpenFOAM found (WSL or native)"
        else:
            diag["openfoam_check"] = "OpenFOAM not found - use 'builtin' solver instead"
    except Exception as e:
        diag["openfoam_check"] = f"OpenFOAM check failed: {str(e)}"

    return jsonify(diag)

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'files' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['files']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    unique_id = str(uuid.uuid4())[:8]
    save_name = f"{unique_id}_{filename}"
    save_path = UPLOAD_FOLDER / save_name
    
    file.save(save_path)

    # Convert CAD to STL for frontend preview (VTKLoader doesn't support UNSTRUCTURED_GRID)
    preview_url = None
    if filename.lower().endswith('.msh'):
        stl_name = f"{unique_id}_{filename.rsplit('.', 1)[0]}.stl"
        stl_path = UPLOAD_FOLDER / stl_name
        try:
            import gmsh
            # Use a separate model for the preview conversion
            if not gmsh.isInitialized():
                gmsh.initialize()
            
            gmsh.open(str(save_path))
            gmsh.write(str(stl_path))
            # No clear() in gmsh, but we can finalize or just leave it
            # Since we are threaded=False, we should probably finalize or just use a context
            gmsh.finalize() # Full reset for safety
            
            preview_url = f"/api/uploads/{stl_name}"
            print(f"Generated preview STL: {preview_url}")
        except Exception as e:
            print(f"Failed to generate STL preview: {e}")
            if gmsh.isInitialized():
                gmsh.finalize()
    
    response_data = {
        'message': 'File uploaded successfully',
        'filename': filename,
        'saved_as': save_name,
        'path': str(save_path.absolute()),
        'url': f"/api/uploads/{save_name}",
        'preview_url': preview_url
    }

    
    return jsonify(response_data)

@app.route('/api/uploads/<path:filename>')
def serve_upload(filename):
    """Serve uploaded files (for frontend preview)"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/results/<path:filename>')
def serve_results(filename):
    print(f"DEBUG: serve_results called with {filename}")
    full_path = OUTPUT_FOLDER / filename
    print(f"DEBUG: Looking for {full_path} (exists: {full_path.exists()})")
    
    try:
        response = send_from_directory(str(OUTPUT_FOLDER), filename)
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    except Exception as e:
        print(f"DEBUG: send_from_directory failed: {e}")
        return jsonify({'error': str(e)}), 404

@app.route('/api/simulate', methods=['POST'])
def trigger_simulation():
    if not run_simops_pipeline:
        return jsonify({'error': 'SimOps pipeline not available'}), 500

    data = request.json
    filename = data.get('filename') # Saved filename from upload
    config_dict = data.get('config')
    
    if not filename:
         return jsonify({'error': 'Filename is required'}), 400

    # Map Config Dict to SimOpsConfig
    # We assume config_dict matches schema or we pass it via kwargs if possible
    # For now, let's just use defaults + whatever we can map
    
    # Construct config object
    pipeline_config = SimOpsConfig()
    
    if config_dict:
        # Map known fields manually
        if 'heat_source_power' in config_dict: pipeline_config.heat_source_power = float(config_dict['heat_source_power'])
        if 'ambient_temperature' in config_dict: pipeline_config.ambient_temperature = float(config_dict['ambient_temperature'])
        if 'initial_temperature' in config_dict: pipeline_config.initial_temperature = float(config_dict['initial_temperature'])
        if 'convection_coefficient' in config_dict: pipeline_config.convection_coefficient = float(config_dict['convection_coefficient'])
        if 'material' in config_dict: pipeline_config.material = str(config_dict['material'])
        if 'simulation_type' in config_dict: pipeline_config.simulation_type = str(config_dict['simulation_type'])
        if 'time_step' in config_dict: pipeline_config.time_step = float(config_dict['time_step'])
        if 'duration' in config_dict: pipeline_config.duration = float(config_dict['duration'])
        if 'max_iterations' in config_dict: pipeline_config.max_iterations = int(config_dict['max_iterations'])
        if 'tolerance' in config_dict: pipeline_config.tolerance = float(config_dict['tolerance'])
        if 'write_interval' in config_dict: pipeline_config.write_interval = int(config_dict['write_interval'])
        if 'colormap' in config_dict: pipeline_config.colormap = str(config_dict['colormap'])
        if 'solver' in config_dict: pipeline_config.solver = str(config_dict['solver'])

    # Input file path
    # We need to find where the file was saved.
    # The frontend should pass the 'saved_as' name, or we need to look it up.
    # Let's assume frontend passes the full filename we returned in /upload
    
    input_path = UPLOAD_FOLDER / filename
    if not input_path.exists():
        return jsonify({'error': f'File not found: {filename}'}), 404

    # Run Pipeline (In-Process for Sidecar compatibility)
    try:
        # Create a unique output sub-folder for this run
        job_id = str(uuid.uuid4())[:8]
        job_output_dir = OUTPUT_FOLDER / job_id
        
        print(f"Running pipeline for {filename}...")
        
        # Direct function call
        results_metadata = run_simops_pipeline(
            cad_file=str(input_path),
            output_dir=str(job_output_dir),
            config=pipeline_config,
            verbose=True
        )
        
        # Use returned metadata directly
        results_data = results_metadata

        # Add relative URL paths for frontend
        results_data['vtk_url'] = f"/api/results/{job_id}/thermal_result.vtk"
        results_data['png_url'] = f"/api/results/{job_id}/temperature_map.png"

        # Add PDF URL if generated
        if 'pdf_file' in results_data:
            pdf_filename = Path(results_data['pdf_file']).name
            results_data['pdf_url'] = f"/api/results/{job_id}/{pdf_filename}"
            print(f"PDF Report generated: {results_data['pdf_url']}")

        # Generate HTML Report
        if generate_html_report:
            report_path = generate_html_report(job_id, results_data, job_output_dir)
            results_data['report_url'] = f"/api/results/{job_id}/report.html"
            print(f"HTML Report generated: {report_path}")

        return jsonify({
            "status": "success",
            "job_id": job_id,
            "results": results_data
        })
        


    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai/generate-config', methods=['POST'])
def ai_generate_config():
    """Generate simulation configuration from natural language prompt."""
    if not AISetupGenerator:
        return jsonify({'error': 'AI generator not available'}), 500
    
    try:
        data = request.json
        
        # Validate required fields
        if not data or 'prompt' not in data or 'cad_file' not in data:
            return jsonify({'error': 'Missing required fields: prompt, cad_file'}), 400
        
        prompt = data['prompt']
        cad_file = data['cad_file']
        use_mock = data.get('use_mock', False)
        
        # Generate configuration
        generator = AISetupGenerator()
        config = generator.generate_config(prompt, cad_file, use_mock=use_mock)
        
        # Return as dict
        return jsonify({
            'success': True,
            'config': config.model_dump()
        }), 200
        
    except ValueError as e:
        return jsonify({'error': f'AI generation failed: {str(e)}'}), 422
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

if __name__ == '__main__':
    # Run on port 8000 (standard for this triage plan)
    port = int(os.environ.get('PORT', 8000))
    print(f"STARTING SIMOPS BACKEND ON PORT {port}...")
    print(f"OUTPUT_FOLDER: {OUTPUT_FOLDER}")
    # IMPORTANT: threaded=False is required for gmsh to work correctly
    # gmsh.initialize() registers signal handlers which only work in the main thread
    app.run(host='0.0.0.0', port=port, debug=False, threaded=False)

