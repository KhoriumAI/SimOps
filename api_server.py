#!/usr/bin/env python3
"""
Flask API Server for Mesh Generation
Provides REST API endpoints for the React frontend
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from pathlib import Path
import json
import sys
import subprocess
import uuid
import os
from datetime import datetime
from threading import Thread, Lock
import traceback

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configuration
UPLOAD_FOLDER = Path(__file__).parent / "uploads"
OUTPUT_FOLDER = Path(__file__).parent / "outputs"
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

# Project storage (in-memory for now - could be replaced with database)
projects = {}
projects_lock = Lock()


class MeshProject:
    """Represents a mesh generation project"""

    def __init__(self, project_id: str, filename: str, filepath: str):
        self.id = project_id
        self.filename = filename
        self.filepath = filepath
        self.status = "uploaded"
        self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
        self.error_message = None
        self.output_file = None
        self.quality_metrics = None
        self.logs = []
        self.strategy = None
        self.strategy = None
        self.score = None
        self.iterations = []  # List of {timestamp, output_file, metrics, strategy}


    def update_status(self, status: str, error_message: str = None):
        """Update project status"""
        self.status = status
        self.updated_at = datetime.now().isoformat()
        if error_message:
            self.error_message = error_message

    def add_log(self, message: str):
        """Add log message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")

    def to_dict(self):
        """Convert to dictionary"""
        return {
            "id": self.id,
            "filename": self.filename,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error_message": self.error_message,
            "output_file": self.output_file,
            "quality_metrics": self.quality_metrics,
            "logs": self.logs,
            "strategy": self.strategy,
            "score": self.score,
            "iterations": self.iterations
        }



def run_mesh_generation(project_id: str, quality_params: dict = None):
    """Run mesh generation in background thread"""
    with projects_lock:
        project = projects.get(project_id)

    if not project:
        return

    try:
        project.update_status("executing_mesh_generation")
        project.add_log("[INFO] Starting mesh generation...")

        # Get path to mesh worker subprocess
        worker_script = Path(__file__).parent.parent / "mesh_worker_subprocess.py"

        # Prepare arguments
        cmd = [sys.executable, str(worker_script), project.filepath, str(OUTPUT_FOLDER)]
        
        # Add quality params if provided
        if quality_params:
            cmd.extend(['--quality-params', json.dumps(quality_params)])

        # Run mesh generation subprocess
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        # Read output line by line
        for line in process.stdout:
            line = line.strip()
            if line:
                # Check if it's the final JSON result
                if line.startswith('{') and '"success"' in line:
                    result = json.loads(line)
                    if result.get('success'):
                        project.add_log("[SUCCESS] Meshing completed!")
                        project.update_status("completed")
                        project.output_file = result.get('output_file')
                        project.quality_metrics = result.get('metrics')
                        project.strategy = result.get('strategy')
                        project.score = result.get('score')
                        
                        # Save as an iteration
                        iteration = {
                            "id": len(project.iterations) + 1,
                            "timestamp": datetime.now().isoformat(),
                            "output_file": project.output_file,
                            "quality_metrics": project.quality_metrics,
                            "strategy": project.strategy,
                            "score": project.score
                        }
                        project.iterations.append(iteration)

                    else:
                        error = result.get('error', 'Unknown error')
                        project.add_log(f"[ERROR] Meshing failed: {error}")
                        project.update_status("error", error)
                else:
                    # Regular log line
                    project.add_log(line)

        # Wait for process to complete
        process.wait()

        # Check for stderr errors
        stderr = process.stderr.read()
        if stderr and project.status != "completed":
            project.add_log(f"[ERROR] {stderr}")
            project.update_status("error", stderr)

    except Exception as e:
        error_msg = f"Exception during mesh generation: {str(e)}"
        project.add_log(f"[ERROR] {error_msg}")
        project.update_status("error", error_msg)
        print(traceback.format_exc())


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "mesh-generation-api"})


@app.route('/api/upload', methods=['POST'])
def upload_cad_file():
    """Upload CAD file and create new project"""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Check file extension
    allowed_extensions = {'.step', '.stp', '.stl'}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        return jsonify({"error": f"Invalid file type. Allowed: {allowed_extensions}"}), 400

    # Create project
    project_id = str(uuid.uuid4())
    filename = file.filename
    filepath = UPLOAD_FOLDER / f"{project_id}_{filename}"

    # Save file
    file.save(str(filepath))

    # Create project object
    project = MeshProject(project_id, filename, str(filepath))

    with projects_lock:
        projects[project_id] = project

    return jsonify({
        "project_id": project_id,
        "filename": filename,
        "status": "uploaded"
    })


@app.route('/api/projects/<project_id>/generate', methods=['POST'])
def generate_mesh(project_id: str):
    """Start mesh generation for a project"""
    with projects_lock:
        project = projects.get(project_id)

    if not project:
        return jsonify({"error": "Project not found"}), 404

    if project.status not in ["uploaded", "error"]:
        return jsonify({"error": f"Cannot generate mesh - project status is {project.status}"}), 400

    # Get quality params from request
    quality_params = request.json if request.is_json else None

    # Start mesh generation in background thread
    thread = Thread(target=run_mesh_generation, args=(project_id, quality_params))
    thread.daemon = True
    thread.start()

    return jsonify({"message": "Mesh generation started", "project_id": project_id})


@app.route('/api/projects/<project_id>/status', methods=['GET'])
def get_project_status(project_id: str):
    """Get project status and details"""
    with projects_lock:
        project = projects.get(project_id)

    if not project:
        return jsonify({"error": "Project not found"}), 404

    return jsonify(project.to_dict())


@app.route('/api/projects/<project_id>/logs', methods=['GET'])
def get_project_logs(project_id: str):
    """Get project logs"""
    with projects_lock:
        project = projects.get(project_id)

    if not project:
        return jsonify({"error": "Project not found"}), 404

    return jsonify({"logs": project.logs})


@app.route('/api/projects/<project_id>/download', methods=['GET'])
def download_mesh(project_id: str):
    """Download generated mesh file"""
    with projects_lock:
        project = projects.get(project_id)

    if not project:
        return jsonify({"error": "Project not found"}), 404

    if project.status != "completed" or not project.output_file:
        return jsonify({"error": "Mesh not ready for download"}), 400

    output_path = Path(project.output_file)
    if not output_path.exists():
        return jsonify({"error": "Output file not found"}), 404

    return send_file(
        str(output_path),
        as_attachment=True,
        download_name=f"{Path(project.filename).stem}_mesh.msh"
    )


@app.route('/api/projects/<project_id>/mesh-data', methods=['GET'])
def get_mesh_data(project_id: str):
    """Get mesh data for 3D visualization"""
    with projects_lock:
        project = projects.get(project_id)

    if not project:
        return jsonify({"error": "Project not found"}), 404

    if project.status != "completed" or not project.output_file:
        return jsonify({"error": "Mesh not ready"}), 400

    try:
        # Parse .msh file and return geometry
        return parse_msh_file_optimized(project.output_file)
    except Exception as e:
        return jsonify({"error": f"Failed to parse mesh file: {str(e)}"}), 500


def parse_msh_file_optimized(msh_filepath: str):
    """
    Parse Gmsh .msh file and return optimized flat arrays for Three.js BufferGeometry.
    Includes ALL quality metrics for visualization.
    """
    vertices = []
    colors = [] # Default RGB
    
    # Store quality data arrays for client-side filtering/coloring
    quality_arrays = {
        "sicn": [],
        "gamma": [],
        "skewness": [],
        "aspect_ratio": []
    }
    
    # Check for quality file
    quality_filepath = Path(msh_filepath).with_suffix('.quality.json')
    per_element_data = {
        "quality": {}, # SICN
        "gamma": {},
        "skewness": {},
        "aspect_ratio": {}
    }
    
    if quality_filepath.exists():
        try:
            with open(quality_filepath, 'r') as f:
                qdata = json.load(f)
                # Load all available metrics
                per_element_data["quality"] = {str(k): v for k, v in qdata.get('per_element_quality', {}).items()}
                per_element_data["gamma"] = {str(k): v for k, v in qdata.get('per_element_gamma', {}).items()}
                per_element_data["skewness"] = {str(k): v for k, v in qdata.get('per_element_skewness', {}).items()}
                per_element_data["aspect_ratio"] = {str(k): v for k, v in qdata.get('per_element_aspect_ratio', {}).items()}
        except Exception as e:
            print(f"Failed to load quality file: {e}")

    # Helper to get color from value (0.0 to 1.0) - Green to Red
    # Note: client side can re-color, but we provide defaults
    def get_color(val):
        # 0.0 (Bad) -> Red, 1.0 (Good) -> Green
        # Simple lerp
        r = 1.0 - val
        g = val
        b = 0.0
        # Boost for visibility
        if val < 0.1: r, g, b = 1.0, 0.0, 0.0
        elif val > 0.9: r, g, b = 0.0, 1.0, 0.0
        return [r, g, b]

    try:
        with open(msh_filepath, 'r') as f:
            lines = f.readlines()
            
        # First pass: Load all nodes
        nodes_coords = {}
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line == "$Nodes":
                i += 2
                while i < len(lines) and lines[i].strip() != "$EndNodes":
                    parts = lines[i].strip().split()
                    if len(parts) == 4:
                        num_nodes = int(parts[3])
                        i += 1
                        tags = []
                        for _ in range(num_nodes):
                            tags.append(int(lines[i].strip()))
                            i += 1
                        for tag in tags:
                            coords = lines[i].strip().split()
                            nodes_coords[tag] = [float(coords[0]), float(coords[1]), float(coords[2])]
                            i += 1
                    else:
                        i += 1
            else:
                i += 1
                
        # Second pass: Elements -> Vertices + Colors + Metrics
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line == "$Elements":
                i += 2
                while i < len(lines) and lines[i].strip() != "$EndElements":
                    parts = lines[i].strip().split()
                    if len(parts) == 4:
                        elem_type = int(parts[2])
                        num_elems = int(parts[3])
                        i += 1
                        
                        # Type 2 (Triangle) - Surface
                        # Type 4 (Tetrahedron) - Volume (we only show surface faces for performance unless requested)
                        # For now, we only render Triangles (surfaces)
                        if elem_type == 2: 
                            for _ in range(num_elems):
                                el_parts = lines[i].strip().split()
                                if len(el_parts) >= 4:
                                    el_tag = str(el_parts[0])
                                    n1, n2, n3 = int(el_parts[1]), int(el_parts[2]), int(el_parts[3])
                                    
                                    if n1 in nodes_coords and n2 in nodes_coords and n3 in nodes_coords:
                                        # Add 3 vertices
                                        vertices.extend(nodes_coords[n1])
                                        vertices.extend(nodes_coords[n2])
                                        vertices.extend(nodes_coords[n3])
                                        
                                        # Get metrics (fallback to defaults)
                                        sicn = per_element_data["quality"].get(el_tag, 1.0)
                                        gamma = per_element_data["gamma"].get(el_tag, 1.0)
                                        skew = per_element_data["skewness"].get(el_tag, 0.0)
                                        ar = per_element_data["aspect_ratio"].get(el_tag, 1.0)
                                        
                                        # Add to quality arrays (3 times per face)
                                        for _ in range(3):
                                            quality_arrays["sicn"].append(sicn)
                                            quality_arrays["gamma"].append(gamma)
                                            quality_arrays["skewness"].append(skew)
                                            quality_arrays["aspect_ratio"].append(ar)

                                        # Base color (SICN-based)
                                        c = get_color(sicn)
                                        colors.extend(c * 3) # 3 vertices
                                        
                                i += 1
                        else:
                            # Skip other types
                            for _ in range(num_elems):
                                i += 1
                    else:
                        i += 1
            else:
                i += 1
                
    except Exception as e:
        return {"error": str(e)}

    return {
        "vertices": vertices,
        "colors": colors,
        "metrics": quality_arrays,
        "numVertices": len(vertices) // 3,
        "numTriangles": len(vertices) // 9
    }



@app.route('/api/projects', methods=['GET'])
def list_projects():
    """List all projects"""
    with projects_lock:
        project_list = [p.to_dict() for p in projects.values()]

    # Sort by creation time (newest first)
    project_list.sort(key=lambda x: x['created_at'], reverse=True)

    return jsonify({"projects": project_list})


if __name__ == '__main__':
    print("=" * 70)
    print("MESH GENERATION API SERVER")
    print("=" * 70)
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print("\nStarting server on http://localhost:5000")
    print("=" * 70)

    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)

# ==========================================
# TEMPORARY: Dummy Login for Frontend Access
# ==========================================
@app.route('/login', methods=['POST'])
@app.route('/api/login', methods=['POST'])
def dummy_login():
    """Accepts any login attempt and returns a fake token"""
    return jsonify({
        "access_token": "fake-super-user-token",
        "token_type": "bearer",
        "user": {
            "id": 1,
            "username": "admin",
            "email": "admin@khorium.com",
            "roles": ["admin"]
        }
    }), 200
