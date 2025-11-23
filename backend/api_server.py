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
        # Parse .msh file and extract nodes/elements
        mesh_data = parse_msh_file(project.output_file)
        return jsonify(mesh_data)
    except Exception as e:
        return jsonify({"error": f"Failed to parse mesh file: {str(e)}"}), 500


    return parse_msh_file_optimized(msh_filepath)


def parse_msh_file_optimized(msh_filepath: str):
    """
    Parse Gmsh .msh file and return optimized flat arrays for Three.js BufferGeometry.
    Also loads quality data if available.
    """
    vertices = []
    indices = []
    colors = []  # RGB 0-1
    
    # Check for quality file
    quality_filepath = Path(msh_filepath).with_suffix('.quality.json')
    per_element_quality = {}
    quality_metrics = {}
    
    if quality_filepath.exists():
        try:
            with open(quality_filepath, 'r') as f:
                qdata = json.load(f)
                per_element_quality = qdata.get('per_element_quality', {})
                # Normalize keys to strings just in case
                per_element_quality = {str(k): v for k, v in per_element_quality.items()}
                
                # Get threshold for coloring
                quality_metrics = {
                    'sicn_10_percentile': qdata.get('quality_threshold_10', 0.3)
                }
        except Exception as e:
            print(f"Failed to load quality file: {e}")

    # 1. Read Nodes
    nodes_map = {} # tag -> index in vertices array (0, 1, 2...)
    
    try:
        with open(msh_filepath, 'r') as f:
            lines = f.readlines()
            
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line == "$Nodes":
                i += 2 # Skip header
                while i < len(lines) and lines[i].strip() != "$EndNodes":
                    parts = lines[i].strip().split()
                    if len(parts) == 4: # Block header
                        num_nodes = int(parts[3])
                        i += 1
                        # Read tags
                        tags = []
                        for _ in range(num_nodes):
                            tags.append(int(lines[i].strip()))
                            i += 1
                        # Read coords
                        for tag in tags:
                            coords = lines[i].strip().split()
                            # Append to flat vertices array
                            vertices.extend([float(coords[0]), float(coords[1]), float(coords[2])])
                            # Map tag to vertex index (vertices array index / 3)
                            nodes_map[tag] = (len(vertices) // 3) - 1
                            i += 1
                    else:
                        i += 1
                        
            elif line == "$Elements":
                i += 2 # Skip header
                while i < len(lines) and lines[i].strip() != "$EndElements":
                    parts = lines[i].strip().split()
                    if len(parts) == 4: # Block header
                        elem_type = int(parts[2])
                        num_elems = int(parts[3])
                        i += 1
                        
                        # We only care about triangles (type 2) for surface rendering
                        # If we want to see inside, we'd need a different approach (clipping volume)
                        # But standard WebGL usually renders surfaces.
                        if elem_type == 2: # Triangle
                            for _ in range(num_elems):
                                # tag node1 node2 node3
                                el_parts = lines[i].strip().split()
                                if len(el_parts) >= 4:
                                    el_tag = el_parts[0]
                                    n1, n2, n3 = int(el_parts[1]), int(el_parts[2]), int(el_parts[3])
                                    
                                    if n1 in nodes_map and n2 in nodes_map and n3 in nodes_map:
                                        indices.extend([nodes_map[n1], nodes_map[n2], nodes_map[n3]])
                                        
                                        # Determine color based on quality
                                        # Default green
                                        r, g, b = 0.2, 0.7, 0.4
                                        
                                        if per_element_quality:
                                            q = per_element_quality.get(el_tag)
                                            if q is not None:
                                                threshold = quality_metrics.get('sicn_10_percentile', 0.3)
                                                if q <= threshold:
                                                    r, g, b = 1.0, 0.0, 0.0 # Red (Worst)
                                                elif q < 0.3:
                                                    r, g, b = 1.0, 0.5, 0.0 # Orange
                                                elif q < 0.5:
                                                    r, g, b = 1.0, 1.0, 0.0 # Yellow
                                                elif q < 0.7:
                                                    r, g, b = 0.5, 1.0, 0.0 # Yellow-Green
                                                else:
                                                    r, g, b = 0.2, 0.7, 0.4 # Green (Good)
                                        
                                        # Add color for EACH vertex of the triangle (flat shading look requires splitting vertices usually, 
                                        # but for smooth shading we share. For flat look in Three.js with shared vertices, 
                                        # we might need to use 'flatShading: true' material, but vertex colors blend.
                                        # To get true flat faceted colors with shared vertices is tricky in indexed geometry.
                                        # We'll provide colors per vertex. If vertices are shared, color will blend.
                                        # To match desktop "Flat" look perfectly, we might need non-indexed geometry (de-indexed).
                                        # BUT de-indexed is 3x memory.
                                        # Let's try to map colors to faces? Three.js BufferGeometry colors are per vertex.
                                        # We will rely on the frontend to de-index if it wants perfect flat shading with colors,
                                        # OR we just de-index here. De-indexing is safer for "Low Poly" look.
                                        pass 
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
        print(f"Error parsing mesh: {e}")
        return {"error": str(e)}

    # Post-processing: To achieve the "Flat" look with per-face colors like the desktop app,
    # we MUST de-index the geometry (explode it).
    # This means every triangle has 3 unique vertices.
    
    flat_vertices = []
    flat_colors = []
    # We don't need indices for non-indexed geometry
    
    # Re-read elements to build de-indexed arrays
    # This is inefficient (reading twice), but safer for memory in Python than holding massive arrays in memory if we can stream.
    # Actually, let's just process the indices we collected.
    
    # We need to reconstruct the color logic per face
    # We have indices [n1, n2, n3, n4, n5, n6...]
    # We need to map these back to the element tags to get quality... 
    # The previous loop didn't store element tags with indices.
    # Let's redo the loop structure slightly to be single-pass de-indexed.
    
    vertices = [] # Clear
    colors = []
    
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
                
        # Second pass: Elements -> De-indexed Vertices + Colors
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
                        
                        if elem_type == 2: # Triangle
                            for _ in range(num_elems):
                                el_parts = lines[i].strip().split()
                                if len(el_parts) >= 4:
                                    el_tag = el_parts[0]
                                    n1, n2, n3 = int(el_parts[1]), int(el_parts[2]), int(el_parts[3])
                                    
                                    if n1 in nodes_coords and n2 in nodes_coords and n3 in nodes_coords:
                                        # Add 3 vertices
                                        vertices.extend(nodes_coords[n1])
                                        vertices.extend(nodes_coords[n2])
                                        vertices.extend(nodes_coords[n3])
                                        
                                        # Determine color
                                        r, g, b = 0.2, 0.7, 0.4 # Default Green
                                        if per_element_quality:
                                            q = per_element_quality.get(el_tag)
                                            if q is not None:
                                                threshold = quality_metrics.get('sicn_10_percentile', 0.3)
                                                if q <= threshold:
                                                    r, g, b = 1.0, 0.0, 0.0
                                                elif q < 0.3:
                                                    r, g, b = 1.0, 0.5, 0.0
                                                elif q < 0.5:
                                                    r, g, b = 1.0, 1.0, 0.0
                                                elif q < 0.7:
                                                    r, g, b = 0.5, 1.0, 0.0
                                        
                                        # Add color for 3 vertices
                                        colors.extend([r, g, b, r, g, b, r, g, b])
                                i += 1
                        else:
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
