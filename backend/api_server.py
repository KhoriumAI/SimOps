#!/usr/bin/env python3
"""
Flask API Server for Mesh Generation
With JWT Authentication and SQLAlchemy Database
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_jwt_extended import JWTManager, jwt_required, get_jwt_identity
from pathlib import Path
import json
import sys
import os
import subprocess
import uuid
from datetime import datetime
from threading import Thread, Lock
import traceback
import shutil

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config
from models import db, User, Project, MeshResult, TokenBlocklist, ActivityLog, DownloadRecord
from werkzeug.utils import secure_filename
from routes.auth import auth_bp, check_if_token_revoked
from routes.batch import batch_bp
from storage import get_storage, S3Storage, LocalStorage
from slicing import generate_slice_mesh, parse_msh_for_slicing
import numpy as np
try:
    import meshio
except ImportError:
    meshio = None


def create_app(config_class=None):
    """Application factory"""
    app = Flask(__name__)
    
    # Load configuration
    if config_class is None:
        config_class = get_config()
    app.config.from_object(config_class)
    
    # Ensure folders exist
    Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)
    Path(app.config['OUTPUT_FOLDER']).mkdir(parents=True, exist_ok=True)

    # Initialize CORS - Allow frontend origins
    default_origins = [
        "http://localhost:5173",
        "http://localhost:3000", 
        "http://127.0.0.1:5173",
        "http://muaz-mesh-web-dev.s3-website-us-west-1.amazonaws.com"
    ]
    
    # Get origins from config or environment
    config_origins = app.config.get('CORS_ORIGINS', [])
    if isinstance(config_origins, str):
        config_origins = [o.strip() for o in config_origins.split(',') if o.strip()]
        
    env_origins = os.environ.get("CORS_ORIGINS", "")
    if env_origins:
        env_origins_list = [o.strip() for o in env_origins.split(',') if o.strip()]
    else:
        env_origins_list = []

    # Combine all origins
    cors_origins = sorted(list(set(default_origins + config_origins + env_origins_list)))
    
    # Check if we should allow all (if '*' is present)
    if '*' in cors_origins:
        cors_origins = "*"

    print(f"[CORS] Allowed Origins: {cors_origins}")
    
    CORS(app, resources={
        r"/api/*": {
            "origins": cors_origins,
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
            "supports_credentials": True
        },
        r"/auth/*": {
            "origins": cors_origins,
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
            "supports_credentials": True
        }
    })
    
    # Initialize database
    db.init_app(app)
    
    # Initialize JWT
    jwt = JWTManager(app)
    
    @jwt.token_in_blocklist_loader
    def check_token_revoked_callback(jwt_header, jwt_payload):
        return check_if_token_revoked(jwt_header, jwt_payload)
    
    @jwt.expired_token_loader
    def expired_token_callback(jwt_header, jwt_payload):
        # Token expired - don't log payload details
        return jsonify({'error': 'Token has expired'}), 401
    
    @jwt.invalid_token_loader
    def invalid_token_callback(error):
        # Invalid token
        return jsonify({'error': f'Invalid token: {error}'}), 401
    
    @jwt.unauthorized_loader
    def missing_token_callback(error):
        # Missing token
        return jsonify({'error': 'Authorization token required'}), 401
    
    @jwt.token_verification_failed_loader
    def token_verification_failed_callback(jwt_header, jwt_payload):
        # Token verification failed
        return jsonify({'error': 'Token verification failed'}), 401
    
    # Register blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(batch_bp)
    
    # Ensure instance folder exists for SQLite database
    instance_dir = Path(__file__).parent / 'instance'
    instance_dir.mkdir(parents=True, exist_ok=True)
    
    # Create database tables
    with app.app_context():
        db.create_all()
    
    # Register routes
    register_routes(app)
    
    return app


generation_lock = Lock()
running_processes = {}


def run_mesh_generation(app, project_id: str, quality_params: dict = None):
    """Run mesh generation in background thread"""
    import time
    import tempfile
    start_time = time.time()
    
    with app.app_context():
        project = Project.query.get(project_id)
        user = User.query.get(project.user_id) if project else None

        if not project:
            print(f"[MESH GEN] Project {project_id} not found")
            return

        try:
            project.status = 'processing'
            project.mesh_count = (project.mesh_count or 0) + 1
            db.session.commit()
            print(f"[MESH GEN] Started for project {project_id}")

            worker_script = Path(__file__).parent.parent / "apps" / "cli" / "mesh_worker_subprocess.py"
            output_folder = Path(app.config['OUTPUT_FOLDER'])
            output_folder.mkdir(parents=True, exist_ok=True)
            
            # Get storage backend
            storage = get_storage()
            use_s3 = app.config.get('USE_S3', False)
            
            # If using S3, download file to local temp folder for processing
            if use_s3 and project.filepath.startswith('s3://'):
                local_input_dir = Path(tempfile.mkdtemp(prefix='meshgen_input_'))
                local_input_file = local_input_dir / Path(project.filename).name
                print(f"[MESH GEN] Downloading from S3: {project.filepath}")
                storage.download_to_local(project.filepath, str(local_input_file))
                input_filepath = str(local_input_file)
            else:
                input_filepath = project.filepath
            
            print(f"[MESH GEN] Worker script: {worker_script}")
            print(f"[MESH GEN] Input file: {input_filepath}")

            cmd = [sys.executable, str(worker_script), input_filepath, str(output_folder)]

            if quality_params:
                cmd.extend(['--quality-params', json.dumps(quality_params)])

            logs = []
            
            # Create initial mesh result to store logs during processing
            mesh_result = MeshResult(
                project_id=project_id,
                strategy='processing',
                logs=logs,
                params=quality_params
            )
            db.session.add(mesh_result)
            db.session.commit()
            result_id = mesh_result.id

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Track running process
            with generation_lock:
                running_processes[project_id] = process

            try:
                line_count = 0
                for line in process.stdout:
                    line = line.strip()
                    if line:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        log_line = f"[{timestamp}] {line}"
                        logs.append(log_line)
                        print(f"[MESH GEN] {log_line}")
                        line_count += 1
                        
                        # Update logs in database every 5 lines
                        if line_count % 5 == 0:
                            mesh_result = db.session.get(MeshResult, result_id)
                            if mesh_result:
                                mesh_result.logs = logs.copy()
                                db.session.commit()

                        if line.startswith('{') and '"success"' in line:
                            try:
                                result = json.loads(line)
                                mesh_result = db.session.get(MeshResult, result_id)
                                if result.get('success'):
                                    mesh_result.strategy = result.get('strategy')
                                    mesh_result.score = result.get('score')
                                    local_output_path = result.get('output_file')
                                    mesh_result.quality_metrics = result.get('quality_metrics', {})
                                    mesh_result.completed_at = datetime.utcnow()
                                    mesh_result.processing_time = time.time() - start_time
                                    mesh_result.node_count = result.get('total_nodes')
                                    mesh_result.element_count = result.get('total_elements')
                                    
                                    # Get output file size
                                    if local_output_path and Path(local_output_path).exists():
                                        mesh_result.output_size = Path(local_output_path).stat().st_size
                                        
                                        # Try to load full results from the _result.json file if available
                                        # (Contains heavy arrays that are sanitized from the stdout JSON)
                                        full_result_path = Path(local_output_path).with_name(Path(local_output_path).stem + "_result.json")
                                        if full_result_path.exists():
                                            try:
                                                with open(full_result_path, 'r') as f:
                                                    full_data = json.load(f)
                                                    # Merge quality data back into result for processing
                                                    for key in ['per_element_quality', 'per_element_gamma', 'per_element_skewness', 'per_element_aspect_ratio']:
                                                        if key in full_data and (not result.get(key)):
                                                            result[key] = full_data[key]
                                                print(f"[MESH GEN] Loaded quality data from {full_result_path.name}")
                                            except Exception as e:
                                                print(f"[MESH GEN] Warning: Could not load full result JSON: {e}")

                                        # Save quality data to a JSON file alongside the mesh
                                        per_element_quality = result.get('per_element_quality', {})
                                        per_element_gamma = result.get('per_element_gamma', {})
                                        per_element_skewness = result.get('per_element_skewness', {})
                                        per_element_aspect_ratio = result.get('per_element_aspect_ratio', {})
                                        quality_metrics = result.get('quality_metrics', {})
                                        
                                        if per_element_quality:
                                            quality_filepath = Path(local_output_path).with_suffix('.quality.json')
                                            quality_data = {
                                                'per_element_quality': per_element_quality,
                                                'per_element_gamma': per_element_gamma,
                                                'per_element_skewness': per_element_skewness,
                                                'per_element_aspect_ratio': per_element_aspect_ratio,
                                                'quality_metrics': quality_metrics,
                                                'quality_threshold_10': 0.1,  # 10% threshold
                                            }
                                            with open(quality_filepath, 'w') as f:
                                                json.dump(quality_data, f)
                                            print(f"[MESH GEN] Saved quality data: {len(per_element_quality)} elements")
                                        
                                        # If using S3, upload mesh result to S3
                                        if use_s3 and user:
                                            mesh_filename = Path(local_output_path).name
                                            s3_path = storage.save_local_file(
                                                local_path=local_output_path,
                                                filename=mesh_filename,
                                                user_email=user.email,
                                                file_type='mesh'
                                            )
                                            mesh_result.output_path = s3_path
                                            print(f"[MESH GEN] Uploaded mesh to S3: {s3_path}")
                                            
                                            # Also upload quality file to S3
                                            if per_element_quality:
                                                quality_filename = quality_filepath.name
                                                storage.save_local_file(
                                                    local_path=str(quality_filepath),
                                                    filename=quality_filename,
                                                    user_email=user.email,
                                                    file_type='mesh'
                                                )
                                        else:
                                            mesh_result.output_path = local_output_path
                                    
                                    project.status = 'completed'
                                    project.last_accessed = datetime.utcnow()
                                    logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] [SUCCESS] Meshing completed in {mesh_result.processing_time:.1f}s!")
                                else:
                                    error = result.get('error', 'Unknown error')
                                    project.status = 'error'
                                    project.error_message = error
                                    logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] [ERROR] {error}")
                                mesh_result.logs = logs
                                db.session.commit()
                            except json.JSONDecodeError:
                                pass
            finally:
                with generation_lock:
                    if project_id in running_processes:
                        del running_processes[project_id]

            process.wait()
            print(f"[MESH GEN] Process exited with code {process.returncode}")

            # Final update
            mesh_result = db.session.get(MeshResult, result_id)
            if mesh_result:
                mesh_result.logs = logs
            
            if project.status == 'processing':
                # Process finished but no success message - check for errors
                if process.returncode != 0:
                    project.status = 'error'
                    project.error_message = f"Process exited with code {process.returncode}"
                else:
                    project.status = 'error'
                    project.error_message = "Mesh generation finished without result"

            db.session.commit()
            print(f"[MESH GEN] Final status: {project.status}")

        except Exception as e:
            print(f"[MESH GEN ERROR] {e}")
            project.status = 'error'
            project.error_message = str(e)
            db.session.commit()
            print(traceback.format_exc())


def register_routes(app):
    """Register API routes"""
    
    @app.before_request
    def log_request_info():
        # Debug: Log Authorization header for all requests
        # Request logging (without sensitive auth data)
        if request.path.startswith('/api/') and os.environ.get('FLASK_DEBUG'):
            print(f"[API] {request.method} {request.path}")

    @app.route('/', methods=['GET'])
    def index():
        return "<h1>Khorium MeshGen Backend</h1><p>Status: Running</p><p>API: /api/health</p>"

    @app.route('/api/health', methods=['GET'])
    def health_check():
        return jsonify({"status": "healthy", "service": "mesh-generation-api", "version": "2.0"})

    @app.route('/api/strategies', methods=['GET'])
    def get_mesh_strategies():
        """
        Return available mesh generation strategies.
        Frontend should fetch this on load instead of hardcoding.
        """
        strategies = [
            {
                'id': 'fast_tet_delaunay',
                'name': 'Tet (Fast)',
                'description': 'Fast single-pass HXT Delaunay - ideal for batch processing',
                'element_type': 'tet',
                'recommended': True,
                'fast': True
            },
            {
                'id': 'tetrahedral_delaunay',
                'name': 'Tetrahedral (Delaunay)',
                'description': 'Exhaustive strategy search - best quality, slower',
                'element_type': 'tet',
                'recommended': False
            },
            {
                'id': 'tetrahedral_frontal',
                'name': 'Tetrahedral (Frontal)',
                'description': 'Advancing front method - good for boundary layers',
                'element_type': 'tet',
                'recommended': False
            },
            {
                'id': 'tetrahedral_hxt',
                'name': 'Tetrahedral (HXT)',
                'description': 'High-performance parallel meshing',
                'element_type': 'tet',
                'recommended': False
            },
            {
                'id': 'hex_dominant',
                'name': 'Hex-Dominant',
                'description': 'Hexahedral mesh with tet fill - best for CFD',
                'element_type': 'hex',
                'recommended': False
            },
            {
                'id': 'gpu_delaunay',
                'name': 'GPU Delaunay',
                'description': 'GPU-accelerated meshing (requires CUDA)',
                'element_type': 'tet',
                'recommended': False,
                'requires': 'cuda'
            },
            {
                'id': 'polyhedral',
                'name': 'Polyhedral',
                'description': 'Polyhedral cells - experimental',
                'element_type': 'poly',
                'recommended': False,
                'experimental': True
            }
        ]
        
        # Return list of strategy names for simple dropdown use
        # and full details for advanced UI
        return jsonify({
            'strategies': strategies,
            'names': [s['name'] for s in strategies],
            'default': 'Tet (Fast)'
        })

    @app.route('/api/upload', methods=['POST'])
    @jwt_required()
    def upload_cad_file():
        """
        Upload CAD file endpoint
        
        Handles large file uploads (up to 500MB) with proper timeout handling.
        Frontend implements 10-minute client-side timeout with AbortController.
        Backend processes files asynchronously for preview generation.
        """

        current_user_id = int(get_jwt_identity())
        user = User.query.get(current_user_id)
        
        if not user:
            return jsonify({"error": "User not found"}), 404

        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in app.config['ALLOWED_EXTENSIONS']:
            return jsonify({"error": f"Invalid file type"}), 400

        project_id = str(uuid.uuid4())
        original_filename = file.filename
        filename = secure_filename(file.filename) or f"file{file_ext}"
        storage_filename = f"{project_id}_{filename}"
        
        # Calculate file size and hash before saving
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        import hashlib
        file_content = file.read()
        file_hash = hashlib.sha256(file_content).hexdigest()
        file.seek(0)  # Reset again for saving
        
        # Use storage abstraction (local or S3 based on config)
        storage = get_storage()
        
        # 1. Save the incoming file to a temporary path explicitly for preview generation
        import tempfile
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, filename)
        with open(temp_path, 'wb') as f:
            f.write(file_content)
        
        preview_path = None
        try:
            # 2. Generate the preview IMMEDIATELY using the local temp file
            print(f"[PREVIEW] Generating coarse mesh locally for {filename}...", flush=True)
            preview_data = parse_step_file_for_preview(temp_path)
            
            if preview_data and "error" not in preview_data:
                # Save preview data as JSON to upload
                preview_temp_path = os.path.join(temp_dir, f"{project_id}_preview.json")
                with open(preview_temp_path, 'w') as f:
                    json.dump(preview_data, f)
                
                # 3. Upload preview to S3
                preview_filename = f"{project_id}_preview.json"
                preview_path = storage.save_local_file(
                    local_path=preview_temp_path,
                    filename=preview_filename,
                    user_email=user.email,
                    file_type='uploads'
                )
                print(f"[PREVIEW] Preview uploaded: {preview_path}")
        except Exception as e:
            print(f"[PREVIEW ERROR] Failed to generate/upload preview: {e}")
            preview_path = None

        try:
            # 4. Upload original file to storage
            # We can use the temp file we already created
            filepath = storage.save_local_file(
                local_path=temp_path,
                filename=storage_filename,
                user_email=user.email,
                file_type='uploads'
            )
            print(f"[UPLOAD] File saved to: {filepath}")
        except Exception as e:
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"[UPLOAD ERROR] Failed to save file: {e}")
            return jsonify({"error": "Failed to save file"}), 500

        # Cleanup local temp files
        shutil.rmtree(temp_dir, ignore_errors=True)

        project = Project(
            id=project_id,
            user_id=current_user_id,
            filename=filename,
            original_filename=original_filename,
            filepath=filepath,  # Can be local path or S3 URI
            preview_path=preview_path,
            file_size=file_size,
            file_hash=file_hash,
            mime_type=file.content_type,
            status='uploaded'
        )
        
        # Log activity
        activity = ActivityLog(
            user_id=current_user_id,
            action='upload',
            resource_type='project',
            resource_id=project_id,
            details={
                'filename': original_filename,
                'file_size': file_size,
                'file_type': file_ext,
                'storage_path': filepath,
                'preview_generated': preview_path is not None
            },
            ip_address=request.remote_addr,
            user_agent=request.user_agent.string[:500] if request.user_agent else None
        )
        
        # Update user storage
        user.storage_used = (user.storage_used or 0) + file_size

        try:
            db.session.add(project)
            db.session.add(activity)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            import traceback
            print(f"[DATABASE ERROR] Failed to commit project/activity: {e}")
            traceback.print_exc()
            # Try to clean up the uploaded files
            try:
                storage.delete_file(filepath)
                if preview_path:
                    storage.delete_file(preview_path)
            except:
                pass
            return jsonify({"error": f"Failed to create project: {str(e)}"}), 500

        return jsonify({
            "project_id": project_id,
            "filename": filename,
            "status": "uploaded",
            "preview_ready": preview_path is not None
        })

    @app.route('/api/projects/<project_id>/generate', methods=['POST'])
    @jwt_required()
    def generate_mesh(project_id: str):
        current_user_id = int(get_jwt_identity())
        project = Project.query.get(project_id)

        if not project:
            return jsonify({"error": "Project not found"}), 404

        if project.user_id != current_user_id:
            return jsonify({"error": "Access denied"}), 403
        
        if project.status not in ["uploaded", "completed", "error"]:
            return jsonify({"error": f"Cannot generate mesh - status is {project.status}"}), 400

        quality_params = request.json if request.is_json else None

        thread = Thread(target=run_mesh_generation, args=(app, project_id, quality_params))
        thread.daemon = True
        thread.start()

        return jsonify({"message": "Mesh generation started", "project_id": project_id})

    @app.route('/api/projects/<project_id>/stop', methods=['POST'])
    @jwt_required()
    def stop_mesh(project_id: str):
        current_user_id = int(get_jwt_identity())
        project = Project.query.get(project_id)

        if not project:
            return jsonify({"error": "Project not found"}), 404

        if project.user_id != current_user_id:
            return jsonify({"error": "Access denied"}), 403
        
        # Try to find and kill the process
        process_killed = False
        with generation_lock:
            if project_id in running_processes:
                try:
                    print(f"[API] Stopping process for project {project_id}")
                    running_processes[project_id].terminate()
                    # Give it a tiny bit of time to cleanup?
                    process_killed = True
                except Exception as e:
                    print(f"[API] Error stopping process: {e}")
                    # Try kill if terminate failed
                    try:
                        running_processes[project_id].kill()
                        process_killed = True
                    except Exception as e2:
                        print(f"[API] Error killing process: {e2}")

        # Force status update
        if project.status == 'processing':
            project.status = 'stopped'
            db.session.commit()
            return jsonify({"message": "Mesh generation stopped", "process_killed": process_killed})
        else:
            # It might have already finished or been stopped
            return jsonify({"message": f"Project is already in state {project.status}", "process_killed": process_killed})


    @app.route('/api/projects/<project_id>/status', methods=['GET'])
    @jwt_required()
    def get_project_status(project_id: str):
        current_user_id = int(get_jwt_identity())
        project = Project.query.get(project_id)

        if not project:
            return jsonify({"error": "Project not found"}), 404

        if project.user_id != current_user_id:
            return jsonify({"error": "Access denied"}), 403

        return jsonify(project.to_dict(include_results=True))

    @app.route('/api/projects/<project_id>/slice', methods=['POST'])
    @jwt_required()
    def slice_mesh(project_id: str):
        current_user_id = int(get_jwt_identity())
        project = Project.query.get(project_id)

        if not project:
            return jsonify({"error": "Project not found"}), 404

        if project.user_id != current_user_id:
            return jsonify({"error": "Access denied"}), 403
            
        data = request.json
        axis = data.get('axis', 'x').lower()
        offset_percent = data.get('offset', 0)
        
        # Get latest successful mesh result
        result = MeshResult.query.filter_by(project_id=project_id, success=True).order_by(MeshResult.created_at.desc()).first()
        if not result or not result.output_path:
            return jsonify({"error": "No mesh available for this project"}), 404
            
        msh_path = Path(result.output_path)
        if not msh_path.exists():
            return jsonify({"error": "Mesh file not found on disk"}), 404
            
        # Quality file
        quality_path = msh_path.with_name(msh_path.stem + "_result.json")
        quality_map = {}
        if quality_path.exists():
            try:
                with open(quality_path, 'r') as f:
                    res_data = json.load(f)
                    quality_map = res_data.get('per_element_quality', {})
            except: pass
            
        # Parse mesh for volume elements
        print(f"[SLICE] Parsing mesh for slice: {msh_path.name}...")
        nodes, elements = parse_msh_for_slicing(msh_path)
        
        if not nodes or not elements:
            return jsonify({"error": "Could not parse volume elements from mesh"}), 400
            
        # Calculate bounds for plane positioning
        pts = np.array(list(nodes.values()))
        bbox_min = pts.min(axis=0)
        bbox_max = pts.max(axis=0)
        center = (bbox_min + bbox_max) / 2.0
        size = bbox_max - bbox_min
        
        # Define plane
        plane_origin = center.tolist()
        if axis == 'x':
            plane_normal = [1, 0, 0]
            plane_origin[0] = bbox_max[0] - (offset_percent / 100.0) * size[0]
        elif axis == 'y':
            plane_normal = [0, 1, 0]
            plane_origin[1] = bbox_max[1] - (offset_percent / 100.0) * size[1]
        else: # z
            plane_normal = [0, 0, 1]
            plane_origin[2] = bbox_max[2] - (offset_percent / 100.0) * size[2]
            
        # Generate slice
        print(f"[SLICE] Generating slice mesh on {axis}={offset_percent}%...")
        slice_data = generate_slice_mesh(nodes, elements, quality_map, plane_origin, plane_normal)
        
        return jsonify({
            "success": True,
            "axis": axis,
            "offset": offset_percent,
            "mesh": slice_data
        })

    @app.route('/api/projects/<project_id>/logs', methods=['GET'])
    @jwt_required()
    def get_project_logs(project_id: str):
        current_user_id = int(get_jwt_identity())
        project = Project.query.get(project_id)

        if not project:
            return jsonify({"error": "Project not found"}), 404

        if project.user_id != current_user_id:
            return jsonify({"error": "Access denied"}), 403
        
        latest_result = project.mesh_results.first()
        logs = latest_result.logs if latest_result and latest_result.logs else []
        
        return jsonify({"logs": logs})

    @app.route('/api/projects/<project_id>/download', methods=['GET'])
    @jwt_required()
    def download_mesh(project_id: str):
        current_user_id = int(get_jwt_identity())
        project = Project.query.get(project_id)

        if not project:
            return jsonify({"error": "Project not found"}), 404

        if project.user_id != current_user_id:
            return jsonify({"error": "Access denied"}), 403

        if project.status != "completed":
            return jsonify({"error": "Mesh not ready"}), 400

        latest_result = project.mesh_results.first()
        if not latest_result or not latest_result.output_path:
            return jsonify({"error": "Output file not found"}), 404

        output_path = latest_result.output_path
        storage = get_storage()
        use_s3 = app.config.get('USE_S3', False)
        
        # Check if file exists
        if use_s3 and output_path.startswith('s3://'):
            if not storage.file_exists(output_path):
                return jsonify({"error": "Output file not found"}), 404
            file_size = latest_result.output_size or 0
        else:
            local_path = Path(output_path)
            if not local_path.exists():
                return jsonify({"error": "Output file not found"}), 404
            file_size = local_path.stat().st_size

        # Track download
        download_record = DownloadRecord(
            project_id=project_id,
            mesh_result_id=latest_result.id,
            user_id=current_user_id,
            download_type='mesh',
            file_format='msh',
            file_size=file_size,
            ip_address=request.remote_addr,
            user_agent=request.user_agent.string[:500] if request.user_agent else None
        )
        
        # Update counters
        project.download_count = (project.download_count or 0) + 1
        project.last_accessed = datetime.utcnow()
        
        # Log activity
        activity = ActivityLog(
            user_id=current_user_id,
            action='download',
            resource_type='mesh_result',
            resource_id=str(latest_result.id),
            details={
                'project_id': project_id,
                'filename': project.filename,
                'file_size': file_size,
                'storage_path': output_path
            },
            ip_address=request.remote_addr,
            user_agent=request.user_agent.string[:500] if request.user_agent else None
        )
        
        try:
            db.session.add(download_record)
            db.session.add(activity)
            db.session.commit()
        except Exception:
            db.session.rollback()

        # For S3, return a presigned URL; for local, serve the file
        if use_s3 and output_path.startswith('s3://'):
            presigned_url = storage.get_file_url(output_path, expires_in=3600)
            return jsonify({
                "download_url": presigned_url,
                "filename": f"{Path(project.filename).stem}_mesh.msh",
                "expires_in": 3600
            })
        else:
            return send_file(
                output_path,
                as_attachment=True,
                download_name=f"{Path(project.filename).stem}_mesh.msh"
            )

    @app.route('/api/projects/<project_id>/mesh-data', methods=['GET'])
    @jwt_required()
    def get_mesh_data(project_id: str):
        import tempfile
        current_user_id = int(get_jwt_identity())
        project = Project.query.get(project_id)

        if not project:
            return jsonify({"error": "Project not found"}), 404

        if project.user_id != current_user_id:
            return jsonify({"error": "Access denied"}), 403
        
        if project.status != "completed":
            return jsonify({"error": "Mesh not ready"}), 400

        latest_result = project.mesh_results.first()
        if not latest_result or not latest_result.output_path:
            return jsonify({"error": "Mesh file not found"}), 404
        
        try:
            output_path = latest_result.output_path
            storage = get_storage()
            use_s3 = app.config.get('USE_S3', False)
            
            # If S3, download to temp file for parsing
            if use_s3 and output_path.startswith('s3://'):
                local_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.msh')
                storage.download_to_local(output_path, local_temp.name)
                mesh_data = parse_msh_file(local_temp.name)
                # Clean up temp file
                Path(local_temp.name).unlink(missing_ok=True)
            else:
                mesh_data = parse_msh_file(output_path)
            
            return jsonify(mesh_data)
        except Exception as e:
            return jsonify({"error": f"Failed to parse mesh: {str(e)}"}), 500

    @app.route('/api/projects/<project_id>/preview', methods=['GET'])
    @jwt_required()
    def get_cad_preview(project_id: str):
        """Get triangulated preview of CAD file before meshing"""
        import tempfile
        current_user_id = int(get_jwt_identity())
        project = Project.query.get(project_id)
        
        if not project:
            return jsonify({"error": "Project not found"}), 404
        
        if project.user_id != current_user_id:
            return jsonify({"error": "Access denied"}), 403
        
        if not project.filepath:
            return jsonify({"error": "CAD file not found"}), 404
        
        try:
            storage = get_storage()
            use_s3 = app.config.get('USE_S3', False)
            
            # 1. Check if we already have a cached preview
            if project.preview_path:
                try:
                    print(f"[PREVIEW] Using cached preview for project {project_id}: {project.preview_path}")
                    preview_bytes = storage.get_file(project.preview_path)
                    preview_data = json.loads(preview_bytes)
                    
                    # Ensure it has the geometry/volumes info 
                    # If it's an old version or corrupt, fallback
                    if "vertices" in preview_data:
                        return jsonify(preview_data)
                    print(f"[PREVIEW] Cached preview data incomplete, regenerating...")
                except Exception as e:
                    print(f"[PREVIEW] Failed to load cached preview: {e}")
            
            # 2. Fallback: Generate preview if not found (Double Hop - for legacy files)
            filepath = project.filepath
            
            # If S3, download to temp file for parsing
            if use_s3 and filepath.startswith('s3://'):
                file_ext = Path(project.filename).suffix
                local_temp = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
                storage.download_to_local(filepath, local_temp.name)
                preview_data = parse_step_file_for_preview(local_temp.name)
                # Clean up temp file
                Path(local_temp.name).unlink(missing_ok=True)
            else:
                if not Path(filepath).exists():
                    return jsonify({"error": "CAD file not found"}), 404
                preview_data = parse_step_file_for_preview(filepath)
            
            return jsonify(preview_data)
        except Exception as e:
            return jsonify({"error": f"Failed to generate preview: {str(e)}"}), 500
    
    @app.route('/api/projects', methods=['GET'])
    @jwt_required()
    def list_projects():
        current_user_id = int(get_jwt_identity())
        
        projects = Project.query.filter_by(user_id=current_user_id)\
            .order_by(Project.created_at.desc()).all()
        
        return jsonify({"projects": [p.to_dict() for p in projects]})
    
    @app.route('/api/projects/<project_id>', methods=['DELETE'])
    @jwt_required()
    def delete_project(project_id: str):
        current_user_id = int(get_jwt_identity())
        project = Project.query.get(project_id)
        
        if not project:
            return jsonify({"error": "Project not found"}), 404
        
        if project.user_id != current_user_id:
            return jsonify({"error": "Access denied"}), 403
        
        try:
            if project.filepath and Path(project.filepath).exists():
                Path(project.filepath).unlink()
            
            for result in project.mesh_results.all():
                if result.output_path and Path(result.output_path).exists():
                    Path(result.output_path).unlink()
            
            db.session.delete(project)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            return jsonify({"error": str(e)}), 500
        
        return jsonify({"message": "Project deleted"})

    @app.route('/api/user/stats', methods=['GET'])
    @jwt_required()
    def get_user_stats():
        """Get user usage statistics"""
        current_user_id = int(get_jwt_identity())
        user = User.query.get(current_user_id)
        
        if not user:
            return jsonify({"error": "User not found"}), 404
        
        # Get project stats
        total_projects = Project.query.filter_by(user_id=current_user_id).count()
        completed_projects = Project.query.filter_by(user_id=current_user_id, status='completed').count()
        
        # Get mesh stats
        total_meshes = db.session.query(MeshResult).join(Project).filter(Project.user_id == current_user_id).count()
        
        # Get download stats
        total_downloads = db.session.query(DownloadRecord).filter_by(user_id=current_user_id).count()
        
        # Get recent activity
        recent_activity = ActivityLog.query.filter_by(user_id=current_user_id)\
            .order_by(ActivityLog.created_at.desc()).limit(20).all()
        
        return jsonify({
            "user": user.to_dict(),
            "stats": {
                "total_projects": total_projects,
                "completed_projects": completed_projects,
                "total_meshes": total_meshes,
                "total_downloads": total_downloads,
                "storage_used": user.storage_used,
                "storage_quota": user.storage_quota,
                "storage_percent": round((user.storage_used or 0) / user.storage_quota * 100, 1) if user.storage_quota else 0
            },
            "recent_activity": [a.to_dict() for a in recent_activity]
        })

    @app.route('/api/user/activity', methods=['GET'])
    @jwt_required()
    def get_user_activity():
        """Get user activity log with pagination"""
        current_user_id = int(get_jwt_identity())
        
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        action_filter = request.args.get('action', None)
        
        query = ActivityLog.query.filter_by(user_id=current_user_id)
        
        if action_filter:
            query = query.filter_by(action=action_filter)
        
        activities = query.order_by(ActivityLog.created_at.desc())\
            .paginate(page=page, per_page=per_page, error_out=False)
        
        return jsonify({
            "activities": [a.to_dict() for a in activities.items],
            "total": activities.total,
            "pages": activities.pages,
            "current_page": page
        })


def parse_step_file_for_preview(step_filepath: str):
    """Convert STEP file to triangulated mesh for preview (matches desktop behavior)"""
    import subprocess
    import tempfile
    import json as json_lib
    
    print(f"[PREVIEW] Starting preview generation for: {step_filepath}")
    
    # ============================================================================
    # STEP 1: Try forwarding to Threadripper workstation via SSH tunnel
    # ============================================================================
    try:
        import requests
        print("[PREVIEW] Attempting to forward to Threadripper via SSH tunnel (localhost:8080)...")
        
        with open(step_filepath, 'rb') as f:
            # SSH tunnel maps localhost:8080 -> Threadripper:8080
            response = requests.post(
                "http://localhost:8080/mesh",
                files={'file': ('model.step', f, 'application/octet-stream')},
                timeout=120
            )
            
            if response.status_code == 200:
                print("[PREVIEW] ✓ Threadripper completed successfully - using remote result")
                result = response.json()
                # Convert Threadripper response format to our format if needed
                return result
            else:
                print(f"[PREVIEW] Threadripper returned error status {response.status_code}: {response.text[:200]}")
                
    except requests.exceptions.ConnectionError as e:
        print(f"[PREVIEW] Threadripper unavailable (connection refused): {e}")
    except requests.exceptions.Timeout as e:
        print(f"[PREVIEW] Threadripper request timed out after 120s: {e}")
    except Exception as e:
        print(f"[PREVIEW] Threadripper forwarding failed: {e}")
    
    print("[PREVIEW] Falling back to AWS local compute...")
    
    # ============================================================================
    # STEP 2: Fallback to AWS local GMSH computation
    # ============================================================================
    
    # Create temp file for STL output
    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp:
        tmp_stl = tmp.name
    
    # Use subprocess like desktop version to avoid gmsh state issues
    # More robust meshing with fallback options for complex geometries
    # Use raw string and manual replacement for better reliability with braces
    gmsh_script = r'''
import gmsh
import sys
import os
import json

def apply_robust_settings():
    """Apply standard robust settings to Gmsh"""
    # CRITICAL: Disable OCC auto-fix early to prevent "Could not fix wire" crashes
    gmsh.option.setNumber("Geometry.OCCAutoFix", 0)
    gmsh.option.setNumber("Geometry.Tolerance", 1e-2)
    gmsh.option.setNumber("Geometry.ToleranceBoolean", 1e-1)
    
    # DISABLE OpenMP for preview generation (often the source of SIGSEGV during meshing)
    gmsh.option.setNumber("General.NumThreads", 1)
    
    # Fast rendering - disable perfectionism for preview speed
    gmsh.option.setNumber("Mesh.Optimize", 0)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)
    gmsh.option.setNumber("Mesh.MaxRetries", 1)
    gmsh.option.setNumber("Mesh.RecombineAll", 0)
    gmsh.option.setNumber("Mesh.Smoothing", 0)
    gmsh.option.setNumber("Mesh.StlRemoveDuplicateTriangles", 1)

def try_mesh(mesh_factor, algorithm=1, tolerance=1e-3):
    """Try to generate mesh with given parameters"""
    gmsh.model.mesh.clear()
    
    # Get bounding box for sizing
    bbox = gmsh.model.getBoundingBox(-1, -1)
    bbox_size = [bbox[3]-bbox[0], bbox[4]-bbox[1], bbox[5]-bbox[2]]
    diag = (bbox_size[0]**2 + bbox_size[1]**2 + bbox_size[2]**2)**0.5
    
    # Mesh sizing
    mesh_size = diag / mesh_factor
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size * 5)
    
    # Apply local tolerance and algorithm
    gmsh.option.setNumber("Mesh.Algorithm", algorithm)
    gmsh.option.setNumber("Geometry.Tolerance", tolerance)
    gmsh.option.setNumber("Mesh.ToleranceInitialDelaunay", tolerance)
    
    # Extra robustness
    gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", 0.9)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 4)
    
    # Re-enforce single-thread and no-autofix right before generation
    gmsh.option.setNumber("General.NumThreads", 1)
    gmsh.option.setNumber("Geometry.OCCAutoFix", 0)
    
    # Generate 2D surface mesh
    gmsh.model.mesh.generate(2)
    return True

try:
    step_filepath = r"__CAD_FILE__"
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("General.Verbosity", 3)
    
    apply_robust_settings()

    print(f"[PREVIEW] Loading CAD file: {step_filepath}")
    try:
        # Ultra-lenient tolerance for loading
        gmsh.option.setNumber("Geometry.Tolerance", 1.0)
        gmsh.model.occ.importShapes(step_filepath)
        gmsh.model.occ.synchronize()
    except Exception as e:
        print(f"[PREVIEW] Initial open failed ({e}), attempting fallback...")
        try:
            gmsh.merge(step_filepath)
            gmsh.model.occ.synchronize()
        except Exception as e2:
            print(f"[PREVIEW] all load attempts failed: {e2}")
    
    # Check if we actually loaded anything
    volumes = gmsh.model.getEntities(3)
    surfaces = gmsh.model.getEntities(2)
    if not volumes and not surfaces:
        print("[PREVIEW] ERROR: No geometry loaded.", file=sys.stderr)
        sys.exit(1)
    
    print(f"[PREVIEW] CAD loaded. Volumes: {len(volumes)}, Surfaces: {len(surfaces)}")
    
    # Extract CAD info
    curves = gmsh.model.getEntities(1)
    points = gmsh.model.getEntities(0)
    bbox = gmsh.model.getBoundingBox(-1, -1)
    bbox_size = [bbox[3]-bbox[0], bbox[4]-bbox[1], bbox[5]-bbox[2]]
    diag = (bbox_size[0]**2 + bbox_size[1]**2 + bbox_size[2]**2)**0.5
    
    # Calculate volume
    total_volume = 0.0
    for dim, tag in volumes:
        try: total_volume += gmsh.model.occ.getMass(dim, tag)
        except: pass
    
    # Try meshing attempts - progress from simple to complex but always single-threaded
    mesh_attempts = [
        (15, 1, 1e-2), (10, 1, 1e-2), (5, 1, 1e-1), (3, 1, 1.0)
    ]
    
    mesh_success = False
    for i, (mesh_factor, algo, tol) in enumerate(mesh_attempts):
        try:
            print(f"[PREVIEW] Attempt {i+1}/{len(mesh_attempts)}: sizing=diag/{mesh_factor}, algo={algo}, tolerance={tol}")
            try_mesh(mesh_factor, algorithm=algo, tolerance=tol)
            mesh_success = True
            print(f"MESH_OK:factor={mesh_factor},algo={algo}", file=sys.stderr)
            break
        except Exception as mesh_err:
            print(f"[PREVIEW] Attempt {i+1} failed ({mesh_err})...")
            continue
    
    if not mesh_success:
        print(f"[PREVIEW] All meshing attempts failed. Falling back to Bounding Box...", file=sys.stderr)
        try:
            gmsh.model.mesh.clear()
            # Restore stable settings for fallback
            apply_robust_settings()
            gmsh.model.add("BBox_Fallback")
            
            # Simple box
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
            dx, dy, dz = max(1e-3, xmax-xmin), max(1e-3, ymax-ymin), max(1e-3, zmax-zmin)
            gmsh.model.occ.addBox(xmin, ymin, zmin, dx, dy, dz)
            gmsh.model.occ.synchronize()
            
            # Mesh the box coarsely
            diag = (dx**2 + dy**2 + dz**2)**0.5
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", diag / 10.0)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", diag / 5.0)
            gmsh.model.mesh.generate(2)
            mesh_success = True
            print("MESH_OK:strategy=bounding_box", file=sys.stderr)
        except Exception as bbox_err:
            print(f"[PREVIEW] Bounding Box fallback also failed: {bbox_err}", file=sys.stderr)
            raise Exception("All meshing attempts including Bounding Box fallback failed")
    
    # Get triangles
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    nodes = {}
    for i, tag in enumerate(node_tags):
        nodes[int(tag)] = [node_coords[3*i], node_coords[3*i+1], node_coords[3*i+2]]
    
    vertices = []
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2)
    for etype, etags, enodes in zip(elem_types, elem_tags, elem_node_tags):
        if etype == 2:  # 3-node triangle
            enodes = list(enodes)
            for i in range(0, len(enodes), 3):
                n1, n2, n3 = int(enodes[i]), int(enodes[i+1]), int(enodes[i+2])
                if n1 in nodes and n2 in nodes and n3 in nodes:
                    vertices.extend(nodes[n1])
                    vertices.extend(nodes[n2])
                    vertices.extend(nodes[n3])
    
    gmsh.finalize()
    result = {
        "vertices": vertices, 
        "count": len(vertices)//9,
        "geometry": {
            "volumes": len(volumes), "surfaces": len(surfaces), "curves": len(curves),
            "points": len(points), "bbox": bbox_size, "bbox_diagonal": diag,
            "total_volume": total_volume
        }
    }
    print("MESH_DATA:" + json.dumps(result))
    
except Exception as e:
    print("ERROR:" + str(e))
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''.replace("__CAD_FILE__", step_filepath.replace("\\", "/"))
    
    try:
        # Stream output in real-time so diagnostics appear immediately in logs
        import time
        process = subprocess.Popen(
            [sys.executable, '-u', '-c', gmsh_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # Line buffered
        )
        
        print("[PREVIEW] Subprocess started, streaming output...")
        
        # Stream output line by line
        stdout_lines = []
        start_time = time.time()
        
        while True:
            # Check for timeout
            if time.time() - start_time > 600:
                process.kill()
                raise subprocess.TimeoutExpired(process.args, 600)
            
            # Read line (with timeout)
            line = process.stdout.readline()
            
            if not line and process.poll() is not None:
                break  # Process finished
                
            if line:
                line = line.strip()
                stdout_lines.append(line)
                # Print immediately to logs
                print(f"[GMSH-SUBPROCESS] {line}", flush=True)
        
        process.wait()
        stdout = '\n'.join(stdout_lines)
        
        print(f"[PREVIEW] Subprocess finished with return code: {process.returncode}")
        
        # Parse output
        for line in stdout.split('\n'):
            if line.startswith('MESH_DATA:'):
                data = json_lib.loads(line[10:])
                vertices = data['vertices']
                num_triangles = data['count']
                geometry = data.get('geometry', {})

                # Default blue color for preview
                colors = [0.3, 0.6, 0.9] * (len(vertices) // 3)

                print(f"[PREVIEW] Success: {len(vertices)//3} vertices, {num_triangles} triangles")
                if geometry:
                    print(f"[PREVIEW] Geometry: {geometry.get('volumes')} volumes, {geometry.get('surfaces')} surfaces, {geometry.get('curves')} curves")

                return {
                    "vertices": vertices,
                    "colors": colors,
                    "numVertices": len(vertices) // 3,
                    "numTriangles": num_triangles,
                    "isPreview": True,
                    "geometry": geometry
                }
            elif line.startswith('ERROR:'):
                raise Exception(line[6:])
        
        # Check for all failed message
        if 'All meshing attempts failed' in stdout:
            raise Exception("Could not mesh this geometry - it may be too complex or have invalid surfaces")
        
        raise Exception(f"No mesh data in output: {stdout[:200] if stdout else 'empty'}")
        
    except subprocess.TimeoutExpired:
        print("[PREVIEW] Subprocess timed out after 600 seconds")
        return {"error": "Preview generation timed out - geometry is too complex", "vertices": [], "colors": [], "numVertices": 0, "numTriangles": 0, "isPreview": True}
    except Exception as e:
        import traceback
        print(f"[PREVIEW ERROR] {e}")
        traceback.print_exc()
        return {"error": str(e), "vertices": [], "colors": [], "numVertices": 0, "numTriangles": 0}
    finally:
        # Cleanup temp file
        try:
            Path(tmp_stl).unlink()
        except:
            pass


def parse_msh_file(msh_filepath: str):
    """
    Parse Gmsh .msh file for Three.js visualization using native Python parsing.
    This replaces the GMSH API subprocess to avoid "mesh soup" issues with assemblies
    (non-contiguous node IDs) and improves stability.
    """
    import json
    import sys
    from collections import defaultdict
    
    print(f"[MESH PARSE] Parsing mesh file (native): {msh_filepath}")
    
    # Load quality data if available
    # The worker saves detailed metrics to _result.json, while legacy saves to .quality.json
    quality_filepath = Path(msh_filepath).with_suffix('.quality.json')
    result_filepath = Path(str(msh_filepath).replace('.msh', '_result.json'))
    
    per_element_quality = {}
    per_element_gamma = {}
    per_element_skewness = {}
    per_element_aspect_ratio = {}
    per_element_min_angle = {}
    
    data_file = None
    if result_filepath.exists():
        data_file = result_filepath
    elif quality_filepath.exists():
        data_file = quality_filepath
        
    if data_file:
        try:
            with open(data_file, 'r') as f:
                qdata = json.load(f)
                # Metrics might be nested under quality_metrics or at root
                metrics_container = qdata.get('quality_metrics', qdata)
                
                per_element_quality = {int(k): v for k, v in metrics_container.get('per_element_quality', {}).items()}
                per_element_gamma = {int(k): v for k, v in metrics_container.get('per_element_gamma', {}).items()}
                per_element_skewness = {int(k): v for k, v in metrics_container.get('per_element_skewness', {}).items()}
                per_element_aspect_ratio = {int(k): v for k, v in metrics_container.get('per_element_aspect_ratio', {}).items()}
                per_element_min_angle = {int(k): v for k, v in metrics_container.get('per_element_min_angle', {}).items()}
                print(f"[MESH PARSE] Loaded quality data from {data_file.name} for {len(per_element_quality)} elements")
        except Exception as e:
            print(f"[MESH PARSE] Failed to load quality data from {data_file}: {e}")

    # Unified Vivid Color Scale used by all parsing paths
    def get_color(q, metric='sicn'):
        if q is None:
            return 0.29, 0.56, 0.89  # Blue default (Quality missing)
        
        val = q
        
        if metric == 'sicn':
            if val < 0: return 0.9, 0.1, 0.1 # VIVID RED for inverted!
            # Sharper transition for SICN [0..1]
            if val < 0.3:
                # 0..0.3 maps Red -> Yellow
                t = val / 0.3
                return 0.9, 0.1 + 0.8 * t, 0.1
            else:
                # 0.3..1.0 maps Yellow -> Green
                t = (val - 0.3) / 0.7
                return 0.9 - 0.8 * t, 0.9, 0.1
        elif metric == 'skewness':
            val = max(0.0, min(1.0, 1.0 - q))
        elif metric == 'aspect_ratio':
            val = max(0.0, min(1.0, 1.0 - (q - 1.0) / 4.0))
        elif metric == 'minAngle':
            val = max(0.0, min(1.0, q / 60.0))
            
        t = max(0.0, min(1.0, val))
        if t < 0.5:
            return 1.0, t * 2, 0.0
        else:
            return 1.0 - (t - 0.5) * 2, 1.0, 0.0

    def get_q_value(tag, data_dict):
        if not data_dict: return None
        try:
            v = data_dict.get(int(tag))
            if v is None: v = data_dict.get(str(tag))
            return v
        except: return None

    try:
        # Try to detect format
        with open(msh_filepath, 'rb') as f:
            header = f.read(100)
        
        is_binary = b'\x00' in header[:50]  # Binary files have null bytes in header
        
        # For binary MSH files, use meshio (simpler and more robust)
        if is_binary and meshio is not None:
            print(f"[MESH PARSE] Binary MSH file detected, using meshio")
            try:
                mesh = meshio.read(msh_filepath)
                
                # Get nodes
                nodes = {}
                node_id_to_index = {}
                for i, point in enumerate(mesh.points):
                    nodes[i+1] = point.tolist()
                    node_id_to_index[i+1] = i
                
                print(f"[MESH PARSE] Loaded {len(nodes)} nodes via meshio")
                
                # Process cells to extract boundary faces
                face_map = {}
                
                def add_face(face_nodes, el_tag, entity_tag=0):
                    key = tuple(sorted(face_nodes))
                    if key not in face_map:
                        face_map[key] = {'nodes': face_nodes, 'count': 0, 'element_tag': el_tag, 'entity_tag': entity_tag}
                    face_map[key]['count'] += 1
                
                el_tag_seq = 0
                for cell_block_idx, cell_block in enumerate(mesh.cells):
                    cell_type = cell_block.type
                    cells = cell_block.data
                    
                    # Try to find real Gmsh IDs in cell_data
                    real_ids = None
                    if hasattr(mesh, 'cell_data') and 'gmsh:element_id' in mesh.cell_data:
                        try:
                            # gmsh:element_id is a list of arrays (one per cell block)
                            real_ids = mesh.cell_data['gmsh:element_id'][cell_block_idx]
                            print(f"[MESH PARSE] Found {len(real_ids)} real Gmsh IDs for block {cell_block_idx}")
                        except Exception as e:
                            print(f"[MESH PARSE] Warning: Could not get Gmsh IDs for block {cell_block_idx}: {e}")
                    
                    def get_elem_id(i):
                        if real_ids is not None and i < len(real_ids):
                            return int(real_ids[i])
                        return el_tag_seq + i + 1  # Fallback to 1-indexed sequential
                        
                    if cell_type == 'tetra':  # Tet4
                        for i, cell in enumerate(cells):
                            tag = get_elem_id(i)
                            n = [node_id_to_index[int(nid)+1] for nid in cell[:4]]
                            add_face((n[0], n[2], n[1]), tag)
                            add_face((n[0], n[1], n[3]), tag)
                            add_face((n[0], n[3], n[2]), tag)
                            add_face((n[1], n[2], n[3]), tag)
                    elif cell_type == 'hexahedron':  # Hex8
                        for i, cell in enumerate(cells):
                            tag = get_elem_id(i)
                            n = [node_id_to_index[int(nid)+1] for nid in cell[:8]]
                            quads = [
                                (n[0], n[3], n[2], n[1]), (n[4], n[5], n[6], n[7]),
                                (n[0], n[1], n[5], n[4]), (n[2], n[3], n[7], n[6]),
                                (n[1], n[2], n[6], n[5]), (n[4], n[7], n[3], n[0])
                            ]
                            for q in quads:
                                add_face((q[0], q[1], q[2]), tag)
                                add_face((q[0], q[2], q[3]), tag)
                    elif cell_type == 'triangle':  # Surface triangle
                        for i, cell in enumerate(cells):
                            tag = get_elem_id(i)
                            n = [node_id_to_index[int(nid)+1] for nid in cell[:3]]
                            key = tuple(sorted(n))
                            face_map[key] = {'nodes': n, 'count': 1, 'element_tag': tag, 'entity_tag': 0, 'is_surface': True}
                    
                    el_tag_seq += len(cells)
                
                print(f"[MESH PARSE] Processed {len(face_map)} unique faces")
                
                # Build output from meshio data
                indexed_nodes = [None] * len(nodes)
                for nid, idx in node_id_to_index.items():
                    indexed_nodes[idx] = nodes[nid]
                
                vertices = []
                element_tags = []
                entity_tags = []
                
                for key, data in face_map.items():
                    if data.get('is_surface') or data['count'] == 1:  # Boundary face or explicit surface
                        face_nodes = data['nodes']
                        for idx in face_nodes:
                            vertices.extend(indexed_nodes[idx])
                        element_tags.append(data['element_tag'])
                        entity_tags.append(data['entity_tag'])
                
                print(f"[MESH PARSE] Extracted {len(element_tags)} boundary triangles")
                
                # Apply quality colors using the loaded quality data
                colors_sicn = []
                colors_gamma = []
                colors_skewness = []
                colors_aspect_ratio = []
                colors_min_angle = []
                
                for el_tag in element_tags:
                    tag_key = el_tag # Could be int or str
                    q_sicn = get_q_value(tag_key, per_element_quality)
                    q_gamma = get_q_value(tag_key, per_element_gamma)
                    q_skew = get_q_value(tag_key, per_element_skewness)
                    q_ar = get_q_value(tag_key, per_element_aspect_ratio)
                    q_ang = get_q_value(tag_key, per_element_min_angle)
                    
                    colors_sicn.extend(get_color(q_sicn, 'sicn') * 3)
                    colors_gamma.extend(get_color(q_gamma, 'gamma') * 3)
                    colors_skewness.extend(get_color(q_skew, 'skewness') * 3)
                    colors_aspect_ratio.extend(get_color(q_ar, 'aspect_ratio') * 3)
                    colors_min_angle.extend(get_color(q_ang, 'minAngle') * 3)
                
                # Build quality summary
                quality_values = [per_element_quality.get(int(t)) for t in element_tags if per_element_quality.get(int(t)) is not None]
                quality_summary = {}
                histogram_data = []
                if quality_values:
                    quality_summary = {
                        "min": min(quality_values),
                        "max": max(quality_values),
                        "avg": sum(quality_values) / len(quality_values),
                        "count": len(quality_values)
                    }
                
                return {
                    "vertices": vertices,
                    "element_tags": element_tags,
                    "entity_tags": entity_tags,
                    "num_nodes": len(nodes),
                    "colors": colors_sicn,
                    "qualityColors": {
                        "sicn": colors_sicn,
                        "gamma": colors_gamma,
                        "skewness": colors_skewness,
                        "aspect_ratio": colors_aspect_ratio,
                        "min_angle": colors_min_angle
                    },
                    "qualityMetrics": quality_summary,
                    "histogramData": histogram_data,
                    "hasQualityData": bool(per_element_quality)
                }
                
            except Exception as e:
                print(f"[MESH PARSE] meshio failed: {e}, falling back to text parsing")
                import traceback
                traceback.print_exc()
                is_binary = False
        
        # For ASCII files or if meshio failed, use native parsing
        if not is_binary:
            with open(msh_filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

        # --- Detect Version ---
        msh_version = '2.2' # Default
        if not is_binary and '$MeshFormat' in content:
            try:
                format_section = content.split('$MeshFormat')[1].split('$EndMeshFormat')[0].strip().split('\n')
                if format_section:
                    msh_version = format_section[0].split()[0]
                    print(f"[MESH PARSE] MSH version: {msh_version}")
            except: pass

        # --- Parse Nodes ---
        if '$Nodes' not in content:
            return {"error": "Invalid MSH file: No $Nodes section"}
            
        nodes_section = content.split('$Nodes')[1].split('$EndNodes')[0].strip().split('\n')
        
        nodes = {}  # node_tag -> [x, y, z]
        node_id_to_index = {}  # node_tag -> sequential_index
        
        # Detect format based on first line structure
        header = nodes_section[0].split()
        
        if msh_version.startswith('4'):
            # MSH 4.x format: numEntityBlocks numNodes minNodeTag maxNodeTag
            num_blocks = int(header[0])
            total_nodes = int(header[1])
            
            curr_line = 1
            for _ in range(num_blocks):
                block_header = nodes_section[curr_line].split()
                curr_line += 1
                num_nodes_in_block = int(block_header[3])
                
                node_tags = []
                for i in range(num_nodes_in_block):
                    tag = int(nodes_section[curr_line + i])
                    node_tags.append(tag)
                curr_line += num_nodes_in_block
                
                for i in range(num_nodes_in_block):
                    coords = list(map(float, nodes_section[curr_line + i].split()))
                    node_tag = node_tags[i]
                    nodes[node_tag] = coords[:3]
                    node_id_to_index[node_tag] = len(node_id_to_index)
                curr_line += num_nodes_in_block
        else:
            # MSH 2.x format: numNodes on first line, then "id x y z" per line
            total_nodes = int(header[0])
            
            for i in range(1, total_nodes + 1):
                parts = nodes_section[i].split()
                node_tag = int(parts[0])
                coords = [float(parts[1]), float(parts[2]), float(parts[3])]
                nodes[node_tag] = coords
                node_id_to_index[node_tag] = len(node_id_to_index)

        print(f"[MESH PARSE] Parsed {len(nodes)} nodes")

        # --- Parse Elements ---
        if '$Elements' not in content:
            return {"error": "Invalid MSH file: No $Elements section"}
            
        elements_section = content.split('$Elements')[1].split('$EndElements')[0].strip().split('\n')
        header = elements_section[0].split()
        
        # Element types mapping (same for MSH 2.x and 4.x)
        # 2: 3-node triangle, 4: 4-node tet, 11: 10-node tet, 5: 8-node hex, 12: 27-node hex
        
        vertices = []
        element_tags = []
        entity_tags = []
        
        colors_sicn = []
        colors_gamma = []
        colors_skewness = []
        colors_aspect_ratio = []
        colors_min_angle = []
        
        face_map = {}
        
        def process_element(el_tag, el_type, node_ids, entity_tag=0):
            """Process a single element and add faces to face_map"""
            if el_type in [4, 11]:  # Tetrahedra
                try:
                    n = [node_id_to_index[nid] for nid in node_ids[:4]]
                    faces = [(n[0], n[2], n[1]), (n[0], n[1], n[3]), (n[0], n[3], n[2]), (n[1], n[2], n[3])]
                    for face in faces:
                        key = tuple(sorted(face))
                        if key not in face_map:
                            face_map[key] = {'nodes': face, 'count': 0, 'element_tag': el_tag, 'entity_tag': entity_tag}
                        face_map[key]['count'] += 1
                except KeyError: pass
            elif el_type in [5, 12]:  # Hexahedra
                try:
                    n = [node_id_to_index[nid] for nid in node_ids[:8]]
                    qs = [(n[0], n[3], n[2], n[1]), (n[4], n[5], n[6], n[7]),
                          (n[0], n[1], n[5], n[4]), (n[2], n[3], n[7], n[6]),
                          (n[1], n[2], n[6], n[5]), (n[4], n[7], n[3], n[0])]
                    for q in qs:
                        for tri in [(q[0], q[1], q[2]), (q[0], q[2], q[3])]:
                            key = tuple(sorted(tri))
                            if key not in face_map:
                                face_map[key] = {'nodes': tri, 'count': 0, 'element_tag': el_tag, 'entity_tag': entity_tag}
                            face_map[key]['count'] += 1
                except KeyError: pass
            elif el_type in [2, 9]:  # Triangles (surface mesh)
                try:
                    # Assembly logic: Triangles are often redundant or internal.
                    # We only include them if they are NOT clearly part of a volume-only assembly.
                    # For now, let's treat them as surfaces but don't force 'is_surface' if we have volumes.
                    n = [node_id_to_index[nid] for nid in node_ids[:3]]
                    key = tuple(sorted(n))
                    if key not in face_map:
                        face_map[key] = {'nodes': n, 'count': 0, 'element_tag': el_tag, 'entity_tag': entity_tag, 'is_surface': True}
                    face_map[key]['is_surface'] = True
                except KeyError: pass

        if msh_version.startswith('4'):
            # MSH 4.x: numEntityBlocks numElements minTag maxTag
            num_blocks = int(header[0])
            curr_line = 1
            for _ in range(num_blocks):
                line_parts = elements_section[curr_line].split()
                curr_line += 1
                if not line_parts: continue
                entity_tag_block = int(line_parts[1])
                el_type = int(line_parts[2])
                num_els = int(line_parts[3])
                for i in range(num_els):
                    el_line = list(map(int, elements_section[curr_line + i].split()))
                    process_element(el_line[0], el_type, el_line[1:], entity_tag_block)
                curr_line += num_els
        else:
            # MSH 2.x: numElements on first line, then "id type numTags tags... nodes..."
            num_elements = int(header[0])
            for i in range(1, num_elements + 1):
                parts = elements_section[i].split()
                el_tag = int(parts[0])
                el_type = int(parts[1])
                num_tags = int(parts[2])
                # Skip tags, node IDs start after tags
                node_ids = [int(x) for x in parts[3 + num_tags:]]
                entity_tag = int(parts[3]) if num_tags > 0 else 0
                process_element(el_tag, el_type, node_ids, entity_tag)
        print(f"[MESH PARSE] Processed faces, extracting boundary...")
        
        # Reconstruct nodes list from dict for fast access by index
        # We need this because node_id_to_index maps ID -> 0..N
        # We need 0..N -> [x,y,z]
        indexed_nodes = [None] * len(nodes)
        for nid, idx in node_id_to_index.items():
            indexed_nodes[idx] = nodes[nid]
        
        # DEBUG: Sample element tags to compare with quality data
        sample_face_tags = [d['element_tag'] for d in list(face_map.values())[:10]]
        sample_quality_keys = list(per_element_quality.keys())[:10] if per_element_quality else []
        print(f"[MESH PARSE DEBUG] Sample face element_tags: {sample_face_tags}")
        print(f"[MESH PARSE DEBUG] Sample quality dict keys: {sample_quality_keys}")
        print(f"[MESH PARSE DEBUG] Total faces in face_map: {len(face_map)}")
        print(f"[MESH PARSE DEBUG] Quality dict has {len(per_element_quality)} entries")
        
        boundary_face_count = 0
        faces_with_quality = 0
        faces_without_quality = 0
            
        # Extract boundary faces (count == 1)
        for key, data in face_map.items():
            if data['count'] == 1:
                boundary_face_count += 1
                # This is a boundary face
                face_nodes = data['nodes']
                el_tag = data['element_tag']
                ent_tag = data['entity_tag']
                
                # Update buffers
                for idx in face_nodes:
                    vertices.extend(indexed_nodes[idx])
                
                element_tags.append(el_tag)
                entity_tags.append(ent_tag)
                
                # Colors
                q_sicn = get_q_value(el_tag, per_element_quality)
                if q_sicn is not None:
                    faces_with_quality += 1
                else:
                    faces_without_quality += 1
                colors_sicn.extend(get_color(q_sicn, 'sicn') * 3)
                
                q_gamma = get_q_value(el_tag, per_element_gamma)
                colors_gamma.extend(get_color(q_gamma, 'gamma') * 3)
                
                q_skew = get_q_value(el_tag, per_element_skewness)
                colors_skewness.extend(get_color(q_skew, 'skewness') * 3)
                
                q_ar = get_q_value(el_tag, per_element_aspect_ratio)
                colors_aspect_ratio.extend(get_color(q_ar, 'aspect_ratio') * 3)
                
                q_ang = get_q_value(el_tag, per_element_min_angle)
                colors_min_angle.extend(get_color(q_ang, 'minAngle') * 3)

        print(f"[MESH PARSE DEBUG] Boundary faces extracted: {boundary_face_count}")
        print(f"[MESH PARSE DEBUG] Faces WITH quality data: {faces_with_quality}")
        print(f"[MESH PARSE DEBUG] Faces WITHOUT quality data: {faces_without_quality}")
        print(f"[MESH PARSE] Generated {len(vertices)//3} vertices")
        
        # Calculate quality summary and histogram from per_element_quality
        quality_values = list(per_element_quality.values()) if per_element_quality else []
        quality_summary = None
        histogram_data = None
        
        if quality_values:
            # Summary statistics
            quality_summary = {
                "sicn_min": min(quality_values),
                "sicn_max": max(quality_values),
                "sicn_avg": sum(quality_values) / len(quality_values),
                "total_elements": len(quality_values),
                "poor_elements": sum(1 for q in quality_values if q < 0.1),
            }
            
            # Load additional metrics from quality.json if available
            try:
                if quality_filepath.exists():
                    with open(quality_filepath, 'r') as f:
                        qdata = json.load(f)
                        qmetrics = qdata.get('quality_metrics', {})
                        # Add gamma, skewness, aspect ratio if available
                        if 'gamma_min' in qmetrics:
                            quality_summary['gamma_min'] = qmetrics['gamma_min']
                            quality_summary['gamma_avg'] = qmetrics['gamma_avg']
                            quality_summary['gamma_max'] = qmetrics['gamma_max']
                        if 'skewness_min' in qmetrics:
                            quality_summary['skewness_min'] = qmetrics['skewness_min']
                            quality_summary['skewness_avg'] = qmetrics['skewness_avg']
                            quality_summary['skewness_max'] = qmetrics['skewness_max']
                        if 'aspect_ratio_min' in qmetrics:
                            quality_summary['aspect_ratio_min'] = qmetrics['aspect_ratio_min']
                            quality_summary['aspect_ratio_avg'] = qmetrics['aspect_ratio_avg']
                            quality_summary['aspect_ratio_max'] = qmetrics['aspect_ratio_max']
            except Exception as e:
                print(f"[MESH PARSE] Warning: Could not load additional metrics: {e}")
            
            # Compute histogram bins (10 bins from 0.0 to 1.0)
            num_bins = 10
            histogram_bins = [0] * num_bins
            for q in quality_values:
                bin_idx = min(int(q * num_bins), num_bins - 1)
                if bin_idx >= 0:
                    histogram_bins[bin_idx] += 1
            
            max_count = max(histogram_bins) if histogram_bins else 1
            histogram_data = {
                "bins": [
                    {
                        "rangeStart": i / num_bins,
                        "rangeEnd": (i + 1) / num_bins,
                        "count": histogram_bins[i],
                        "percentage": round(histogram_bins[i] / len(quality_values) * 100, 1) if quality_values else 0,
                        "normalized": histogram_bins[i] / max_count if max_count > 0 else 0,
                    }
                    for i in range(num_bins)
                ],
                "totalElements": len(quality_values),
                "maxCount": max_count,
            }
            print(f"[MESH PARSE] Histogram computed: {histogram_bins}")

        mesh_data = {
            "vertices": vertices,
            "element_tags": element_tags,
            "entity_tags": entity_tags,
            "num_nodes": len(nodes),
            "colors": colors_sicn, # Default color set
            "qualityColors": {
                "sicn": colors_sicn,
                "gamma": colors_gamma,
                "skewness": colors_skewness,
                "aspect_ratio": colors_aspect_ratio,
                "min_angle": colors_min_angle
            },
            "qualityMetrics": quality_summary,
            "histogramData": histogram_data,
            "hasQualityData": bool(per_element_quality)
        }
        
        return mesh_data

    except Exception as e:
        print(f"[MESH PARSE ERROR] {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "vertices": [], "colors": []}


app = create_app()

if __name__ == '__main__':
    print("=" * 70)
    print("MESH GENERATION API SERVER v2.0")
    print("With JWT Authentication & SQLAlchemy Database")
    print("=" * 70)
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Output folder: {app.config['OUTPUT_FOLDER']}")
    print(f"JWT_SECRET_KEY: {'[SET]' if app.config.get('JWT_SECRET_KEY') else '[NOT SET - USING DEFAULT]'}")
    print("\nStarting server on http://localhost:5000")
    print("=" * 70)

    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
