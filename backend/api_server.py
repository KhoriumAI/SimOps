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
                                mesh_result.quality_metrics = result.get('quality_metrics') # Fix: use correct key
                                mesh_result.completed_at = datetime.utcnow()
                                mesh_result.processing_time = time.time() - start_time
                                mesh_result.node_count = result.get('total_nodes')
                                mesh_result.element_count = result.get('total_elements')
                                
                                # Get output file size
                                if local_output_path and Path(local_output_path).exists():
                                    mesh_result.output_size = Path(local_output_path).stat().st_size
                                    
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
            # Try to clean up the uploaded files
            try:
                storage.delete_file(filepath)
                if preview_path:
                    storage.delete_file(preview_path)
            except:
                pass
            return jsonify({"error": "Failed to create project"}), 500

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
            plane_origin[0] = bbox_min[0] + (offset_percent + 50) / 100.0 * size[0]
        elif axis == 'y':
            plane_normal = [0, 1, 0]
            plane_origin[1] = bbox_min[1] + (offset_percent + 50) / 100.0 * size[1]
        else: # z
            plane_normal = [0, 0, 1]
            plane_origin[2] = bbox_min[2] + (offset_percent + 50) / 100.0 * size[2]
            
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
    gmsh_script = f'''
import gmsh
import sys
import os

def try_mesh(mesh_factor, algorithm=5, tolerance=1e-3):
    """Try to generate mesh with given parameters"""
    gmsh.model.mesh.clear()
    
    # Get bounding box for sizing
    bbox = gmsh.model.getBoundingBox(-1, -1)
    bbox_size = [bbox[3]-bbox[0], bbox[4]-bbox[1], bbox[5]-bbox[2]]
    diag = (bbox_size[0]**2 + bbox_size[1]**2 + bbox_size[2]**2)**0.5
    
    # Mesh sizing - larger factor = coarser mesh
    mesh_size = diag / mesh_factor
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size * 5)
    
    # Algorithm: 1=MeshAdapt, 5=Delaunay, 6=Frontal-Delaunay, 7=BAMG
    gmsh.option.setNumber("Mesh.Algorithm", algorithm)
    
    # Very tolerant settings for complex geometry
    gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", 0.9)
    gmsh.option.setNumber("Mesh.AllowSwapAngle", 120)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 4)
    gmsh.option.setNumber("Mesh.MinimumCircleNodes", 3)
    gmsh.option.setNumber("Mesh.ToleranceInitialDelaunay", tolerance)
    
    # Increase tolerance for difficult geometries
    gmsh.option.setNumber("Geometry.Tolerance", tolerance)
    gmsh.option.setNumber("Geometry.ToleranceBoolean", tolerance * 10)
    
    # Generate 2D surface mesh
    gmsh.model.mesh.generate(2)
    # Force GMSH to skip the 'stuck' healing process
    gmsh.option.setNumber("Geometry.OCCAutoFix", 0) 
    gmsh.option.setNumber("Geometry.Tolerance", 1e-3)
    gmsh.option.setNumber("Mesh.Algorithm", 1) # MeshAdapt is more stable for dirty CAD
    
    return True

try:
    print(f"[PREVIEW] Initializing Gmsh for CAD: {step_filepath}")
    gmsh.initialize()
    # Increase verbosity for debugging
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("General.Verbosity", 3)
    
    # Aggressive OCC healing options for problematic CAD files
    gmsh.option.setNumber("Geometry.OCCFixDegenerated", 1)
    gmsh.option.setNumber("Geometry.OCCFixSmallEdges", 1)
    gmsh.option.setNumber("Geometry.OCCFixSmallFaces", 1)
    gmsh.option.setNumber("Geometry.OCCSewFaces", 1)
    gmsh.option.setNumber("Geometry.OCCMakeSolids", 1)
    
    # Dirty Geometry Workaround
    gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)
    gmsh.option.setNumber("Geometry.Tolerance", 1e-3) # Millimeter tolerance
    
    gmsh.option.setNumber("Geometry.ToleranceBoolean", 0.5)
    gmsh.option.setNumber("Geometry.OCCScaling", 1)
    gmsh.option.setNumber("Geometry.OCCParallel", 1)
    
    # Fast rendering - disable perfectionism for preview speed
    gmsh.option.setNumber("Mesh.Optimize", 0)               # Disable the optimization loop
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)         # Disable Netgen optimizer
    gmsh.option.setNumber("Mesh.Algorithm", 1)              # Use MeshAdapt (most forgiving)
    gmsh.option.setNumber("Mesh.MaxRetries", 1)             # Don't try to re-mesh failed surfaces
    gmsh.option.setNumber("Mesh.RecombineAll", 0)
    gmsh.option.setNumber("Mesh.Smoothing", 0)              # Disable smoothing for speed
    gmsh.option.setNumber("Mesh.StlRemoveDuplicateTriangles", 1)
    
    ext = os.path.splitext(r"{step_filepath}")[1].lower()
    print(f"[PREVIEW] Loading CAD file ({{ext}}): {step_filepath}")
    
    # Use importShapes for best compatibility with various formats
    if ext in ['.stl', '.obj', '.ply']:
        gmsh.merge(r"{step_filepath}")
    else:
        gmsh.model.occ.importShapes(r"{step_filepath}")
        
    print("[PREVIEW] Synchronizing OCC model...")
    gmsh.model.occ.synchronize()
    print("[PREVIEW] CAD loaded successfully.")
    
    # Extract CAD geometry info BEFORE meshing
    volumes = gmsh.model.getEntities(3)  # 3D entities (volumes)
    surfaces = gmsh.model.getEntities(2)  # 2D entities (surfaces)
    curves = gmsh.model.getEntities(1)    # 1D entities (curves)
    points = gmsh.model.getEntities(0)    # 0D entities (points)
    
    # Get bounding box
    bbox = gmsh.model.getBoundingBox(-1, -1)
    bbox_size = [bbox[3]-bbox[0], bbox[4]-bbox[1], bbox[5]-bbox[2]]
    diag = (bbox_size[0]**2 + bbox_size[1]**2 + bbox_size[2]**2)**0.5
    
    print(f"[PREVIEW] Geometry Info:")
    print(f"  - Volumes: {{len(volumes)}}")
    print(f"  - Surfaces: {{len(surfaces)}}")
    print(f"  - Curves: {{len(curves)}}")
    print(f"  - Points: {{len(points)}}")
    print(f"  - Bounding Box: {{bbox_size[0]:.2f}} x {{bbox_size[1]:.2f}} x {{bbox_size[2]:.2f}}")
    print(f"  - Diagonal: {{diag:.2f}}")
    
    # Calculate volume if possible
    total_volume = 0.0
    for dim, tag in volumes:
        try:
            total_volume += gmsh.model.occ.getMass(dim, tag)
        except:
            pass
    
    # Try meshing with different settings (progressively coarser and more tolerant)
    # Format: (mesh_factor, algorithm, tolerance)
    # Algorithms: 1=MeshAdapt, 5=Delaunay, 6=Frontal-Delaunay
    mesh_attempts = [
        (20, 6, 1e-3),   # Fine Frontal-Delaunay (more robust than pure Delaunay)
        (15, 6, 1e-2),   # Medium Frontal-Delaunay
        (10, 1, 1e-2),   # Coarser MeshAdapt
        (8, 1, 1e-1),    # MeshAdapt with high tolerance
        (5, 6, 0.5),     # Very coarse Frontal with high tolerance
        (3, 1, 1.0),     # Ultra coarse MeshAdapt
    ]
    
    mesh_success = False
    for i, (mesh_factor, algo, tol) in enumerate(mesh_attempts):
        try:
            print(f"[PREVIEW] Attempt {{i+1}}/{{len(mesh_attempts)}}: sizing=diag/{{mesh_factor}}, algo={{algo}}, tolerance={{tol}}")
            try_mesh(mesh_factor, algorithm=algo, tolerance=tol)
            mesh_success = True
            print(f"MESH_OK:factor={{mesh_factor}},algo={{algo}}", file=sys.stderr)
            break
        except Exception as mesh_err:
            print(f"RETRY:factor={{mesh_factor}},algo={{algo}} failed: {{mesh_err}}", file=sys.stderr)
            continue
    
    if not mesh_success:
        # Last resort: try OCC native tessellation
        print("RETRY: Trying OCC native tessellation as last resort", file=sys.stderr)
        try:
            gmsh.model.mesh.clear()
            # Use very high tolerance for OCC tessellation
            gmsh.option.setNumber("Geometry.Tolerance", 1.0)
            gmsh.option.setNumber("Geometry.OCCTargetMesh", 1)  # Enable OCC target meshing
            # Try minimal surface mesh generation
            for dim, tag in gmsh.model.getEntities(2):
                try:
                    gmsh.model.mesh.setTransfinite(dim, tag)
                except:
                    pass
            gmsh.model.mesh.generate(2)
            mesh_success = True
            print("MESH_OK:OCC_fallback", file=sys.stderr)
        except Exception as occ_err:
            print(f"RETRY:OCC_fallback failed: {{occ_err}}", file=sys.stderr)
    
    if not mesh_success:
        raise Exception("All meshing attempts failed - geometry may be too complex")
    
    # Get nodes
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    
    # Get triangles
    vertices = []
    nodes = {{}}
    for i, tag in enumerate(node_tags):
        nodes[int(tag)] = [node_coords[3*i], node_coords[3*i+1], node_coords[3*i+2]]
    
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
    
    # Output as JSON with geometry info
    import json
    result = {{
        "vertices": vertices, 
        "count": len(vertices)//9,
        "geometry": {{
            "volumes": len(volumes),
            "surfaces": len(surfaces),
            "curves": len(curves),
            "points": len(points),
            "bbox": bbox_size,
            "bbox_diagonal": diag,
            "total_volume": total_volume
        }}
    }}
    print("MESH_DATA:" + json.dumps(result))
    
except Exception as e:
    print("ERROR:" + str(e))
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''
    
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
    """Parse Gmsh .msh file for Three.js visualization using gmsh API"""
    import subprocess
    import tempfile
    from collections import defaultdict
    
    print(f"[MESH PARSE] Parsing mesh file: {msh_filepath}")
    
    # Load quality data if available
    quality_filepath = Path(msh_filepath).with_suffix('.quality.json')
    per_element_quality = {}
    quality_threshold = 0.3
    
    if quality_filepath.exists():
        try:
            with open(quality_filepath, 'r') as f:
                qdata = json.load(f)
                per_element_quality = {int(k): v for k, v in qdata.get('per_element_quality', {}).items()}
                quality_threshold = qdata.get('quality_threshold_10', 0.3)
                print(f"[MESH PARSE] Loaded quality data for {len(per_element_quality)} elements")
        except Exception as e:
            print(f"[MESH PARSE] Failed to load quality data: {e}")

    # Use gmsh in subprocess to extract surface mesh
    gmsh_script = f'''
import gmsh
import json
import sys
from collections import defaultdict

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)

try:
    gmsh.open("{msh_filepath}")
    
    # Get all nodes
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    nodes = {{}}
    for i, tag in enumerate(node_tags):
        nodes[int(tag)] = [node_coords[i*3], node_coords[i*3+1], node_coords[i*3+2]]
    
    # Map elements to entities (Surface IDs)
    element_to_entity = {{}}
    entities = gmsh.model.getEntities(2) # Get all surfaces
    for dim, entity_tag in entities:
        # Get elements for this specific surface
        try:
            e_types, e_tags, _ = gmsh.model.mesh.getElements(2, entity_tag)
            for i in range(len(e_types)):
                for tag in e_tags[i]:
                    element_to_entity[int(tag)] = int(entity_tag)
        except:
            pass

    # Get all unique elements scan
    vertices = []
    element_tags = []
    entity_tags_list = [] # Parralel to element_tags
    
    face_count = {{}}  # face_key -> count
    face_data = {{}}   # face_key -> (original_vertices, element_tag)
    direct_triangles = [] # (n1, n2, n3, el_tag, entity_tag)
    
    # Get all elements efficiently
    elem_types, elem_tags_list, node_tags_list = gmsh.model.mesh.getElements(-1, -1)
    
    # Pre-calculate qualities for 2D elements if we need fallbacks
    # This ensures older meshes or missing quality.json still works
    fallback_quality = {{}} # tag -> quality
    
    try:
        # Get all 2D element tags
        tri_tags_all = []
        for et, tags in zip(elem_types, elem_tags_list):
            if et in [2, 9]: # Triangles
                 tri_tags_all.extend(tags)
        
        if tri_tags_all:
             min_sicn = gmsh.model.mesh.getElementQualities(tri_tags_all, "minSICN")
             for i, tag in enumerate(tri_tags_all):
                 fallback_quality[int(tag)] = float(min_sicn[i])
    except:
        pass

    # Debug: Print found types
    print(f"DEBUG: Found element types: {{list(elem_types)}}", file=sys.stderr)

    for et, tags, nodes_per_elem in zip(elem_types, elem_tags_list, node_tags_list):
        if et == 2:  # Triangle (2D surface element)
            nodes_per = 3
            for i, tag in enumerate(tags):
                el_tag = int(tag)
                n1 = int(nodes_per_elem[i*nodes_per])
                n2 = int(nodes_per_elem[i*nodes_per + 1])
                n3 = int(nodes_per_elem[i*nodes_per + 2])
                ent_tag = element_to_entity.get(el_tag, 0)
                direct_triangles.append((n1, n2, n3, el_tag, ent_tag))
                
        elif et == 4:  # 4-node Tetrahedron
            nodes_per = 4
            for i, tag in enumerate(tags):
                n1 = int(nodes_per_elem[i*nodes_per])
                n2 = int(nodes_per_elem[i*nodes_per + 1])
                n3 = int(nodes_per_elem[i*nodes_per + 2])
                n4 = int(nodes_per_elem[i*nodes_per + 3])
                
                # Check faces
                tet_faces = [
                    ((n1, n2, n3), tuple(sorted([n1, n2, n3]))),
                    ((n1, n2, n4), tuple(sorted([n1, n2, n4]))),
                    ((n1, n3, n4), tuple(sorted([n1, n3, n4]))),
                    ((n2, n3, n4), tuple(sorted([n2, n3, n4])))
                ]
                for orig_face, face_key in tet_faces:
                    if face_key in face_count:
                        face_count[face_key] += 1
                    else:
                        face_count[face_key] = 1
                        face_data[face_key] = (orig_face, int(tag))
                        
        elif et == 11:  # 10-node Tet
            nodes_per = 10
            for i, tag in enumerate(tags):
                n1 = int(nodes_per_elem[i*nodes_per])
                n2 = int(nodes_per_elem[i*nodes_per + 1])
                n3 = int(nodes_per_elem[i*nodes_per + 2])
                n4 = int(nodes_per_elem[i*nodes_per + 3])
                tet_faces = [
                    ((n1, n2, n3), tuple(sorted([n1, n2, n3]))),
                    ((n1, n2, n4), tuple(sorted([n1, n2, n4]))),
                    ((n1, n3, n4), tuple(sorted([n1, n3, n4]))),
                    ((n2, n3, n4), tuple(sorted([n2, n3, n4])))
                ]
                for orig_face, face_key in tet_faces:
                    if face_key in face_count:
                        face_count[face_key] += 1
                    else:
                        face_count[face_key] = 1
                        face_data[face_key] = (orig_face, int(tag))
    
    # Extract boundary faces from volumes
    surface_triangles = [] # (n1, n2, n3, el_tag, entity_tag)
    for face_key, count in face_count.items():
        if count == 1:
            orig_face, el_tag = face_data[face_key]
            # Boundary faces form tets don't have explicit entity tags usually
            # unless we map them spatially. Use 0 for now.
            surface_triangles.append((orig_face[0], orig_face[1], orig_face[2], el_tag, 0))
    
    # Strategy: If we have direct surface triangles, use them (they have entity tags).
    # If not, use boundary faces. 
    # Or combine them?
    # Simple merge for now.
    all_triangles = direct_triangles + surface_triangles
    
    # deduplicate based on node sets if needed? 
    # For now, just output.
    
    for n1, n2, n3, el_tag, ent_tag in all_triangles:
        if n1 in nodes and n2 in nodes and n3 in nodes:
            vertices.extend(nodes[n1])
            vertices.extend(nodes[n2])
            vertices.extend(nodes[n3])
            element_tags.append(el_tag)
            entity_tags_list.append(ent_tag)
            
    result = {{
        "vertices": vertices,
        "element_tags": element_tags,
        "entity_tags": entity_tags_list,
        "num_nodes": len(nodes),
        "fallback_quality": fallback_quality # Return on-the-fly quality
    }}
    print("MESH_DATA:" + json.dumps(result))
    
except Exception as e:
    print("ERROR:" + str(e))
    import traceback
    traceback.print_exc()
finally:
    gmsh.finalize()
'''

    try:
        result = subprocess.run(
            [sys.executable, '-c', gmsh_script],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        print(f"[MESH PARSE] Subprocess stderr: {result.stderr[:500] if result.stderr else 'empty'}")
        
        # Parse output
        mesh_data = None
        for line in result.stdout.split('\n'):
            if line.startswith('MESH_DATA:'):
                mesh_data = json.loads(line[10:])
                break
            elif line.startswith('ERROR:'):
                return {"error": line[6:], "vertices": [], "colors": [], "numVertices": 0, "numTriangles": 0}
        
        if not mesh_data:
            return {"error": "Failed to parse mesh - no data returned", "vertices": [], "colors": [], "numVertices": 0, "numTriangles": 0}
        
        vertices = mesh_data['vertices']
        element_tags = mesh_data['element_tags']
        entity_tags = mesh_data.get('entity_tags', []) # New field
        fallback_quality = mesh_data.get('fallback_quality', {}) # Parse fallback

        def get_color(q, metric='sicn'):
            if q is None:
                return 0.29, 0.56, 0.89  # Default blue
                
            # Normalize q so that 0 is WORST and 1 is BEST for coloring
            val = q
            if metric == 'skewness':
                val = max(0.0, min(1.0, 1.0 - q))
            elif metric == 'aspect_ratio':
                # AR 1.0 is best, 5.0+ is poor
                val = max(0.0, min(1.0, 1.0 - (q - 1.0) / 4.0))
            else:
                val = max(0.0, min(1.0, q))
                
            # Smooth gradient from red (0) to green (1)
            if val <= 0.1:
                return 0.8, 0.0, 0.0  # Dark red - very bad
            elif val < 0.3:
                # Red to Orange
                t = (val - 0.1) / 0.2
                return 1.0, 0.3 * t, 0.0
            elif val < 0.5:
                # Orange to Yellow
                t = (val - 0.3) / 0.2
                return 1.0, 0.3 + 0.7 * t, 0.0
            elif val < 0.7:
                # Yellow to Light Green
                t = (val - 0.5) / 0.2
                return 1.0 - 0.5 * t, 1.0, 0.0
            else:
                # Light Green to Green
                t = min(1.0, (val - 0.7) / 0.3)
                return 0.5 - 0.5 * t, 0.8 + 0.2 * t, 0.2 * t

        # Load all metrics from quality.json
        per_element_gamma = {}
        per_element_skewness = {}
        per_element_aspect_ratio = {}
        
        try:
            if quality_filepath.exists():
                with open(quality_filepath, 'r') as f:
                    qdata = json.load(f)
                    per_element_gamma = {int(k): v for k, v in qdata.get('per_element_gamma', {}).items()}
                    per_element_skewness = {int(k): v for k, v in qdata.get('per_element_skewness', {}).items()}
                    per_element_aspect_ratio = {int(k): v for k, v in qdata.get('per_element_aspect_ratio', {}).items()}
        except: pass

        # Build colors for all metrics
        colors_sicn = []
        colors_gamma = []
        colors_skewness = []
        colors_aspect_ratio = []
        
        matched_count = 0
        unmatched_count = 0
        
        # If per_element_quality is empty (no file), initialize it with fallback
        if not per_element_quality and fallback_quality:
            per_element_quality = {k: v for k, v in fallback_quality.items()}

        for el_tag in element_tags:
            tag_key = int(el_tag)
            
            # 1. SICN
            q_sicn = per_element_quality.get(tag_key)
            if q_sicn is None: q_sicn = fallback_quality.get(tag_key)
            if q_sicn is None: q_sicn = fallback_quality.get(str(tag_key))
            
            # 2. Others
            q_gamma = per_element_gamma.get(tag_key)
            q_skew = per_element_skewness.get(tag_key)
            q_ar = per_element_aspect_ratio.get(tag_key)
            
            if q_sicn is not None: matched_count += 1
            else: unmatched_count += 1
            
            # Calculate RGBs
            r_s, g_s, b_s = get_color(q_sicn, 'sicn')
            r_g, g_g, b_g = get_color(q_gamma, 'gamma')
            r_k, g_k, b_k = get_color(q_skew, 'skewness')
            r_a, g_a, b_a = get_color(q_ar, 'aspect_ratio')
            
            colors_sicn.extend([r_s, g_s, b_s] * 3)
            colors_gamma.extend([r_g, g_g, b_g] * 3)
            colors_skewness.extend([r_k, g_k, b_k] * 3)
            colors_aspect_ratio.extend([r_a, g_a, b_a] * 3)
        
        print(f"[MESH PARSE] Mapped colors for {matched_count} elements ({unmatched_count} missing)")
        
        print(f"[MESH PARSE] Quality color mapping: {matched_count} matched, {unmatched_count} unmatched")
                
        print(f"[MESH PARSE] Extracted {len(vertices)//9} triangles, {mesh_data['num_nodes']} nodes")
        
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

        return {
            "vertices": vertices,
            "colors": colors_sicn, # Compatibility
            "qualityColors": {
                "sicn": colors_sicn,
                "gamma": colors_gamma,
                "skewness": colors_skewness,
                "aspectRatio": colors_aspect_ratio
            },
            "entityTags": entity_tags,  # Pass raw array of surface IDs per triangle
            "numVertices": len(vertices) // 3,
            "numTriangles": len(vertices) // 9,
            "qualityMetrics": quality_summary,
            "histogramData": histogram_data,
            "hasQualityData": len(quality_values) > 0
        }

    except subprocess.TimeoutExpired:
        return {"error": "Mesh parsing timed out", "vertices": [], "colors": [], "numVertices": 0, "numTriangles": 0}
    except Exception as e:
        import traceback
        print(f"[MESH PARSE ERROR] {e}")
        traceback.print_exc()
        return {"error": str(e), "vertices": [], "colors": [], "numVertices": 0, "numTriangles": 0}


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
