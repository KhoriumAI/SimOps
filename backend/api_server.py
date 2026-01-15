#!/usr/bin/env python3
"""
Flask API Server for Mesh Generation
With JWT Authentication and SQLAlchemy Database
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_jwt_extended import JWTManager, jwt_required, get_jwt_identity
from flask_socketio import SocketIO
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

# Add current directory and parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config
from models import db, User, Project, MeshResult, TokenBlocklist, ActivityLog, DownloadRecord, Feedback
from werkzeug.utils import secure_filename
from routes.auth import auth_bp, check_if_token_revoked
from routes.batch import batch_bp
from routes.webhooks import webhook_bp, init_socketio, register_socketio_handlers
from storage import get_storage, S3Storage, LocalStorage
from modal_client import modal_client
from slicing import generate_slice_mesh, parse_msh_for_slicing
from job_logger import generate_job_id, log_mesh_job, log_job_update
import numpy as np
try:
    import meshio
except ImportError:
    meshio = None


def cleanup_stuck_jobs(app):
    """
    On development server startup, reset any jobs stuck in 'processing' state.
    This prevents 'zombie' jobs from looking like they are running forever.
    """
    # Only run in development or if explicitly requested
    if app.config.get('FLASK_ENV') == 'development' or os.environ.get('FLASK_DEBUG') == '1':
        try:
            print("==================================================")
            print("[DEV] Checking for stuck jobs on startup...")
            stuck_projects = Project.query.filter_by(status='processing').all()
            count = 0
            for p in stuck_projects:
                print(f"[DEV] Resetting stuck project: {p.id} ({p.filename})")
                p.status = 'uploaded'
                p.error_message = 'Job killed on server restart'
                
                latest = p.mesh_results.order_by(MeshResult.created_at.desc()).first()
                if latest and latest.status in ['pending', 'processing', 'queued']:
                    latest.status = 'failed'
                    latest.error_message = 'Server restarted'
                    latest.logs = (latest.logs or []) + [f"[{datetime.utcnow().isoformat()}] Server restart detected. Job killed."]
                count += 1
            
            if count > 0:
                db.session.commit()
                print(f"[DEV] Reset {count} stuck projects.")
            else:
                print("[DEV] No stuck jobs found.")
            print("==================================================")
        except Exception as e:
            print(f"[DEV] Warning: Job cleanup failed: {e}")

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
        "http://muaz-mesh-web-dev.s3-website-us-west-1.amazonaws.com",
        "https://app.khorium.ai",
        "http://app.khorium.ai"
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
    app.register_blueprint(webhook_bp)

    # Initialize SocketIO
    # We use threading mode for compatibility with standard Flask run
    socketio_url = app.config.get('SOCKETIO_MESSAGE_QUEUE')
    print(f"[SocketIO] initializing with message queue: {socketio_url}")
    socketio = SocketIO(app, cors_allowed_origins="*", message_queue=socketio_url, async_mode='threading')
    app.socketio = socketio
    
    # Initialize routes with socketio instance
    init_socketio(socketio)
    register_socketio_handlers(socketio)
    
    # Ensure instance folder exists for SQLite database
    instance_dir = Path(__file__).parent / 'instance'
    instance_dir.mkdir(parents=True, exist_ok=True)
    
    # Create database tables
    with app.app_context():
        db.create_all()
    
    # Register routes
    register_routes(app)
    
    # Run cleanup of stuck jobs on development startup
    if app.config.get('FLASK_ENV') == 'development' or os.environ.get('FLASK_DEBUG') == '1':
        with app.app_context():
            cleanup_stuck_jobs(app)
    return app


generation_lock = Lock()
running_processes = {}


def submit_mesh_job_sync(app, project_id: str, quality_params: dict = None, use_modal_override: bool = None):
    """
    Synchronously submit mesh job and return job_id.
    Does NOT block for completion. Returns (job_id, mode, result_id).
    """
    import tempfile
    from compute_backend import get_compute_provider
    
    with app.app_context():
        project = Project.query.get(project_id)
        user = User.query.get(project.user_id) if project else None

        if not project:
            raise ValueError(f"Project {project_id} not found")

        # Update Project Status
        project.status = 'processing'
        project.mesh_count = (project.mesh_count or 0) + 1
        db.session.commit()
        
        print(f"[MESH GEN] Started for project {project_id}")
        
        # Determine Mode
        mode = None
        if use_modal_override is True:
            mode = 'CLOUD'
        elif use_modal_override is False:
            mode = 'LOCAL'
        else:
            # First check for explicit COMPUTE_MODE
            mode = app.config.get('COMPUTE_MODE')
            if not mode:
                # Fallback to USE_MODAL_COMPUTE flag
                if app.config.get('USE_MODAL_COMPUTE', False):
                    mode = 'CLOUD'
                else:
                    mode = 'LOCAL'
            
        # Get Provider
        provider = get_compute_provider(mode)
        print(f"[MESH GEN] Using Provider: {provider.name} (Mode: {mode})")
        
        # Handle File Access / Input Path
        storage = get_storage()
        input_path = project.filepath
        output_folder = Path(app.config['OUTPUT_FOLDER'])
        output_folder.mkdir(parents=True, exist_ok=True)
        
        if mode == 'CLOUD':
            if not project.filepath.startswith('s3://'):
                raise ValueError("Cloud compute requires S3 storage. Please upload to S3 first.")
            
            # Diagnostic Check for Modal
            from modal_client import modal_client
            diag = modal_client.diagnose()
            if not diag['ready']:
                 issues = "; ".join(diag['issues'])
                 print(f"[MESH GEN ERROR] Modal Misconfiguration: {issues}")
                 raise RuntimeError(f"Cloud Compute Unavailable: {issues}")

        else:
            # Local Mode: Need file on disk
            if input_path.startswith('s3://'):
                local_input_dir = Path(tempfile.mkdtemp(prefix='meshgen_input_'))
                local_input_file = local_input_dir / Path(project.filename).name
                if not storage.file_exists(project.filepath):
                    raise Exception(f"CAD file not found on S3: {project.filepath}")
                
                print(f"[MESH GEN] Downloading from S3: {project.filepath}")
                storage.download_to_local(project.filepath, str(local_input_file))
                input_path = str(local_input_file)
            else:
                if not Path(input_path).exists():
                     print(f"[MESH GEN ERROR] Local file missing: {input_path}")

        # Submit Job (Synchronous start)
        # For Modal, this spawns the function and returns ID quickly
        # For Local, this starts subprocess and returns ID quickly
        job_id = provider.submit_job(
            task_type='mesh',
            input_path=input_path,
            output_dir=str(output_folder),
            quality_params=quality_params,
            project_id=project_id
        )
        
        # Create DB Record
        internal_job_id = generate_job_id('MSH')
        logs = []
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] [INFO] Job Started: {internal_job_id} on {provider.name}")
        
        mesh_result = MeshResult(
            project_id=project_id,
            job_id=internal_job_id,
            strategy='processing',
            logs=logs,
            params=quality_params,
            modal_job_id=job_id, 
            modal_status='pending'
        )
        db.session.add(mesh_result)
        db.session.commit()
        result_id = mesh_result.id
        
            
        # Track cancellation handle
        with generation_lock:
            running_processes[project_id] = {'provider': provider, 'job_id': job_id}
            
        return job_id, mode, result_id, internal_job_id


def monitor_mesh_job(app, project_id, job_id, mode, result_id):
    """
    Monitor running job in background thread.
    """
    import time
    from compute_backend import get_compute_provider
    
    # Re-instantiate provider in thread (stateless)
    provider = get_compute_provider(mode)
    start_time = time.time()
    
    with app.app_context():
        # Re-fetch objects
        mesh_result = db.session.get(MeshResult, result_id)
        if not mesh_result:
            print(f"[MONITOR] Result {result_id} not found, aborting")
            return
            
        logs = list(mesh_result.logs) if mesh_result.logs else []
        project = Project.query.get(project_id)
        # Note: 'user' access needs re-query if needed, but we used email stored in project/storage call
        user = User.query.get(project.user_id) if project else None

        try:
            # Monitor Loop
            last_log_idx = len(logs)
            while True:
                status_data = provider.get_status(job_id)
                status = status_data.get('status')
                new_logs = status_data.get('logs', [])
                
                # Append new logs
                for line in new_logs:
                     timestamp = datetime.now().strftime("%H:%M:%S")
                     logs.append(f"[{timestamp}] {line}")
                
                 # Update DB if logs changed
                if len(logs) > last_log_idx:
                    mesh_result.logs = logs.copy()
                    db.session.commit()
                    last_log_idx = len(logs)
                
                # Check cancellation by user
                db.session.refresh(project)
                if project.status == 'stopped':
                    print(f"[MESH GEN] Project stopped by user.")
                    provider.cancel_job(job_id)
                    break

                if status in ['completed', 'failed', 'cancelled']:
                    break
                
                time.sleep(1) # Poll interval
            
            # Handle Completion
            if status == 'completed':
                # Fetch results
                if mode == 'CLOUD':
                    res = status_data.get('result', {})
                    # If result not in status, try explicit fetch
                    if not res: 
                        try:
                            res = provider.fetch_results(job_id)
                        except:
                            res = {}
                    
                    mesh_result.output_path = res.get('s3_output_path')
                    mesh_result.strategy = res.get('strategy')
                    mesh_result.quality_metrics = res.get('quality_metrics', {})
                    mesh_result.processing_time = res.get('metrics', {}).get('total_time_seconds', 0)
                    mesh_result.node_count = res.get('total_nodes', 0)
                    mesh_result.element_count = res.get('total_elements', 0)
                    
                    project.status = 'completed'
                    if mesh_result.output_path:
                        project.mesh_path = mesh_result.output_path
                    
                else:
                    # LOCAL Mode
                    res = provider.fetch_results(job_id)
                    
                    if res and res.get('success'):
                        mesh_result.strategy = res.get('strategy')
                        local_output_path = res.get('output_file')
                        mesh_result.quality_metrics = res.get('quality_metrics', {})
                            
                        # Handle S3 upload
                        storage = get_storage()
                        use_s3 = app.config.get('USE_S3', False)
                        
                        if use_s3 and user and local_output_path:
                            mesh_filename = Path(local_output_path).name
                            s3_path = storage.save_local_file(
                                local_path=local_output_path,
                                filename=mesh_filename,
                                user_email=user.email,
                                file_type='mesh'
                            )
                            mesh_result.output_path = s3_path
                        elif local_output_path:
                            mesh_result.output_path = local_output_path
                        
                        project.status = 'completed'
                    else:
                        project.status = 'error'
                        project.error_message = res.get('error') or res.get('message') or 'Unknown local error'
                
                final_msg = f"[{datetime.now().strftime('%H:%M:%S')}] [SUCCESS] Meshing completed in {time.time()-start_time:.1f}s"
                logs.append(final_msg)
                
            elif status == 'failed':
                project.status = 'error'
                err = status_data.get('error', 'Unknown error')
                project.error_message = err
                logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] [ERROR] {err}")
            
            mesh_result.completed_at = datetime.utcnow()
            mesh_result.logs = logs
            db.session.commit()
            
            # Legacy Logger Hook
            log_mesh_job(
                job_id=mesh_result.job_id, # Internal ID
                user_email=user.email if user else 'unknown',
                project_id=project_id,
                filename=project.filename,
                status='completed' if project.status == 'completed' else 'error',
                strategy=mesh_result.strategy,
                details={'processing_time': mesh_result.processing_time}
            )

        except Exception as e:
            print(f"[MESH GEN ERROR] {e}")
            project.status = 'error'
            project.error_message = str(e)
            db.session.commit()
            print(traceback.format_exc())
        finally:
             with generation_lock:
                if project_id in running_processes:
                    del running_processes[project_id]


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

    # In-memory storage for validation results
    _validation_results = {}

    @app.route('/api/dev/validate', methods=['POST'])
    def trigger_validation():
        """
        DEV-ONLY: Trigger the happy path validation script.
        Returns immediately with a job ID, validation runs in background.
        """
        # Only allow in development mode
        if not app.config.get('FLASK_ENV') == 'development' and not os.environ.get('FLASK_DEBUG') == '1':
            return jsonify({"error": "This endpoint is only available in development mode"}), 403
        
        import uuid
        validation_id = str(uuid.uuid4())
        
        # Initialize result storage
        _validation_results[validation_id] = {
            'status': 'running',
            'output': [],
            'exit_code': None
        }
        
        # Run validation script in background thread
        def run_validation():
            import subprocess
            try:
                result = subprocess.run(
                    [sys.executable, '-u', 'scripts/validate_happy_path.py', '--url', 'http://127.0.0.1:5000'],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                # Store results
                output_lines = result.stdout.split('\n') if result.stdout else []
                if result.stderr:
                    output_lines.extend(['', '=== STDERR ==='] + result.stderr.split('\n'))
                
                _validation_results[validation_id] = {
                    'status': 'completed',
                    'output': output_lines,
                    'exit_code': result.returncode
                }
                
                print(f"[VALIDATION {validation_id}] Completed with exit code: {result.returncode}")
                
            except subprocess.TimeoutExpired:
                _validation_results[validation_id] = {
                    'status': 'timeout',
                    'output': ['Validation timed out after 5 minutes'],
                    'exit_code': -1
                }
            except Exception as e:
                _validation_results[validation_id] = {
                    'status': 'error',
                    'output': [f"Error: {str(e)}"],
                    'exit_code': -1
                }
        
        thread = Thread(target=run_validation, daemon=True)
        thread.start()
        
        return jsonify({
            "message": "Validation started",
            "validation_id": validation_id
        })

    @app.route('/api/dev/validate/<validation_id>', methods=['GET'])
    def get_validation_result(validation_id: str):
        """
        DEV-ONLY: Get validation results by ID.
        """
        # Only allow in development mode
        if not app.config.get('FLASK_ENV') == 'development' and not os.environ.get('FLASK_DEBUG') == '1':
            return jsonify({"error": "This endpoint is only available in development mode"}), 403
        
        result = _validation_results.get(validation_id)
        if not result:
            return jsonify({"error": "Validation ID not found"}), 404
        
        return jsonify(result)



    @app.route('/api/strategies', methods=['GET'])
    def get_mesh_strategies():
        """
        Return available mesh generation strategies.
        Frontend should fetch this on load instead of hardcoding.
        """
        strategies = [
            {
                'id': 'tetrahedral_hxt',
                'name': 'Tetrahedral (HXT)',
                'description': 'High-performance parallel meshing - recommended default',
                'element_type': 'tet',
                'recommended': True
            },
            {
                'id': 'tet_delaunay',
                'name': 'Tet Delaunay',
                'description': 'Standard Delaunay algorithm - robust and reliable',
                'element_type': 'tet',
                'recommended': False
            },
            {
                'id': 'tet_frontal',
                'name': 'Tet Frontal',
                'description': 'Advancing front method - good for boundary layers',
                'element_type': 'tet',
                'recommended': False
            },
            {
                'id': 'tet_meshadapt',
                'name': 'Tet MeshAdapt',
                'description': 'Classic MeshAdapt algorithm - very stable',
                'element_type': 'tet',
                'recommended': False
            },
            {
                'id': 'highspeed_gpu',
                'name': 'Tetrahedral (HighSpeed GPU)',
                'description': 'GPU-accelerated with SDF visibility - robust for multi-body assemblies',
                'element_type': 'tet',
                'recommended': False,
                'requires': 'gpu'
            },
            {
                'id': 'hex_subdivision',
                'name': 'Hex Subdivision',
                'description': 'Hexahedral subdivision - experimental',
                'element_type': 'hex',
                'recommended': False,
                'experimental': True
            },
            {
                'id': 'hex_cartesian',
                'name': 'Hex Cartesian',
                'description': 'SnappyHexMesh cartesian grid - requires OpenFOAM',
                'element_type': 'hex',
                'recommended': False,
                'requires': 'openfoam'
            }
        ]
        
        # Return list of strategy names for simple dropdown use
        # and full details for advanced UI
        return jsonify({
            'strategies': strategies,
            'names': [s['name'] for s in strategies],
            'default': 'Tetrahedral (HXT)'
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
        user = db.session.get(User, current_user_id)
        
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
        
        try:
            # 4. Upload original file to storage (S3 or local)
            # We upload first so Modal has access to the file on S3
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

        preview_path = None
        preview_path = None
        try:
            # 5. Generate the preview
            # Refactored to use ComputeProvider
            from compute_backend import get_compute_provider
            
            # Determine mode
            mode = app.config.get('COMPUTE_MODE', 'LOCAL')
            # Legacy config override
            if app.config.get('USE_MODAL_COMPUTE', False):
                mode = 'CLOUD'
            
            provider = get_compute_provider(mode)
            print(f"[PREVIEW] Generating preview using {provider.name} (Mode: {mode})...")
            
            # Prepare input
            if mode == 'CLOUD':
                 if filepath.startswith('s3://'):
                     preview_input = filepath
                 else:
                     # Fallback to local if cloud requested but no S3
                     print("[PREVIEW] Cloud mode requires S3, falling back to local for preview.")
                     provider = get_compute_provider('LOCAL')
                     preview_input = temp_path
            else:
                 preview_input = temp_path

            result = provider.generate_preview(preview_input)
            
            if result.get('success') or result.get('status') == 'success':
                if result.get('preview_path'):
                    # Cloud provider returns path
                    preview_path = result.get('preview_path')
                    print(f"[PREVIEW] Preview generated: {preview_path}")
                else:
                    # Local provider returns data
                    preview_data = result
                    if "vertices" in preview_data:
                         # Save preview data as JSON to upload
                        preview_temp_path = os.path.join(temp_dir, f"{project_id}_preview.json")
                        with open(preview_temp_path, 'w') as f:
                            json.dump(preview_data, f)
                        
                        # Upload preview to S3/Local storage
                        preview_filename = f"{project_id}_preview.json"
                        preview_path = storage.save_local_file(
                            local_path=preview_temp_path,
                            filename=preview_filename,
                            user_email=user.email,
                            file_type='uploads'
                        )
                        print(f"[PREVIEW] Preview uploaded: {preview_path}")
            else:
                 print(f"[PREVIEW ERROR] Preview generation failed: {result.get('error') or result.get('message')}")

        except Exception as e:
            print(f"[PREVIEW ERROR] Failed to generate/upload preview: {e}")
            import traceback
            traceback.print_exc()
            preview_path = None

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

        data = request.json if request.is_json else {}
        
        # Robust parameter extraction
        quality_params = data.get('quality_params')
        if not quality_params:
            # Fallback: Check if the root level contains meshing parameters
            # This handles the current web frontend behavior and simple CLI-style requests
            root_params = {}
            possible_keys = [
                'mesh_strategy', 'target_elements', 'max_size_mm', 'min_size_mm', 
                'curvature_adaptive', 'element_order', 'ansys_mode', 'defer_quality', 
                'score_threshold', 'strategy_order', 'save_stl', 'worker_count',
                'quality_preset'
            ]
            for key in possible_keys:
                if key in data:
                    root_params[key] = data[key]
            
            if root_params:
                quality_params = root_params
                print(f"[DEBUG] Extracted quality_params from root body: {list(quality_params.keys())}")
            else:
                print(f"[DEBUG] No quality_params found in request")

        # Check for Modal override (only allowed in development/staging or if enabled)
        # We allow 'use_modal' to override if configured
        use_modal_override = data.get('use_modal') 

        try:
            # Synchronously submit job to get ID
            job_id, mode, result_id, internal_job_id = submit_mesh_job_sync(app, project_id, quality_params, use_modal_override)
            
            # Start background monitoring
            thread = Thread(target=monitor_mesh_job, args=(app, project_id, job_id, mode, result_id))
            thread.daemon = True
            thread.start()

            return jsonify({
                "message": "Mesh generation started", 
                "project_id": project_id,
                "job_id": job_id,
                "internal_job_id": internal_job_id
            })
            
        except Exception as e:
            print(f"[API ERROR] Failed to start mesh job: {e}")
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    @app.route('/api/projects/<project_id>/stop', methods=['POST'])
    @jwt_required()
    def stop_mesh(project_id: str):
        current_user_id = int(get_jwt_identity())
        project = Project.query.get(project_id)

        if not project:
            return jsonify({"error": "Project not found"}), 404

        if project.user_id != current_user_id:
            return jsonify({"error": "Access denied"}), 403
        
        # Try to find and kill the process or cancel Modal job
        process_killed = False
        with generation_lock:
            if project_id in running_processes:
                obj = running_processes[project_id]
                try:
                    if hasattr(obj, 'terminate'):
                        print(f"[API] Stopping local process for project {project_id}")
                        obj.terminate()
                        process_killed = True
                    elif hasattr(obj, 'cancel'):
                        print(f"[API] Canceling Modal job for project {project_id}")
                        obj.cancel()
                        process_killed = True
                except Exception as e:
                    print(f"[API] Error stopping/canceling: {e}")
                    # Try kill if terminate failed
                    if hasattr(obj, 'kill'):
                        try:
                            obj.kill()
                            process_killed = True
                        except Exception as e2:
                            print(f"[API] Error killing process: {e2}")

        # Force status update
        if project.status == 'processing':
            project.status = 'stopped'
            
            # Add stop log
            try:
                # Find the latest mesh result for this project that is 'processing' or 'pending'
                latest_result = MeshResult.query.filter_by(project_id=project_id)\
                    .order_by(MeshResult.created_at.desc()).first()
                
                if latest_result:
                    logs = list(latest_result.logs) if latest_result.logs else []
                    logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] [INFO] Process stopped by user.")
                    latest_result.logs = logs
                    # Also update modal_status if applicable
                    if latest_result.modal_job_id:
                        latest_result.modal_status = 'stopped'
            except Exception as e:
                print(f"[API] Error updating stop log: {e}")

            db.session.commit()
            return jsonify({"message": "Mesh generation stopped", "process_killed": process_killed})
        else:
            # It might have already finished or been stopped
            return jsonify({"message": f"Project is already in state {project.status}", "process_killed": process_killed})


    @app.route('/api/projects/<project_id>/status', methods=['GET'])
    @jwt_required()
    def get_project_status(project_id: str):
        try:
            current_user_id = int(get_jwt_identity())
            project = Project.query.get(project_id)

            if not project:
                return jsonify({"error": "Project not found"}), 404

            if project.user_id != current_user_id:
                return jsonify({"error": "Access denied"}), 403

            return jsonify(project.to_dict(include_results=True))
        except Exception as e:
            print(f"[STATUS ERROR] Failed to get status for {project_id}: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

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
        
        # Handle S3 paths
        use_s3 = app.config.get('USE_S3', False)
        is_s3 = str(result.output_path).startswith('s3://')
        
        temp_dir = None
        try:
            if use_s3 and is_s3:
                # Download to temp
                storage = get_storage()
                import tempfile
                import shutil
                temp_dir = tempfile.mkdtemp()
                local_msh_path = Path(temp_dir) / msh_path.name
                print(f"[SLICE] Downloading mesh from S3: {result.output_path}")
                try:
                    storage.download_to_local(str(result.output_path), str(local_msh_path))
                    msh_path = local_msh_path
                except Exception as e:
                     print(f"[SLICE ERROR] Failed to download from S3: {e}")
                     return jsonify({"error": f"Failed to retrieve mesh: {e}"}), 500
            
            if not msh_path.exists():
                return jsonify({"error": "Mesh file not found on disk"}), 404
                
            # Quality file
            quality_path = msh_path.with_name(msh_path.stem + "_result.json")
            
            # If S3, we need to try to download the quality file too if we want it
            if use_s3 and is_s3:
                 s3_quality_path = str(result.output_path).replace('.msh', '_result.json')
                 local_quality_path = Path(temp_dir) / Path(s3_quality_path).name
                 try:
                     storage.download_to_local(s3_quality_path, str(local_quality_path))
                     quality_path = local_quality_path
                 except:
                     print(f"[SLICE] Quality file not found on S3 or download failed: {s3_quality_path}")
            
            quality_map = {}
            if quality_path.exists():
                try:
                    with open(quality_path, 'r') as f:
                        res_data = json.load(f)
                        quality_map = res_data.get('per_element_quality', {})
                except: pass
                
            # Parse mesh for volume elements
            print(f"[SLICE] Parsing mesh for slice: {msh_path.name}...")
            nodes, elements = parse_msh_for_slicing(str(msh_path))
            
            if not nodes or not elements:
                return jsonify({"error": "Could not parse volume elements from mesh"}), 400
                
            # Calculate bounds for plane positioning
            pts = np.array(list(nodes.values()))
            bbox_min = pts.min(axis=0)
            bbox_max = pts.max(axis=0)
            center = (bbox_min + bbox_max) / 2.0
            size = bbox_max - bbox_min
            
            print(f"[SLICE DEBUG] Mesh Bounds: Min={bbox_min}, Max={bbox_max}")
            print(f"[SLICE DEBUG] Center={center}, Size={size}")
            
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
            
            print(f"[SLICE DEBUG] Cut Plane: Normal={plane_normal}, Origin={plane_origin}")
            
            # Generate slice
            print(f"[SLICE] Generating slice mesh on {axis}={offset_percent}%...")
            slice_data = generate_slice_mesh(nodes, elements, quality_map, plane_origin, plane_normal)
            
            v_count = len(slice_data.get('vertices', [])) // 3
            print(f"[SLICE] Generated {v_count} vertices")
            
            return jsonify({
                "success": True,
                "axis": axis,
                "offset": offset_percent,
                "mesh": slice_data
            })
        finally:
            if temp_dir:
                try:
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except: pass

    @app.route('/api/feedback', methods=['POST'])
    def submit_feedback():
        """Handle feedback submission"""
        # Optional auth - try to get user but don't enforce it strictly if we want anonymous feedback
        # But frontend sends token, so let's try to capture it.
        user_id = None
        try:
            # Manually check header to avoid @jwt_required throwing 401
            auth_header = request.headers.get('Authorization')
            if auth_header:
                # We can just decode it or use verify_jwt_in_request(optional=True) if configured
                # For now, let's trust the frontend passed email in body
                pass
        except:
            pass
            
        data = request.json
        if not data or not data.get('message'):
            return jsonify({"error": "Message required"}), 400
            
        feedback = Feedback(
            type=data.get('type', 'feedback'),
            message=data.get('message'),
            user_email=data.get('userEmail'),
            url=data.get('url'),
            user_agent=data.get('userAgent'),
            job_id=data.get('jobId')
        )
        
        # If we have a user in the system matching this email, link it
        if data.get('userEmail'):
            user = User.query.filter_by(email=data.get('userEmail')).first()
            if user:
                feedback.user_id = user.id
                
        try:
            db.session.add(feedback)
            db.session.commit()
            
            # Send Slack notification
            slack_webhook = app.config.get('SLACK_WEBHOOK_URL') or os.environ.get('SLACK_WEBHOOK_URL')
            if slack_webhook:
                try:
                    import requests
                    
                    # Formatting helpers
                    f_type = data.get('type', 'feedback').lower()
                    f_email = data.get('userEmail') or 'Anonymous'
                    f_msg = data.get('message')
                    f_url = data.get('url')
                    f_job = data.get('jobId') or 'N/A'
                    
                    # Style based on type
                    if 'bug' in f_type:
                        color = "#dd2e44" # Red
                        icon = ":bug:"
                        title = "New Bug Report"
                    elif 'feature' in f_type:
                        color = "#ffcc4d" # Yellow
                        icon = ":bulb:"
                        title = "New Feature Request"
                    else:
                        color = "#3b88c3" # Blue
                        icon = ":speech_balloon:"
                        title = "New Feedback"
                        
                    # Construct rich payload
                    slack_msg = {
                        "text": "Attention: @Aaron Wu @Mark Mukminov", # Restoring mention line
                        "attachments": [
                            {
                                "color": color,
                                "blocks": [
                                    {
                                        "type": "header",
                                        "text": {
                                            "type": "plain_text",
                                            "text": f"{icon} {title}",
                                            "emoji": True
                                        }
                                    },
                                    {
                                        "type": "section",
                                        "fields": [
                                            {
                                                "type": "mrkdwn",
                                                "text": f"*From:*\n{f_email}"
                                            },
                                            {
                                                "type": "mrkdwn",
                                                "text": f"*Type:*\n{f_type.title()}"
                                            }
                                        ]
                                    },
                                    {
                                        "type": "section",
                                        "text": {
                                            "type": "mrkdwn",
                                            "text": f"*Message:*\n{f_msg}"
                                        }
                                    },
                                    {
                                        "type": "context",
                                        "elements": [
                                            {
                                                "type": "mrkdwn",
                                                "text": f":calendar: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}  |  :globe_with_meridians: <{f_url}|View Page>  |  :label: `{f_job}`"
                                            }
                                        ]
                                    }
                                ]
                            }
                        ]
                    }
                    requests.post(slack_webhook, json=slack_msg, timeout=5)
                except Exception as slack_err:
                    print(f"[SLACK ERROR] Failed to send notification: {slack_err}")

            
            return jsonify({"success": True, "id": feedback.id})

        except Exception as e:
            db.session.rollback()
            print(f"[FEEDBACK ERROR] {e}")
            return jsonify({"error": "Failed to save feedback"}), 500

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
        import tempfile
        import shutil
        
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
        
        # Get requested format (default: msh, options: msh, fluent)
        requested_format = request.args.get('format', 'msh').lower()
        
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
            file_format=requested_format,
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
                'storage_path': output_path,
                'format': requested_format
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

        # Handle Fluent format conversion
        temp_dir = None
        try:
            if requested_format == 'fluent':
                print(f"[DOWNLOAD] Converting to Fluent format for project {project_id}")
                temp_dir = tempfile.mkdtemp()
                
                # Get the source MSH file locally
                if use_s3 and output_path.startswith('s3://'):
                    local_msh = Path(temp_dir) / "source.msh"
                    storage.download_to_local(output_path, str(local_msh))
                else:
                    local_msh = Path(output_path)
                
                # Convert to Fluent format
                fluent_output = Path(temp_dir) / f"{Path(project.filename).stem}_fluent.msh"
                
                try:
                    # Import the Fluent converter
                    import sys
                    core_path = Path(__file__).parent.parent / 'core'
                    if str(core_path) not in sys.path:
                        sys.path.insert(0, str(core_path))
                    
                    from write_fluent_mesh import convert_gmsh_to_fluent
                    convert_gmsh_to_fluent(str(local_msh), str(fluent_output))
                    
                    if fluent_output.exists():
                        print(f"[DOWNLOAD] Fluent conversion successful: {fluent_output.stat().st_size} bytes")
                        return send_file(
                            str(fluent_output),
                            as_attachment=True,
                            download_name=f"{Path(project.filename).stem}_fluent.msh"
                        )
                    else:
                        return jsonify({"error": "Fluent conversion failed - no output file"}), 500
                        
                except ImportError as ie:
                    print(f"[DOWNLOAD] Fluent converter not available: {ie}")
                    return jsonify({"error": "Fluent format export not available on this server"}), 501
                except Exception as conv_err:
                    print(f"[DOWNLOAD] Fluent conversion error: {conv_err}")
                    return jsonify({"error": f"Fluent conversion failed: {str(conv_err)}"}), 500
            
            elif requested_format == 'mechanical':
                print(f"[DOWNLOAD] Converting to Mechanical format for project {project_id}")
                temp_dir = tempfile.mkdtemp()
                
                # Get the source MSH file locally
                if use_s3 and output_path.startswith('s3://'):
                    local_msh = Path(temp_dir) / "source.msh"
                    storage.download_to_local(output_path, str(local_msh))
                else:
                    local_msh = Path(output_path)
                
                # Convert to Mechanical format (.inp)
                mech_output = Path(temp_dir) / f"{Path(project.filename).stem}_mechanical.inp"
                
                try:
                    import meshio
                    print("[DOWNLOAD] Loading mesh with meshio...")
                    mesh = meshio.read(str(local_msh))
                    print(f"[DOWNLOAD] Writing to Abaqus format: {mech_output}")
                    mesh.write(str(mech_output), file_format="abaqus")
                    
                    if mech_output.exists():
                        print(f"[DOWNLOAD] Mechanical conversion successful: {mech_output.stat().st_size} bytes")
                        return send_file(
                            str(mech_output),
                            as_attachment=True,
                            download_name=f"{Path(project.filename).stem}_mechanical.inp"
                        )
                    else:
                        return jsonify({"error": "Mechanical conversion failed - no output file"}), 500
                except ImportError:
                    print("[DOWNLOAD] meshio not installed")
                    return jsonify({"error": "Mechanical format export requires 'meshio' library"}), 501
                except Exception as e:
                    print(f"[DOWNLOAD] Mechanical conversion error: {e}")
                    return jsonify({"error": f"Mechanical conversion failed: {str(e)}"}), 500
            
            # Default: return original MSH format
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
        finally:
            # Cleanup temp directory
            if temp_dir:
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except:
                    pass

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
                
                # Also try to download result JSON if it exists (modern worker format)
                result_s3_path = output_path.replace('.msh', '_result.json')
                local_result_path = local_temp.name.replace('.msh', '_result.json')
                
                # Legacy quality JSON path
                quality_s3_path = output_path.replace('.msh', '.quality.json')
                local_quality_path = local_temp.name.replace('.msh', '.quality.json')
                
                print(f"[MESH DATA] Mesh S3 path: {output_path}")
                
                # Try result file first (richer data)
                if storage.file_exists(result_s3_path):
                    try:
                        storage.download_to_local(result_s3_path, local_result_path)
                        print(f"[MESH DATA] Downloaded result file: {local_result_path}")
                    except Exception as r_err:
                        print(f"[MESH DATA] Error downloading result file: {r_err}")
                
                # Try quality file as fallback
                if storage.file_exists(quality_s3_path):
                    try:
                        storage.download_to_local(quality_s3_path, local_quality_path)
                        print(f"[MESH DATA] Downloaded quality file: {local_quality_path}")
                    except Exception as q_err:
                        print(f"[MESH DATA] Error downloading quality file: {q_err}")

                
                mesh_data = parse_msh_file(local_temp.name)
                
                # Merge in richer quality metrics from result file if available
                # This ensures we get Gamma, Skewness, AR, and CFD metrics which might not be in the MSH tags
                if os.path.exists(local_result_path):
                    try:
                        with open(local_result_path, 'r') as f:
                            result_json = json.load(f)
                            if 'quality_metrics' in result_json:
                                print(f"[MESH DATA] Merging quality metrics from result file")
                                # Ensure we don't overwrite existing summary if it's better, but usually JSON is better
                                rich_metrics = result_json['quality_metrics']
                                
                                # If mesh_data already has qualityMetrics, merge them
                                if 'qualityMetrics' not in mesh_data or not mesh_data['qualityMetrics']:
                                    mesh_data['qualityMetrics'] = {}
                                
                                # Copy all metrics from JSON to mesh_data['qualityMetrics']
                                mesh_data['qualityMetrics'].update(rich_metrics)
                                mesh_data['hasQualityData'] = True
                    except Exception as e:
                        print(f"[MESH DATA] Failed to merge result JSON: {e}")

                # Clean up temp files
                Path(local_temp.name).unlink(missing_ok=True)
                Path(local_result_path).unlink(missing_ok=True)
                Path(local_quality_path).unlink(missing_ok=True)
            else:
                mesh_data = parse_msh_file(output_path)
                
                # For local files, also look for sibling _result.json
                try:
                    local_msh_path = Path(output_path)
                    local_result_json = local_msh_path.with_name(local_msh_path.stem + "_result.json")
                    if local_result_json.exists():
                        with open(local_result_json, 'r') as f:
                            result_json = json.load(f)
                            if 'quality_metrics' in result_json:
                                print(f"[MESH DATA] Merging quality metrics from local result file")
                                rich_metrics = result_json['quality_metrics']
                                if 'qualityMetrics' not in mesh_data or not mesh_data['qualityMetrics']:
                                    mesh_data['qualityMetrics'] = {}
                                mesh_data['qualityMetrics'].update(rich_metrics)
                                mesh_data['hasQualityData'] = True
                except Exception as e:
                    print(f"[MESH DATA] Failed to merge local result JSON: {e}")
            
            return jsonify(mesh_data)
        except Exception as e:
            return jsonify({"error": f"Failed to parse mesh: {str(e)}"}), 500

    @app.route('/api/projects/<project_id>/preview', methods=['GET'])
    @jwt_required()
    def get_cad_preview(project_id: str):
        """Get triangulated preview of CAD file before meshing"""
        import tempfile
        import time
        current_user_id = int(get_jwt_identity())
        project = Project.query.get(project_id)
        
        if not project:
            return jsonify({"error": "Project not found"}), 404
        
        if project.user_id != current_user_id:
            return jsonify({"error": "Access denied"}), 403
        
        if not project.filepath:
            return jsonify({"error": "CAD file not found"}), 404
        
        # Check for fast_mode query parameter
        fast_mode = request.args.get('fast_mode', 'false').lower() == 'true'
        
        try:
            storage = get_storage()
            use_s3 = app.config.get('USE_S3', False)
            
            # 1. Check for cached preview first (always preferred for speed)
            if project.preview_path:
                try:
                    print(f"[PREVIEW] Using cached preview for project {project_id}")
                    preview_bytes = storage.get_file(project.preview_path)
                    preview_data = json.loads(preview_bytes)
                    if "vertices" in preview_data:
                        return jsonify(preview_data)
                except Exception as e:
                    print(f"[PREVIEW] Failed to load cached preview: {e}")

            # 2. Generation Path (Unified - Always Optimized)
            start_time = time.time()
            filepath = project.filepath
            filename = project.filename or "unknown"
            
            # Detect file type
            file_ext = Path(filename).suffix.lower()
            is_msh_file = file_ext in ['.msh', '.mesh']
            
            # Download from S3 if needed
            local_path = None
            if use_s3 and filepath.startswith('s3://'):
                local_temp = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
                local_path = local_temp.name
                storage.download_to_local(filepath, local_path)
                filepath = local_path
            elif not Path(filepath).exists():
                return jsonify({"error": "CAD file not found"}), 404

            try:
                if is_msh_file:
                    print(f"[PREVIEW] Parsing MSH file directly: {filename}")
                    result = parse_msh_file(filepath)
                else:
                    # Optimized preview path (standard for visualization)
                    from compute_backend import get_preferred_backend
                    backend = get_preferred_backend()
                    print(f"[PREVIEW] Generating optimized preview using: {backend.name}")
                    
                    result = backend.generate_preview(filepath, timeout=120)
                    
                    if "error" in result:
                        print(f"[PREVIEW] Backend failed: {result['error']}. Using robust fallback...")
                        result = parse_step_file_for_preview(filepath)
                
                # Post-process for consistency
                if "vertices" in result and "colors" not in result:
                    result["colors"] = [0.3, 0.6, 0.9] * (len(result["vertices"]) // 3)
                
                result["_elapsed"] = time.time() - start_time
                return jsonify(result)
                
            finally:
                if local_path:
                    try: Path(local_path).unlink(missing_ok=True)
                    except: pass
                    
        except Exception as e:
            print(f"[PREVIEW ERROR] {e}")
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
    # STEP 1: Try Modal.com Compute (Preferred on Staging/Production)
    # ============================================================================
    from flask import current_app
    if current_app.config.get('USE_MODAL_COMPUTE', False):
        try:
            print("[PREVIEW] Attempting to use Modal for preview...")
            use_s3 = current_app.config.get('USE_S3', False)
            
            # Modal works best with S3-based files
            # If it's a local file (temp upload), we might need to upload it first
            # but usually by this point it's already on S3 in production.
            
            # Since we only have a local filepath here, let's see if we can find 
            # the corresponding S3 bucket/key if present.
            # (Note: parse_step_file_for_preview is usually called with a local temp path)
            
            # TODO: Add local-to-modal upload logic here if needed.
            # For now, if Modal is requested but we have no S3 context, we might skip.
            
            # If we are in production/staging, we likely want to use Modal.
            # But the Modal worker expects a bucket and key.
            pass # Placeholder for now, continue to Threadripper/Local fallback
        except Exception as e:
            print(f"[PREVIEW] Modal preview failed: {e}")

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
    
    # IMPORTANT: Standard_ConstructionError is a C++ exception that crashes the subprocess
    # BEFORE Python can catch it. So we cannot loop through tolerances - we must use
    # the safest approach first.
    #
    # Strategy:
    # 1. Try gmsh.merge() - this is often more lenient and less crash-prone
    # 2. If that fails, try importShapes with a moderate tolerance (1e-3)
    #
    # We set Geometry.Tolerance BEFORE loading to give OCC the best chance.
    
    loaded = False
    
    # Attempt 1: gmsh.merge() with moderate tolerance (safest approach)
    try:
        print("[PREVIEW] Loading attempt 1: gmsh.merge() with Geometry.Tolerance=1e-3...")
        gmsh.option.setNumber("Geometry.Tolerance", 1e-3)
        gmsh.option.setNumber("Geometry.OCCFixDegenerated", 0)  # Don't fix, just load
        gmsh.option.setNumber("Geometry.OCCFixSmallEdges", 0)
        gmsh.option.setNumber("Geometry.OCCFixSmallFaces", 0)
        gmsh.option.setNumber("Geometry.OCCSewFaces", 0)
        gmsh.option.setNumber("Geometry.OCCMakeSolids", 0)
        gmsh.merge(step_filepath)
        
        # Check if we got anything
        if gmsh.model.getEntities(3) or gmsh.model.getEntities(2):
            loaded = True
            print("[PREVIEW] Success loading with gmsh.merge()")
        else:
            print("[PREVIEW] gmsh.merge() loaded but no geometry entities found")
    except Exception as e:
        print(f"[PREVIEW] gmsh.merge() failed: {e}")
    
    # Attempt 2: importShapes with moderate tolerance
    if not loaded:
        try:
            print("[PREVIEW] Loading attempt 2: importShapes with Geometry.Tolerance=1e-2...")
            gmsh.option.setNumber("Geometry.Tolerance", 1e-2)
            gmsh.model.occ.importShapes(step_filepath)
            gmsh.model.occ.synchronize()
            
            if gmsh.model.getEntities(3) or gmsh.model.getEntities(2):
                loaded = True
                print("[PREVIEW] Success loading with importShapes()")
        except Exception as e:
            print(f"[PREVIEW] importShapes() failed: {e}")
    
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
    
    # Try meshing attempts - simplified for visualization speed
    mesh_attempts = [
        (10, 1, 1e-2), (5, 1, 1e-1)
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
                # per_element_* arrays are stored at root level in quality.json
                # (quality_metrics is a separate key with summary stats only)
                per_element_quality = {int(k): v for k, v in qdata.get('per_element_quality', {}).items()}
                per_element_gamma = {int(k): v for k, v in qdata.get('per_element_gamma', {}).items()}
                per_element_skewness = {int(k): v for k, v in qdata.get('per_element_skewness', {}).items()}
                per_element_aspect_ratio = {int(k): v for k, v in qdata.get('per_element_aspect_ratio', {}).items()}
                per_element_min_angle = {int(k): v for k, v in qdata.get('per_element_min_angle', {}).items()}
                print(f"[MESH PARSE] Loaded quality data for {len(per_element_quality)} elements")
        except Exception as e:
            print(f"[MESH PARSE] Failed to load quality data: {e}")
    else:
        print(f"[MESH PARSE] No sidecar result file found. Colors will be default.")
    
    # =====================================================================
    # COMPUTE QUALITY ON-THE-FLY if no pre-computed data exists
    # This enables quality coloring for uploaded meshes without companion JSON
    # =====================================================================
    if not per_element_quality:
        try:
            import gmsh
            print("[MESH PARSE] No pre-computed quality data - computing with Gmsh...")
            
            # Initialize Gmsh and load the mesh
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)  # Suppress output
            gmsh.merge(msh_filepath)
            
            # Get all 3D elements (tets, hexes)
            vol_types, vol_tags, vol_nodes = gmsh.model.mesh.getElements(3)
            
            for etype, etags, enodes in zip(vol_types, vol_tags, vol_nodes):
                if etype in [4, 11, 5, 12]:  # Tet4, Tet10, Hex8, Hex27
                    try:
                        sicn_vals = gmsh.model.mesh.getElementQualities(etags.tolist(), "minSICN")
                        gamma_vals = gmsh.model.mesh.getElementQualities(etags.tolist(), "gamma")
                        
                        for i, tag in enumerate(etags):
                            tag_int = int(tag)
                            sicn = float(sicn_vals[i])
                            gamma = float(gamma_vals[i])
                            
                            per_element_quality[tag_int] = sicn
                            per_element_gamma[tag_int] = gamma
                            per_element_skewness[tag_int] = max(0, 1.0 - sicn)
                            per_element_aspect_ratio[tag_int] = 1.0 / sicn if sicn > 0.01 else 100.0
                            per_element_min_angle[tag_int] = 60.0  # Default for tets
                    except Exception as e:
                        print(f"[MESH PARSE] Warning: Quality calc failed for element type {etype}: {e}")
            
            # Also get 2D surface elements (triangles, quads)
            surf_types, surf_tags, surf_nodes = gmsh.model.mesh.getElements(2)
            for etype, etags, enodes in zip(surf_types, surf_tags, surf_nodes):
                if etype in [2, 9, 3, 16]:  # Tri3, Tri6, Quad4, Quad9
                    try:
                        sicn_vals = gmsh.model.mesh.getElementQualities(etags.tolist(), "minSICN")
                        gamma_vals = gmsh.model.mesh.getElementQualities(etags.tolist(), "gamma")
                        
                        for i, tag in enumerate(etags):
                            tag_int = int(tag)
                            if tag_int not in per_element_quality:  # Don't overwrite 3D-inherited quality
                                sicn = float(sicn_vals[i])
                                per_element_quality[tag_int] = sicn
                                per_element_gamma[tag_int] = float(gamma_vals[i])
                                per_element_skewness[tag_int] = max(0, 1.0 - sicn)
                                per_element_aspect_ratio[tag_int] = 1.0 / sicn if sicn > 0.01 else 100.0
                                per_element_min_angle[tag_int] = 60.0
                    except:
                        pass
            
            gmsh.finalize()
            print(f"[MESH PARSE] Computed quality for {len(per_element_quality)} elements on-the-fly")
        except Exception as e:
            print(f"[MESH PARSE] On-the-fly quality computation failed: {e}")
            try:
                gmsh.finalize()
            except:
                pass

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
            val = q
            
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
                
                # Build quality summary with key names matching frontend expectations
                quality_values = [per_element_quality.get(int(t)) for t in element_tags if per_element_quality.get(int(t)) is not None]
                gamma_values = [per_element_gamma.get(int(t)) for t in element_tags if per_element_gamma.get(int(t)) is not None]
                quality_summary = {}
                histogram_data = []
                if quality_values:
                    quality_summary = {
                        # Frontend expects these key names
                        "sicn_min": min(quality_values),
                        "sicn_max": max(quality_values),
                        "sicn_avg": sum(quality_values) / len(quality_values),
                        "total_elements": len(per_element_quality),
                        "element_count": len(per_element_quality),
                        # Also add alternate names for compatibility
                        "min_sicn": min(quality_values),
                        "max_sicn": max(quality_values),
                        "avg_sicn": sum(quality_values) / len(quality_values),
                    }
                    # Add gamma if available
                    if gamma_values:
                        quality_summary["gamma_min"] = min(gamma_values)
                        quality_summary["gamma_max"] = max(gamma_values)
                        quality_summary["gamma_avg"] = sum(gamma_values) / len(gamma_values)
                    # Count poor elements (SICN < 0.1)
                    poor_count = len([q for q in quality_values if q < 0.1])
                    quality_summary["poor_elements"] = poor_count
                
                if not vertices or not element_tags:
                    raise Exception("Meshio returned empty geometry (no boundary faces found)")
                
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
                    # When we have volume elements, explicit triangles are redundant with tet faces.
                    # We skip counting them to avoid inflating the count (which would hide boundary faces).
                    # We only add them if they're NOT already in face_map (from tet processing).
                    n = [node_id_to_index[nid] for nid in node_ids[:3]]
                    key = tuple(sorted(n))
                    if key not in face_map:
                        # This triangle is NOT a face of any tet - it's a standalone surface element
                        face_map[key] = {'nodes': n, 'count': 1, 'element_tag': el_tag, 'entity_tag': entity_tag, 'is_surface': True}
                    else:
                        # Already exists from tet processing - just mark it as surface
                        face_map[key]['is_surface'] = True
                        # CRITICAL: Overwrite the entity_tag (which was Volume ID) with the Surface ID
                        # This ensures the frontend receives the specific Surface ID for selection,
                        # rather than the generic Volume ID.
                        face_map[key]['entity_tag'] = entity_tag
                except KeyError: pass

        if msh_version.startswith('4'):
            # MSH 4.x: numEntityBlocks numElements minTag maxTag
            num_blocks = int(header[0])
            curr_line = 1
            for _ in range(num_blocks):
                line_parts = elements_section[curr_line].split()
                curr_line += 1
                if not line_parts: continue
                # DEBUG: Print first few blocks to see what we are parsing
                if curr_line < 20:
                     print(f"[MESH PARSE DEBUG] Block Header Raw: {line_parts}")
                entity_dim = int(line_parts[0])
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
            
        # Extract boundary faces (count == 1 OR explicitly marked as surface)
        for key, data in face_map.items():
            if data['count'] == 1 or data.get('is_surface'):
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
                
                if q_sicn is not None:
                    faces_with_quality += 1
                else:
                    faces_without_quality += 1

        print(f"[MESH PARSE DEBUG] Boundary faces extracted: {boundary_face_count}")
        print(f"[MESH PARSE DEBUG] Faces WITH quality data: {faces_with_quality}")
        unique_ent_tags = set(entity_tags)
        print(f"[MESH PARSE DEBUG] Unique Entity Tags found: {len(unique_ent_tags)}")
        print(f"[MESH PARSE DEBUG] Sample tags: {list(unique_ent_tags)[:20]}")
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
                        if 'cfd' in qdata:
                            quality_summary['cfd'] = qdata['cfd']
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

# Note: cleanup_stuck_jobs(app) is called within create_app() in development.
if __name__ == '__main__':
    print("=" * 70)
    print("MESH GENERATION API SERVER v2.0")
    print("With JWT Authentication & SQLAlchemy Database")
    print("=" * 70)
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Output folder: {app.config['OUTPUT_FOLDER']}")
    print(f"JWT_SECRET_KEY: {'[SET]' if app.config.get('JWT_SECRET_KEY') else '[NOT SET - USING DEFAULT]'}")
    print(f"Docker mode: {os.environ.get('IS_DOCKER_CONTAINER', 'false')}")
    print(f"WebSocket Message Queue: {app.config.get('SOCKETIO_MESSAGE_QUEUE', 'redis://localhost:6379/1')}")
    print("\nStarting server on http://localhost:5000")
    print("WebSocket endpoint: ws://localhost:5000")
    print("=" * 70)

    # Use SocketIO's run method if available, otherwise fall back to Flask's run
    if hasattr(app, 'socketio') and app.socketio:
        try:
            app.socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
        except Exception as e:
            print(f"[WARNING] SocketIO run failed: {e}, falling back to Flask run")
            app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
    else:
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
