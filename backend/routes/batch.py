"""
Batch Processing API Routes

Endpoints for batch mesh generation:
- POST /api/batch/create - Create a new batch session
- POST /api/batch/<id>/upload - Upload files to batch
- GET /api/batch/<id> - Get batch status and progress
- POST /api/batch/<id>/start - Start batch processing
- POST /api/batch/<id>/cancel - Cancel batch processing
- GET /api/batch/<id>/download - Download all meshes as ZIP
- GET /api/batches - List user's batches
"""

import os
import io
import re
import zipfile
import hashlib
import tempfile
from pathlib import Path
from datetime import datetime
from flask import Blueprint, request, jsonify, send_file, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from werkzeug.utils import secure_filename

from models import (
    db, User, Batch, BatchFile, BatchJob, 
    MESH_PRESETS, create_batch_jobs_for_file
)
from storage import get_storage
from middleware.rate_limit import check_batch_quota

batch_bp = Blueprint('batch', __name__, url_prefix='/api/batch')

# UUID validation regex
UUID_REGEX = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.I)

# Store SocketIO instance
socketio_instance = None

def init_socketio(socketio):
    """Initialize SocketIO instance for use in routes"""
    global socketio_instance
    socketio_instance = socketio

def validate_uuid(uuid_string):
    """Validate UUID format to prevent injection attacks"""
    return bool(UUID_REGEX.match(uuid_string))

def emit_batch_progress(batch_id, job_id, progress, status, message=None):
    """
    Emit progress update via WebSocket for batch jobs
    """
    if socketio_instance:
        try:
            socketio_instance.emit('job_progress', {
                'batch_id': batch_id,
                'job_id': job_id,
                'progress': progress,
                'status': status,
                'message': message,
                'timestamp': datetime.utcnow().isoformat()
            }, room=f'batch_{batch_id}')
        except Exception as e:
            print(f"[BATCH WS] Error emitting progress: {e}")
    else:
        print(f"[BATCH] {status}: {message or ''} ({progress}%)")

def emit_batch_status(batch_id, status, totals=None):
    """
    Emit batch-level status update
    """
    if socketio_instance:
        try:
            payload = {
                'batch_id': batch_id,
                'status': status,
                'timestamp': datetime.utcnow().isoformat()
            }
            if totals:
                payload.update(totals)
            socketio_instance.emit('batch_status', payload, room=f'batch_{batch_id}')
        except Exception as e:
            print(f"[BATCH WS] Error emitting batch status: {e}")


@batch_bp.route('/create', methods=['POST'])
@jwt_required()
def create_batch():
    """
    Create a new batch session.
    
    Request body:
    {
        "name": "Optional batch name",
        "mesh_independence": true/false,
        "mesh_strategy": "Tet (Fast)",  # Uses HXT algorithm (tet_delaunay_optimized) - fast single-pass
        "curvature_adaptive": true
    }
    
    Returns:
    {
        "batch_id": "uuid",
        "status": "pending"
    }
    """
    current_user_id = int(get_jwt_identity())
    user = User.query.get(current_user_id)
    
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    data = request.get_json() or {}
    
    # Validate and sanitize batch name
    batch_name = data.get('name', '')
    if batch_name:
        batch_name = batch_name.strip()[:255]  # Limit length
        # Remove potentially dangerous characters
        batch_name = ''.join(c for c in batch_name if c.isalnum() or c in ' -_.')
    
    batch = Batch(
        user_id=current_user_id,
        name=batch_name or None,
        mesh_independence=data.get('mesh_independence', False),
        mesh_strategy=data.get('mesh_strategy', 'Tet (Fast)'),
        curvature_adaptive=data.get('curvature_adaptive', True),
        parallel_limit=current_app.config.get('BATCH_PARALLEL_JOBS', 6),
        status='pending'
    )
    
    db.session.add(batch)
    db.session.commit()
    
    return jsonify({
        "batch_id": batch.id,
        "status": batch.status,
        "mesh_independence": batch.mesh_independence
    }), 201


@batch_bp.route('/<batch_id>/upload', methods=['POST'])
@jwt_required()
def upload_batch_files(batch_id):
    """
    Upload one or more files to a batch.
    
    Request: multipart/form-data with 'files' field containing multiple files
    
    Returns:
    {
        "uploaded": 3,
        "files": [...],
        "total_jobs": 9  // if mesh_independence=True, 3 files × 3 presets = 9 jobs
    }
    """
    # Validate UUID format
    if not validate_uuid(batch_id):
        return jsonify({"error": "Invalid batch ID format"}), 400
    
    current_user_id = int(get_jwt_identity())
    
    batch = Batch.query.get(batch_id)
    if not batch:
        return jsonify({"error": "Batch not found"}), 404
    
    if batch.user_id != current_user_id:
        return jsonify({"error": "Access denied"}), 403
    
    if batch.status not in ['pending', 'uploading']:
        return jsonify({"error": f"Cannot upload to batch (status: {batch.status})"}), 400
    
    if 'files' not in request.files:
        return jsonify({"error": "No files provided"}), 400
    
    files = request.files.getlist('files')
    if not files:
        return jsonify({"error": "No files selected"}), 400
    
    # Check batch limits
    max_files = current_app.config.get('BATCH_MAX_FILES', 10)
    current_count = BatchFile.query.filter_by(batch_id=batch_id).count()
    
    if current_count + len(files) > max_files:
        return jsonify({
            "error": f"Batch limit exceeded. Max {max_files} files. Current: {current_count}, Trying to add: {len(files)}"
        }), 400
    
    user = User.query.get(current_user_id)
    storage = get_storage()
    max_file_size = current_app.config.get('BATCH_MAX_FILE_SIZE', 500 * 1024 * 1024)
    allowed_extensions = current_app.config.get('ALLOWED_EXTENSIONS', {'.step', '.stp', '.stl', '.msh', '.iges', '.igs', '.brep', '.obj', '.vtk'})
    
    batch.status = 'uploading'
    db.session.commit()
    
    uploaded_files = []
    errors = []
    
    for file in files:
        if not file or file.filename == '':
            continue
        
        # Validate extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            errors.append(f"{file.filename}: Invalid file type")
            continue
        
        # Check file size
        file.seek(0, 2)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > max_file_size:
            errors.append(f"{file.filename}: File too large (max {max_file_size // (1024*1024)}MB)")
            continue
        
        try:
            # Create batch file record
            original_filename = file.filename
            filename = secure_filename(file.filename) or f"file{file_ext}"
            
            # Calculate hash
            file_content = file.read()
            file_hash = hashlib.sha256(file_content).hexdigest()
            file.seek(0)
            
            batch_file = BatchFile(
                batch_id=batch_id,
                filename=filename,
                original_filename=original_filename,
                file_size=file_size,
                file_type=file_ext.lstrip('.'),
                file_hash=file_hash,
                status='uploading'
            )
            db.session.add(batch_file)
            db.session.flush()  # Get ID before saving file
            
            # Save file
            storage_filename = f"{batch_file.id}_{filename}"
            filepath = storage.save_file(
                file_data=file,
                filename=storage_filename,
                user_email=user.email,
                file_type='uploads'
            )
            
            batch_file.storage_path = filepath
            batch_file.status = 'uploaded'
            batch_file.uploaded_at = datetime.utcnow()
            
            # Create mesh jobs based on independence setting
            jobs = create_batch_jobs_for_file(batch_file, batch, batch.mesh_independence)
            
            db.session.commit()
            
            uploaded_files.append({
                'id': batch_file.id,
                'filename': original_filename,
                'file_size': file_size,
                'jobs_created': len(jobs)
            })
            
        except Exception as e:
            db.session.rollback()
            errors.append(f"{file.filename}: {str(e)}")
    
    # Update batch totals
    batch.total_files = BatchFile.query.filter_by(batch_id=batch_id).count()
    batch.total_jobs = BatchJob.query.filter_by(batch_id=batch_id).count()
    
    # Set status to ready if files uploaded successfully
    if uploaded_files:
        batch.status = 'ready'
    elif not errors:
        batch.status = 'pending'
    
    db.session.commit()
    
    response = {
        "uploaded": len(uploaded_files),
        "files": uploaded_files,
        "total_files": batch.total_files,
        "total_jobs": batch.total_jobs,
        "batch_status": batch.status
    }
    
    if errors:
        response["errors"] = errors
    
    return jsonify(response)


@batch_bp.route('/file/<file_id>/preview', methods=['GET'])
@jwt_required()
def get_batch_file_preview(file_id):
    """
    Get CAD preview for a batch file.
    Returns triangulated preview data for 3D visualization.
    """
    import tempfile
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from api_server import parse_step_file_for_preview
    
    if not validate_uuid(file_id):
        return jsonify({"error": "Invalid file ID format"}), 400
    
    current_user_id = int(get_jwt_identity())
    
    # Get the batch file
    batch_file = BatchFile.query.get(file_id)
    if not batch_file:
        return jsonify({"error": "File not found"}), 404
    
    # Verify ownership via batch
    batch = Batch.query.get(batch_file.batch_id)
    if not batch or batch.user_id != current_user_id:
        return jsonify({"error": "Access denied"}), 403
    
    # Get file path
    filepath = batch_file.storage_path
    if not filepath:
        return jsonify({"error": "CAD file not found"}), 404
    
    try:
        storage = get_storage()
        use_s3 = current_app.config.get('USE_S3', False)
        
        # If S3, download to temp file for parsing
        if use_s3 and filepath.startswith('s3://'):
            file_ext = Path(batch_file.original_filename).suffix
            local_temp = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
            storage.download_to_local(filepath, local_temp.name)
            preview_data = parse_step_file_for_preview(local_temp.name)
            Path(local_temp.name).unlink(missing_ok=True)
        else:
            if not Path(filepath).exists():
                return jsonify({"error": "CAD file not found"}), 404
            preview_data = parse_step_file_for_preview(filepath)
        
        return jsonify(preview_data)
    except Exception as e:
        return jsonify({"error": f"Failed to generate preview: {str(e)}"}), 500


@batch_bp.route('/<batch_id>', methods=['GET'])
@jwt_required()
def get_batch_status(batch_id):
    """
    Get detailed batch status and progress.
    
    Query params:
    - include_files=true/false (default: true)
    - include_jobs=true/false (default: false)
    
    Returns full batch object with files and optionally jobs.
    """
    if not validate_uuid(batch_id):
        return jsonify({"error": "Invalid batch ID format"}), 400
    
    current_user_id = int(get_jwt_identity())
    
    batch = Batch.query.get(batch_id)
    if not batch:
        return jsonify({"error": "Batch not found"}), 404
    
    if batch.user_id != current_user_id:
        return jsonify({"error": "Access denied"}), 403
    
    include_files = request.args.get('include_files', 'true').lower() == 'true'
    include_jobs = request.args.get('include_jobs', 'false').lower() == 'true'
    
    return jsonify(batch.to_dict(include_files=include_files, include_jobs=include_jobs))


@batch_bp.route('/<batch_id>/start', methods=['POST'])
@jwt_required()
def start_batch(batch_id):
    """
    Start processing all jobs in the batch.
    
    In production: Uses Celery for parallel processing
    In development: Uses threading for sequential processing
    """
    if not validate_uuid(batch_id):
        return jsonify({"error": "Invalid batch ID format"}), 400
    
    current_user_id = int(get_jwt_identity())
    
    batch = Batch.query.get(batch_id)
    if not batch:
        return jsonify({"error": "Batch not found"}), 404
    
    if batch.user_id != current_user_id:
        return jsonify({"error": "Access denied"}), 403
    
    if batch.status != 'ready':
        return jsonify({"error": f"Batch not ready to start (status: {batch.status})"}), 400
    
    if batch.total_jobs == 0:
        return jsonify({"error": "No jobs in batch"}), 400
    
    # Check quota before starting batch
    allowed, current_count, quota, remaining = check_batch_quota(current_user_id, batch.total_jobs)
    if not allowed:
        return jsonify({
            "error": "Daily job quota exceeded",
            "quota": quota,
            "used": current_count,
            "batch_jobs": batch.total_jobs,
            "remaining": remaining,
            "message": f"Batch would exceed daily limit. You have {remaining} jobs remaining today, but batch contains {batch.total_jobs} jobs."
        }), 429
    
    # Update batch status
    batch.status = 'processing'
    batch.started_at = datetime.utcnow()
    db.session.commit()
    
    # Try Celery first, fall back to sync processing for local dev
    try:
        from tasks import start_batch_processing
        task = start_batch_processing.delay(batch_id)
        return jsonify({
            "message": "Batch processing started (Celery)",
            "batch_id": batch_id,
            "total_jobs": batch.total_jobs,
            "task_id": task.id
        })
    except ImportError:
        # Celery not available - use local processing
        import threading
        
        def process_batch_locally(app, bid):
            """Process batch jobs locally without Celery"""
            import sys
            import traceback
            print(f"[BATCH] Thread started for batch {bid[:8]}", flush=True)
            
            try:
                with app.app_context():
                    from models import Batch, BatchJob
                    from storage import get_storage
                    from middleware.rate_limit import create_job_usage_record, update_job_usage_status
                    import subprocess
                    
                    batch = Batch.query.get(bid)
                    if not batch:
                        print(f"[BATCH] Error: Batch {bid[:8]} not found", flush=True)
                        return
                    
                    jobs = BatchJob.query.filter_by(batch_id=bid, status='pending').all()
                    total_jobs = len(jobs)
                    print(f"[BATCH] Found {total_jobs} pending jobs", flush=True)
                    storage = get_storage()
                    
                    # Process jobs SEQUENTIALLY using HXT (tet_delaunay_optimized)
                    # This is faster than parallel internal strategy search because:
                    # 1. HXT is already parallelized internally by Gmsh
                    # 2. Avoids memory overhead of multiple strategy workers
                    # 3. Sequential processing = predictable resource usage
                    # 4. Each job gets full CPU for Gmsh's internal parallelization
                    for job_idx, job in enumerate(jobs):
                        # Check if batch was cancelled
                        db.session.refresh(batch)
                        if batch.status == 'cancelled':
                            print(f"[BATCH] Batch cancelled, stopping processing", flush=True)
                            break
                        
                        print(f"[BATCH] === Starting job {job_idx + 1}/{total_jobs} ===", flush=True)
                        try:
                            job.status = 'processing'
                            job.started_at = datetime.utcnow()
                            db.session.commit()
                            
                            # Create JobUsage record
                            job_usage = create_job_usage_record(
                                user_id=batch.user_id,
                                job_type='batch_job',
                                job_id=job.id, # Using model ID as tracking ID for local
                                batch_id=bid,
                                batch_job_id=job.id,
                                compute_backend='local',
                                status='processing'
                            )
                            
                            emit_batch_progress(bid, job.id, 0, 'processing', f'Starting job {job_idx + 1}/{total_jobs}')
                            
                            # Get the source file
                            batch_file = job.source_file
                            if not batch_file:
                                job.status = 'failed'
                                job.error_message = 'Source file not found'
                                db.session.commit()
                                continue
                            
                            # Get file from storage
                            # For local storage, storage_path IS the full path
                            # For S3, download to temp file first
                            use_s3 = app.config.get('USE_S3', False)
                            input_path = batch_file.storage_path
                            local_temp_file = None
                            
                            if use_s3 and input_path and input_path.startswith('s3://'):
                                # Download S3 file to local temp directory
                                file_ext = Path(batch_file.original_filename).suffix
                                local_temp_dir = Path(tempfile.mkdtemp(prefix='meshgen_batch_'))
                                local_temp_file = local_temp_dir / f"{batch_file.id}{file_ext}"
                                try:
                                    print(f"[BATCH] Downloading from S3: {input_path}", flush=True)
                                    storage.download_to_local(input_path, str(local_temp_file))
                                    input_path = str(local_temp_file)
                                    print(f"[BATCH] Downloaded to: {input_path}", flush=True)
                                except Exception as e:
                                    job.status = 'failed'
                                    job.error_message = f'Failed to download from S3: {str(e)}'
                                    print(f"[BATCH] Error downloading from S3: {e}", flush=True)
                                    db.session.commit()
                                    continue
                            
                            if not input_path or not os.path.exists(input_path):
                                job.status = 'failed'
                                job.error_message = f'Input file not found: {batch_file.storage_path}'
                                print(f"[BATCH] Error: Input file not found: {input_path}", flush=True)
                                db.session.commit()
                                continue
                            
                            print(f"[BATCH] Processing job {job.id[:8]} - {batch_file.original_filename} ({job.quality_preset})", flush=True)
                            
                            # Create output directory - use app.config NOT current_app.config in thread
                            output_dir = Path(app.config['OUTPUT_FOLDER'])
                            output_dir.mkdir(parents=True, exist_ok=True)
                            
                            # Build quality params JSON
                            import json as json_module
                            
                            # Get preset sizing based on quality_preset
                            from models import MESH_PRESETS
                            preset = MESH_PRESETS.get(job.quality_preset, MESH_PRESETS['medium'])
                            
                            quality_params = {
                                'mesh_strategy': job.mesh_strategy or 'Tet (Fast)',
                                'quality_preset': job.quality_preset,
                                'target_elements': job.target_elements or preset['target_elements'],
                                'max_size_mm': job.max_element_size or preset['max_element_size'],
                                'min_size_mm': (job.max_element_size or preset['max_element_size']) / 10.0,  # Min = 1/10 of max
                                'curvature_adaptive': job.curvature_adaptive if job.curvature_adaptive is not None else preset['curvature_adaptive'],
                                'output_prefix': f"{Path(batch_file.original_filename).stem}_{job.quality_preset}"
                            }
                            
                            print(f"[BATCH] Job params: preset={job.quality_preset}, max_size={quality_params['max_size_mm']}, target={quality_params['target_elements']}", flush=True)
                            
                            # Run mesh worker subprocess
                            mesh_worker = Path(__file__).parent.parent.parent / 'apps' / 'cli' / 'mesh_worker_subprocess.py'
                            
                            cmd = [
                                sys.executable,
                                str(mesh_worker),
                                str(input_path),
                                str(output_dir),
                                '--quality-params', json_module.dumps(quality_params)
                            ]
                            
                            print(f"[BATCH] Running subprocess...", flush=True)
                            print(f"[BATCH]   Input: {input_path}", flush=True)
                            print(f"[BATCH]   Output dir: {output_dir}", flush=True)
                            print(f"[BATCH]   Params: {json_module.dumps(quality_params)}", flush=True)
                            
                            # Use Popen for real-time output streaming
                            import time
                            
                            # Force Python unbuffered output in subprocess
                            env = os.environ.copy()
                            env['PYTHONUNBUFFERED'] = '1'
                            
                            process = subprocess.Popen(
                                cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                                text=True,
                                bufsize=1,  # Line buffered
                                env=env
                            )
                            
                            stdout_lines = []
                            last_log_time = time.time()
                            start_time = time.time()
                            timeout = 1200  # 20 minute timeout for complex geometries
                            
                            print(f"[BATCH] Subprocess PID: {process.pid}", flush=True)
                            
                            # Stream output in real-time with select for non-blocking
                            import select
                            
                            while True:
                                # Check timeout
                                elapsed = time.time() - start_time
                                if elapsed > timeout:
                                    process.kill()
                                    print(f"[BATCH] ⚠ Process killed after {timeout}s timeout", flush=True)
                                    break
                                
                                # Check if process has finished
                                ret = process.poll()
                                if ret is not None:
                                    # Process finished - read any remaining output
                                    remaining = process.stdout.read()
                                    if remaining:
                                        for line in remaining.strip().split('\n'):
                                            stdout_lines.append(line)
                                    print(f"[BATCH] Process exited with code {ret}", flush=True)
                                    break
                                
                                # Use select to check if data is available (with 1s timeout)
                                try:
                                    ready, _, _ = select.select([process.stdout], [], [], 1.0)
                                except (ValueError, OSError):
                                    # Pipe closed
                                    break
                                
                                if ready:
                                    line = process.stdout.readline()
                                    if line:
                                        line = line.rstrip()
                                        stdout_lines.append(line)
                                        # Log important lines
                                        if any(x in line for x in ['[OK]', 'Error', 'WARNING', 'BEST', 'Success', 'score:', 'Generated', 'Starting', 'Loading', 'Mesh', 'Strategy', 'FAST-TET', 'SICN', 'Gamma', 'Elements']):
                                            print(f"[BATCH]   {line}", flush=True)
                                else:
                                    # No data available - log heartbeat every 10 seconds
                                    if time.time() - last_log_time > 10:
                                        print(f"[BATCH] ... waiting for output ({int(elapsed)}s elapsed)", flush=True)
                                        last_log_time = time.time()
                                        emit_batch_progress(bid, job.id, min(95, 10 + int(elapsed/10)), 'processing', 'Generating mesh...')
                            
                            # Get return code
                            return_code = process.poll()
                            stdout_text = '\n'.join(stdout_lines)
                            elapsed = int(time.time() - start_time)
                            print(f"[BATCH] Process completed in {elapsed}s with {len(stdout_lines)} output lines", flush=True)
                            
                            print(f"[BATCH] Exit code: {return_code}", flush=True)
                            
                            # Create a result-like object for compatibility
                            class Result:
                                pass
                            result = Result()
                            result.returncode = return_code
                            result.stdout = stdout_text
                            result.stderr = ""
                            
                            # Parse result JSON from stdout
                            output_path = None
                            if result.returncode == 0 and result.stdout:
                                try:
                                    mesh_result = json_module.loads(result.stdout.strip().split('\n')[-1])
                                    # mesh worker returns 'output_file', not 'mesh_file'
                                    if mesh_result.get('success') and mesh_result.get('output_file'):
                                        output_path = Path(mesh_result['output_file'])
                                        
                                        # Extract metrics - worker returns nested structure
                                        metrics = mesh_result.get('metrics', {})
                                        job.element_count = metrics.get('total_elements') or mesh_result.get('total_elements')
                                        job.node_count = metrics.get('total_nodes') or mesh_result.get('total_nodes')
                                        job.score = mesh_result.get('score')
                                        job.quality_metrics = mesh_result.get('quality_metrics')
                                except Exception as parse_err:
                                    print(f"[BATCH] JSON parse error: {parse_err}", flush=True)
                                    # Try to find any .msh file in output dir
                                    pattern = f"*{job.quality_preset}*.msh"
                                    msh_files = list(output_dir.glob(pattern))
                                    if msh_files:
                                        output_path = msh_files[0]
                            
                            if output_path and output_path.exists():
                                job.status = 'completed'
                                job.output_file_size = output_path.stat().st_size
                                job.completed_at = datetime.utcnow()
                                
                                # Upload to S3 if configured
                                if use_s3:
                                    try:
                                        # Get user email for S3 folder structure
                                        user_email = batch.user.email if batch.user else None
                                        mesh_filename = f"{batch_file.id}_{job.quality_preset}_mesh.msh"
                                        s3_path = storage.save_local_file(
                                            local_path=str(output_path),
                                            filename=mesh_filename,
                                            user_email=user_email,
                                            file_type='mesh'
                                        )
                                        job.output_path = s3_path
                                        print(f"[BATCH] Uploaded mesh to S3: {s3_path}", flush=True)
                                        
                                        # Also upload quality file if it exists
                                        quality_path = output_path.with_suffix('.quality.json')
                                        if quality_path.exists():
                                            storage.save_local_file(
                                                local_path=str(quality_path),
                                                filename=quality_path.name,
                                                user_email=user_email,
                                                file_type='mesh'
                                            )
                                    except Exception as s3_err:
                                        print(f"[BATCH] Warning: Failed to upload to S3: {s3_err}", flush=True)
                                        job.output_path = str(output_path)  # Fallback to local path
                                else:
                                    job.output_path = str(output_path)
                                
                                # Calculate processing time
                                if job.started_at:
                                    job.processing_time = (job.completed_at - job.started_at).total_seconds()
                                print(f"[BATCH] ✓ Job completed: {job.output_path}", flush=True)
                                
                                # Update tracking status
                                update_job_usage_status(job.id, 'completed', 'local')
                                
                                emit_batch_progress(bid, job.id, 100, 'completed', f'Job {job_idx + 1} completed successfully')
                                
                                # Update batch counts
                                batch.completed_jobs = BatchJob.query.filter_by(
                                    batch_id=bid, status='completed'
                                ).count()
                            else:
                                error_msg = result.stderr[:500] if result.stderr else f'Mesh generation failed (exit code: {result.returncode})'
                                job.status = 'failed'
                                job.error_message = error_msg
                                print(f"[BATCH] ✗ Job failed: {error_msg}", flush=True)
                                batch.failed_jobs = BatchJob.query.filter_by(
                                    batch_id=bid, status='failed'
                                ).count()
                            
                            db.session.commit()
                            
                            # Cleanup temp S3 download file
                            if local_temp_file and Path(local_temp_file).exists():
                                try:
                                    Path(local_temp_file).unlink()
                                    Path(local_temp_file).parent.rmdir()
                                except Exception:
                                    pass
                            
                            print(f"[BATCH] === Job {job_idx + 1}/{total_jobs} complete, continuing... ===", flush=True)
                            
                        except Exception as e:
                            import traceback as tb
                            job.status = 'failed'
                            job.error_message = str(e)[:500]
                            # Update tracking status on exception
                            update_job_usage_status(job.id, 'failed', 'local')
                            
                            print(f"[BATCH] Job exception: {e}", flush=True)
                            tb.print_exc()
                            batch.failed_jobs = BatchJob.query.filter_by(
                                batch_id=bid, status='failed'
                            ).count()
                            db.session.commit()
                            
                            # Cleanup temp S3 download file on error too
                            if 'local_temp_file' in locals() and local_temp_file and Path(local_temp_file).exists():
                                try:
                                    Path(local_temp_file).unlink()
                                    Path(local_temp_file).parent.rmdir()
                                except Exception:
                                    pass
                            
                            print(f"[BATCH] === Job {job_idx + 1}/{total_jobs} failed, continuing... ===", flush=True)
                    
                    print(f"[BATCH] === All {total_jobs} jobs processed ===", flush=True)
                
                    # Update final batch status
                    batch = Batch.query.get(bid)
                    if batch.failed_jobs == batch.total_jobs:
                        batch.status = 'failed'
                    elif batch.completed_jobs == batch.total_jobs:
                        batch.status = 'completed'
                        batch.completed_at = datetime.utcnow()
                    elif batch.completed_jobs + batch.failed_jobs == batch.total_jobs:
                        batch.status = 'completed'  # partial completion
                        batch.completed_at = datetime.utcnow()
                    
                    db.session.commit()
                    print(f"[BATCH] Batch {bid[:8]} completed. Status: {batch.status}", flush=True)
                    
                    emit_batch_status(bid, batch.status, {
                        'completed_jobs': batch.completed_jobs,
                        'failed_jobs': batch.failed_jobs,
                        'total_jobs': batch.total_jobs
                    })
            except Exception as e:
                print(f"[BATCH] CRITICAL ERROR in thread: {e}", flush=True)
                traceback.print_exc()
        
        # Start background thread
        thread = threading.Thread(
            target=process_batch_locally,
            args=(current_app._get_current_object(), batch_id)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "message": "Batch processing started (local mode)",
            "batch_id": batch_id,
            "total_jobs": batch.total_jobs
        })


@batch_bp.route('/<batch_id>/cancel', methods=['POST'])
@jwt_required()
def cancel_batch_route(batch_id):
    """
    Cancel all pending/queued jobs in the batch.
    Running jobs will complete but no new jobs will start.
    """
    if not validate_uuid(batch_id):
        return jsonify({"error": "Invalid batch ID format"}), 400
    
    current_user_id = int(get_jwt_identity())
    
    batch = Batch.query.get(batch_id)
    if not batch:
        return jsonify({"error": "Batch not found"}), 404
    
    if batch.user_id != current_user_id:
        return jsonify({"error": "Access denied"}), 403
    
    if batch.status not in ['ready', 'processing']:
        return jsonify({"error": f"Cannot cancel batch (status: {batch.status})"}), 400
    
    try:
        from tasks import cancel_batch
        
        # Queue the cancel task
        result = cancel_batch.delay(batch_id)
        
        return jsonify({
            "message": "Batch cancellation requested",
            "batch_id": batch_id
        })
        
    except Exception as e:
        # Fallback: cancel directly without Celery
        batch.status = 'cancelled'
        batch.completed_at = datetime.utcnow()
        
        BatchJob.query.filter(
            BatchJob.batch_id == batch_id,
            BatchJob.status.in_(['pending', 'queued'])
        ).update({'status': 'cancelled', 'completed_at': datetime.utcnow()})
        
        db.session.commit()
        
        return jsonify({
            "message": "Batch cancelled",
            "batch_id": batch_id
        })


@batch_bp.route('/<batch_id>/download', methods=['GET'])
@jwt_required()
def download_batch_meshes(batch_id):
    """
    Download all completed meshes as a ZIP file.
    
    ZIP structure:
    - batch_name/
      - FileA/
        - FileA_coarse_mesh.msh
        - FileA_medium_mesh.msh
        - FileA_fine_mesh.msh
      - FileB/
        - FileB_medium_mesh.msh
    """
    if not validate_uuid(batch_id):
        return jsonify({"error": "Invalid batch ID format"}), 400
    
    current_user_id = int(get_jwt_identity())
    
    batch = Batch.query.get(batch_id)
    if not batch:
        return jsonify({"error": "Batch not found"}), 404
    
    if batch.user_id != current_user_id:
        return jsonify({"error": "Access denied"}), 403
    
    if batch.completed_jobs == 0:
        return jsonify({"error": "No completed meshes to download"}), 400
    
    # Get completed jobs with their files
    completed_jobs = BatchJob.query.filter_by(
        batch_id=batch_id,
        status='completed'
    ).all()
    
    if not completed_jobs:
        return jsonify({"error": "No completed meshes"}), 400
    
    storage = get_storage()
    use_s3 = current_app.config.get('USE_S3', False)
    
    # Create ZIP in memory
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        batch_name = batch.name or f"batch_{batch.id[:8]}"
        
        for job in completed_jobs:
            if not job.output_path:
                continue
            
            source_file = BatchFile.query.get(job.file_id)
            if not source_file:
                continue
            
            # Get file content
            try:
                output_path = Path(job.output_path)
                
                # Security: Validate path is within allowed directories
                if not use_s3:
                    allowed_dirs = [
                        Path(current_app.config['OUTPUT_FOLDER']).resolve(),
                        Path(current_app.config['UPLOAD_FOLDER']).resolve(),
                        Path('/tmp').resolve()
                    ]
                    try:
                        resolved = output_path.resolve()
                        if not any(str(resolved).startswith(str(d)) for d in allowed_dirs):
                            print(f"[DOWNLOAD] Security: Path outside allowed dirs: {job.output_path}")
                            continue
                    except Exception:
                        continue
                
                if use_s3 and job.output_path.startswith('s3://'):
                    # Download from S3 to temp file
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.msh')
                    storage.download_to_local(job.output_path, temp_file.name)
                    with open(temp_file.name, 'rb') as f:
                        mesh_content = f.read()
                    Path(temp_file.name).unlink(missing_ok=True)
                else:
                    # Read local file
                    with open(job.output_path, 'rb') as f:
                        mesh_content = f.read()
                
                # Create path in ZIP
                file_stem = Path(source_file.original_filename).stem
                if batch.mesh_independence:
                    zip_path = f"{batch_name}/{file_stem}/{file_stem}_{job.quality_preset}_mesh.msh"
                else:
                    zip_path = f"{batch_name}/{file_stem}_mesh.msh"
                
                zip_file.writestr(zip_path, mesh_content)
                
            except Exception as e:
                print(f"[DOWNLOAD] Failed to add {job.output_path}: {e}")
                continue
    
    zip_buffer.seek(0)
    
    # Generate filename
    download_name = f"{batch_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    
    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name=download_name
    )


@batch_bp.route('/<batch_id>/jobs', methods=['GET'])
@jwt_required()
def get_batch_jobs(batch_id):
    """
    Get all jobs for a batch with their current status.
    
    Query params:
    - status: filter by status (pending, queued, processing, completed, failed)
    - file_id: filter by source file
    """
    if not validate_uuid(batch_id):
        return jsonify({"error": "Invalid batch ID format"}), 400
    
    current_user_id = int(get_jwt_identity())
    
    batch = Batch.query.get(batch_id)
    if not batch:
        return jsonify({"error": "Batch not found"}), 404
    
    if batch.user_id != current_user_id:
        return jsonify({"error": "Access denied"}), 403
    
    query = BatchJob.query.filter_by(batch_id=batch_id)
    
    status_filter = request.args.get('status')
    if status_filter:
        query = query.filter_by(status=status_filter)
    
    file_filter = request.args.get('file_id')
    if file_filter:
        query = query.filter_by(file_id=file_filter)
    
    jobs = query.order_by(BatchJob.created_at).all()
    
    return jsonify({
        "batch_id": batch_id,
        "total": len(jobs),
        "jobs": [job.to_dict() for job in jobs]
    })


@batch_bp.route('/list', methods=['GET'])
@jwt_required()
def list_batches():
    """
    List all batches for the current user.
    
    Query params:
    - status: filter by status
    - limit: max results (default 20)
    - offset: pagination offset
    """
    current_user_id = int(get_jwt_identity())
    
    query = Batch.query.filter_by(user_id=current_user_id)
    
    status_filter = request.args.get('status')
    if status_filter:
        query = query.filter_by(status=status_filter)
    
    limit = request.args.get('limit', 20, type=int)
    offset = request.args.get('offset', 0, type=int)
    
    total = query.count()
    batches = query.order_by(Batch.created_at.desc()).offset(offset).limit(limit).all()
    
    return jsonify({
        "total": total,
        "limit": limit,
        "offset": offset,
        "batches": [b.to_dict() for b in batches]
    })


@batch_bp.route('/<batch_id>', methods=['DELETE'])
@jwt_required()
def delete_batch(batch_id):
    """
    Delete a batch and all its files and jobs.
    """
    if not validate_uuid(batch_id):
        return jsonify({"error": "Invalid batch ID format"}), 400
    
    current_user_id = int(get_jwt_identity())
    
    batch = Batch.query.get(batch_id)
    if not batch:
        return jsonify({"error": "Batch not found"}), 404
    
    if batch.user_id != current_user_id:
        return jsonify({"error": "Access denied"}), 403
    
    if batch.status == 'processing':
        return jsonify({"error": "Cannot delete batch while processing"}), 400
    
    try:
        storage = get_storage()
        use_s3 = current_app.config.get('USE_S3', False)
        
        # Delete files from storage
        for batch_file in batch.files.all():
            if batch_file.storage_path:
                try:
                    if use_s3 and batch_file.storage_path.startswith('s3://'):
                        storage.delete_file(batch_file.storage_path)
                    elif Path(batch_file.storage_path).exists():
                        Path(batch_file.storage_path).unlink()
                except Exception as e:
                    print(f"[DELETE] Failed to delete file: {e}")
        
        # Delete output meshes
        for job in batch.jobs.all():
            if job.output_path:
                try:
                    if use_s3 and job.output_path.startswith('s3://'):
                        storage.delete_file(job.output_path)
                    elif Path(job.output_path).exists():
                        Path(job.output_path).unlink()
                except Exception as e:
                    print(f"[DELETE] Failed to delete mesh: {e}")
        
        # Delete batch (cascades to files and jobs)
        db.session.delete(batch)
        db.session.commit()
        
        return jsonify({"message": "Batch deleted", "batch_id": batch_id})
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500
