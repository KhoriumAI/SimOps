"""
Celery Tasks for Batch Mesh Processing

This module contains async tasks for:
- Individual mesh generation jobs
- Batch coordination
- Progress updates via WebSocket
"""

import os
import sys
import json
import time
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime
from celery import shared_task, current_task
from celery.exceptions import SoftTimeLimitExceeded

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_flask_app():
    """Lazy import Flask app to avoid circular imports"""
    from api_server import create_app
    return create_app()


def emit_progress(batch_id, job_id, progress, status, message=None):
    """
    Emit progress update via WebSocket (if SocketIO is available)
    Falls back to logging if WebSocket not configured
    """
    try:
        from flask_socketio import SocketIO
        from celery_app import celery
        
        socketio = SocketIO(message_queue=celery.conf.get('SOCKETIO_MESSAGE_QUEUE'))
        socketio.emit('job_progress', {
            'batch_id': batch_id,
            'job_id': job_id,
            'progress': progress,
            'status': status,
            'message': message,
            'timestamp': datetime.utcnow().isoformat()
        }, room=f'batch_{batch_id}')
    except Exception as e:
        # Fallback to logging
        print(f"[TASK] Batch {batch_id} Job {job_id}: {progress}% - {status} - {message}")


@shared_task(bind=True, max_retries=2)
def process_mesh_job(self, job_id: str, batch_id: str):
    """
    Process a single mesh generation job.
    
    This task runs the mesh_worker_subprocess.py script for a single file
    with the specified quality parameters.
    
    Args:
        job_id: UUID of the BatchJob
        batch_id: UUID of the parent Batch
    
    Returns:
        dict with job results or error info
    """
    app = get_flask_app()
    start_time = time.time()
    
    with app.app_context():
        from models import db, BatchJob, BatchFile, Batch, User
        from storage import get_storage
        
        job = BatchJob.query.get(job_id)
        if not job:
            return {'success': False, 'error': 'Job not found'}
        
        batch = Batch.query.get(batch_id)
        source_file = BatchFile.query.get(job.file_id)
        user = User.query.get(batch.user_id) if batch else None
        
        if not source_file:
            job.status = 'failed'
            job.error_message = 'Source file not found'
            db.session.commit()
            return {'success': False, 'error': 'Source file not found'}
        
        try:
            # Update job status
            job.status = 'processing'
            job.started_at = datetime.utcnow()
            job.celery_task_id = self.request.id
            db.session.commit()
            
            emit_progress(batch_id, job_id, 0, 'processing', f'Starting mesh generation for {source_file.original_filename}')
            
            # Get paths
            worker_script = Path(__file__).parent.parent / "apps" / "cli" / "mesh_worker_subprocess.py"
            output_folder = Path(app.config['OUTPUT_FOLDER'])
            output_folder.mkdir(parents=True, exist_ok=True)
            
            storage = get_storage()
            use_s3 = app.config.get('USE_S3', False)
            
            # If S3, download file to local temp folder
            if use_s3 and source_file.storage_path and source_file.storage_path.startswith('s3://'):
                local_input_dir = Path(tempfile.mkdtemp(prefix='meshgen_batch_'))
                local_input_file = local_input_dir / source_file.original_filename
                storage.download_to_local(source_file.storage_path, str(local_input_file))
                input_filepath = str(local_input_file)
            else:
                input_filepath = source_file.storage_path
            
            if not Path(input_filepath).exists():
                raise FileNotFoundError(f"Input file not found: {input_filepath}")
            
            # Build quality params based on preset
            quality_params = {
                'quality_preset': job.quality_preset,
                'target_elements': job.target_elements,
                'max_element_size': job.max_element_size,
                'mesh_strategy': job.mesh_strategy,
                'curvature_adaptive': job.curvature_adaptive,
            }
            
            if job.custom_params:
                quality_params.update(job.custom_params)
            
            # Build command
            cmd = [sys.executable, str(worker_script), input_filepath, str(output_folder)]
            cmd.extend(['--quality-params', json.dumps(quality_params)])
            
            emit_progress(batch_id, job_id, 10, 'processing', 'Running mesh worker...')
            
            # Run mesh generation
            logs = []
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            result_data = None
            line_count = 0
            
            for line in process.stdout:
                line = line.strip()
                if line:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    log_line = f"[{timestamp}] {line}"
                    logs.append(log_line)
                    line_count += 1
                    
                    # Update progress periodically
                    if line_count % 10 == 0:
                        progress = min(80, 10 + (line_count // 5))
                        emit_progress(batch_id, job_id, progress, 'processing', line[:100])
                    
                    # Check for JSON result
                    if line.startswith('{') and '"success"' in line:
                        try:
                            result_data = json.loads(line)
                        except json.JSONDecodeError:
                            pass
            
            process.wait()
            
            # Process result
            if result_data and result_data.get('success'):
                local_output_path = result_data.get('output_file')
                
                job.output_filename = Path(local_output_path).name if local_output_path else None
                job.quality_metrics = result_data.get('metrics')
                job.node_count = result_data.get('total_nodes')
                job.element_count = result_data.get('total_elements')
                job.score = result_data.get('score')
                job.logs = logs
                job.status = 'completed'
                job.completed_at = datetime.utcnow()
                job.processing_time = time.time() - start_time
                
                if local_output_path and Path(local_output_path).exists():
                    job.output_file_size = Path(local_output_path).stat().st_size
                    
                    # Upload to S3 if configured
                    if use_s3 and user:
                        mesh_filename = f"{source_file.id}_{job.quality_preset}_mesh.msh"
                        s3_path = storage.save_local_file(
                            local_path=local_output_path,
                            filename=mesh_filename,
                            user_email=user.email,
                            file_type='mesh'
                        )
                        job.output_path = s3_path
                    else:
                        job.output_path = local_output_path
                
                db.session.commit()
                
                # Update batch counters
                batch.completed_jobs = BatchJob.query.filter_by(
                    batch_id=batch_id, status='completed'
                ).count()
                
                # Check if batch is complete
                total_done = batch.completed_jobs + batch.failed_jobs
                if total_done >= batch.total_jobs:
                    batch.status = 'completed'
                    batch.completed_at = datetime.utcnow()
                
                db.session.commit()
                
                emit_progress(batch_id, job_id, 100, 'completed', 
                             f'Completed: {job.element_count} elements, score: {job.score:.2f}' if job.score else 'Completed')
                
                return {
                    'success': True,
                    'job_id': job_id,
                    'output_path': job.output_path,
                    'element_count': job.element_count,
                    'score': job.score,
                    'processing_time': job.processing_time
                }
            else:
                error_msg = result_data.get('error', 'Unknown error') if result_data else f'Process exited with code {process.returncode}'
                raise Exception(error_msg)
                
        except SoftTimeLimitExceeded:
            job.status = 'failed'
            job.error_message = 'Task timed out (exceeded 10 minutes)'
            job.completed_at = datetime.utcnow()
            job.processing_time = time.time() - start_time
            db.session.commit()
            
            batch.failed_jobs = BatchJob.query.filter_by(batch_id=batch_id, status='failed').count()
            db.session.commit()
            
            emit_progress(batch_id, job_id, 0, 'failed', 'Task timed out')
            return {'success': False, 'error': 'Task timed out'}
            
        except Exception as e:
            job.status = 'failed'
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            job.processing_time = time.time() - start_time
            job.logs = logs if 'logs' in dir() else []
            db.session.commit()
            
            batch.failed_jobs = BatchJob.query.filter_by(batch_id=batch_id, status='failed').count()
            
            # Check if batch should be marked failed/complete
            total_done = batch.completed_jobs + batch.failed_jobs
            if total_done >= batch.total_jobs:
                if batch.completed_jobs == 0:
                    batch.status = 'failed'
                else:
                    batch.status = 'completed'  # Partial success
                batch.completed_at = datetime.utcnow()
            
            db.session.commit()
            
            emit_progress(batch_id, job_id, 0, 'failed', str(e)[:200])
            
            # Retry on transient errors
            if self.request.retries < self.max_retries:
                raise self.retry(exc=e, countdown=30)
            
            return {'success': False, 'error': str(e)}


@shared_task
def start_batch_processing(batch_id: str):
    """
    Start processing all jobs in a batch.
    
    This task queues individual mesh jobs respecting the parallel limit.
    
    Args:
        batch_id: UUID of the Batch to process
    """
    app = get_flask_app()
    
    with app.app_context():
        from models import db, Batch, BatchJob
        
        batch = Batch.query.get(batch_id)
        if not batch:
            return {'success': False, 'error': 'Batch not found'}
        
        if batch.status != 'ready':
            return {'success': False, 'error': f'Batch not ready (status: {batch.status})'}
        
        # Update batch status
        batch.status = 'processing'
        batch.started_at = datetime.utcnow()
        db.session.commit()
        
        # Get all pending jobs
        pending_jobs = BatchJob.query.filter_by(
            batch_id=batch_id, 
            status='pending'
        ).all()
        
        # Queue each job (Celery will handle concurrency)
        for job in pending_jobs:
            job.status = 'queued'
            db.session.commit()
            
            # Queue the task
            process_mesh_job.delay(job.id, batch_id)
        
        emit_progress(batch_id, None, 0, 'processing', f'Started {len(pending_jobs)} mesh generation jobs')
        
        return {
            'success': True,
            'batch_id': batch_id,
            'jobs_queued': len(pending_jobs)
        }


@shared_task
def cancel_batch(batch_id: str):
    """
    Cancel all pending/queued jobs in a batch.
    
    Running jobs will complete but no new jobs will start.
    """
    app = get_flask_app()
    
    with app.app_context():
        from models import db, Batch, BatchJob
        from celery_app import celery
        
        batch = Batch.query.get(batch_id)
        if not batch:
            return {'success': False, 'error': 'Batch not found'}
        
        # Cancel pending and queued jobs
        jobs_to_cancel = BatchJob.query.filter(
            BatchJob.batch_id == batch_id,
            BatchJob.status.in_(['pending', 'queued'])
        ).all()
        
        cancelled_count = 0
        for job in jobs_to_cancel:
            job.status = 'cancelled'
            job.completed_at = datetime.utcnow()
            
            # Revoke Celery task if it exists
            if job.celery_task_id:
                celery.control.revoke(job.celery_task_id, terminate=True)
            
            cancelled_count += 1
        
        batch.status = 'cancelled'
        batch.completed_at = datetime.utcnow()
        db.session.commit()
        
        return {
            'success': True,
            'batch_id': batch_id,
            'cancelled_jobs': cancelled_count
        }
