"""
Webhook and WebSocket routes for Modal job completion and log streaming
"""
from flask import Blueprint, request, jsonify, current_app
from flask_socketio import emit, join_room, leave_room
from typing import Optional, Dict, Any
from datetime import datetime
import json
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import db, MeshResult, Project
from webhook_utils import verify_webhook_signature, extract_signature_from_header
from cloudwatch_logs import CloudWatchLogTailer, create_log_tailer_for_job
from middleware.rate_limit import update_job_usage_status

webhook_bp = Blueprint('webhooks', __name__)

# Store active log tailers per job_id
active_log_tailers: dict[str, CloudWatchLogTailer] = {}
# Store SocketIO instance (will be set by api_server)
socketio_instance: Optional[Any] = None


def init_socketio(socketio):
    """Initialize SocketIO instance for use in routes"""
    global socketio_instance
    socketio_instance = socketio


@webhook_bp.route('/api/webhooks/modal', methods=['POST'])
def modal_webhook():
    """
    Webhook endpoint for Modal job completion notifications.
    
    Expected payload:
    {
        "job_id": "modal-job-id",
        "status": "completed" | "failed",
        "result": {
            "success": bool,
            "s3_output_path": str,
            "strategy": str,
            "quality_metrics": dict,
            ...
        },
        "error": str (if failed)
    }
    """
    try:
        # Get raw payload for signature verification
        payload = request.get_data()
        
        # Verify webhook signature
        signature_header = request.headers.get('X-Modal-Signature')
        signature = extract_signature_from_header(signature_header)
        
        if not verify_webhook_signature(payload, signature):
            print(f"[WEBHOOK] Invalid signature for webhook")
            return jsonify({'error': 'Invalid signature'}), 401
        
        # Parse JSON payload
        data = json.loads(payload.decode('utf-8'))
        job_id = data.get('job_id')
        
        if not job_id:
            return jsonify({'error': 'Missing job_id'}), 400
        
        print(f"[WEBHOOK] Received completion notification for job {job_id}")
        
        # Find MeshResult by modal_job_id
        mesh_result = MeshResult.query.filter_by(modal_job_id=job_id).first()
        
        if not mesh_result:
            print(f"[WEBHOOK] Warning: No MeshResult found for job_id {job_id}")
            return jsonify({'error': 'Job not found'}), 404
        
        # Update MeshResult with completion status
        status = data.get('status', 'failed')
        mesh_result.modal_status = status
        mesh_result.modal_completed_at = datetime.utcnow()
        mesh_result.completed_at = datetime.utcnow()
        
        # Update JobUsage record
        job_usage_status = 'completed' if status == 'completed' else 'failed'
        update_job_usage_status(job_id, job_usage_status, 'modal', datetime.utcnow())
        
        if status == 'completed' and data.get('result'):
            result = data['result']
            mesh_result.output_path = result.get('s3_output_path')
            mesh_result.strategy = result.get('strategy')
            mesh_result.quality_metrics = result.get('quality_metrics')
            mesh_result.processing_time = result.get('metrics', {}).get('total_time_seconds', 0)
            mesh_result.node_count = result.get('total_nodes', 0)
            mesh_result.element_count = result.get('total_elements', 0)
            
            # Update project status
            project = mesh_result.project
            project.status = 'completed'
            db.session.commit()
            
            # Emit WebSocket event to notify frontend
            if socketio_instance:
                socketio_instance.emit('job_completed', {
                    'project_id': project.id,
                    'job_id': job_id,
                    'status': 'completed',
                    'result': mesh_result.to_dict()
                }, room=f'project_{project.id}')
            
            print(f"[WEBHOOK] Job {job_id} marked as completed")
            
        elif status == 'failed':
            error_msg = data.get('error', 'Unknown error')
            mesh_result.project.status = 'error'
            mesh_result.project.error_message = error_msg
            db.session.commit()
            
            # Emit WebSocket event
            if socketio_instance:
                socketio_instance.emit('job_failed', {
                    'project_id': mesh_result.project_id,
                    'job_id': job_id,
                    'error': error_msg
                }, room=f'project_{mesh_result.project_id}')
            
            print(f"[WEBHOOK] Job {job_id} marked as failed: {error_msg}")
        
        db.session.commit()
        
        return jsonify({'status': 'ok', 'job_id': job_id}), 200
        
    except json.JSONDecodeError as e:
        print(f"[WEBHOOK] Invalid JSON: {e}")
        return jsonify({'error': 'Invalid JSON'}), 400
    except Exception as e:
        print(f"[WEBHOOK] Error processing webhook: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def register_socketio_handlers(socketio):
    """
    Register WebSocket event handlers for log streaming.
    """
    
    @socketio.on('connect')
    def handle_connect():
        """Handle WebSocket connection"""
        print(f"[WS] Client connected: {request.sid}")  # type: ignore[attr-defined]
    
    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle WebSocket disconnection"""
        print(f"[WS] Client disconnected: {request.sid}")  # type: ignore[attr-defined]
        
        # Stop any log tailers for this client
        for job_id, tailer in list(active_log_tailers.items()):
            if hasattr(tailer, '_client_sid') and tailer._client_sid == request.sid:  # type: ignore[attr-defined]
                tailer.stop()
                del active_log_tailers[job_id]
    
    @socketio.on('subscribe_logs')
    def handle_subscribe_logs(data):
        """
        Subscribe to logs for a specific job.
        
        Expected data:
        {
            "job_id": "modal-job-id" or "local-{result_id}",
            "project_id": "project-uuid"
        }
        """
        try:
            job_id = data.get('job_id')
            project_id = data.get('project_id')
            
            if not job_id:
                emit('error', {'message': 'Missing job_id'})
                return
            
            print(f"[WS] Client {request.sid} subscribing to logs for job {job_id}")  # type: ignore[attr-defined]
            
            # Join project room for status updates
            if project_id:
                join_room(f'project_{project_id}')
            
            # Check if this is a local job (format: "local-{result_id}")
            is_local_job = job_id.startswith('local-')
            
            if is_local_job:
                # Local jobs: logs are emitted directly from subprocess, just acknowledge subscription
                print(f"[WS] Local job {job_id} - logs will stream directly from subprocess")
                print(f"[WS] Client {request.sid} joined room 'project_{project_id}' for local job logs")  # type: ignore[attr-defined]
                emit('subscribed', {'job_id': job_id, 'status': 'started', 'type': 'local'})
                print(f"[WS] Sent 'subscribed' confirmation to client {request.sid}")  # type: ignore[attr-defined]
            else:
                # Modal jobs: Create CloudWatch log tailer
                if job_id not in active_log_tailers:
                    def log_callback(message: str, timestamp: datetime):
                        """Callback for each log line"""
                        if socketio_instance:
                            socketio_instance.emit('log_line', {
                                'job_id': job_id,
                                'message': message,
                                'timestamp': timestamp.isoformat()
                            }, room=f'project_{project_id}' if project_id else None)
                    
                    # Get AWS region from config
                    region = current_app.config.get('AWS_REGION', 'us-west-1')
                    
                    try:
                        tailer = create_log_tailer_for_job(job_id, log_callback, region=region)
                        tailer._client_sid = request.sid  # type: ignore[attr-defined]
                        tailer.start()
                        active_log_tailers[job_id] = tailer
                        emit('subscribed', {'job_id': job_id, 'status': 'started', 'type': 'modal'})
                    except Exception as tailer_error:
                        print(f"[WS] Failed to create CloudWatch tailer: {tailer_error}")
                        # Still acknowledge subscription - logs might come from other sources
                        emit('subscribed', {'job_id': job_id, 'status': 'started', 'type': 'modal', 'warning': 'CloudWatch tailer failed'})
                else:
                    emit('subscribed', {'job_id': job_id, 'status': 'already_active'})
                
        except Exception as e:
            print(f"[WS] Error subscribing to logs: {e}")
            import traceback
            traceback.print_exc()
            emit('error', {'message': str(e)})
    
    @socketio.on('unsubscribe_logs')
    def handle_unsubscribe_logs(data):
        """
        Unsubscribe from logs for a specific job.
        
        Expected data:
        {
            "job_id": "modal-job-id"
        }
        """
        try:
            job_id = data.get('job_id')
            
            if job_id in active_log_tailers:
                tailer = active_log_tailers[job_id]
                tailer.stop()
                del active_log_tailers[job_id]
                
                emit('unsubscribed', {'job_id': job_id})
                print(f"[WS] Client {request.sid} unsubscribed from logs for job {job_id}")  # type: ignore[attr-defined]
            else:
                emit('error', {'message': 'Not subscribed to this job'})
                
        except Exception as e:
            print(f"[WS] Error unsubscribing from logs: {e}")
            emit('error', {'message': str(e)})
    
    @socketio.on('subscribe_project')
    def handle_subscribe_project(data):
        """
        Subscribe to status updates for a project.
        
        Expected data:
        {
            "project_id": "project-uuid"
        }
        """
        try:
            project_id = data.get('project_id')
            
            if project_id:
                join_room(f'project_{project_id}')
                emit('subscribed_project', {'project_id': project_id})
                print(f"[WS] Client {request.sid} subscribed to project {project_id}")  # type: ignore[attr-defined]
            else:
                emit('error', {'message': 'Missing project_id'})
                
        except Exception as e:
            print(f"[WS] Error subscribing to project: {e}")
            emit('error', {'message': str(e)})

    @socketio.on('join_batch')
    def handle_join_batch(data):
        """
        Subscribe to status updates for a batch.
        
        Expected data:
        {
            "batch_id": "batch-uuid"
        }
        """
        try:
            batch_id = data.get('batch_id')
            
            if batch_id:
                join_room(f'batch_{batch_id}')
                emit('joined_batch', {'batch_id': batch_id})
                print(f"[WS] Client {request.sid} subscribed to batch {batch_id}")  # type: ignore[attr-defined]
            else:
                emit('error', {'message': 'Missing batch_id'})
                
        except Exception as e:
            print(f"[WS] Error subscribing to batch: {e}")
            emit('error', {'message': str(e)})


