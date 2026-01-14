"""
Rate Limiting Middleware for Job Quota Enforcement

This module provides decorators to enforce daily job quotas before dispatching
jobs to Modal or local compute backends.
"""
from functools import wraps
from datetime import datetime, date, timezone
from flask import jsonify, current_app
from flask_jwt_extended import get_jwt_identity
from sqlalchemy import func, and_

from models import db, JobUsage, User


def get_today_range():
    """
    Get start and end of today in UTC.
    
    Returns:
        tuple: (today_start, today_end) as timezone-aware datetime objects
    """
    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_end = now.replace(hour=23, minute=59, second=59, microsecond=999999)
    return today_start, today_end


def check_job_quota(job_type):
    """
    Decorator to check daily job quota before allowing job dispatch.
    
    This decorator:
    1. Extracts user_id from JWT token
    2. Counts jobs started today for this user
    3. Compares against DEFAULT_JOB_QUOTA
    4. Logs the attempt (blocked or allowed)
    5. Returns HTTP 429 if quota exceeded
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            user_id = int(get_jwt_identity())
            
            # Get quota from config
            quota = current_app.config.get('DEFAULT_JOB_QUOTA', 10)
            
            # Count jobs started today (UTC date)
            today_start, today_end = get_today_range()
            
            try:
                # Flush to ensure accurate count
                db.session.flush()
                
                jobs_today = db.session.query(func.count(JobUsage.id)).filter(
                    and_(
                        JobUsage.user_id == user_id,
                        JobUsage.started_at >= today_start,
                        JobUsage.started_at <= today_end
                    )
                ).scalar() or 0
                
                # Check quota
                if jobs_today >= quota:
                    print(f"[RATE_LIMIT] BLOCKED: User {user_id} exceeded quota ({jobs_today} >= {quota})")
                    
                    # LOG BLOCKED ATTEMPT
                    try:
                        blocked_record = JobUsage(
                            user_id=user_id,
                            job_type=job_type,
                            status='blocked',
                            started_at=datetime.now(timezone.utc),
                            compute_backend='none'
                        )
                        db.session.add(blocked_record)
                        db.session.commit()
                    except Exception as log_err:
                        db.session.rollback()
                        print(f"[RATE_LIMIT] Failed to log blocked attempt: {log_err}")

                    return jsonify({
                        'error': 'Daily job quota exceeded',
                        'quota': quota,
                        'used': jobs_today,
                        'message': f'You have reached your daily limit of {quota} jobs. Please try again tomorrow.'
                    }), 429
                
                print(f"[RATE_LIMIT] ALLOWED: User {user_id} has {jobs_today}/{quota} jobs, proceeding")
            except Exception as e:
                print(f"[RATE_LIMIT] ERROR: Exception checking quota for user {user_id}: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({
                    'error': 'Rate limit check failed',
                    'message': 'Unable to verify job quota. Please try again later.'
                }), 500
            
            # Quota check passed - proceed with function
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator


def create_job_usage_record(user_id, job_type, job_id=None, project_id=None, 
                            batch_id=None, batch_job_id=None, compute_backend=None, 
                            status='pending'):
    """
    Create a JobUsage record to track a job attempt.
    
    This should be called BEFORE dispatching the job to Modal/local compute.
    
    Args:
        user_id: User ID from JWT
        job_type: 'single_mesh', 'batch_job', or 'preview'
        job_id: modal_job_id, celery_task_id, or model ID
        project_id: Project ID (for single mesh jobs)
        batch_id: Batch ID (for batch jobs)
        batch_job_id: BatchJob ID (for individual batch jobs)
        compute_backend: 'modal', 'local', or 'celery'
        status: 'pending', 'processing', 'completed', 'failed', 'cancelled'
    
    Returns:
        JobUsage instance or None if creation failed
    
    Raises:
        Exception: Re-raises database errors after rollback
    """
    try:
        job_usage = JobUsage(
            user_id=user_id,
            job_id=job_id,
            job_type=job_type,
            status=status,
            compute_backend=compute_backend,
            project_id=project_id,
            batch_id=batch_id,
            batch_job_id=batch_job_id,
            started_at=datetime.now(timezone.utc)
        )
        
        db.session.add(job_usage)
        db.session.commit()
        
        return job_usage
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(
            f"Failed to create JobUsage record for user {user_id}, job {job_id}: {e}",
            exc_info=True
        )
        # Re-raise to allow caller to handle
        raise


def update_job_usage_status(job_id, status, compute_backend=None, completed_at=None):
    """
    Update JobUsage record status after job completion/failure.
    
    Args:
        job_id: modal_job_id, celery_task_id, or model ID used to find record
        status: 'processing', 'completed', 'failed', 'cancelled'
        compute_backend: 'modal', 'local', or 'celery' (optional update)
        completed_at: datetime for completion timestamp (optional)
    
    Returns:
        JobUsage instance or None if record not found or update failed
    """
    try:
        job_usage = JobUsage.query.filter_by(job_id=job_id).first()
        
        if not job_usage:
            current_app.logger.warning(f"JobUsage record not found for job_id: {job_id}")
            return None
        
        job_usage.status = status
        if compute_backend:
            job_usage.compute_backend = compute_backend
        if completed_at:
            job_usage.completed_at = completed_at
        elif status in ['completed', 'failed', 'cancelled']:
            job_usage.completed_at = datetime.now(timezone.utc)
        
        db.session.commit()
        return job_usage
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(
            f"Failed to update JobUsage status for job_id {job_id}: {e}",
            exc_info=True
        )
        return None


def get_user_job_count_today(user_id):
    """
    Get the number of jobs started today for a user.
    
    Args:
        user_id: User ID
    
    Returns:
        int: Count of jobs started today
    """
    today_start, today_end = get_today_range()
    
    return JobUsage.query.filter(
        and_(
            JobUsage.user_id == user_id,
            JobUsage.started_at >= today_start,
            JobUsage.started_at <= today_end
        )
    ).count()


def check_batch_quota(user_id, batch_job_count):
    """
    Check if a batch would exceed daily quota.
    
    Args:
        user_id: User ID
        batch_job_count: Number of jobs in the batch
    
    Returns:
        tuple: (allowed: bool, current_count: int, quota: int, remaining: int)
    """
    quota = current_app.config.get('DEFAULT_JOB_QUOTA', 10)
    current_count = get_user_job_count_today(user_id)
    remaining = quota - current_count
    
    if batch_job_count > remaining:
        return False, current_count, quota, remaining
    
    return True, current_count, quota, remaining

