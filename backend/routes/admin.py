"""
Admin API Routes

Endpoints for administrative functions:
- /api/admin/usage - Top users by job count for analytics
"""
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from datetime import datetime, timedelta, timezone
from sqlalchemy import func, and_, case

from models import db, User, JobUsage

admin_bp = Blueprint('admin', __name__, url_prefix='/api/admin')


def require_admin(f):
    """
    Decorator to require admin role.
    
    Checks that the authenticated user has role='admin'.
    """
    from functools import wraps
    
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user_id = int(get_jwt_identity())
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({"error": "User not found"}), 404
        
        if user.role != 'admin':
            return jsonify({"error": "Admin access required"}), 403
        
        return f(*args, **kwargs)
    
    return decorated_function


@admin_bp.route('/usage', methods=['GET'])
@jwt_required()
@require_admin
def get_usage_stats():
    """
    Get top 5 users by job count for the current week.
    
    Returns JSON with:
    {
        "period": "2026-01-01 to 2026-01-07",
        "top_users": [
            {
                "user_id": 1,
                "email": "user@example.com",
                "job_count": 150,
                "completed": 145,
                "failed": 5
            }
        ]
    }
    """
    # Calculate current week (Monday to Sunday) in UTC
    today = datetime.now(timezone.utc).date()
    # Get Monday of current week
    days_since_monday = today.weekday()
    week_start = today - timedelta(days=days_since_monday)
    week_end = week_start + timedelta(days=6)
    
    # Convert to datetime for query (start of day and end of day) with timezone
    week_start_dt = datetime.combine(week_start, datetime.min.time(), timezone.utc)
    week_end_dt = datetime.combine(week_end, datetime.max.time(), timezone.utc)
    
    # Optimized query: Get top 5 users with job counts, completed, and failed in single query
    # This eliminates N+1 query problem
    top_users_query = db.session.query(
        JobUsage.user_id,
        User.email,
        func.count(JobUsage.id).label('job_count'),
        func.sum(
            case(
                (JobUsage.status == 'completed', 1),
                else_=0
            )
        ).label('completed'),
        func.sum(
            case(
                (JobUsage.status == 'failed', 1),
                else_=0
            )
        ).label('failed')
    ).join(
        User, JobUsage.user_id == User.id
    ).filter(
        and_(
            JobUsage.started_at >= week_start_dt,
            JobUsage.started_at <= week_end_dt
        )
    ).group_by(
        JobUsage.user_id, User.email
    ).order_by(
        func.count(JobUsage.id).desc()
    ).limit(5).all()
    
    # Format results
    top_users = [{
        'user_id': user_id,
        'email': email,
        'job_count': job_count,
        'completed': int(completed or 0),
        'failed': int(failed or 0)
    } for user_id, email, job_count, completed, failed in top_users_query]
    
    return jsonify({
        'period': f"{week_start.isoformat()} to {week_end.isoformat()}",
        'top_users': top_users
    })

