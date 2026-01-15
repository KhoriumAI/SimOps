"""
Database models for the Mesh Generation application
"""
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import uuid

db = SQLAlchemy()


class User(db.Model):
    """User model for authentication"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    name = db.Column(db.String(100), nullable=True)
    role = db.Column(db.String(20), default='user')
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)
    last_api_request_at = db.Column(db.DateTime, nullable=True)  # Track last API request for time online calculation
    
    # Password reset tokens
    reset_token = db.Column(db.String(100), unique=True, nullable=True, index=True)
    reset_token_expires = db.Column(db.DateTime, nullable=True)
    
    # Storage quota (in bytes) - default 1GB
    storage_quota = db.Column(db.BigInteger, default=1073741824)
    storage_used = db.Column(db.BigInteger, default=0)
    
    projects = db.relationship('Project', backref='owner', lazy='dynamic', cascade='all, delete-orphan')
    activities = db.relationship('ActivityLog', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    job_usage = db.relationship('JobUsage', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'email': self.email,
            'name': self.name,
            'role': self.role,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'storage_used': self.storage_used,
            'storage_quota': self.storage_quota,
        }


class Project(db.Model):
    """Project model for CAD files"""
    __tablename__ = 'projects'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=True)  # Original name before sanitization
    filepath = db.Column(db.String(500), nullable=False)
    preview_path = db.Column(db.String(500), nullable=True)  # Path to generated preview JSON
    file_size = db.Column(db.BigInteger, default=0)  # Size in bytes
    file_hash = db.Column(db.String(64), nullable=True)  # SHA256 hash for deduplication
    mime_type = db.Column(db.String(100), nullable=True)
    status = db.Column(db.String(50), default='uploaded', index=True)
    error_message = db.Column(db.Text, nullable=True)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Usage tracking
    mesh_count = db.Column(db.Integer, default=0)  # Number of times meshed
    download_count = db.Column(db.Integer, default=0)  # Number of downloads
    last_accessed = db.Column(db.DateTime, nullable=True)
    
    mesh_results = db.relationship('MeshResult', backref='project', lazy='dynamic', 
                                   cascade='all, delete-orphan', order_by='MeshResult.created_at.desc()')
    downloads = db.relationship('DownloadRecord', backref='project', lazy='dynamic',
                               cascade='all, delete-orphan')
    
    def to_dict(self, include_results=False):
        data = {
            'id': self.id,
            'filename': self.filename,
            'original_filename': self.original_filename,
            'file_size': self.file_size,
            'preview_path': self.preview_path,
            'status': self.status,
            'error_message': self.error_message,
            'mesh_count': self.mesh_count,
            'download_count': self.download_count,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
        if include_results:
            data['mesh_results'] = [r.to_dict() for r in self.mesh_results.all()]
        latest = self.mesh_results.first()
        if latest:
            data['latest_result'] = latest.to_dict()
        return data


class MeshResult(db.Model):
    """Mesh generation result"""
    __tablename__ = 'mesh_results'
    
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.String(36), db.ForeignKey('projects.id'), nullable=False, index=True)
    strategy = db.Column(db.String(100), nullable=True)
    score = db.Column(db.Float, nullable=True)
    output_path = db.Column(db.String(500), nullable=True)
    output_size = db.Column(db.BigInteger, default=0)  # Output file size in bytes
    quality_metrics = db.Column(db.JSON, nullable=True)
    logs = db.Column(db.JSON, nullable=True)
    params = db.Column(db.JSON, nullable=True)
    boundary_zones = db.Column(db.JSON, nullable=True)  # { "ZoneName": [face_indices], ... }
    job_id = db.Column(db.String(50), nullable=True)  # Human-readable job ID (e.g. MSH-0101-ABCD)
    
    # NEW: Modal compute tracking
    modal_job_id = db.Column(db.String(100), nullable=True, index=True)
    modal_status = db.Column(db.String(20), nullable=True)
    modal_started_at = db.Column(db.DateTime, nullable=True)
    modal_completed_at = db.Column(db.DateTime, nullable=True)
    
    
    # Processing info
    processing_time = db.Column(db.Float, nullable=True)  # Time in seconds
    node_count = db.Column(db.Integer, nullable=True)
    element_count = db.Column(db.Integer, nullable=True)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime, nullable=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'project_id': self.project_id,
            'strategy': self.strategy,
            'job_id': getattr(self, 'job_id', None),
            'score': self.score,
            'output_path': self.output_path,
            'output_size': self.output_size,
            'quality_metrics': self.quality_metrics,
            'boundary_zones': getattr(self, 'boundary_zones', None),
            'logs': self.logs,
            'processing_time': self.processing_time,
            'node_count': self.node_count,
            'element_count': self.element_count,
            'modal_job_id': self.modal_job_id,
            'modal_status': self.modal_status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
        }


class DownloadRecord(db.Model):
    """Track file downloads"""
    __tablename__ = 'download_records'
    
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.String(36), db.ForeignKey('projects.id'), nullable=False, index=True)
    mesh_result_id = db.Column(db.Integer, db.ForeignKey('mesh_results.id'), nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    
    download_type = db.Column(db.String(50), nullable=False)  # 'mesh', 'cad', 'quality_report'
    file_format = db.Column(db.String(20), nullable=True)  # 'msh', 'vtk', 'stl', etc.
    file_size = db.Column(db.BigInteger, default=0)
    
    ip_address = db.Column(db.String(45), nullable=True)  # IPv6 compatible
    user_agent = db.Column(db.String(500), nullable=True)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class ActivityLog(db.Model):
    """Track user activities for audit trail"""
    __tablename__ = 'activity_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    
    action = db.Column(db.String(50), nullable=False, index=True)  # 'upload', 'generate', 'download', 'login', 'logout'
    resource_type = db.Column(db.String(50), nullable=True)  # 'project', 'mesh_result'
    resource_id = db.Column(db.String(36), nullable=True)
    
    details = db.Column(db.JSON, nullable=True)  # Additional context
    ip_address = db.Column(db.String(45), nullable=True)
    user_agent = db.Column(db.String(500), nullable=True)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'action': self.action,
            'resource_type': self.resource_type,
            'resource_id': self.resource_id,
            'details': self.details,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


class TokenBlocklist(db.Model):
    """Blocklist for revoked JWT tokens"""
    __tablename__ = 'token_blocklist'
    
    id = db.Column(db.Integer, primary_key=True)
    jti = db.Column(db.String(36), nullable=False, unique=True, index=True)
    token_type = db.Column(db.String(10), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime, nullable=False)


class Feedback(db.Model):
    """User feedback storage"""
    __tablename__ = 'feedbacks'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)  # Nullable for anonymous if needed
    user_email = db.Column(db.String(120), nullable=True)  # Capture email even if not logged in
    
    type = db.Column(db.String(50), default='feedback')  # feedback, bug, feature
    message = db.Column(db.Text, nullable=False)
    
    # Context
    url = db.Column(db.String(500), nullable=True)
    user_agent = db.Column(db.String(500), nullable=True)
    job_id = db.Column(db.String(50), nullable=True)  # Related job context
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'type': self.type,
            'message': self.message,
            'url': self.url,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


# ============================================================================
# BATCH PROCESSING MODELS
# ============================================================================

class Batch(db.Model):
    """
    A batch session that groups multiple file uploads.
    
    Example: User drags 10 STEP files -> Creates 1 Batch with 10 BatchFiles
    If mesh_independence=True -> Creates 30 BatchJobs (3 per file: coarse, medium, fine)
    """
    __tablename__ = 'batches'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    name = db.Column(db.String(255), nullable=True)  # Optional user-assigned name
    
    # Settings
    mesh_independence = db.Column(db.Boolean, default=False)  # If True, creates Coarse/Med/Fine jobs
    parallel_limit = db.Column(db.Integer, default=6)         # Max concurrent jobs
    
    # Mesh parameters (applied to all jobs unless overridden)
    mesh_strategy = db.Column(db.String(50), default='Tetrahedral (Delaunay)')
    curvature_adaptive = db.Column(db.Boolean, default=True)
    
    # Status tracking
    status = db.Column(db.String(20), default='pending', index=True)  # pending, uploading, ready, processing, completed, failed, cancelled
    total_files = db.Column(db.Integer, default=0)
    total_jobs = db.Column(db.Integer, default=0)
    completed_jobs = db.Column(db.Integer, default=0)
    failed_jobs = db.Column(db.Integer, default=0)
    
    # Error handling
    error_message = db.Column(db.Text, nullable=True)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    started_at = db.Column(db.DateTime, nullable=True)
    completed_at = db.Column(db.DateTime, nullable=True)
    
    # Relationships
    files = db.relationship('BatchFile', backref='batch', lazy='dynamic', 
                           cascade='all, delete-orphan', order_by='BatchFile.created_at')
    jobs = db.relationship('BatchJob', backref='batch', lazy='dynamic',
                          cascade='all, delete-orphan', order_by='BatchJob.created_at')
    
    @property
    def progress(self):
        """Calculate overall progress percentage"""
        if self.total_jobs == 0:
            return 0
        return round((self.completed_jobs + self.failed_jobs) / self.total_jobs * 100, 1)
    
    @property
    def is_complete(self):
        """Check if all jobs are finished (success or failure)"""
        return self.status in ['completed', 'failed', 'cancelled']
    
    def to_dict(self, include_files=False, include_jobs=False):
        data = {
            'id': self.id,
            'user_id': self.user_id,
            'name': self.name,
            'mesh_independence': self.mesh_independence,
            'parallel_limit': self.parallel_limit,
            'mesh_strategy': self.mesh_strategy,
            'curvature_adaptive': self.curvature_adaptive,
            'status': self.status,
            'total_files': self.total_files,
            'total_jobs': self.total_jobs,
            'completed_jobs': self.completed_jobs,
            'failed_jobs': self.failed_jobs,
            'progress': self.progress,
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
        }
        if include_files:
            # Pass include_jobs to each file so jobs are nested inside files
            data['files'] = [f.to_dict(include_jobs=include_jobs) for f in self.files.all()]
        if include_jobs:
            # Also include flat list of all jobs for backward compat
            data['jobs'] = [j.to_dict() for j in self.jobs.all()]
        return data


class BatchFile(db.Model):
    """
    Original CAD file uploaded as part of a batch.
    
    Each BatchFile can have multiple BatchJobs:
    - If mesh_independence=False: 1 job (medium quality)
    - If mesh_independence=True: 3 jobs (coarse, medium, fine)
    """
    __tablename__ = 'batch_files'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    batch_id = db.Column(db.String(36), db.ForeignKey('batches.id'), nullable=False, index=True)
    
    # File info
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_size = db.Column(db.BigInteger, default=0)  # bytes
    file_type = db.Column(db.String(20), default='step')  # step, stp, stl
    file_hash = db.Column(db.String(64), nullable=True)  # SHA256 for deduplication
    
    # Storage location
    storage_path = db.Column(db.String(500), nullable=True)  # Local path or S3 URI
    
    # CAD Geometry info (extracted on upload)
    geometry_info = db.Column(db.JSON, nullable=True)  # volumes, surfaces, curves, bbox
    
    # Status
    status = db.Column(db.String(20), default='uploading', index=True)  # uploading, uploaded, processing, ready, error
    error_message = db.Column(db.Text, nullable=True)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    uploaded_at = db.Column(db.DateTime, nullable=True)
    
    # Relationships
    jobs = db.relationship('BatchJob', backref='source_file', lazy='dynamic',
                          cascade='all, delete-orphan', order_by='BatchJob.quality_preset')
    
    def to_dict(self, include_jobs=False):
        data = {
            'id': self.id,
            'batch_id': self.batch_id,
            'filename': self.filename,
            'original_filename': self.original_filename,
            'file_size': self.file_size,
            'file_type': self.file_type,
            'geometry_info': self.geometry_info,
            'status': self.status,
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'uploaded_at': self.uploaded_at.isoformat() if self.uploaded_at else None,
        }
        if include_jobs:
            data['jobs'] = [j.to_dict() for j in self.jobs.all()]
        return data


class BatchJob(db.Model):
    """
    Individual mesh generation job within a batch.
    
    For Mesh Independence study:
    - 1 source file creates 3 jobs (coarse, medium, fine)
    - Each job runs independently and can be parallelized (up to 6 concurrent)
    """
    __tablename__ = 'batch_jobs'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    batch_id = db.Column(db.String(36), db.ForeignKey('batches.id'), nullable=False, index=True)
    file_id = db.Column(db.String(36), db.ForeignKey('batch_files.id'), nullable=False, index=True)
    
    # Quality preset for mesh independence
    quality_preset = db.Column(db.String(20), default='medium', index=True)  # coarse, medium, fine
    
    # Mesh parameters
    mesh_strategy = db.Column(db.String(50), default='Tetrahedral (Delaunay)')
    target_elements = db.Column(db.Integer, default=5000)
    max_element_size = db.Column(db.Float, default=5.0)
    min_element_size = db.Column(db.Float, nullable=True)
    curvature_adaptive = db.Column(db.Boolean, default=True)
    
    # Custom parameters (JSON for flexibility)
    custom_params = db.Column(db.JSON, nullable=True)
    
    # Output file info
    output_filename = db.Column(db.String(255), nullable=True)
    output_path = db.Column(db.String(500), nullable=True)  # Local path or S3 URI
    output_file_size = db.Column(db.BigInteger, nullable=True)
    
    # Mesh quality metrics
    quality_metrics = db.Column(db.JSON, nullable=True)  # sicn_min, sicn_avg, element_count, etc.
    node_count = db.Column(db.Integer, nullable=True)
    element_count = db.Column(db.Integer, nullable=True)
    score = db.Column(db.Float, nullable=True)
    
    # Status tracking
    status = db.Column(db.String(20), default='pending', index=True)  # pending, queued, processing, completed, failed, cancelled
    progress = db.Column(db.Integer, default=0)  # 0-100
    error_message = db.Column(db.Text, nullable=True)
    logs = db.Column(db.JSON, nullable=True)  # Processing logs
    
    # Celery task tracking
    celery_task_id = db.Column(db.String(50), nullable=True, index=True)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    started_at = db.Column(db.DateTime, nullable=True)
    completed_at = db.Column(db.DateTime, nullable=True)
    processing_time = db.Column(db.Float, nullable=True)  # seconds
    
    def to_dict(self):
        return {
            'id': self.id,
            'batch_id': self.batch_id,
            'file_id': self.file_id,
            'quality_preset': self.quality_preset,
            'mesh_strategy': self.mesh_strategy,
            'target_elements': self.target_elements,
            'max_element_size': self.max_element_size,
            'curvature_adaptive': self.curvature_adaptive,
            'output_filename': self.output_filename,
            'output_path': self.output_path,
            'output_file_size': self.output_file_size,
            'quality_metrics': self.quality_metrics,
            'node_count': self.node_count,
            'element_count': self.element_count,
            'score': self.score,
            'status': self.status,
            'progress': self.progress,
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'processing_time': self.processing_time,
        }


# Preset configurations for mesh independence study
MESH_PRESETS = {
    'coarse': {
        'target_elements': 1000,
        'max_element_size': 10.0,
        'curvature_adaptive': False,
        'description': 'Fast preview mesh, lower accuracy'
    },
    'medium': {
        'target_elements': 5000,
        'max_element_size': 5.0,
        'curvature_adaptive': True,
        'description': 'Balanced mesh for general analysis'
    },
    'fine': {
        'target_elements': 20000,
        'max_element_size': 2.0,
        'curvature_adaptive': True,
        'description': 'High-quality mesh for detailed analysis'
    }
}


def create_batch_jobs_for_file(batch_file, batch, mesh_independence=False):
    """
    Create BatchJob records for a source file.
    
    Args:
        batch_file: BatchFile instance
        batch: Batch instance  
        mesh_independence: If True, creates 3 jobs (coarse, medium, fine)
                          If False, creates 1 job (medium only)
    
    Returns:
        List of created BatchJob instances
    """
    jobs = []
    
    if mesh_independence:
        presets = ['coarse', 'medium', 'fine']
    else:
        presets = ['medium']
    
    for preset_name in presets:
        preset = MESH_PRESETS[preset_name]
        job = BatchJob(
            batch_id=batch.id,
            file_id=batch_file.id,
            quality_preset=preset_name,
            mesh_strategy=batch.mesh_strategy,
            target_elements=preset['target_elements'],
            max_element_size=preset['max_element_size'],
            curvature_adaptive=preset['curvature_adaptive'],
        )
        db.session.add(job)
        jobs.append(job)
    
    return jobs


# ============================================================================
# JOB USAGE TRACKING MODEL
# ============================================================================

class JobUsage(db.Model):
    """
    Track all job attempts for rate limiting and usage analytics.
    
    Every job attempt is logged here, even if it fails or is blocked by quota.
    This enables:
    - Daily quota enforcement (COUNT jobs per user per day)
    - Power user identification for sales outreach
    - Cost tracking and analysis
    """
    __tablename__ = 'job_usage'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    
    # Job identifier - modal_job_id, celery_task_id, or model ID
    job_id = db.Column(db.String(100), nullable=True, index=True)
    
    # Job type classification
    job_type = db.Column(db.String(50), nullable=False)  # 'single_mesh', 'batch_job', 'preview'
    
    # Timestamps
    started_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    completed_at = db.Column(db.DateTime, nullable=True)
    
    # Status tracking
    status = db.Column(db.String(20), nullable=False, index=True)  # 'pending', 'processing', 'completed', 'failed', 'cancelled'
    
    # Compute backend used
    compute_backend = db.Column(db.String(20), nullable=True)  # 'modal', 'local', 'celery'
    
    # Foreign key references (nullable for flexibility)
    project_id = db.Column(db.String(36), db.ForeignKey('projects.id'), nullable=True)
    batch_id = db.Column(db.String(36), db.ForeignKey('batches.id'), nullable=True)
    batch_job_id = db.Column(db.String(36), db.ForeignKey('batch_jobs.id'), nullable=True)
    
    # Composite index for efficient daily quota queries
    __table_args__ = (
        db.Index('ix_job_usage_user_started', 'user_id', 'started_at'),
    )
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'job_id': self.job_id,
            'job_type': self.job_type,
            'status': self.status,
            'compute_backend': self.compute_backend,
            'project_id': self.project_id,
            'batch_id': self.batch_id,
            'batch_job_id': self.batch_job_id,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
        }
