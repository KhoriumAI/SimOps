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
    
    # Storage quota (in bytes) - default 1GB
    storage_quota = db.Column(db.BigInteger, default=1073741824)
    storage_used = db.Column(db.BigInteger, default=0)
    
    projects = db.relationship('Project', backref='owner', lazy='dynamic', cascade='all, delete-orphan')
    activities = db.relationship('ActivityLog', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    
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
            'score': self.score,
            'output_path': self.output_path,
            'output_size': self.output_size,
            'quality_metrics': self.quality_metrics,
            'logs': self.logs,
            'processing_time': self.processing_time,
            'node_count': self.node_count,
            'element_count': self.element_count,
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
