"""
Configuration for the Flask application
"""
import os
from datetime import timedelta
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

BASE_DIR = Path(__file__).parent


def get_database_url():
    """Build database URL from environment variables"""
    # Check for full DATABASE_URL first (used by AWS RDS)
    if os.environ.get('DATABASE_URL'):
        url = os.environ.get('DATABASE_URL')
        # Fix: replace postgres:// with postgresql://
        if url.startswith('postgres://'):
            url = url.replace('postgres://', 'postgresql://', 1)
        return url
    
    # Check for PostgreSQL config (AWS RDS or local)
    db_host = os.environ.get('DB_HOST')
    if db_host:
        db_user = os.environ.get('DB_USER', 'meshgen_user')
        db_password = os.environ.get('DB_PASSWORD', 'meshgen_password_123')
        db_name = os.environ.get('DB_NAME', 'meshgen_db')
        db_port = os.environ.get('DB_PORT', '5432')
        return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    # Default to SQLite for local development
    # Ensure the instance directory exists
    instance_dir = BASE_DIR / 'instance'
    instance_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{instance_dir / 'mesh_app.db'}"


class Config:
    """
    Base configuration
    
    LOCAL DEVELOPMENT: Uses local filesystem (backend/uploads, backend/outputs)
    PRODUCTION (AWS): Uses S3 bucket (muaz-webdev-assets) with user email folders
    
    The switch is controlled by FLASK_ENV environment variable:
    - FLASK_ENV=development (default) → Local storage
    - FLASK_ENV=production → AWS S3 storage
    """
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # Database
    SQLALCHEMY_DATABASE_URI = get_database_url()
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_pre_ping': True,
        'pool_recycle': 300,
    }
    
    # JWT
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'jwt-secret-key-change-in-production')
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30)
    
    # File uploads - Local storage (used in development)
    UPLOAD_FOLDER = BASE_DIR / "uploads"
    OUTPUT_FOLDER = BASE_DIR / "outputs"
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_FILE_SIZE_MB', 500)) * 1024 * 1024  # Default 500MB
    ALLOWED_EXTENSIONS = {'.step', '.stp', '.stl'}
    
    # Batch Processing Settings (configurable via .env)
    BATCH_MAX_FILES = int(os.environ.get('BATCH_MAX_FILES', 10))  # Max files per batch
    BATCH_MAX_FILE_SIZE = int(os.environ.get('BATCH_MAX_FILE_SIZE_MB', 500)) * 1024 * 1024  # Per file limit
    BATCH_PARALLEL_JOBS = int(os.environ.get('BATCH_PARALLEL_JOBS', 6))  # Concurrent mesh jobs
    
    # Celery / Redis Configuration
    CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
    CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
    
    # WebSocket (Flask-SocketIO)
    SOCKETIO_MESSAGE_QUEUE = os.environ.get('SOCKETIO_MESSAGE_QUEUE', 'redis://localhost:6379/1')
    
    # AWS S3 Configuration (only used when USE_S3=true)
    USE_S3 = False  # Default: use local storage
    AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
    AWS_REGION = os.environ.get('AWS_REGION', 'us-west-1')
    S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME', 'muaz-webdev-assets')
    # S3 Folder structure: {user_email}/uploads/ and {user_email}/mesh/
    S3_UPLOADS_FOLDER = 'uploads'
    S3_MESH_FOLDER = 'mesh'
    
    # CORS - Add your domain
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', 'http://localhost:5173,http://localhost:3000').split(',')


class DevelopmentConfig(Config):
    """Local development - uses local filesystem, no AWS"""
    DEBUG = True
    USE_S3 = False  # Always use local storage in development
    
    # Local file paths (same as before)
    UPLOAD_FOLDER = BASE_DIR / "uploads"
    OUTPUT_FOLDER = BASE_DIR / "outputs"


class ProductionConfig(Config):
    """Production on AWS - uses S3 and RDS"""
    DEBUG = False
    USE_S3 = os.environ.get('USE_S3', 'true').lower() == 'true'  # Default to S3 in production
    
    # Still need local folders for temporary mesh processing
    UPLOAD_FOLDER = Path('/tmp/meshgen/uploads')
    OUTPUT_FOLDER = Path('/tmp/meshgen/outputs')
    
    # Production database pool settings
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_pre_ping': True,
        'pool_recycle': 300,
        'pool_size': 10,
        'max_overflow': 20,
    }


class StagingConfig(ProductionConfig):
    """Staging environment - similar to production but may have different resources"""
    pass


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'staging': StagingConfig,
    'default': DevelopmentConfig
}


def get_config():
    env = os.environ.get('FLASK_ENV', 'development')
    return config.get(env, config['default'])
