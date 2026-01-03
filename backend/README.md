# Backend API

This directory contains the server-side logic for the cloud-enabled version of the mesh generator.

## Structure

- **api_server.py**: Main Flask application entry point.
- **models.py**: SQLAlchemy database models (User, Project, MeshResult, etc.).
- **tasks.py**: Celery task definitions for asynchronous mesh processing.
- **storage.py**: Abstraction layer for file storage (Local vs S3).
- **routes/**: API route definitions organized by domain (auth, batch, etc.).
- **config.py**: Application configuration (Development/Production).

## Configuration

The backend uses environment variables for configuration. In production, these are managed via a `.env` file or GitHub Secrets.

See `backend/.env.example` for the required variables:
- `DATABASE_URL`: PostgreSQL connection string.
- `SECRET_KEY` & `JWT_SECRET_KEY`: Random strings for security.
- `USE_S3`: Set to `true` to use AWS S3 for storage.
- `S3_BUCKET_NAME`: Target S3 bucket for assets.
- `CORS_ORIGINS`: Comma-separated list of allowed origins.

## Diagnostics

Use `diagnose_auth.py` to verify database connectivity and environment setup:
```bash
python diagnose_auth.py
```
