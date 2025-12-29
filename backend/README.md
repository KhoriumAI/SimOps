# Backend API

This directory contains the server-side logic for the cloud-enabled version of the mesh generator.

## Structure

- **api_server.py**: Main Flask application entry point.
- **models.py**: SQLAlchemy database models (User, Project, MeshResult, etc.).
- **tasks.py**: Celery task definitions for asynchronous mesh processing.
- **storage.py**: Abstraction layer for file storage (Local vs S3).
- **routes/**: API route definitions organized by domain (auth, batch, etc.).
- **config.py**: Application configuration (Development/Production).
