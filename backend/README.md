# Backend Directory

The Khorium MeshGen backend server, implemented using FastAPI. It manages job queues, database storage, and orchestrates workers.

## Core Files

- `api_server.py`: The entry point for the FastAPI application. Defines REST endpoints.
- `models.py`: SQLAlchemy database models for Jobs, Results, and Users.
- `tasks.py`: Celery/Redis task definitions for background processing.
- `celery_app.py`: Configuration for the Celery worker and message broker.
- `storage.py`: Interaction with local and cloud storage (S3/Azure).
- `config.py`: Backend-specific settings (database URLs, API keys).

## API Documentation

When the server is running, documentation is available at `/docs` (Swagger UI) or `/redoc`.

## Local Development

1. Install backend requirements:
   ```bash
   pip install -r backend/requirements.txt
   ```
2. Run the server:
   ```bash
   python backend/api_server.py
   ```
3. (Optional) Run workers:
   ```bash
   celery -A backend.tasks worker --loglevel=info
   ```
