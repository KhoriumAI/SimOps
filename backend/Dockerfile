# SimOps Backend API Server Dockerfile
# =====================================
# Flask + Gunicorn API server for mesh generation
# Build from project root: docker build -f backend/Dockerfile .

# =================================================================
# STAGE 1: Build dependencies
# =================================================================
FROM python:3.10-slim-bullseye AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY backend/requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# =================================================================
# STAGE 2: Production image
# =================================================================
FROM python:3.10-slim-bullseye AS production

LABEL maintainer="SimOps Team"
LABEL description="SimOps Backend API Server"
LABEL version="1.0"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy wheels from builder and install
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* \
    && rm -rf /wheels

# Create non-root user
RUN useradd --create-home --shell /bin/bash simops
WORKDIR /app

# Copy application code (from project root context)
COPY --chown=simops:simops backend/ ./backend/
COPY --chown=simops:simops core/ ./core/
COPY --chown=simops:simops simops/ ./simops/
COPY --chown=simops:simops apps/ ./apps/
COPY --chown=simops:simops strategies/ ./strategies/

# Create required directories
RUN mkdir -p /app/backend/instance /app/backend/outputs /app/uploads /app/output /app/data \
    && chown -R simops:simops /app

# Switch to non-root user
USER simops

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV FLASK_APP=backend.api_server:create_app

EXPOSE 5000

WORKDIR /app/backend

# Start with Gunicorn (eventlet for SocketIO support)
CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", "--bind", "0.0.0.0:5000", "api_server:create_app()"]
