FROM python:3.10-slim
WORKDIR /app

# Install system dependencies (e.g. for Gmsh/OpenFOAM if needed, or just basic utils)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy backend code
COPY . /app
COPY requirements.txt /app/requirements.txt

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Install Gunicorn for production serving
RUN pip install gunicorn

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

# Expose port
EXPOSE 5000

# Run with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "api_server:app"]
