# Lightweight python image for fast builds
FROM python:3.9-slim

WORKDIR /app

# No dependencies needed for this dummy worker
# If you need them later, uncomment:
# COPY requirements.txt .
# RUN pip install -r requirements.txt

# Copy the worker script
COPY dummy_worker.py .

# Entrypoint allows arguments to be passed directly
ENTRYPOINT ["python", "dummy_worker.py"]
