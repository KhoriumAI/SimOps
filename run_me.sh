#!/bin/bash

echo ""
echo "========================================================"
echo "  SIMOPS ENGINEERING WORKBENCH - INSTALLER / RUNNER"
echo "========================================================"
echo ""

# Check for Docker
if ! command -v docker &> /dev/null
then
    echo "[ERROR] Docker not found. Please install Docker."
    echo "Visit: https://www.docker.com/products/docker-desktop"
    exit 1
fi

echo "[1/4] Pulling latest images (optional)..."
docker-compose pull

echo "[2/4] Starting SimOps services in background..."
docker-compose up -d

echo "[3/4] Waiting for services to initialize..."
echo "(This may take 15-30 seconds on first run)"
sleep 15

echo "[4/4] Launching SimOps Workbench in your browser..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    open http://localhost:3000
else
    xdg-open http://localhost:3000 || echo "Please open http://localhost:3000 manually"
fi

echo ""
echo "========================================================"
echo "  SIMOPS IS RUNNING"
echo "  API: http://localhost:8000"
echo "  Frontend: http://localhost:3000"
echo "========================================================"
echo ""
echo "To stop SimOps, run 'docker-compose down'."
echo ""
