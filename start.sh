#!/bin/bash
# =================================================================
# SimOps Startup Script (Linux/Mac)
# =================================================================
# Starts the SimOps Docker appliance with all services
#
# Usage:
#   ./start.sh              # Start with defaults
#   ./start.sh --build      # Rebuild images
#   ./start.sh --scale 8    # Start with 8 workers
# =================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}"
echo "============================================================"
echo "   SIMOPS - Thermal Analysis Vending Machine"
echo "============================================================"
echo -e "${NC}"

# Parse arguments
BUILD_FLAG=""
SCALE_WORKERS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --build)
            BUILD_FLAG="--build"
            shift
            ;;
        --scale)
            SCALE_WORKERS="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Create directories
echo "Creating directories..."
mkdir -p input output logs

# Copy .env if not exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}Creating .env from .env.example...${NC}"
    cp .env.example .env
fi

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo -e "${RED}Error: Docker daemon is not running${NC}"
    exit 1
fi

# Start services
echo ""
echo "Starting SimOps services..."

if [ -n "$SCALE_WORKERS" ]; then
    echo -e "${YELLOW}Scaling to $SCALE_WORKERS workers${NC}"
    docker-compose up -d $BUILD_FLAG --scale worker=$SCALE_WORKERS
else
    docker-compose up -d $BUILD_FLAG
fi

# Wait for services
echo ""
echo "Waiting for services to be healthy..."
sleep 5

# Show status
echo ""
docker-compose ps

echo ""
echo -e "${GREEN}============================================================"
echo "   SimOps is running!"
echo "============================================================${NC}"
echo ""
echo "   Dashboard:  http://localhost:9181"
echo "   Input:      $(pwd)/input/"
echo "   Output:     $(pwd)/output/"
echo ""
echo "   Drop STEP files into the input folder to start simulations."
echo ""
echo "   Commands:"
echo "     docker-compose logs -f          # View logs"
echo "     docker-compose down             # Stop services"
echo "     docker-compose ps               # Check status"
echo ""
