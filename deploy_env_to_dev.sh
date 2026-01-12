#!/bin/bash
# Deploy .env configuration to AWS Dev Environment
# Run this script via SSM on the dev EC2 instance

set -e

echo "=== Deploying Modal Configuration to Dev Environment ==="

# Navigate to backend directory
cd ~/MeshPackageLean/backend || cd ~/backend || { echo "Error: backend directory not found"; exit 1; }

echo "Current directory: $(pwd)"

# Backup existing .env if it exists
if [ -f .env ]; then
    echo "Backing up existing .env to .env.backup.$(date +%Y%m%d_%H%M%S)"
    cp .env .env.backup.$(date +%Y%m%d_%H%M%S)
fi

# Create new .env file with Modal configuration
echo "Creating new .env file..."
cat > .env << 'EOF'
# Khorium MeshGen - AWS Dev Environment Configuration
# Auto-generated - DO NOT COMMIT TO GIT

# Flask Environment
FLASK_ENV=development
FLASK_DEBUG=true
SECRET_KEY=dev-aws-secret-key-change-me
JWT_SECRET_KEY=dev-aws-jwt-secret-change-me

# Modal Compute - ENABLED
USE_MODAL_COMPUTE=true
MODAL_APP_NAME=khorium-production
MODAL_MESH_FUNCTION=generate_mesh
MODAL_PREVIEW_FUNCTION=generate_preview_mesh

# AWS S3 - Required for Modal
USE_S3=true
AWS_REGION=us-west-1
S3_BUCKET_NAME=muaz-webdev-assets

# CORS - Development origins
CORS_ORIGINS=http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,https://app.khorium.ai

# File Upload Limits
MAX_FILE_SIZE_MB=500
BATCH_MAX_FILES=10
BATCH_PARALLEL_JOBS=6

# Compute Backend
COMPUTE_BACKEND=auto
EOF

echo "✅ .env file created"

# Set proper permissions
chmod 600 .env
echo "✅ Permissions set to 600"

# Show configuration (without secrets)
echo ""
echo "=== Configuration Summary ==="
grep -v "SECRET" .env | grep -v "AWS_" | head -20
echo ""

# Restart Gunicorn to pick up new environment
echo "=== Restarting Gunicorn ==="
sudo systemctl restart gunicorn

# Wait for service to start
sleep 3

# Check service status
echo "=== Service Status ==="
sudo systemctl status gunicorn --no-pager | head -10

# Test API health
echo ""
echo "=== Testing API Health ==="
curl -s http://localhost:3000/api/health || echo "Warning: Health check failed"

echo ""
echo "=== Deployment Complete ==="
echo "Modal configuration has been deployed to dev environment"
echo "To verify Modal is being used, check logs:"
echo "  sudo journalctl -u gunicorn -f"
echo "Look for: [MESH GEN] Using MODAL compute"
