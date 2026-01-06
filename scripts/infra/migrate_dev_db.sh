#!/bin/bash
# Run database migration on AWS Dev to add Modal columns
# Execute via SSM on i-0070fec97d713f06f

set -e

echo "=== Running Database Migration for Modal Support ==="

# Navigate to backend directory
cd /home/ec2-user/MeshPackageLean/backend || cd /home/ec2-user/backend || { echo "Error: backend directory not found"; exit 1; }

echo "Current directory: $(pwd)"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
fi

# Run migration script
echo ""
echo "Running migration script..."
python3 run_migration.py

# Check exit code
if [ $? -eq 0 ]; then
    echo "✅ Migration completed successfully"
else
    echo "❌ Migration failed"
    exit 1
fi

# Restart Gunicorn to pick up changes
echo ""
echo "Restarting Gunicorn..."
sudo systemctl restart gunicorn

# Wait for service to start
sleep 3

# Check service status
echo "✅ Service restarted"
sudo systemctl status gunicorn --no-pager | head -10

# Test API
echo ""
echo "Testing API health..."
curl -s http://localhost:3000/api/health || echo "Health check pending..."

echo ""
echo "=== Migration Complete ==="
echo "Modal columns added to mesh_results table:"
echo "  - modal_job_id VARCHAR(100)"
echo "  - modal_status VARCHAR(20)"  
echo "  - modal_started_at TIMESTAMP"
echo "  - modal_completed_at TIMESTAMP"
