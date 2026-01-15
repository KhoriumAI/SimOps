#!/bin/bash
# Cleanup script for EC2 instance disk space

echo "=== Checking disk usage before cleanup ==="
df -h

echo ""
echo "=== Finding large directories ==="
du -sh /home/ec2-user/* 2>/dev/null | sort -hr | head -20

echo ""
echo "=== Cleaning up mesh outputs ==="
rm -rf /home/ec2-user/backend/outputs/* 2>/dev/null || true
rm -rf /home/ec2-user/MeshPackageLean/backend/outputs/* 2>/dev/null || true
rm -rf /home/ec2-user/MeshPackageLean/output/* 2>/dev/null || true

echo ""
echo "=== Cleaning up uploads ==="
rm -rf /home/ec2-user/backend/uploads/* 2>/dev/null || true
rm -rf /home/ec2-user/MeshPackageLean/backend/uploads/* 2>/dev/null || true

echo ""
echo "=== Cleaning pip cache ==="
pip cache purge 2>/dev/null || true
rm -rf /home/ec2-user/.cache/pip 2>/dev/null || true

echo ""
echo "=== Cleaning Python cache ==="
find /home/ec2-user -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

echo ""
echo "=== Cleaning system logs (requires sudo) ==="
sudo journalctl --vacuum-time=7d

echo ""
echo "=== Checking disk usage after cleanup ==="
df -h
