#!/bin/bash

# Quick test script for webhook endpoint

echo "=========================================="
echo "Testing Webhook Endpoint"
echo "=========================================="
echo ""

# Test 1: Health check
echo "1. Testing backend health..."
curl -s http://localhost:5000/api/health | python3 -m json.tool
echo -e "\n"

# Test 2: Webhook without signature (dev mode)
echo "2. Testing webhook endpoint (no signature)..."
JOB_ID="test-job-$(date +%s)"
curl -X POST http://localhost:5000/api/webhooks/modal \
  -H "Content-Type: application/json" \
  -d "{
    \"job_id\": \"$JOB_ID\",
    \"status\": \"completed\",
    \"result\": {
      \"success\": true,
      \"s3_output_path\": \"s3://test-bucket/test.msh\",
      \"strategy\": \"tet_hxt\",
      \"total_nodes\": 1000,
      \"total_elements\": 5000,
      \"quality_metrics\": {
        \"min_quality\": 0.85,
        \"avg_quality\": 0.95
      },
      \"metrics\": {
        \"total_time_seconds\": 45.2
      }
    }
  }" | python3 -m json.tool 2>/dev/null || echo "Response received"
echo -e "\n"

# Test 3: Webhook with invalid job_id
echo "3. Testing webhook with invalid job_id..."
curl -X POST http://localhost:5000/api/webhooks/modal \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "non-existent-job",
    "status": "completed"
  }' | python3 -m json.tool 2>/dev/null || echo "Response received"
echo -e "\n"

echo "=========================================="
echo "Tests Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Open http://localhost:3000 in browser"
echo "2. Open DevTools → Network tab"
echo "3. Filter by 'Fetch/XHR'"
echo "4. Upload a CAD file and start mesh generation"
echo "5. Verify NO polling requests to /api/projects/{id}/status"
echo "6. Check Console for WebSocket connection messages"

