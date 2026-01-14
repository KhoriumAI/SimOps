#!/bin/bash
# Test script to capture HTTP 429 response
# Usage: ./test_429_response.sh <email> <password>

EMAIL=$1
PASSWORD=$2

if [ -z "$EMAIL" ] || [ -z "$PASSWORD" ]; then
    echo "Usage: ./test_429_response.sh <email> <password>"
    exit 1
fi

echo "======================================================================"
echo "HTTP 429 RATE LIMIT TEST"
echo "======================================================================"
echo ""
echo "Step 1: Login and get token..."
TOKEN=$(curl -s -X POST http://localhost:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"$EMAIL\",\"password\":\"$PASSWORD\"}" \
  | python3 -c "import sys, json; print(json.load(sys.stdin)['access_token'])")

if [ -z "$TOKEN" ] || [ "$TOKEN" == "None" ]; then
    echo "❌ Login failed"
    exit 1
fi

echo "✅ Logged in successfully"
echo ""

echo "Step 2: Get user projects..."
PROJECTS=$(curl -s -X GET http://localhost:5000/api/projects \
  -H "Authorization: Bearer $TOKEN")

PROJECT_ID=$(echo $PROJECTS | python3 -c "import sys, json; projects = json.load(sys.stdin).get('projects', []); print([p['id'] for p in projects if p.get('status') in ['uploaded', 'completed', 'error']][0] if projects else '')")

if [ -z "$PROJECT_ID" ]; then
    echo "❌ No ready projects found"
    exit 1
fi

echo "✅ Found project: $PROJECT_ID"
echo ""

echo "Step 3: Attempt to generate mesh (should return 429 if quota exceeded)..."
echo ""
echo "Request:"
echo "  POST /api/projects/$PROJECT_ID/generate"
echo "  Authorization: Bearer $TOKEN"
echo ""

RESPONSE=$(curl -s -w "\nHTTP_STATUS:%{http_code}" -X POST \
  "http://localhost:5000/api/projects/$PROJECT_ID/generate" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"mesh_strategy":"Tet (Fast)"}')

HTTP_STATUS=$(echo "$RESPONSE" | grep "HTTP_STATUS" | cut -d: -f2)
BODY=$(echo "$RESPONSE" | sed '/HTTP_STATUS/d')

echo "Response:"
echo "  Status Code: $HTTP_STATUS"
echo "  Body:"
echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
echo ""

if [ "$HTTP_STATUS" == "429" ]; then
    echo "======================================================================"
    echo "✅ HTTP 429 RECEIVED - RATE LIMIT WORKING!"
    echo "======================================================================"
    echo ""
    echo "📸 Take a screenshot of the above response"
else
    echo "======================================================================"
    echo "⚠️  Expected HTTP 429 but got $HTTP_STATUS"
    echo "======================================================================"
    echo ""
    echo "Current quota may not be exceeded yet."
    echo "Check current job count and set DEFAULT_JOB_QUOTA accordingly."
fi

