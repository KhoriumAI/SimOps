#!/bin/bash
# Quick test for HTTP 429 after server restart

echo "======================================================================"
echo "TESTING HTTP 429 RATE LIMIT RESPONSE"
echo "======================================================================"
echo ""

# Login
echo "1. Logging in..."
TOKEN=$(curl -s -X POST http://localhost:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"johndoe@gmail.com","password":"qwerty123"}' \
  | python3 -c "import sys, json; print(json.load(sys.stdin)['access_token'])" 2>/dev/null)

if [ -z "$TOKEN" ]; then
    echo "❌ Login failed"
    exit 1
fi

echo "✅ Logged in successfully"
echo ""

# Get ready project
echo "2. Finding ready project..."
PROJECT_ID=$(curl -s -X GET http://localhost:5000/api/projects \
  -H "Authorization: Bearer $TOKEN" \
  | python3 -c "import sys, json; projects = json.load(sys.stdin).get('projects', []); ready = [p for p in projects if p.get('status') in ['uploaded', 'completed', 'error']]; print(ready[0]['id'] if ready else '')" 2>/dev/null)

if [ -z "$PROJECT_ID" ]; then
    echo "❌ No ready projects found"
    exit 1
fi

echo "✅ Found project: $PROJECT_ID"
echo ""

# Attempt job generation (should return 429)
echo "3. Attempting to generate job (should return HTTP 429)..."
echo ""
RESPONSE=$(curl -s -w "\nHTTP_STATUS:%{http_code}" -X POST \
  "http://localhost:5000/api/projects/$PROJECT_ID/generate" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"mesh_strategy":"Tet (Fast)"}')

HTTP_STATUS=$(echo "$RESPONSE" | grep "HTTP_STATUS" | cut -d: -f2)
BODY=$(echo "$RESPONSE" | sed '/HTTP_STATUS/d')

echo "Response Status: $HTTP_STATUS"
echo "Response Body:"
echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
echo ""

if [ "$HTTP_STATUS" == "429" ]; then
    echo "======================================================================"
    echo "✅ SUCCESS: HTTP 429 RECEIVED - RATE LIMITING WORKING!"
    echo "======================================================================"
    echo ""
    echo "📸 Take a screenshot of the above HTTP 429 response"
else
    echo "======================================================================"
    echo "⚠️  Expected HTTP 429 but got $HTTP_STATUS"
    echo "======================================================================"
    echo ""
    echo "Current quota may not be exceeded yet."
    echo "Check: DEFAULT_JOB_QUOTA in .env and restart server if changed."
fi
