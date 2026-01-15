#!/bin/bash

echo "=========================================="
echo "Task Completion Verification"
echo "=========================================="
echo ""

ERRORS=0

# 1. Check webhook endpoint
echo "1. Checking webhook endpoint..."
if grep -q "/api/webhooks/modal" backend/routes/webhooks.py; then
    echo "   ✅ Webhook endpoint exists"
else
    echo "   ❌ Webhook endpoint missing"
    ERRORS=$((ERRORS + 1))
fi

# 2. Check backend polling removal (mesh generation only, preview can still poll)
echo "2. Checking backend polling removal..."
# Check for polling loop in mesh generation (should not have while True with get_job_result)
if grep -A 5 "Wait for results with polling" backend/api_server.py > /dev/null 2>&1; then
    echo "   ❌ Backend still polling (polling loop found)"
    ERRORS=$((ERRORS + 1))
elif grep -A 10 "Modal job.*spawned" backend/api_server.py | grep -q "get_job_result.*timeout.*2"; then
    echo "   ❌ Backend still polling (get_job_result with timeout=2 found)"
    ERRORS=$((ERRORS + 1))
else
    echo "   ✅ Backend polling removed for mesh generation"
fi

# 3. Check send_webhook function
echo "3. Checking Modal webhook implementation..."
if grep -q "def send_webhook" backend/modal_service.py; then
    echo "   ✅ send_webhook function exists"
else
    echo "   ❌ send_webhook function missing"
    ERRORS=$((ERRORS + 1))
fi

# 4. Check WebSocket implementation
echo "4. Checking WebSocket implementation..."
if grep -q "subscribe_logs" backend/routes/webhooks.py; then
    echo "   ✅ WebSocket log subscription exists"
else
    echo "   ❌ WebSocket log subscription missing"
    ERRORS=$((ERRORS + 1))
fi

# 5. Check signature verification
echo "5. Checking webhook signature verification..."
if grep -q "verify_webhook_signature" backend/routes/webhooks.py; then
    echo "   ✅ Signature verification implemented"
else
    echo "   ❌ Signature verification missing"
    ERRORS=$((ERRORS + 1))
fi

# 6. Check CloudWatch logging
echo "6. Checking CloudWatch logging..."
if grep -q "put_log_events" backend/modal_service.py; then
    echo "   ✅ CloudWatch logging implemented"
else
    echo "   ❌ CloudWatch logging missing"
    ERRORS=$((ERRORS + 1))
fi

echo ""
echo "=========================================="
if [ $ERRORS -eq 0 ]; then
    echo "✅ All checks passed! Task is complete."
    exit 0
else
    echo "❌ Found $ERRORS issue(s) that need to be fixed."
    exit 1
fi
echo "=========================================="

