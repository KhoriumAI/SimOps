# Testing Guide: Webhook & WebSocket Implementation

This guide helps you test the new webhook and WebSocket functionality that replaces polling.

## Prerequisites

1. **Backend running**: `http://localhost:5000`
2. **Frontend running**: `http://localhost:3000`
3. **Browser DevTools**: Open Network tab to verify no polling requests

## Test 1: Verify Polling is Removed

### Steps:
1. Open browser DevTools (F12)
2. Go to **Network** tab
3. Filter by **Fetch/XHR**
4. Navigate to `http://localhost:3000`
5. Upload a CAD file and start mesh generation

### Expected Result:
- ✅ **NO** requests to `/api/projects/{id}/status` every 2 seconds
- ✅ Only initial requests (upload, generate, etc.)
- ✅ WebSocket connection visible in Network tab (WS filter)

### How to Verify:
- Look for `setInterval` calls in Console (should be none for status polling)
- Check Network tab - no repeated status requests

---

## Test 2: WebSocket Connection

### Steps:
1. Open browser DevTools → **Console** tab
2. Navigate to `http://localhost:3000`
3. Upload a CAD file
4. Start mesh generation

### Expected Console Output:
```
[WS] Connected to project WebSocket
[WS] Subscribed to project: {project_id}
[WS] Subscribing to logs for job: {job_id}
```

### How to Verify:
- Check Console for WebSocket connection messages
- Network tab → WS filter → Should see WebSocket connection
- Connection should show as "101 Switching Protocols"

---

## Test 3: Real-time Log Streaming

### Steps:
1. Start a mesh generation job
2. Watch the Terminal/Console component in the UI
3. Observe logs appearing line-by-line

### Expected Result:
- ✅ Logs appear in real-time (not every 2 seconds)
- ✅ Logs stream as they're generated
- ✅ Terminal auto-scrolls to latest log
- ✅ Logs include timestamps

### How to Verify:
- Open browser Console → Look for `[WS] log_line` events
- Terminal component should show logs immediately
- No delay between log generation and display

---

## Test 4: Webhook Endpoint

### Manual Test with curl:

```bash
# Test webhook endpoint (without signature - should work in dev)
curl -X POST http://localhost:5000/api/webhooks/modal \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "test-job-123",
    "status": "completed",
    "result": {
      "success": true,
      "s3_output_path": "s3://bucket/test.msh",
      "strategy": "tet_hxt",
      "total_nodes": 1000,
      "total_elements": 5000
    }
  }'
```

### Expected Response:
```json
{
  "status": "ok",
  "job_id": "test-job-123"
}
```

### With Signature (Production-like):

```bash
# Generate signature
SECRET="your-webhook-secret"
PAYLOAD='{"job_id":"test-123","status":"completed"}'
SIGNATURE=$(echo -n "$PAYLOAD" | openssl dgst -sha256 -hmac "$SECRET" | cut -d' ' -f2)

# Send with signature
curl -X POST http://localhost:5000/api/webhooks/modal \
  -H "Content-Type: application/json" \
  -H "X-Modal-Signature: sha256=$SIGNATURE" \
  -d "$PAYLOAD"
```

---

## Test 5: Job Completion via WebSocket

### Steps:
1. Start a mesh generation job
2. Keep browser tab open
3. Simulate job completion via webhook (see Test 4)
4. Watch for completion notification

### Expected Result:
- ✅ Frontend receives `job_completed` event via WebSocket
- ✅ Mesh data automatically loads
- ✅ Status updates without page refresh
- ✅ No polling requests

### How to Verify:
- Browser Console → Look for `[WS] Job completed:` message
- UI should update automatically
- Network tab → No status polling requests

---

## Test 6: CloudWatch Log Streaming (If Modal Configured)

### Prerequisites:
- Modal job running with CloudWatch logging enabled
- AWS credentials configured
- CloudWatch log group exists: `/modal/jobs/{job_id}`

### Steps:
1. Start a Modal mesh job
2. Check CloudWatch Logs console
3. Verify logs are being written
4. Verify WebSocket streams logs to frontend

### Expected Result:
- ✅ Logs appear in CloudWatch Logs
- ✅ Backend tails logs and streams to frontend
- ✅ Frontend displays logs in real-time

---

## Test 7: Full End-to-End Flow

### Steps:
1. **Upload CAD file** → Should see upload progress
2. **Start mesh generation** → Should see:
   - WebSocket connection established
   - Logs streaming in Terminal
   - Progress updates
3. **Wait for completion** → Should see:
   - Completion notification via WebSocket
   - Mesh loads automatically
   - No polling requests

### Expected Timeline:
- **0s**: Job starts, WebSocket connects
- **1-2s**: First logs appear
- **Throughout**: Logs stream continuously
- **Completion**: Instant notification, mesh loads

### Verification Checklist:
- [ ] No `setInterval` polling in Network tab
- [ ] WebSocket connection active
- [ ] Logs stream in real-time
- [ ] Completion happens instantly
- [ ] No manual refresh needed

---

## Test 8: Error Handling

### Test Webhook with Invalid Signature:

```bash
curl -X POST http://localhost:5000/api/webhooks/modal \
  -H "Content-Type: application/json" \
  -H "X-Modal-Signature: sha256=invalid" \
  -d '{"job_id":"test","status":"completed"}'
```

### Expected Result:
- ✅ Returns 401 Unauthorized (if MODAL_WEBHOOK_SECRET is set)
- ✅ Or accepts in development mode (if secret not set)

### Test WebSocket Disconnection:

1. Start a job
2. Close browser tab
3. Reopen tab
4. Navigate to same project

### Expected Result:
- ✅ WebSocket reconnects automatically
- ✅ Resubscribes to logs
- ✅ Continues receiving updates

---

## Test 9: Performance Comparison

### Before (Polling):
- Network requests: ~30 requests/minute (every 2s)
- Latency: Up to 2 seconds delay
- Server load: Constant polling

### After (WebSocket):
- Network requests: 1 WebSocket connection
- Latency: Real-time (<100ms)
- Server load: Event-driven only

### Measure:
1. Open Network tab
2. Start a 1-minute mesh job
3. Count requests:
   - **Before**: ~30 status requests
   - **After**: 1 WebSocket connection

---

## Test 10: Screen Recording Validation

As per task requirements, create a screen recording showing:

1. **Job starts** → WebSocket connects
2. **Logs appear line-by-line** → Real-time streaming
3. **Mesh completes** → Instant notification
4. **Network tab shows zero polling requests**

### Recording Checklist:
- [ ] Show browser DevTools Network tab
- [ ] Filter by Fetch/XHR
- [ ] Start mesh generation
- [ ] Show logs streaming in Terminal
- [ ] Show completion without polling
- [ ] Verify Network tab has no repeated status requests

---

## Quick Test Script

Save this as `test_webhook.sh`:

```bash
#!/bin/bash

# Test webhook endpoint
echo "Testing webhook endpoint..."
curl -X POST http://localhost:5000/api/webhooks/modal \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "test-job-'$(date +%s)'",
    "status": "completed",
    "result": {
      "success": true,
      "s3_output_path": "s3://test/test.msh",
      "strategy": "test"
    }
  }'

echo -e "\n\nTesting health endpoint..."
curl http://localhost:5000/api/health

echo -e "\n\nDone!"
```

Run: `chmod +x test_webhook.sh && ./test_webhook.sh`

---

## Troubleshooting

### WebSocket Not Connecting:
- Check backend logs for SocketIO errors
- Verify CORS settings allow frontend origin
- Check browser Console for connection errors

### Logs Not Streaming:
- Verify CloudWatch log group exists
- Check AWS credentials are configured
- Verify job_id matches between Modal and backend

### Webhook Not Working:
- Check MODAL_WEBHOOK_SECRET is set (or not set for dev)
- Verify webhook URL is correct
- Check backend logs for webhook processing errors

---

## Success Criteria Checklist

- [ ] ✅ No polling requests in Network tab
- [ ] ✅ WebSocket connection established
- [ ] ✅ Logs stream in real-time
- [ ] ✅ Job completion via WebSocket
- [ ] ✅ Webhook endpoint accepts requests
- [ ] ✅ Signature verification works (if secret set)
- [ ] ✅ Screen recording shows zero polling requests

