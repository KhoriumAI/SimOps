/**
 * WebSocket Readiness Checker
 * 
 * Run this in browser console BEFORE recording to verify WebSocket is working.
 * 
 * Usage:
 * 1. Open browser console (F12)
 * 2. Copy-paste this entire script
 * 3. Press Enter
 * 4. Check the output
 */

(function() {
  console.log('%c🔍 WebSocket Readiness Check', 'font-size: 16px; font-weight: bold; color: blue;');
  console.log('='.repeat(50));
  
  // Check 1: WebSocket URL
  const wsUrl = window.WS_URL || 'http://localhost:5000';
  console.log('✅ WebSocket URL:', wsUrl);
  
  // Check 2: Socket.IO available
  if (typeof io === 'undefined') {
    console.error('❌ Socket.IO not loaded! Check if socket.io-client is installed.');
    return;
  }
  console.log('✅ Socket.IO library loaded');
  
  // Check 3: Test connection
  console.log('🔄 Testing WebSocket connection...');
  const testSocket = io(wsUrl, {
    transports: ['websocket', 'polling'],
    timeout: 5000
  });
  
  let connected = false;
  let testComplete = false;
  
  const timeout = setTimeout(() => {
    if (!testComplete) {
      testComplete = true;
      testSocket.disconnect();
      console.error('❌ Connection test timed out after 5 seconds');
      console.log('💡 Check if backend is running on port 5000');
    }
  }, 5000);
  
  testSocket.on('connect', () => {
    connected = true;
    console.log('✅ WebSocket connection successful!');
    console.log('   Socket ID:', testSocket.id);
    
    setTimeout(() => {
      if (!testComplete) {
        testComplete = true;
        clearTimeout(timeout);
        testSocket.disconnect();
        console.log('\n%c✅ ALL CHECKS PASSED - Ready to record!', 'font-size: 14px; font-weight: bold; color: green;');
        console.log('\n📋 Next steps:');
        console.log('1. Open Network tab (F12 → Network)');
        console.log('2. Filter by "Fetch/XHR"');
        console.log('3. Clear existing requests');
        console.log('4. Enable "Preserve log"');
        console.log('5. Start recording!');
      }
    }, 1000);
  });
  
  testSocket.on('connect_error', (err) => {
    if (!testComplete) {
      testComplete = true;
      clearTimeout(timeout);
      console.error('❌ WebSocket connection failed:', err.message);
      console.log('💡 Troubleshooting:');
      console.log('   - Is backend running? (python api_server.py)');
      console.log('   - Is CORS configured correctly?');
      console.log('   - Check backend logs for errors');
    }
  });
  
  testSocket.on('disconnect', () => {
    if (connected && testComplete) {
      console.log('✅ Test connection closed (expected)');
    }
  });
})();

