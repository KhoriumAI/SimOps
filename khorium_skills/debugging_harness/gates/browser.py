import sys
import time

def run_browser_check(url="http://localhost:3001", timeout_ms=5000):
    """
    Gate 2: Headless Browser Check for Physics Bugs
    Monitors the console for errors, NaNs, and context loss.
    """
    print(f"[*] Running Gate 2 (Browser) on {url}...")
    
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False, "Gate 2 Error: 'playwright' package not found. Please run 'pip install playwright && playwright install' to enable Gate 2."

    logs = []
    errors = []
    
    try:
        with sync_playwright() as p:
            # Invisible browser
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            # Capture console logs
            def handle_console(msg):
                text = msg.text
                logs.append(text)
                # Detection logic for "Antigravity" issues
                if "Error" in text or "NaN" in text or "Context Lost" in text or "Exception" in text:
                    errors.append(f"Console {msg.type}: {text}")

            page.on("console", handle_console)
            # Also catch page errors (uncaught exceptions)
            page.on("pageerror", lambda exc: errors.append(f"Page Error: {exc}"))

            try:
                print(f"    Navigating to {url}...")
                page.goto(url, wait_until="networkidle", timeout=30000)
                # Wait for physics to settle or crash
                print(f"    Monitoring for {timeout_ms/1000}s...")
                time.sleep(timeout_ms / 1000) 
            except Exception as e:
                errors.append(f"Navigation/Timeout Error: {str(e)}")
                
            browser.close()
    except Exception as e:
        return False, f"Playwright Execution Error: {str(e)}"
    
    success = len(errors) == 0
    return success, "\n".join(errors) if not success else "No browser errors detected."

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:3001"
    success, result = run_browser_check(url)
    if success:
        print(f"✅ Gate 2 Passed: {result}")
    else:
        print(f"❌ Gate 2 Failed:\n{result}")
