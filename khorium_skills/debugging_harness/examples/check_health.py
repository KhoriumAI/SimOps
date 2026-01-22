import requests
import sys

def check_system_health():
    """Example success condition: Checks if the backend health endpoint returns healthy."""
    print("Checking backend health...")
    try:
        response = requests.get("http://localhost:5000/api/health", timeout=2)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "healthy":
                print("✅ Backend is healthy.")
                return True
            else:
                print(f"❌ Backend is unhealthy: {data.get('status')}")
        else:
            print(f"❌ Backend returned status code {response.status_code}")
    except Exception as e:
        print(f"❌ Failed to connect to backend: {str(e)}")
    
    return False

if __name__ == "__main__":
    if check_system_health():
        sys.exit(0)
    else:
        sys.exit(1)
