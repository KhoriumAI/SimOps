
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from api_server import create_app

def test_cors():
    # Force development env for consistency in test
    os.environ['FLASK_ENV'] = 'development'
    app = create_app()
    
    # Simulate requests from the AWS origins
    alb_origin = "http://webdev-alb-1882895883.us-west-1.elb.amazonaws.com"
    ip_origin = "http://54.67.128.4"
    new_elastic_ip = "http://10.0.8.121"
    
    test_origins = [alb_origin, ip_origin, new_elastic_ip, "http://localhost:5173"]
    
    with app.test_client() as client:
        for origin in test_origins:
            # Test health check with origin
            response = client.options('/api/health', headers={
                'Origin': origin, 
                'Access-Control-Request-Method': 'GET',
                'Access-Control-Request-Headers': 'Content-Type'
            })
            print(f"Origin Test ({origin}): {response.status_code}")
            allowed = response.headers.get('Access-Control-Allow-Origin')
            print(f"Allow-Origin: {allowed}")
            
            if allowed != origin:
                print(f"FAIL: Expected {origin}, got {allowed}")
                sys.exit(1)

    print("\nCORS tests passed for all AWS and local origins!")

if __name__ == "__main__":
    try:
        test_cors()
    except Exception as e:
        print(f"CORS test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
