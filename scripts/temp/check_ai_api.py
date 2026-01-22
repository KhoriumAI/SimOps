#!/usr/bin/env python3
"""
AI Generator API Endpoint Verification
======================================
Test that /api/ai/generate-config endpoint works correctly.
Following Khorium Rulebook Rule #1: Write verification BEFORE implementation.
"""

import sys
import requests
import json
from pathlib import Path

API_BASE = "http://localhost:5000"

def test_health():
    """Verify API server is running."""
    print("Checking API health...")
    try:
        response = requests.get(f"{API_BASE}/api/health", timeout=2)
        assert response.status_code == 200
        print("✅ API server is healthy")
        return True
    except requests.exceptions.ConnectionError:
        print("❌ API server not running. Start with: python simops-backend/api_server.py")
        return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_ai_generate_config_endpoint():
    """Verify /api/ai/generate-config endpoint exists and returns valid SimConfig."""
    print("Testing /api/ai/generate-config endpoint...")
    try:
        payload = {
            "prompt": "Simulate this heatsink with 50W heat source",
            "cad_file": "test_heatsink.step",
            "use_mock": True
        }
        
        response = requests.post(
            f"{API_BASE}/api/ai/generate-config",
            json=payload,
            timeout=5
        )
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert "config" in data, "Response should contain 'config' key"
        
        config = data["config"]
        assert "job_name" in config, "Config should have job_name"
        assert "cad_file" in config, "Config should have cad_file"
        assert "materials" in config, "Config should have materials"
        assert "boundary_conditions" in config, "Config should have boundary_conditions"
        
        print(f"✅ AI endpoint returned valid config: {config['job_name']}")
        return True
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API")
        return False
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_ai_error_handling():
    """Verify endpoint handles invalid requests gracefully."""
    print("Testing error handling...")
    try:
        # Missing required fields
        response = requests.post(
            f"{API_BASE}/api/ai/generate-config",
            json={},
            timeout=5
        )
        
        assert response.status_code in [400, 422], "Should return error for invalid request"
        print("✅ Error handling works")
        return True
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("AI Generator API Verification")
    print("=" * 60)
    print()
    
    tests = [
        test_health,
        test_ai_generate_config_endpoint,
        test_ai_error_handling
    ]
    
    results = [test() for test in tests]
    
    print()
    print("=" * 60)
    if all(results):
        print("🎉 ALL API TESTS PASSED")
        print("=" * 60)
        sys.exit(0)
    else:
        print("❌ API TESTS FAILED")
        print("=" * 60)
        sys.exit(1)
