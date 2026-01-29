#!/usr/bin/env python3
"""
Example verification script following the Khorium Developer Rulebook.

This demonstrates the test-first protocol:
1. Define success criteria
2. Create minimal reproducible test
3. Return 0 on success, non-zero on failure
"""

import sys
import subprocess
from pathlib import Path


def main() -> int:
    """
    Verify that a simple Python function works correctly.
    
    Success criteria:
    - Function exists and is importable
    - Function returns expected output for known input
    - Function handles edge cases (empty input, None, etc.)
    
    Returns:
        0 on success, 1 on failure
    """
    print("🔍 Running verification checks...")
    
    try:
        # Example: Verify a utility function
        # from backend.utils import sanitize_filename
        
        # Test 1: Normal input
        # result = sanitize_filename("test file.stl")
        # assert result == "test_file.stl", f"Expected 'test_file.stl', got '{result}'"
        # print("✅ Test 1: Normal input passed")
        
        # Test 2: Edge case - empty string
        # result = sanitize_filename("")
        # assert result == "unnamed", f"Expected 'unnamed', got '{result}'"
        # print("✅ Test 2: Empty string passed")
        
        # Test 3: Edge case - special characters
        # result = sanitize_filename("../../../etc/passwd")
        # assert "/" not in result, f"Path traversal vulnerability: {result}"
        # print("✅ Test 3: Security check passed")
        
        print("\n✅ All checks passed!")
        return 0
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return 1
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
