#!/usr/bin/env python3
"""
Environment Variable Audit Script

Scans backend code for os.getenv() and os.environ.get() calls,
then verifies all referenced variables exist in .env.example.

Exit code 0 if all variables are documented, 1 otherwise.
"""
import os
import re
import sys
from pathlib import Path
from typing import Set


def find_env_vars_in_code(backend_dir: Path) -> Set[str]:
    """Scan Python files for environment variable references."""
    env_vars: Set[str] = set()
    
    # Pattern to match os.getenv('VAR') or os.environ.get('VAR')
    patterns = [
        r"os\.getenv\(['\"]([^'\"]+)['\"]",
        r"os\.environ\.get\(['\"]([^'\"]+)['\"]",
    ]
    
    for py_file in backend_dir.rglob("*.py"):
        # Skip __pycache__ and virtual environments
        if '__pycache__' in str(py_file) or 'venv' in str(py_file):
            continue
            
        try:
            content = py_file.read_text(encoding='utf-8')
            for pattern in patterns:
                matches = re.findall(pattern, content)
                env_vars.update(matches)
        except Exception as e:
            print(f"Warning: Could not read {py_file}: {e}", file=sys.stderr)
    
    return env_vars


def parse_env_example(env_example_path: Path) -> Set[str]:
    """Parse .env.example and extract variable names."""
    if not env_example_path.exists():
        return set()
    
    env_vars: Set[str] = set()
    content = env_example_path.read_text(encoding='utf-8')
    
    for line in content.split('\n'):
        line = line.strip()
        # Skip comments and empty lines
        if not line or line.startswith('#'):
            continue
        # Extract variable name (before =)
        if '=' in line:
            var_name = line.split('=')[0].strip()
            env_vars.add(var_name)
    
    return env_vars


def main() -> int:
    """Main entry point for the environment variable audit."""
    project_root = Path(__file__).parent.parent
    backend_dir = project_root / "backend"
    env_example = backend_dir / ".env.example"
    
    if not backend_dir.exists():
        print(f"Error: Backend directory not found: {backend_dir}", file=sys.stderr)
        return 1
    
    if not env_example.exists():
        print(f"Error: .env.example not found: {env_example}", file=sys.stderr)
        print("Please create backend/.env.example with all required environment variables.", file=sys.stderr)
        return 1
    
    # Find all env vars used in code
    code_vars = find_env_vars_in_code(backend_dir)
    
    # Find all vars in .env.example
    example_vars = parse_env_example(env_example)
    
    # Find missing vars
    missing = code_vars - example_vars
    
    if missing:
        print("❌ FAILED: Environment variables used in code but missing from .env.example:")
        for var in sorted(missing):
            print(f"  - {var}")
        print(f"\nTotal: {len(missing)} missing variable(s)")
        return 1
    else:
        print(f"✅ PASSED: All {len(code_vars)} environment variables are documented in .env.example")
        return 0


if __name__ == "__main__":
    sys.exit(main())


