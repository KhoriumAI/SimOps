"""
Pytest Configuration for E2E Test Suite
========================================
Defines fixtures, markers, and test configuration.
"""

import pytest
import sys
import json
from pathlib import Path
from typing import List

# Add SimOps root to path
TASK_ROOT = Path(__file__).parent
SIMOPS_ROOT = TASK_ROOT.parent.parent
sys.path.insert(0, str(SIMOPS_ROOT))

from test_models import E2ETestCase


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (require solver execution)"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (full pipeline)"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as fast unit tests"
    )


@pytest.fixture(scope="session")
def simops_root():
    """Path to SimOps repository root."""
    return SIMOPS_ROOT


@pytest.fixture(scope="session")
def task_root():
    """Path to TASK_05 directory."""
    return TASK_ROOT


@pytest.fixture(scope="session")
def test_data_dir(task_root):
    """Path to test_data directory."""
    return task_root / "test_data"


@pytest.fixture(scope="session")
def test_cases(test_data_dir) -> List[E2ETestCase]:
    """Load test cases from JSON file."""
    test_cases_file = test_data_dir / "test_cases.json"
    
    if not test_cases_file.exists():
        pytest.fail(f"Test cases file not found: {test_cases_file}")
    
    with open(test_cases_file, 'r') as f:
        case_dicts = json.load(f)
    
    return [E2ETestCase(**case) for case in case_dicts]


@pytest.fixture(scope="function")
def temp_output_dir(tmp_path):
    """Temporary directory for test outputs."""
    output_dir = tmp_path / "e2e_output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture(scope="session")
def mock_ai_mode():
    """Enable mock mode for AI generation (no real API calls)."""
    return True


@pytest.fixture(scope="session")
def ai_generator(mock_ai_mode):
    """AI setup generator instance."""
    sys.path.insert(0, str(SIMOPS_ROOT))
    from core.ai_setup.generator import AISetupGenerator
    return AISetupGenerator()
