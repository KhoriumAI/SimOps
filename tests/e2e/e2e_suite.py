"""
E2E Test Suite Orchestration
============================
Main orchestration module for running and reporting E2E tests.
"""

import sys
import json
import time
from pathlib import Path
from typing import List

# Add SimOps root to path
TASK_ROOT = Path(__file__).parent
SIMOPS_ROOT = TASK_ROOT.parent.parent.parent
sys.path.insert(0, str(SIMOPS_ROOT))

from test_models import E2ETestCase, E2ETestResult, E2ETestReport
from test_e2e_integration import E2EPipelineRunner


class E2ETestRunner:
    """Orchestrates execution of E2E test suite."""
    
    def __init__(self, test_cases_file: Path):
        self.test_cases_file = test_cases_file
        self.test_cases: List[E2ETestCase] = []
        self.report = E2ETestReport()
    
    def load_test_cases(self):
        """Load test cases from JSON file."""
        with open(self.test_cases_file, 'r') as f:
            case_dicts = json.load(f)
        
        self.test_cases = [E2ETestCase(**case) for case in case_dicts]
        print(f"Loaded {len(self.test_cases)} test cases")
    
    def run_all_tests(self, use_mocks=True) -> E2ETestReport:
        """Run all test cases and generate report."""
        print("\n" + "="*70)
        print("E2E Test Suite Execution")
        print("="*70)
        
        pipeline_runner = E2EPipelineRunner(use_mocks=use_mocks)
        
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"\n[{i}/{len(self.test_cases)}] Running test: {test_case.name}")
            print(f"  Prompt: {test_case.prompt[:60]}...")
            
            start_time = time.time()
            result = pipeline_runner.run_pipeline(test_case)
            
            self.report.add_result(result)
            
            # Print result
            status = "✅ PASS" if result.success else "❌ FAIL"
            print(f"  {status} ({result.total_execution_time:.2f}s)")
            
            if not result.success:
                print(f"  Errors: {', '.join(result.errors)}")
            
            print(f"  Stages completed: {len(result.stages_passed)}/4")
        
        return self.report
    
    def generate_report(self, output_file: Path):
        """Generate JSON report."""
        report_dict = self.report.model_dump()
        
        with open(output_file, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        print(f"\nReport saved to: {output_file}")
    
    def print_summary(self):
        """Print summary to console."""
        print("\n" + "="*70)
        print("E2E Test Suite Summary")
        print("="*70)
        print(f"Total Tests:     {self.report.total_tests}")
        print(f"Passed:          {self.report.tests_passed} ✅")
        print(f"Failed:          {self.report.tests_failed} ❌")
        print(f"Success Rate:    {self.report.success_rate:.1f}%")
        print(f"Total Time:      {self.report.total_execution_time:.2f}s")
        
        threshold = 80.0
        meets_threshold = self.report.meets_threshold(threshold)
        
        print(f"\nThreshold ({threshold}%): {'✅ MET' if meets_threshold else '❌ NOT MET'}")
        
        if not meets_threshold:
            print(f"\n⚠️  Success rate {self.report.success_rate:.1f}% is below threshold {threshold}%")
        
        return meets_threshold


def run_test_suite(test_cases_file: Path, output_file: Path, use_mocks=True) -> bool:
    """
    Run the complete E2E test suite.
    
    Args:
        test_cases_file: Path to test_cases.json
        output_file: Path for output report (e2e_results.json)
        use_mocks: Use mocked components for fast testing
    
    Returns:
        bool: True if success rate meets threshold (≥80%)
    """
    runner = E2ETestRunner(test_cases_file)
    runner.load_test_cases()
    runner.run_all_tests(use_mocks=use_mocks)
    runner.generate_report(output_file)
    meets_threshold = runner.print_summary()
    
    return meets_threshold


if __name__ == "__main__":
    # Run test suite
    test_cases_file = TASK_ROOT / "test_data" / "test_cases.json"
    output_file = TASK_ROOT / "e2e_results.json"
    
    success = run_test_suite(test_cases_file, output_file, use_mocks=True)
    
    sys.exit(0 if success else 1)
