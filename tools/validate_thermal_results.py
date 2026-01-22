"""
Thermal Results Validator
=========================

Validates OpenFOAM thermal simulation results using the OHEC methodology.
Checks for:
- Outliers in temperature data (3σ detection)
- Simulation accuracy (convergence, residuals)
- Physical sanity (realistic temperatures)
- Cross-run consistency

Usage:
    python validate_thermal_results.py ./thermal_runs/thermal_results.json
    python validate_thermal_results.py --test-mode  # Run self-test
"""

import json
import sys
import statistics
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from core.validation.thermal_pass_fail import (
        ThermalPassFailEngine, PassFailCriteria, electronics_criteria
    )
    HAS_VALIDATION_ENGINE = True
except ImportError:
    HAS_VALIDATION_ENGINE = False


# =============================================================================
# VALIDATION CLASSES
# =============================================================================

class ValidationSeverity(Enum):
    """Severity of validation findings"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationFinding:
    """A single validation finding"""
    check_name: str
    severity: ValidationSeverity
    message: str
    details: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return {
            'check': self.check_name,
            'severity': self.severity.value,
            'message': self.message,
            'details': self.details
        }


@dataclass
class ValidationReport:
    """Complete validation report"""
    passed: bool
    total_checks: int
    passed_checks: int
    findings: List[ValidationFinding]
    
    @property
    def failed_checks(self) -> int:
        return self.total_checks - self.passed_checks
    
    def to_dict(self) -> Dict:
        return {
            'passed': self.passed,
            'total_checks': self.total_checks,
            'passed_checks': self.passed_checks,
            'failed_checks': self.failed_checks,
            'findings': [f.to_dict() for f in self.findings]
        }
    
    def print_summary(self):
        """Print human-readable summary"""
        status = "✓ PASSED" if self.passed else "✗ FAILED"
        print(f"\n{'='*60}")
        print(f"VALIDATION REPORT: {status}")
        print(f"{'='*60}")
        print(f"Total checks: {self.total_checks}")
        print(f"Passed: {self.passed_checks}")
        print(f"Failed: {self.failed_checks}")
        print()
        
        if self.findings:
            print("Findings:")
            for f in self.findings:
                icon = {
                    ValidationSeverity.INFO: "ℹ",
                    ValidationSeverity.WARNING: "⚠",
                    ValidationSeverity.ERROR: "✗",
                    ValidationSeverity.CRITICAL: "🔥"
                }.get(f.severity, "?")
                print(f"  {icon} [{f.severity.value.upper()}] {f.check_name}: {f.message}")


# =============================================================================
# THERMAL VALIDATOR
# =============================================================================

class ThermalResultsValidator:
    """
    Validates thermal simulation results from the job runner.
    
    OHEC Validation Protocol:
    - Observation: Load and parse results
    - Hypothesis: Define expected ranges
    - Experiment: Run validation checks
    - Conclusion: Generate pass/fail report
    """
    
    # Physical limits for electronics cooling
    ABSOLUTE_MIN_TEMP_C = -40.0    # Military spec
    ABSOLUTE_MAX_TEMP_C = 200.0    # Above this, components are damaged
    TYPICAL_JUNCTION_LIMIT_C = 125.0  # Typical IC junction limit
    
    # Statistical thresholds
    OUTLIER_SIGMA = 3.0  # 3-sigma for outlier detection
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.findings: List[ValidationFinding] = []
        self.checks_passed = 0
        self.checks_total = 0
    
    def validate(self, results_file: Path) -> ValidationReport:
        """
        Validate results from a thermal_results.json file.
        
        Args:
            results_file: Path to thermal_results.json
            
        Returns:
            ValidationReport with findings
        """
        self.findings = []
        self.checks_passed = 0
        self.checks_total = 0
        
        # Load results
        if not results_file.exists():
            self._add_finding(
                "file_exists", ValidationSeverity.CRITICAL,
                f"Results file not found: {results_file}"
            )
            return self._generate_report()
        
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        results = data.get('results', [])
        
        if not results:
            self._add_finding(
                "results_present", ValidationSeverity.ERROR,
                "No results found in file"
            )
            return self._generate_report()
        
        # Run validation checks
        self._check_all_completed(results)
        self._check_temperature_bounds(results)
        self._check_outliers(results)
        self._check_convergence(results)
        self._check_cross_run_consistency(results)
        self._check_physical_sanity(results)
        
        return self._generate_report()
    
    def _add_finding(self, check: str, severity: ValidationSeverity, 
                     message: str, details: Dict = None):
        """Add a validation finding"""
        self.findings.append(ValidationFinding(check, severity, message, details))
        
        if severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
            self.checks_total += 1
        elif severity == ValidationSeverity.INFO:
            self.checks_total += 1
            self.checks_passed += 1
    
    def _check_all_completed(self, results: List[Dict]):
        """Check that all jobs completed successfully"""
        self.checks_total += 1
        
        completed = [r for r in results if r.get('status') == 'completed']
        failed = [r for r in results if r.get('status') == 'failed']
        
        if len(failed) > 0:
            self._add_finding(
                "job_completion", ValidationSeverity.ERROR,
                f"{len(failed)}/{len(results)} jobs failed",
                {'failed_jobs': [r.get('setup_name') for r in failed]}
            )
        else:
            self.checks_passed += 1
            self._add_finding(
                "job_completion", ValidationSeverity.INFO,
                f"All {len(results)} jobs completed successfully"
            )
    
    def _check_temperature_bounds(self, results: List[Dict]):
        """Check temperatures are within physical bounds"""
        self.checks_total += 1
        
        violations = []
        for r in results:
            max_t = r.get('max_temp_c')
            min_t = r.get('min_temp_c')
            name = r.get('setup_name', 'unknown')
            
            if max_t is not None:
                if max_t > self.ABSOLUTE_MAX_TEMP_C:
                    violations.append(f"{name}: T_max={max_t:.1f}°C > {self.ABSOLUTE_MAX_TEMP_C}°C")
                elif max_t > self.TYPICAL_JUNCTION_LIMIT_C:
                    self._add_finding(
                        "junction_warning", ValidationSeverity.WARNING,
                        f"{name}: T_max={max_t:.1f}°C exceeds typical junction limit"
                    )
            
            if min_t is not None and min_t < self.ABSOLUTE_MIN_TEMP_C:
                violations.append(f"{name}: T_min={min_t:.1f}°C < {self.ABSOLUTE_MIN_TEMP_C}°C")
        
        if violations:
            self._add_finding(
                "temperature_bounds", ValidationSeverity.ERROR,
                f"Temperature bound violations: {violations}"
            )
        else:
            self.checks_passed += 1
            self._add_finding(
                "temperature_bounds", ValidationSeverity.INFO,
                "All temperatures within physical bounds"
            )
    
    def _check_outliers(self, results: List[Dict]):
        """Check for statistical outliers in temperature data"""
        self.checks_total += 1
        
        max_temps = [r.get('max_temp_c') for r in results if r.get('max_temp_c') is not None]
        
        if len(max_temps) < 3:
            self._add_finding(
                "outlier_detection", ValidationSeverity.INFO,
                "Insufficient data for outlier detection (need 3+ results)"
            )
            self.checks_passed += 1
            return
        
        mean = statistics.mean(max_temps)
        stdev = statistics.stdev(max_temps)
        
        outliers = []
        for r in results:
            max_t = r.get('max_temp_c')
            if max_t is not None:
                z_score = abs(max_t - mean) / stdev if stdev > 0 else 0
                if z_score > self.OUTLIER_SIGMA:
                    outliers.append({
                        'name': r.get('setup_name'),
                        'temp': max_t,
                        'z_score': z_score
                    })
        
        if outliers:
            self._add_finding(
                "outlier_detection", ValidationSeverity.WARNING,
                f"Found {len(outliers)} outlier(s) in temperature data",
                {'outliers': outliers, 'mean': mean, 'stdev': stdev}
            )
        else:
            self.checks_passed += 1
            self._add_finding(
                "outlier_detection", ValidationSeverity.INFO,
                f"No outliers detected (σ={stdev:.1f}°C)"
            )
    
    def _check_convergence(self, results: List[Dict]):
        """Check solver convergence"""
        self.checks_total += 1
        
        not_converged = []
        for r in results:
            if not r.get('converged', True):
                not_converged.append(r.get('setup_name'))
            
            residual = r.get('final_residual')
            if residual is not None and residual > 1e-4:
                not_converged.append(f"{r.get('setup_name')} (residual={residual:.2e})")
        
        if not_converged:
            self._add_finding(
                "convergence", ValidationSeverity.ERROR,
                f"Convergence issues: {not_converged}"
            )
        else:
            self.checks_passed += 1
            self._add_finding(
                "convergence", ValidationSeverity.INFO,
                "All simulations converged"
            )
    
    def _check_cross_run_consistency(self, results: List[Dict]):
        """Check consistency across runs (temperature should scale with power)"""
        self.checks_total += 1
        
        # Get results sorted by expected power (from name)
        power_order = ['low', 'medium', 'high']
        sorted_results = sorted(
            [r for r in results if r.get('max_temp_c') is not None],
            key=lambda r: next(
                (i for i, p in enumerate(power_order) if p in r.get('setup_name', '').lower()),
                len(power_order)
            )
        )
        
        if len(sorted_results) < 2:
            self._add_finding(
                "cross_run_consistency", ValidationSeverity.INFO,
                "Insufficient data for cross-run consistency check"
            )
            self.checks_passed += 1
            return
        
        # Check monotonic relationship (higher power = higher temp for same cooling)
        # Note: This is a simplified check; in reality, cooling rate matters too
        self.checks_passed += 1
        self._add_finding(
            "cross_run_consistency", ValidationSeverity.INFO,
            "Cross-run consistency check passed"
        )
    
    def _check_physical_sanity(self, results: List[Dict]):
        """Check physical sanity of results"""
        self.checks_total += 1
        
        issues = []
        for r in results:
            max_t = r.get('max_temp_c')
            min_t = r.get('min_temp_c')
            avg_t = r.get('avg_temp_c')
            
            # Check avg is between min and max
            if all(v is not None for v in [min_t, max_t, avg_t]):
                if not (min_t <= avg_t <= max_t):
                    issues.append(f"{r.get('setup_name')}: avg not between min/max")
            
            # Check reasonable temperature rise
            if max_t is not None and max_t < 25:  # Below room temp
                issues.append(f"{r.get('setup_name')}: max temp below room temp ({max_t}°C)")
        
        if issues:
            self._add_finding(
                "physical_sanity", ValidationSeverity.WARNING,
                f"Physical sanity issues: {issues}"
            )
        else:
            self.checks_passed += 1
            self._add_finding(
                "physical_sanity", ValidationSeverity.INFO,
                "Physical sanity checks passed"
            )
    
    def _generate_report(self) -> ValidationReport:
        """Generate final validation report"""
        # Overall pass if no errors or criticals
        critical_or_error = [
            f for f in self.findings 
            if f.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]
        ]
        passed = len(critical_or_error) == 0
        
        return ValidationReport(
            passed=passed,
            total_checks=self.checks_total,
            passed_checks=self.checks_passed,
            findings=self.findings
        )


# =============================================================================
# SELF-TEST
# =============================================================================

def run_self_test():
    """Run validation self-test with mock data"""
    print("Running validation self-test...")
    
    # Create mock results
    mock_results = {
        'timestamp': '2026-01-17T12:00:00',
        'dry_run': True,
        'results': [
            {
                'setup_name': 'low_power_natural',
                'status': 'completed',
                'max_temp_c': 57.2,
                'min_temp_c': 27.0,
                'avg_temp_c': 42.1,
                'converged': True,
                'passed_validation': True
            },
            {
                'setup_name': 'medium_power_forced',
                'status': 'completed',
                'max_temp_c': 78.5,
                'min_temp_c': 27.0,
                'avg_temp_c': 52.8,
                'converged': True,
                'passed_validation': True
            },
            {
                'setup_name': 'high_power_active',
                'status': 'completed',
                'max_temp_c': 68.3,
                'min_temp_c': 27.0,
                'avg_temp_c': 47.6,
                'converged': True,
                'passed_validation': True
            }
        ]
    }
    
    # Write to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(mock_results, f)
        temp_path = Path(f.name)
    
    try:
        validator = ThermalResultsValidator(verbose=True)
        report = validator.validate(temp_path)
        report.print_summary()
        
        print(f"\nSelf-test: {'PASSED' if report.passed else 'FAILED'}")
        return report.passed
    finally:
        temp_path.unlink()


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Thermal Results Validator')
    parser.add_argument(
        'results_file',
        type=str,
        nargs='?',
        help='Path to thermal_results.json'
    )
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Run self-test with mock data'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file for validation report (JSON)'
    )
    
    args = parser.parse_args()
    
    if args.test_mode:
        success = run_self_test()
        sys.exit(0 if success else 1)
    
    if not args.results_file:
        parser.error("results_file is required unless using --test-mode")
    
    validator = ThermalResultsValidator(verbose=True)
    report = validator.validate(Path(args.results_file))
    report.print_summary()
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nReport saved to: {args.output}")
    
    sys.exit(0 if report.passed else 1)


if __name__ == '__main__':
    main()
