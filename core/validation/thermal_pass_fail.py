"""
Thermal Pass/Fail Criteria Engine
=================================

Physics-based pass/fail determination for thermal simulations.
Goes beyond log parsing to validate actual simulation results.

Criteria Types:
- Temperature bounds (max/min limits)
- Gradient limits (thermal stress indicator)
- Convergence requirements
- Mesh quality validation
- Physical sanity checks
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import numpy as np


class FailureCategory(Enum):
    """Categories of simulation failures"""
    TEMPERATURE_EXCEEDED = "temperature_exceeded"
    TEMPERATURE_TOO_LOW = "temperature_too_low"
    GRADIENT_TOO_HIGH = "gradient_too_high"
    NOT_CONVERGED = "not_converged"
    MESH_QUALITY = "mesh_quality"
    SOLVER_CRASH = "solver_crash"
    PHYSICAL_VIOLATION = "physical_violation"
    UNKNOWN = "unknown"


@dataclass
class PassFailCriteria:
    """Criteria for thermal simulation pass/fail determination"""
    
    # Temperature bounds (in Celsius)
    max_temp_limit_c: float = 150.0
    min_temp_limit_c: float = -40.0
    
    # Temperature uniformity
    max_temp_variance_c2: Optional[float] = None  # Max variance allowed
    
    # Gradient limits
    max_gradient_c_per_mm: Optional[float] = None  # Max spatial gradient
    
    # Convergence requirements
    require_convergence: bool = True
    max_residual: float = 1e-4
    min_iterations: int = 3
    
    # Mesh quality thresholds
    min_jacobian: float = 0.1
    max_aspect_ratio: float = 20.0
    min_elements: int = 100
    
    # Physical sanity checks
    check_energy_conservation: bool = False
    max_energy_imbalance_pct: float = 5.0
    
    # Material-based limits (auto-checked if set)
    material_max_service_temp_c: Optional[float] = None
    material_melting_point_c: Optional[float] = None


@dataclass
class PassFailResult:
    """Result of pass/fail evaluation"""
    passed: bool
    category: FailureCategory
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def is_critical(self) -> bool:
        """Check if failure is critical (solver crash, physical violation)"""
        return self.category in [
            FailureCategory.SOLVER_CRASH,
            FailureCategory.PHYSICAL_VIOLATION
        ]


class ThermalPassFailEngine:
    """
    Evaluate thermal simulation results against pass/fail criteria.
    
    Example:
        engine = ThermalPassFailEngine()
        criteria = PassFailCriteria(max_temp_limit_c=100.0)
        result = engine.evaluate(simulation_result, criteria)
        
        if not result.passed:
            print(f"FAIL: {result.reason} ({result.category.value})")
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    def _log(self, msg: str):
        if self.verbose:
            print(f"[PassFail] {msg}")
    
    def evaluate(
        self,
        simulation_result: Dict,
        criteria: PassFailCriteria
    ) -> PassFailResult:
        """
        Evaluate simulation result against criteria.
        
        Args:
            simulation_result: Dict containing:
                - success: bool
                - min_temp: float (Celsius)
                - max_temp: float (Celsius)
                - avg_temp: float (Celsius, optional)
                - temp_variance: float (optional)
                - max_gradient: float (optional)
                - converged: bool (optional)
                - final_residual: float (optional)
                - num_elements: int (optional)
                - mesh_quality: dict (optional)
            criteria: PassFailCriteria to check against
        
        Returns:
            PassFailResult with pass/fail status and details
        """
        warnings = []
        details = {}
        
        # Check 1: Solver success
        if not simulation_result.get('success', False):
            error = simulation_result.get('error', 'Unknown solver error')
            return PassFailResult(
                passed=False,
                category=FailureCategory.SOLVER_CRASH,
                reason=f"Solver failed: {error}",
                details={'error': error}
            )
        
        # Check 2: Max temperature limit
        max_temp = simulation_result.get('max_temp')
        if max_temp is not None:
            details['max_temp_c'] = max_temp
            
            if max_temp > criteria.max_temp_limit_c:
                return PassFailResult(
                    passed=False,
                    category=FailureCategory.TEMPERATURE_EXCEEDED,
                    reason=f"Max temperature {max_temp:.1f}°C exceeds limit {criteria.max_temp_limit_c}°C",
                    details=details
                )
            
            # Check against material limits
            if criteria.material_max_service_temp_c and max_temp > criteria.material_max_service_temp_c:
                warnings.append(
                    f"Temperature {max_temp:.1f}°C exceeds material service limit "
                    f"{criteria.material_max_service_temp_c}°C"
                )
            
            if criteria.material_melting_point_c and max_temp > criteria.material_melting_point_c * 0.9:
                return PassFailResult(
                    passed=False,
                    category=FailureCategory.PHYSICAL_VIOLATION,
                    reason=f"Temperature {max_temp:.1f}°C exceeds 90% of melting point {criteria.material_melting_point_c}°C",
                    details=details
                )
        
        # Check 3: Min temperature limit
        min_temp = simulation_result.get('min_temp')
        if min_temp is not None:
            details['min_temp_c'] = min_temp
            
            if min_temp < criteria.min_temp_limit_c:
                return PassFailResult(
                    passed=False,
                    category=FailureCategory.TEMPERATURE_TOO_LOW,
                    reason=f"Min temperature {min_temp:.1f}°C below limit {criteria.min_temp_limit_c}°C",
                    details=details
                )
        
        # Check 4: Temperature variance
        variance = simulation_result.get('temp_variance')
        if variance is not None and criteria.max_temp_variance_c2 is not None:
            details['temp_variance_c2'] = variance
            
            if variance > criteria.max_temp_variance_c2:
                warnings.append(
                    f"Temperature variance {variance:.1f}°C² exceeds limit {criteria.max_temp_variance_c2}°C²"
                )
        
        # Check 5: Gradient limit
        gradient = simulation_result.get('max_gradient')
        if gradient is not None and criteria.max_gradient_c_per_mm is not None:
            details['max_gradient_c_mm'] = gradient
            
            if gradient > criteria.max_gradient_c_per_mm:
                return PassFailResult(
                    passed=False,
                    category=FailureCategory.GRADIENT_TOO_HIGH,
                    reason=f"Max gradient {gradient:.2f}°C/mm exceeds limit {criteria.max_gradient_c_per_mm}°C/mm",
                    details=details,
                    warnings=warnings
                )
        
        # Check 6: Convergence
        if criteria.require_convergence:
            converged = simulation_result.get('converged', True)  # Assume converged if not specified
            residual = simulation_result.get('final_residual')
            
            if residual is not None:
                details['final_residual'] = residual
                
                if residual > criteria.max_residual:
                    return PassFailResult(
                        passed=False,
                        category=FailureCategory.NOT_CONVERGED,
                        reason=f"Final residual {residual:.2e} exceeds limit {criteria.max_residual:.2e}",
                        details=details,
                        warnings=warnings
                    )
            
            if not converged:
                return PassFailResult(
                    passed=False,
                    category=FailureCategory.NOT_CONVERGED,
                    reason="Simulation did not converge",
                    details=details,
                    warnings=warnings
                )
        
        # Check 7: Mesh quality
        num_elements = simulation_result.get('num_elements', 0)
        if num_elements > 0:
            details['num_elements'] = num_elements
            
            if num_elements < criteria.min_elements:
                warnings.append(
                    f"Mesh has only {num_elements} elements (minimum: {criteria.min_elements})"
                )
        
        mesh_quality = simulation_result.get('mesh_quality', {})
        if mesh_quality:
            min_jac = mesh_quality.get('min_jacobian')
            max_ar = mesh_quality.get('max_aspect_ratio')
            
            if min_jac is not None:
                details['min_jacobian'] = min_jac
                if min_jac < criteria.min_jacobian:
                    return PassFailResult(
                        passed=False,
                        category=FailureCategory.MESH_QUALITY,
                        reason=f"Min Jacobian {min_jac:.3f} below limit {criteria.min_jacobian}",
                        details=details,
                        warnings=warnings
                    )
            
            if max_ar is not None:
                details['max_aspect_ratio'] = max_ar
                if max_ar > criteria.max_aspect_ratio:
                    return PassFailResult(
                        passed=False,
                        category=FailureCategory.MESH_QUALITY,
                        reason=f"Max aspect ratio {max_ar:.1f} exceeds limit {criteria.max_aspect_ratio}",
                        details=details,
                        warnings=warnings
                    )
        
        # Check 8: Physical sanity
        if self._check_physical_sanity(simulation_result, criteria, details):
            warnings_list = details.pop('sanity_warnings', [])
            warnings.extend(warnings_list)
        
        # All checks passed
        self._log(f"PASSED - T_max: {max_temp:.1f}°C, T_min: {min_temp:.1f}°C")
        
        return PassFailResult(
            passed=True,
            category=FailureCategory.UNKNOWN,  # Not applicable for pass
            reason="All criteria satisfied",
            details=details,
            warnings=warnings
        )
    
    def _check_physical_sanity(
        self,
        result: Dict,
        criteria: PassFailCriteria,
        details: Dict
    ) -> bool:
        """Check for physical sanity violations"""
        
        sanity_warnings = []
        
        # Check for unrealistic temperatures (e.g., T > 5000K indicates explosion)
        max_temp = result.get('max_temp')
        if max_temp is not None:
            if max_temp > 5000:
                details['sanity_violation'] = f"Unrealistic temperature: {max_temp}°C"
                return False
        
        # Check for negative absolute temperature (impossible)
        min_temp = result.get('min_temp')
        if min_temp is not None:
            if min_temp < -273.15:
                details['sanity_violation'] = f"Temperature below absolute zero: {min_temp}°C"
                return False
        
        # Temperature should be somewhat close to boundary conditions
        # (unless geometry prevents it)
        
        details['sanity_warnings'] = sanity_warnings
        return True
    
    def evaluate_batch(
        self,
        results: List[Dict],
        criteria: PassFailCriteria
    ) -> List[PassFailResult]:
        """Evaluate multiple simulation results"""
        return [self.evaluate(r, criteria) for r in results]


# =============================================================================
# QUICK CRITERIA BUILDERS
# =============================================================================

def electronics_criteria(max_junction_temp_c: float = 100.0) -> PassFailCriteria:
    """Standard criteria for electronics thermal analysis"""
    return PassFailCriteria(
        max_temp_limit_c=max_junction_temp_c,
        min_temp_limit_c=-40.0,  # Military spec
        max_gradient_c_per_mm=5.0,  # Avoid thermal stress
        require_convergence=True,
        max_residual=1e-5,
        min_elements=5000,
    )


def automotive_criteria(max_component_temp_c: float = 150.0) -> PassFailCriteria:
    """Standard criteria for automotive thermal analysis"""
    return PassFailCriteria(
        max_temp_limit_c=max_component_temp_c,
        min_temp_limit_c=-40.0,
        require_convergence=True,
        max_residual=1e-4,
    )


def aerospace_criteria(
    material_service_temp_c: float,
    material_melting_c: float
) -> PassFailCriteria:
    """Strict criteria for aerospace thermal analysis"""
    return PassFailCriteria(
        max_temp_limit_c=material_service_temp_c,
        min_temp_limit_c=-269.0,  # Near LHe
        max_gradient_c_per_mm=10.0,
        require_convergence=True,
        max_residual=1e-6,
        min_jacobian=0.2,
        max_aspect_ratio=10.0,
        min_elements=10000,
        material_max_service_temp_c=material_service_temp_c,
        material_melting_point_c=material_melting_c,
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'FailureCategory',
    'PassFailCriteria',
    'PassFailResult',
    'ThermalPassFailEngine',
    'electronics_criteria',
    'automotive_criteria',
    'aerospace_criteria',
]
