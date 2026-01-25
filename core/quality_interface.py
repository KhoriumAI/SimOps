"""
Quality Interface: Mesh Quality Analysis Abstraction
=====================================================

Abstract interface for mesh quality computation.
Enables testing without Gmsh dependency.

Usage:
    from core.quality_interface import IQualityAnalyzer, get_quality_analyzer
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass
class QualityReport:
    """Mesh quality statistics."""
    sicn_min: float = 0.0
    sicn_avg: float = 0.0
    sicn_max: float = 1.0
    sicn_10_percentile: float = 0.0
    gamma_min: float = 0.0
    gamma_avg: float = 0.0
    gamma_max: float = 1.0
    inverted_count: int = 0
    total_elements: int = 0
    
    @property
    def has_inverted(self) -> bool:
        return self.inverted_count > 0


class IQualityAnalyzer(ABC):
    """Abstract interface for mesh quality analysis."""
    
    @abstractmethod
    def analyze(self, points: np.ndarray, cells: np.ndarray) -> QualityReport:
        """
        Compute quality metrics from raw mesh data.
        
        Args:
            points: (N, 3) vertex coordinates
            cells: (M, 4) tetrahedral connectivity
            
        Returns:
            QualityReport with statistics
        """
        pass
    
    @abstractmethod
    def get_per_element_quality(self, metric: str = "sicn") -> Dict[int, float]:
        """
        Get per-element quality values.
        
        Args:
            metric: Quality metric ("sicn", "gamma", "skewness")
            
        Returns:
            Dict mapping element ID to quality value
        """
        pass


class MockQualityAnalyzer(IQualityAnalyzer):
    """Mock analyzer returning uniform quality."""
    
    def analyze(self, points: np.ndarray, cells: np.ndarray) -> QualityReport:
        return QualityReport(
            sicn_min=0.5,
            sicn_avg=0.75,
            sicn_max=0.95,
            sicn_10_percentile=0.55,
            total_elements=len(cells)
        )
    
    def get_per_element_quality(self, metric: str = "sicn") -> Dict[int, float]:
        return {}  # Would be populated by real analyzer


# Analyzer registry
_analyzer_instance: Optional[IQualityAnalyzer] = None


def get_quality_analyzer() -> IQualityAnalyzer:
    """Get current quality analyzer (mock by default)."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = MockQualityAnalyzer()
    return _analyzer_instance


def register_quality_analyzer(analyzer: IQualityAnalyzer):
    """Register a real analyzer implementation."""
    global _analyzer_instance
    _analyzer_instance = analyzer
