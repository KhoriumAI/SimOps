"""
AI Integration Module
=====================

Centralized AI recommendation system for mesh quality improvement.
Provides Claude AI integration with intelligent fallback strategies.
"""

import requests
import json
import re
from typing import Dict, List, Optional, Any
from .config import Config, get_default_config


class RecommendationType:
    """Recommendation type constants"""
    PARAMETER_ADJUSTMENT = "parameter_adjustment"
    ALGORITHM_CHANGE = "algorithm_change"
    ELEMENT_ORDER_CHANGE = "element_order_change"
    REFINEMENT_STRATEGY = "refinement_strategy"


class MeshRecommendation:
    """Represents a mesh improvement recommendation"""

    def __init__(self, rec_type: str, parameter: str, value: Any, reason: str):
        self.type = rec_type
        self.parameter = parameter
        self.value = value
        self.reason = reason

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'type': self.type,
            'parameter': self.parameter,
            'value': self.value,
            'reason': self.reason
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'MeshRecommendation':
        """Create from dictionary"""
        return cls(
            rec_type=data.get('type', ''),
            parameter=data.get('parameter', ''),
            value=data.get('value'),
            reason=data.get('reason', '')
        )

    def __repr__(self) -> str:
        return f"Recommendation({self.type}: {self.parameter}={self.value})"


class AIRecommendationEngine:
    """AI-powered mesh quality recommendation engine"""

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize AI recommendation engine

        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config or get_default_config()
        self.request_count = 0
        self.success_count = 0
        self.fallback_count = 0

    def get_recommendations(
        self,
        quality_metrics: Dict,
        current_params: Dict,
        iteration: int,
        history: Optional[List[Dict]] = None
    ) -> List[MeshRecommendation]:
        """
        Get mesh improvement recommendations

        Args:
            quality_metrics: Current mesh quality metrics
            current_params: Current mesh parameters
            iteration: Current iteration number
            history: Previous iteration history (optional)

        Returns:
            List of MeshRecommendation objects
        """
        # Check if AI is enabled and configured
        if not self.config.is_ai_enabled():
            return self._get_fallback_recommendations(quality_metrics, current_params)

        # Try Claude API
        try:
            recommendations = self._get_ai_recommendations(
                quality_metrics, current_params, iteration, history
            )
            if recommendations:
                self.success_count += 1
                return recommendations
        except Exception as e:
            print(f"AI recommendation failed: {e}")

        # Fallback to rule-based recommendations
        self.fallback_count += 1
        return self._get_fallback_recommendations(quality_metrics, current_params)

    def _get_ai_recommendations(
        self,
        quality_metrics: Dict,
        current_params: Dict,
        iteration: int,
        history: Optional[List[Dict]]
    ) -> Optional[List[MeshRecommendation]]:
        """Get recommendations from Claude AI"""
        self.request_count += 1

        # Build prompt
        prompt = self._build_ai_prompt(quality_metrics, current_params, iteration, history)

        # API request
        headers = {
            "x-api-key": self.config.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }

        data = {
            "model": self.config.ai_config.model,
            "max_tokens": self.config.ai_config.max_tokens,
            "messages": [{"role": "user", "content": prompt}]
        }

        response = requests.post(
            self.config.ai_config.api_url,
            headers=headers,
            json=data,
            timeout=self.config.ai_config.timeout
        )

        if response.status_code == 200:
            result = response.json()
            response_text = result.get('content', [{}])[0].get('text', '')

            # Parse JSON response
            recommendations = self._parse_ai_response(response_text)
            if recommendations:
                return recommendations

        return None

    def _build_ai_prompt(
        self,
        quality_metrics: Dict,
        current_params: Dict,
        iteration: int,
        history: Optional[List[Dict]]
    ) -> str:
        """Build prompt for Claude AI"""
        prompt_parts = [
            "You are an expert FEA mesh analyst. Analyze this mesh quality data and provide specific recommendations for IMPROVING quality metrics:",
            "",
            f"Iteration: {iteration}",
            "Mesh Statistics:",
            f"- Total Elements: {quality_metrics.get('total_elements', 0):,}",
            f"- Total Nodes: {quality_metrics.get('total_nodes', 0):,}",
        ]

        # Quality metrics
        if quality_metrics.get('skewness'):
            s = quality_metrics['skewness']
            prompt_parts.append(f"- Skewness: Max={s['max']:.4f}, Avg={s['avg']:.4f}")

        if quality_metrics.get('aspect_ratio'):
            a = quality_metrics['aspect_ratio']
            prompt_parts.append(f"- Aspect Ratio: Max={a['max']:.4f}, Avg={a['avg']:.4f}")

        if quality_metrics.get('min_angle'):
            m = quality_metrics['min_angle']
            prompt_parts.append(f"- Min Angle: {m['min']:.2f}°")

        # Current parameters
        prompt_parts.extend([
            "",
            "Current Parameters:",
            f"- CL_min: {current_params.get('cl_min', 'N/A')}",
            f"- CL_max: {current_params.get('cl_max', 'N/A')}",
        ])

        # Quality targets
        targets = self.config.quality_targets
        prompt_parts.extend([
            "",
            "Quality Targets:",
            f"- Skewness <= {targets.skewness_max}",
            f"- Aspect Ratio <= {targets.aspect_ratio_max}",
            f"- Min Angle >= {targets.min_angle_min}°",
        ])

        # History context (if available)
        if history and len(history) > 1:
            prompt_parts.extend([
                "",
                "Previous Iterations:",
            ])
            for i, hist in enumerate(history[-3:], start=max(1, iteration-2)):
                metrics = hist.get('metrics', {})
                if metrics.get('skewness'):
                    skew = metrics['skewness']['max']
                    prompt_parts.append(f"  Iteration {i}: Skewness={skew:.4f}")

        # Instructions
        prompt_parts.extend([
            "",
            "IMPORTANT: Focus on strategies that will actually IMPROVE these metrics, not just make the mesh finer. Consider:",
            "- Different meshing algorithms (Delaunay=1, HXT=4, MMG3D=7, Frontal-Delaunay=10)",
            "- Adaptive refinement strategies",
            "- Quality-focused parameter adjustments",
            "- Element order changes (linear=1, quadratic=2)",
            "",
            "Respond in JSON format:",
            '{',
            '    "recommendations": [',
            '        {',
            '            "type": "algorithm_change",',
            '            "parameter": "Mesh.Algorithm3D",',
            '            "value": 4,',
            '            "reason": "Use HXT algorithm for better skewness"',
            '        }',
            '    ]',
            '}'
        ])

        return "\n".join(prompt_parts)

    def _parse_ai_response(self, response_text: str) -> Optional[List[MeshRecommendation]]:
        """Parse AI response and extract recommendations"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                return None

            json_str = json_match.group()
            data = json.loads(json_str)

            recommendations = data.get('recommendations', [])
            if not recommendations:
                return None

            # Convert to MeshRecommendation objects
            return [MeshRecommendation.from_dict(rec) for rec in recommendations]

        except Exception as e:
            print(f"Failed to parse AI response: {e}")
            return None

    def _get_fallback_recommendations(
        self,
        quality_metrics: Dict,
        current_params: Dict
    ) -> List[MeshRecommendation]:
        """
        Generate intelligent rule-based recommendations

        This is used when AI is not available or fails.
        Implements sophisticated quality-driven strategies.
        """
        recommendations = []
        targets = self.config.quality_targets

        # Extract metrics
        max_skewness = 0.0
        avg_skewness = 0.0
        if quality_metrics.get('skewness'):
            max_skewness = quality_metrics['skewness']['max']
            avg_skewness = quality_metrics['skewness']['avg']

        max_aspect_ratio = 0.0
        avg_aspect_ratio = 0.0
        if quality_metrics.get('aspect_ratio'):
            max_aspect_ratio = quality_metrics['aspect_ratio']['max']
            avg_aspect_ratio = quality_metrics['aspect_ratio']['avg']

        min_angle = 90.0
        if quality_metrics.get('min_angle'):
            min_angle = quality_metrics['min_angle']['min']

        # Get current parameters
        cl_min = current_params.get('cl_min', 1.0)
        cl_max = current_params.get('cl_max', 10.0)

        # Strategy 1: Address severe quality issues with algorithm changes
        if max_skewness > 0.85:
            recommendations.append(MeshRecommendation(
                RecommendationType.ALGORITHM_CHANGE,
                "Mesh.Algorithm3D",
                4,  # HXT
                f"Severe skewness ({max_skewness:.4f}) - use HXT algorithm"
            ))

        if max_aspect_ratio > 15.0:
            recommendations.append(MeshRecommendation(
                RecommendationType.ALGORITHM_CHANGE,
                "Mesh.Algorithm3D",
                7,  # MMG3D
                f"Severe aspect ratio ({max_aspect_ratio:.4f}) - use MMG3D algorithm"
            ))

        if min_angle < 5.0:
            recommendations.append(MeshRecommendation(
                RecommendationType.ALGORITHM_CHANGE,
                "Mesh.Algorithm3D",
                10,  # Frontal-Delaunay 3D
                f"Very small angles ({min_angle:.2f}°) - use Frontal-Delaunay"
            ))

        # Strategy 2: Adaptive parameter adjustments
        skewness_poor = max_skewness > targets.skewness_max
        aspect_ratio_poor = max_aspect_ratio > targets.aspect_ratio_max

        if skewness_poor and aspect_ratio_poor:
            # Both poor - analyze which is worse
            skew_severity = max_skewness / targets.skewness_max
            aspect_severity = max_aspect_ratio / targets.aspect_ratio_max

            if aspect_severity > skew_severity * 1.5:
                # Aspect ratio much worse - try coarsening
                recommendations.append(MeshRecommendation(
                    RecommendationType.PARAMETER_ADJUSTMENT,
                    "cl_max",
                    cl_max * 1.3,
                    f"Coarsen mesh to improve aspect ratio (severity: {aspect_severity:.2f})"
                ))
                recommendations.append(MeshRecommendation(
                    RecommendationType.PARAMETER_ADJUSTMENT,
                    "cl_min",
                    cl_min * 1.1,
                    "Increase minimum element size"
                ))
            else:
                # Balanced refinement
                recommendations.append(MeshRecommendation(
                    RecommendationType.PARAMETER_ADJUSTMENT,
                    "cl_min",
                    cl_min * 0.85,
                    f"Balanced refinement for both metrics"
                ))

        elif skewness_poor:
            # Only skewness poor - local refinement
            if avg_skewness > targets.skewness_max * 0.8:
                # Widespread issue - uniform refinement
                recommendations.append(MeshRecommendation(
                    RecommendationType.PARAMETER_ADJUSTMENT,
                    "cl_min",
                    cl_min * 0.7,
                    f"Widespread skewness issues (avg={avg_skewness:.4f})"
                ))
            else:
                # Localized issue - moderate refinement
                recommendations.append(MeshRecommendation(
                    RecommendationType.PARAMETER_ADJUSTMENT,
                    "cl_min",
                    cl_min * 0.8,
                    f"Localized skewness issues (max={max_skewness:.4f})"
                ))

        elif aspect_ratio_poor:
            # Only aspect ratio poor - consider coarsening
            if avg_aspect_ratio > targets.aspect_ratio_max * 0.7:
                # Widespread issue - try coarsening
                recommendations.append(MeshRecommendation(
                    RecommendationType.PARAMETER_ADJUSTMENT,
                    "cl_max",
                    cl_max * 1.2,
                    f"Widespread aspect ratio issues - try coarser mesh"
                ))
            else:
                # Localized issue - slight refinement
                recommendations.append(MeshRecommendation(
                    RecommendationType.PARAMETER_ADJUSTMENT,
                    "cl_min",
                    cl_min * 0.9,
                    f"Localized aspect ratio issues"
                ))

        # Strategy 3: Element order adjustments for extreme cases
        if max_skewness > 0.95 or max_aspect_ratio > 30.0:
            recommendations.append(MeshRecommendation(
                RecommendationType.ELEMENT_ORDER_CHANGE,
                "Mesh.ElementOrder",
                1,  # Linear elements
                "Extreme quality issues - try linear elements"
            ))

        # Strategy 4: If quality is good, suggest conservative refinement
        if not recommendations:
            # Quality is acceptable, suggest conservative improvement
            recommendations.append(MeshRecommendation(
                RecommendationType.PARAMETER_ADJUSTMENT,
                "cl_min",
                cl_min * 0.9,
                "Conservative refinement to further improve quality"
            ))

        return recommendations

    def get_statistics(self) -> Dict[str, int]:
        """Get recommendation engine statistics"""
        return {
            'total_requests': self.request_count,
            'ai_success': self.success_count,
            'fallback_used': self.fallback_count,
            'ai_success_rate': (self.success_count / self.request_count * 100)
            if self.request_count > 0 else 0
        }


# Convenience functions for backward compatibility

def get_quality_driven_recommendations(
    quality_metrics: Dict,
    current_params: Dict,
    config: Optional[Config] = None
) -> List[Dict]:
    """
    Get quality-driven recommendations (legacy interface)

    Returns list of dictionaries for backward compatibility
    """
    engine = AIRecommendationEngine(config)
    recommendations = engine.get_recommendations(
        quality_metrics=quality_metrics,
        current_params=current_params,
        iteration=1
    )
    return [rec.to_dict() for rec in recommendations]
