"""
Configuration Management Module
===============================

Centralized configuration for mesh generation.
Handles API keys, quality targets, and mesh parameters.
"""

import os
import json
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class QualityTargets:
    """Quality target thresholds for mesh generation"""
    skewness_max: float = 0.7
    aspect_ratio_max: float = 5.0
    min_angle_min: float = 10.0  # degrees
    jacobian_min: float = 0.0
    min_improvement_threshold: float = 0.05  # 5% improvement required


@dataclass
class MeshParameters:
    """Default mesh generation parameters"""
    algorithm_2d: int = 6  # Frontal-Delaunay
    algorithm_3d: int = 1  # Delaunay
    element_order: int = 2  # Quadratic elements
    high_order_optimize: int = 2
    refinement_levels: int = 3
    max_iterations: int = 5


@dataclass
class AIConfig:
    """AI integration configuration - DISABLED (not used in exhaustive strategy)"""
    enabled: bool = False  # Disabled - AI not providing value, adds complexity
    api_url: str = "https://api.anthropic.com/v1/messages"
    model: str = "claude-3-5-sonnet-20241022"
    max_tokens: int = 500
    timeout: int = 30
    use_fallback: bool = True


class Config:
    """Central configuration manager"""

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration

        Args:
            config_file: Path to JSON config file (optional)
        """
        self.config_file = config_file
        self.quality_targets = QualityTargets()
        self.mesh_params = MeshParameters()
        self.ai_config = AIConfig()
        self._api_key: Optional[str] = None

        # Load configuration
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
        else:
            self.load_from_env()

    @property
    def api_key(self) -> Optional[str]:
        """
        Get Claude API key with secure fallback chain:
        1. Environment variable CLAUDE_API_KEY
        2. .env file in current directory
        3. .env file in home directory
        4. Return None (will disable AI features)
        """
        if self._api_key:
            return self._api_key

        # Try environment variable
        api_key = os.getenv('CLAUDE_API_KEY')
        if api_key:
            self._api_key = api_key
            return api_key

        # Try .env file in current directory
        env_file = Path('.env')
        if env_file.exists():
            api_key = self._load_api_key_from_env_file(env_file)
            if api_key:
                self._api_key = api_key
                return api_key

        # Try .env file in home directory
        home_env_file = Path.home() / '.env'
        if home_env_file.exists():
            api_key = self._load_api_key_from_env_file(home_env_file)
            if api_key:
                self._api_key = api_key
                return api_key

        return None

    @api_key.setter
    def api_key(self, value: str):
        """Set API key directly"""
        self._api_key = value

    def _load_api_key_from_env_file(self, env_file: Path) -> Optional[str]:
        """Load API key from .env file"""
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('CLAUDE_API_KEY='):
                        return line.split('=', 1)[1].strip().strip('"\'')
        except Exception:
            pass
        return None

    def load_from_env(self):
        """Load configuration from environment variables"""
        # Quality targets
        self.quality_targets.skewness_max = float(
            os.getenv('MESH_SKEWNESS_TARGET', self.quality_targets.skewness_max)
        )
        self.quality_targets.aspect_ratio_max = float(
            os.getenv('MESH_ASPECT_RATIO_TARGET', self.quality_targets.aspect_ratio_max)
        )
        self.quality_targets.min_angle_min = float(
            os.getenv('MESH_MIN_ANGLE_TARGET', self.quality_targets.min_angle_min)
        )

        # Mesh parameters
        self.mesh_params.max_iterations = int(
            os.getenv('MESH_MAX_ITERATIONS', self.mesh_params.max_iterations)
        )
        self.mesh_params.element_order = int(
            os.getenv('MESH_ELEMENT_ORDER', self.mesh_params.element_order)
        )

        # AI configuration
        self.ai_config.enabled = os.getenv('AI_ENABLED', 'true').lower() == 'true'
        self.ai_config.model = os.getenv('AI_MODEL', self.ai_config.model)

    def load_from_file(self, config_file: str):
        """
        Load configuration from JSON file

        Args:
            config_file: Path to JSON configuration file
        """
        try:
            with open(config_file, 'r') as f:
                data = json.load(f)

            # Load quality targets
            if 'quality_targets' in data:
                for key, value in data['quality_targets'].items():
                    if hasattr(self.quality_targets, key):
                        setattr(self.quality_targets, key, value)

            # Load mesh parameters
            if 'mesh_parameters' in data:
                for key, value in data['mesh_parameters'].items():
                    if hasattr(self.mesh_params, key):
                        setattr(self.mesh_params, key, value)

            # Load AI configuration
            if 'ai_config' in data:
                for key, value in data['ai_config'].items():
                    if hasattr(self.ai_config, key):
                        setattr(self.ai_config, key, value)

            # API key from config file (not recommended, but supported)
            if 'api_key' in data:
                self._api_key = data['api_key']

        except Exception as e:
            print(f"Warning: Could not load config file {config_file}: {e}")

    def save_to_file(self, config_file: str, include_api_key: bool = False):
        """
        Save configuration to JSON file

        Args:
            config_file: Path to save configuration
            include_api_key: Include API key in file (NOT RECOMMENDED)
        """
        data = {
            'quality_targets': asdict(self.quality_targets),
            'mesh_parameters': asdict(self.mesh_params),
            'ai_config': asdict(self.ai_config)
        }

        if include_api_key and self._api_key:
            data['api_key'] = self._api_key

        try:
            with open(config_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Configuration saved to {config_file}")
        except Exception as e:
            print(f"Error saving configuration: {e}")

    def get_quality_targets_dict(self) -> Dict[str, float]:
        """Get quality targets as dictionary"""
        return asdict(self.quality_targets)

    def get_mesh_params_dict(self) -> Dict[str, Any]:
        """Get mesh parameters as dictionary"""
        return asdict(self.mesh_params)

    def get_ai_config_dict(self) -> Dict[str, Any]:
        """Get AI configuration as dictionary"""
        return asdict(self.ai_config)

    def is_ai_enabled(self) -> bool:
        """Check if AI features are enabled and configured"""
        return self.ai_config.enabled and self.api_key is not None

    def __repr__(self) -> str:
        """String representation of configuration"""
        api_key_status = "configured" if self.api_key else "not configured"
        return (
            f"Config(\n"
            f"  API Key: {api_key_status}\n"
            f"  Quality Targets: skewness<={self.quality_targets.skewness_max}, "
            f"aspect_ratio<={self.quality_targets.aspect_ratio_max}\n"
            f"  Max Iterations: {self.mesh_params.max_iterations}\n"
            f"  AI Enabled: {self.ai_config.enabled}\n"
            f")"
        )


# Global default configuration instance
_default_config: Optional[Config] = None


def get_default_config() -> Config:
    """Get or create default configuration instance"""
    global _default_config
    if _default_config is None:
        _default_config = Config()
    return _default_config


def set_default_config(config: Config):
    """Set the default configuration instance"""
    global _default_config
    _default_config = config


def create_example_env_file(output_path: str = ".env.example"):
    """
    Create an example .env file with all configuration options

    Args:
        output_path: Path where to save the example file
    """
    example_content = """# Claude API Configuration
# Get your API key from: https://console.anthropic.com/
CLAUDE_API_KEY=your_api_key_here

# Quality Target Configuration
MESH_SKEWNESS_TARGET=0.7
MESH_ASPECT_RATIO_TARGET=5.0
MESH_MIN_ANGLE_TARGET=10.0

# Mesh Generation Parameters
MESH_MAX_ITERATIONS=5
MESH_ELEMENT_ORDER=2

# AI Configuration
AI_ENABLED=true
AI_MODEL=claude-3-5-sonnet-20241022
"""

    try:
        with open(output_path, 'w') as f:
            f.write(example_content)
        print(f"Example .env file created at: {output_path}")
    except Exception as e:
        print(f"Error creating example .env file: {e}")


def create_example_config_file(output_path: str = "mesh_config.json"):
    """
    Create an example configuration JSON file

    Args:
        output_path: Path where to save the example file
    """
    config = Config()
    config.save_to_file(output_path, include_api_key=False)
    print(f"Example configuration file created at: {output_path}")
