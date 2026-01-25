"""
SimOps Configuration Loader
===========================

Handles loading and validation of simulation configurations.
supports JSON sidecars and default fallbacks.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from core.schemas.config_schema import SimulationConfig
except ImportError:
    # Fallback for direct execution testing
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from core.schemas.config_schema import SimulationConfig

logger = logging.getLogger(__name__)

def load_simops_config(config_path: str) -> SimulationConfig:
    """
    Load configuration from a JSON file.
    If path is None or file doesn't exist, returns defaults.
    """
    if not config_path:
        logger.info("No config path provided, using defaults (Golden Template)")
        return SimulationConfig()
        
    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Config file not found at {path}, using defaults")
        return SimulationConfig()
        
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            
        config = SimulationConfig(**data)
        logger.info(f"Loaded configuration for job: {config.job_name or 'Unnamed'}")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load config from {path}: {e}")
        logger.warning("Falling back to defaults due to config error")
        return SimulationConfig()

if __name__ == "__main__":
    # Test
    print(load_simops_config("nonexistent.json"))
