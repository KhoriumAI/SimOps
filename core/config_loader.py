import json
import os
import logging
from typing import Optional
from core.schemas.config_schema import SimulationConfig

logger = logging.getLogger(__name__)

def load_simops_config(config_path: Optional[str] = None) -> SimulationConfig:
    """
    Load simulation configuration from a JSON sidecar file.
    If path is None or invalid, returns default configuration.
    """
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
            # Validate and parse via Pydantic
            config = SimulationConfig(**data)
            return config
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            logger.info("Falling back to default configuration.")
            return SimulationConfig()
            
    return SimulationConfig()