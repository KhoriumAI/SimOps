import logging
import sys
from typing import Any, Optional

class SimLogger:
    """
    Standardized logger for SimOps simulations.
    Produces machine-readable logs with structured prefixes while maintaining human readability.
    
    Format:
    [STAGE] <Stage Name>
    [METADATA] key=value
    [METRIC] name=value unit=unit
    [ERROR] <code>: <message>
    [INFO] <message>
    """
    
    def __init__(self, name: str = "SimOps"):
        self.logger = logging.getLogger(name)
        # Ensure we don't add duplicate handlers if re-initialized
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(message)s') # Raw message, we add prefixes manually
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            
    def set_stream(self, stream: Any):
        """Update the output stream for all StreamHandlers."""
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setStream(stream)
            
    def _format_kv(self, key: str, value: Any) -> str:
        """Formats key-value pair, handling quoting if needed."""
        v_str = str(value)
        if " " in v_str:
            return f'{key}="{v_str}"'
        return f'{key}={v_str}'

    def log_stage(self, stage_name: str):
        """Log a major simulation stage transition."""
        self.logger.info(f"[STAGE] {stage_name}")

    def log_metadata(self, key: str, value: Any):
        """Log configuration or environment metadata."""
        self.logger.info(f"[METADATA] {self._format_kv(key, value)}")

    def log_metric(self, name: str, value: float, unit: str = ""):
        """Log a numerical metric (e.g. residual, temperature, progress)."""
        unit_str = f" unit={unit}" if unit else ""
        self.logger.info(f"[METRIC] {name}={value}{unit_str}")

    def log_error(self, code: str, message: str):
        """Log a structured error."""
        self.logger.error(f"[ERROR] {code}: {message}")

    def info(self, message: str):
        """Standard info log."""
        self.logger.info(f"[INFO] {message}")

    def warning(self, message: str):
        """Standard warning log."""
        self.logger.warning(f"[WARNING] {message}")
        
    def error(self, message: str):
        """Standard error log (unstructured)."""
        self.logger.error(f"[ERROR] {message}")

# Global instance for easy access if needed
sim_logger = SimLogger()
