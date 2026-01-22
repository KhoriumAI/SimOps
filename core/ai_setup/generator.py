"""
AI-Powered Simulation Setup Generator
=====================================
Converts natural language prompts into validated SimConfig objects.
Merged from Forge TASK_01_AI_SCHEMA.
"""

import json
import logging
import os
from typing import Optional
from core.schemas import SimConfig

logger = logging.getLogger(__name__)

class AISetupGenerator:
    """
    Generates simulation configurations from user prompts using LLM.
    Uses Anthropic Claude API for production, with fallback logic.
    """
    
    SYSTEM_PROMPT = """
You are a Thermal Simulation Expert. Convert user requests into valid JSON matching the SimConfig schema.

Rules:
- Default materials: Use engineering standards (Al6061, Copper, etc.)
- Units: Temperature (Celsius), Power (Watts), Time (Seconds), Size (mm)
- Be conservative with boundary conditions
- Output ONLY valid JSON, no explanations
"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the AI generator.
        
        Args:
            api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.
        """
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self._client = None
        
    def _get_client(self):
        """Lazy-load the Anthropic client."""
        if self._client is None and self.api_key:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                logger.warning("anthropic package not installed. Install with: pip install anthropic")
        return self._client

    def generate_config(self, prompt: str, cad_file: str, use_mock: bool = False) -> SimConfig:
        """
        Main entry point for generating a config from a prompt.
        
        Args:
            prompt: Natural language description of simulation
            cad_file: Path to the CAD file
            use_mock: If True, use mock responses instead of real API
            
        Returns:
            Validated SimConfig object
            
        Raises:
            ValueError: If LLM response is invalid or API fails
        """
        if use_mock or not self.api_key:
            logger.info("Using mock LLM response")
            llm_json_response = self._mock_llm_call(prompt, cad_file)
        else:
            llm_json_response = self._real_llm_call(prompt)
        
        # Parse and validate
        try:
            config_dict = json.loads(llm_json_response)
            config_dict['cad_file'] = cad_file  # Ensure CAD path is set
            config = SimConfig(**config_dict)
            return config
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            raise ValueError(f"AI failed to generate valid configuration: {e}")

    def _real_llm_call(self, prompt: str) -> str:
        """Call the real Anthropic API."""
        client = self._get_client()
        if not client:
            raise ValueError("Anthropic client not available. Check API key and package installation.")
        
        try:
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise ValueError(f"LLM API call failed: {e}")

    def _mock_llm_call(self, prompt: str, cad_file: str) -> str:
        """
        Simulates an LLM response based on keywords in the prompt.
        Used for testing and fallback when API is unavailable.
        """
        p_lower = prompt.lower()
        
        # Default mock config
        mock_data = {
            "job_name": f"Simulation_{cad_file.split('/')[-1].replace('.step', '')}",
            "materials": [{"name": "Aluminum_6061_T6"}],
            "boundary_conditions": [],
            "solver_settings": {"transient": True, "duration": 30.0, "time_step": 1.0},
            "mesh_settings": {"max_size_mm": 5.0, "min_size_mm": 1.0}
        }
        
        # Parse heat source
        if "heatsink" in p_lower or "heat" in p_lower:
            value = 50.0  # default
            if "w" in p_lower:
                # Extract wattage
                import re
                match = re.search(r'(\d+)\s*w', p_lower)
                if match:
                    value = float(match.group(1))
            
            mock_data["boundary_conditions"].append({
                "type": "heat_source",
                "target": "Base_Surface",
                "value": value
            })
            
        # Parse convection
        if "convection" in p_lower or "ambient" in p_lower:
            mock_data["boundary_conditions"].append({
                "type": "convection",
                "target": "Exterior",
                "value": 20.0,
                "ambient_temperature": 25.0
            })
            
        return json.dumps(mock_data)
