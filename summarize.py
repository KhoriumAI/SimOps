import os
import json
from google import genai
from typing import Dict, Any, Optional

class CaseSummarizer:
    def __init__(self, api_key: str, model_name: str = "gemini-3-flash-preview"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.system_prompt = """
        You are an expert OpenFOAM computational fluid dynamics (CFD) engineer.
        Your task is to take a JSON representation of an OpenFOAM case and translate it into a natural language "Pre-Flight Check" summary for the user.
        
        The summary should be a single, cohesive paragraph that:
        1. Identifies the type of simulation.
        2. Describes the physical properties and key boundary conditions.
        3. States the run parameters.
        4. Ends with "Proceed?".
        
        Keep it concise (under 100 words).
        
        Example Output:
        "I am setting up a Laminar flow simulation. Air will enter at 10 m/s and the wall is heated to 27°C (300K). The sim will run for 1000 iterations (0 to 10s). Proceed?"
        """

    def summarize(self, case_json: Dict[str, Any]) -> str:
        case_str = json.dumps(case_json, indent=2)
        
        prompt = f"""
        Please summarize the following OpenFOAM case configuration:
        
        {case_str}
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "system_instruction": self.system_prompt,
                    "temperature": 0.0,
                }
            )
            return response.text
        except Exception as e:
            return f"Error generating summary: {str(e)}"

if __name__ == "__main__":
    if "GOOGLE_API_KEY" not in os.environ:
        print("Error: GOOGLE_API_KEY not set.")
        exit(1)
        
    input_path = "gemini_generated.json"
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Run gemini.py first.")
        exit(1)
        
    print(f"Reading configuration from {input_path}...")
    with open(input_path, "r") as f:
        case_data = json.load(f)
        
    summarizer = CaseSummarizer(api_key=os.environ["GOOGLE_API_KEY"])
    print("Generating summary...")
    summary = summarizer.summarize(case_data)
    
    print("\n--- Pre-Flight Check ---")
    print(summary)
    print("------------------------")
