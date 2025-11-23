"""
Claude API Integration for Mesh Iteration
==========================================

Provides intelligent mesh analysis and iteration suggestions using Claude API.

Features:
- Analyzes mesh quality and identifies problems
- Suggests parameter improvements
- Generates modified strategy code
- Iterative mesh refinement assistance

Requirements:
    pip install anthropic python-dotenv
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

try:
    from anthropic import Anthropic
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("[!] Anthropic SDK not installed. Install with: pip install anthropic")

try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("[!] python-dotenv not installed. Install with: pip install python-dotenv")


class ClaudeMeshAssistant:
    """
    AI assistant for mesh generation and quality improvement

    Uses Claude API to provide intelligent suggestions for mesh refinement.
    """

    def __init__(self, api_key: Optional[str] = None, use_thinking: bool = False):
        """
        Initialize Claude assistant

        Args:
            api_key: Anthropic API key (REQUIRED - must be provided as parameter)
            use_thinking: Enable extended thinking for more accurate responses (slower, more expensive)
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

        # API key must be provided as parameter (not from environment for security)
        self.api_key = api_key

        if not self.api_key:
            raise ValueError(
                "No API key provided. You must provide your API key when creating ClaudeMeshAssistant.\n"
                "Get your API key from: https://console.anthropic.com/\n\n"
                "SECURITY: Never store your API key in code or .env files that are committed to git.\n"
                "The key should be entered at runtime only."
            )

        # Initialize client
        self.client = Anthropic(api_key=self.api_key)

        # Model configuration
        self.model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
        self.max_tokens = int(os.getenv("MAX_TOKENS", "4096"))
        self.use_thinking = use_thinking

        # Conversation history
        self.conversation_history = []

        # System prompt for mesh generation expertise
        self.system_prompt = """You are an expert in finite element mesh generation and CAD geometry processing. You specialize in:
- Analyzing mesh quality metrics (SICN, Gamma, skewness, aspect ratio)
- Identifying problem areas in meshes (inverted elements, poor quality regions)
- Suggesting mesh parameter improvements
- Optimizing meshing strategies for better quality
- Understanding gmsh, tetrahedral meshing, and quality metrics

When analyzing meshes:
1. Always check SICN (target >0.3 acceptable, >0.5 good, >0.7 excellent)
2. Check Gamma (target >0.2 acceptable, >0.4 good)
3. Identify specific problem areas (corners, edges, thin features)
4. Suggest concrete parameter changes
5. Explain the reasoning behind your suggestions

When suggesting code changes:
- Provide specific parameter values
- Explain why each change will help
- Consider trade-offs (quality vs element count vs time)
- Format code changes clearly with before/after examples

Be concise but thorough. Focus on actionable improvements."""

        thinking_status = "ON (slower, more accurate)" if self.use_thinking else "OFF (faster)"
        print(f"[OK] Claude assistant initialized (model: {self.model}, thinking: {thinking_status})")

    def set_thinking_mode(self, enabled: bool):
        """Enable or disable extended thinking mode"""
        self.use_thinking = enabled
        status = "enabled" if enabled else "disabled"
        print(f"Extended thinking {status}")

    def analyze_mesh_quality(self, mesh_data: Dict) -> str:
        """
        Analyze mesh quality and provide detailed assessment

        Args:
            mesh_data: Dictionary with mesh metrics and info

        Returns:
            Analysis text from Claude
        """
        # Build context message
        context = self._build_mesh_context(mesh_data)

        prompt = f"""Please analyze this mesh and identify any quality issues:

{context}

Provide:
1. Overall quality assessment
2. Specific problem areas (if any)
3. Suggested improvements
4. Expected impact of improvements"""

        return self._send_message(prompt)

    def suggest_improvements(self, mesh_data: Dict, current_config: Dict) -> str:
        """
        Suggest specific parameter improvements for mesh generation

        Args:
            mesh_data: Current mesh quality metrics
            current_config: Current meshing configuration

        Returns:
            Improvement suggestions from Claude
        """
        context = self._build_mesh_context(mesh_data)
        config_str = json.dumps(current_config, indent=2)

        prompt = f"""Current mesh quality:
{context}

Current configuration:
{config_str}

Please suggest specific parameter changes to improve mesh quality. Include:
1. Which parameters to change and to what values
2. Why these changes will help
3. Expected quality improvement
4. Any trade-offs (element count, computation time, etc.)

Format suggestions as actionable configuration changes."""

        return self._send_message(prompt)

    def generate_strategy_code(self, mesh_data: Dict, goal: str) -> str:
        """
        Generate Python code for a modified meshing strategy

        Args:
            mesh_data: Current mesh info
            goal: What to improve (e.g., "improve corner quality")

        Returns:
            Python code for new strategy
        """
        context = self._build_mesh_context(mesh_data)

        prompt = f"""Current mesh state:
{context}

Goal: {goal}

Please generate Python code for a modified meshing strategy that addresses this goal.
Include:
1. Strategy function implementation
2. Parameter settings
3. Comments explaining the changes
4. Expected outcome

Use gmsh API and follow the existing strategy pattern."""

        return self._send_message(prompt)

    def iterate_on_mesh(self,
                       previous_result: Dict,
                       new_result: Dict,
                       what_changed: str) -> str:
        """
        Analyze iteration results and suggest next steps

        Args:
            previous_result: Previous iteration metrics
            new_result: New iteration metrics
            what_changed: Description of what was changed

        Returns:
            Analysis and next step suggestions
        """
        prev_sicn = previous_result.get('gmsh_sicn', {}).get('min', 'N/A')
        new_sicn = new_result.get('gmsh_sicn', {}).get('min', 'N/A')

        prompt = f"""Iteration results:

Previous mesh:
- SICN min: {prev_sicn}
- Elements: {previous_result.get('total_elements', 'N/A')}

Changes made: {what_changed}

New mesh:
- SICN min: {new_sicn}
- Elements: {new_result.get('total_elements', 'N/A')}

Analysis:
1. Did quality improve?
2. What worked well?
3. What didn't work?
4. Should we continue iterating?
5. If yes, what to try next?"""

        return self._send_message(prompt)

    def chat(self, user_message: str, mesh_data: Optional[Dict] = None) -> str:
        """
        General chat interface with optional mesh context

        Args:
            user_message: User's question or request
            mesh_data: Optional current mesh data for context

        Returns:
            Claude's response
        """
        # Add mesh context if provided
        if mesh_data:
            context = self._build_mesh_context(mesh_data)
            full_message = f"Current mesh:\n{context}\n\nUser question: {user_message}"
        else:
            full_message = user_message

        return self._send_message(full_message)

    def _send_message(self, content: str, max_retries: int = 3) -> str:
        """
        Send message to Claude API and get response with retry logic

        Args:
            content: Message content
            max_retries: Maximum number of retry attempts for overload errors

        Returns:
            Claude's response text
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": content
        })

        last_error = None

        for attempt in range(max_retries):
            try:
                # Call Claude API with optional extended thinking
                api_params = {
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "system": self.system_prompt,
                    "messages": self.conversation_history
                }

                # Add thinking parameter if enabled
                if self.use_thinking:
                    api_params["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": 5000  # Allow up to 5k tokens for thinking
                    }

                response = self.client.messages.create(**api_params)

                # Extract response text
                response_text = response.content[0].text

                # Add assistant response to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response_text
                })

                return response_text

            except anthropic.RateLimitError as e:
                last_error = e
                wait_time = (2 ** attempt) * 1  # Exponential backoff: 1s, 2s, 4s
                print(f"[!] Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                continue

            except anthropic.APIStatusError as e:
                # Check if it's an overload error (529)
                if e.status_code == 529:
                    last_error = e
                    wait_time = (2 ** attempt) * 2  # Exponential backoff: 2s, 4s, 8s
                    print(f"[!] API overloaded (529), waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                    if attempt < max_retries - 1:
                        time.sleep(wait_time)
                    continue
                else:
                    # Other status errors (auth, etc.) - don't retry
                    error_msg = f"Claude API error: {e}"
                    print(f"[X] {error_msg}")
                    # Remove user message from history since it failed
                    self.conversation_history.pop()
                    return f"Error: {error_msg}\n\nPlease check your API key and connection."

            except anthropic.APIError as e:
                error_msg = f"Claude API error: {e}"
                print(f"[X] {error_msg}")
                # Remove user message from history since it failed
                self.conversation_history.pop()
                return f"Error: {error_msg}\n\nPlease check your API key and connection."

            except Exception as e:
                error_msg = f"Unexpected error: {e}"
                print(f"[X] {error_msg}")
                # Remove user message from history since it failed
                self.conversation_history.pop()
                return f"Error: {error_msg}"

        # All retries exhausted
        error_msg = f"Claude API error: {last_error}"
        print(f"[X] Max retries exhausted: {error_msg}")
        # Remove user message from history since it failed
        self.conversation_history.pop()
        return f"Error: {error_msg}\n\nThe API is currently overloaded. Please try again in a few minutes."

    def _build_mesh_context(self, mesh_data: Dict) -> str:
        """
        Build formatted context string from mesh data

        Args:
            mesh_data: Dictionary with mesh information

        Returns:
            Formatted context string
        """
        lines = []

        # Basic info
        if 'file_name' in mesh_data:
            lines.append(f"File: {mesh_data['file_name']}")

        lines.append(f"Elements: {mesh_data.get('total_elements', 'N/A'):,}")
        lines.append(f"Nodes: {mesh_data.get('total_nodes', 'N/A'):,}")

        # Quality metrics
        lines.append("\nQuality Metrics:")

        if 'gmsh_sicn' in mesh_data and mesh_data['gmsh_sicn']:
            sicn = mesh_data['gmsh_sicn']
            lines.append(f"  SICN: min={sicn['min']:.4f}, avg={sicn['avg']:.4f}, max={sicn['max']:.4f}")

        if 'gmsh_gamma' in mesh_data and mesh_data['gmsh_gamma']:
            gamma = mesh_data['gmsh_gamma']
            lines.append(f"  Gamma: min={gamma['min']:.4f}, avg={gamma['avg']:.4f}, max={gamma['max']:.4f}")

        if 'skewness' in mesh_data and mesh_data['skewness']:
            skew = mesh_data['skewness']
            lines.append(f"  Skewness: min={skew['min']:.4f}, avg={skew['avg']:.4f}, max={skew['max']:.4f}")

        if 'aspect_ratio' in mesh_data and mesh_data['aspect_ratio']:
            ar = mesh_data['aspect_ratio']
            lines.append(f"  Aspect Ratio: min={ar['min']:.2f}, avg={ar['avg']:.2f}, max={ar['max']:.2f}")

        # Problem identification
        problems = []
        if 'gmsh_sicn' in mesh_data and mesh_data['gmsh_sicn']:
            sicn_min = mesh_data['gmsh_sicn']['min']
            if sicn_min < 0:
                problems.append("CRITICAL: Inverted elements detected!")
            elif sicn_min < 0.3:
                problems.append("Poor quality elements (SICN < 0.3)")
            elif sicn_min < 0.5:
                problems.append("Below target quality (SICN < 0.5)")

        if problems:
            lines.append("\nIssues Detected:")
            for problem in problems:
                lines.append(f"  - {problem}")

        return "\n".join(lines)

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("Conversation history cleared")

    def get_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.conversation_history.copy()

    def save_conversation(self, filepath: str):
        """
        Save conversation history to file

        Args:
            filepath: Path to save JSON file
        """
        data = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "conversation": self.conversation_history
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"[OK] Conversation saved to {filepath}")

    def load_conversation(self, filepath: str):
        """
        Load conversation history from file

        Args:
            filepath: Path to JSON file
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.conversation_history = data.get("conversation", [])
        print(f"[OK] Conversation loaded from {filepath}")


def setup_api_key():
    """
    Interactive setup for API key

    Guides user through getting and setting up their Anthropic API key.
    """
    print("=" * 70)
    print("CLAUDE API KEY SETUP")
    print("=" * 70)
    print()

    # Check if already set
    existing_key = os.getenv("ANTHROPIC_API_KEY")
    if existing_key:
        print(f"[OK] API key already set: {existing_key[:15]}...")
        response = input("Do you want to update it? [y/N]: ").lower()
        if response != 'y':
            print("Keeping existing API key")
            return

    print("To use the AI assistant, you need an Anthropic API key.")
    print()
    print("Steps:")
    print("1. Go to: https://console.anthropic.com/")
    print("2. Sign up or log in")
    print("3. Navigate to API Keys")
    print("4. Create a new key")
    print()

    api_key = input("Paste your API key here: ").strip()

    if not api_key:
        print("[X] No API key provided")
        return

    # Save to .env file
    env_file = Path(__file__).parent.parent / ".env"

    # Read existing .env if it exists
    env_lines = []
    if env_file.exists():
        with open(env_file, 'r') as f:
            env_lines = [line for line in f.readlines() if not line.startswith("ANTHROPIC_API_KEY=")]

    # Add new API key
    env_lines.append(f"ANTHROPIC_API_KEY={api_key}\n")

    # Write back
    with open(env_file, 'w') as f:
        f.writelines(env_lines)

    print()
    print(f"[OK] API key saved to {env_file}")
    print("[OK] Restart the application to use the AI assistant")
    print()


if __name__ == "__main__":
    # Test / setup mode
    if not ANTHROPIC_AVAILABLE:
        print("Installing required packages...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "anthropic", "python-dotenv"])
        print("\nPackages installed. Please restart the application.")
        sys.exit(0)

    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        setup_api_key()
        sys.exit(0)

    # Test connection
    try:
        assistant = ClaudeMeshAssistant()
        print("\n[OK] Claude assistant initialized successfully")
        print("Ready to provide mesh analysis and suggestions!")

        # Test message
        test_response = assistant.chat("Hello! Can you help me improve mesh quality?")
        print("\nTest response:")
        print(test_response)

    except Exception as e:
        print(f"\n[X] Setup failed: {e}")
        print("\nRun this script again to set up your API key:")
        print("  python core/claude_integration.py")
