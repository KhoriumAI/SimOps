import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from google import genai

NOISE_KEY_NAMES: Set[str] = {
    # Parser / metadata only, present in many dictionaries.
    "foam_class",
    "foam_object",
    "foam_location",
    "extra_data",
}

NOISE_PATHS: Set[str] = {
    "system.controlDict.writeControl",
    "system.controlDict.writeInterval",
    "system.controlDict.purgeWrite",
    "system.controlDict.writeFormat",
    "system.controlDict.writePrecision",
    "system.controlDict.writeCompression",
    "system.controlDict.timeFormat",
    "system.controlDict.timePrecision",
}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass
class ConfigChange:
    path: str
    change_type: str
    old_value: Any
    new_value: Any
    numeric_delta: Optional[float] = None
    numeric_pct: Optional[float] = None


def _format_value(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return repr(value)


def _compute_numeric_change(old: Any, new: Any) -> (Optional[float], Optional[float]):
    if not (_is_number(old) and _is_number(new)):
        return None, None

    delta = float(new) - float(old)
    if float(old) == 0.0:
        return delta, None

    pct = (delta / float(old)) * 100.0
    return delta, pct


def _is_noise(path: str, key: str) -> bool:
    """
    Return True if this (path, key) should be treated as noise and ignored.

    - Key-based: parser / metadata fields that don't affect physics.
    - Path-based: known logging / write-control settings in controlDict.
    """
    key_lower = key.lower()
    if key_lower in NOISE_KEY_NAMES:
        return True
    if path in NOISE_PATHS:
        return True
    return False


def _join_path(parent: str, key: str) -> str:
    if parent:
        return f"{parent}.{key}"
    return key


def _compare_dicts(
    base: Dict[str, Any],
    new: Dict[str, Any],
    parent_path: str,
    changes: List[ConfigChange],
) -> None:
    base_keys = set(base.keys())
    new_keys = set(new.keys())

    for key in sorted(base_keys - new_keys):
        path = _join_path(parent_path, key)
        changes.append(
            ConfigChange(
                path=path,
                change_type="removed",
                old_value=base[key],
                new_value=None,
            )
        )

    for key in sorted(new_keys - base_keys):
        path = _join_path(parent_path, key)
        if _is_noise(path, key):
            continue
        changes.append(
            ConfigChange(
                path=path,
                change_type="added",
                old_value=None,
                new_value=new[key],
            )
        )

    for key in sorted(base_keys & new_keys):
        old_val = base[key]
        new_val = new[key]
        path = _join_path(parent_path, key)
        if _is_noise(path, key):
            continue

        if isinstance(old_val, dict) and isinstance(new_val, dict):
            _compare_dicts(old_val, new_val, path, changes)
        elif isinstance(old_val, list) and isinstance(new_val, list):
            if (
                len(old_val) == 2
                and len(new_val) == 2
                and isinstance(old_val[1], dict)
                and isinstance(new_val[1], dict)
            ):
                if old_val[0] != new_val[0]:
                    changes.append(
                        ConfigChange(
                            path=path,
                            change_type="modified",
                            old_value=old_val[0],
                            new_value=new_val[0],
                        )
                    )
                _compare_dicts(old_val[1], new_val[1], path, changes)
            elif old_val != new_val:
                delta, pct = None, None
                if all(_is_number(v) for v in old_val + new_val):
                    delta, pct = _compute_numeric_change(
                        sum(old_val), sum(new_val)
                    )
                changes.append(
                    ConfigChange(
                        path=path,
                        change_type="modified",
                        old_value=old_val,
                        new_value=new_val,
                        numeric_delta=delta,
                        numeric_pct=pct,
                    )
                )
        else:
            if old_val != new_val:
                delta, pct = _compute_numeric_change(old_val, new_val)
                changes.append(
                    ConfigChange(
                        path=path,
                        change_type="modified",
                        old_value=old_val,
                        new_value=new_val,
                        numeric_delta=delta,
                        numeric_pct=pct,
                    )
                )


def compare_configs(
    base_config: Dict[str, Any], new_config: Dict[str, Any]
) -> List[ConfigChange]:
    """
    Compare two configuration dictionaries and return a list of changes.

    Numeric changes include absolute and percentage deltas where applicable.
    Structural changes (added/removed keys) are also captured.
    """
    changes: List[ConfigChange] = []

    if not isinstance(base_config, dict) or not isinstance(new_config, dict):
        raise ValueError("Both configurations must be JSON objects at the top level.")

    _compare_dicts(base_config, new_config, parent_path="", changes=changes)
    return changes


def format_changes_markdown(changes: List[ConfigChange]) -> str:
    """
    Format a list of ConfigChange objects into a Markdown bullet list.
    """
    if not changes:
        return "No differences found."

    lines: List[str] = []
    for change in changes:
        path = change.path

        if change.change_type == "added":
            lines.append(
                f"- **{path}**: added with value `{_format_value(change.new_value)}`"
            )
        elif change.change_type == "removed":
            lines.append(
                f"- **{path}**: removed (was `{_format_value(change.old_value)}`)"
            )
        elif change.change_type == "modified":
            old_str = _format_value(change.old_value)
            new_str = _format_value(change.new_value)

            if (
                change.numeric_delta is not None
                and _is_number(change.old_value)
                and _is_number(change.new_value)
            ):
                direction = (
                    "increased"
                    if float(change.new_value) > float(change.old_value)
                    else "decreased"
                )
                if change.numeric_pct is not None:
                    pct_abs = abs(change.numeric_pct)
                    lines.append(
                        f"- **{path}**: {direction} by {pct_abs:.1f}% "
                        f"({old_str} → {new_str})"
                    )
                else:
                    lines.append(
                        f"- **{path}**: changed from {old_str} to {new_str} "
                        "(percent change undefined for zero baseline)"
                    )
            else:
                lines.append(
                    f"- **{path}**: changed from `{old_str}` to `{new_str}`"
                )

    return "\n".join(lines)


class ConfigDiffSummarizer:
    """
    Uses Gemini to generate a semantic summary of configuration differences.
    """

    def __init__(self, api_key: str, model_name: str = "gemini-3-flash-preview"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.system_prompt = """
        You are an expert CFD / physics simulation engineer.
        You are given a low-level list of JSON configuration differences between
        a baseline (working) simulation and a new (possibly broken) simulation.

        Your task:
        - Produce a concise, human-readable, high-level "Semantic Summary" of
          the differences, focusing on physical meaning and algorithm choices.
        - Use clear bullet points like:
          - "Inlet velocity increased by 50% (10 → 15 m/s)"
          - "Turbulence model switched from k-Epsilon to k-Omega-SST"
        - Group related parameters when appropriate.
        - Avoid restating raw JSON paths; instead, infer the physical concept
          when possible.
        - Keep the answer under 150 words.
        """

    def summarize(self, diff_markdown: str) -> str:
        prompt = f"""
        Here is a machine-generated list of low-level JSON config differences
        between an OLD base configuration and a NEW configuration:

        {diff_markdown}

        Please rewrite these as a high-level semantic summary for a CFD engineer,
        following the instructions above.
        """
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "system_instruction": self.system_prompt,
                    "temperature": 0.0,
                },
            )
            return response.text
        except Exception as exc:
            return f"Error generating semantic summary: {exc}"


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def main() -> None:
    """
    Compare two JSON configuration files and optionally ask Gemini
    for a semantic summary of the differences.

    Usage example:
        python compare_configs.py --old run_1.json --new run_2.json
    """
    parser = argparse.ArgumentParser(
        description="Compare two JSON config files and summarize differences."
    )
    parser.add_argument(
        "--old",
        required=True,
        help="Path to base / old configuration JSON file.",
    )
    parser.add_argument(
        "--new",
        required=True,
        help="Path to new configuration JSON file.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.old):
        raise SystemExit(f"Old config not found: {args.old}")
    if not os.path.exists(args.new):
        raise SystemExit(f"New config not found: {args.new}")

    base_config = _load_json(args.old)
    new_config = _load_json(args.new)

    changes = compare_configs(base_config, new_config)
    diff_md = format_changes_markdown(changes)

    print("\n=== Raw Config Differences ===")
    print(diff_md)

    has_removed = any(c.change_type == "removed" for c in changes)
    if has_removed:
        print(
            "\nDetected missing keys/structures (marked as 'removed').\n"
            "Please restore or explicitly decide on these changes, then re-run\n"
            "this tool. Skipping Gemini semantic summary."
        )
        return

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print(
            "\nGOOGLE_API_KEY not set; skipping Gemini semantic summary.\n"
            "Set GOOGLE_API_KEY to enable high-level summaries."
        )
        return

    summarizer = ConfigDiffSummarizer(api_key=api_key)
    print("\n=== Gemini Semantic Summary ===")
    summary = summarizer.summarize(diff_md)
    print(summary)

    try:
        with open("summary.md", "w") as f:
            f.write("# Config Difference Semantic Summary\n\n")
            f.write(summary)
            f.write("\n")
    except Exception as exc:
        print(f"\nWarning: failed to write summary.md: {exc}")


if __name__ == "__main__":
    main()
