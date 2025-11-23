"""
Experimental Mesh Iterator
===========================

Manages experimental mesh generation iterations for AI-assisted improvement.

Features:
- Creates isolated experiment folders
- Copies and modifies strategy files
- Generates experimental meshes
- Compares results across iterations
- Maintains iteration history

Directory structure:
    experiments/
        iteration_001/
            strategy.py
            config.json
            mesh.msh
            results.json
        iteration_002/
        ...
"""

import os
import shutil
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import subprocess
import sys


class MeshIterationExperiment:
    """
    Manages a single mesh iteration experiment

    Each experiment has:
    - Unique ID and folder
    - Modified strategy code
    - Configuration parameters
    - Generated mesh
    - Quality metrics
    """

    def __init__(self, experiment_id: int, base_dir: Path):
        self.id = experiment_id
        self.folder = base_dir / f"iteration_{experiment_id:03d}"
        self.folder.mkdir(parents=True, exist_ok=True)

        self.strategy_file = self.folder / "strategy.py"
        self.config_file = self.folder / "config.json"
        self.mesh_file = self.folder / "mesh.msh"
        self.results_file = self.folder / "results.json"
        self.log_file = self.folder / "log.txt"

        self.results = {}
        self.timestamp = datetime.now().isoformat()

    def save_strategy(self, strategy_code: str):
        """Save modified strategy code"""
        with open(self.strategy_file, 'w') as f:
            f.write(strategy_code)

    def save_config(self, config: Dict):
        """Save configuration parameters"""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)

    def save_results(self, metrics: Dict):
        """Save mesh quality results"""
        self.results = {
            'timestamp': self.timestamp,
            'metrics': metrics,
            'mesh_file': str(self.mesh_file),
            'config': self._load_config()
        }

        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

    def _load_config(self) -> Dict:
        """Load configuration from file"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return {}

    def get_results(self) -> Dict:
        """Load results from file"""
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                return json.load(f)
        return self.results

    def __repr__(self):
        return f"Experiment({self.id}, quality={self.results.get('metrics', {}).get('gmsh_sicn', {}).get('min', 'N/A')})"


class ExperimentalMeshIterator:
    """
    Manages iterative mesh improvement experiments

    Workflow:
    1. Create experiment from current settings
    2. Modify parameters or strategy code
    3. Generate experimental mesh
    4. Compare with previous iterations
    5. Repeat or apply best result
    """

    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize iterator

        Args:
            base_dir: Base directory for experiments (default: ./experiments)
        """
        if base_dir is None:
            base_dir = Path(__file__).parent.parent / "experiments"

        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

        # Create .gitignore for experiments folder
        gitignore = self.base_dir / ".gitignore"
        if not gitignore.exists():
            with open(gitignore, 'w') as f:
                f.write("# Experimental mesh iterations (do not commit)\n")
                f.write("*\n")
                f.write("!.gitignore\n")

        self.experiments: List[MeshIterationExperiment] = []
        self._load_existing_experiments()

        print(f"[OK] Experimental iterator initialized")
        print(f"  Base directory: {self.base_dir}")
        print(f"  Existing experiments: {len(self.experiments)}")

    def _load_existing_experiments(self):
        """Load existing experiments from disk"""
        for folder in sorted(self.base_dir.glob("iteration_*")):
            if folder.is_dir():
                try:
                    exp_id = int(folder.name.split("_")[1])
                    experiment = MeshIterationExperiment(exp_id, self.base_dir.parent / "experiments")
                    if experiment.results_file.exists():
                        experiment.results = experiment.get_results()
                    self.experiments.append(experiment)
                except (ValueError, IndexError):
                    continue

    def create_experiment(self,
                         description: str,
                         cad_file: str,
                         config: Dict,
                         strategy_modifications: Optional[str] = None) -> MeshIterationExperiment:
        """
        Create a new experiment

        Args:
            description: Description of what's being tested
            cad_file: Path to CAD file
            config: Meshing configuration
            strategy_modifications: Optional custom strategy code

        Returns:
            New experiment object
        """
        # Get next experiment ID
        next_id = len(self.experiments) + 1
        experiment = MeshIterationExperiment(next_id, self.base_dir)

        # Save description in config
        config['description'] = description
        config['cad_file'] = str(cad_file)
        config['created_at'] = datetime.now().isoformat()

        experiment.save_config(config)

        # Copy CAD file to experiment folder
        cad_dest = experiment.folder / Path(cad_file).name
        if not cad_dest.exists():
            shutil.copy(cad_file, cad_dest)

        # Save strategy modifications if provided
        if strategy_modifications:
            experiment.save_strategy(strategy_modifications)

        self.experiments.append(experiment)

        print(f"[OK] Created experiment {next_id}: {description}")
        return experiment

    def run_experiment(self, experiment: MeshIterationExperiment) -> Dict:
        """
        Execute mesh generation for an experiment

        Args:
            experiment: Experiment to run

        Returns:
            Quality metrics from generated mesh
        """
        print(f"\n🔬 Running experiment {experiment.id}...")

        config = experiment._load_config()
        cad_file = config.get('cad_file')

        if not cad_file or not Path(cad_file).exists():
            # Try to find CAD file in experiment folder
            cad_files = list(experiment.folder.glob("*.step")) + list(experiment.folder.glob("*.stp"))
            if cad_files:
                cad_file = str(cad_files[0])
            else:
                raise FileNotFoundError(f"CAD file not found for experiment {experiment.id}")

        # Use subprocess to run mesh generation
        worker_script = Path(__file__).parent.parent / "mesh_worker_subprocess.py"

        # Prepare quality params for worker
        quality_params = {
            'quality_preset': config.get('quality_preset', 'Medium'),
            'target_elements': config.get('target_elements', 10000),
            'max_size_mm': config.get('max_size_mm', 100),
            'curvature_adaptive': config.get('curvature_adaptive', False)
        }

        # DEBUG: Log what we're actually sending to mesh generator
        print(f"\n📋 Parameters being sent to mesh generator:")
        print(f"   quality_preset: {quality_params['quality_preset']}")
        print(f"   target_elements: {quality_params['target_elements']:,}")
        print(f"   max_size_mm: {quality_params['max_size_mm']}")
        print(f"   curvature_adaptive: {quality_params['curvature_adaptive']}")

        cmd = [
            sys.executable,
            str(worker_script),
            cad_file,
            '--quality-params',
            json.dumps(quality_params)
        ]

        # Run mesh generation
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            # Parse output
            for line in result.stdout.split('\n'):
                if line.strip().startswith('{') and '"success"' in line:
                    result_data = json.loads(line)

                    if result_data.get('success'):
                        # Copy mesh file to experiment folder
                        source_mesh = result_data.get('output_file')
                        if source_mesh and Path(source_mesh).exists():
                            shutil.copy(source_mesh, experiment.mesh_file)

                        # Save results
                        metrics = result_data.get('metrics', {})
                        experiment.save_results(metrics)

                        # Save log
                        with open(experiment.log_file, 'w') as f:
                            f.write(result.stdout)

                        print(f"[OK] Experiment {experiment.id} completed")
                        print(f"  Elements: {metrics.get('total_elements', 'N/A'):,}")
                        if 'gmsh_sicn' in metrics:
                            print(f"  SICN min: {metrics['gmsh_sicn']['min']:.4f}")

                        return metrics

            # If we get here, no success message found
            print(f"[X] Experiment {experiment.id} failed")
            print(f"Output: {result.stdout}")
            return {}

        except subprocess.TimeoutExpired:
            print(f"[X] Experiment {experiment.id} timed out (>5 minutes)")
            return {}

        except Exception as e:
            print(f"[X] Experiment {experiment.id} error: {e}")
            return {}

    def compare_experiments(self, exp_ids: Optional[List[int]] = None) -> str:
        """
        Compare results across experiments

        Args:
            exp_ids: Optional list of experiment IDs to compare (default: all)

        Returns:
            Formatted comparison text
        """
        if exp_ids is None:
            experiments = self.experiments
        else:
            experiments = [e for e in self.experiments if e.id in exp_ids]

        if not experiments:
            return "No experiments to compare"

        lines = []
        lines.append("=" * 70)
        lines.append("EXPERIMENT COMPARISON")
        lines.append("=" * 70)
        lines.append()

        # Sort by quality (SICN)
        experiments_with_data = []
        for exp in experiments:
            results = exp.get_results()
            if results and 'metrics' in results:
                metrics = results['metrics']
                sicn = metrics.get('gmsh_sicn', {}).get('min', -999)
                experiments_with_data.append((exp, sicn, metrics))

        experiments_with_data.sort(key=lambda x: x[1], reverse=True)

        # Display table
        lines.append(f"{'ID':<6} {'SICN':<10} {'Elements':<12} {'Description':<30}")
        lines.append("-" * 70)

        for exp, sicn, metrics in experiments_with_data:
            config = exp._load_config()
            desc = config.get('description', 'N/A')[:28]
            elements = metrics.get('total_elements', 0)

            lines.append(f"{exp.id:<6} {sicn:>8.4f}  {elements:>10,}  {desc}")

        lines.append()

        # Show best experiment
        if experiments_with_data:
            best_exp, best_sicn, best_metrics = experiments_with_data[0]
            lines.append("=" * 70)
            lines.append(f"BEST RESULT: Experiment {best_exp.id}")
            lines.append(f"  SICN: {best_sicn:.4f}")
            lines.append(f"  Elements: {best_metrics.get('total_elements', 0):,}")
            if 'gmsh_gamma' in best_metrics:
                lines.append(f"  Gamma: {best_metrics['gmsh_gamma']['min']:.4f}")
            lines.append(f"  Mesh: {best_exp.mesh_file}")

        return "\n".join(lines)

    def get_best_experiment(self) -> Optional[MeshIterationExperiment]:
        """
        Get the experiment with best quality

        Returns:
            Best experiment or None
        """
        best_exp = None
        best_sicn = -999

        for exp in self.experiments:
            results = exp.get_results()
            if results and 'metrics' in results:
                sicn = results['metrics'].get('gmsh_sicn', {}).get('min', -999)
                if sicn > best_sicn:
                    best_sicn = sicn
                    best_exp = exp

        return best_exp

    def apply_best_experiment(self, output_mesh: str) -> bool:
        """
        Copy the best experimental mesh to output location

        Args:
            output_mesh: Destination path for mesh

        Returns:
            True if successful
        """
        best = self.get_best_experiment()

        if not best:
            print("[X] No successful experiments found")
            return False

        if not best.mesh_file.exists():
            print(f"[X] Best experiment mesh not found: {best.mesh_file}")
            return False

        # Copy mesh
        shutil.copy(best.mesh_file, output_mesh)
        print(f"[OK] Applied best experiment {best.id}")
        print(f"  Copied: {best.mesh_file} -> {output_mesh}")

        return True

    def quick_experiment(self,
                        cad_file: str,
                        description: str,
                        **config_overrides) -> Tuple[MeshIterationExperiment, Dict]:
        """
        Convenience method: create and run experiment in one call

        Args:
            cad_file: Path to CAD file
            description: Experiment description
            **config_overrides: Configuration parameters to override

        Returns:
            (experiment, metrics) tuple
        """
        # Default config
        config = {
            'quality_preset': 'Medium',
            'target_elements': 10000,
            'max_size_mm': 100,
            'curvature_adaptive': False
        }
        config.update(config_overrides)

        # Create experiment
        experiment = self.create_experiment(description, cad_file, config)

        # Run it
        metrics = self.run_experiment(experiment)

        return experiment, metrics

    def generate_fast_baseline(self, cad_file: str) -> Tuple[MeshIterationExperiment, Dict]:
        """
        Generate a fast "crappy" baseline mesh for iteration

        Args:
            cad_file: Path to CAD file

        Returns:
            (experiment, metrics) tuple
        """
        return self.quick_experiment(
            cad_file,
            "Fast baseline (coarse)",
            quality_preset='Coarse',
            target_elements=2000,
            max_size_mm=200
        )

    def cleanup_old_experiments(self, keep_count: int = 10):
        """
        Remove old experiment folders, keeping only the most recent

        Args:
            keep_count: Number of experiments to keep
        """
        if len(self.experiments) <= keep_count:
            print(f"Only {len(self.experiments)} experiments, no cleanup needed")
            return

        # Sort by ID (newest first)
        sorted_exps = sorted(self.experiments, key=lambda e: e.id, reverse=True)

        # Keep recent, remove old
        to_remove = sorted_exps[keep_count:]

        for exp in to_remove:
            try:
                shutil.rmtree(exp.folder)
                print(f"Removed experiment {exp.id}")
            except Exception as e:
                print(f"Failed to remove experiment {exp.id}: {e}")

        # Reload experiments
        self.experiments = []
        self._load_existing_experiments()

        print(f"[OK] Cleanup complete, kept {len(self.experiments)} experiments")

    def multi_iteration_ai_experiment(self,
                                      cad_file: str,
                                      current_mesh_data: Dict,
                                      current_config: Dict,
                                      claude_assistant,
                                      num_iterations: int = 5) -> Tuple[List[str], List[MeshIterationExperiment], List[Dict]]:
        """
        Run multiple AI-driven experiments with code modifications and learning

        Args:
            cad_file: Path to CAD file
            current_mesh_data: Current mesh quality metrics
            current_config: Current configuration
            claude_assistant: ClaudeMeshAssistant instance
            num_iterations: Number of iterations to run (default: 5)

        Returns:
            (explanations, experiments, all_metrics) tuple
        """
        print(f"\n[ROCKET] Starting Multi-Iteration AI Experiment ({num_iterations} trials)")
        print("=" * 70)

        # Load existing strategy code to show AI
        strategy_code = self._load_strategy_examples()

        explanations = []
        experiments = []
        all_metrics = []

        # Baseline
        baseline_sicn = current_mesh_data.get('gmsh_sicn', {}).get('min', 0)
        best_sicn = baseline_sicn
        best_iteration = 0

        print(f"\n📊 Baseline Quality: SICN = {baseline_sicn:.4f}")
        print(f"   Target: Improve mesh quality through algorithmic changes\n")

        for iteration in range(1, num_iterations + 1):
            print(f"\n{'='*70}")
            print(f"🔬 ITERATION {iteration}/{num_iterations}")
            print(f"{'='*70}")

            # Build prompt with history
            if iteration == 1:
                prompt = self._build_initial_prompt(current_mesh_data, current_config, strategy_code)
            else:
                prompt = self._build_iteration_prompt(
                    current_mesh_data,
                    current_config,
                    strategy_code,
                    all_metrics,
                    explanations,
                    iteration
                )

            print(f"🤖 Asking AI for iteration {iteration} strategy...")

            # Get AI response
            response = claude_assistant.chat(prompt, current_mesh_data)

            # Parse response
            explanation = self._extract_explanation(response)
            strategy_modifications = self._extract_strategy_code(response)
            param_changes = self._extract_parameter_changes(response)

            print(f"\n💡 AI Strategy for Iteration {iteration}:")
            print(f"   {explanation}")

            # Create experiment with code modifications
            new_config = current_config.copy()
            if param_changes:
                new_config.update(param_changes)
                print(f"\n🔧 Parameter changes:")
                for key, value in param_changes.items():
                    print(f"   {key}: {current_config.get(key)} -> {value}")

            description = f"AI Iteration {iteration}: {explanation[:80]}"
            experiment = self.create_experiment(description, cad_file, new_config, strategy_modifications)

            # Save AI explanation for this iteration
            with open(experiment.folder / "ai_explanation.txt", 'w') as f:
                f.write(f"Iteration {iteration}\n")
                f.write(f"="*50 + "\n\n")
                f.write(explanation + "\n\n")
                f.write(f"Full AI Response:\n")
                f.write("="*50 + "\n")
                f.write(response)

            print(f"\n⚙️  Running mesh generation...")
            metrics = self.run_experiment(experiment)

            if metrics:
                new_sicn = metrics.get('gmsh_sicn', {}).get('min', 0)
                improvement = ((new_sicn - baseline_sicn) / baseline_sicn * 100) if baseline_sicn > 0 else 0

                print(f"\n[OK] Iteration {iteration} Complete!")
                print(f"   SICN: {new_sicn:.4f} (baseline: {baseline_sicn:.4f})")
                print(f"   Improvement: {improvement:+.1f}%")
                print(f"   Elements: {metrics.get('total_elements', 0):,}")

                if new_sicn > best_sicn:
                    best_sicn = new_sicn
                    best_iteration = iteration
                    print(f"   🌟 NEW BEST RESULT!")

                explanations.append(explanation)
                experiments.append(experiment)
                all_metrics.append(metrics)
            else:
                print(f"\n[X] Iteration {iteration} failed")
                explanations.append(f"Failed: {explanation}")
                experiments.append(experiment)
                all_metrics.append({})

        # Final summary
        print(f"\n{'='*70}")
        print(f"🎯 FINAL RESULTS - {num_iterations} Iterations Complete")
        print(f"{'='*70}")
        print(f"\nBaseline SICN: {baseline_sicn:.4f}")
        print(f"Best SICN:     {best_sicn:.4f} (Iteration {best_iteration})")
        print(f"Improvement:   {((best_sicn - baseline_sicn) / baseline_sicn * 100):+.1f}%\n")

        print("Iteration Summary:")
        for i, (exp, metrics) in enumerate(zip(experiments, all_metrics), 1):
            if metrics:
                sicn = metrics.get('gmsh_sicn', {}).get('min', 0)
                elements = metrics.get('total_elements', 0)
                print(f"  {i}. SICN: {sicn:.4f}, Elements: {elements:,}")
            else:
                print(f"  {i}. Failed")

        print(f"\n[FILE] All results saved to: {self.base_dir}")

        return (explanations, experiments, all_metrics)

    def _load_strategy_examples(self) -> str:
        """Load existing strategy code to show AI"""
        strategies_dir = Path(__file__).parent.parent / "strategies"

        # Load adaptive strategy as example
        adaptive_file = strategies_dir / "adaptive_strategy.py"
        if adaptive_file.exists():
            with open(adaptive_file, 'r') as f:
                return f.read()

        return "# No strategy examples found"

    def _build_initial_prompt(self, mesh_data: Dict, config: Dict, strategy_code: str) -> str:
        """Build prompt for first iteration"""
        geom_acc = mesh_data.get('geometric_accuracy', 0)
        sicn = mesh_data.get('gmsh_sicn', {}).get('min', 0)

        quality_summary = []
        if geom_acc > 0:
            quality_summary.append(f"Shape Accuracy: {geom_acc:.3f}")
        quality_summary.append(f"SICN: {sicn:.4f}")

        return f"""You are a mesh generation expert. Current mesh quality: {', '.join(quality_summary)}

Your task: Generate a MODIFIED MESHING STRATEGY to improve mesh quality.

IMPORTANT: Geometric accuracy (how well mesh matches CAD shape) is the PRIMARY metric. Element quality (SICN) is secondary.

Current mesh metrics:
- Geometric Accuracy: {geom_acc:.3f} (target: > 0.90) - MOST IMPORTANT
- SICN: {json.dumps(mesh_data.get('gmsh_sicn', {}), indent=2)}
- Skewness: {json.dumps(mesh_data.get('skewness', {}), indent=2)}
- Elements: {mesh_data.get('total_elements', 0):,}

Here's an example strategy you can learn from and modify:

```python
{strategy_code[:3000]}
```

Provide your response in this format:

EXPLANATION:
[Explain WHAT changes you're making and WHY - be specific about algorithmic changes AND any mesh size adjustments]

STRATEGY_CODE:
```python
[Complete Python code for modified strategy - must be runnable]
```

CHANGES:
- target_elements: [number - adjust if needed based on geometry complexity]
- max_size_mm: [number - adjust if mesh is too coarse/fine for the model]
- quality_preset: [Coarse/Medium/Fine/Ultra]
- curvature_adaptive: [true/false]

You should explore:
1. Algorithm changes (mesh method, optimization, smoothing)
2. Mesh sizing (if current mesh is clearly too coarse/fine for the geometry)
3. Quality thresholds and refinement strategies
4. Curvature/geometry adaptation

Be intelligent: Use mesh size changes when appropriate (e.g., if mesh is way too coarse for small features),
but don't just brute-force by making meshes smaller - prefer smarter algorithmic approaches."""

    def _build_iteration_prompt(self, mesh_data: Dict, config: Dict, strategy_code: str,
                                 previous_metrics: List[Dict], previous_explanations: List[str],
                                 iteration: int) -> str:
        """Build prompt for subsequent iterations with learning"""

        # Summarize previous attempts (full explanations for better learning)
        history = f"Previous {iteration-1} attempts:\n"
        for i, (metrics, explanation) in enumerate(zip(previous_metrics, previous_explanations), 1):
            if metrics:
                geom = metrics.get('geometric_accuracy', 0)
                sicn = metrics.get('gmsh_sicn', {}).get('min', 0)
                if geom > 0:
                    history += f"\n{i}. Shape: {geom:.3f}, SICN: {sicn:.4f} - {explanation}"
                else:
                    history += f"\n{i}. SICN: {sicn:.4f} - {explanation}"
            else:
                history += f"\n{i}. FAILED - {explanation}"

        baseline_geom = mesh_data.get('geometric_accuracy', 0)
        baseline_sicn = mesh_data.get('gmsh_sicn', {}).get('min', 0)
        baseline_str = f"Shape Accuracy: {baseline_geom:.3f}, SICN: {baseline_sicn:.4f}" if baseline_geom > 0 else f"SICN: {baseline_sicn:.4f}"

        return f"""Iteration {iteration}: Learn from previous attempts and try a NEW approach.

IMPORTANT: Prioritize GEOMETRIC ACCURACY (shape fidelity) over element quality (SICN).

Baseline mesh: {baseline_str}

{history}

Based on what worked and didn't work, generate a NEW strategy that:
1. Tries a DIFFERENT approach than previous iterations
2. Learns from failures (what algorithms/sizes didn't work?)
3. Makes meaningful changes (algorithmic improvements + size adjustments if needed)

Provide response in this format:

EXPLANATION:
[What you're changing based on learning from previous attempts - explain algorithmic changes AND size adjustments]

STRATEGY_CODE:
```python
[Modified strategy code]
```

CHANGES:
- target_elements: [number - adjust based on what you learned]
- max_size_mm: [number - adjust based on geometry needs]
- quality_preset: [Coarse/Medium/Fine/Ultra]
- curvature_adaptive: [true/false]

Be intelligent and adaptive:
- If previous iterations had mesh too coarse/fine, adjust sizing
- If algorithm choices failed, try different meshing methods
- Combine smart sizing WITH algorithmic improvements
- Don't just make meshes smaller as a crutch - be strategic!"""

    def _extract_strategy_code(self, ai_response: str) -> Optional[str]:
        """Extract Python strategy code from AI response"""
        # Look for code blocks
        patterns = [
            r'STRATEGY_CODE:\s*```python\s*(.*?)```',
            r'```python\s*(.*?)```',
            r'STRATEGY_CODE:\s*```\s*(.*?)```',
        ]

        for pattern in patterns:
            match = re.search(pattern, ai_response, re.DOTALL | re.IGNORECASE)
            if match:
                code = match.group(1).strip()
                if len(code) > 100:  # Sanity check - should be substantial code
                    return code

        return None

    def auto_experiment_with_ai(self,
                                cad_file: str,
                                current_mesh_data: Dict,
                                current_config: Dict,
                                claude_assistant) -> Tuple[str, Optional[MeshIterationExperiment], Dict]:
        """
        Single iteration experiment (kept for compatibility)

        For multi-iteration experiments with learning, use multi_iteration_ai_experiment()
        """
        # Redirect to multi-iteration with just 1 iteration
        explanations, experiments, all_metrics = self.multi_iteration_ai_experiment(
            cad_file, current_mesh_data, current_config, claude_assistant, num_iterations=5
        )

        if experiments and all_metrics:
            # Return best result
            best_idx = 0
            best_sicn = -999
            for i, metrics in enumerate(all_metrics):
                if metrics:
                    sicn = metrics.get('gmsh_sicn', {}).get('min', -999)
                    if sicn > best_sicn:
                        best_sicn = sicn
                        best_idx = i

            # Create summary explanation
            summary = f"Ran {len(experiments)} iterations. Best result from iteration {best_idx+1}:\n{explanations[best_idx]}"
            return (summary, experiments[best_idx], all_metrics[best_idx])

        return ("All experiments failed", None, {})

    def _extract_explanation(self, ai_response: str) -> str:
        """Extract explanation text from AI response"""
        # Look for EXPLANATION: section - capture until we hit STRATEGY_CODE or CHANGES
        match = re.search(r'EXPLANATION:\s*\n(.*?)(?:\n(?:STRATEGY_CODE|CHANGES):|```python|\Z)',
                         ai_response, re.DOTALL | re.IGNORECASE)
        if match:
            explanation = match.group(1).strip()
            # Clean up - remove markdown, extra whitespace, but preserve line breaks
            explanation = re.sub(r'\*\*|\*|`', '', explanation)
            # Remove excessive newlines (more than 2 in a row)
            explanation = re.sub(r'\n{3,}', '\n\n', explanation)
            return explanation

        # Fallback: use everything before code blocks
        code_start = re.search(r'```|STRATEGY_CODE|CHANGES:', ai_response, re.IGNORECASE)
        if code_start:
            text_before_code = ai_response[:code_start.start()].strip()
            if text_before_code:
                # Clean and return
                text_before_code = re.sub(r'\*\*|\*|`', '', text_before_code)
                text_before_code = re.sub(r'\n{3,}', '\n\n', text_before_code)
                return text_before_code

        # Last resort: take first substantial paragraph
        lines = ai_response.split('\n')
        for i, line in enumerate(lines):
            if line.strip() and not line.strip().startswith(('#', '-', 'CHANGES')):
                end_idx = min(i + 5, len(lines))
                explanation = '\n'.join(lines[i:end_idx]).strip()
                return re.sub(r'\*\*|\*|`', '', explanation)

        return "AI suggested improvements to mesh quality parameters."

    def _extract_parameter_changes(self, ai_response: str) -> Dict:
        """Extract parameter changes from AI response"""
        changes = {}

        # Look for CHANGES: section (more flexible matching)
        match = re.search(r'CHANGES:?\s*\n(.*?)(?:\n\n|STRATEGY_CODE|```|\Z)', ai_response, re.DOTALL | re.IGNORECASE)
        if not match:
            # Try to find parameter patterns anywhere in response
            text = ai_response
        else:
            text = match.group(1)

        # Extract parameters with more flexible patterns (handle dash prefix, colons, equals, etc.)
        # target_elements
        match = re.search(r'[-*]?\s*target_elements[:\s=]+(\d+)', text, re.IGNORECASE)
        if match:
            changes['target_elements'] = int(match.group(1))

        # max_size_mm
        match = re.search(r'[-*]?\s*max_size_mm[:\s=]+([\d.]+)', text, re.IGNORECASE)
        if match:
            changes['max_size_mm'] = float(match.group(1))

        # quality_preset
        match = re.search(r'[-*]?\s*quality_preset[:\s=]+(Coarse|Medium|Fine|Ultra|Very\s*Fine)', text, re.IGNORECASE)
        if match:
            preset = match.group(1).strip()
            # Normalize preset names
            if 'very' in preset.lower() and 'fine' in preset.lower():
                changes['quality_preset'] = 'Very Fine'
            else:
                changes['quality_preset'] = preset.capitalize()

        # curvature_adaptive
        match = re.search(r'[-*]?\s*curvature_adaptive[:\s=]+(true|false|yes|no|on|off)', text, re.IGNORECASE)
        if match:
            val = match.group(1).lower()
            changes['curvature_adaptive'] = val in ['true', 'yes', 'on']

        # Also look for comments like "increase to 500" or "set to 20mm"
        # These are often in the EXPLANATION section
        if 'target_elements' not in changes:
            match = re.search(r'(?:increase|decrease|set|use|try)\s+(?:target_)?elements?\s+(?:to|:)?\s*(\d+)', text, re.IGNORECASE)
            if match:
                changes['target_elements'] = int(match.group(1))

        if 'max_size_mm' not in changes:
            match = re.search(r'(?:increase|decrease|set|use|try)\s+(?:max_)?size\s+(?:to|:)?\s*([\d.]+)\s*mm', text, re.IGNORECASE)
            if match:
                changes['max_size_mm'] = float(match.group(1))

        print(f"\n[DEBUG] Extracted parameter changes: {changes}")
        return changes


if __name__ == "__main__":
    # Test mode
    print("Experimental Mesh Iterator - Test Mode")
    print()

    iterator = ExperimentalMeshIterator()

    # Check for test CAD file
    test_cad = Path("cad_files/Cube.step")
    if not test_cad.exists():
        print(f"[X] Test CAD file not found: {test_cad}")
        print("  Run this from the MeshTest directory")
        sys.exit(1)

    # Generate baseline
    print("Generating fast baseline...")
    exp1, metrics1 = iterator.generate_fast_baseline(str(test_cad))

    print()
    print("Comparison:")
    print(iterator.compare_experiments())
