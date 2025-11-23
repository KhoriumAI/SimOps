"""
AI Chatbox Widget for Mesh Iteration
=====================================

Interactive chat interface with GenMesh AI for mesh quality improvement.

Features:
- Toggleable sidebar
- Message history display
- User input field
- Real-time AI responses
- Mesh context awareness
- Experiment suggestions
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Callable

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit,
    QPushButton, QLabel, QFrame, QScrollArea, QSizePolicy, QProgressBar, QCheckBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QTextCursor, QColor, QPalette

sys.path.insert(0, str(Path(__file__).parent.parent))


class AIResponseThread(QThread):
    """Background thread for AI API calls to prevent GUI freezing"""
    response_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, ai_assistant, message: str, mesh_data: Optional[Dict] = None):
        super().__init__()
        self.ai_assistant = ai_assistant
        self.message = message
        self.mesh_data = mesh_data

    def run(self):
        """Execute API call in background"""
        try:
            response = self.ai_assistant.chat(self.message, self.mesh_data)
            self.response_ready.emit(response)
        except Exception as e:
            self.error_occurred.emit(str(e))


class AutoExperimentThread(QThread):
    """Background thread for automatic AI experiments with multiple iterations"""
    progress_update = pyqtSignal(str)  # Progress messages
    iteration_complete = pyqtSignal(int, str, dict, str)  # (iteration_num, explanation, metrics, mesh_path)
    all_complete = pyqtSignal(list, list, list)  # (explanations, experiments, all_metrics)
    error_occurred = pyqtSignal(str)

    def __init__(self, cad_file: str, mesh_data: Dict, config: Dict, ai_assistant, num_iterations: int = 5):
        super().__init__()
        self.cad_file = cad_file
        self.mesh_data = mesh_data
        self.config = config
        self.ai_assistant = ai_assistant
        self.num_iterations = num_iterations

    def run(self):
        """Execute multi-iteration auto experiment in background"""
        try:
            from core.mesh_iterator import ExperimentalMeshIterator

            self.progress_update.emit(f"Starting {self.num_iterations}-iteration experiment with AI code generation...")
            iterator = ExperimentalMeshIterator()

            # Custom version with progress callbacks
            explanations = []
            experiments = []
            all_metrics = []

            # Use geometric accuracy as primary metric (shape fidelity)
            baseline_geom = self.mesh_data.get('geometric_accuracy', 0)
            baseline_sicn = self.mesh_data.get('gmsh_sicn', {}).get('min', 0)

            if baseline_geom > 0:
                self.progress_update.emit(f"Baseline: Shape Accuracy = {baseline_geom:.3f} (SICN = {baseline_sicn:.4f})")
            else:
                self.progress_update.emit(f"Baseline: SICN = {baseline_sicn:.4f}")

            # Load strategy code
            self.progress_update.emit("Loading existing strategy examples...")
            strategy_code = iterator._load_strategy_examples()
            self.progress_update.emit(f"Loaded adaptive_strategy.py ({len(strategy_code)} chars)")

            for iteration in range(1, self.num_iterations + 1):
                self.progress_update.emit(f"\n{'='*50}\nITERATION {iteration}/{self.num_iterations}\n{'='*50}")

                # Build prompt
                self.progress_update.emit("Building prompt for AI...")
                if iteration == 1:
                    prompt = iterator._build_initial_prompt(self.mesh_data, self.config, strategy_code)
                else:
                    prompt = iterator._build_iteration_prompt(
                        self.mesh_data, self.config, strategy_code,
                        all_metrics, explanations, iteration
                    )
                self.progress_update.emit(f"Prompt ready ({len(prompt)} chars)")

                self.progress_update.emit(f"Sending to AI Model (this may take 30-60s)...")

                # Get AI response
                response = self.ai_assistant.chat(prompt, self.mesh_data)

                self.progress_update.emit(f"Received AI response ({len(response)} chars)")

                # Parse
                self.progress_update.emit("Parsing AI response...")
                explanation = iterator._extract_explanation(response)
                strategy_mods = iterator._extract_strategy_code(response)
                param_changes = iterator._extract_parameter_changes(response)

                if strategy_mods:
                    num_lines = len(strategy_mods.split('\n'))
                    self.progress_update.emit(f"Extracted {num_lines} lines of strategy code")

                    # Check for potentially problematic patterns
                    if strategy_mods.count('for ') > 5:
                        self.progress_update.emit(f"[!]️ Strategy has {strategy_mods.count('for ')} loops - may be complex")
                    if 'while' in strategy_mods.lower():
                        self.progress_update.emit(f"[!]️ WARNING: Strategy contains 'while' loop - could be slow or infinite!")
                else:
                    self.progress_update.emit("[!] No strategy code found, using parameter changes only")

                # Show full explanation (not truncated)
                self.progress_update.emit(f"Strategy:")
                self.progress_update.emit(f"   {explanation}")

                # Create experiment
                self.progress_update.emit("Creating experiment folder...")
                new_config = self.config.copy()
                if param_changes:
                    new_config.update(param_changes)
                    # Show what parameters AI is using
                    self.progress_update.emit(f"🔧 AI Parameters:")
                    for key, value in param_changes.items():
                        self.progress_update.emit(f"   * {key}: {value}")

                    # Sanity check for problematic values
                    target = new_config.get('target_elements', 0)
                    max_size = new_config.get('max_size_mm', 0)

                    if target > 100000:
                        self.progress_update.emit(f"[!]️ WARNING: target_elements={target:,} is very high - this may take a long time!")
                    if max_size < 1.0:
                        self.progress_update.emit(f"[!]️ WARNING: max_size_mm={max_size} is very small - this may take a long time!")
                else:
                    self.progress_update.emit("[!] No parameter changes specified")

                description = f"AI Iter {iteration}: {explanation[:80]}"
                experiment = iterator.create_experiment(description, self.cad_file, new_config, strategy_mods)
                self.progress_update.emit(f"Created {experiment.folder.name}")

                # Save AI explanation
                self.progress_update.emit("Saving AI explanation...")
                with open(experiment.folder / "ai_explanation.txt", 'w') as f:
                    f.write(f"Iteration {iteration}\n{'='*50}\n\n")
                    f.write(explanation + "\n\n")
                    f.write(f"Full AI Response:\n{'='*50}\n{response}")
                self.progress_update.emit("Saved ai_explanation.txt")

                # Copy CAD file
                self.progress_update.emit(f"Copying CAD file to experiment folder...")

                self.progress_update.emit(f"⚙️ Starting mesh generation subprocess...")
                self.progress_update.emit(f"   This may take 1-3 minutes...")

                # Run with progress
                metrics = self._run_experiment_with_progress(iterator, experiment)

                if metrics:
                    # Prioritize geometric accuracy over SICN
                    new_geom = metrics.get('geometric_accuracy', 0)
                    new_sicn = metrics.get('gmsh_sicn', {}).get('min', 0)

                    if new_geom > 0 and baseline_geom > 0:
                        improvement = ((new_geom - baseline_geom) / baseline_geom * 100) if baseline_geom > 0 else 0
                        result_msg = f"[OK] Iteration {iteration}: Shape Accuracy = {new_geom:.3f} ({improvement:+.1f}%), SICN = {new_sicn:.4f}"
                    else:
                        improvement = ((new_sicn - baseline_sicn) / baseline_sicn * 100) if baseline_sicn > 0 else 0
                        result_msg = f"[OK] Iteration {iteration}: SICN = {new_sicn:.4f} ({improvement:+.1f}%)"

                    self.progress_update.emit(result_msg)
                    self.progress_update.emit(f"Mesh file saved ({metrics.get('total_elements', 0):,} elements)")

                    explanations.append(explanation)
                    experiments.append(experiment)
                    all_metrics.append(metrics)

                    # Send individual iteration result with mesh path
                    mesh_path = str(experiment.mesh_file) if hasattr(experiment, 'mesh_file') else ""
                    self.iteration_complete.emit(iteration, explanation, metrics, mesh_path)
                else:
                    self.progress_update.emit(f"Iteration {iteration} failed to generate mesh")
                    explanations.append(f"Failed: {explanation}")
                    experiments.append(experiment)
                    all_metrics.append({})

            # All iterations complete
            self.all_complete.emit(explanations, experiments, all_metrics)

        except Exception as e:
            import traceback
            self.error_occurred.emit(f"Experiment error: {str(e)}\n{traceback.format_exc()}")

    def _run_experiment_with_progress(self, iterator, experiment):
        """Run experiment with progress updates"""
        import subprocess
        import sys
        import json
        from pathlib import Path

        config = experiment._load_config()
        cad_file = config.get('cad_file')

        if not cad_file or not Path(cad_file).exists():
            # Try to find CAD file in experiment folder
            cad_files = list(experiment.folder.glob("*.step")) + list(experiment.folder.glob("*.stp"))
            if cad_files:
                cad_file = str(cad_files[0])
            else:
                self.progress_update.emit("CAD file not found")
                return {}

        # Use subprocess to run mesh generation
        worker_script = Path(__file__).parent.parent / "apps" / "cli" / "mesh_worker_subprocess.py"

        # Prepare quality params for worker
        quality_params = {
            'quality_preset': config.get('quality_preset', 'Medium'),
            'target_elements': config.get('target_elements', 10000),
            'max_size_mm': config.get('max_size_mm', 100),
            'curvature_adaptive': config.get('curvature_adaptive', False)
        }

        cmd = [
            sys.executable,
            str(worker_script),
            cad_file,
            '--quality-params',
            json.dumps(quality_params)
        ]

        self.progress_update.emit(f"Running: mesh_worker_subprocess.py")

        # Run mesh generation
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )

            # Read output line by line
            output_lines = []
            # Track which milestones we've already shown (show each only ONCE)
            shown_milestones = set()
            last_info_line = None  # Track last Gmsh "Info:" line

            for line in process.stdout:
                output_lines.append(line)

                # Show Gmsh Info lines (but throttle to every 100th line to avoid spam)
                if "Info    :" in line and line.strip():
                    # Store it but only emit periodically
                    last_info_line = line.strip()
                    if len(output_lines) % 100 == 0:  # Show every 100 lines
                        self.progress_update.emit(f"  🔄 {last_info_line[-60:]}")  # Last 60 chars

                # Show key progress indicators ONLY ONCE per milestone
                if ("Loading CAD" in line or "Importing" in line) and "loading_cad" not in shown_milestones:
                    self.progress_update.emit("  Loading CAD geometry...")
                    shown_milestones.add("loading_cad")
                elif ("Generating mesh" in line or "Meshing" in line) and "generating_mesh" not in shown_milestones:
                    self.progress_update.emit("  Generating mesh...")
                    shown_milestones.add("generating_mesh")
                elif ("Quality" in line or "SICN" in line) and "analyzing_quality" not in shown_milestones:
                    self.progress_update.emit("  Analyzing quality...")
                    shown_milestones.add("analyzing_quality")

            process.wait(timeout=300)

            # Parse result
            result_text = ''.join(output_lines)
            for line in result_text.split('\n'):
                if line.strip().startswith('{') and '"success"' in line:
                    result_data = json.loads(line)

                    if result_data.get('success'):
                        # Copy mesh file to experiment folder
                        source_mesh = result_data.get('output_file')
                        if source_mesh and Path(source_mesh).exists():
                            import shutil
                            shutil.copy(source_mesh, experiment.mesh_file)
                            self.progress_update.emit(f"  Copied mesh to experiment folder")

                        # Save results
                        metrics = result_data.get('metrics', {})
                        experiment.save_results(metrics)
                        self.progress_update.emit(f"  Saved results.json")

                        # Save log
                        with open(experiment.log_file, 'w') as f:
                            f.write(result_text)

                        return metrics

            self.progress_update.emit("  Mesh generation failed (no success message)")
            return {}

        except subprocess.TimeoutExpired:
            self.progress_update.emit("  Mesh generation timed out (>5 minutes)")
            return {}
        except Exception as e:
            self.progress_update.emit(f"  Error: {str(e)}")
            return {}


class ChatMessage(QFrame):
    """Single chat message widget"""

    def __init__(self, text: str, is_user: bool = False, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel)

        # Style based on sender
        if is_user:
            self.setStyleSheet("""
                QFrame {
                    background-color: #0d6efd;
                    border-radius: 10px;
                    padding: 8px 12px;
                    margin: 5px 50px 5px 10px;
                }
            """)
            text_color = "white"
        else:
            self.setStyleSheet("""
                QFrame {
                    background-color: #e9ecef;
                    border-radius: 10px;
                    padding: 8px 12px;
                    margin: 5px 10px 5px 50px;
                }
            """)
            text_color = "#212529"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Message label
        message_label = QLabel(text)
        message_label.setWordWrap(True)
        message_label.setStyleSheet(f"color: {text_color}; font-size: 12px; background: transparent; border: none;")
        message_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(message_label)


class ChatboxWidget(QFrame):
    """
    AI Chat interface for mesh iteration assistance

    Provides:
    - Chat with AI about mesh quality
    - Get improvement suggestions
    - Create and run experiments
    - Compare iteration results
    """

    # Signals
    experiment_requested = pyqtSignal(dict)  # Emitted when user wants to run experiment
    mesh_analysis_requested = pyqtSignal()   # Emitted when AI needs mesh data
    close_requested = pyqtSignal()           # Emitted when user wants to close chatbox

    def __init__(self, parent=None):
        super().__init__(parent)

        self.ai_assistant = None
        self.current_mesh_data = None
        self.current_cad_file = None
        self.current_config = None
        self.response_thread = None
        self.experiment_thread = None

        # Setup UI
        self.init_ui()

        # Try to initialize AI
        self.initialize_ai()

    def init_ui(self):
        """Initialize the chat interface"""
        self.setFrameStyle(QFrame.Box | QFrame.Raised)
        self.setLineWidth(2)
        self.setStyleSheet("""
            ChatboxWidget {
                background-color: #ffffff;
                border: 2px solid #dee2e6;
                border-radius: 8px;
            }
            QWidget {
                background-color: #ffffff;
                color: #212529;
            }
            QLabel {
                color: #212529;
                background-color: transparent;
            }
        """)
        self.setMinimumWidth(350)
        self.setMaximumWidth(450)
        self.setMaximumHeight(900)  # Prevent overflow on screen
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)  # Reduced margins
        layout.setSpacing(6)  # Reduced spacing

        # Header
        header = QHBoxLayout()

        title_label = QLabel("GenMesh AI")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #212529;")  # Explicit black text
        header.addWidget(title_label)

        header.addStretch()

        # Clear button
        clear_btn = QPushButton("Clear")
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 4px;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #5a6268; }
        """)
        clear_btn.clicked.connect(self.clear_chat)
        header.addWidget(clear_btn)

        # Close button
        close_btn = QPushButton("X")
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: none;
                padding: 5px 12px;
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #c82333; }
        """)
        close_btn.setToolTip("Close AI Chat")
        close_btn.clicked.connect(self.close_requested.emit)
        header.addWidget(close_btn)

        layout.addLayout(header)

        # Status label
        self.status_label = QLabel("Initializing...")
        self.status_label.setStyleSheet("color: #6c757d; font-size: 10px; padding: 5px;")
        layout.addWidget(self.status_label)

        # Message display area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: 1px solid #dee2e6;
                border-radius: 4px;
                background-color: #f8f9fa;
            }
        """)

        self.message_container = QWidget()
        self.message_container.setStyleSheet("background-color: #f8f9fa;")  # Explicit light background
        self.message_layout = QVBoxLayout(self.message_container)
        self.message_layout.setContentsMargins(3, 3, 3, 3)  # More compact
        self.message_layout.setSpacing(3)  # Tighter spacing
        self.message_layout.addStretch()

        self.scroll_area.setWidget(self.message_container)
        layout.addWidget(self.scroll_area, 1)

        # API key setup area (hidden by default)
        self.api_key_widget = self.create_api_key_widget()
        layout.addWidget(self.api_key_widget)
        self.api_key_widget.setVisible(False)

        # Quick actions
        actions_layout = QHBoxLayout()

        analyze_btn = QPushButton("Analyze")
        analyze_btn.setToolTip("Analyze current mesh quality")
        analyze_btn.setStyleSheet(self._button_style("#0d6efd"))
        analyze_btn.clicked.connect(self.analyze_mesh)
        actions_layout.addWidget(analyze_btn)

        suggest_btn = QPushButton("Suggest")
        suggest_btn.setToolTip("Get improvement suggestions")
        suggest_btn.setStyleSheet(self._button_style("#198754"))
        suggest_btn.clicked.connect(self.suggest_improvements)
        actions_layout.addWidget(suggest_btn)

        experiment_btn = QPushButton("Experiment")
        experiment_btn.setToolTip("Create test variation")
        experiment_btn.setStyleSheet(self._button_style("#ffc107"))
        experiment_btn.clicked.connect(self.create_experiment)
        actions_layout.addWidget(experiment_btn)

        # Copy log button
        copy_btn = QPushButton("Copy")
        copy_btn.setToolTip("Copy AI conversation log to clipboard")
        copy_btn.setFixedHeight(32)
        copy_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                padding: 4px;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #5c636a; }
        """)
        copy_btn.clicked.connect(self.copy_conversation_log)
        actions_layout.addWidget(copy_btn)

        layout.addLayout(actions_layout)

        # Extended thinking toggle
        thinking_layout = QHBoxLayout()
        self.thinking_checkbox = QCheckBox("Extended Thinking")
        self.thinking_checkbox.setToolTip(
            "Enable extended thinking for more accurate responses\n"
            "[!] Slower and more expensive, but better quality"
        )
        self.thinking_checkbox.setStyleSheet("""
            QCheckBox {
                font-size: 11px;
                color: #495057;
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                background-color: white;
                border: 2px solid #6c757d;
                border-radius: 3px;
            }
            QCheckBox::indicator:checked {
                background-color: #0d6efd;
                border: 2px solid #0d6efd;
            }
        """)
        self.thinking_checkbox.stateChanged.connect(self.on_thinking_toggled)
        thinking_layout.addWidget(self.thinking_checkbox)
        thinking_layout.addStretch()

        # Status indicator for thinking mode
        self.thinking_status = QLabel("Fast mode")
        self.thinking_status.setStyleSheet("font-size: 10px; color: #6c757d;")
        thinking_layout.addWidget(self.thinking_status)

        layout.addLayout(thinking_layout)

        # Input area
        input_layout = QHBoxLayout()

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Ask about mesh quality...")
        self.input_field.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                font-size: 12px;
            }
            QLineEdit:focus {
                border: 2px solid #0d6efd;
            }
        """)
        self.input_field.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.input_field)

        self.send_btn = QPushButton("Send")
        self.send_btn.setStyleSheet("""
            QPushButton {
                background-color: #0d6efd;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #0b5ed7; }
            QPushButton:disabled { background-color: #e9ecef; color: #adb5bd; }
        """)
        self.send_btn.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_btn)

        layout.addLayout(input_layout)

    def _button_style(self, color: str) -> str:
        """Generate button stylesheet"""
        return f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                padding: 6px 10px;
                border-radius: 4px;
                font-size: 10px;
            }}
            QPushButton:hover {{
                opacity: 0.9;
            }}
        """

    def create_api_key_widget(self) -> QFrame:
        """Create API key input widget"""
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: #fff3cd;
                border: 2px solid #ffc107;
                border-radius: 8px;
                padding: 10px;
            }
        """)

        layout = QVBoxLayout(frame)
        layout.setSpacing(5)  # More compact

        # Title
        title = QLabel("API Key Required")
        title.setStyleSheet("color: #856404; font-weight: bold; font-size: 13px; background: transparent; border: none;")
        layout.addWidget(title)

        # Instructions
        instructions = QLabel(
            "Enter your API key to enable AI assistance.\n"
            "Get your key from your AI provider"
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #856404; font-size: 11px; background: transparent; border: none;")
        layout.addWidget(instructions)

        # Input field
        input_layout = QHBoxLayout()
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("sk-ant-...")
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setStyleSheet("""
            QLineEdit {
                padding: 6px;
                border: 1px solid #856404;
                border-radius: 4px;
                background-color: white;
                color: #212529;
                font-size: 11px;
            }
        """)
        input_layout.addWidget(self.api_key_input)

        # Save button
        save_btn = QPushButton("Save")
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #ffc107;
                color: #212529;
                border: none;
                padding: 6px 15px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #ffca2c;
            }
        """)
        save_btn.clicked.connect(self.save_api_key)
        input_layout.addWidget(save_btn)

        layout.addLayout(input_layout)

        # Show/hide toggle
        show_key_btn = QPushButton("Show")
        show_key_btn.setCheckable(True)
        show_key_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #856404;
                border: 1px solid #856404;
                padding: 3px 8px;
                border-radius: 3px;
                font-size: 9px;
            }
            QPushButton:checked {
                background-color: #856404;
                color: white;
            }
        """)
        show_key_btn.clicked.connect(lambda checked:
            self.api_key_input.setEchoMode(QLineEdit.Normal if checked else QLineEdit.Password)
        )
        layout.addWidget(show_key_btn)

        return frame

    def initialize_ai(self):
        """Initialize AI API connection - always prompt for API key"""
        try:
            from core.claude_integration import ANTHROPIC_AVAILABLE

            if not ANTHROPIC_AVAILABLE:
                self.status_label.setText("[!] Install: pip install anthropic")
                self.status_label.setStyleSheet("color: #dc3545; font-size: 10px;")
                self.send_btn.setEnabled(False)
                self.add_system_message(
                    "AI Model not available. Install with:\n"
                    "pip install anthropic python-dotenv\n\n"
                    "Then restart the application"
                )
                return

            # Always prompt for API key (never load from environment for security)
            self.status_label.setText("[!] API key required")
            self.status_label.setStyleSheet("color: #ffc107; font-size: 10px;")
            self.send_btn.setEnabled(False)

            # Show API key input widget
            self.api_key_widget.setVisible(True)

            self.add_system_message(
                "🔐 API Key Required\n\n"
                "For security, your API key is never saved to disk.\n"
                "You'll need to enter it each time you launch the application.\n\n"
                "Get your API key from your AI provider"
            )

        except Exception as e:
            # Handle any initialization errors
            self.status_label.setText("Initialization failed")
            self.status_label.setStyleSheet("color: #dc3545; font-size: 10px;")
            self.send_btn.setEnabled(False)
            self.add_system_message(f"Error initializing:\n{str(e)}")

    def save_api_key(self):
        """Save API key for this session only (not persisted to disk)"""
        api_key = self.api_key_input.text().strip()

        if not api_key:
            self.add_system_message("Please enter an API key")
            return

        if not api_key.startswith("sk-ant-"):
            self.add_system_message("[!] API key should start with 'sk-ant-'")
            return

        # Store API key in memory only (session-only, never saved to disk)
        try:
            self.add_system_message("API key accepted (session only - will not be saved to disk)")

            # Hide API key widget
            self.api_key_widget.setVisible(False)

            # Try to initialize AI with new key
            try:
                from core.claude_integration import ClaudeMeshAssistant
                self.ai_assistant = ClaudeMeshAssistant(api_key=api_key)
                self.status_label.setText("AI connected")
                self.status_label.setStyleSheet("color: #198754; font-size: 10px;")
                self.send_btn.setEnabled(True)

                # Welcome message
                self.add_message(
                    "Hello! I'm GenMesh AI, your mesh quality assistant. I can help you:\n\n"
                    "* Analyze mesh quality\n"
                    "* Suggest improvements\n"
                    "* Create experimental variations\n"
                    "* Iterate towards better results\n\n"
                    "What would you like to do?",
                    is_user=False
                )

                # Clear the API key input field for security
                self.api_key_input.clear()

            except Exception as e:
                self.add_system_message(f"Failed to connect: {str(e)}\nPlease check your API key.")
                self.api_key_widget.setVisible(True)

        except Exception as e:
            self.add_system_message(f"Error: {str(e)}")

    def add_message(self, text: str, is_user: bool = False):
        """Add message to chat display"""
        # Remove stretch before adding message
        if self.message_layout.count() > 0:
            stretch_item = self.message_layout.takeAt(self.message_layout.count() - 1)
            if stretch_item:
                stretch_item.widget()

        # Add message
        message = ChatMessage(text, is_user)
        self.message_layout.addWidget(message)

        # Re-add stretch at end
        self.message_layout.addStretch()

        # Auto-scroll to bottom
        QTimer.singleShot(100, self._scroll_to_bottom)

    def add_system_message(self, text: str):
        """Add system/info message"""
        # Determine style based on content
        if text.startswith("[OK]") or text.startswith("  [OK]"):
            # Grey checkmark style for progress
            label = QLabel(text)
            label.setWordWrap(True)
            label.setStyleSheet("""
                QLabel {
                    background-color: #f8f9fa;
                    color: #6c757d;
                    padding: 4px 10px;
                    border-radius: 3px;
                    border-left: 3px solid #6c757d;
                    font-size: 10px;
                    font-family: monospace;
                }
            """)
        elif text.startswith("[X]") or text.startswith("[ERROR]") or text.startswith("  [X]"):
            # Red for errors
            label = QLabel(text)
            label.setWordWrap(True)
            label.setStyleSheet("""
                QLabel {
                    background-color: #f8d7da;
                    color: #842029;
                    padding: 4px 10px;
                    border-radius: 3px;
                    border-left: 3px solid #dc3545;
                    font-size: 10px;
                    font-family: monospace;
                }
            """)
        elif text.startswith("[AI]") or text.startswith("⚙️"):
            # Blue for active operations
            label = QLabel(text)
            label.setWordWrap(True)
            label.setStyleSheet("""
                QLabel {
                    background-color: #cfe2ff;
                    color: #084298;
                    padding: 6px 10px;
                    border-radius: 3px;
                    border-left: 3px solid #0d6efd;
                    font-size: 11px;
                }
            """)
        else:
            # Yellow for general info
            label = QLabel(text)
            label.setWordWrap(True)
            label.setStyleSheet("""
                QLabel {
                    background-color: #fff3cd;
                    color: #856404;
                    padding: 10px;
                    border-radius: 4px;
                    border-left: 4px solid #ffc107;
                    font-size: 11px;
                }
            """)

        label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        if self.message_layout.count() > 0:
            stretch_item = self.message_layout.takeAt(self.message_layout.count() - 1)

        self.message_layout.addWidget(label)
        self.message_layout.addStretch()

        QTimer.singleShot(100, self._scroll_to_bottom)

    def _scroll_to_bottom(self):
        """Scroll message area to bottom - only if user is already at bottom"""
        if not hasattr(self, 'scroll_area'):
            return

        scrollbar = self.scroll_area.verticalScrollBar()

        # Check if user is already near the bottom (within 30px)
        current_value = scrollbar.value()
        max_value = scrollbar.maximum()
        is_at_bottom = (max_value - current_value) <= 30

        # Only auto-scroll if user is already at bottom
        if is_at_bottom:
            scrollbar.setValue(scrollbar.maximum())

    def send_message(self):
        """Send user message to AI"""
        if not self.ai_assistant:
            self.add_system_message("AI not connected. Check configuration.")
            return

        text = self.input_field.text().strip()
        if not text:
            return

        # Add user message
        self.add_message(text, is_user=True)
        self.input_field.clear()

        # Disable input while processing
        self.input_field.setEnabled(False)
        self.send_btn.setEnabled(False)
        self.send_btn.setText("Thinking...")

        # Call AI in background thread
        self.response_thread = AIResponseThread(
            self.ai_assistant,
            text,
            self.current_mesh_data
        )
        self.response_thread.response_ready.connect(self._handle_response)
        self.response_thread.error_occurred.connect(self._handle_error)
        self.response_thread.start()

    def _handle_response(self, response: str):
        """Handle AI response"""
        self.add_message(response, is_user=False)

        # Re-enable input
        self.input_field.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.send_btn.setText("Send")
        self.input_field.setFocus()

    def _handle_error(self, error: str):
        """Handle error from AI API"""
        # Check for specific error types
        if "529" in error or "overloaded" in error.lower():
            self.add_system_message(
                "[!]️ AI Model is currently overloaded (high demand).\n\n"
                "This is temporary - please try again in a moment.\n"
                "The message will be automatically retried."
            )
        elif "401" in error or "authentication" in error.lower():
            self.add_system_message(
                "Authentication failed.\n\n"
                "Please check your API key in the .env file."
            )
        else:
            self.add_system_message(f"Error: {error}\n\nPlease check your API key and connection.")

        self.input_field.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.send_btn.setText("Send")

    def update_mesh_data(self, mesh_data: Dict, cad_file: Optional[str] = None, config: Optional[Dict] = None):
        """Update current mesh context"""
        self.current_mesh_data = mesh_data
        if cad_file:
            self.current_cad_file = cad_file
        if config:
            self.current_config = config
        elements = mesh_data.get('total_elements', 'N/A')
        self.status_label.setText(f"Mesh loaded ({elements:,} elements)")

    def _on_experiment_progress(self, message: str):
        """Handle experiment progress updates"""
        self.add_system_message(message)

    def _on_iteration_complete(self, iteration: int, explanation: str, metrics: Dict, mesh_path: str):
        """Handle single iteration completion - auto-display mesh"""
        logging.info(f"_on_iteration_complete: iteration={iteration}, mesh_path={mesh_path}")

        # Find the main GUI window (traverse up the parent chain)
        widget = self
        main_gui = None
        while widget is not None:
            if hasattr(widget, 'on_ai_iteration_mesh_ready'):
                main_gui = widget
                break
            widget = widget.parent() if hasattr(widget, 'parent') else None

        if main_gui:
            logging.info(f"Found main GUI, calling on_ai_iteration_mesh_ready")
            main_gui.on_ai_iteration_mesh_ready(mesh_path, metrics)
        else:
            logging.error("Could not find main GUI with on_ai_iteration_mesh_ready method")

    def _on_all_experiments_complete(self, explanations: list, experiments: list, all_metrics: list):
        """Handle all iterations complete"""
        # Re-enable UI
        self.send_btn.setEnabled(True)
        self.input_field.setEnabled(True)

        # Build comprehensive results - prioritize geometric accuracy
        baseline_geom = self.current_mesh_data.get('geometric_accuracy', 0)
        baseline_sicn = self.current_mesh_data.get('gmsh_sicn', {}).get('min', 0)
        use_geom = baseline_geom > 0

        # Find best based on geometric accuracy (or SICN as fallback)
        best_idx = 0
        best_score = -999
        for i, metrics in enumerate(all_metrics):
            if metrics:
                if use_geom:
                    score = metrics.get('geometric_accuracy', -999)
                else:
                    score = metrics.get('gmsh_sicn', {}).get('min', -999)
                if score > best_score:
                    best_score = score
                    best_idx = i

        # Build results message
        results_text = f"🎯 ALL 5 ITERATIONS COMPLETE!\n\n"
        results_text += f"Results Summary:\n"
        if use_geom:
            results_text += f"Baseline: Shape Accuracy = {baseline_geom:.3f}, SICN = {baseline_sicn:.4f}\n\n"
        else:
            results_text += f"Baseline: SICN = {baseline_sicn:.4f}\n\n"

        for i, metrics in enumerate(all_metrics, 1):
            if metrics:
                geom = metrics.get('geometric_accuracy', 0)
                sicn = metrics.get('gmsh_sicn', {}).get('min', 0)
                marker = "🌟" if i == best_idx + 1 else "  "

                if use_geom and geom > 0:
                    improvement = ((geom - baseline_geom) / baseline_geom * 100) if baseline_geom > 0 else 0
                    results_text += f"{marker} Iter {i}: Shape {geom:.3f} ({improvement:+.1f}%), SICN {sicn:.4f}\n"
                else:
                    improvement = ((sicn - baseline_sicn) / baseline_sicn * 100) if baseline_sicn > 0 else 0
                    results_text += f"{marker} Iter {i}: SICN {sicn:.4f} ({improvement:+.1f}%)\n"
            else:
                results_text += f"  Iter {i}: FAILED\n"

        results_text += f"\n🏆 BEST RESULT: Iteration {best_idx + 1}\n"

        if all_metrics and all_metrics[best_idx]:
            best_metrics = all_metrics[best_idx]
            best_explanation = explanations[best_idx]

            results_text += f"\nWhat worked:\n{best_explanation}\n\n"
            results_text += f"📈 Final Quality:\n"

            if use_geom:
                best_geom = best_metrics.get('geometric_accuracy', 0)
                best_sicn_val = best_metrics.get('gmsh_sicn', {}).get('min', 0)
                geom_improvement = ((best_geom - baseline_geom) / baseline_geom * 100) if baseline_geom > 0 else 0
                results_text += f"   Shape Accuracy: {baseline_geom:.3f} -> {best_geom:.3f} ({geom_improvement:+.1f}%)\n"
                results_text += f"   SICN: {baseline_sicn:.4f} -> {best_sicn_val:.4f}\n"
                improvement = geom_improvement
            else:
                improvement = ((best_score - baseline_sicn) / baseline_sicn * 100) if baseline_sicn > 0 else 0
                results_text += f"   SICN: {baseline_sicn:.4f} -> {best_score:.4f}\n"
                results_text += f"   Improvement: {improvement:+.1f}%\n"
            results_text += f"   Elements: {best_metrics.get('total_elements', 0):,}\n\n"

            if improvement > 10:
                results_text += "🎉 Significant improvement! AI found better algorithms."
            elif improvement > 0:
                results_text += "[OK] Small improvement. Check experiments/ folder."
            else:
                results_text += "[!]️ No improvement yet. Try more iterations or different geometry."

            results_text += f"\n\n[FILE] Check experiments/ folder for generated code and details."
        else:
            results_text += "\nAll iterations failed. Check console for errors."

        self.add_message(results_text, is_user=False)

    def _on_experiment_error(self, error: str):
        """Handle experiment error"""
        self.send_btn.setEnabled(True)
        self.input_field.setEnabled(True)
        self.add_system_message(f"Experiment failed: {error}")

    def analyze_mesh(self):
        """Request mesh analysis from AI"""
        if not self.ai_assistant:
            self.add_system_message("AI not available")
            return

        if not self.current_mesh_data:
            self.add_system_message("No mesh loaded. Generate a mesh first.")
            return

        self.input_field.setText("Please analyze the current mesh quality and identify any issues.")
        self.send_message()

    def suggest_improvements(self):
        """Request improvement suggestions"""
        if not self.ai_assistant:
            self.add_system_message("AI not available")
            return

        if not self.current_mesh_data:
            self.add_system_message("No mesh loaded. Generate a mesh first.")
            return

        self.input_field.setText("What parameters should I change to improve mesh quality?")
        self.send_message()

    def on_thinking_toggled(self, state):
        """Handle thinking mode toggle"""
        enabled = state == Qt.Checked
        if self.ai_assistant:
            self.ai_assistant.set_thinking_mode(enabled)

        if enabled:
            self.thinking_status.setText("Thinking mode")
            self.thinking_status.setStyleSheet("font-size: 10px; color: #0d6efd; font-weight: bold;")
            self.add_system_message("Extended thinking enabled - responses will be more accurate but slower")
        else:
            self.thinking_status.setText("Fast mode")
            self.thinking_status.setStyleSheet("font-size: 10px; color: #6c757d;")
            self.add_system_message("Extended thinking disabled - responses will be faster")

    def copy_conversation_log(self):
        """Copy entire AI conversation to clipboard"""
        from PyQt5.QtWidgets import QApplication

        # Get all text from the chat display
        full_log = self.chat_display.toPlainText()

        if not full_log:
            self.add_system_message("No conversation to copy")
            return

        # Add metadata header
        log_text = "="*70 + "\n"
        log_text += "KHORIUM MESHGEN - AI CONVERSATION LOG\n"
        log_text += "="*70 + "\n\n"

        # Add mesh context if available
        if self.current_mesh_data:
            log_text += "MESH CONTEXT:\n"
            log_text += f"  File: {self.current_mesh_data.get('file_name', 'N/A')}\n"
            log_text += f"  Elements: {self.current_mesh_data.get('total_elements', 'N/A'):,}\n"
            log_text += f"  Nodes: {self.current_mesh_data.get('total_nodes', 'N/A'):,}\n"

            # Add quality metrics
            if 'geometric_accuracy' in self.current_mesh_data:
                log_text += f"  Shape Accuracy: {self.current_mesh_data['geometric_accuracy']:.3f}\n"
            if 'gmsh_sicn' in self.current_mesh_data:
                log_text += f"  SICN (min): {self.current_mesh_data['gmsh_sicn'].get('min', 'N/A'):.4f}\n"

            log_text += "\n"

        log_text += "CONVERSATION:\n"
        log_text += "="*70 + "\n"
        log_text += full_log
        log_text += "\n" + "="*70 + "\n"

        # Copy to clipboard
        clipboard = QApplication.clipboard()
        clipboard.setText(log_text)

        self.add_system_message("Conversation copied to clipboard!")

    def create_experiment(self):
        """Automatically create and run experimental variation"""
        if not self.ai_assistant:
            self.add_system_message("AI not available")
            return

        if not self.current_mesh_data:
            self.add_system_message("No mesh loaded. Generate a mesh first.")
            return

        if not self.current_cad_file or not self.current_config:
            self.add_system_message("Missing CAD file or config. Please generate a mesh first.")
            return

        # Show progress message
        self.add_message("Running 5-iteration AI experiment with code generation...", is_user=True)
        self.add_system_message("GenMesh AI will automatically:\n"
                               "1. Read existing meshing algorithms\n"
                               "2. Generate MODIFIED strategy code\n"
                               "3. Run 5 different approaches\n"
                               "4. Learn from each iteration\n"
                               "5. Show comparative results\n\n"
                               "This will take 5-10 minutes for 5 iterations...")

        # Disable buttons during experiment
        self.send_btn.setEnabled(False)
        self.input_field.setEnabled(False)

        # Run experiment in background
        self.experiment_thread = AutoExperimentThread(
            self.current_cad_file,
            self.current_mesh_data,
            self.current_config,
            self.ai_assistant,
            num_iterations=5
        )
        self.experiment_thread.progress_update.connect(self._on_experiment_progress)
        self.experiment_thread.iteration_complete.connect(self._on_iteration_complete)
        self.experiment_thread.all_complete.connect(self._on_all_experiments_complete)
        self.experiment_thread.error_occurred.connect(self._on_experiment_error)
        self.experiment_thread.start()

    def clear_chat(self):
        """Clear all messages"""
        # Remove all messages except system info
        while self.message_layout.count() > 1:
            item = self.message_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if self.ai_assistant:
            self.ai_assistant.clear_history()

        self.add_system_message("Chat history cleared")


if __name__ == "__main__":
    # Test mode
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    chatbox = ChatboxWidget()
    chatbox.show()

    # Test with sample mesh data
    test_data = {
        'total_elements': 10000,
        'total_nodes': 5000,
        'gmsh_sicn': {'min': 0.32, 'avg': 0.55, 'max': 0.89},
        'gmsh_gamma': {'min': 0.25, 'avg': 0.48, 'max': 0.75}
    }
    chatbox.update_mesh_data(test_data)

    sys.exit(app.exec_())
