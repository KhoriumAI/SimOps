import subprocess
import time
import sys
import os
import signal
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict

# Configuration
BACKEND_PORT = 5001
FRONTEND_PORT = 1420

@dataclass
class ServiceConfig:
    name: str
    command: List[str]
    cwd: Path
    env: Dict[str, str] = field(default_factory=dict)
    ready_pattern: Optional[str] = None
    color: str = "\033[0m" # Default white

class ServiceManager:
    def __init__(self):
        self.processes: List[subprocess.Popen] = []
        self.running = True

    def log(self, service_name: str, message: str, color: str = "\033[0m"):
        timestamp = time.strftime("%H:%M:%S")
        print(f"{color}[{timestamp}] [{service_name}] {message}\033[0m")

    def _stream_output(self, process: subprocess.Popen, config: ServiceConfig):
        for line in iter(process.stdout.readline, ''):
            if not self.running:
                break
            if line:
                self.log(config.name, line.strip(), config.color)
                
    def start_service(self, config: ServiceConfig):
        self.log("ORCHESTRATOR", f"Starting {config.name}...", "\033[96m") # Cyan
        
        # Merge environment variables
        env = os.environ.copy()
        env.update(config.env)

        try:
            # Use shell=True for npm commands on Windows to resolve properly, 
            # but try to keep it minimal for Python
            # For scalability/security, direct execution is better but 'npm' is a batch file on Windows
            is_shell = config.command[0] == 'npm' and os.name == 'nt'
            
            process = subprocess.Popen(
                config.command,
                cwd=str(config.cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                bufsize=1,
                universal_newlines=True,
                shell=is_shell 
            )
            self.processes.append(process)

            # Start logger thread
            thread = threading.Thread(target=self._stream_output, args=(process, config))
            thread.daemon = True
            thread.start()
            
            return process
        except Exception as e:
            self.log("ERROR", f"Failed to start {config.name}: {e}", "\033[91m")
            return None

    def stop_all(self):
        self.running = False
        self.log("ORCHESTRATOR", "Shutting down all services...", "\033[96m")
        for p in self.processes:
            try:
                # Windows requires more forceful termination for process trees (like npm starting node)
                if os.name == 'nt':
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
                else:
                    p.terminate()
            except Exception:
                pass
        sys.exit(0)

def main():
    manager = ServiceManager()
    
    # Handle Ctrl+C
    def signal_handler(sig, frame):
        manager.stop_all()
    signal.signal(signal.SIGINT, signal_handler)

    # 1. Define Paths
    root_dir = Path(__file__).parent.parent
    backend_dir = root_dir / "simops-backend"
    config_dir = root_dir / "desktop-app" / "src-tauri" # Run tauri dev from here usually or root of desktop-app

    # 2. Define Services
    
    # Service: Backend (Flask)
    backend_service = ServiceConfig(
        name="BACKEND",
        command=[sys.executable, "api_server.py"],
        cwd=backend_dir,
        env={"PORT": str(BACKEND_PORT), "FLASK_ENV": "development"},
        color="\033[92m" # Green
    )

    # Service: Frontend (Tauri/Vite)
    # running "npm run tauri dev" handles both the Vite dev server and the Rust sidecar
    # We run it from desktop-app directory
    desktop_app_dir = root_dir / "desktop-app"
    frontend_service = ServiceConfig(
        name="DESKTOP",
        command=["npm", "run", "tauri", "dev"],
        cwd=desktop_app_dir,
        color="\033[94m" # Blue
    )

    # 3. Start Services
    manager.start_service(backend_service)
    
    # Give backend a moment to warm up (optional but effectively reduces race conditions logging)
    time.sleep(2) 
    
    manager.start_service(frontend_service)

    # Keep main thread alive
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
