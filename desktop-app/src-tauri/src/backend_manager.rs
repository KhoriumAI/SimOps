use std::process::{Command, Stdio, Child};
use std::sync::{Arc, Mutex};
use std::net::TcpStream;
use std::time::Duration;
use std::path::PathBuf;

/// Backend manager for SimOps Flask API server
pub struct BackendManager {
    process: Arc<Mutex<Option<Child>>>,
    port: u16,
}

impl BackendManager {
    pub fn new() -> Self {
        Self {
            process: Arc::new(Mutex::new(None)),
            port: 5001,
        }
    }

    /// Check if backend is running by testing port 5000
    fn is_backend_running(&self) -> bool {
        TcpStream::connect_timeout(
            &format!("127.0.0.1:{}", self.port).parse().unwrap(),
            Duration::from_millis(500)
        ).is_ok()
    }

    /// Start the Flask backend server
    /// executable_path: Absolute path to the sidecar executable
    pub fn start_backend(&self, executable_path: String) -> Result<String, String> {
        // Check if already running
        if self.is_backend_running() {
            return Ok("Backend already running".to_string());
        }

        let exe_path = std::path::PathBuf::from(&executable_path);
        if !exe_path.exists() {
            return Err(format!("Sidecar not found at: {}", executable_path));
        }

        println!("Starting backend sidecar from: {}", executable_path);
        
        // Log to users home to avoid permission issues
        let home = std::env::var("USERPROFILE").map(PathBuf::from)
            .or_else(|_| std::env::var("HOME").map(PathBuf::from))
            .unwrap_or_else(|_| PathBuf::from("."));
            
        let simops_dir = home.join(".simops");
        std::fs::create_dir_all(&simops_dir).map_err(|e| format!("Failed to create storage dir: {}", e))?;
        
        let log_path = simops_dir.join("backend.log");
        let log_file = std::fs::File::create(&log_path)
            .map_err(|e| format!("Failed to create log file {}: {}", log_path.display(), e))?;

        let child = Command::new(executable_path)
            .env("PORT", "5001")
            .stdout(Stdio::from(log_file.try_clone().unwrap()))
            .stderr(Stdio::from(log_file))
            .spawn()
            .map_err(|e| format!("Failed to spawn sidecar: {}", e))?;

        *self.process.lock().unwrap() = Some(child);

        Ok(format!("Backend sidecar started. Logs at: {}", log_path.display()))
    }

    /// Stop the backend server
    pub fn stop_backend(&self) -> Result<String, String> {
        let mut process_guard = self.process.lock().unwrap();
        if let Some(mut child) = process_guard.take() {
            println!("Killing backend sidecar...");
            let _ = child.kill();
            Ok("Backend stopped successfully".to_string())
        } else {
            Ok("Backend not running".to_string())
        }
    }

    /// Check backend health endpoint
    pub fn check_backend_health(&self) -> Result<String, String> {
        if self.is_backend_running() {
            Ok("Backend is running and accessible".to_string())
        } else {
            Err("Backend is not accessible on port 5000".to_string())
        }
    }
}

impl Default for BackendManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for BackendManager {
    fn drop(&mut self) {
        // Clean up backend process when manager is dropped
        let _ = self.stop_backend();
    }
}

#[tauri::command]
pub fn start_backend(_app: tauri::AppHandle, manager: tauri::State<BackendManager>) -> Result<String, String> {
    // Same logic as setup - try bundled name first, then dev name
    let current_exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let exe_dir = current_exe.parent().unwrap();
    
    // Bundled apps: sidecar is named without target triple
    let bundled_sidecar = exe_dir.join("api_server.exe");
    // Dev mode: sidecar has target triple suffix
    let target_triple = "x86_64-pc-windows-msvc";
    let dev_sidecar = exe_dir.join(format!("api_server-{}.exe", target_triple));
    
    let sidecar_path = if bundled_sidecar.exists() {
        bundled_sidecar
    } else {
        dev_sidecar
    };
    
    manager.start_backend(sidecar_path.to_string_lossy().to_string())
}

#[tauri::command]
pub fn check_backend_health(manager: tauri::State<BackendManager>) -> Result<String, String> {
    manager.check_backend_health()
}
