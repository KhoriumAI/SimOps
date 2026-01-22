use serde::{Deserialize, Serialize};
use std::process::Command;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SystemHealth {
    pub docker_installed: bool,
    pub docker_running: bool,
    pub wsl2_available: bool,
    pub disk_space_gb: f64,
    pub recommended_action: Option<String>,
}

/// Check if Docker is installed by running `docker --version`
fn check_docker_installed() -> bool {
    Command::new("docker")
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

/// Check if Docker daemon is running by running `docker ps`
fn check_docker_running() -> bool {
    Command::new("docker")
        .arg("ps")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

/// Check if WSL2 is available (Windows only)
#[cfg(target_os = "windows")]
fn check_wsl2() -> bool {
    Command::new("wsl")
        .arg("--status")
        .output()
        .map(|output| {
            output.status.success() && 
            String::from_utf8_lossy(&output.stdout).contains("WSL")
        })
        .unwrap_or(false)
}

#[cfg(not(target_os = "windows"))]
fn check_wsl2() -> bool {
    false // WSL2 is Windows-only
}

/// Get available disk space in GB
fn get_disk_space() -> f64 {
    #[cfg(target_os = "windows")]
    {
        // Check C: drive space
        use std::fs;
        if let Ok(metadata) = fs::metadata("C:\\") {
            // This is a rough estimate - on Windows we'd ideally use WinAPI
            // For now, return a placeholder value
            return 100.0;
        }
    }
    
    #[cfg(target_os = "macos")]
    {
        // Check root filesystem
        if let Ok(output) = Command::new("df")
            .arg("-g")
            .arg("/")
            .output()
        {
            let s = String::from_utf8_lossy(&output.stdout);
            // Parse df output to get available space
            if let Some(line) = s.lines().nth(1) {
                if let Some(available) = line.split_whitespace().nth(3) {
                    return available.parse::<f64>().unwrap_or(0.0);
                }
            }
        }
    }
    
    #[cfg(target_os = "linux")]
    {
        // Check root filesystem
        if let Ok(output) = Command::new("df")
            .arg("-BG")
            .arg("/")
            .output()
        {
            let s = String::from_utf8_lossy(&output.stdout);
            // Parse df output
            if let Some(line) = s.lines().nth(1) {
                if let Some(available) = line.split_whitespace().nth(3) {
                    let clean = available.trim_end_matches('G');
                    return clean.parse::<f64>().unwrap_or(0.0);
                }
            }
        }
    }
    
    0.0
}

/// Get OS-specific Docker installation instructions
fn get_docker_install_instructions() -> String {
    #[cfg(target_os = "windows")]
    {
        "Docker is not installed. Please install Docker Desktop for Windows.\n\
         1. Download from: https://www.docker.com/products/docker-desktop\n\
         2. Ensure WSL2 is enabled\n\
         3. Restart your computer after installation".to_string()
    }
    
    #[cfg(target_os = "macos")]
    {
        "Docker is not installed. Please install Docker Desktop for Mac.\n\
         Download from: https://www.docker.com/products/docker-desktop".to_string()
    }
    
    #[cfg(target_os = "linux")]
    {
        "Docker is not installed. Please install Docker:\n\
         Ubuntu/Debian: sudo apt-get install docker.io\n\
         Fedora: sudo dnf install docker\n\
         Or download Docker Desktop: https://www.docker.com/products/docker-desktop".to_string()
    }
    
    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    {
        "Docker is not installed. Please visit https://www.docker.com/products/docker-desktop".to_string()
    }
}

/// Tauri command to check system health
#[tauri::command]
pub fn check_system_health() -> SystemHealth {
    let docker_installed = check_docker_installed();
    let docker_running = if docker_installed { check_docker_running() } else { false };
    let wsl2_available = check_wsl2();
    let disk_space_gb = get_disk_space();
    
    let recommended_action = if !docker_installed {
        Some(get_docker_install_instructions())
    } else if !docker_running {
        Some("Docker is installed but not running. Please start Docker Desktop and try again.".to_string())
    } else if disk_space_gb < 10.0 {
        Some(format!("Low disk space: {:.1}GB available. Simulations require at least 10GB free space.", disk_space_gb))
    } else {
        None
    };

    SystemHealth {
        docker_installed,
        docker_running,
        wsl2_available,
        disk_space_gb,
        recommended_action,
    }
}
