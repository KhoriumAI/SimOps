// Prevents additional console window on Windows in release, do not remove!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod system_check;
mod backend_manager;
mod updater;

use system_check::check_system_health;
use backend_manager::BackendManager;
use tauri::Manager;

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! Welcome to SimOps!", name)
}

fn main() {
    let backend_manager = BackendManager::new();

    tauri::Builder::default()
        .manage(backend_manager)
        .invoke_handler(tauri::generate_handler![
            greet,
            updater::check_for_updates,
            updater::install_update,
            check_system_health,
            backend_manager::start_backend,
            backend_manager::check_backend_health
        ])
        .setup(|app| {
            // Auto-start backend on app launch
            let backend = app.state::<BackendManager>();
            
            // Resolve sidecar path - bundled apps use base name, dev uses target triple
            let current_exe = std::env::current_exe().unwrap();
            let exe_dir = current_exe.parent().unwrap();
            
            // Bundled apps: sidecar is named without target triple
            let bundled_sidecar = exe_dir.join("api_server.exe");
            // Dev mode: sidecar has target triple suffix
            let target_triple = "x86_64-pc-windows-msvc";
            let dev_sidecar = exe_dir.join(format!("api_server-{}.exe", target_triple));
            // Fallback: binaries directory during dev
            let fallback_sidecar = std::path::PathBuf::from(format!("binaries/api_server-{}.exe", target_triple));
            
            let final_path = if bundled_sidecar.exists() {
                bundled_sidecar
            } else if dev_sidecar.exists() {
                dev_sidecar
            } else {
                fallback_sidecar
            };

            if let Err(e) = backend.start_backend(final_path.to_string_lossy().to_string()) {
                eprintln!("Warning: Could not auto-start backend: {}", e);
            }

            #[cfg(debug_assertions)]
            {
                let window = app.get_window("main").unwrap();
                window.open_devtools();
            }
            
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while running tauri application")
        .run(|app_handle, event| {
            if let tauri::RunEvent::ExitRequested { .. } = event {
                // Stop backend on app exit
                let backend = app_handle.state::<BackendManager>();
                let _ = backend.stop_backend();
            }
        });
}
