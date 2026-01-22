use tauri::{AppHandle, Manager, Wry};
use tauri::updater::UpdateResponse;

// Define a struct to serialize update info for the frontend
#[derive(Clone, serde::Serialize)]
struct UpdateInfo {
    version: String,
    date: String,
    body: String,
}

#[tauri::command]
pub async fn check_for_updates(app: AppHandle<Wry>) -> Result<bool, String> {
    println!("Checking for updates...");
    
    // Check for updates
    match app.updater().check().await {
        Ok(update) => {
            if update.is_update_available() {
                println!("Update available: {}", update.latest_version());
                
                // Emit an event to the frontend
                app.emit_all("update-available", UpdateInfo {
                    version: update.latest_version().to_string(),
                    date: update.date().map(|d| d.to_string()).unwrap_or_default(),
                    body: update.body().map(|b| b.to_string()).unwrap_or_default(),
                }).map_err(|e| e.to_string())?;
                
                // For this task, we can also trigger the download/install immediately if configured
                // But usually we wait for user confirmation (via frontend calling another command)
                // For simplicity in this demo, let's just return true
                return Ok(true);
            } else {
                println!("No updates found.");
                return Ok(false);
            }
        }
        Err(e) => {
            println!("Error checking for updates: {}", e);
            return Err(e.to_string());
        }
    }
}

#[tauri::command]
pub async fn install_update(app: AppHandle<Wry>) -> Result<(), String> {
    println!("Installing update...");
    match app.updater().check().await {
        Ok(update) => {
            if update.is_update_available() {
                update.download_and_install().await.map_err(|e| e.to_string())?;
                println!("Update installed. Restarting...");
                app.restart();
            } else {
                return Err("No update available to install".to_string());
            }
        }
        Err(e) => return Err(e.to_string()),
    }
    Ok(())
}
