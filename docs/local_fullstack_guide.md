# Local Full-Stack Development Guide

This guide explains how to run the entire application (Flask Backend + React Frontend) locally on your machine. This allows you to test changes rapidly without deploying to the shared development environment.

## 1. Prerequisites

Ensure you have the following installed:
- **Python 3.11+**
- **Node.js 18+**
- **PowerShell** (Standard on Windows)

## 2. One-Time Setup

We have provided a setup script to install all necessary dependencies for both the backend and frontend.

1. Open PowerShell in the root of the `MeshPackageLean` repository.
2. Run the setup script:
   ```powershell
   scripts\setup_local_dev.ps1
   ```
   
   This script will:
   - Create `backend/.env` from `.env.example` (if it doesn't exist).
   - Install Python requirements (`pip install -r requirements.txt`).
   - Install Node.js modules (`npm install` in `web-frontend`).

3. **Configure Secrets (Optional)**
   - Open `backend/.env` in your editor.
   - If you want to use **Modal** for meshing, ensure `USE_MODAL_COMPUTE=true` and your Modal tokens are set up on your machine.
   - If you want to use **AWS S3**, ensure `USE_S3=true` and your AWS credentials are valid.
   - *Default Behavior:* By default, the app uses local storage and local GMSH processing if these are not configured.

## 3. Running the App

To start the application, use the launcher script:

```powershell
scripts\run_local_stack.ps1
```

This will open two new command windows:
1. **Flask Backend**: Running on `http://localhost:5000`
2. **Vite Frontend**: Running on `http://localhost:3000`

### 3. Verify Local Stack
1. Open your browser to [http://localhost:3000](http://localhost:3000)
   - You should see the MeshGen interface
   - The top-right should show "MeshGen"
2. Check the Backend Console
   - You should see `[API] GET /api/strategies` when the frontend loads
3. **Run on Modal (Local Dev Only)**
   - In the "Mesh Settings" panel, you will see a **"☁️ Run on Modal"** checkbox.
   - This toggle is **ONLY visible in local development** (`npm run dev`).
   - Checking this will force the job to run on Modal's cloud infrastructure instead of your local machine, allowing you to test cloud integration from localhost.
4. **Force Kill / Cleanup**
   - If the backend is restarted while a job is running, it will automatically clean up "stuck" jobs on the next startup (marking them as failed).
   - A **"☠️ Force Kill Job (Dev)"** button is also available in the Mesh Settings panel to manually reset a stuck job's status.

## 4. Using the App

- Open your browser to **[http://localhost:3000](http://localhost:3000)**.
- You can now upload STEP files, generate meshes, and view them entirely on your local machine.
- Changes to the code in `backend/` or `web-frontend/` will typically trigger hot-reloads (auto-restarts) in their respective windows.

## 5. Mocking / Testing Without Cloud

If you do not have AWS/Modal credentials or want to test completely offline:

1. **Edit `backend/.env`**:
   ```ini
   FLASK_ENV=development
   USE_S3=false
   USE_MODAL_COMPUTE=false
   COMPUTE_BACKEND=local
   ```
   
2. **Use Local GMSH**:
   - Ensure `gmsh` is installed/available in your python environment (installed via requirements.txt).
   - The app will now run meshing processes on your local CPU.

## Troubleshooting

- **"ModuleNotFoundError"**: Re-run `scripts\setup_local_dev.ps1`.
- **Port In Use**: Ensure no other instances of python or node are running on ports 5000 or 3000.
- **Frontend can't connect**: Check the `web-frontend/vite.config.js` proxy settings if you changed backend ports.
