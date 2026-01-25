"""
REAL API Integration Tests
===========================

These tests hit the ACTUAL HTTP API endpoints (not mocks).
They require the backend server to be running.

Usage:
    # Start backend first
    python simops-backend/api_server.py

    # Then run tests
    pytest tests/integration/test_api_real.py -v
"""

import pytest
import requests
import time
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

API_BASE = "http://localhost:8000"
TIMEOUT = 120  # 2 minutes for simulation


@pytest.fixture(scope="session")
def backend_running():
    """Check if backend is running before tests"""
    try:
        response = requests.get(f"{API_BASE}/api/health", timeout=5)
        if response.status_code != 200:
            pytest.skip("Backend is not running. Start with: python simops-backend/api_server.py")
        return True
    except requests.exceptions.ConnectionError:
        pytest.skip("Backend is not running. Start with: python simops-backend/api_server.py")


@pytest.fixture
def test_mesh_file():
    """Return path to a test mesh file"""
    # Look for test meshes in order of preference
    candidates = [
        PROJECT_ROOT / "simops/templates/Golden_Thermal_Case/test_geom.msh",
        PROJECT_ROOT / "ExampleOF_Cases/Cube_medium_fast_tet.msh",
        PROJECT_ROOT / "cad_files/test.msh",
    ]

    for path in candidates:
        if path.exists():
            return path

    pytest.skip("No test mesh file found")


class TestAPIEndpoints:
    """Test basic API endpoints"""

    def test_health_endpoint(self, backend_running):
        """Test /api/health returns 200"""
        response = requests.get(f"{API_BASE}/api/health", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "service" in data

    def test_diagnostics_endpoint(self, backend_running):
        """Test /api/diagnostics returns system info"""
        response = requests.get(f"{API_BASE}/api/diagnostics", timeout=5)
        assert response.status_code == 200
        data = response.json()

        # Should contain diagnostic info
        assert "upload_folder" in data
        assert "upload_folder_exists" in data
        assert "gmsh_available" in data

        print(f"\nDiagnostics: {data}")


class TestFileUpload:
    """Test file upload functionality"""

    def test_upload_mesh_file(self, backend_running, test_mesh_file):
        """Test uploading a .msh file"""
        with open(test_mesh_file, 'rb') as f:
            files = {'files': (test_mesh_file.name, f, 'application/octet-stream')}
            response = requests.post(f"{API_BASE}/api/upload", files=files, timeout=30)

        assert response.status_code == 200
        data = response.json()

        assert "saved_as" in data
        assert "message" in data
        assert data["filename"] == test_mesh_file.name

        # Store for later tests
        return data["saved_as"]


class TestSimulationBuiltin:
    """Test simulation with builtin solver (no OpenFOAM required)"""

    def test_simulate_builtin_solver(self, backend_running, test_mesh_file):
        """
        Test simulation with builtin solver.
        This should work even without OpenFOAM/WSL installed.
        """
        # Step 1: Upload file
        with open(test_mesh_file, 'rb') as f:
            files = {'files': (test_mesh_file.name, f, 'application/octet-stream')}
            upload_response = requests.post(f"{API_BASE}/api/upload", files=files, timeout=30)

        assert upload_response.status_code == 200
        uploaded_filename = upload_response.json()["saved_as"]

        # Step 2: Run simulation with BUILTIN solver
        config = {
            "heat_source_power": 50.0,
            "ambient_temperature": 293.15,
            "initial_temperature": 293.15,
            "convection_coefficient": 20.0,
            "material": "Aluminum",
            "simulation_type": "steady_state",
            "time_step": 0.1,
            "duration": 10.0,
            "max_iterations": 1000,
            "tolerance": 1e-6,
            "write_interval": 50,
            "colormap": "jet",
            "solver": "builtin"  # Use builtin, not OpenFOAM
        }

        payload = {
            "filename": uploaded_filename,
            "config": config
        }

        print(f"\nRunning builtin simulation with {uploaded_filename}...")
        sim_response = requests.post(
            f"{API_BASE}/api/simulate",
            json=payload,
            timeout=TIMEOUT
        )

        # Check response status
        print(f"Response status: {sim_response.status_code}")

        if sim_response.status_code != 200:
            # Print error for debugging
            try:
                error_data = sim_response.json()
                print(f"Error response: {error_data}")
                pytest.fail(f"Simulation failed: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"Raw error response: {sim_response.text}")
                pytest.fail(f"Simulation failed with status {sim_response.status_code}")

        # Parse success response
        data = sim_response.json()
        assert data["status"] == "success"
        assert "results" in data

        results = data["results"]

        # Validate results structure
        assert "min_temp" in results or "min_temperature_K" in results
        assert "max_temp" in results or "max_temperature_K" in results
        assert "vtk_url" in data["results"]

        # Validate temperature values are reasonable
        min_temp = results.get("min_temp", results.get("min_temperature_K", 0))
        max_temp = results.get("max_temp", results.get("max_temperature_K", 0))

        print(f"\nSimulation Results:")
        print(f"  Min Temp: {min_temp:.2f} K ({min_temp - 273.15:.2f} °C)")
        print(f"  Max Temp: {max_temp:.2f} K ({max_temp - 273.15:.2f} °C)")
        print(f"  VTK URL: {results['vtk_url']}")

        # Sanity checks
        assert 200 < min_temp < 500, f"Min temp {min_temp}K seems unrealistic"
        assert min_temp < max_temp, "Max temp should be higher than min temp"
        assert max_temp < 1000, f"Max temp {max_temp}K seems too high"


class TestSimulationOpenFOAM:
    """Test simulation with OpenFOAM solver (requires WSL on Windows)"""

    def test_simulate_openfoam_solver(self, backend_running, test_mesh_file):
        """
        Test simulation with OpenFOAM solver.
        This may skip if OpenFOAM is not available.
        """
        # Step 1: Check if OpenFOAM is available via diagnostics
        diag_response = requests.get(f"{API_BASE}/api/diagnostics", timeout=5)
        # Note: diagnostics doesn't currently check OpenFOAM, but we'll try anyway

        # Step 2: Upload file
        with open(test_mesh_file, 'rb') as f:
            files = {'files': (test_mesh_file.name, f, 'application/octet-stream')}
            upload_response = requests.post(f"{API_BASE}/api/upload", files=files, timeout=30)

        assert upload_response.status_code == 200
        uploaded_filename = upload_response.json()["saved_as"]

        # Step 3: Run simulation with OPENFOAM solver
        config = {
            "heat_source_power": 50.0,
            "ambient_temperature": 293.15,
            "initial_temperature": 293.15,
            "convection_coefficient": 20.0,
            "material": "Aluminum",
            "simulation_type": "steady_state",
            "time_step": 0.1,
            "duration": 10.0,
            "max_iterations": 1000,
            "tolerance": 1e-6,
            "write_interval": 50,
            "colormap": "jet",
            "solver": "openfoam"  # Use OpenFOAM
        }

        payload = {
            "filename": uploaded_filename,
            "config": config
        }

        print(f"\nRunning OpenFOAM simulation with {uploaded_filename}...")
        sim_response = requests.post(
            f"{API_BASE}/api/simulate",
            json=payload,
            timeout=TIMEOUT
        )

        # Check response status
        print(f"Response status: {sim_response.status_code}")

        if sim_response.status_code != 200:
            # Try to parse error
            try:
                error_data = sim_response.json()
                error_msg = error_data.get('error', 'Unknown error')
                print(f"Error response: {error_msg}")

                # If OpenFOAM is not available, skip instead of failing
                if "OpenFOAM" in error_msg or "WSL" in error_msg:
                    pytest.skip(f"OpenFOAM not available: {error_msg}")
                else:
                    pytest.fail(f"Simulation failed: {error_msg}")
            except:
                print(f"Raw error response: {sim_response.text}")
                pytest.fail(f"Simulation failed with status {sim_response.status_code}")

        # Parse success response
        data = sim_response.json()
        assert data["status"] == "success"
        assert "results" in data

        results = data["results"]
        print(f"\nOpenFOAM Simulation Results:")
        print(f"  Results: {results}")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
