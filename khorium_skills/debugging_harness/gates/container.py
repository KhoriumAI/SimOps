import subprocess
import requests
import time

def run_container_check(url="http://localhost:5000/api/health", compose_file="docker-compose.yml"):
    """
    Gate 3: Container Integration (The AWS Proxy)
    Ensures the application is running in the Docker environment.
    """
    print(f"[*] Running Gate 3 (Container) on {url}...")
    
    errors = []
    
    # 1. Check if docker-compose is up (simplified check)
    try:
        process = subprocess.run(
            ["docker", "compose", "ps", "--format", "json"],
            capture_output=True,
            text=True
        )
        if process.returncode != 0:
            errors.append(f"Docker Compose Check Failed: Is Docker running?\n{process.stderr}")
        elif not process.stdout.strip():
            errors.append("Docker Compose appears to be down (no containers found). Use 'docker compose up -d' to start the environment.")
    except FileNotFoundError:
        errors.append("Docker command not found. Please ensure Docker is installed and in your PATH.")
    except Exception as e:
        errors.append(f"Unexpected error checking Docker: {str(e)}")

    if errors:
        return False, "\n".join(errors)

    # 2. Ping API Endpoint
    max_pings = 5
    ping_count = 0
    while ping_count < max_pings:
        try:
            print(f"    Pinging API ({ping_count+1}/{max_pings})...")
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print("    API is Up (200 OK).")
                break
            else:
                errors.append(f"API returned {response.status_code} at {url}")
        except requests.exceptions.RequestException as e:
            errors.append(f"Ping failed: {str(e)}")
        
        ping_count += 1
        time.sleep(2)

    if ping_count == max_pings:
        return False, f"Gate 3 Failed: API unreachable after {max_pings} attempts.\n" + "\n".join(errors[-3:])

    # 3. Check Logs for Crashes (Optional but recommended)
    try:
        log_process = subprocess.run(
            ["docker", "compose", "logs", "--tail", "50"],
            capture_output=True,
            text=True
        )
        if "ERROR" in log_process.stdout.upper() or "CRASH" in log_process.stdout.upper():
            print("    [WARN] Errors detected in container logs.")
            # We might not fail the gate just for logs, but we can report them
    except:
        pass

    return True, "Container environment is healthy."

if __name__ == "__main__":
    success, result = run_container_check()
    if success:
        print(f"✅ Gate 3 Passed: {result}")
    else:
        print(f"❌ Gate 3 Failed:\n{result}")
