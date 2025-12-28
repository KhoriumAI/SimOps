import os
import time
import random
import shutil

LOG_DIR = os.path.join(os.path.dirname(__file__), 'sim_logs')
USE_STANDARD_LOGGING = True

def setup_logs(num_logs=50):
    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    print(f"Created {LOG_DIR}")
    return [os.path.join(LOG_DIR, f"job_{i:03d}.log") for i in range(num_logs)]

def generate_header(job_name):
    # Mimic SimLogger init
    header = f"=== LOG START: {time.ctime()} ===\n"
    header += f"[INFO] SimOps Worker Initialized\n"
    header += f"[STAGE] Initializing\n"
    header += f"[METADATA] job_name={job_name}\n"
    header += f"[METADATA] solver=OpenFOAM_Native\n"
    header += f"[INFO] Configuration loaded for {job_name}\n"
    return header

def generate_step(time_val, iteration, behavior="normal"):
    # Courant: Normal < 1, Crash > 1 explodes
    courant = random.uniform(0.1, 0.8)
    if behavior == "crash":
        courant = random.uniform(1.2, 50.0)
    elif behavior == "converging":
        courant = random.uniform(0.1, 0.3)

    # Residuals
    res_base = 0.1 * (0.95 ** iteration)
    if behavior == "crash":
        res_base = 0.1 * (1.1 ** iteration)
    
    res_p = res_base * random.uniform(0.8, 1.2)
    res_U = res_base * random.uniform(0.8, 1.2)
    temp = 300 + (iteration * 0.5) + random.uniform(-2, 2)
    
    # 1. Standard Metrics
    lines = [
        f"[METRIC] iteration={iteration}",
        f"[METRIC] time={time_val}",
        f"[METRIC] courant_max={courant:.4f}",
        f"[METRIC] residual_p={res_p:.6e}",
        f"[METRIC] residual_U={res_U:.6e}",
        f"[METRIC] temp_max={temp:.2f}"
    ]
    
    # 2. Legacy OpenFOAM output (for realism/robustness testing)
    lines.append(f"Time = {time_val}")
    lines.append(f"Courant Number mean: {courant:.6f} max: {courant * 1.5:.6f}")
    lines.append(f"Solving for p, Initial residual = {res_p:.6e}, Final residual = {res_p/10:.6e}, No Iterations 12")
    
    return "\n".join(lines) + "\n"

def main():
    log_files = setup_logs(50)
    
    # Initialize states
    states = {} 
    for f in log_files:
        job_name = os.path.basename(f).replace('.log', '')
        r = random.random()
        if r < 0.1: state = "crash"
        elif r < 0.3: state = "stuck"
        else: state = "normal"
        
        states[f] = {
            "state": state,
            "time": 0,
            "iter": 0,
            "finished": False,
            "stage": "Initializing"
        }
        
        with open(f, 'w') as fh:
            fh.write(generate_header(job_name))

    print("Starting simulation loop... Press Ctrl+C to stop.")
    
    while True:
        for f in log_files:
            s = states[f]
            if s["finished"]: continue
            if s["state"] == "stuck" and random.random() > 0.05: continue
            
            # State Machine / Logic
            # 0 -> 5: Meshing
            if s["iter"] == 5 and s["stage"] == "Initializing":
                s["stage"] = "Meshing"
                with open(f, 'a') as fh:
                    fh.write(f"\n[STAGE] Meshing\n[INFO] Starting mesh generation strategy...\n")
                continue # Spend a tick here
                
            # 5 -> 10: Solving
            if s["iter"] == 10 and s["stage"] == "Meshing":
                s["stage"] = "Solving"
                with open(f, 'a') as fh:
                    fh.write(f"\n[STAGE] Solving (CFD)\n[METADATA] mesh_cells={random.randint(50000, 200000)}\n")
            
            if s["stage"] != "Solving" and s["iter"] > 10:
                s["stage"] = "Solving" # Catchup

            # Crash logic
            if s["state"] == "crash" and s["iter"] > 25:
                with open(f, 'a') as fh:
                    fh.write("\n[ERROR] SIGSEGV: Segmentation fault in bubble bursting logic\n")
                    fh.write("FOAM FATAL ERROR: bubble bursting logic failed\n") # Legacy too
                s["finished"] = True
                continue
                
            # Step
            s["time"] += 1
            s["iter"] += 1
            
            # Only generate steps if solving
            if s["iter"] > 10:
                chunk = generate_step(s["time"], s["iter"]-10, behavior=s["state"])
                with open(f, 'a') as fh:
                    fh.write(chunk)
                
            # Convergence logic
            if s["iter"] > 80 and s["state"] == "normal":
                 with open(f, 'a') as fh:
                    fh.write(f"\n[STAGE] Post-Processing\n")
                    fh.write(f"[METRIC] solve_time={s['iter']*0.5}\n")
                    fh.write(f"ExecutionTime = {s['iter']*0.5} s\nEnd\n")
                 s["finished"] = True
                 
        time.sleep(0.5) # Speed up for testing

if __name__ == "__main__":
    main()
