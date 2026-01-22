import os
import shutil
import json
import re
import argparse
from pathlib import Path
from collections import defaultdict

# --- Configuration ---
FAILED_DIR_NAME = "_FAILED"
SUMMARY_FILENAME = "summary.html"

# --- Physics Guardrail (Red Filter) ---

def check_openfoam_log(log_path):
    """
    Parses OpenFOAM log file for failure keywords.
    Returns: (is_passed, reason)
    """
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
            # 1. Crash / Fatal Error
            if "Floating point exception" in content:
                return False, "Floating point exception (Crash)"
            if "FOAM FATAL ERROR" in content:
                return False, "FOAM FATAL ERROR found"
            if "Segmentation fault" in content:
                return False, "Segmentation fault (Crash)"

            # 2. Bounding validation
            # "bounding k", "bounding omega", "bounding T" often indicate divergence
            if re.search(r"bounding\s+(?!box\b)\w+", content):
                 return False, "Found 'bounding' variable warning (divergence)"

            # 3. Continuity errors
            # Check for localized explosions in continuity
            # "time step continuity errors : sum local = 1.23199e+85"
            # Regex to find "sum local" values
            matches = re.findall(r"sum local\s*=\s*([\d\.eE\+\-]+)", content)
            for m in matches:
                try:
                    val = float(m)
                    if val > 1e10: # Threshold for explosion
                        return False, f"Continuity error explosion: {val}"
                except ValueError: pass

            # 4. Completion check
            # Only if strictly required, but crashes are usually caught above.

    except Exception as e:
        print(f"  [WARN] Could not read {log_path}: {e}")
        return False, f"Could not read log: {e}"
    
    return True, "OK"

def check_calculix_sta(sta_path):
    """
    Parses CalculiX .sta file for time increment issues.
    Returns: (is_passed, reason)
    """
    try:
        with open(sta_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # .sta format: ..  TIME  TIME-INC (Last column typically is dt)
            # We look for a small dt < 1e-5
            
            for line in lines:
                parts = line.split()
                if len(parts) < 3: continue
                # Look for floating point numbers
                try:
                    # Parse the last column as dt
                    dt_str = parts[-1] 
                    if 'E' in dt_str or '.' in dt_str:
                         dt = float(dt_str)
                         if 0.0 < dt < 1e-5:
                             return False, f"Time increment (dt) {dt} < 1e-5 detected (Divergence)"
                except ValueError:
                    continue
                    
    except Exception as e:
        return False, f"Could not read .sta: {e}"
        
    return True, "OK"

def check_sanity_bounds(run_name, root_dir):
    """
    Checks result.json or other outputs for physical sanity.
    Also checks for mesh QC failures.
    Returns: (is_passed, reason)
    """
    # Check for QC failure first
    qc_files = list(root_dir.glob(f"{run_name}*_qc.json"))
    if qc_files:
        try:
            with open(qc_files[0], 'r') as f:
                qc_data = json.load(f)
            if not qc_data.get('passed', True):
                return False, f"Mesh QC: {qc_data.get('reason', 'Unknown QC failure')}"
        except Exception:
            pass
    
    # Look for {run_name}_result.json or similar
    json_path = root_dir / f"{run_name}_result.json"
    if not json_path.exists():
        # Maybe just "result.json" inside case dir?
        json_path = root_dir / f"{run_name}_case" / "result.json"
    
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            # Check Max T
            t_max = data.get("max_temperature") or data.get("T_max")
            if t_max and float(t_max) > 5000:
                return False, f"Max Temperature {t_max}K > 5000K (Physical sanity breach)"
            
            # Check Max Velocity (CFD/Thermal only)
            if data.get('sim_type', 'fluid') != 'structural':
                u_max = data.get("max_velocity") or data.get("U_max")
                if u_max and float(u_max) > 343:
                     return False, f"Max Velocity {u_max} m/s > Mach 1 (Physical sanity breach)"
            
            # Check Structural Stress (Static Structural)
            s_max = data.get("max_stress_pa") or data.get("max_stress")
            if s_max and float(s_max) > 1e11: # 100 GPa
                 return False, f"Max Stress {float(s_max)/1e6:.1f} MPa > 100,000 MPa (Simulation exploded)"

            # Check Structural Displacement
            d_max = data.get("max_displacement_mm") or data.get("max_disp")
            if d_max and float(d_max) > 2000: # 2 meters
                 return False, f"Max Displacement {float(d_max):.1f} mm > 2000mm (Simulation exploded)"
            
            # Check for QC failure metadata in result.json
            qc_failure = data.get("qc_failure_reason")
            if qc_failure:
                return False, f"Mesh QC: {qc_failure}"

        except Exception as e:
            pass # Non-fatal if json is unreadable/missing, usually
            
    return True, "OK"


def group_runs(root_dir):
    """
    Scans directory and groups files/folders into logical "Runs".
    Returns: dict { run_name: { 'paths': [], 'type': 'OF'/'CCX'/'UNKNOWN' } }
    """
    runs = defaultdict(lambda: {'paths': [], 'type': 'UNKNOWN', 'logs': []})
    
    for item in root_dir.iterdir():
        if item.name == FAILED_DIR_NAME or item.name.endswith(".html"):
            continue
            
        # 1. Case Directories
        if item.is_dir() and item.name.endswith("_case"):
            run_name = item.name[:-5] # remove _case
            runs[run_name]['paths'].append(item)
            # Check for internal logs
            internal_logs = list(item.glob("log.*"))
            if internal_logs:
                runs[run_name]['logs'].extend(internal_logs)
                if runs[run_name]['type'] == 'UNKNOWN': runs[run_name]['type'] = 'OF'
                
        # 2. Loose Files (CalculiX or Results)
        elif item.is_file():
            # Heuristic for run name: 
            # USED_Cylinder_HighFi_Layered.sta -> USED_Cylinder_HighFi_Layered
            # Airfoil_velocity.png -> Airfoil ?? (Maybe dangerous if name has underscores)
            # Let's rely on .sta and .log and .inp
            
            if item.suffix == '.sta':
                run_name = item.stem
                runs[run_name]['paths'].append(item)
                runs[run_name]['logs'].append(item) # Treat sta as log
                runs[run_name]['type'] = 'CCX'
                
            elif item.suffix == '.log':
                run_name = item.stem
                runs[run_name]['paths'].append(item)
                runs[run_name]['logs'].append(item)
                
            elif item.suffix in ['.json', '.png', '.pdf', '.inp', '.msh', '.vtk']:
                # Attach to existing run if matches, or potential new run
                # To be safe, we only "create" a run entry if we see a case folder or a .sta/.log/inp
                # But we should add these files to the run if they match so valid_batch can move them.
                pass 

    # Second pass to associate other files by prefix
    all_files = list(root_dir.iterdir())
    for run_name in list(runs.keys()):
        for f in all_files:
            if f.is_file() and f.name.startswith(run_name) and f not in runs[run_name]['paths']:
                 # Avoid partial matches? e.g. run "A" matching "A_B.png"
                 # Check if next char is . or _ or end
                 avg_char = f.name[len(run_name)] if len(f.name) > len(run_name) else ''
                 if avg_char in ['.', '_', '-']:
                     runs[run_name]['paths'].append(f)

    return runs

def validate_run_object(run_name, run_data, root_dir):
    """
    Validates a run object.
    """
    # 1. Physics Logs
    for log in run_data['logs']:
        if log.name.endswith('.sta'):
            ok, reason = check_calculix_sta(log)
            if not ok: return False, f"CalculiX STA: {reason}"
        elif 'log.simpleFoam' in log.name or log.suffix == '.log': # Generic log check
             # If it's a CCX log, check_openfoam might get confused but usually ok
             # CCX logs don't typically have "bounding" etc.
             # Only apply OF check if it looks like OF (log.*) or user specified .log
             # SimOps uses .log for CCX too (seen in file list).
             # Let's peek at header? Or just run OF checks (harmless usually)
             ok, reason = check_openfoam_log(log)
             if not ok: return False, f"Log Check {log.name}: {reason}"

    # 2. Sanity
    ok, reason = check_sanity_bounds(run_name, root_dir)
    if not ok: return False, f"Sanity: {reason}"
    
    return True, "PASSED"

# --- Contact Sheet (Yellow Filter) ---

def generate_contact_sheet(passed_run_names, root_dir, output_html="summary.html"):
    """
    Generates HTML contact sheet for passed runs.
    """
    
    html_header = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SimOps Contact Sheet</title>
        <style>
            body { font-family: sans-serif; background: #222; color: #eee; }
            table { width: 100%; border-collapse: collapse; }
            td { padding: 10px; border: 1px solid #444; vertical-align: top; }
            img { max-width: 400px; max-height: 300px; display: block; margin-bottom: 5px; background: #000; }
            .run-name { font-weight: bold; font-size: 1.1em; margin-bottom: 5px; color: #8f8; }
            .meta { font-size: 0.8em; color: #aaa; }
        </style>
    </head>
    <body>
    <h1>Simulation Validation Contact Sheet</h1>
    <table>
    """
    
    html_rows = ""
    
    for run_name in sorted(passed_run_names):
        # Look for images in the root associated with this run
        # Priority: {run_name}_velocity.png, {run_name}_stress.png
        
        all_pngs = sorted(list(root_dir.glob(f"{run_name}*.png")))
        
        def get_img(keyword):
            for p in all_pngs:
                if keyword in p.name.lower(): return p
            return None
            
        img_main = get_img("velocity") or get_img("stress") or get_img("streamlines")
        img_res = get_img("residual")
        
        # Fallback: look inside case folder if no flat pngs
        if not img_main:
             case_dir = root_dir / f"{run_name}_case"
             if case_dir.exists():
                 case_pngs = sorted(list(case_dir.glob("*.png")))
                 if case_pngs: img_main = case_pngs[0] # Just take first

        html_rows += "<tr>"
        html_rows += f"<td><div class='run-name'>{run_name}</div></td>"
        
        # Main Image
        html_rows += "<td>"
        if img_main:
            rel_path = os.path.relpath(img_main, start=os.path.dirname(output_html))
            html_rows += f"<img src='{rel_path}' title='Main Viz'>"
        else:
            html_rows += "No Image"
        html_rows += "</td>"
        
        # Residuals
        html_rows += "<td>"
        if img_res:
             rel_path = os.path.relpath(img_res, start=os.path.dirname(output_html))
             html_rows += f"<img src='{rel_path}' title='Residuals'>"
        html_rows += "</td>"
        
        html_rows += "</tr>"

    html_footer = """
    </table>
    </body>
    </html>
    """
    
    with open(output_html, "w", encoding='utf-8') as f:
        f.write(html_header + html_rows + html_footer)
    
    print(f"Generated Contact Sheet: {os.path.abspath(output_html)}")


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Traffic Light Validation System")
    parser.add_argument("--root", default="output", help="Root directory containing run folders")
    parser.add_argument("--test-mode", action="store_true", help="Don't move files, just print")
    args = parser.parse_args()
    
    root_dir = Path(args.root)
    if not root_dir.exists():
        print(f"Error: Directory '{root_dir}' not found.")
        return

    failed_dir = root_dir / FAILED_DIR_NAME
    if not args.test_mode:
        failed_dir.mkdir(exist_ok=True)
        
    print(f"Scanning runs in {root_dir}...")
    
    runs = group_runs(root_dir)
    passed_run_names = []
    
    print(f"Found {len(runs)} potential runs.")
    
    for run_name, run_data in runs.items():
        is_valid, reason = validate_run_object(run_name, run_data, root_dir)
        
        if is_valid:
            passed_run_names.append(run_name)
        else:
            print(f" [RED] FAILED: {run_name} -> {reason}")
            if not args.test_mode:
                # Move ALL associated files to _FAILED/{run_name}/
                target_folder = failed_dir / run_name
                # Handle collision
                if target_folder.exists():
                     import time
                     target_folder = failed_dir / f"{run_name}_{int(time.time())}"
                
                target_folder.mkdir(parents=True, exist_ok=True)
                
                for path in run_data['paths']:
                    try:
                        # If path is inside root_dir, move it. 
                        # (It should be, unless logic is wrong)
                         if path.exists():
                            shutil.move(str(path), str(target_folder / path.name))
                    except Exception as e:
                        print(f"       Failed to move {path.name}: {e}")
                print(f"       Moved to {FAILED_DIR_NAME}/{target_folder.name}")

    # Generate Contact Sheet
    if passed_run_names:
        generate_contact_sheet(passed_run_names, root_dir, output_html=str(root_dir / SUMMARY_FILENAME))
    else:
        print("No passed runs found.")

if __name__ == "__main__":
    main()
