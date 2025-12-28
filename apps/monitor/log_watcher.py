import os
import glob
import re
import pandas as pd
import time
from datetime import datetime

class LogWatcher:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.file_offsets = {}  # filepath -> last_byte_read
        self.sim_data = {}      # filepath -> dict of metrics
        
        # Regex Patterns
        self.patterns = {
            'time': re.compile(r'^Time = (\d+\.?\d*)'),
            'courant': re.compile(r'Courant Number mean: (\d+\.?\d*(?:e[-+]?\d+)?) max: (\d+\.?\d*(?:e[-+]?\d+)?)'),
            'residual': re.compile(r'Solving for (\w+), Initial residual = (\d+\.?\d*(?:e[-+]?\d+)?)'),
            'temp_max': re.compile(r'T max: (\d+\.?\d*)'),
            # Process/Stage Detection
            'stage_meshing': re.compile(r'Meshing with strategy:|Generating mesh', re.IGNORECASE),
            'stage_solving': re.compile(r'Solving \(|Running Thermal Solver|SimpleFoam|Static analysis was selected|Decascading the MPC', re.IGNORECASE),
            'stage_reporting': re.compile(r'Generating report|Exporting VTK|Job finished', re.IGNORECASE),
            
            # Standardized SimLogger Tags
            'std_stage': re.compile(r'\[STAGE\]\s*(.+)'),
            'std_metric': re.compile(r'\[METRIC\]\s*(\w+)=([\d\.e\-\+]+)'),
            'std_error': re.compile(r'\[ERROR\]\s*(?:(\w+):)?\s*(.+)'),
            'std_metadata': re.compile(r'\[METADATA\]\s*(\w+)=(.+)')
        }

    def reset(self):
        """
        Clears all cached state. Call this when the user wants to start fresh.
        """
        self.file_offsets.clear()
        self.sim_data.clear()

    def poll(self):
        """
        Scans the log directory (RECURSIVELY) for new files and updates valid files.
        Also removes entries for files that no longer exist.
        """
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
            self.sim_data.clear()
            self.file_offsets.clear()
            return

        # Recursive Glob
        all_logs = glob.glob(os.path.join(self.log_dir, "**", "*.log"), recursive=True)
        # Filter out worker_console.log
        log_files = {f for f in all_logs if os.path.basename(f) != "worker_console.log"}
        
        # Remove stale entries for deleted files
        stale_paths = [p for p in self.sim_data.keys() if p not in log_files]
        for stale in stale_paths:
            del self.sim_data[stale]
            if stale in self.file_offsets:
                del self.file_offsets[stale]
        
        for filepath in log_files:
            if filepath not in self.sim_data:
                # Initialize state for new file
                self.sim_data[filepath] = {
                    'Job ID': os.path.basename(filepath).replace('.log', ''),
                    'Status': 'Queued',
                    'Stage': 'Initializing',      # New: Current Stage
                    'Stage Duration': '0s',       # New: Duration in current stage
                    'Start Time': datetime.fromtimestamp(os.path.getctime(filepath)).strftime('%H:%M:%S'),
                    'Run Time': '0s',
                    'Time': 0,
                    'Iterations': 0,
                    'Courant Max': 0.0,
                    'Residuals': {'Ux': [], 'Uy': [], 'Uz': [], 'p': [], 'T': []},
                    'History': {
                        'Time': [],
                        'Courant Max': [],
                        'Ux': [], 'Uy': [], 'Uz': [], 'p': [], 'T': []
                    },
                    'Max Temp': 0,
                    'Error Lines': [],
                    'Metrics': {},                # New: Storage for arbitrary metrics
                    'Last Update': time.time(),
                    '_start_ts': os.path.getctime(filepath),
                    '_stage_ts': time.time()      # Internal timestamp for stage duration
                }
                self.file_offsets[filepath] = 0
            
            self._process_file(filepath)

    def _process_file(self, filepath):
        """
        Reads new lines from a specific log file.
        """
        try:
            # Re-check existence just in case
            if not os.path.exists(filepath):
                return

            # Update Run Time (Only if running)
            if self.sim_data[filepath]['Status'] in ['Running', 'Queued']:
                start_ts = self.sim_data[filepath].get('_start_ts', time.time())
                duration = time.time() - start_ts
                m, s = divmod(duration, 60)
                h, m = divmod(m, 60)
                self.sim_data[filepath]['Run Time'] = f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

            # Update Stage Duration
            stage_ts = self.sim_data[filepath].get('_stage_ts', time.time())
            stage_dur = time.time() - stage_ts
            m, s = divmod(stage_dur, 60)
            self.sim_data[filepath]['Stage Duration'] = f"{int(m):02d}:{int(s):02d}"

            with open(filepath, 'r') as f:
                f.seek(self.file_offsets[filepath])
                new_lines = f.readlines()
                self.file_offsets[filepath] = f.tell()
                
                if new_lines:
                    # Sync status if it was queued
                    if self.sim_data[filepath]['Status'] == 'Queued':
                        self.sim_data[filepath]['Status'] = 'Running'
                    
                    self.sim_data[filepath]['Last Update'] = time.time()
                    
                    for line in new_lines:
                        self._parse_line(filepath, line.strip())
                        
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            self.sim_data[filepath]['Status'] = 'Error'

    def _parse_line(self, filepath, line):
        data = self.sim_data[filepath]
        
        # 0. Standardized SimLogger Parsing (High Priority)
        # [STAGE]
        s_match = self.patterns['std_stage'].search(line)
        if s_match:
            new_stage = s_match.group(1).strip()
            if data['Stage'] != new_stage:
                data['Stage'] = new_stage
                data['_stage_ts'] = time.time()
            return
            
        # [METADATA]
        meta_match = self.patterns['std_metadata'].search(line)
        if meta_match:
            key = meta_match.group(1)
            val = meta_match.group(2).strip()
            # Store in Metrics for now, or a separate Metadata dict if I added one
            # Using Metrics to keep it simple and visible in verification
            data['Metrics'][key] = val
            return

        # [METRIC]
        m_match = self.patterns['std_metric'].search(line)
        if m_match:
            key = m_match.group(1)
            try:
                val = float(m_match.group(2))
                data['Metrics'][key] = val
                # Specific mappings for UI
                if key == 'reynolds_number':
                    data['Reynolds'] = val  # Special field if we want to show it
            except:
                pass
            return

        # [ERROR]
        e_match = self.patterns['std_error'].search(line)
        if e_match:
            msg = e_match.group(2)
            data['Error Lines'].append(msg)
            if data['Status'] != 'Failed':
                data['Status'] = 'Failed'
            return

        # Legacy Stage Detection
        if self.patterns['stage_meshing'].search(line):
            if data['Stage'] != 'Meshing':
                data['Stage'] = 'Meshing'
                data['_stage_ts'] = time.time()
        elif self.patterns['stage_solving'].search(line):
             if data['Stage'] != 'Solving':
                data['Stage'] = 'Solving'
                data['_stage_ts'] = time.time()
        elif self.patterns['stage_reporting'].search(line):
             if data['Stage'] != 'Post-Processing':
                data['Stage'] = 'Post-Processing'
                data['_stage_ts'] = time.time()
        
        # Time / Iteration
        t_match = self.patterns['time'].search(line)
        if t_match:
            data['Time'] = float(t_match.group(1))
            data['Iterations'] += 1
            data['History']['Time'].append(data['Time'])
            return

        # Courant
        c_match = self.patterns['courant'].search(line)
        if c_match:
            c_max = float(c_match.group(2))
            data['Courant Max'] = c_max
            data['History']['Courant Max'].append(c_max)
            return

        # Residuals
        r_match = self.patterns['residual'].search(line)
        if r_match:
            field = r_match.group(1)
            val = float(r_match.group(2))
            
            if field in data['Residuals']:
                data['Residuals'][field].append(val)
                data['History'][field].append(val)
            elif field == 'U': 
                 data['Residuals']['Ux'].append(val)
                 data['History']['Ux'].append(val)
            return

        # Max Temp
        tm_match = self.patterns['temp_max'].search(line)
        if tm_match:
            data['Max Temp'] = float(tm_match.group(1))
            return
            
        # Detect Completion/Failure/Errors
        # Check for ExecutionTime first to set precise duration
        if "ExecutionTime =" in line:
            # Parse: ExecutionTime = 123.45 s
            try:
                parts = line.split('=')
                if len(parts) >= 2:
                    val_str = parts[1].split('s')[0].strip()
                    exec_seconds = float(val_str)
                    m, s = divmod(exec_seconds, 60)
                    h, m = divmod(m, 60)
                    data['Run Time'] = f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
            except:
                pass # Fallback to existing run time
            data['Status'] = 'Converged'

        # CalculiX Success Detection
        elif "Job finished" in line:
            data['Status'] = 'Converged'
            # If we see success, clear previous error lines as they might be from a retry
            data['Error Lines'] = [] 
            
        elif "Total CalculiX Time:" in line:
             try:
                parts = line.split(':')
                if len(parts) >= 2:
                    val_str = parts[1].strip()
                    exec_seconds = float(val_str)
                    data['Run Time'] = f"{exec_seconds:.2f}s"
             except:
                pass

        elif "FOAM FATAL ERROR" in line or "segmentation fault" in line.lower() or "error" in line.lower():
            if data['Status'] != 'Failed':
                data['Status'] = 'Failed'
                # Freeze time on failure is automatic since loop skips update for 'Failed'
            data['Error Lines'].append(line)

    def get_dataframe(self):
        """
        Returns a Pandas DataFrame suitable for the Orchestrator Grid.
        """
        rows = []
        for filepath, data in self.sim_data.items():
            # Calculate a simple "Current Residual" (e.g., max of p/U) for the table
            curr_res = 0
            if data['Residuals']['p']:
                curr_res = data['Residuals']['p'][-1]
            
            rows.append({
                'Job ID': data['Job ID'],
                'Status': data['Status'],
                'Stage': data['Stage'],
                'Stage Duration': data['Stage Duration'],
                'Start Time': data['Start Time'],
                'Run Time': data['Run Time'],
                'Iterations': data['Iterations'],
                'Time': data['Time'],
                'Courant Max': data['Courant Max'],
                'Max Temp': data['Max Temp'],
                'Current Residual': curr_res,
                'Error Count': len(data['Error Lines']),
                # Store full object for Deep Dive
                '_raw_data': data 
            })
        
        return pd.DataFrame(rows)

    def get_details(self, job_id):
        """
        Retrieve full history for a specific job.
        """
        for filepath, data in self.sim_data.items():
            if data['Job ID'] == job_id:
                return data
                
        return None

    def get_recent_logs(self, job_id, lines=50):
        # Find path
        target_path = None
        for filepath, data in self.sim_data.items():
            if data['Job ID'] == job_id:
                target_path = filepath
                break
        
        if target_path and os.path.exists(target_path):
            with open(target_path, 'r') as f:
                # Simple tail
                # For huge files, seek to end minus approx bytes would be better
                # But for now, read all is easiest or use deque
                from collections import deque
                return "".join(deque(f, maxlen=lines))
        return "Log file not found."
