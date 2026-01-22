"""
Local Docker Compute Provider
==============================
Executes simulations using local Docker containers.
"""

import os
import json
import uuid
import subprocess
import logging
from typing import Dict, Any
from .provider import ComputeProvider, JobStatus
from core.schemas import SimConfig

logger = logging.getLogger(__name__)

class LocalDockerProvider(ComputeProvider):
    """Runs simulation jobs in local Docker containers."""
    
    def __init__(self, image_name: str = "simops-worker:latest"):
        self.image_name = image_name
        self.jobs: Dict[str, Dict[str, Any]] = {}
    
    def submit_job(self, config: SimConfig) -> str:
        """Submit job to local Docker."""
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {"status": JobStatus.PENDING, "config": config}
        
        try:
            # Serialize config for worker
            config_json = config.model_dump_json()
            
            cmd = [
                "docker", "run", "--rm",
                "-v", f"{os.getcwd()}:/data",
                self.image_name,
                "python", "/app/simops_worker.py",
                f"/data/{config.cad_file}",
                "--config", config_json
            ]
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.jobs[job_id]["process"] = process
            self.jobs[job_id]["status"] = JobStatus.RUNNING
            logger.info(f"Job {job_id} submitted to local Docker")
            
        except Exception as e:
            logger.error(f"Job {job_id} submission failed: {e}")
            self.jobs[job_id]["status"] = JobStatus.FAILED
            self.jobs[job_id]["error"] = str(e)
        
        return job_id
    
    def get_status(self, job_id: str) -> JobStatus:
        """Check job status."""
        job = self.jobs.get(job_id)
        if not job:
            return JobStatus.FAILED
        
        if job["status"] == JobStatus.RUNNING:
            process = job.get("process")
            if process and process.poll() is not None:
                job["status"] = JobStatus.COMPLETED if process.returncode == 0 else JobStatus.FAILED
        
        return job["status"]
    
    def get_results(self, job_id: str) -> Dict[str, Any]:
        """Retrieve job results."""
        job = self.jobs.get(job_id)
        if not job or job["status"] != JobStatus.COMPLETED:
            return {"success": False, "error": "Job not completed"}
        
        return {
            "success": True,
            "job_id": job_id,
            "output_dir": f"output/{job_id}"
        }
