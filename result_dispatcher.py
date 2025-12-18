
import os
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class ResultDispatcher:
    """
    Dispatches simulation results to organized output folders.
    
    Features:
    - Creates per-job output directories
    - Copies artifacts to the destination (preserving originals for debug)
    - Generates job manifests
    - Handles cross-platform path compatibility
    """
    
    def __init__(self, output_base_dir: Path):
        self.output_base = output_base_dir
        
    def dispatch_result(self, job_result: Dict) -> Optional[Path]:
        """
        Dispatch the results of a simulation to the final output directory.
        
        Args:
            job_result (Dict): The result dictionary from the worker.
            
        Returns:
            Optional[Path]: The path to the created job directory, or None if failed.
        """
        try:
            job_name = job_result.get('job_name', 'unknown_job')
            strategy = job_result.get('strategy', 'unknown_strategy')
            success = job_result.get('success', False)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 1. Create job folder
            status_suffix = "OK" if success else "FAILED"
            folder_name = f"{job_name}_{timestamp}_{status_suffix}"
            job_dir = self._create_job_folder(folder_name)
            
            logger.info(f"📁 Dispatching results to: {job_dir}")
            
            # 2. Collect files to copy
            files_to_copy = []
            
            # Add files from result dict if they exist
            for key in ['mesh_file', 'vtk_file', 'png_file', 'report_file']:
                if path_str := job_result.get(key):
                    files_to_copy.append(Path(path_str))
                    
            # Add metadata file separately if it wasn't passed, though it usually isn't in the result dict directly
            # The worker writes a separate json file, let's try to find it
            if 'output_dir' in job_result:
                 # If we had the output dir we could look for it, but we might have to infer it
                 pass

            # 3. Copy artifacts
            self._copy_artifacts(files_to_copy, job_dir)
            
            # 4. Create a manifest for this dispatch
            self._create_dispatch_manifest(job_dir, job_result)
            
            logger.info(f"✅ Dispatch complete for {job_name}")
            return job_dir
            
        except Exception as e:
            logger.error(f"❌ Failed to dispatch results: {e}")
            return None

    def _create_job_folder(self, folder_name: str) -> Path:
        """Create the per-job output directory."""
        job_dir = self.output_base / folder_name
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    def _copy_artifacts(self, source_files: List[Path], dest_dir: Path) -> None:
        """Copy artifact files to the destination directory."""
        for src_path in source_files:
            try:
                if src_path.exists():
                    shutil.copy2(src_path, dest_dir)
                    logger.debug(f"  Copied: {src_path.name}")
                else:
                    logger.warning(f"  Artifact missing, cannot copy: {src_path}")
            except Exception as e:
                logger.error(f"  Failed to copy {src_path.name}: {e}")

    def _create_dispatch_manifest(self, job_dir: Path, job_result: Dict) -> None:
        """Create a summary JSON in the dispatch folder."""
        import json
        manifest_path = job_dir / "dispatch_manifest.json"
        
        manifest = {
            "dispatched_at": datetime.now().isoformat(),
            "original_result": job_result
        }
        
        try:
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write manifest: {e}")
