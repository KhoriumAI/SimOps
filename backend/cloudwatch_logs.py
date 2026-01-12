"""
CloudWatch Logs tailing service for streaming Modal job logs
"""
import boto3
import time
import os
from typing import Optional, Callable, Dict, List
from datetime import datetime, timedelta
from threading import Thread, Event
import json


class CloudWatchLogTailer:
    """
    Tails CloudWatch logs for a specific log group/stream and calls
    a callback function for each new log line.
    """
    
    def __init__(self, log_group_name: str, log_stream_name: Optional[str] = None,
                 region: str = 'us-west-1', aws_access_key: Optional[str] = None,
                 aws_secret_key: Optional[str] = None):
        """
        Initialize CloudWatch log tailer.
        
        Args:
            log_group_name: CloudWatch log group name (e.g., '/modal/jobs')
            log_stream_name: Optional specific log stream name
            region: AWS region
            aws_access_key: Optional AWS access key (uses boto3 default if not provided)
            aws_secret_key: Optional AWS secret key
        """
        self.log_group_name = log_group_name
        self.log_stream_name = log_stream_name
        self.region = region
        
        # Initialize boto3 client
        if aws_access_key and aws_secret_key:
            self.client = boto3.client(
                'logs',
                region_name=region,
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key
            )
        else:
            self.client = boto3.client('logs', region_name=region)
        
        self.running = False
        self.thread = None
        self.stop_event = Event()
        self.last_token = None
        self.callbacks: List[Callable[[str, datetime], None]] = []
    
    def add_callback(self, callback: Callable[[str, datetime], None]):
        """Add a callback function to be called for each log line."""
        self.callbacks.append(callback)
    
    def _get_log_streams(self) -> List[str]:
        """Get list of log streams in the log group."""
        try:
            kwargs = {'logGroupName': self.log_group_name}
            if self.log_stream_name:
                kwargs['logStreamNamePrefix'] = self.log_stream_name
            
            response = self.client.describe_log_streams(**kwargs)
            return [stream['logStreamName'] for stream in response.get('logStreams', [])]
        except Exception as e:
            print(f"[CloudWatch] Error getting log streams: {e}")
            return []
    
    def _get_log_events(self, log_stream: str, start_time: Optional[int] = None) -> tuple[List[Dict], Optional[str]]:
        """
        Get log events from a stream.
        
        Returns:
            Tuple of (events list, next_token)
        """
        try:
            kwargs = {
                'logGroupName': self.log_group_name,
                'logStreamName': log_stream,
                'limit': 1000
            }
            
            if start_time:
                kwargs['startTime'] = start_time
            
            if self.last_token:
                kwargs['nextToken'] = self.last_token
            
            response = self.client.get_log_events(**kwargs)
            events = response.get('events', [])
            next_token = response.get('nextForwardToken')
            
            return events, next_token
        except Exception as e:
            print(f"[CloudWatch] Error getting log events: {e}")
            return [], None
    
    def _process_events(self, events: List[Dict]):
        """Process log events and call callbacks."""
        for event in events:
            message = event.get('message', '').strip()
            if not message:
                continue
            
            timestamp = datetime.fromtimestamp(event.get('timestamp', 0) / 1000)
            
            # Call all registered callbacks
            for callback in self.callbacks:
                try:
                    callback(message, timestamp)
                except Exception as e:
                    print(f"[CloudWatch] Callback error: {e}")
    
    def start(self):
        """Start tailing logs in a background thread."""
        if self.running:
            return
        
        self.running = True
        self.stop_event.clear()
        self.thread = Thread(target=self._tail_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop tailing logs."""
        self.running = False
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)
    
    def _tail_loop(self):
        """Main tailing loop running in background thread."""
        # Start from 1 minute ago to catch recent logs
        start_time = int((datetime.now() - timedelta(minutes=1)).timestamp() * 1000)
        
        while self.running and not self.stop_event.is_set():
            try:
                # Get log streams
                streams = self._get_log_streams()
                
                if not streams:
                    # No streams yet, wait and retry
                    time.sleep(2)
                    continue
                
                # Process each stream
                for stream in streams:
                    if not self.running:
                        break
                    
                    events, next_token = self._get_log_events(stream, start_time)
                    
                    if events:
                        self._process_events(events)
                        # Update start time to last event timestamp
                        if events:
                            start_time = events[-1].get('timestamp', start_time) + 1
                    
                    if next_token:
                        self.last_token = next_token
                
                # Poll every 2 seconds
                time.sleep(2)
                
            except Exception as e:
                print(f"[CloudWatch] Error in tail loop: {e}")
                time.sleep(5)  # Wait longer on error
    
    @staticmethod
    def get_log_group_for_job(job_id: str, base_group: str = '/modal/jobs') -> str:
        """
        Get CloudWatch log group name for a specific job.
        
        Args:
            job_id: Modal job ID
            base_group: Base log group name
        
        Returns:
            Full log group name
        """
        return f"{base_group}/{job_id}"
    
    @staticmethod
    def get_log_stream_for_job(job_id: str) -> str:
        """
        Get CloudWatch log stream name for a specific job.
        Modal typically uses the function name or job ID as stream name.
        """
        return f"job-{job_id}"


def create_log_tailer_for_job(job_id: str, callback: Callable[[str, datetime], None],
                              region: str = 'us-west-1') -> CloudWatchLogTailer:
    """
    Convenience function to create a log tailer for a specific Modal job.
    
    Args:
        job_id: Modal job ID
        callback: Function to call for each log line: callback(message, timestamp)
        region: AWS region
    
    Returns:
        Configured CloudWatchLogTailer instance
    """
    log_group = CloudWatchLogTailer.get_log_group_for_job(job_id)
    log_stream = CloudWatchLogTailer.get_log_stream_for_job(job_id)
    
    tailer = CloudWatchLogTailer(
        log_group_name=log_group,
        log_stream_name=log_stream,
        region=region
    )
    tailer.add_callback(callback)
    
    return tailer


