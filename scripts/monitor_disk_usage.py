#!/usr/bin/env python3
"""
EC2 Disk Usage Monitor with Slack Alerts

Monitors disk usage and sends Slack notifications when usage exceeds threshold.
Designed to run as a cron job on EC2 instances.

Usage:
  python monitor_disk_usage.py [--threshold 80] [--webhook-url URL]
"""
import subprocess
import os
import sys
import json
import argparse
from datetime import datetime

try:
    import requests
except ImportError:
    print("ERROR: requests module not installed. Install with: pip install requests")
    sys.exit(1)


def get_disk_usage():
    """Get disk usage percentage for root filesystem"""
    try:
        result = subprocess.run(['df', '-h', '/'], capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        if len(lines) < 2:
            return None
        
        # Parse the output (second line contains the data)
        parts = lines[1].split()
        # Use% column is typically index 4
        usage_str = parts[4].rstrip('%')
        return int(usage_str)
    except Exception as e:
        print(f"ERROR getting disk usage: {e}")
        return None


def get_hostname():
    """Get instance hostname or EC2 instance ID"""
    try:
        # Try to get EC2 instance ID
        result = subprocess.run(
            ['ec2-metadata', '--instance-id'],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            return result.stdout.strip().split(': ')[1]
    except:
        pass
    
    # Fallback to hostname
    try:
        result = subprocess.run(['hostname'], capture_output=True, text=True)
        return result.stdout.strip()
    except:
        return "unknown-host"


def send_slack_alert(webhook_url, usage_percent, threshold, hostname):
"""Send Slack notification about disk usage"""
    message = {
        "text": f"⚠️ *Disk Space Alert - {hostname}*",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"⚠️ Disk Space Alert - {hostname}"
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Current Usage:*\n{usage_percent}%"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Threshold:*\n{threshold}%"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Time:*\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Host:*\n`{hostname}`"
                    }
                ]
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "💡 *Suggested Actions:*\n• Run cleanup script: `sudo /home/ec2-user/MeshPackageLean/scripts/cleanup.py`\n• Check large files: `du -sh /home/ec2-user/* | sort -hr | head -10`\n• Clear pip cache: `pip cache purge`"
                }
            }
        ]
    }
    
    try:
        response = requests.post(webhook_url, json=message, timeout=10)
        if response.status_code == 200:
            print(f"✓ Slack alert sent successfully (Usage: {usage_percent}%)")
            return True
        else:
            print(f"✗ Slack alert failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Failed to send Slack alert: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Monitor EC2 disk usage and send Slack alerts')
    parser.add_argument('--threshold', type=int, default=80, help='Alert threshold percentage (default: 80)')
    parser.add_argument('--webhook-url', help='Slack webhook URL (or set SLACK_WEBHOOK_URL env var)')
    parser.add_argument('--force', action='store_true', help='Send alert even if below threshold (for testing)')
    args = parser.parse_args()
    
    # Get webhook URL from args or environment
    webhook_url = args.webhook_url or os.environ.get('SLACK_WEBHOOK_URL')
    if not webhook_url:
        print("ERROR: No Slack webhook URL provided. Use --webhook-url or set SLACK_WEBHOOK_URL environment variable")
        sys.exit(1)
    
    # Get disk usage
    usage = get_disk_usage()
    if usage is None:
        print("ERROR: Could not determine disk usage")
        sys.exit(1)
    
    hostname = get_hostname()
    print(f"Disk usage on {hostname}: {usage}% (threshold: {args.threshold}%)")
    
    # Send alert if threshold exceeded or forced
    if usage >= args.threshold or args.force:
        if send_slack_alert(webhook_url, usage, args.threshold, hostname):
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        print("✓ Disk usage below threshold, no alert needed")
        sys.exit(0)


if __name__ == '__main__':
    main()
