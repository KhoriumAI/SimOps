#!/usr/bin/env python3
"""
S3 Storage Monitor with Slack Alerts

Monitors S3 bucket size and sends Slack notifications when approaching limits.
Can run as cron job or AWS Lambda function.

Usage:
  python monitor_s3_usage.py --bucket BUCKET_NAME [--threshold-gb 100]
"""
import os
import sys
import json
import argparse
from datetime import datetime

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    print("ERROR: boto3 not installed. Install with: pip install boto3")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Install with: pip install requests")
    sys.exit(1)


def get_bucket_size(bucket_name, region='us-west-1'):
    """Get total size of S3 bucket in GB"""
    try:
        s3_client = boto3.client('s3', region_name=region)
        cloudwatch = boto3.client('cloudwatch', region_name=region)
        
        # Get bucket size from CloudWatch metrics (more efficient than listing all objects)
        response = cloudwatch.get_metric_statistics(
            Namespace='AWS/S3',
            MetricName='BucketSizeBytes',
            Dimensions=[
                {'Name': 'BucketName', 'Value': bucket_name},
                {'Name': 'StorageType', 'Value': 'StandardStorage'}
            ],
            StartTime=datetime.now().replace(hour=0, minute=0, second=0),
            EndTime=datetime.now(),
            Period=86400,  # 1 day
            Statistics=['Average']
        )
        
        if response['Datapoints']:
            size_bytes = response['Datapoints'][0]['Average']
            size_gb = size_bytes / (1024 ** 3)
            return size_gb
        
        # Fallback: count objects manually (slower)
        print("CloudWatch metrics not available, counting objects manually...")
        total_size = 0
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket_name):
            for obj in page.get('Contents', []):
                total_size += obj['Size']
        
        return total_size / (1024 ** 3)
        
    except ClientError as e:
        print(f"ERROR accessing S3 bucket: {e}")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def get_object_count(bucket_name, region='us-west-1'):
    """Get total number of objects in bucket"""
    try:
        s3_client = boto3.client('s3', region_name=region)
        cloudwatch = boto3.client('cloudwatch', region_name=region)
        
        response = cloudwatch.get_metric_statistics(
            Namespace='AWS/S3',
            MetricName='NumberOfObjects',
            Dimensions=[
                {'Name': 'BucketName', 'Value': bucket_name},
                {'Name': 'StorageType', 'Value': 'AllStorageTypes'}
            ],
            StartTime=datetime.now().replace(hour=0, minute=0, second=0),
            EndTime=datetime.now(),
            Period=86400,
            Statistics=['Average']
        )
        
        if response['Datapoints']:
            return int(response['Datapoints'][0]['Average'])
        return 0
        
    except Exception:
        return 0


def send_slack_alert(webhook_url, bucket_name, size_gb, threshold_gb, object_count):
    """Send Slack notification about S3 usage"""
    usage_percent = int((size_gb / threshold_gb) * 100) if threshold_gb > 0 else 0
    
    message = {
        "text": f"⚠️ *S3 Storage Alert - {bucket_name}*",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"⚠️ S3 Storage Alert - {bucket_name}"
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Current Usage:*\n{size_gb:.2f} GB ({usage_percent}%)"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Threshold:*\n{threshold_gb} GB"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Object Count:*\n{object_count:,} files"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Time:*\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    }
                ]
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "💡 *Suggested Actions:*\n• Review old uploads and meshes\n• Implement lifecycle policies for automatic cleanup\n• Archive old data to S3 Glacier"
                }
            }
        ]
    }
    
    try:
        response = requests.post(webhook_url, json=message, timeout=10)
        if response.status_code == 200:
            print(f"✓ Slack alert sent successfully")
            return True
        else:
            print(f"✗ Slack alert failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Failed to send Slack alert: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Monitor S3 bucket usage and send Slack alerts')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--threshold-gb', type=float, default=100, help='Alert threshold in GB (default: 100)')
    parser.add_argument('--region', default='us-west-1', help='AWS region (default: us-west-1)')
    parser.add_argument('--webhook-url', help='Slack webhook URL (or set SLACK_WEBHOOK_URL env var)')
    parser.add_argument('--force', action='store_true', help='Send alert even if below threshold (for testing)')
    args = parser.parse_args()
    
    # Get webhook URL from args or environment
    webhook_url = args.webhook_url or os.environ.get('SLACK_WEBHOOK_URL')
    if not webhook_url:
        print("ERROR: No Slack webhook URL provided")
        sys.exit(1)
    
    # Get bucket size
    print(f"Checking S3 bucket: {args.bucket}")
    size_gb = get_bucket_size(args.bucket, args.region)
    if size_gb is None:
        print("ERROR: Could not determine bucket size")
        sys.exit(1)
    
    object_count = get_object_count(args.bucket, args.region)
    
    print(f"Bucket size: {size_gb:.2f} GB ({object_count:,} objects)")
    print(f"Threshold: {args.threshold_gb} GB")
    
    # Send alert if threshold exceeded or forced
    if size_gb >= args.threshold_gb or args.force:
        if send_slack_alert(webhook_url, args.bucket, size_gb, args.threshold_gb, object_count):
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        print("✓ Bucket size below threshold, no alert needed")
        sys.exit(0)


if __name__ == '__main__':
    main()
